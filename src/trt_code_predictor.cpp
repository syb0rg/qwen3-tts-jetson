#include "trt_code_predictor.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <fstream>
#include <cstring>
#include <cstdio>
#include <utility>

// CUDA kernel for FP32 → FP16 conversion (defined in trt_cuda_kernels.cu)
extern "C" void gpu_fp32_to_fp16(const float * in, void * out, int n,
                                  cudaStream_t stream);
extern "C" void gpu_argmax_f32(const float * in, int32_t * out, int n,
                                cudaStream_t stream);
extern "C" void gpu_embedding_lookup_by_gpu_id(const int32_t * token_id_ptr,
                                                const float * table,
                                                float * output,
                                                int embd_dim,
                                                cudaStream_t stream);
extern "C" void gpu_sample_topk_f32(const float * logits, const float * rand_val,
                                     int32_t * out, float temperature,
                                     int32_t top_k, int32_t vocab_size,
                                     cudaStream_t stream);

namespace qwen3_tts {

class TRTCodePredLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char * msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TRT-CodePred] %s\n", msg);
        }
    }
};

static TRTCodePredLogger g_cp_trt_logger;

// Add bias using cuBLAS saxpy: output += 1.0 * bias
static cublasStatus_t add_bias_cublas(cublasHandle_t handle, float * output,
                                       const float * bias, int n) {
    float alpha = 1.0f;
    return cublasSaxpy(handle, n, &alpha, bias, 1, output, 1);
}

TRTCodePredictor::TRTCodePredictor() = default;

TRTCodePredictor::~TRTCodePredictor() {
    unload();
}

void TRTCodePredictor::unload() {
    if (d_hidden_in_) { cudaFree(d_hidden_in_); d_hidden_in_ = nullptr; }
    if (d_position_id_) { cudaFree(d_position_id_); d_position_id_ = nullptr; }
    if (d_hidden_out_) { cudaFree(d_hidden_out_); d_hidden_out_ = nullptr; }
    if (d_hidden_out_fp16_) { cudaFree(d_hidden_out_fp16_); d_hidden_out_fp16_ = nullptr; }
    if (d_logits_) { cudaFree(d_logits_); d_logits_ = nullptr; }
    if (d_token_id_) { cudaFree(d_token_id_); d_token_id_ = nullptr; }
    if (d_all_token_ids_) { cudaFree(d_all_token_ids_); d_all_token_ids_ = nullptr; }
    if (d_rand_vals_) { cudaFree(d_rand_vals_); d_rand_vals_ = nullptr; }
    if (d_mtp_proj_w_) { cudaFree(d_mtp_proj_w_); d_mtp_proj_w_ = nullptr; }
    if (d_mtp_proj_b_) { cudaFree(d_mtp_proj_b_); d_mtp_proj_b_ = nullptr; }
    if (d_proj_input_) { cudaFree(d_proj_input_); d_proj_input_ = nullptr; }

    for (int i = 0; i < 5; i++) {
        if (d_past_key_[i]) { cudaFree(d_past_key_[i]); d_past_key_[i] = nullptr; }
        if (d_past_value_[i]) { cudaFree(d_past_value_[i]); d_past_value_[i] = nullptr; }
        if (d_present_key_[i]) { cudaFree(d_present_key_[i]); d_present_key_[i] = nullptr; }
        if (d_present_value_[i]) { cudaFree(d_present_value_[i]); d_present_value_[i] = nullptr; }
    }
    for (int i = 0; i < 15; i++) {
        if (d_lm_heads_[i]) { cudaFree(d_lm_heads_[i]); d_lm_heads_[i] = nullptr; }
        if (d_codec_embds_[i]) { cudaFree(d_codec_embds_[i]); d_codec_embds_[i] = nullptr; }
        codec_embd_vocab_[i] = 0;
    }
    n_codec_embds_ = 0;
    codec_embd_dim_ = 0;

    if (cublas_handle_) { cublasDestroy(cublas_handle_); cublas_handle_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    if (context_) { delete context_; context_ = nullptr; }
    if (engine_) { delete engine_; engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
    loaded_ = false;
    has_mtp_proj_ = false;
}

bool TRTCodePredictor::load_engine(const std::string & engine_path) {
    unload();

    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        error_msg_ = "Failed to open TRT code predictor engine: " + engine_path;
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    if (!file.read(engine_data.data(), size)) {
        error_msg_ = "Failed to read TRT engine file";
        return false;
    }
    file.close();

    runtime_ = nvinfer1::createInferRuntime(g_cp_trt_logger);
    if (!runtime_) { error_msg_ = "Failed to create TRT runtime"; return false; }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) { error_msg_ = "Failed to deserialize TRT code predictor engine"; return false; }

    context_ = engine_->createExecutionContext();
    if (!context_) { error_msg_ = "Failed to create TRT execution context"; return false; }

    // Create dedicated CUDA stream
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        error_msg_ = "Failed to create CUDA stream";
        return false;
    }

    // Create cuBLAS handle and bind to stream
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        error_msg_ = "Failed to create cuBLAS handle";
        return false;
    }
    cublasSetStream(cublas_handle_, stream_);

    // Allocate TRT I/O buffers
    size_t hidden_bytes = 1 * 1 * hidden_size_ * sizeof(float);
    size_t pos_bytes = 1 * sizeof(int64_t);
    size_t kv_bytes = 1 * n_kv_heads_ * max_kv_ * head_dim_ * sizeof(float);

    if (cudaMalloc(&d_hidden_in_, hidden_bytes) != cudaSuccess ||
        cudaMalloc(&d_position_id_, pos_bytes) != cudaSuccess ||
        cudaMalloc(&d_hidden_out_, hidden_bytes) != cudaSuccess ||
        cudaMalloc(&d_hidden_out_fp16_, hidden_size_ * sizeof(uint16_t)) != cudaSuccess) {
        error_msg_ = "Failed to allocate GPU memory for TRT I/O";
        return false;
    }

    for (int i = 0; i < n_layers_; i++) {
        if (cudaMalloc(&d_past_key_[i], kv_bytes) != cudaSuccess ||
            cudaMalloc(&d_past_value_[i], kv_bytes) != cudaSuccess ||
            cudaMalloc(&d_present_key_[i], kv_bytes) != cudaSuccess ||
            cudaMalloc(&d_present_value_[i], kv_bytes) != cudaSuccess) {
            error_msg_ = "Failed to allocate GPU memory for KV cache";
            return false;
        }
    }

    // Allocate greedy/sampling loop buffers
    if (cudaMalloc(&d_token_id_, sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_all_token_ids_, 15 * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_rand_vals_, 15 * sizeof(float)) != cudaSuccess) {
        error_msg_ = "Failed to allocate GPU memory for greedy/sampling loop";
        return false;
    }

    reset_kv_cache();

    fprintf(stderr, "  TRT code predictor loaded: %s (%.1f MB engine)\n",
            engine_path.c_str(), (float)size / 1024.0f / 1024.0f);

    loaded_ = true;
    return true;
}

bool TRTCodePredictor::upload_lm_heads(const std::vector<std::vector<float>> & weights,
                                        int32_t vocab_size, int32_t hidden_size) {
    vocab_size_ = vocab_size;
    size_t head_bytes_fp16 = (size_t)vocab_size * hidden_size * sizeof(uint16_t);

    // Allocate FP32 logits buffer
    if (cudaMalloc(&d_logits_, vocab_size * sizeof(float)) != cudaSuccess) {
        error_msg_ = "Failed to allocate GPU memory for logits";
        return false;
    }

    // Upload FP32 weights to temp GPU buffer, convert to FP16 on GPU
    size_t head_bytes_fp32 = (size_t)vocab_size * hidden_size * sizeof(float);
    void * d_temp_fp32 = nullptr;
    if (cudaMalloc(&d_temp_fp32, head_bytes_fp32) != cudaSuccess) {
        error_msg_ = "Failed to allocate temp GPU buffer for FP16 conversion";
        return false;
    }

    for (int i = 0; i < (int)weights.size() && i < 15; i++) {
        if (cudaMalloc(&d_lm_heads_[i], head_bytes_fp16) != cudaSuccess) {
            cudaFree(d_temp_fp32);
            error_msg_ = "Failed to allocate GPU memory for lm_head " + std::to_string(i);
            return false;
        }
        // Upload FP32 to temp, convert to FP16 on GPU
        cudaMemcpyAsync(d_temp_fp32, weights[i].data(), head_bytes_fp32,
                        cudaMemcpyHostToDevice, stream_);
        gpu_fp32_to_fp16((const float *)d_temp_fp32, d_lm_heads_[i],
                         vocab_size * hidden_size, stream_);
    }
    cudaStreamSynchronize(stream_);
    cudaFree(d_temp_fp32);

    fprintf(stderr, "  TRT code predictor: %d lm_heads uploaded as FP16 (%.1f MB total)\n",
            (int)weights.size(),
            (float)(weights.size() * head_bytes_fp16) / 1024.0f / 1024.0f);
    return true;
}

bool TRTCodePredictor::upload_mtp_proj(const std::vector<float> & weight,
                                        const std::vector<float> & bias,
                                        int32_t cp_hidden, int32_t talker_hidden) {
    talker_hidden_ = talker_hidden;
    size_t w_bytes = (size_t)cp_hidden * talker_hidden * sizeof(float);

    if (cudaMalloc(&d_mtp_proj_w_, w_bytes) != cudaSuccess) {
        error_msg_ = "Failed to allocate GPU memory for mtp_proj weight";
        return false;
    }
    if (cudaMemcpy(d_mtp_proj_w_, weight.data(), w_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        error_msg_ = "Failed to copy mtp_proj weight to GPU";
        return false;
    }

    if (!bias.empty()) {
        if (cudaMalloc(&d_mtp_proj_b_, cp_hidden * sizeof(float)) != cudaSuccess) {
            error_msg_ = "Failed to allocate GPU memory for mtp_proj bias";
            return false;
        }
        if (cudaMemcpy(d_mtp_proj_b_, bias.data(), cp_hidden * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            error_msg_ = "Failed to copy mtp_proj bias to GPU";
            return false;
        }
    }

    // Allocate staging buffer for talker-space input
    if (cudaMalloc(&d_proj_input_, talker_hidden * sizeof(float)) != cudaSuccess) {
        error_msg_ = "Failed to allocate GPU memory for proj input";
        return false;
    }

    has_mtp_proj_ = true;
    return true;
}

bool TRTCodePredictor::upload_codec_embeddings(const std::vector<std::vector<float>> & weights,
                                                int32_t embd_dim) {
    codec_embd_dim_ = embd_dim;
    n_codec_embds_ = (int32_t)weights.size();
    if (n_codec_embds_ > 15) n_codec_embds_ = 15;

    size_t total_bytes = 0;
    for (int i = 0; i < n_codec_embds_; i++) {
        size_t bytes = weights[i].size() * sizeof(float);
        if (cudaMalloc(&d_codec_embds_[i], bytes) != cudaSuccess) {
            error_msg_ = "Failed to allocate GPU memory for codec embedding " + std::to_string(i);
            return false;
        }
        if (cudaMemcpy(d_codec_embds_[i], weights[i].data(), bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            error_msg_ = "Failed to copy codec embedding to GPU";
            return false;
        }
        codec_embd_vocab_[i] = (int32_t)(weights[i].size() / embd_dim);
        total_bytes += bytes;
    }

    fprintf(stderr, "  TRT code predictor: %d codec embeddings on GPU (%.1f MB total)\n",
            n_codec_embds_, (float)total_bytes / 1024.0f / 1024.0f);
    return true;
}

void TRTCodePredictor::warmup_cublas() {
    if (!loaded_ || !d_lm_heads_[0] || vocab_size_ == 0) return;

    // Dummy FP16 convert + GemmEx to trigger cuBLAS JIT compilation
    gpu_fp32_to_fp16((const float *)d_hidden_out_, d_hidden_out_fp16_,
                     hidden_size_, stream_);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_handle_,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 vocab_size_, 1, hidden_size_,
                 &alpha,
                 d_lm_heads_[0], CUDA_R_16F, hidden_size_,
                 d_hidden_out_fp16_, CUDA_R_16F, hidden_size_,
                 &beta,
                 d_logits_, CUDA_R_32F, vocab_size_,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaStreamSynchronize(stream_);
    fprintf(stderr, "  TRT code predictor: cuBLAS warmed up\n");
}

void TRTCodePredictor::reset_kv_cache() {
    size_t kv_bytes = 1 * n_kv_heads_ * max_kv_ * head_dim_ * sizeof(float);
    for (int i = 0; i < n_layers_; i++) {
        if (d_past_key_[i]) cudaMemset(d_past_key_[i], 0, kv_bytes);
        if (d_past_value_[i]) cudaMemset(d_past_value_[i], 0, kv_bytes);
        if (d_present_key_[i]) cudaMemset(d_present_key_[i], 0, kv_bytes);
        if (d_present_value_[i]) cudaMemset(d_present_value_[i], 0, kv_bytes);
    }
}

// Shared inference logic: d_hidden_in_ must already be populated.
// Runs TRT inference → KV swap → FP16 convert → cuBLAS lm_head → D2H output.
bool TRTCodePredictor::run_trt_and_lm_head(int32_t position_id, int32_t lm_head_idx,
                                            float * output, int32_t output_size) {
    // Upload position_id
    int64_t pos_i64 = position_id;
    cudaMemcpyAsync(d_position_id_, &pos_i64, sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream_);

    // Set TRT tensor addresses
    context_->setTensorAddress("hidden_states", d_hidden_in_);
    context_->setTensorAddress("position_id", d_position_id_);
    context_->setTensorAddress("output", d_hidden_out_);

    for (int i = 0; i < n_layers_; i++) {
        char name[64];
        snprintf(name, sizeof(name), "past_key_%d", i);
        context_->setTensorAddress(name, d_past_key_[i]);
        snprintf(name, sizeof(name), "past_value_%d", i);
        context_->setTensorAddress(name, d_past_value_[i]);
        snprintf(name, sizeof(name), "present_key_%d", i);
        context_->setTensorAddress(name, d_present_key_[i]);
        snprintf(name, sizeof(name), "present_value_%d", i);
        context_->setTensorAddress(name, d_present_value_[i]);
    }

    // Run TRT inference on dedicated stream
    if (!context_->enqueueV3(stream_)) {
        error_msg_ = "TRT code predictor inference failed";
        return false;
    }

    // Swap past ↔ present pointers (zero-copy KV cache update)
    for (int i = 0; i < n_layers_; i++) {
        std::swap(d_past_key_[i], d_present_key_[i]);
        std::swap(d_past_value_[i], d_present_value_[i]);
    }

    // Apply lm_head or return hidden states
    if (lm_head_idx >= 0 && lm_head_idx < 15 && d_lm_heads_[lm_head_idx]) {
        // Convert FP32 hidden_out → FP16 on GPU
        gpu_fp32_to_fp16((const float *)d_hidden_out_, d_hidden_out_fp16_,
                         hidden_size_, stream_);

        // cuBLAS GemmEx: FP16 weights × FP16 hidden → FP32 logits
        float alpha = 1.0f, beta = 0.0f;
        if (cublasGemmEx(cublas_handle_,
                         CUBLAS_OP_T, CUBLAS_OP_N,
                         vocab_size_, 1, hidden_size_,
                         &alpha,
                         d_lm_heads_[lm_head_idx], CUDA_R_16F, hidden_size_,
                         d_hidden_out_fp16_, CUDA_R_16F, hidden_size_,
                         &beta,
                         d_logits_, CUDA_R_32F, vocab_size_,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
            error_msg_ = "cuBLAS lm_head GemmEx failed";
            return false;
        }

        cudaStreamSynchronize(stream_);

        if (cudaMemcpy(output, d_logits_, output_size * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            error_msg_ = "Failed to copy logits from GPU";
            return false;
        }
    } else {
        // Return hidden states
        cudaStreamSynchronize(stream_);
        if (cudaMemcpy(output, d_hidden_out_, output_size * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            error_msg_ = "Failed to copy hidden_out from GPU";
            return false;
        }
    }

    return true;
}

bool TRTCodePredictor::step(const float * input, int32_t input_size,
                             int32_t position_id, int32_t lm_head_idx,
                             bool needs_proj,
                             float * output, int32_t output_size) {
    if (!loaded_) {
        error_msg_ = "TRT code predictor not loaded";
        return false;
    }

    // Load input to GPU and optionally project through mtp_proj
    if (needs_proj && has_mtp_proj_) {
        cudaMemcpyAsync(d_proj_input_, input, input_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        float alpha = 1.0f, beta = 0.0f;
        if (cublasSgemv(cublas_handle_, CUBLAS_OP_T,
                        talker_hidden_, hidden_size_,
                        &alpha,
                        (const float *)d_mtp_proj_w_, talker_hidden_,
                        (const float *)d_proj_input_, 1,
                        &beta,
                        (float *)d_hidden_in_, 1) != CUBLAS_STATUS_SUCCESS) {
            error_msg_ = "cuBLAS mtp_proj gemv failed";
            return false;
        }

        if (d_mtp_proj_b_) {
            if (add_bias_cublas(cublas_handle_, (float *)d_hidden_in_,
                                (const float *)d_mtp_proj_b_,
                                hidden_size_) != CUBLAS_STATUS_SUCCESS) {
                error_msg_ = "cuBLAS mtp_proj bias add failed";
                return false;
            }
        }
    } else {
        cudaMemcpyAsync(d_hidden_in_, input, hidden_size_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }

    return run_trt_and_lm_head(position_id, lm_head_idx, output, output_size);
}

bool TRTCodePredictor::step_with_token(int32_t token_id, int32_t embedding_idx,
                                        int32_t position_id, int32_t lm_head_idx,
                                        float * output, int32_t output_size) {
    if (!loaded_) {
        error_msg_ = "TRT code predictor not loaded";
        return false;
    }
    if (embedding_idx < 0 || embedding_idx >= n_codec_embds_ || !d_codec_embds_[embedding_idx]) {
        error_msg_ = "Invalid embedding index: " + std::to_string(embedding_idx);
        return false;
    }
    if (token_id < 0 || token_id >= codec_embd_vocab_[embedding_idx]) {
        error_msg_ = "Token ID out of range: " + std::to_string(token_id);
        return false;
    }

    // D2D copy embedding row from GPU table
    size_t offset = (size_t)token_id * codec_embd_dim_ * sizeof(float);

    if (has_mtp_proj_) {
        // Copy to proj_input staging buffer, then project
        cudaMemcpyAsync(d_proj_input_,
                        (const char *)d_codec_embds_[embedding_idx] + offset,
                        codec_embd_dim_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);

        float alpha = 1.0f, beta = 0.0f;
        if (cublasSgemv(cublas_handle_, CUBLAS_OP_T,
                        talker_hidden_, hidden_size_,
                        &alpha,
                        (const float *)d_mtp_proj_w_, talker_hidden_,
                        (const float *)d_proj_input_, 1,
                        &beta,
                        (float *)d_hidden_in_, 1) != CUBLAS_STATUS_SUCCESS) {
            error_msg_ = "cuBLAS mtp_proj gemv failed";
            return false;
        }

        if (d_mtp_proj_b_) {
            if (add_bias_cublas(cublas_handle_, (float *)d_hidden_in_,
                                (const float *)d_mtp_proj_b_,
                                hidden_size_) != CUBLAS_STATUS_SUCCESS) {
                error_msg_ = "cuBLAS mtp_proj bias add failed";
                return false;
            }
        }
    } else {
        // No projection: copy directly to hidden_in (talker_hidden == cp_hidden)
        cudaMemcpyAsync(d_hidden_in_,
                        (const char *)d_codec_embds_[embedding_idx] + offset,
                        hidden_size_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    return run_trt_and_lm_head(position_id, lm_head_idx, output, output_size);
}


bool TRTCodePredictor::run_greedy_loop(const float * hidden, int32_t hidden_size,
                                        int32_t cb0_token, bool needs_proj,
                                        int32_t * output_tokens, int32_t n_outputs) {
    if (!loaded_) return false;

    reset_kv_cache();

    auto run_trt_step = [&](int32_t position) {
        int64_t pos_i64 = position;
        cudaMemcpyAsync(d_position_id_, &pos_i64, sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream_);

        for (int i = 0; i < n_layers_; i++) {
            char name[64];
            snprintf(name, sizeof(name), "past_key_%d", i);
            context_->setTensorAddress(name, d_past_key_[i]);
            snprintf(name, sizeof(name), "past_value_%d", i);
            context_->setTensorAddress(name, d_past_value_[i]);
            snprintf(name, sizeof(name), "present_key_%d", i);
            context_->setTensorAddress(name, d_present_key_[i]);
            snprintf(name, sizeof(name), "present_value_%d", i);
            context_->setTensorAddress(name, d_present_value_[i]);
        }

        context_->setTensorAddress("hidden_states", d_hidden_in_);
        context_->setTensorAddress("position_id", d_position_id_);
        context_->setTensorAddress("output", d_hidden_out_);

        context_->enqueueV3(stream_);

        for (int i = 0; i < n_layers_; i++) {
            std::swap(d_past_key_[i], d_present_key_[i]);
            std::swap(d_past_value_[i], d_present_value_[i]);
        }
    };

    auto lm_head_and_argmax = [&](int32_t head_idx, int32_t * token_out) {
        gpu_fp32_to_fp16((const float *)d_hidden_out_, d_hidden_out_fp16_,
                         hidden_size_, stream_);
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle_,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     vocab_size_, 1, hidden_size_,
                     &alpha,
                     d_lm_heads_[head_idx], CUDA_R_16F, hidden_size_,
                     d_hidden_out_fp16_, CUDA_R_16F, hidden_size_,
                     &beta,
                     d_logits_, CUDA_R_32F, vocab_size_,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        gpu_argmax_f32((const float *)d_logits_, token_out, vocab_size_, stream_);
    };

    auto apply_mtp_proj = [&]() {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(cublas_handle_, CUBLAS_OP_T,
                    talker_hidden_, hidden_size_,
                    &alpha,
                    (const float *)d_mtp_proj_w_, talker_hidden_,
                    (const float *)d_proj_input_, 1,
                    &beta,
                    (float *)d_hidden_in_, 1);
        if (d_mtp_proj_b_) {
            add_bias_cublas(cublas_handle_, (float *)d_hidden_in_,
                            (const float *)d_mtp_proj_b_, hidden_size_);
        }
    };

    // Step 0: feed talker hidden state, no lm_head
    if (needs_proj && has_mtp_proj_) {
        cudaMemcpyAsync(d_proj_input_, hidden, hidden_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        apply_mtp_proj();
    } else {
        cudaMemcpyAsync(d_hidden_in_, hidden, hidden_size_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }
    run_trt_step(0);

    // Step 1: CB0 embedding -> TRT -> lm_head[0] -> argmax
    cudaMemcpyAsync(d_token_id_, &cb0_token, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream_);

    float * embd_target = has_mtp_proj_ ? (float *)d_proj_input_ : (float *)d_hidden_in_;
    gpu_embedding_lookup_by_gpu_id(
        (const int32_t *)d_token_id_,
        (const float *)d_codec_embds_[0],
        embd_target, codec_embd_dim_, stream_);

    if (has_mtp_proj_) apply_mtp_proj();

    run_trt_step(1);
    lm_head_and_argmax(0, (int32_t *)d_all_token_ids_);

    // Steps 2-15: fully on GPU, no CPU sync
    for (int step = 1; step < n_outputs; step++) {
        int32_t * prev_token = (int32_t *)d_all_token_ids_ + (step - 1);

        embd_target = has_mtp_proj_ ? (float *)d_proj_input_ : (float *)d_hidden_in_;
        gpu_embedding_lookup_by_gpu_id(
            prev_token,
            (const float *)d_codec_embds_[step],
            embd_target, codec_embd_dim_, stream_);

        if (has_mtp_proj_) apply_mtp_proj();

        run_trt_step(step + 1);
        lm_head_and_argmax(step, (int32_t *)d_all_token_ids_ + step);
    }

    // Single sync + batch D2H
    cudaStreamSynchronize(stream_);
    cudaMemcpy(output_tokens, d_all_token_ids_,
               n_outputs * sizeof(int32_t), cudaMemcpyDeviceToHost);

    return true;
}

bool TRTCodePredictor::run_sampling_loop(const float * hidden, int32_t hidden_size,
                                          int32_t cb0_token, bool needs_proj,
                                          float temperature, int32_t top_k,
                                          const float * rand_vals,
                                          int32_t * output_tokens, int32_t n_outputs) {
    if (!loaded_) return false;

    reset_kv_cache();

    // Upload pre-generated random values (async, no sync needed)
    cudaMemcpyAsync(d_rand_vals_, rand_vals, n_outputs * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    auto run_trt_step = [&](int32_t position) {
        int64_t pos_i64 = position;
        cudaMemcpyAsync(d_position_id_, &pos_i64, sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream_);

        for (int i = 0; i < n_layers_; i++) {
            char name[64];
            snprintf(name, sizeof(name), "past_key_%d", i);
            context_->setTensorAddress(name, d_past_key_[i]);
            snprintf(name, sizeof(name), "past_value_%d", i);
            context_->setTensorAddress(name, d_past_value_[i]);
            snprintf(name, sizeof(name), "present_key_%d", i);
            context_->setTensorAddress(name, d_present_key_[i]);
            snprintf(name, sizeof(name), "present_value_%d", i);
            context_->setTensorAddress(name, d_present_value_[i]);
        }

        context_->setTensorAddress("hidden_states", d_hidden_in_);
        context_->setTensorAddress("position_id", d_position_id_);
        context_->setTensorAddress("output", d_hidden_out_);

        context_->enqueueV3(stream_);

        for (int i = 0; i < n_layers_; i++) {
            std::swap(d_past_key_[i], d_present_key_[i]);
            std::swap(d_past_value_[i], d_present_value_[i]);
        }
    };

    auto lm_head_and_sample = [&](int32_t head_idx, int32_t * token_out, int step_idx) {
        gpu_fp32_to_fp16((const float *)d_hidden_out_, d_hidden_out_fp16_,
                         hidden_size_, stream_);
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle_,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     vocab_size_, 1, hidden_size_,
                     &alpha,
                     d_lm_heads_[head_idx], CUDA_R_16F, hidden_size_,
                     d_hidden_out_fp16_, CUDA_R_16F, hidden_size_,
                     &beta,
                     d_logits_, CUDA_R_32F, vocab_size_,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        gpu_sample_topk_f32((const float *)d_logits_,
                            (const float *)d_rand_vals_ + step_idx,
                            token_out, temperature, top_k, vocab_size_, stream_);
    };

    auto apply_mtp_proj = [&]() {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(cublas_handle_, CUBLAS_OP_T,
                    talker_hidden_, hidden_size_,
                    &alpha,
                    (const float *)d_mtp_proj_w_, talker_hidden_,
                    (const float *)d_proj_input_, 1,
                    &beta,
                    (float *)d_hidden_in_, 1);
        if (d_mtp_proj_b_) {
            add_bias_cublas(cublas_handle_, (float *)d_hidden_in_,
                            (const float *)d_mtp_proj_b_, hidden_size_);
        }
    };

    // Step 0: feed talker hidden state, no lm_head
    if (needs_proj && has_mtp_proj_) {
        cudaMemcpyAsync(d_proj_input_, hidden, hidden_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        apply_mtp_proj();
    } else {
        cudaMemcpyAsync(d_hidden_in_, hidden, hidden_size_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
    }
    run_trt_step(0);

    // Step 1: CB0 embedding -> TRT -> lm_head[0] -> sample
    cudaMemcpyAsync(d_token_id_, &cb0_token, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream_);

    float * embd_target = has_mtp_proj_ ? (float *)d_proj_input_ : (float *)d_hidden_in_;
    gpu_embedding_lookup_by_gpu_id(
        (const int32_t *)d_token_id_,
        (const float *)d_codec_embds_[0],
        embd_target, codec_embd_dim_, stream_);

    if (has_mtp_proj_) apply_mtp_proj();

    run_trt_step(1);
    lm_head_and_sample(0, (int32_t *)d_all_token_ids_, 0);

    // Steps 2-15: fully on GPU, no CPU sync
    for (int step = 1; step < n_outputs; step++) {
        int32_t * prev_token = (int32_t *)d_all_token_ids_ + (step - 1);

        embd_target = has_mtp_proj_ ? (float *)d_proj_input_ : (float *)d_hidden_in_;
        gpu_embedding_lookup_by_gpu_id(
            prev_token,
            (const float *)d_codec_embds_[step],
            embd_target, codec_embd_dim_, stream_);

        if (has_mtp_proj_) apply_mtp_proj();

        run_trt_step(step + 1);
        lm_head_and_sample(step, (int32_t *)d_all_token_ids_ + step, step);
    }

    // Single sync + batch D2H
    cudaStreamSynchronize(stream_);
    cudaMemcpy(output_tokens, d_all_token_ids_,
               n_outputs * sizeof(int32_t), cudaMemcpyDeviceToHost);

    return true;
}

} // namespace qwen3_tts

