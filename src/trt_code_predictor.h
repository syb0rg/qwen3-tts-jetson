#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Forward declarations for TRT types
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

struct cublasContext;
typedef struct cublasContext * cublasHandle_t;

struct CUstream_st;
typedef struct CUstream_st * cudaStream_t;

namespace qwen3_tts {

// TensorRT-based code predictor (5 shared transformer layers + output norm)
// Includes GPU-side lm_head matmul via cuBLAS and optional mtp_proj.
// Uses FP16 lm_head weights, dedicated CUDA stream, and GPU codec embeddings.
class TRTCodePredictor {
public:
    TRTCodePredictor();
    ~TRTCodePredictor();

    // Load TRT engine from file
    bool load_engine(const std::string & engine_path);

    // Upload lm_head weights to GPU as FP16 (called once after model load)
    // weights[i]: row-major [vocab_size, hidden_size] float data for codebook i
    bool upload_lm_heads(const std::vector<std::vector<float>> & weights,
                         int32_t vocab_size, int32_t hidden_size);

    // Upload mtp_proj weights to GPU (for 1.7B model, optional)
    bool upload_mtp_proj(const std::vector<float> & weight,
                         const std::vector<float> & bias,
                         int32_t cp_hidden, int32_t talker_hidden);

    // Upload codec embedding tables to GPU (called once after model load)
    // weights[0]: codec_embd (codebook 0), weights[1..14]: code_pred_embd[0..13]
    // embd_dim: embedding dimension (talker_hidden)
    bool upload_codec_embeddings(const std::vector<std::vector<float>> & weights,
                                 int32_t embd_dim);

    // Reset KV cache (call at start of each frame)
    void reset_kv_cache();

    // Run one step with CPU input: project → TRT layers → lm_head → logits
    bool step(const float * input, int32_t input_size,
              int32_t position_id, int32_t lm_head_idx,
              bool needs_proj,
              float * output, int32_t output_size);

    // Run one step using GPU codec embedding lookup (no CPU→GPU transfer)
    // token_id: which row to look up in the embedding table
    // embedding_idx: which embedding table (0=codec_embd, 1..14=code_pred_embd)
    // Always applies mtp_proj if available
    bool step_with_token(int32_t token_id, int32_t embedding_idx,
                         int32_t position_id, int32_t lm_head_idx,
                         float * output, int32_t output_size);

    // Full greedy autoregressive loop (GPU-only, 1 sync at end)
    // Returns n_outputs token IDs via output_tokens array.
    bool run_greedy_loop(const float * hidden, int32_t hidden_size,
                         int32_t cb0_token, bool needs_proj,
                         int32_t * output_tokens, int32_t n_outputs);

    // Full sampling autoregressive loop (GPU-only, 1 sync at end)
    // Like run_greedy_loop but with temperature + top_k sampling.
    // rand_vals: [n_outputs] pre-generated uniform random floats in [0,1)
    bool run_sampling_loop(const float * hidden, int32_t hidden_size,
                           int32_t cb0_token, bool needs_proj,
                           float temperature, int32_t top_k,
                           const float * rand_vals,
                           int32_t * output_tokens, int32_t n_outputs);

    // Pre-warm cuBLAS (call after upload_lm_heads to avoid first-frame JIT penalty)
    void warmup_cublas();

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }

    void unload();

private:
    // Shared TRT inference + lm_head logic (input must already be in d_hidden_in_)
    bool run_trt_and_lm_head(int32_t position_id, int32_t lm_head_idx,
                              float * output, int32_t output_size);

    nvinfer1::IRuntime * runtime_ = nullptr;
    nvinfer1::ICudaEngine * engine_ = nullptr;
    nvinfer1::IExecutionContext * context_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Architecture constants
    int32_t n_layers_ = 5;
    int32_t hidden_size_ = 1024;   // code_pred hidden
    int32_t n_kv_heads_ = 8;
    int32_t head_dim_ = 128;
    int32_t max_kv_ = 16;
    int32_t vocab_size_ = 0;
    int32_t talker_hidden_ = 0;

    // GPU buffers - TRT I/O
    void * d_hidden_in_ = nullptr;      // [1, 1, hidden_size] float
    void * d_position_id_ = nullptr;    // [1] int64
    void * d_hidden_out_ = nullptr;     // [1, 1, hidden_size] float

    // KV cache GPU buffers: [1, n_kv_heads, max_kv, head_dim] per layer
    void * d_past_key_[5] = {};
    void * d_past_value_[5] = {};
    void * d_present_key_[5] = {};
    void * d_present_value_[5] = {};

    // GPU buffers - FP16 lm_heads and conversion
    void * d_lm_heads_[15] = {};       // [vocab_size, hidden_size] FP16 per head
    void * d_logits_ = nullptr;         // [vocab_size] FP32
    void * d_hidden_out_fp16_ = nullptr; // [hidden_size] FP16 conversion buffer

    // GPU buffers - mtp_proj (FP32)
    void * d_mtp_proj_w_ = nullptr;  // [cp_hidden, talker_hidden] float
    void * d_mtp_proj_b_ = nullptr;  // [cp_hidden] float
    void * d_proj_input_ = nullptr;  // [talker_hidden] float (input staging)
    bool has_mtp_proj_ = false;

    // GPU codec embeddings
    void * d_codec_embds_[15] = {};    // [vocab_size * embd_dim] FP32 per codebook
    int32_t codec_embd_vocab_[15] = {}; // vocab_size per embedding table
    int32_t n_codec_embds_ = 0;
    int32_t codec_embd_dim_ = 0;       // embedding dimension (talker_hidden)

    // GPU buffers for greedy autoregressive loop
    void * d_token_id_ = nullptr;        // [1] int32 - current token on GPU
    void * d_all_token_ids_ = nullptr;   // [15] int32 - output token IDs
    void * d_rand_vals_ = nullptr;       // [15] float - pre-generated random values

    bool loaded_ = false;
    std::string error_msg_;
};

} // namespace qwen3_tts
