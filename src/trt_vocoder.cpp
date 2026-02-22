#include "trt_vocoder.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace qwen3_tts {

// TRT logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char * msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

static TRTLogger g_trt_logger;

TRTVocoderDecoder::TRTVocoderDecoder() = default;

TRTVocoderDecoder::~TRTVocoderDecoder() {
    unload();
}

void TRTVocoderDecoder::unload() {
    if (d_codes_) { cudaFree(d_codes_); d_codes_ = nullptr; }
    if (d_audio_) { cudaFree(d_audio_); d_audio_ = nullptr; }
    if (context_) { delete context_; context_ = nullptr; }
    if (engine_) { delete engine_; engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
    loaded_ = false;
}

bool TRTVocoderDecoder::load_engine(const std::string & engine_path, int32_t fixed_frames) {
    unload();
    fixed_frames_ = fixed_frames;

    // Read engine file
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        error_msg_ = "Failed to open TRT engine: " + engine_path;
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

    // Create runtime and engine
    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) {
        error_msg_ = "Failed to create TRT runtime";
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        error_msg_ = "Failed to deserialize TRT engine";
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        error_msg_ = "Failed to create TRT execution context";
        return false;
    }

    // Allocate GPU buffers
    size_t codes_bytes = 1 * n_codebooks_ * fixed_frames_ * sizeof(int64_t);
    size_t audio_bytes = 1 * 1 * fixed_frames_ * samples_per_frame_ * sizeof(float);

    if (cudaMalloc(&d_codes_, codes_bytes) != cudaSuccess) {
        error_msg_ = "Failed to allocate GPU memory for codes";
        return false;
    }
    if (cudaMalloc(&d_audio_, audio_bytes) != cudaSuccess) {
        error_msg_ = "Failed to allocate GPU memory for audio";
        return false;
    }

    fprintf(stderr, "  TRT vocoder loaded: %s (%d fixed frames, %.1f MB engine)\n",
            engine_path.c_str(), fixed_frames_, (float)size / 1024.0f / 1024.0f);

    loaded_ = true;
    return true;
}

bool TRTVocoderDecoder::decode(const int32_t * codes, int32_t n_frames,
                                int32_t n_codebooks,
                                std::vector<float> & samples) {
    if (!loaded_) {
        error_msg_ = "TRT vocoder not loaded";
        return false;
    }

    n_codebooks_ = n_codebooks;
    samples.clear();

    // Process in chunks of fixed_frames_
    int32_t offset = 0;
    while (offset < n_frames) {
        int32_t chunk_frames = std::min(fixed_frames_, n_frames - offset);

        // Prepare input: convert int32 codes to int64 and transpose
        // Input codes: [n_frames, n_codebooks] row-major
        // TRT expects: [1, n_codebooks, fixed_frames_] int64
        std::vector<int64_t> codes_i64(n_codebooks_ * fixed_frames_, 0);
        for (int32_t f = 0; f < chunk_frames; f++) {
            for (int32_t c = 0; c < n_codebooks_; c++) {
                codes_i64[c * fixed_frames_ + f] = codes[(offset + f) * n_codebooks_ + c];
            }
        }
        // Pad remaining frames with zeros (already zeroed)

        // Copy to GPU
        size_t codes_bytes = n_codebooks_ * fixed_frames_ * sizeof(int64_t);
        if (cudaMemcpy(d_codes_, codes_i64.data(), codes_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            error_msg_ = "Failed to copy codes to GPU";
            return false;
        }

        // Set tensor addresses
        if (!context_->setTensorAddress("codes", d_codes_)) {
            error_msg_ = "Failed to set input tensor address";
            return false;
        }
        if (!context_->setTensorAddress("audio", d_audio_)) {
            error_msg_ = "Failed to set output tensor address";
            return false;
        }

        // Run inference
        if (!context_->enqueueV3(0)) {
            error_msg_ = "TRT inference failed";
            return false;
        }
        cudaStreamSynchronize(0);

        // Copy output back
        int32_t total_samples = fixed_frames_ * samples_per_frame_;
        int32_t valid_samples = chunk_frames * samples_per_frame_;
        std::vector<float> chunk_audio(total_samples);
        if (cudaMemcpy(chunk_audio.data(), d_audio_,
                       total_samples * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            error_msg_ = "Failed to copy audio from GPU";
            return false;
        }

        // Append only the valid portion
        samples.insert(samples.end(),
                       chunk_audio.begin(),
                       chunk_audio.begin() + valid_samples);

        offset += chunk_frames;
    }

    return true;
}

} // namespace qwen3_tts
