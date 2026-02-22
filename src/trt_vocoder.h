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

namespace qwen3_tts {

// TensorRT-based vocoder decoder
// Drop-in replacement for AudioTokenizerDecoder::decode()
class TRTVocoderDecoder {
public:
    TRTVocoderDecoder();
    ~TRTVocoderDecoder();

    // Load TRT engine from file
    // engine_path: path to serialized TRT engine (.trt)
    // fixed_frames: the fixed sequence length this engine was built for
    bool load_engine(const std::string & engine_path, int32_t fixed_frames);

    // Decode audio codes to waveform
    // codes: [n_frames, n_codebooks] row-major int32_t
    // n_codebooks: number of codebooks (16)
    // n_frames: number of frames (can differ from fixed_frames; will pad/chunk)
    // samples: output audio samples normalized to [-1, 1]
    bool decode(const int32_t * codes, int32_t n_frames, int32_t n_codebooks,
                std::vector<float> & samples);

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }
    int32_t get_fixed_frames() const { return fixed_frames_; }

    // Release all resources
    void unload();

private:
    nvinfer1::IRuntime * runtime_ = nullptr;
    nvinfer1::ICudaEngine * engine_ = nullptr;
    nvinfer1::IExecutionContext * context_ = nullptr;

    int32_t fixed_frames_ = 0;
    int32_t n_codebooks_ = 16;
    int32_t samples_per_frame_ = 1920;  // 24kHz / 12.5Hz

    // GPU buffers
    void * d_codes_ = nullptr;     // input: [1, 16, fixed_frames] int64
    void * d_audio_ = nullptr;     // output: [1, 1, fixed_frames * 1920] float

    bool loaded_ = false;
    std::string error_msg_;
};

} // namespace qwen3_tts
