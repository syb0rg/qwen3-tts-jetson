#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "coreml_code_predictor.h"
#ifdef QWEN3_TTS_TENSORRT
#include "trt_code_predictor.h"
#endif

#include <string>
#include <map>
#include <vector>
#include <random>
#include <functional>
namespace qwen3_tts {

#define QWEN3_TTS_MAX_NODES 16384

// TTS Transformer configuration (Qwen2-based Talker)
struct tts_transformer_config {
    // Text embedding
    int32_t text_vocab_size = 151936;
    int32_t text_embd_dim = 2048;
    
    // Talker transformer
    int32_t hidden_size = 1024;
    int32_t n_layers = 28;
    int32_t n_attention_heads = 16;
    int32_t n_key_value_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    
    // M-RoPE sections [time, freq, channel] = [24, 20, 20]
    int32_t mrope_section[3] = {24, 20, 20};
    
    // Codec vocabulary
    int32_t codec_vocab_size = 3072;  // talker.codec_embd/codec_head
    int32_t n_codebooks = 16;
    
    // Code predictor
    int32_t code_pred_layers = 5;
    int32_t code_pred_vocab_size = 2048;  // Per-codebook vocab

    // Code predictor architecture (may differ from talker in 1.7B model)
    int32_t code_pred_hidden_size = 0;       // 0 = same as hidden_size (set in parse_config)
    int32_t code_pred_intermediate_size = 0; // 0 = same as intermediate_size
    int32_t code_pred_n_attention_heads = 0; // 0 = same as n_attention_heads
    int32_t code_pred_n_kv_heads = 0;        // 0 = same as n_key_value_heads
    int32_t code_pred_head_dim = 0;          // 0 = same as head_dim
    
    // Special codec tokens
    int32_t codec_pad_id = 2148;
    int32_t codec_bos_id = 2149;
    int32_t codec_eos_id = 2150;

    int32_t tts_bos_token_id = 151672;
    int32_t tts_eos_token_id = 151673;
    int32_t tts_pad_token_id = 151671;

    int32_t codec_think_id = 2154;
    int32_t codec_nothink_id = 2155;
    int32_t codec_think_bos_id = 2156;
    int32_t codec_think_eos_id = 2157;

    int32_t english_language_id = 2050;
};

// Transformer layer weights
struct transformer_layer {
    struct ggml_tensor * attn_norm = nullptr;
    
    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    
    struct ggml_tensor * ffn_norm = nullptr;
    
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

// TTS Transformer model weights
struct tts_transformer_model {
    tts_transformer_config config;
    
    // Text embedding and projection
    struct ggml_tensor * text_embd = nullptr;      // [text_embd_dim, text_vocab_size]
    struct ggml_tensor * text_proj_fc1 = nullptr;  // [text_embd_dim, text_embd_dim]
    struct ggml_tensor * text_proj_fc1_bias = nullptr;
    struct ggml_tensor * text_proj_fc2 = nullptr;  // [text_embd_dim, hidden_size]
    struct ggml_tensor * text_proj_fc2_bias = nullptr;
    
    // Codec embedding (for autoregressive input)
    struct ggml_tensor * codec_embd = nullptr;     // [hidden_size, codec_vocab_size]
    
    // Talker transformer layers
    std::vector<transformer_layer> layers;
    
    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr;    // [hidden_size]
    
    // Codec head (for first codebook prediction)
    struct ggml_tensor * codec_head = nullptr;     // [hidden_size, codec_vocab_size]
    
     // Code predictor layers
     std::vector<transformer_layer> code_pred_layers;

     // Code predictor output norm (final RMS norm before lm_head)
     struct ggml_tensor * code_pred_output_norm = nullptr;  // [code_pred_hidden_size]

     // Code predictor per-codebook embeddings and heads (15 codebooks, 0 uses talker output)
     std::vector<struct ggml_tensor *> code_pred_embd;  // [hidden_size, code_pred_vocab_size] x 15 (talker space)
     std::vector<struct ggml_tensor *> code_pred_head;  // [code_pred_hidden_size, code_pred_vocab_size] x 15

     // Code predictor input projection (talker hidden -> code pred hidden, only for 1.7B)
     struct ggml_tensor * code_pred_mtp_proj = nullptr;       // [hidden_size, code_pred_hidden_size]
     struct ggml_tensor * code_pred_mtp_proj_bias = nullptr;  // [code_pred_hidden_size]
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// KV cache for autoregressive generation
struct tts_kv_cache {
    std::vector<struct ggml_tensor *> k_cache;
    std::vector<struct ggml_tensor *> v_cache;
    
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int32_t n_ctx = 0;
    int32_t n_used = 0;
    int32_t head_dim = 128;
    int32_t n_kv_heads = 8;
    int32_t n_layers = 28;
};

// TTS Transformer state
struct tts_transformer_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    tts_kv_cache cache;           // Talker KV cache (28 layers)
    tts_kv_cache code_pred_cache; // Code predictor KV cache (5 layers)
};

// TTS Transformer class
class TTSTransformer {
public:
    TTSTransformer();
    ~TTSTransformer();
    
    // Load model from GGUF file
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Initialize KV cache
    bool init_kv_cache(int32_t n_ctx);
    
    // Clear KV cache
    void clear_kv_cache();
    
    // Initialize code predictor KV cache (5 layers, max 16 context)
    bool init_code_pred_kv_cache(int32_t n_ctx);
    
    // Clear code predictor KV cache
    void clear_code_pred_kv_cache();
    
    // Forward pass for text tokens (prefill phase)
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size] (optional, can be nullptr)
    // n_past: number of tokens already in KV cache
    // output: hidden states [n_tokens, hidden_size]
    bool forward_text(const int32_t * text_tokens, int32_t n_tokens,
                      const float * speaker_embd, int32_t n_past,
                      std::vector<float> & output);

    bool forward_prefill(const float * prefill_embd, int32_t n_tokens,
                         int32_t n_past, std::vector<float> & output,
                         std::vector<float> * logits_out = nullptr);
    
    // Forward pass for codec tokens (generation phase)
    // codec_token: single codec token for first codebook
    // n_past: number of tokens already in KV cache
    // output: logits for next codec token [codec_vocab_size]
    bool forward_codec(int32_t codec_token, int32_t n_past,
                       std::vector<float> & output);

    bool forward_step(const float * step_embd, int32_t n_past,
                      std::vector<float> & output,
                      std::vector<float> * hidden_out = nullptr);
    
    // Get hidden states from last forward pass (for code predictor)
    bool get_hidden_states(std::vector<float> & hidden) const;
    
    // Run code predictor autoregressively to generate 15 codes (codebooks 1-15)
    // hidden: hidden states from talker [hidden_size]
    // codebook_0_token: the codebook 0 token (used to create 2-token prefill input)
    // output: generated codes for codebooks 1-15 [15]
    bool predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token, 
                                       std::vector<int32_t> & output,
                                       float temperature = 0.9f,
                                       int32_t top_k = 50);
    
    // Frame callback: called with (codes, n_frames, n_codebooks), return false to stop
    using frame_callback_t = std::function<bool(const int32_t *, int32_t, int32_t)>;

    // Generate speech codes autoregressively
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size]
    // max_len: maximum number of frames to generate
    // output: generated speech codes [n_frames, n_codebooks]
    // on_frames: optional callback fired every callback_interval frames
    // callback_interval: frames between callback invocations (default: 40)
    bool generate(const int32_t * text_tokens, int32_t n_tokens,
                  const float * speaker_embd, int32_t max_len,
                  std::vector<int32_t> & output,
                  int32_t language_id = 2050,
                  float repetition_penalty = 1.05f,
                  float temperature = 0.9f,
                  int32_t top_k = 50,
                  frame_callback_t on_frames = nullptr,
                  int32_t callback_interval = 40);
    
    const tts_transformer_config & get_config() const { return model_.config; }
    
    const std::string & get_error() const { return error_msg_; }
    
private:
    bool try_init_coreml_code_predictor(const std::string & model_path);
    bool predict_codes_autoregressive_coreml(const float * hidden, int32_t codebook_0_token,
                                             std::vector<int32_t> & output,
                                             float temperature,
                                             int32_t top_k);

    bool build_prefill_graph(const int32_t * text_tokens, int32_t n_tokens,
                             const float * speaker_embd, int32_t language_id,
                             std::vector<float> & prefill_embd,
                             std::vector<float> & trailing_text_hidden,
                             std::vector<float> & tts_pad_embed);

    struct ggml_cgraph * build_prefill_forward_graph(int32_t n_tokens, int32_t n_past);

    struct ggml_cgraph * build_step_graph(int32_t n_past);

    bool project_text_tokens(const int32_t * text_tokens, int32_t n_tokens,
                             std::vector<float> & output);

    bool lookup_embedding_rows(struct ggml_tensor * embedding, const int32_t * token_ids,
                               int32_t n_tokens, const char * input_name,
                               const char * output_name, std::vector<float> & output);
    bool lookup_single_embedding_row(struct ggml_tensor * embedding, int32_t token_id,
                                     float * out_row);
    
    // Build computation graph for single-step autoregressive code predictor
    // n_past: number of tokens already in KV cache (0-14)
    // generation_step: which codebook we're predicting (0-14)
    struct ggml_cgraph * build_code_pred_step_graph(int32_t n_past, int32_t generation_step);
    
    // Build computation graph for 2-token prefill of code predictor
    // Processes [past_hidden, codec_embd(codebook_0_token)] together
    struct ggml_cgraph * build_code_pred_prefill_graph();
    
    // Parse hyperparameters from GGUF
    bool parse_config(struct gguf_context * ctx);
    
    // Create tensor structures
    bool create_tensors(struct gguf_context * ctx);
    
    // Load tensor data from file
    bool load_tensor_data(const std::string & path, struct gguf_context * ctx);
    
    tts_transformer_model model_;
    tts_transformer_state state_;
    std::string error_msg_;
    
    // Cached hidden states from last forward pass
    std::vector<float> last_hidden_;
    std::vector<ggml_fp16_t> embd_row_fp16_scratch_;
    std::mt19937 rng_{std::random_device{}()};
    CoreMLCodePredictor coreml_code_predictor_;
    bool use_coreml_code_predictor_ = false;
    std::string coreml_code_predictor_path_;
    bool skip_ggml_code_pred_layers_ = false;

#ifdef QWEN3_TTS_TENSORRT
    TRTCodePredictor trt_code_predictor_;
    bool use_trt_code_predictor_ = false;
    bool predict_codes_autoregressive_trt(const float * hidden, int32_t codebook_0_token,
                                          std::vector<int32_t> & output,
                                          float temperature, int32_t top_k);
    bool read_tensor_f32(struct ggml_tensor * tensor, std::vector<float> & buf);
#endif
};

// Free model resources
void free_transformer_model(tts_transformer_model & model);

// Free KV cache resources
void free_tts_kv_cache(tts_kv_cache & cache);

} // namespace qwen3_tts
