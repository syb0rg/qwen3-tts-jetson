#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required unless --serve)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  -e, --embedding <file> Cached speaker embedding (.bin)\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  --serve                Server mode: read requests from stdin\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Server mode:\n");
    fprintf(stderr, "  %s -m ./models -e speaker.bin --serve\n", program);
    fprintf(stderr, "  Then send lines: text<TAB>output.wav\n");
    fprintf(stderr, "  Responds with:   OK<TAB>duration_s<TAB>time_ms<TAB>output.wav\n");
    fprintf(stderr, "  Send 'quit' to exit.\n");
}

// Load cached speaker embedding from file, returns empty vector on failure
static std::vector<float> load_embedding(const std::string & path) {
    std::vector<float> embd;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return embd;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    embd.resize(fsize / sizeof(float));
    size_t n_read = fread(embd.data(), sizeof(float), embd.size(), f);
    fclose(f);
    if (n_read != embd.size()) embd.clear();
    return embd;
}

// Save speaker embedding to file
static bool save_embedding(const std::string & path, const std::vector<float> & embd) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(embd.data(), sizeof(float), embd.size(), f);
    fclose(f);
    return true;
}

// Resolve speaker embedding: load from cache, or encode and cache
static bool resolve_embedding(qwen3_tts::Qwen3TTS & tts,
                               const std::string & embedding_file,
                               const std::string & reference_audio,
                               std::vector<float> & embd) {
    if (embedding_file.empty()) return false;

    embd = load_embedding(embedding_file);
    if (!embd.empty()) {
        fprintf(stderr, "Loaded cached speaker embedding: %s (%zu floats)\n",
                embedding_file.c_str(), embd.size());
        return true;
    }

    if (reference_audio.empty()) {
        fprintf(stderr, "Error: embedding file not found and no --reference provided\n");
        return false;
    }

    fprintf(stderr, "Encoding speaker embedding from: %s\n", reference_audio.c_str());
    if (!tts.encode_speaker(reference_audio, embd)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return false;
    }

    if (save_embedding(embedding_file, embd)) {
        fprintf(stderr, "Saved speaker embedding to: %s (%zu floats)\n",
                embedding_file.c_str(), embd.size());
    }
    return true;
}

// Synthesize one utterance
static qwen3_tts::tts_result synthesize_one(qwen3_tts::Qwen3TTS & tts,
                                             const std::string & text,
                                             const std::vector<float> & speaker_embd,
                                             const std::string & reference_audio,
                                             const qwen3_tts::tts_params & params) {
    if (!speaker_embd.empty()) {
        return tts.synthesize_with_embedding(text, speaker_embd, params);
    } else if (!reference_audio.empty()) {
        return tts.synthesize_with_voice(text, reference_audio, params);
    } else {
        return tts.synthesize(text, params);
    }
}

// Server loop: read "text\toutput.wav" lines from stdin
static int run_server(qwen3_tts::Qwen3TTS & tts,
                      const std::vector<float> & speaker_embd,
                      const std::string & reference_audio,
                      const qwen3_tts::tts_params & params) {
    fprintf(stderr, "\nServer ready. Send: text<TAB>output.wav  (or 'quit' to exit)\n");
    fflush(stderr);

    char line[8192];
    while (fgets(line, sizeof(line), stdin)) {
        // Strip trailing newline
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';

        if (len == 0) continue;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;

        // Parse: text\toutput_file
        std::string req_text;
        std::string req_output = "output.wav";
        const char * tab = strchr(line, '\t');
        if (tab) {
            req_text = std::string(line, tab - line);
            req_output = std::string(tab + 1);
        } else {
            req_text = line;
        }

        fprintf(stderr, "Synthesizing: \"%s\" -> %s\n", req_text.c_str(), req_output.c_str());

        qwen3_tts::tts_result result = synthesize_one(tts, req_text, speaker_embd, reference_audio, params);

        if (!result.success) {
            fprintf(stdout, "ERR\t%s\n", result.error_msg.c_str());
            fflush(stdout);
            continue;
        }

        if (!qwen3_tts::save_audio_file(req_output, result.audio, result.sample_rate)) {
            fprintf(stdout, "ERR\tfailed to save %s\n", req_output.c_str());
            fflush(stdout);
            continue;
        }

        float duration = (float)result.audio.size() / result.sample_rate;
        fprintf(stdout, "OK\t%.2f\t%lld\t%s\n", duration, (long long)result.t_total_ms, req_output.c_str());
        fflush(stdout);

        fprintf(stderr, "  Done: %.2fs audio in %lldms (RTF=%.1f)\n",
                duration, (long long)result.t_total_ms,
                (float)result.t_total_ms / 1000.0f / duration);
    }

    fprintf(stderr, "Server shutting down.\n");
    return 0;
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string embedding_file;
    bool serve_mode = false;

    qwen3_tts::tts_params params;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "Error: missing model directory\n"); return 1; }
            model_dir = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) { fprintf(stderr, "Error: missing text\n"); return 1; }
            text = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { fprintf(stderr, "Error: missing output file\n"); return 1; }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) { fprintf(stderr, "Error: missing reference audio\n"); return 1; }
            reference_audio = argv[i];
        } else if (arg == "-e" || arg == "--embedding") {
            if (++i >= argc) { fprintf(stderr, "Error: missing embedding file\n"); return 1; }
            embedding_file = argv[i];
        } else if (arg == "--temperature") {
            if (++i >= argc) { fprintf(stderr, "Error: missing temperature value\n"); return 1; }
            params.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) { fprintf(stderr, "Error: missing top-k value\n"); return 1; }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) { fprintf(stderr, "Error: missing top-p value\n"); return 1; }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) { fprintf(stderr, "Error: missing max-tokens value\n"); return 1; }
            params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) { fprintf(stderr, "Error: missing repetition-penalty value\n"); return 1; }
            params.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) { fprintf(stderr, "Error: missing threads value\n"); return 1; }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "--serve") {
            serve_mode = true;
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!serve_mode && text.empty()) {
        fprintf(stderr, "Error: text is required (or use --serve)\n");
        print_usage(argv[0]);
        return 1;
    }

    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;

    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }

    // Resolve speaker embedding (auto-cache when -r is used without -e)
    std::vector<float> speaker_embd;
    if (embedding_file.empty() && !reference_audio.empty()) {
        embedding_file = reference_audio + ".embd";
    }
    if (!embedding_file.empty()) {
        if (!resolve_embedding(tts, embedding_file, reference_audio, speaker_embd)) {
            return 1;
        }
    }

    // Server mode
    if (serve_mode) {
        return run_server(tts, speaker_embd, reference_audio, params);
    }

    // Single-shot mode
    fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
    if (!reference_audio.empty() && speaker_embd.empty()) {
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
    }

    qwen3_tts::tts_result result = synthesize_one(tts, text, speaker_embd, reference_audio, params);

    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }

    fprintf(stderr, "\n");

    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
        return 1;
    }

    fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
    fprintf(stderr, "Audio duration: %.2f seconds\n",
            (float)result.audio.size() / result.sample_rate);

    // Print timing
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Load:      %6lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }

    return 0;
}
