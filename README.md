# qwen3-tts-jetson

Realtime text-to-speech on [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) (8 GB). C++ inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) using [GGML](https://github.com/ggml-org/ggml) and [TensorRT](https://developer.nvidia.com/tensorrt).

**1.01x realtime** (RTF 0.994) with voice cloning, 839 MB peak memory.

## How It Works

```
        Text ──► Tokenizer ────── token IDs ──┐
                                               ├─► Talker ─► Code Predictor ─► Vocoder ─► 24kHz audio
Ref. Audio ──► Speaker Encoder ── embedding ──┘    (GGML)       (TRT)           (TRT)
```

Per-frame pipeline (80 ms/frame, 12 Hz = 83 ms budget):

| Stage | Engine | ms/frame | Key optimizations |
|-------|--------|----------|-------------------|
| Talker (28 layers) | GGML CUDA | 27.3 | Flash attention, F16 weights without cast |
| Code Predictor (5 layers × 16 steps) | TensorRT BF16 | 45.1 | BF16 precision, GPU-side sampling |
| Vocoder | TensorRT | ~2.2 | Streaming decode during generation |
| Overhead | — | ~5.4 | Graph build/alloc, embed lookups |

## Prerequisites

- Jetson Orin Nano Super (JetPack 6.x)
- CUDA 12.x + TensorRT 10.x (included in JetPack)
- CMake 3.14+, GCC 9+
- Python 3.10+ with [uv](https://github.com/astral-sh/uv) (model setup only)

## Build

```bash
git clone https://github.com/syb0rg/qwen3-tts-jetson.git
cd qwen3-tts-jetson
git submodule update --init --recursive

# Build GGML with CUDA
cmake -S ggml -B ggml/build -DGGML_CUDA=ON
cmake --build ggml/build -j4

# Build qwen3-tts-jetson with TensorRT
cmake -S . -B build -DQWEN3_TTS_TENSORRT=ON
cmake --build build -j4
```

## Model Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install huggingface_hub gguf torch safetensors numpy tqdm onnx onnxruntime
python scripts/setup_pipeline_models.py
```

This downloads the 0.6B model and generates `models/qwen3-tts-0.6b-f16.gguf` and `models/qwen3-tts-tokenizer-f16.gguf`.

### TensorRT Engines

Build the BF16 code predictor and vocoder engines (device-specific, must be built on the Jetson):

```bash
# Export code predictor to ONNX, then build BF16 TRT engine
python scripts/export_code_predictor.py
python scripts/build_fp16_engine.py

# Build vocoder TRT engine (40-frame fixed chunks)
trtexec --onnx=models/vocoder_decoder.onnx \
  --saveEngine=models/vocoder_decoder_40.trt \
  --fp16 --memPoolSize=workspace:512MiB
```

BF16 is critical for audio quality — FP16 and INT8 produce garbled output due to precision compounding over 15 autoregressive code predictor steps.

## Usage

```bash
# Basic synthesis
./build/qwen3-tts-cli -m models -t "Hello, world!" -o hello.wav

# Voice cloning
./build/qwen3-tts-cli -m models -t "Hello!" -r reference.wav -o cloned.wav

# Server mode (for streaming requests)
./build/qwen3-tts-cli -m models -e speaker.bin --serve
```

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model <dir>` | Model directory | required |
| `-t, --text <text>` | Text to synthesize | required |
| `-o, --output <file>` | Output WAV path | `output.wav` |
| `-r, --reference <file>` | Reference audio for voice cloning | — |
| `-e, --embedding <file>` | Cached speaker embedding | — |
| `--temperature <val>` | Sampling temperature (0 = greedy) | 0.9 |
| `--top-k <n>` | Top-k sampling | 50 |
| `--max-tokens <n>` | Max audio frames | 4096 |
| `--repetition-penalty <val>` | Codebook-0 repetition penalty | 1.05 |
| `--serve` | Server mode (read from stdin) | — |

## Memory Budget

| Component | Memory |
|-----------|--------|
| GGUF transformer (F16) | ~400 MB |
| TRT code predictor (BF16 + lm_heads + embeddings) | ~335 MB |
| TRT vocoder | ~100 MB |
| Runtime overhead | ~4 MB |
| **Total peak RSS** | **839 MB** |
| **System RAM available** | 7.4 GB |

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) by Georgi Gerganov
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture
