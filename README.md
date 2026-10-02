# IndexTTS-2.5 Rust

Pure Rust inference library for [IndexTTS-2.5](https://github.com/index-tts/index-tts), a multilingual text-to-speech model.

## Features

- **Zero Python Runtime**: Fully native Rust implementation
- **GPU Acceleration**: CUDA support via Candle and ONNX Runtime
- **Stable C ABI**: Bindings for Go, C/C++, and other languages
- **Cross-Platform**: Windows with NVIDIA CUDA (first-class support)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust Control Layer                       │
├─────────────────────────────────────────────────────────────┤
│  Candle GPT        │  ONNX Runtime                         │
│  - Semantic GPT    │  - Wav2Vec2-BERT                      │
│  - KV Cache        │  - CAMPPlus                           │
│  - Sampling        │  - BigVGAN                            │
│                    │  - S2Mel DiT                          │
├─────────────────────────────────────────────────────────────┤
│  Rust Native                                           │
│  - Text Normalization  - Audio I/O                       │
│  - Tokenizer           - Resampling                       │
├─────────────────────────────────────────────────────────────┤
│  C ABI + Go Bindings                                      │
└─────────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Description |
|-------|-------------|
| `indextts-core` | Core data structures and error types |
| `indextts-text` | Text normalization |
| `indextts-tokenizer` | Tiktoken BPE tokenizer |
| `indextts-audio` | Audio reading, resampling, WAV I/O |
| `indextts-gpt` | GPT semantic token generation |
| `indextts-ort` | ONNX Runtime wrapper |
| `indextts-pipeline` | Complete inference pipeline |
| `indextts-ffi` | C ABI interface |
| `indextts-cli` | Command-line tool |

## Quick Start

### Build

```bash
cargo build --release
```

### CLI Usage

```bash
indextts synth \
  --model D:\models\IndexTTS-2.5 \
  --voice reference.wav \
  --text "相信姐姐。" \
  --language ZH \
  --seed 1234 \
  --output result.wav
```

### Go Usage

```go
package main

/*
#cgo LDFLAGS: -lindextts
#include "indextts.h"
*/
import "C"

func main() {
    var model C.indextts_model_t
    var opts C.indextts_model_options_t
    opts.model_dir = C.CString("/models/IndexTTS-2.5")
    
    if C.indextts_model_load(&opts, &model) != 0 {
        println(C.GoString(C.indextts_last_error()))
        return
    }
    defer C.indextts_model_free(model)
    
    // ... generate speech
}
```

## Current Status

### Phase 0: Repository Setup ✅
- ✅ Workspace and crate structure (9 crates)
- ✅ Core data structures (`ModelConfig`, `GenerationConfig`, `AudioBuffer`, `SemanticCodes`)
- ✅ Text normalization (Chinese number/date/time)
- ✅ Tokenizer framework (Tiktoken BPE)
- ✅ Audio processing (WAV I/O)
- ✅ C ABI draft with `indextts.h`
- ✅ CLI tool framework
- ✅ CI configuration

### Phase 1: GPT Semantic Token PoC (In Progress)
- ✅ GPT configuration matching IndexTTS-2.5 (1280-dim, 24 layers, 20 heads)
- ✅ Safetensors export and Candle loading for 296 required tensors
- ✅ Weight names, shapes, and Conv1D layout validation
- ✅ Real 24-layer Attention/MLP/LayerNorm forward pass
- ✅ Prefill and incremental KV-cache decode
- ✅ Greedy generation control loop
- ✅ Python/Rust first-token logits agree (`max_abs_error=2.5749207e-5` on the fixed CPU fixture)
- ✅ Full greedy sequence parity: all 45 semantic codes and EOS position are identical
- 🔲 Checked-in compact regression fixture (model weights remain external)

### Later generation modes
- ✅ Temperature, top-k, top-p, and repetition-penalty framework
- 🔲 Sampling RNG parity
- 🔲 Beam search implementation

### Next Steps
- 🔲 Add a distributable parity fixture and automated external-model test
- 🔲 ONNX backend integration
- 🔲 End-to-end pipeline completion

See [docs/2026-10-02-IndexTTS-2.5纯Rust推理库实施方案.md](docs/2026-10-02-IndexTTS-2.5纯Rust推理库实施方案.md) for full roadmap.

## Model Architecture

Based on the official IndexTTS-2.5 implementation:

| Parameter | Value |
|-----------|-------|
| Model Dimension | 1280 |
| Attention Heads | 20 |
| Layers | 24 |
| Max Mel Tokens | 1815 |
| Max Text Tokens | 600 |
| Vocab Size (mel) | 8194 |
| Vocab Size (text) | 60509 |
| Start Mel Token | 8192 |
| Stop Mel Token | 8193 |
| Condition Type | Conformer + Perceiver |
| Speaker Mode | CAMPPlus |

## Requirements

- Rust 1.85+
- CUDA 11.8+ / 12.x (for GPU support)
- ONNX Runtime with CUDA EP (for ONNX models)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

**Important**: IndexTTS-2.5 model weights are governed by the Bilibili Model Use License Agreement. See the original [IndexTTS license](https://github.com/index-tts/index-tts/blob/main/LICENSE.md).

## Contributing

Contributions are welcome! Please read the [architecture documentation](docs/) before making significant changes.

## References

- [IndexTTS Official](https://github.com/index-tts/index-tts)
- [Candle](https://github.com/huggingface/candle)
- [ONNX Runtime](https://onnxruntime.ai/)
