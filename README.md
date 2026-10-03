# IndexTTS-2.5 Rust

Python-free **runtime** for IndexTTS-2.5. Rust owns preprocessing, tokenization, orchestration, CFM integration, PCM processing, and WAV output; Candle runs the autoregressive semantic GPT and ONNX Runtime runs the remaining neural components. Python/PyTorch is used only to convert official checkpoints offline.

## Current support

- Windows CPU runtime and validated NVIDIA CUDA runtime (RTX 4070, CUDA 12.8)
- Chinese-first greedy synthesis from text + reference WAV to 22.05 kHz mono WAV
- Candle GPT with KV cache and exact fixed-fixture semantic-token parity
- Wav2Vec2-BERT, CAMPPlus, conditioning, semantic codec, length regulator, bucketed DiT and bucketed BigVGAN through ONNX Runtime
- Rust API, CLI, C ABI, and Go wrapper
- Fixed frame buckets (default export: 256, 512, 1024, 2048)

Not yet validated: sampling/beam search, non-Chinese quality, and every possible long-input bucket. CUDA is opt-in at build and runtime.

## Build

```powershell
cargo build --release -p indextts-cli
cargo build --release -p indextts-ffi

# CUDA build (requires CUDA 12.8 and MSVC on Windows)
cargo build --release -p indextts-cli --features cuda
```

The ONNX Runtime dynamic library must be discoverable by the operating system. Alternatively set `ORT_DYLIB_PATH` to the matching ONNX Runtime 2.0.0-rc.13 library.

## Offline model conversion

Conversion requires the official IndexTTS source/checkpoints and Python dependencies used by that project. It is not part of deployment.

```powershell
python tools/export_all.py `
  --source E:\path\to\index-tts `
  --checkpoints E:\path\to\index-tts\checkpoints `
  --output E:\models\indextts25-rust `
  --buckets 256 512 1024 2048
```

The command exports all runtime models, tokenizer assets, frame buckets, and a SHA-256 package manifest. Model artifacts are intentionally not committed.

## CLI

```powershell
target\release\indextts.exe --device cuda --device-index 0 synth `
  --model E:\models\indextts25-rust `
  --voice reference.wav `
  --text "相信姐姐。" `
  --language zh `
  --seed 1234 `
  --output result.wav
```

Semantic codes only:

```powershell
target\release\indextts.exe tokens --model E:\models\indextts25-rust --voice reference.wav --text "你好"
```

## Rust API

```rust,no_run
use indextts_core::{DeviceConfig, DeviceKind, GenerationConfig, ModelConfig, Precision};
use indextts_pipeline::IndexTtsPipeline;
use std::path::PathBuf;

let mut pipeline = IndexTtsPipeline::new(ModelConfig {
    model_dir: PathBuf::from(r"E:\models\indextts25-rust"),
    device: DeviceConfig::new(DeviceKind::Cpu, 0),
    precision: Precision::Float32,
});
pipeline.load()?;
let audio = pipeline.synthesize("你好", PathBuf::from("voice.wav").as_path(), &GenerationConfig::default())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## C and Go

The release build produces `indextts.dll`, `indextts.dll.lib`, and `indextts.lib` on Windows. The public header is `crates/indextts-ffi/indextts.h`. Initialize option structures with the supplied init functions, free audio with `indextts_audio_free`, and close voice/model handles exactly once.

The Go module is under `bindings/go`:

```go
model, err := indextts.Load(`E:\models\indextts25-rust`)
if err != nil { panic(err) }
defer model.Close()
voice, err := model.PrepareVoice("reference.wav")
if err != nil { panic(err) }
defer voice.Close()
audio, err := model.Generate(voice, "你好", indextts.Options{Language: "ZH"})
```

## Verified numerical contracts

- GPT: all 45 semantic codes and EOS position match the Python fixture; maximum logit error `2.5749207e-5`
- Wav2Vec2-BERT maximum error `0.00154161453`
- CAMPPlus cosine similarity `0.999825954`
- Semantic codec maximum error below `9e-6`
- Length regulator maximum error below `2e-7`
- DiT 256-frame bucket maximum error about `1.42e-5`
- BigVGAN 256-frame bucket maximum error about `3.12e-5`

## Limitations

- CUDA currently targets the CUDA 12 ABI. CUDA 12.8, cuDNN 9, and a CUDA-enabled ONNX Runtime must be discoverable through `PATH`; CPU remains available with `--device cpu`.
- Inference is greedy only (`do_sample=false`, `num_beams=1`).
- WAV input is the supported public contract.
- Output quality must be assessed with appropriately licensed real speech references; synthetic test tones only establish execution and file correctness.
- The model weights are governed by the upstream Bilibili Model Use License Agreement and are not covered by this repository's Apache-2.0 source license.

## Test

```powershell
cargo test --workspace
python -m unittest discover -s tools -p "test_*.py" -v
cargo build --release -p indextts-cli
cargo build --release -p indextts-ffi
$env:CGO_LDFLAGS="-L$pwd/target/release -lindextts"
go -C bindings/go test ./...
```

See the [IndexTTS Windows PowerShell workflow](docs/WINDOWS_POWERSHELL.md) for project operations. For PowerShell itself, see the separate [PowerShell command reference](docs/powershell/README.md).

See [the implementation plan](2026-10-02-IndexTTS-2.5纯Rust推理库实施方案.md) for architectural background.
