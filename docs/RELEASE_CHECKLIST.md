# Release checklist

## Automated code checks

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python -m unittest discover -s tools -p "test_*.py" -v
$env:CGO_ENABLED = '0'
go -C bindings/go test ./...
```

## Native Windows Go checks

Build `indextts-ffi`, regenerate `libindextts.dll.a`, place `indextts.dll` and ONNX Runtime on `PATH`, then run:

```powershell
$env:CGO_ENABLED = '1'
$env:CGO_LDFLAGS = "-L$PWD/target/release"
go -C bindings/go test -tags indextts_native ./...
```

## Model package checks

```powershell
python tools/rebuild_manifest.py E:\models\indextts25-rust
$env:ORT_DYLIB_PATH = 'path\to\onnxruntime.dll'
cargo run -p indextts-pipeline --example validate_model_contracts -- E:\models\indextts25-rust
```

The manifest validates runtime version, required assets, semantic-token constants, safe bucket sets, ONNX opset/I/O/dtype/shape metadata, file sizes, and SHA-256 hashes.

## Packages

```powershell
powershell -File tools/package_windows_cpu.ps1 -Zip
powershell -File tools/package_windows_cuda.ps1 -Zip
powershell -File tools/verify-runtime.ps1 -RuntimeDir dist/index-tts-rust-win64-cpu -ModelDir E:\models\indextts25-rust
```

CUDA packages explicitly declare their CUDA 12.8/cuDNN 9 and CUDA 13 cuBLAS compatibility dependencies.

## Hardware and quality acceptance

Run these explicitly on the target machine; they are release evidence rather than compile-time checks:

- CPU and CUDA synthesis with identical fixed inputs.
- Emotion reference, vector, and text modes; same seed reproducibility.
- Request cancellation while running and queued.
- 500/1000/2000-character long-text synthesis and ordering review.
- 100 sequential generations and 100 voice prepare/free cycles while recording RAM/VRAM.
- Multi-model/multi-GPU concurrency where hardware is available.
- ASR signal plus human review for omissions, repetitions, added dialogue, and intelligibility.

Record commands, runtime/ABI/model versions, device, hashes, semantic token counts, stage timings, duration, peak/RMS/silence ratio, and memory trends in the release notes.
