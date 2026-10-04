#!/usr/bin/env python3
"""Rebuild a strict manifest for an existing IndexTTS Rust model package."""

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = (
    "gpt.safetensors",
    "wav2vec2bert_stats.safetensors",
    "emotion-prototypes.safetensors",
    "bpe.model",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "pinyin.vocab",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/model.safetensors",
    "onnx/wav2vec2bert/model.onnx",
    "onnx/campplus/model.onnx",
    "onnx/gpt-conditioning/model.onnx",
    "onnx/emotion-conditioner/model.onnx",
    "onnx/semantic-codec/model.onnx",
    "onnx/length-regulator/model.onnx",
    "onnx/s2mel/model-256.onnx",
    "onnx/s2mel/model-512.onnx",
    "onnx/s2mel/model-1024.onnx",
    "onnx/bigvgan/model-256.onnx",
    "onnx/bigvgan/model-512.onnx",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    missing = [relative for relative in REQUIRED if not (model_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing required model components: " + ", ".join(missing))
    files = {}
    for relative in REQUIRED:
        path = model_dir / relative
        print(f"hashing {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "format_version": 1,
        "model": "IndexTTS-2.5",
        "runtime": "index-tts-rust",
        "minimum_runtime_version": "0.1.0",
        "s2mel_buckets": [256, 512, 1024],
        "bigvgan_buckets": [256, 512],
        "sample_rate": 22050,
        "semantic": {"start_token": 8192, "stop_token": 8193, "max_tokens": 1815},
        "files": files,
    }
    output = model_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} with {len(files)} validated components")


if __name__ == "__main__":
    main()
