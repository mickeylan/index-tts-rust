#!/usr/bin/env python3
"""Build a canonical IndexTTS-2.5 runtime model package."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

COMPONENTS = (
    "campplus",
    "wav2vec2bert",
    "gpt-conditioning",
    "emotion-conditioner",
    "semantic-codec",
    "length-regulator",
)
TOKENIZER_FILES = (
    "bpe.model",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "pinyin.vocab",
)


def run(*args: object) -> None:
    command = [sys.executable, *map(str, args)]
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dit-buckets", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--bigvgan-buckets", nargs="+", type=int, default=[256, 512])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    run(root / "tools/export_weights.py", "--input", args.checkpoints / "gpt.pth",
        "--output", output / "gpt.safetensors", "--manifest", output / "gpt.manifest.json")
    run(root / "tools/export_wav2vec_stats.py", "--input", args.checkpoints / "wav2vec2bert_stats.pt",
        "--output", output / "wav2vec2bert_stats.safetensors")
    run(root / "tools/export_emotion_prototypes.py",
        "--speaker", args.checkpoints / "feat1.pt",
        "--emotion", args.checkpoints / "feat2.pt",
        "--output", output / "emotion-prototypes.safetensors")
    for filename in TOKENIZER_FILES:
        shutil.copy2(args.checkpoints / filename, output / filename)
    qwen_source = args.checkpoints / "qwen0.6bemo4-merge"
    qwen_output = output / "qwen0.6bemo4-merge"
    if qwen_output.exists():
        shutil.rmtree(qwen_output)
    shutil.copytree(qwen_source, qwen_output)
    exporter = root / "tools/export_indextts25_onnx.py"
    for component in COMPONENTS:
        run(exporter, component, "--source", args.source, "--model-dir", args.checkpoints,
            "--output", output / "onnx" / component / "model.onnx")
    bucket_sets = {
        "dit": sorted(set(args.dit_buckets)),
        "bigvgan": sorted(set(args.bigvgan_buckets)),
    }
    for component, buckets in bucket_sets.items():
        target = "s2mel" if component == "dit" else component
        for frames in buckets:
            run(exporter, component, "--frames", frames, "--source", args.source,
                "--model-dir", args.checkpoints,
                "--output", output / "onnx" / target / f"model-{frames}.onnx")

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "format_version": 1,
        "model": "IndexTTS-2.5",
        "runtime": "index-tts-rust",
        "minimum_runtime_version": "0.1.0",
        "s2mel_buckets": bucket_sets["dit"],
        "bigvgan_buckets": bucket_sets["bigvgan"],
        "sample_rate": 22050,
        "semantic": {"start_token": 8192, "stop_token": 8193, "max_tokens": 1815},
        "files": {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote package with {len(files)} files to {output}")


if __name__ == "__main__":
    main()
