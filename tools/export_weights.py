#!/usr/bin/env python3
"""Export the official IndexTTS-2.5 greedy GPT weights to safetensors.

The exported package contains the semantic GPT plus text/language/mel
embeddings needed to construct the complete inference prefix. PyTorch Conv1D tensors retain their native
[input, output] layout; the Rust loader is responsible for that contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import save_file

HIDDEN = 1280
LAYERS = 24
MEL_CODES = 8194


def required_specs() -> dict[str, tuple[int, ...]]:
    specs: dict[str, tuple[int, ...]] = {
        "mel_embedding.weight": (MEL_CODES, HIDDEN),
        "mel_pos_embedding.emb.weight": (1818, HIDDEN),
        "text_embedding.weight": (60510, HIDDEN),
        "text_pos_embedding.emb.weight": (602, HIDDEN),
        "lang_embedding.weight": (107, HIDDEN),
    }
    for layer in range(LAYERS):
        prefix = f"gpt.h.{layer}"
        specs.update(
            {
                f"{prefix}.ln_1.weight": (HIDDEN,),
                f"{prefix}.ln_1.bias": (HIDDEN,),
                f"{prefix}.attn.c_attn.weight": (HIDDEN, 3 * HIDDEN),
                f"{prefix}.attn.c_attn.bias": (3 * HIDDEN,),
                f"{prefix}.attn.c_proj.weight": (HIDDEN, HIDDEN),
                f"{prefix}.attn.c_proj.bias": (HIDDEN,),
                f"{prefix}.ln_2.weight": (HIDDEN,),
                f"{prefix}.ln_2.bias": (HIDDEN,),
                f"{prefix}.mlp.c_fc.weight": (HIDDEN, 4 * HIDDEN),
                f"{prefix}.mlp.c_fc.bias": (4 * HIDDEN,),
                f"{prefix}.mlp.c_proj.weight": (4 * HIDDEN, HIDDEN),
                f"{prefix}.mlp.c_proj.bias": (HIDDEN,),
            }
        )
    specs.update(
        {
            "gpt.ln_f.weight": (HIDDEN,),
            "gpt.ln_f.bias": (HIDDEN,),
            "final_norm.weight": (HIDDEN,),
            "final_norm.bias": (HIDDEN,),
            "mel_head.weight": (MEL_CODES, HIDDEN),
            "mel_head.bias": (MEL_CODES,),
        }
    )
    return specs


def load_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint root must be a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def select_and_validate(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    errors: list[str] = []
    for name, expected_shape in required_specs().items():
        tensor = state.get(name)
        if tensor is None:
            errors.append(f"missing: {name}")
            continue
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"not a tensor: {name} ({type(tensor).__name__})")
            continue
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            errors.append(f"shape: {name}: {actual_shape} != {expected_shape}")
            continue
        selected[name] = tensor.detach().cpu().contiguous()
    if errors:
        raise ValueError("invalid GPT checkpoint contract:\n  " + "\n  ".join(errors))
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, source: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    manifest = {
        "format_version": 1,
        "model": "IndexTTS-2.5",
        "component": "greedy-semantic-gpt",
        "source_checkpoint": source.name,
        "source_sha256": sha256(source),
        "tensor_count": len(tensors),
        "parameter_count": sum(t.numel() for t in tensors.values()),
        "hidden_size": HIDDEN,
        "layers": LAYERS,
        "heads": 20,
        "head_dim": 64,
        "mel_codes": MEL_CODES,
        "start_mel_token": 8192,
        "stop_mel_token": 8193,
        "conv1d_layout": "input_output",
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(tensors.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", type=Path, help="output .safetensors file")
    parser.add_argument("--manifest", type=Path, help="output manifest JSON")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input checkpoint does not exist: {args.input}")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is used")

    state = load_checkpoint(args.input)
    selected = select_and_validate(state)
    parameters = sum(t.numel() for t in selected.values())
    print(f"validated {len(selected)} tensors, {parameters} parameters")

    if args.validate_only:
        return

    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(selected, str(args.output), metadata={"format": "indextts-2.5-greedy-gpt-v1"})
    print(f"wrote {args.output}")

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_manifest(manifest_path, args.input, selected)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
