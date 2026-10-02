#!/usr/bin/env python3
"""Convert IndexTTS-2.5 Wav2Vec2-BERT statistics to safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    state = torch.load(args.input, map_location="cpu", weights_only=True)
    expected = {"mean": (1024,), "var": (1024,)}
    tensors = {}
    for name, shape in expected.items():
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"invalid {name}: expected tensor {shape}")
        value = value.detach().float().cpu().contiguous()
        if name == "var" and not torch.all(value > 0):
            raise ValueError("variance must be positive")
        tensors[name] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    print(f"wrote {args.output}: mean={tuple(tensors['mean'].shape)} var={tuple(tensors['var'].shape)}")


if __name__ == "__main__":
    main()
