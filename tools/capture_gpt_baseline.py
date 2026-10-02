#!/usr/bin/env python3
"""Capture a deterministic first-token GPT fixture from official IndexTTS-2.5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source.resolve()))
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    config = OmegaConf.load(args.model_dir / "config.yaml")
    torch.manual_seed(1234)
    model = UnifiedVoice(**config.gpt, use_accel=False, spk_cond_mode="campplus")
    load_checkpoint(model, str(args.model_dir / config.gpt_checkpoint))
    model.eval()
    model.post_init_gpt2_config(kv_cache=True)

    # This fixture deliberately starts after speaker/emotion conditioning. It
    # tests the Phase-1 contract: fixed prefix embeddings -> semantic logits.
    conditioning = torch.linspace(-0.25, 0.25, 3 * config.gpt.model_dim).reshape(
        1, 3, config.gpt.model_dim
    )
    text_tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    language = torch.tensor([[0]], dtype=torch.long)

    fake_ids, prefix, attention_mask = model.prepare_gpt_inputs(
        conditioning, text_tokens, language
    )
    model.inference_model.store_mel_emb(prefix)
    with torch.no_grad():
        output = model.inference_model(
            fake_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    logits = output.logits[:, -1, :].float().contiguous()
    first_token = int(torch.argmax(logits, dim=-1).item())
    semantic_codes: list[int] = []
    reached_eos = False
    current = output
    current_mask = attention_mask
    with torch.no_grad():
        for _ in range(args.steps):
            token = int(torch.argmax(current.logits[:, -1, :], dim=-1).item())
            if token == config.gpt.stop_mel_token:
                reached_eos = True
                break
            semantic_codes.append(token)
            current_mask = torch.nn.functional.pad(current_mask, (0, 1), value=1)
            current = model.inference_model(
                torch.tensor([[token]], dtype=torch.long),
                past_key_values=current.past_key_values,
                attention_mask=current_mask,
                use_cache=True,
                return_dict=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "conditioning": conditioning.float().contiguous(),
            "text_tokens": text_tokens.contiguous(),
            "language": language.contiguous(),
            "prefix": prefix.float().contiguous(),
            "fake_ids": fake_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "first_logits": logits,
        },
        str(args.output),
    )
    metadata = {
        "text_tokens": text_tokens.tolist(),
        "language": 0,
        "first_token": first_token,
        "semantic_codes": semantic_codes,
        "reached_eos": reached_eos,
        "eos_position": len(semantic_codes) if reached_eos else None,
        "prefix_shape": list(prefix.shape),
        "fake_ids_shape": list(fake_ids.shape),
        "logits_shape": list(logits.shape),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
