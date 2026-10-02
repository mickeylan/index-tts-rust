#!/usr/bin/env python3
"""
Export IndexTTS-2.5 GPT weights from PyTorch to safetensors format.

This script loads the PyTorch model checkpoint and exports the GPT weights
to safetensors format for use with the Rust implementation.

Usage:
    python tools/export_weights.py --input gpt.pth --output gpt.safetensors
    
Requirements:
    pip install torch safetensors
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install torch safetensors")
    sys.exit(1)


def load_pytorch_checkpoint(checkpoint_path: Path) -> dict:
    """Load PyTorch checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Try loading with torch.load first
    try:
        # Try direct loading
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        return state_dict
    except Exception as e:
        print(f"Direct load failed: {e}")
        
    try:
        # Try with weights_only for security
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        return state_dict
    except Exception as e:
        print(f"Weights-only load failed: {e}")
        
    raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")


def extract_gpt_weights(state_dict: dict) -> dict:
    """Extract GPT weights from full model state dict."""
    gpt_weights = {}
    
    for key, value in state_dict.items():
        # Filter for GPT-related weights
        if any(prefix in key for prefix in [
            'gpt.', 'text_embedding', 'mel_embedding',
            'text_pos_embedding', 'mel_pos_embedding',
            'emovec', 'campplus_proj', 'perceiver'
        ]):
            # Remove prefix if present
            clean_key = key
            for prefix in ['gpt.', 'model.gpt.']:
                if key.startswith(prefix):
                    clean_key = key[len(prefix):]
                    break
            
            gpt_weights[clean_key] = value
            print(f"  Extracted: {clean_key} -> {value.shape}")
    
    return gpt_weights


def export_safetensors(state_dict: dict, output_path: Path) -> None:
    """Export weights to safetensors format."""
    print(f"\nExporting to {output_path}...")
    
    # Convert to float32 for compatibility
    state_dict_f32 = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float16, torch.bfloat16]:
            tensor = tensor.to(torch.float32)
        state_dict_f32[key] = tensor
    
    save_file(state_dict_f32, str(output_path))
    
    # Print summary
    total_params = sum(t.numel() for t in state_dict_f32.values())
    print(f"Exported {len(state_dict_f32)} tensors")
    print(f"Total parameters: {total_params:,}")


def generate_config_json(gpt_weights: dict, output_path: Path) -> None:
    """Generate a config.json matching the GPT architecture."""
    
    # Infer config from weights
    config = {
        "model_type": "gpt2",
        "architectures": ["IndexTTSGPT"],
        "n_embd": None,
        "n_head": None,
        "n_layer": None,
        "n_vocab": 8194,
        "n_positions": 2048,
        "n_ctx": 2048,
        "activation_function": "gelu_new",
        "resid_dropout": 0.0,
        "attn_dropout": 0.0,
        "max_mel_tokens": 1815,
        "max_text_tokens": 600,
        "number_text_tokens": 60509,
        "number_mel_codes": 8194,
        "start_mel_token": 8192,
        "stop_mel_token": 8193,
        "start_text_token": 0,
        "stop_text_token": 1,
    }
    
    # Infer dimensions from embedding weights
    for key, tensor in gpt_weights.items():
        if 'text_embedding.weight' in key:
            vocab_size, hidden_size = tensor.shape
            config["number_text_tokens"] = vocab_size
            config["n_embd"] = hidden_size
            print(f"  Inferred n_embd={hidden_size} from text_embedding")
        elif 'mel_embedding.weight' in key:
            mel_size, hidden_size = tensor.shape
            config["number_mel_codes"] = mel_size
            config["n_vocab"] = mel_size
            print(f"  Inferred n_vocab={mel_size} from mel_embedding")
    
    # Count layers
    layer_keys = [k for k in gpt_weights.keys() if '.attn.' in k or '.mlp.' in k]
    if layer_keys:
        # Extract layer numbers
        layer_nums = set()
        for key in layer_keys:
            for part in key.split('.'):
                if part.isdigit():
                    layer_nums.add(int(part))
        if layer_nums:
            config["n_layer"] = max(layer_nums) + 1
            print(f"  Inferred n_layer={config['n_layer']}")
    
    # Infer head count from attention weights
    for key, tensor in gpt_weights.items():
        if 'attn.c_attn.weight' in key or 'attn.qkv_proj.weight' in key:
            # Typically (3 * n_embd, n_embd) or (n_embd, n_embd)
            if len(tensor.shape) == 2:
                hidden = tensor.shape[1]
                # Estimate heads (assuming divisible by 4)
                if hidden % 4 == 0:
                    n_heads = hidden // (hidden // 64) if hidden > 64 else 20
                    config["n_head"] = n_heads
                    print(f"  Inferred n_head={n_heads} from attention weights")
                    break
    
    # Write config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig written to {output_path}")
    print(json.dumps(config, indent=2))


def list_weights(state_dict: dict) -> None:
    """List all available weights."""
    print("\nAvailable weights:")
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        print(f"  {key}: {tensor.shape} ({tensor.dtype})")


def main():
    parser = argparse.ArgumentParser(
        description="Export IndexTTS-2.5 GPT weights to safetensors"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input PyTorch checkpoint (.pth)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output safetensors file (.safetensors)"
    )
    parser.add_argument(
        "--config", "-c",
        help="Output config file (.json)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available weights without exporting"
    )
    parser.add_argument(
        "--extract-gpt",
        action="store_true",
        help="Only extract GPT weights (skip encoder/decoders)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Load checkpoint
    state_dict = load_pytorch_checkpoint(input_path)
    
    # Handle different state dict formats
    if isinstance(state_dict, dict):
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            state_dict = state_dict['state_dict']
    
    print(f"Loaded {len(state_dict)} tensors")
    
    # List weights if requested
    if args.list:
        list_weights(state_dict)
        return
    
    # Extract GPT weights if requested
    if args.extract_gpt:
        state_dict = extract_gpt_weights(state_dict)
        print(f"\nExtracted {len(state_dict)} GPT weights")
    
    # Export to safetensors
    if args.output:
        output_path = Path(args.output)
        if args.extract_gpt:
            export_safetensors(state_dict, output_path)
        else:
            export_safetensors(state_dict, output_path)
    
    # Generate config
    if args.config:
        config_path = Path(args.config)
        if args.extract_gpt:
            generate_config_json(state_dict, config_path)
        else:
            gpt_weights = extract_gpt_weights(state_dict)
            generate_config_json(gpt_weights, config_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
