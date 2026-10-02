#!/usr/bin/env python3
"""Export and numerically validate IndexTTS-2.5 non-autoregressive ONNX models.

The exporter intentionally accepts only local checkpoints. It never downloads or
substitutes IndexTTS-2 models. Components are added here one at a time after an
ONNX-vs-PyTorch parity check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def export_campplus(source: Path, model_dir: Path, output: Path) -> None:
    sys.path.insert(0, str(source.resolve()))
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    checkpoint = model_dir / "hf_cache" / "campplus_cn_common.bin"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    torch.manual_seed(1234)
    sample = torch.randn(1, 197, 80)
    with torch.no_grad():
        expected = model(sample).cpu().numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        str(output),
        input_names=["x"],
        output_names=["style"],
        dynamic_axes={"x": {0: "batch", 1: "frames"}, "style": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(["style"], {"x": sample.numpy()})[0]
    difference = np.abs(expected - actual)
    max_error = float(np.max(difference))
    mean_error = float(np.mean(difference))
    cosine = float(np.sum(expected * actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    print(
        f"campplus output_shape={list(actual.shape)} max_abs_error={max_error:.9g} "
        f"mean_abs_error={mean_error:.9g} cosine={cosine:.9g} "
        f"expected_range=[{expected.min():.9g},{expected.max():.9g}]"
    )
    # CAMPPlus contains deep temporal statistics pooling; CPU kernels differ
    # slightly between PyTorch and ORT. Guard both elementwise drift and the
    # direction of the speaker embedding used downstream.
    if max_error > 0.06 or cosine < 0.999:
        raise RuntimeError(
            f"CAMPPlus ONNX parity failed: max_abs_error={max_error}, cosine={cosine}"
        )


def export_wav2vec2bert(model_dir: Path, output: Path) -> None:
    from torch import nn
    from transformers import Wav2Vec2BertModel

    model_path = model_dir / "hf_cache" / "w2v-bert-2.0"
    full_model = Wav2Vec2BertModel.from_pretrained(model_path, local_files_only=True)
    full_model.eval()

    class HiddenState17(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.feature_projection = model.feature_projection
            self.dropout = model.encoder.dropout
            self.embed_positions = model.encoder.embed_positions
            self.layers = nn.ModuleList(model.encoder.layers[:17])

        def forward(self, input_features, attention_mask):
            hidden = self.feature_projection(input_features)[0]
            conv_mask = attention_mask
            hidden = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            expanded_mask = 1.0 - attention_mask[:, None, None, :].to(hidden.dtype)
            expanded_mask = expanded_mask * torch.finfo(hidden.dtype).min
            expanded_mask = expanded_mask.expand(
                expanded_mask.shape[0], 1, expanded_mask.shape[-1], expanded_mask.shape[-1]
            )
            hidden = self.dropout(hidden)
            positions = self.embed_positions(hidden) if self.embed_positions is not None else None
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    attention_mask=expanded_mask,
                    relative_position_embeddings=positions,
                    conv_attention_mask=conv_mask,
                )[0]
            return hidden

    wrapper = HiddenState17(full_model).eval()
    torch.manual_seed(1234)
    features = torch.randn(1, 64, 160)
    mask = torch.ones(1, 64, dtype=torch.long)
    mask[:, -3:] = 0
    with torch.no_grad():
        official = full_model(
            input_features=features,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[17]
        expected = wrapper(features, mask).cpu().numpy()
    wrapper_error = float(np.max(np.abs(official.cpu().numpy() - expected)))
    if wrapper_error > 1e-5:
        raise RuntimeError(f"Wav2Vec2-BERT layer-17 wrapper mismatch: {wrapper_error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (features, mask),
        str(output),
        input_names=["input_features", "attention_mask"],
        output_names=["hidden_states_17"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "frames"},
            "attention_mask": {0: "batch", 1: "frames"},
            "hidden_states_17": {0: "batch", 1: "frames"},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["hidden_states_17"],
        {"input_features": features.numpy(), "attention_mask": mask.numpy()},
    )[0]
    difference = np.abs(expected - actual)
    max_error = float(np.max(difference))
    mean_error = float(np.mean(difference))
    print(
        f"wav2vec2bert output_shape={list(actual.shape)} wrapper_error={wrapper_error:.9g} "
        f"max_abs_error={max_error:.9g} mean_abs_error={mean_error:.9g}"
    )
    if max_error > 0.003 or mean_error > 1e-5:
        raise RuntimeError(
            f"Wav2Vec2-BERT ONNX parity failed: max={max_error}, mean={mean_error}"
        )


def export_gpt_conditioning(source: Path, model_dir: Path, output: Path) -> None:
    sys.path.insert(0, str(source.resolve()))
    from omegaconf import OmegaConf
    from torch import nn
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    config = OmegaConf.load(model_dir / "config.yaml")
    model = UnifiedVoice(**config.gpt, use_accel=False, spk_cond_mode="campplus")
    load_checkpoint(model, str(model_dir / config.gpt_checkpoint))
    model.eval()

    class GptConditioning(nn.Module):
        def __init__(self, unified_voice):
            super().__init__()
            self.speaker_projection = unified_voice.spk_emb_proj
            self.emotion_encoder = unified_voice.emo_conditioning_encoder
            self.emotion_perceiver = unified_voice.emo_perceiver_encoder
            self.emovec_layer = unified_voice.emovec_layer
            self.emo_layer = unified_voice.emo_layer

        def forward(self, speaker_style, semantic_features):
            # Match model_v2.merge_emovec: infer_v2_5 passes shape[-1]
            # (the 1024 feature width) as the conditioning length.
            semantic_lengths = torch.full(
                (semantic_features.shape[0],),
                semantic_features.shape[-1],
                dtype=torch.long,
                device=semantic_features.device,
            )
            encoded, mask = self.emotion_encoder(semantic_features, semantic_lengths)
            perceiver_mask = torch.nn.functional.pad(mask.squeeze(1), (1, 0), value=True)
            emotion = self.emotion_perceiver(encoded, perceiver_mask).squeeze(1)
            emotion = self.emo_layer(self.emovec_layer(emotion))
            speaker = self.speaker_projection(speaker_style)
            first = speaker + emotion
            zeros = torch.zeros(
                first.shape[0], 2, first.shape[1], dtype=first.dtype, device=first.device
            )
            return torch.cat((first.unsqueeze(1), zeros), dim=1)

    wrapper = GptConditioning(model).eval()
    torch.manual_seed(1234)
    speaker = torch.randn(1, 192)
    semantic = torch.randn(1, 49, 1024)
    with torch.no_grad():
        expected = wrapper(speaker, semantic).cpu().numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (speaker, semantic),
        str(output),
        input_names=["speaker_style", "semantic_features"],
        output_names=["conditioning"],
        dynamic_axes={
            "speaker_style": {0: "batch"},
            "semantic_features": {0: "batch", 1: "frames"},
            "conditioning": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["conditioning"],
        {
            "speaker_style": speaker.numpy(),
            "semantic_features": semantic.numpy(),
        },
    )[0]
    difference = np.abs(expected - actual)
    max_error = float(np.max(difference))
    mean_error = float(np.mean(difference))
    print(
        f"gpt_conditioning output_shape={list(actual.shape)} "
        f"max_abs_error={max_error:.9g} mean_abs_error={mean_error:.9g}"
    )
    if max_error > 0.003 or mean_error > 2e-5:
        raise RuntimeError(
            f"GPT conditioning ONNX parity failed: max={max_error}, mean={mean_error}"
        )

    # The reference duration is dynamic. Validate a second frame count so
    # TorchScript tracing cannot silently freeze the example length.
    semantic_dynamic = torch.randn(1, 63, 1024)
    with torch.no_grad():
        expected_dynamic = wrapper(speaker, semantic_dynamic).cpu().numpy()
    actual_dynamic = session.run(
        ["conditioning"],
        {"speaker_style": speaker.numpy(), "semantic_features": semantic_dynamic.numpy()},
    )[0]
    dynamic_error = float(np.max(np.abs(expected_dynamic - actual_dynamic)))
    print(f"gpt_conditioning dynamic_frames=63 max_abs_error={dynamic_error:.9g}")
    if dynamic_error > 0.003:
        raise RuntimeError(f"GPT conditioning dynamic-shape parity failed: {dynamic_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["campplus", "wav2vec2bert", "gpt-conditioning"]) 
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.component == "campplus":
        export_campplus(args.source, args.model_dir, args.output)
    elif args.component == "wav2vec2bert":
        export_wav2vec2bert(args.model_dir, args.output)
    elif args.component == "gpt-conditioning":
        export_gpt_conditioning(args.source, args.model_dir, args.output)


if __name__ == "__main__":
    main()
