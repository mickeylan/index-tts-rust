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


def export_semantic_codec(source: Path, model_dir: Path, output: Path) -> None:
    sys.path.insert(0, str(source.resolve()))
    from omegaconf import OmegaConf
    from torch import nn
    from indextts.codec.models import EnhancedCodec

    config = OmegaConf.load(model_dir / "config.yaml")
    codec = EnhancedCodec(**config.semantic_codec, cfg=config.semantic_codec)
    codec.load_checkpoint(str(model_dir / "codec.pth"))
    codec.eval()

    class Decoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, codes):
            return self.model.decode(codes)

    wrapper = Decoder(codec).eval()
    torch.manual_seed(1234)
    codes = torch.randint(0, config.semantic_codec.codebook_size, (1, 17), dtype=torch.long)
    with torch.no_grad():
        expected = wrapper(codes).cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (codes,),
        str(output),
        input_names=["codes"],
        output_names=["semantic_features"],
        dynamic_axes={
            "codes": {0: "batch", 1: "codes"},
            "semantic_features": {0: "batch", 1: "frames"},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(["semantic_features"], {"codes": codes.numpy()})[0]
    difference = np.abs(expected - actual)
    max_error = float(np.max(difference))
    mean_error = float(np.mean(difference))
    print(
        f"semantic_codec output_shape={list(actual.shape)} "
        f"max_abs_error={max_error:.9g} mean_abs_error={mean_error:.9g}"
    )
    if max_error > 0.003 or mean_error > 2e-5:
        raise RuntimeError(
            f"semantic codec ONNX parity failed: max={max_error}, mean={mean_error}"
        )

    dynamic_codes = torch.randint(0, config.semantic_codec.codebook_size, (1, 23), dtype=torch.long)
    with torch.no_grad():
        dynamic_expected = wrapper(dynamic_codes).cpu().numpy()
    dynamic_actual = session.run(
        ["semantic_features"], {"codes": dynamic_codes.numpy()}
    )[0]
    dynamic_error = float(np.max(np.abs(dynamic_expected - dynamic_actual)))
    print(
        f"semantic_codec dynamic_codes=23 output_shape={list(dynamic_actual.shape)} "
        f"max_abs_error={dynamic_error:.9g}"
    )
    if dynamic_error > 0.003:
        raise RuntimeError(f"semantic codec dynamic parity failed: {dynamic_error}")


def load_s2mel(source: Path, model_dir: Path):
    sys.path.insert(0, str(source.resolve()))
    import types
    from omegaconf import OmegaConf

    # IndexTTS imports VectorQuantize unconditionally although the 2.5 length
    # regulator has vector_quantize=false. Avoid pulling the unrelated
    # audiotools training dependency into this offline inference export.
    quantize_module = "indextts.s2mel.dac.nn.quantize"
    if quantize_module not in sys.modules:
        stub = types.ModuleType(quantize_module)
        class UnusedVectorQuantize:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("VectorQuantize is disabled by the IndexTTS-2.5 config")
        stub.VectorQuantize = UnusedVectorQuantize
        sys.modules[quantize_module] = stub

    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    config = OmegaConf.load(model_dir / "config.yaml")
    model = MyModel(config.s2mel)
    model, _, _, _ = load_checkpoint2(
        model, None, str(model_dir / config.s2mel_checkpoint),
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    return config, model.eval()


def export_length_regulator(source: Path, model_dir: Path, output: Path) -> None:
    from torch import nn

    _, model = load_s2mel(source, model_dir)

    class LengthRegulator(nn.Module):
        def __init__(self, regulator):
            super().__init__()
            self.regulator = regulator

        def forward(self, semantic_features, target_length):
            return self.regulator(
                semantic_features,
                ylens=target_length,
                n_quantizers=3,
                f0=None,
            )[0]

    wrapper = LengthRegulator(model.models["length_regulator"]).eval()
    torch.manual_seed(1234)
    semantic = torch.randn(1, 34, 1024)
    target_length = torch.tensor([58], dtype=torch.long)
    with torch.no_grad():
        expected = wrapper(semantic, target_length).cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (semantic, target_length),
        str(output),
        input_names=["semantic_features", "target_length"],
        output_names=["condition"],
        dynamic_axes={
            "semantic_features": {0: "batch", 1: "semantic_frames"},
            "target_length": {0: "batch"},
            "condition": {0: "batch", 1: "mel_frames"},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["condition"],
        {"semantic_features": semantic.numpy(), "target_length": target_length.numpy()},
    )[0]
    difference = np.abs(expected - actual)
    max_error = float(np.max(difference))
    mean_error = float(np.mean(difference))
    print(
        f"length_regulator output_shape={list(actual.shape)} "
        f"max_abs_error={max_error:.9g} mean_abs_error={mean_error:.9g}"
    )
    if max_error > 0.003 or mean_error > 2e-5:
        raise RuntimeError(
            f"length regulator ONNX parity failed: max={max_error}, mean={mean_error}"
        )

    dynamic_semantic = torch.randn(1, 46, 1024)
    dynamic_length = torch.tensor([79], dtype=torch.long)
    with torch.no_grad():
        expected_dynamic = wrapper(dynamic_semantic, dynamic_length).cpu().numpy()
    actual_dynamic = session.run(
        ["condition"],
        {"semantic_features": dynamic_semantic.numpy(), "target_length": dynamic_length.numpy()},
    )[0]
    dynamic_error = float(np.max(np.abs(expected_dynamic - actual_dynamic)))
    print(
        f"length_regulator dynamic_output_shape={list(actual_dynamic.shape)} "
        f"max_abs_error={dynamic_error:.9g}"
    )
    if dynamic_error > 0.003:
        raise RuntimeError(f"length regulator dynamic parity failed: {dynamic_error}")


def export_dit(source: Path, model_dir: Path, output: Path, frames: int) -> None:
    _, model = load_s2mel(source, model_dir)
    estimator = model.models["cfm"].estimator
    estimator.setup_caches(max_batch_size=2, max_seq_length=8192)
    estimator.eval()
    # Avoid a legacy TorchScript exporter bug for Tensor + bool in the
    # sequence-mask expression; these flags are false in the official config.
    if estimator.style_as_token is False:
        estimator.style_as_token = 0
    if estimator.time_as_token is False:
        estimator.time_as_token = 0

    # The scripted helper cannot be inspected by the legacy ONNX tracer.
    import indextts.s2mel.modules.commons as commons_module
    from indextts.s2mel.modules import wavenet as wavenet_module
    original_common = commons_module.fused_add_tanh_sigmoid_multiply
    original_wavenet = wavenet_module.commons.fused_add_tanh_sigmoid_multiply
    def plain_fused(input_a, input_b, n_channels):
        channels = n_channels[0]
        activation = input_a + input_b
        return torch.tanh(activation[:, :channels, :]) * torch.sigmoid(activation[:, channels:, :])
    commons_module.fused_add_tanh_sigmoid_multiply = plain_fused
    wavenet_module.commons.fused_add_tanh_sigmoid_multiply = plain_fused
    for module in estimator.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
        except ValueError:
            pass

    if frames < 64 or frames > 8192:
        raise ValueError("fixed DiT frame bucket must be in [64, 8192]")
    torch.manual_seed(1234)
    batch = 2
    x = torch.randn(batch, 80, frames)
    prompt = torch.randn(batch, 80, frames)
    lengths = torch.full((batch,), frames, dtype=torch.long)
    time = torch.tensor([0.4, 0.4])
    style = torch.randn(batch, 192)
    condition = torch.randn(batch, frames, 512)
    with torch.no_grad():
        expected = estimator(x, prompt, lengths, time, style, condition).cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            estimator,
            (x, prompt, lengths, time, style, condition),
            str(output),
            input_names=["x", "prompt_x", "x_lens", "t", "style", "condition"],
            output_names=["velocity"],
            # Time is intentionally fixed. The current PyTorch model contains
            # shape-dependent WaveNet padding which cannot be exported safely
            # as a symbolic dimension. Runtime selects and pads to a bucket.
            dynamic_axes={
                "x": {0: "batch"}, "prompt_x": {0: "batch"},
                "x_lens": {0: "batch"}, "t": {0: "batch"}, "style": {0: "batch"},
                "condition": {0: "batch"}, "velocity": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    finally:
        commons_module.fused_add_tanh_sigmoid_multiply = original_common
        wavenet_module.commons.fused_add_tanh_sigmoid_multiply = original_wavenet

    import onnxruntime as ort
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    inputs = {
        "x": x.numpy(), "prompt_x": prompt.numpy(), "x_lens": lengths.numpy(),
        "t": time.numpy(), "style": style.numpy(), "condition": condition.numpy(),
    }
    actual = session.run(["velocity"], inputs)[0]
    difference = np.abs(expected - actual)
    max_error = float(np.max(difference))
    mean_error = float(np.mean(difference))
    print(
        f"dit output_shape={list(actual.shape)} max_abs_error={max_error:.9g} "
        f"mean_abs_error={mean_error:.9g}"
    )
    if max_error > 0.01 or mean_error > 1e-4:
        raise RuntimeError(f"DiT ONNX parity failed: max={max_error}, mean={mean_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "component",
        choices=["campplus", "wav2vec2bert", "gpt-conditioning", "semantic-codec", "length-regulator", "dit"],
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=256, help="fixed DiT frame bucket")
    args = parser.parse_args()

    if args.component == "campplus":
        export_campplus(args.source, args.model_dir, args.output)
    elif args.component == "wav2vec2bert":
        export_wav2vec2bert(args.model_dir, args.output)
    elif args.component == "gpt-conditioning":
        export_gpt_conditioning(args.source, args.model_dir, args.output)
    elif args.component == "semantic-codec":
        export_semantic_codec(args.source, args.model_dir, args.output)
    elif args.component == "length-regulator":
        export_length_regulator(args.source, args.model_dir, args.output)
    elif args.component == "dit":
        export_dit(args.source, args.model_dir, args.output, args.frames)


if __name__ == "__main__":
    main()
