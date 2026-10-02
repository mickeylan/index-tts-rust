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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["campplus"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.component == "campplus":
        export_campplus(args.source, args.model_dir, args.output)


if __name__ == "__main__":
    main()
