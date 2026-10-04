#!/usr/bin/env python3
"""Convert official IndexTTS-2.5 emotion prototype tensors to safetensors."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

GROUP_COUNTS = [3, 17, 2, 8, 4, 5, 10, 24]


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} does not contain one tensor")
    return value.detach().to(dtype=torch.float32, device="cpu").contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True, type=Path, help="official feat1.pt")
    parser.add_argument("--emotion", required=True, type=Path, help="official feat2.pt")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    speaker = load_tensor(args.speaker)
    emotion = load_tensor(args.emotion)
    if list(speaker.shape) != [73, 192]:
        raise ValueError(f"unexpected speaker prototype shape {list(speaker.shape)}")
    if list(emotion.shape) != [73, 1280]:
        raise ValueError(f"unexpected emotion prototype shape {list(emotion.shape)}")
    offsets = [0]
    for count in GROUP_COUNTS:
        offsets.append(offsets[-1] + count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "speaker_prototypes": speaker,
            "emotion_prototypes": emotion,
            "group_offsets": torch.tensor(offsets, dtype=torch.int64),
        },
        str(args.output),
        metadata={"emotion_order": "happy,angry,sad,afraid,disgusted,melancholic,surprised,calm"},
    )
    print(f"wrote {args.output}: speaker={list(speaker.shape)} emotion={list(emotion.shape)} offsets={offsets}")


if __name__ == "__main__":
    main()
