#!/usr/bin/env python3
"""Prepare CIFAR-10 in the directory layout expected by CTM.

CTM's CIFAR loader reads class labels from the parent directory name when
``--class_cond=True``. This script converts the standard CIFAR-10 python batch
files into:

    <output>/<class_id>/<split>_<index>.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


TRAIN_BATCHES = [f"data_batch_{idx}" for idx in range(1, 6)]
TEST_BATCHES = ["test_batch"]


def _load_batch(path: Path) -> tuple[np.ndarray, list[int]]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    data = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch["labels"]
    return data, labels


def _write_split(input_dir: Path, output_dir: Path, batch_names: list[str], split: str) -> int:
    count = 0
    for batch_name in batch_names:
        images, labels = _load_batch(input_dir / batch_name)
        for offset, (image, label) in enumerate(zip(images, labels)):
            class_dir = output_dir / str(int(label))
            class_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image.astype(np.uint8), mode="RGB").save(
                class_dir / f"{split}_{batch_name}_{offset:05d}.png"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_count = _write_split(args.input_dir, args.output_dir, TRAIN_BATCHES, "train")
    test_count = 0
    if args.include_test:
        test_count = _write_split(args.input_dir, args.output_dir, TEST_BATCHES, "test")

    print(f"wrote train={train_count} test={test_count} images to {args.output_dir}")


if __name__ == "__main__":
    main()
