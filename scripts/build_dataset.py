from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for dgfm experiments")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "imagenet32", "imagenet64"], help="Dataset name")
    parser.add_argument("--data-root", required=True, help="Dataset root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        transform = transforms.ToTensor()
        datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        print(f"prepared cifar10 at {data_root}")
        return

    print(
        f"dataset scaffold prepared for {args.dataset} at {data_root}. "
        "Manual ImageNet population is still required in Phase 1.",
        flush=True,
    )


if __name__ == "__main__":
    main()
