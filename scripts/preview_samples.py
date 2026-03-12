from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stitched image preview from sampled tensors")
    parser.add_argument("--samples", required=True, help="Path to samples .pt tensor ([B,C,H,W])")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--nrow", type=int, default=8, help="Number of images per row")
    parser.add_argument("--max-images", type=int, default=64, help="Maximum images used in preview")
    parser.add_argument("--padding", type=int, default=2, help="Padding (pixels) between images")
    return parser.parse_args()


def to_image_tensor(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 3:
        samples = samples.unsqueeze(0)
    if samples.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] tensor, got shape={tuple(samples.shape)}")
    x = samples.detach().cpu().float()
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got C={x.shape[1]}")

    vmin = float(x.min().item())
    vmax = float(x.max().item())
    if vmin >= -1.2 and vmax <= 1.2:
        x = (x + 1.0) * 0.5
    else:
        x = (x - vmin) / max(1e-8, vmax - vmin)
    return torch.clamp(x, 0.0, 1.0)


def make_grid(x: torch.Tensor, nrow: int, padding: int) -> torch.Tensor:
    b, c, h, w = x.shape
    cols = max(1, nrow)
    rows = int(math.ceil(b / cols))
    grid_h = rows * h + (rows - 1) * padding
    grid_w = cols * w + (cols - 1) * padding
    grid = torch.zeros(c, grid_h, grid_w, dtype=x.dtype)
    for idx in range(b):
        r = idx // cols
        col = idx % cols
        top = r * (h + padding)
        left = col * (w + padding)
        grid[:, top : top + h, left : left + w] = x[idx]
    return grid


def resolve_output_path(samples_path: Path, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    stem = samples_path.stem
    return samples_path.parent / f"{stem}_preview.png"


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    payload = torch.load(samples_path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(payload):
        raise TypeError(f"Expected tensor in {samples_path}, got type={type(payload)}")
    x = to_image_tensor(payload)
    x = x[: max(1, min(args.max_images, x.shape[0]))]
    grid = make_grid(x, nrow=args.nrow, padding=max(0, args.padding))
    grid_u8 = (grid.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    image = Image.fromarray(grid_u8)
    out_path = resolve_output_path(samples_path, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"saved_preview: {out_path}")
    print(f"preview_shape: {grid_u8.shape}")
    print(f"images_used: {x.shape[0]}")


if __name__ == "__main__":
    main()

