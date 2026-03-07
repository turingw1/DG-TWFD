import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: (3, H, W), float tensor
    return: (H, W, 3), uint8
    """
    x = img_chw.detach().cpu().float()
    mn, mx = x.min().item(), x.max().item()

    # 常见范围自动处理
    if mn >= 0.0 and mx <= 1.0:
        x = x
    elif mn >= -1.0 and mx <= 1.0:
        x = (x + 1.0) / 2.0
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    x = x.clamp(0, 1)
    x = (x * 255.0).round().byte()          # (3,H,W)
    x = x.permute(1, 2, 0).numpy()          # (H,W,3)
    return x


def concat_h(images, pad=4, pad_value=255):
    """把一组 HWC uint8 图片横向拼接"""
    h = images[0].shape[0]
    arrs = []
    for i, im in enumerate(images):
        arrs.append(im)
        if i != len(images) - 1 and pad > 0:
            arrs.append(np.full((h, pad, 3), pad_value, dtype=np.uint8))
    return np.concatenate(arrs, axis=1)


def save_one_sample(sample: dict, out_dir: Path, idx: int):
    x0 = sample["x0"]          # (3,32,32)
    xg = sample["x_grid"]      # (8,3,32,32)
    tg = sample["t_grid"]      # (8,)

    sample_dir = out_dir / f"sample_{idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 单独保存 x0
    Image.fromarray(chw_to_uint8(x0)).save(sample_dir / "x0.png")

    # 每个时间步单独保存
    frames = []
    for j in range(xg.shape[0]):
        img = chw_to_uint8(xg[j])
        frames.append(img)
        t = float(tg[j].item())
        Image.fromarray(img).save(sample_dir / f"x_grid_{j:02d}_t{t:.3f}.png")

    # 生成一张总览图：x0 | x_grid[0] | ... | x_grid[7]
    overview = concat_h([chw_to_uint8(x0)] + frames, pad=3)
    Image.fromarray(overview).save(sample_dir / "overview.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", type=str, help="输入 .pt 文件")
    parser.add_argument("--out_dir", type=str, default="pt_vis", help="输出目录")
    parser.add_argument("--index", type=int, default=None, help="只可视化某个样本 index")
    args = parser.parse_args()

    data = torch.load(args.pt_path, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError("该 pt 顶层应为 list")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        if args.index < 0 or args.index >= len(data):
            raise IndexError(f"index 越界: {args.index}, 有效范围 [0, {len(data)-1}]")
        save_one_sample(data[args.index], out_dir, args.index)
        print(f"完成：sample_{args.index:03d}")
    else:
        for i, sample in enumerate(data):
            save_one_sample(sample, out_dir, i)
        print(f"完成：共 {len(data)} 个样本，输出目录: {out_dir}")


if __name__ == "__main__":
    main()