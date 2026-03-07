import argparse
import torch

def walk(obj, prefix="root", max_depth=4, depth=0):
    if depth > max_depth:
        print(f"{prefix}: ...")
        return

    if torch.is_tensor(obj):
        print(f"{prefix}: Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device}")
    elif isinstance(obj, dict):
        print(f"{prefix}: dict(len={len(obj)})")
        for k, v in obj.items():
            walk(v, f"{prefix}.{k}", max_depth, depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__}(len={len(obj)})")
        for i, v in enumerate(obj[:20]):  # 防止太长
            walk(v, f"{prefix}[{i}]", max_depth, depth + 1)
        if len(obj) > 20:
            print(f"{prefix}: ... ({len(obj)-20} more)")
    else:
        print(f"{prefix}: {type(obj).__name__}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", type=str)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    data = torch.load(args.pt_path, map_location="cpu")
    walk(data, max_depth=args.max_depth)

if __name__ == "__main__":
    main()