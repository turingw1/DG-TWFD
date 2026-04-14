from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dgfm.config import load_experiment_config
from dgfm.models import build_map_model


class SyntheticMapDataset(Dataset):
    def __init__(self, length: int, channels: int, image_size: int, seed: int) -> None:
        self.length = int(length)
        generator = torch.Generator().manual_seed(seed)
        self.x_t = torch.randn(self.length, channels, image_size, image_size, generator=generator)
        self.delta = 0.08 * torch.randn(self.length, channels, image_size, image_size, generator=generator)
        t = torch.rand(self.length, generator=generator) * 0.55
        gap = 0.15 + 0.25 * torch.rand(self.length, generator=generator)
        s = torch.clamp(t + gap, max=1.0)
        self.t = t
        self.s = s

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x_t = self.x_t[index]
        t = self.t[index]
        s = self.s[index]
        target = x_t + self.delta[index] * (s - t)
        return {
            "x_t": x_t,
            "t": t,
            "s": s,
            "target": target,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal 2-GPU DDP smoke test for the explicit-map model.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-process batch size.")
    parser.add_argument("--steps", type=int, default=4, help="Number of optimizer steps to run.")
    parser.add_argument("--dataset-size", type=int, default=64, help="Synthetic dataset size.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic dataset seed.")
    parser.add_argument("--out", default="", help="Optional json report path.")
    return parser.parse_args()


def _require_env(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required distributed env var: {name}")
    return int(value)


def main() -> None:
    args = parse_args()
    local_rank = _require_env("LOCAL_RANK")
    rank = _require_env("RANK")
    world_size = _require_env("WORLD_SIZE")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DDP smoke test.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    try:
        config = load_experiment_config(args.config)
        dataset_cfg = config["dataset"]
        model = build_map_model(config).to(device)
        ddp_model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)

        dataset = SyntheticMapDataset(
            length=args.dataset_size,
            channels=int(dataset_cfg["channels"]),
            image_size=int(dataset_cfg["image_size"]),
            seed=args.seed,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        losses: list[float] = []
        step_count = 0
        data_iter = iter(loader)
        for epoch in range(8):
            sampler.set_epoch(epoch)
            for batch in loader:
                x_t = batch["x_t"].to(device, non_blocking=True)
                t = batch["t"].to(device, non_blocking=True)
                s = batch["s"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = ddp_model(x_t, t, s, extra={})
                loss = torch.mean((pred - target) ** 2)
                loss.backward()
                optimizer.step()

                reduced = loss.detach().clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced = reduced / float(world_size)
                losses.append(float(reduced.item()))
                step_count += 1
                if step_count >= args.steps:
                    break
            if step_count >= args.steps:
                break

        report = {
            "success": True,
            "world_size": world_size,
            "steps": step_count,
            "rank": rank,
            "local_rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "losses": losses,
            "batch_size_per_rank": args.batch_size,
            "dataset_size": args.dataset_size,
            "config": args.config,
        }

        if args.out:
            out_path = Path(args.out)
            if rank == 0:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if rank == 0:
            print(json.dumps(report, ensure_ascii=False))
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
