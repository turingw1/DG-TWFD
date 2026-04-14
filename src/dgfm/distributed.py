from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


@dataclass(slots=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = ""
    device: torch.device | None = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed(config: dict) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(module):
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def maybe_wrap_ddp(module, ctx: DistributedContext, *, find_unused_parameters: bool = False):
    if not ctx.enabled:
        return module
    if ctx.backend == "nccl":
        return DistributedDataParallel(
            module,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(module, find_unused_parameters=find_unused_parameters)


def is_distributed_sampler(sampler) -> bool:
    return isinstance(sampler, DistributedSampler)


def set_dataloader_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def reduce_scalar(value: float, *, device: torch.device, ctx: DistributedContext, op: str = "mean") -> float:
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor /= float(ctx.world_size)
    elif op != "sum":
        raise ValueError(f"Unsupported reduce op: {op}")
    return float(tensor.item())


def reduce_float_dict(
    payload: dict[str, float],
    *,
    device: torch.device,
    ctx: DistributedContext,
    mean_keys: Iterable[str],
    sum_keys: Iterable[str] = (),
) -> dict[str, float]:
    if not ctx.enabled:
        return payload
    reduced = dict(payload)
    for key in mean_keys:
        if key in reduced:
            reduced[key] = reduce_scalar(reduced[key], device=device, ctx=ctx, op="mean")
    for key in sum_keys:
        if key in reduced:
            reduced[key] = reduce_scalar(reduced[key], device=device, ctx=ctx, op="sum")
    return reduced
