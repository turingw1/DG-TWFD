from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from dgfm.samplers import rollout_trajectory_with_map, rollout_with_map
from dgfm.schedulers import summarize_time_grid


@torch.no_grad()
def build_mode_a_time_grid(
    *,
    warp,
    step_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if not hasattr(warp, "r_to_t"):
        raise TypeError("DGTD Mode A sampling requires a warp with r_to_t()")
    r_grid = torch.linspace(0.0, 1.0, steps=step_count + 1, device=device, dtype=dtype)
    t_grid = warp.r_to_t(r_grid).to(device=device, dtype=dtype)
    t_grid[0] = torch.tensor(0.0, device=device, dtype=dtype)
    t_grid[-1] = torch.tensor(1.0, device=device, dtype=dtype)
    if not torch.all(t_grid[1:] >= t_grid[:-1]):
        raise ValueError("DGTD Mode A generated a non-monotone time grid")
    return t_grid


@torch.no_grad()
def rollout_mode_a(
    *,
    model: torch.nn.Module,
    x_init: torch.Tensor,
    warp,
    step_count: int,
    extra: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_grid = build_mode_a_time_grid(
        warp=warp,
        step_count=step_count,
        device=x_init.device,
        dtype=x_init.dtype,
    )
    samples = rollout_with_map(
        model=model,
        x_init=x_init,
        step_count=step_count,
        time_grid=time_grid,
        extra=extra,
    )
    return samples, time_grid


def _chunk_extra(extra: dict | None, start: int, stop: int) -> dict | None:
    if not extra:
        return None
    chunked: dict[str, Any] = {}
    for key, value in extra.items():
        if torch.is_tensor(value) and value.shape[0] >= stop:
            chunked[key] = value[start:stop]
        else:
            chunked[key] = value
    return chunked


def _validate_schedule_grid(time_grid: torch.Tensor, step_count: int) -> torch.Tensor:
    if time_grid.ndim != 1:
        raise ValueError(f"schedule time_grid must be 1D, got shape={tuple(time_grid.shape)}")
    if time_grid.numel() != step_count + 1:
        raise ValueError(f"schedule time_grid must have {step_count + 1} points, got {time_grid.numel()}")
    if not torch.all(time_grid[1:] >= time_grid[:-1]):
        raise ValueError("schedule time_grid must be monotone non-decreasing")
    if abs(float(time_grid[0].item())) > 1.0e-7 or abs(float(time_grid[-1].item()) - 1.0) > 1.0e-7:
        raise ValueError("schedule time_grid must start at 0 and end at 1")
    return time_grid


def _pairwise_rollout_cost(
    *,
    model: torch.nn.Module,
    states: torch.Tensor,
    time_grid: torch.Tensor,
    i: int,
    j: int,
    extra: dict | None,
    cost_batch_size: int,
) -> float:
    total = 0.0
    count = int(states.shape[0])
    batch_limit = count if cost_batch_size <= 0 else min(count, int(cost_batch_size))
    for start in range(0, count, batch_limit):
        stop = min(count, start + batch_limit)
        x_i = states[start:stop, i]
        t = time_grid[i].to(dtype=x_i.dtype).expand(stop - start)
        s = time_grid[j].to(dtype=x_i.dtype).expand(stop - start)
        pred = model(x_i, t, s, extra=_chunk_extra(extra, start, stop))
        err = (pred - states[start:stop, j]).detach().float().flatten(1).square().mean(dim=1)
        total += float(err.sum().item())
    return total / max(count, 1)


@torch.no_grad()
def search_oss_time_grid(
    *,
    model: torch.nn.Module,
    x_search: torch.Tensor,
    step_count: int,
    reference_steps: int = 32,
    warp=None,
    extra: dict | None = None,
    cost_batch_size: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Search a global explicit-map schedule with DP over dense rollout states.

    The cost for interval `(i, j)` is the MSE between one student map call
    `x_i -> x_j` and the dense reference rollout state at `j`. This is a
    lightweight OSS baseline adapted to this repo's explicit map interface.
    """
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if reference_steps < step_count:
        raise ValueError(f"reference_steps={reference_steps} must be >= step_count={step_count}")
    if warp is None:
        reference_grid = torch.linspace(0.0, 1.0, steps=reference_steps + 1, device=x_search.device, dtype=x_search.dtype)
        reference_source = "uniform_reference"
    else:
        reference_grid = build_mode_a_time_grid(
            warp=warp,
            step_count=reference_steps,
            device=x_search.device,
            dtype=x_search.dtype,
        )
        reference_source = "learned_warp_reference"

    reference_states, reference_grid = rollout_trajectory_with_map(
        model=model,
        x_init=x_search,
        step_count=reference_steps,
        time_grid=reference_grid,
        extra=extra,
    )

    inf = float("inf")
    costs = torch.full((reference_steps + 1, reference_steps + 1), inf, device=x_search.device, dtype=torch.float64)
    for i in range(reference_steps):
        for j in range(i + 1, reference_steps + 1):
            costs[i, j] = _pairwise_rollout_cost(
                model=model,
                states=reference_states,
                time_grid=reference_grid,
                i=i,
                j=j,
                extra=extra,
                cost_batch_size=cost_batch_size,
            )

    dp = torch.full((step_count + 1, reference_steps + 1), inf, device=x_search.device, dtype=torch.float64)
    prev = torch.full((step_count + 1, reference_steps + 1), -1, device=x_search.device, dtype=torch.long)
    dp[0, 0] = 0.0
    for k in range(1, step_count + 1):
        for j in range(1, reference_steps + 1):
            candidates = dp[k - 1, :j] + costs[:j, j]
            best_cost, best_idx = torch.min(candidates, dim=0)
            dp[k, j] = best_cost
            prev[k, j] = best_idx

    if not torch.isfinite(dp[step_count, reference_steps]):
        raise RuntimeError("OSS schedule DP failed to find a finite path")

    indices = [reference_steps]
    cursor = reference_steps
    for k in range(step_count, 0, -1):
        cursor = int(prev[k, cursor].item())
        if cursor < 0:
            raise RuntimeError("OSS schedule DP backtracking failed")
        indices.append(cursor)
    indices.reverse()
    if indices[0] != 0 or indices[-1] != reference_steps or len(indices) != step_count + 1:
        raise RuntimeError(f"Invalid OSS indices after backtracking: {indices}")

    time_grid = reference_grid[torch.tensor(indices, device=reference_grid.device)]
    time_grid[0] = torch.tensor(0.0, device=time_grid.device, dtype=time_grid.dtype)
    time_grid[-1] = torch.tensor(1.0, device=time_grid.device, dtype=time_grid.dtype)
    _validate_schedule_grid(time_grid, step_count)
    interval_costs = [float(costs[i, j].item()) for i, j in zip(indices, indices[1:])]
    total_cost = float(dp[step_count, reference_steps].item())
    payload: dict[str, Any] = {
        "mode": "oss_schedule_search",
        "status": "ok",
        "cost": "student_rollout_state_mse",
        "step_count": int(step_count),
        "reference_steps": int(reference_steps),
        "search_batch_size": int(x_search.shape[0]),
        "reference_source": reference_source,
        "indices": [int(item) for item in indices],
        "reference_time_grid": [float(item) for item in reference_grid.detach().cpu().tolist()],
        "time_grid": [float(item) for item in time_grid.detach().cpu().tolist()],
        "interval_costs": interval_costs,
        "total_cost": total_cost,
        "mean_interval_cost": total_cost / float(step_count),
    }
    payload.update({f"schedule_{key}": value for key, value in summarize_time_grid(time_grid).items()})
    return time_grid, payload


def save_schedule(payload: dict[str, Any], out_path: str | Path) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_time_grid_from_schedule(
    path: str | Path,
    *,
    step_count: int | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "time_grid" not in payload:
        raise ValueError(f"Schedule JSON has no time_grid: {path}")
    values = torch.tensor(payload["time_grid"], device=device, dtype=dtype)
    expected_steps = int(step_count if step_count is not None else payload.get("step_count", values.numel() - 1))
    _validate_schedule_grid(values, expected_steps)
    return values, payload


@torch.no_grad()
def rollout_mode_b_oss(
    *,
    model: torch.nn.Module,
    x_init: torch.Tensor,
    time_grid: torch.Tensor,
    extra: dict | None = None,
) -> torch.Tensor:
    step_count = int(time_grid.numel() - 1)
    return rollout_with_map(model=model, x_init=x_init, step_count=step_count, time_grid=time_grid, extra=extra)


def export_dp_schedule_stub(
    *,
    out_path: str | Path,
    step_count: int,
    note: str = "TODO: estimate interval costs C(i,j) and run DP node selection.",
) -> dict[str, object]:
    payload = {
        "mode": "dp_schedule_todo",
        "step_count": int(step_count),
        "status": "not_implemented",
        "note": note,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_mode_a(time_grid: torch.Tensor) -> dict[str, float | list[float]]:
    return summarize_time_grid(time_grid)
