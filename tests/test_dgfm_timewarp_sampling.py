from __future__ import annotations

import torch

from dgfm.schedulers import TimeWarpMonotone, build_config_time_grid, build_time_grid, list_timewarp_strategies
from dgfm.trainers.map import _compute_timewarp_defect_loss


def test_list_timewarp_strategies_contains_phase_a_set() -> None:
    names = set(list_timewarp_strategies())
    assert {"uniform", "source_dense_power2", "data_dense_power2", "random_dirichlet"} <= names


def test_uniform_grid_has_expected_shape_and_bounds() -> None:
    grid = build_time_grid(4, "uniform", device=torch.device("cpu"), dtype=torch.float32)
    assert grid.shape == (5,)
    assert float(grid[0]) == 0.0
    assert float(grid[-1]) == 1.0
    assert torch.all(grid[1:] >= grid[:-1])


def test_random_dirichlet_grid_is_monotone_and_deterministic() -> None:
    grid_a = build_time_grid(
        8,
        "random_dirichlet",
        device=torch.device("cpu"),
        dtype=torch.float32,
        random_seed=123,
    )
    grid_b = build_time_grid(
        8,
        "random_dirichlet",
        device=torch.device("cpu"),
        dtype=torch.float32,
        random_seed=123,
    )
    assert torch.allclose(grid_a, grid_b)
    assert float(grid_a[0]) == 0.0
    assert float(grid_a[-1]) == 1.0
    assert torch.all(grid_a[1:] >= grid_a[:-1])


def test_parameterized_power_grid_string_is_supported() -> None:
    grid = build_time_grid(
        4,
        "data_dense_power@1.5",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert grid.shape == (5,)
    assert float(grid[0]) == 0.0
    assert float(grid[-1]) == 1.0
    assert torch.all(grid[1:] >= grid[:-1])


def test_learnable_monotone_timewarp_preserves_endpoints() -> None:
    module = TimeWarpMonotone(num_bins=8, init_bias=0.0)
    t = torch.linspace(0.0, 1.0, steps=9)
    warped = module(t)
    assert float(warped[0].detach()) == 0.0
    assert float(warped[-1].detach()) == 1.0
    assert torch.all(warped[1:] >= warped[:-1])


def test_build_config_time_grid_uses_enabled_timewarp() -> None:
    cfg = {
        "scheduler": {
            "timewarp": {
                "enabled": True,
                "type": "data_dense_power@2.0",
            }
        }
    }
    grid = build_config_time_grid(cfg, 4, device=torch.device("cpu"), dtype=torch.float32)
    uniform = torch.linspace(0.0, 1.0, steps=5)
    assert grid.shape == (5,)
    assert float(grid[0]) == 0.0
    assert float(grid[-1]) == 1.0
    assert torch.all(grid[1:] >= grid[:-1])
    assert not torch.allclose(grid, uniform)


class _NonSemigroupMap(torch.nn.Module):
    def forward(self, x_t, t, s, extra=None):
        del extra
        delta = (s - t).view(-1, 1, 1, 1)
        bias = (0.25 + t).view(-1, 1, 1, 1)
        return x_t + delta * bias


def test_defect_driven_timewarp_update_changes_grid_and_reduces_loss() -> None:
    torch.manual_seed(0)
    cfg = {
        "loss": {
            "timewarp_defect_steps": 4,
            "timewarp_batch_size": 4,
            "timewarp_defect_weight": 1.0,
            "timewarp_balance_weight": 0.05,
        }
    }
    model = _NonSemigroupMap()
    timewarp = TimeWarpMonotone(num_bins=8, init_bias=0.0)
    optimizer = torch.optim.Adam(timewarp.parameters(), lr=0.1)
    x_0 = torch.zeros(4, 1, 2, 2)

    initial_loss, initial_stats = _compute_timewarp_defect_loss(
        model=model,
        timewarp=timewarp,
        x_0=x_0,
        config=cfg,
        device=torch.device("cpu"),
    )
    initial_grid = torch.tensor(initial_stats["time_grid"])
    initial_value = float(initial_loss.detach().item())

    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, _stats = _compute_timewarp_defect_loss(
            model=model,
            timewarp=timewarp,
            x_0=x_0,
            config=cfg,
            device=torch.device("cpu"),
        )
        loss.backward()
        optimizer.step()

    final_loss, final_stats = _compute_timewarp_defect_loss(
        model=model,
        timewarp=timewarp,
        x_0=x_0,
        config=cfg,
        device=torch.device("cpu"),
    )
    final_grid = torch.tensor(final_stats["time_grid"])
    final_value = float(final_loss.detach().item())

    assert final_value < initial_value
    assert not torch.allclose(initial_grid, final_grid)
    assert torch.all(final_grid[1:] >= final_grid[:-1])
