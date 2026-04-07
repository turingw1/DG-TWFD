from __future__ import annotations

import torch

from dgfm.schedulers import TimeWarpMonotone, build_config_time_grid, build_time_grid, list_timewarp_strategies


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
