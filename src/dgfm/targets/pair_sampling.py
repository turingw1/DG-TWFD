from __future__ import annotations

import torch


def _sample_jump_delta(max_delta: int, target_cfg: dict, device: torch.device) -> int:
    short_max = min(max_delta, int(target_cfg.get("pair_short_max", 4)))
    mid_max = min(max_delta, int(target_cfg.get("pair_mid_max", 12)))
    long_max = min(max_delta, int(target_cfg.get("pair_long_max", 32)))
    choices: list[tuple[int, int, float]] = []
    if short_max >= 1:
        choices.append((1, short_max, float(target_cfg.get("pair_short_weight", 0.55))))
    if mid_max >= short_max + 1:
        choices.append((short_max + 1, mid_max, float(target_cfg.get("pair_mid_weight", 0.30))))
    if long_max >= mid_max + 1:
        choices.append((mid_max + 1, long_max, float(target_cfg.get("pair_long_weight", 0.15))))
    if not choices:
        return 1
    weights = torch.tensor([weight for _, _, weight in choices], dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    bucket_idx = int(torch.multinomial(weights, 1).item())
    low, high, _ = choices[bucket_idx]
    return int(torch.randint(low, high + 1, (1,), device=device).item())


def sample_pair_indices(num_points: int, target_cfg: dict, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}")
    endpoint_prob = float(target_cfg.get("pair_endpoint_weight", 0.35))
    high_noise_weight = float(target_cfg.get("high_noise_t_weight", 0.75))
    high_noise_fraction = float(target_cfg.get("high_noise_t_fraction", 0.35))
    max_delta = num_points - 1
    t_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    s_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    for idx in range(batch_size):
        if torch.rand(1, device=device).item() < endpoint_prob:
            s_index = num_points - 1
            max_start = max(1, s_index)
            high_noise_limit = max(1, int(round(max_start * high_noise_fraction)))
            if torch.rand(1, device=device).item() < high_noise_weight:
                t_index = int(torch.randint(0, high_noise_limit, (1,), device=device).item())
            else:
                t_index = int(torch.randint(0, max_start, (1,), device=device).item())
        else:
            delta = _sample_jump_delta(max_delta=max_delta, target_cfg=target_cfg, device=device)
            max_start = num_points - delta
            if max_start <= 1:
                t_index = 0
            else:
                high_noise_limit = max(1, int(round(max_start * high_noise_fraction)))
                if torch.rand(1, device=device).item() < high_noise_weight:
                    t_index = int(torch.randint(0, high_noise_limit, (1,), device=device).item())
                else:
                    t_index = int(torch.randint(0, max_start, (1,), device=device).item())
            s_index = t_index + delta
        t_indices[idx] = t_index
        s_indices[idx] = s_index
    return t_indices, s_indices
