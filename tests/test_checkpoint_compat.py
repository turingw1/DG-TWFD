from __future__ import annotations

import torch

from dg_twfd.engine.checkpoint import load_model_state_dict


def test_load_model_state_dict_accepts_orig_mod_prefix() -> None:
    model = torch.nn.Linear(4, 3)
    prefixed = {f"_orig_mod.{key}": value.clone() for key, value in model.state_dict().items()}
    load_model_state_dict(model, prefixed)
    out = model(torch.randn(2, 4))
    assert out.shape == (2, 3)

