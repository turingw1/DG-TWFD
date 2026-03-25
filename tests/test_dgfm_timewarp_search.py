from __future__ import annotations

from pathlib import Path

from dgfm.evaluators.timewarp_search import TimewarpSearchRunner


def test_timewarp_search_aggregate_objective() -> None:
    runner = TimewarpSearchRunner(
        config={
            "timewarp_search": {
                "objective_steps": [8, 16],
                "objective_weights": [0.25, 0.75],
            }
        },
        checkpoint=Path("dummy.pt"),
        eval_root=Path("dummy_eval"),
    )
    result = runner._aggregate_gamma_rows(
        [
            {"gamma": 1.5, "strategy": "data_dense_power@1.5000", "step_count": 8, "fid": 20.0},
            {"gamma": 1.5, "strategy": "data_dense_power@1.5000", "step_count": 16, "fid": 10.0},
        ]
    )
    assert abs(result["objective"] - 12.5) < 1.0e-8
