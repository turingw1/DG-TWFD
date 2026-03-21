from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.evaluators import EvaluationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refactored DGFM evaluation")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--eval-root", required=True, help="Evaluation root directory")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="Few-step evaluation list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    runner = EvaluationRunner(
        config=config,
        checkpoint=Path(args.checkpoint),
        eval_root=Path(args.eval_root),
    )
    runner.run(step_counts=args.steps)


if __name__ == "__main__":
    main()
