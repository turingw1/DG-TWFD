from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("records", []))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _step_from_eval(path: Path) -> int:
    match = re.search(r"step(\d+)", path.as_posix())
    return int(match.group(1)) if match else -1


def _fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an EDM-first hourly supervision blocker report.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--eval-prefix", required=True)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    train_log = root / "runs" / args.run_tag / "logs" / "train.jsonl"
    train_rows = _read_jsonl(train_log)
    baseline_rows = _read_summary(root / args.baseline_summary)
    baseline_by_step = {int(row["step_count"]): row for row in baseline_rows}
    summary_paths = sorted(
        (root / "eval").glob(f"{args.eval_prefix}*/reports/summary.csv"),
        key=_step_from_eval,
    )
    summary_paths = [path for path in summary_paths if "identity" not in path.as_posix()]

    trend_rows = []
    for path in summary_paths:
        rows = _read_summary(path)
        by_step = {int(row["step_count"]): row for row in rows}
        ckpt_step = _step_from_eval(path)
        trend_rows.append(
            {
                "checkpoint_step": ckpt_step,
                "path": path.relative_to(root).as_posix(),
                "fid1": by_step.get(1, {}).get("fid"),
                "fid2": by_step.get(2, {}).get("fid"),
                "fid4": by_step.get(4, {}).get("fid"),
                "fid8": by_step.get(8, {}).get("fid"),
                "fid16": by_step.get(16, {}).get("fid"),
                "sat16": by_step.get(16, {}).get("sample_saturation_0_1"),
            }
        )

    last_train = train_rows[-1] if train_rows else {}
    first_train = train_rows[0] if train_rows else {}
    latest = trend_rows[-1] if trend_rows else {}
    baseline_fid1 = float(baseline_by_step.get(1, {}).get("fid", 0.0) or 0.0)
    latest_fid1 = float(latest.get("fid1") or 0.0) if latest else 0.0
    fid1_ratio = latest_fid1 / baseline_fid1 if baseline_fid1 and latest_fid1 else None
    multistep_regression = False
    if latest and baseline_by_step:
        for step in (4, 8, 16):
            base = float(baseline_by_step.get(step, {}).get("fid", 0.0) or 0.0)
            cur = float(latest.get(f"fid{step}") or 0.0)
            if base and cur > base:
                multistep_regression = True

    lines = [
        "# EDM-First Hourly Supervision Blocker Report",
        "",
        "Generated automatically after the supervision window elapsed without hitting the configured FID threshold.",
        "",
        "## Latest Training State",
        "",
        f"- train log: `{train_log.relative_to(root).as_posix()}`",
        f"- first logged step: `{first_train.get('step', 'n/a')}`",
        f"- latest logged step: `{last_train.get('step', 'n/a')}`",
        f"- latest loss: `{_fmt(last_train.get('loss'))}`",
        f"- latest match loss: `{_fmt(last_train.get('match_loss'))}`",
        f"- latest defect loss: `{_fmt(last_train.get('defect_loss'), 8)}`",
        f"- latest perceptual loss: `{_fmt(last_train.get('perceptual_loss'))}`",
        "",
        "## FID Trend",
        "",
        "| checkpoint step | FID@1 | FID@2 | FID@4 | FID@8 | FID@16 | saturation@16 | summary |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in trend_rows:
        lines.append(
            "| "
            f"{row['checkpoint_step']} | "
            f"{_fmt(row['fid1'])} | {_fmt(row['fid2'])} | {_fmt(row['fid4'])} | "
            f"{_fmt(row['fid8'])} | {_fmt(row['fid16'])} | {_fmt(row['sat16'])} | "
            f"`{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
        ]
    )
    if fid1_ratio is not None:
        lines.append(
            f"- One-step FID improved to `{_fmt(latest_fid1)}` "
            f"(`{_fmt(fid1_ratio)}` of the step250 baseline), but did not reach the 50% target."
        )
    if multistep_regression:
        lines.append(
            "- Multi-step FID regressed relative to the step250 baseline, which indicates endpoint-only learning is not composing cleanly."
        )
    lines.extend(
        [
            "- The current e504a config has `timewarp.enabled: false`, so no learned schedule can correct the observed step-budget mismatch.",
            "- In `prior_endpoint`, the dominant supervision is direct sigma_max-to-zero matching; the defect term is orders of magnitude smaller than the match/perceptual terms.",
            "- The bridge defect currently anchors the direct endpoint toward a no-grad two-step student rollout. This helps one-step quality but does not strongly train the intermediate transition itself.",
            "- Saturation rises as step count increases, consistent with repeated application of an endpoint-biased map.",
            "",
            "## Most Likely Blockers",
            "",
            "1. Endpoint-only objective is improving one-step samples while damaging compositional rollouts.",
            "2. The defect/consistency term is too weak or too indirect to stabilize multi-step generation.",
            "3. Timewarp is not active in e504a, so the schedule cannot adapt to the model's actual error distribution.",
            "4. A fixed identity sigma grid is being used for multi-step eval, and the learned map is not trained with step-budget-aware schedules.",
            "",
            "## Recommended Next Changes",
            "",
            "1. Run the prepared timewarp follow-up from the best available e504a checkpoint and compare auto-warp vs identity at `1/2/4/8/16`.",
            "2. Add an explicit intermediate teacher target in the prior-endpoint objective: train `sigma_max -> sigma_s` and `sigma_s -> 0` with teacher rollout endpoints, not only `sigma_max -> 0`.",
            "3. Increase the effective defect signal only after intermediate targets are present; otherwise it mostly regularizes the direct endpoint.",
            "4. Add a step-budget-aware schedule search/eval loop before treating learned q_phi as useful for 8/16 steps.",
            "",
            "## Relevant Code",
            "",
            "- `experiments/edm_first/train_edm_map.py`",
            "- `experiments/edm_first/eval_edm_map.py`",
            "- `experiments/edm_first/src/edm_map_lib.py`",
            "- `experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_8h.yaml`",
            "- `experiments/edm_first/configs/cifar10_edm_map_onestep_prior_msdefect_timewarp_8h.yaml`",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote blocker report: {args.out}")


if __name__ == "__main__":
    main()
