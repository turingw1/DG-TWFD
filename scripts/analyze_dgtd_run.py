from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_config(path: Path) -> dict[str, Any]:
    try:
        from dgfm.config import load_experiment_config

        return load_experiment_config(str(path))
    except Exception:
        return _load_yaml(path)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_optional_json(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _find_default_run_paths(run_root: Path | None) -> dict[str, Path | None]:
    if run_root is None:
        return {"log": None, "eval_root": None, "sample_root": None, "checkpoint": None}
    return {
        "log": run_root / "logs" / "train.jsonl",
        "eval_root": run_root / "eval",
        "sample_root": run_root / "samples",
        "checkpoint": run_root / "checkpoints" / "best.pt",
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}g}"


def _get_series(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _to_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _first_last_delta(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"first": None, "last": None, "delta": None, "ratio": None, "min": None, "max": None}
    first = values[0]
    last = values[-1]
    return {
        "first": first,
        "last": last,
        "delta": last - first,
        "ratio": last / first if abs(first) > 1.0e-12 else None,
        "min": min(values),
        "max": max(values),
    }


def _best_row(rows: list[dict[str, Any]], key: str, *, mode: str = "min") -> dict[str, Any] | None:
    candidates = [(idx, row, _to_float(row.get(key))) for idx, row in enumerate(rows)]
    candidates = [(idx, row, value) for idx, row, value in candidates if value is not None]
    if not candidates:
        return None
    reverse = mode == "max"
    _, row, _ = sorted(candidates, key=lambda item: item[2], reverse=reverse)[0]
    return row


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("stage", "unknown"))
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def _last_dict(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _latest_existing_file(candidates: list[Path]) -> Path | None:
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _eval_summary(eval_root: Path | None) -> dict[str, Any]:
    if eval_root is None or not eval_root.exists():
        return {"available": False}
    summary_path = eval_root / "reports" / "summary.json"
    csv_path = eval_root / "reports" / "summary.csv"
    step_metrics: dict[str, Any] = {}
    for metrics_path in sorted(eval_root.glob("steps*/metrics.json")):
        step_metrics[metrics_path.parent.name] = _load_json(metrics_path)
    summary = _load_json(summary_path) if summary_path.exists() else None
    csv_rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
    return {
        "available": True,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "csv_path": str(csv_path) if csv_path.exists() else None,
        "summary": summary,
        "csv_rows": csv_rows,
        "step_metrics": step_metrics,
    }


def _metric_value(payload: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _to_float(payload.get(name))
        if value is not None:
            return value
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for name in names:
            value = _to_float(metrics.get(name))
            if value is not None:
                return value
    return None


def _step_fid_table(eval_info: dict[str, Any]) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for step_name, payload in eval_info.get("step_metrics", {}).items():
        if not step_name.startswith("steps"):
            continue
        try:
            step = int(step_name.replace("steps", ""))
        except ValueError:
            continue
        if isinstance(payload, dict):
            fid = _metric_value(payload, ("fid", "approx_fid", "frechet_inception_distance"))
            if fid is not None:
                rows.append((step, fid))
    rows.sort(key=lambda item: item[0])
    return rows


def _sample_tensor_stats(sample_root: Path | None) -> dict[str, Any]:
    if sample_root is None or not sample_root.exists():
        return {"available": False}
    candidates = list(sample_root.glob("**/samples.pt"))
    sample_path = _latest_existing_file(candidates)
    if sample_path is None:
        return {"available": False, "sample_root": str(sample_root)}
    try:
        import torch
    except ImportError:
        return {"available": False, "sample_path": str(sample_path), "error": "torch not importable"}
    samples = torch.load(sample_path, map_location="cpu")
    if not isinstance(samples, torch.Tensor):
        return {"available": False, "sample_path": str(sample_path), "error": "samples.pt is not a tensor"}
    x = samples.float()
    flat = x.flatten()
    saturation = ((flat <= 0.01) | (flat >= 0.99)).float().mean().item()
    if x.ndim >= 4 and x.shape[-1] > 1 and x.shape[-2] > 1:
        h0 = x[..., :, :-1].flatten()
        h1 = x[..., :, 1:].flatten()
        v0 = x[..., :-1, :].flatten()
        v1 = x[..., 1:, :].flatten()
        corr_h = _corrcoef(h0, h1)
        corr_v = _corrcoef(v0, v1)
    else:
        corr_h = None
        corr_v = None
    return {
        "available": True,
        "sample_path": str(sample_path),
        "shape": list(samples.shape),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "saturation_0_1": float(saturation),
        "neighbor_corr_h": corr_h,
        "neighbor_corr_v": corr_v,
    }


def _corrcoef(x, y) -> float | None:
    import torch

    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.clamp((x.square().mean() * y.square().mean()), min=1.0e-12))
    value = (x * y).mean() / denom
    result = float(value.item())
    return result if math.isfinite(result) else None


def _checkpoint_keys(checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None or not checkpoint.exists():
        return {"available": False}
    try:
        import torch
    except ImportError:
        return {"available": False, "checkpoint": str(checkpoint), "error": "torch not importable"}
    try:
        ckpt = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "checkpoint": str(checkpoint), "error": repr(exc)}
    if not isinstance(ckpt, dict):
        return {"available": True, "checkpoint": str(checkpoint), "type": type(ckpt).__name__}
    return {
        "available": True,
        "checkpoint": str(checkpoint),
        "keys": sorted(str(key) for key in ckpt.keys()),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
    }


def _gate_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "pass" if value else "fail"


def _teacher_endpoint_gate(report: dict[str, Any] | None) -> tuple[bool | None, list[str]]:
    if not isinstance(report, dict):
        return None, ["teacher endpoint report missing"]
    verdict = report.get("verdict")
    if not isinstance(verdict, dict):
        return None, ["teacher endpoint report has no verdict"]
    messages = []
    if not bool(verdict.get("u0_is_not_clean_input", False)):
        messages.append("teacher u0 appears too close to clean input")
    if not bool(verdict.get("u1_closer_to_clean_than_u0", False)):
        messages.append("teacher u1 is not closer to clean input than u0")
    if not bool(verdict.get("endpoint_order_ok", False)):
        messages.append("teacher endpoint time order is invalid")
    return bool(verdict.get("pass", False)), messages


def _step_curve_gate(eval_info: dict[str, Any]) -> tuple[bool | None, list[str]]:
    fids = _step_fid_table(eval_info)
    if len(fids) < 2:
        return None, ["step FID table missing or too short"]
    values = [fid for _, fid in fids]
    messages = []
    for (prev_step, prev_fid), (next_step, next_fid) in zip(fids, fids[1:]):
        if next_fid > prev_fid * 1.10:
            messages.append(f"FID worsens by >10% from steps{prev_step} to steps{next_step}")
    if values[-1] > values[0] * 1.05:
        messages.append("final step FID is >5% worse than first step FID")
    return not messages, messages


def _q_phi_gate(rows: list[dict[str, Any]], config: dict[str, Any] | None) -> tuple[bool | None, list[str]]:
    if not rows:
        return None, ["training rows missing"]
    last = rows[-1]
    q_phi = last.get("q_phi")
    if not isinstance(q_phi, list) or not q_phi:
        return None, ["q_phi missing from train log"]
    entropy = _to_float(last.get("entropy_q_phi"))
    max_ratio = _to_float(last.get("max_qphi_over_qbase"))
    num_bins = len(q_phi)
    min_entropy = 0.5 * math.log(max(num_bins, 2))
    ratio_cap = 10.0
    if config is not None:
        ratio_cap = float(config.get("dgtd", {}).get("gate_max_qphi_over_qbase", ratio_cap))
        min_entropy = float(config.get("dgtd", {}).get("gate_min_entropy_q_phi", min_entropy))
    messages = []
    if entropy is not None and entropy < min_entropy:
        messages.append(f"entropy_q_phi={entropy:.4g} below gate {min_entropy:.4g}")
    if max_ratio is not None and max_ratio > ratio_cap:
        messages.append(f"max_qphi_over_qbase={max_ratio:.4g} above gate {ratio_cap:.4g}")
    return not messages, messages


def _sample_gate(sample_info: dict[str, Any]) -> tuple[bool | None, list[str]]:
    if not sample_info.get("available"):
        return None, ["sample tensor stats missing"]
    corr_h = _to_float(sample_info.get("neighbor_corr_h"))
    corr_v = _to_float(sample_info.get("neighbor_corr_v"))
    saturation = _to_float(sample_info.get("saturation_0_1"))
    messages = []
    if corr_h is None or corr_v is None:
        messages.append("neighbor correlation unavailable")
    elif max(corr_h, corr_v) < 0.15:
        messages.append("neighbor correlation below 0.15; samples are likely noise-like")
    if saturation is not None and saturation > 0.95:
        messages.append("sample saturation above 0.95")
    return not messages, messages


def _online_continuation_gate(rows: list[dict[str, Any]]) -> tuple[bool | None, list[str]]:
    if not rows:
        return None, ["training rows missing"]
    last = rows[-1]
    online_rate = _to_float(last.get("online_continuation_rate"))
    fallback_rate = _to_float(last.get("cached_fallback_rate"))
    messages = []
    if online_rate is None:
        return None, ["online_continuation_rate missing"]
    if online_rate < 0.9:
        messages.append(f"online_continuation_rate={online_rate:.4g} below 0.9")
    if fallback_rate is not None and fallback_rate > 0.1:
        messages.append(f"cached_fallback_rate={fallback_rate:.4g} above 0.1")
    return not messages, messages


def _gate_verdict(
    rows: list[dict[str, Any]],
    train_info: dict[str, Any],
    eval_info: dict[str, Any],
    sample_info: dict[str, Any],
    checkpoint_info: dict[str, Any],
    config: dict[str, Any] | None,
    teacher_endpoint_report: dict[str, Any] | None,
) -> dict[str, Any]:
    del train_info
    checks: dict[str, dict[str, Any]] = {}
    for name, result in {
        "teacher_endpoint_ok": _teacher_endpoint_gate(teacher_endpoint_report),
        "online_continuation_primary": _online_continuation_gate(rows),
        "sample_not_noise_like": _sample_gate(sample_info),
        "q_phi_not_collapsed": _q_phi_gate(rows, config),
        "step_curve_not_regressive": _step_curve_gate(eval_info),
        "checkpoint_readable": (bool(checkpoint_info.get("available")), [] if checkpoint_info.get("available") else ["checkpoint missing or unreadable"]),
    }.items():
        passed, messages = result
        checks[name] = {
            "status": _gate_bool(passed),
            "pass": passed,
            "messages": messages,
        }
    failed = [name for name, item in checks.items() if item["pass"] is False]
    unknown = [name for name, item in checks.items() if item["pass"] is None]
    if failed:
        status = "fail"
        next_action = "Do not launch full run; patch the failed module and rerun the same diagnostic budget."
    elif unknown:
        status = "incomplete"
        next_action = "Collect missing diagnostics before deciding on a full run."
    else:
        status = "pass"
        next_action = "Diagnostic gate passed; full run can be considered."
    return {
        "status": status,
        "failed": failed,
        "unknown": unknown,
        "checks": checks,
        "next_action": next_action,
    }


def _training_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "train_loss",
        "val_loss",
        "train_defect",
        "val_defect",
        "train_direct_teacher_error",
        "val_direct_teacher_error",
        "train_direct_bridge_gap",
        "val_direct_bridge_gap",
        "train_bridge_state_teacher_error",
        "val_bridge_state_teacher_error",
        "train_bridge_u_teacher_error",
        "val_bridge_u_teacher_error",
        "train_noisy_endpoint_error",
        "val_noisy_endpoint_error",
        "train_endpoint_anchor_loss",
        "val_endpoint_anchor_loss",
        "entropy_q_phi",
        "kl_qD_qphi",
        "max_qphi_over_qbase",
        "train_online_teacher_traj_sec",
        "train_target_build_sec",
        "train_forward_sec",
    ]
    trends = {key: _first_last_delta(_get_series(rows, key)) for key in keys}
    best_val_loss = _best_row(rows, "val_loss", mode="min")
    best_val_defect = _best_row(rows, "val_defect", mode="min")
    last = rows[-1] if rows else {}
    return {
        "num_rows": len(rows),
        "first_epoch": rows[0].get("epoch") if rows else None,
        "last_epoch": last.get("epoch"),
        "stage_counts": _stage_counts(rows),
        "trends": trends,
        "best_val_loss": _row_excerpt(best_val_loss),
        "best_val_defect": _row_excerpt(best_val_defect),
        "last": _row_excerpt(last),
        "last_continuation_sources": _last_dict(rows, "continuation_sources"),
    }


def _row_excerpt(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keep = [
        "epoch",
        "global_step",
        "stage",
        "train_loss",
        "val_loss",
        "train_defect",
        "val_defect",
        "train_direct_teacher_error",
        "val_direct_teacher_error",
        "train_direct_bridge_gap",
        "val_direct_bridge_gap",
        "train_bridge_state_teacher_error",
        "val_bridge_state_teacher_error",
        "train_bridge_u_teacher_error",
        "val_bridge_u_teacher_error",
        "train_noisy_endpoint_error",
        "val_noisy_endpoint_error",
        "train_endpoint_anchor_loss",
        "val_endpoint_anchor_loss",
        "train_teacher_rel_error_mean",
        "val_teacher_rel_error_mean",
        "eta",
        "beta",
        "flatten_mix",
        "entropy_q_phi",
        "kl_qD_qphi",
        "argmax_q_phi",
        "max_qphi_over_qbase",
        "online_continuation_rate",
        "cached_fallback_rate",
        "train_online_teacher_traj_sec",
        "train_target_build_sec",
        "train_forward_sec",
    ]
    return {key: row.get(key) for key in keep if key in row}


def _diagnoses(
    rows: list[dict[str, Any]],
    train_info: dict[str, Any],
    eval_info: dict[str, Any],
    sample_info: dict[str, Any],
    config: dict[str, Any] | None,
) -> list[str]:
    diagnoses: list[str] = []
    last = rows[-1] if rows else {}
    trends = train_info.get("trends", {})
    source = train_info.get("last_continuation_sources", {})
    online_rate = _to_float(last.get("online_continuation_rate"))
    if online_rate is not None and online_rate >= 0.9:
        diagnoses.append("online continuation is the dominant path; this run did not test cached fallback behavior.")
    elif source:
        diagnoses.append(f"continuation source is not cleanly online-dominant: {source}")

    train_loss = trends.get("train_loss", {})
    val_loss = trends.get("val_loss", {})
    train_ratio = train_loss.get("ratio") if isinstance(train_loss, dict) else None
    val_ratio = val_loss.get("ratio") if isinstance(val_loss, dict) else None
    if train_ratio is not None and val_ratio is not None and train_ratio < 0.1 and val_ratio > train_ratio * 2.0:
        diagnoses.append("optimization is healthy but validation gap remains; avoid interpreting train loss as sample quality.")

    teacher_time = _to_float(last.get("train_online_teacher_traj_sec"))
    forward_time = _to_float(last.get("train_forward_sec"))
    target_time = _to_float(last.get("train_target_build_sec"))
    if teacher_time is not None and forward_time is not None and target_time is not None:
        denom = max(teacher_time + forward_time + target_time, 1.0e-9)
        share = teacher_time / denom
        if share > 0.8:
            diagnoses.append(
                f"online teacher trajectory generation dominates runtime ({share:.1%} of measured teacher/target/forward time)."
            )

    noisy = trends.get("train_noisy_endpoint_error", {})
    defect = trends.get("train_defect", {})
    if isinstance(noisy, dict) and isinstance(defect, dict):
        noisy_ratio = noisy.get("ratio")
        defect_ratio = defect.get("ratio")
        if noisy_ratio is not None and defect_ratio is not None and noisy_ratio > defect_ratio * 3.0:
            diagnoses.append("noisy endpoint error improves much less than defect; few-step sampling may stay noise-like.")

    fids = _step_fid_table(eval_info)
    if fids:
        if all(fid > 100.0 for _, fid in fids):
            diagnoses.append("FID remains extremely high; this is consistent with noise-like samples despite stable training loss.")
        if len(fids) >= 2 and fids[-1][1] >= min(fid for _, fid in fids[:-1]) * 0.98:
            diagnoses.append("more sampling steps do not clearly improve quality; check map direction and teacher trajectory semantics.")

    if sample_info.get("available"):
        corr_h = _to_float(sample_info.get("neighbor_corr_h"))
        corr_v = _to_float(sample_info.get("neighbor_corr_v"))
        if corr_h is not None and corr_v is not None and max(corr_h, corr_v) < 0.15:
            diagnoses.append("sample tensor has very low neighbor correlation; generated images are likely noise-like.")

    if config:
        dgtd_cfg = config.get("dgtd", {})
        teacher_cfg = config.get("teacher", {})
        if (
            bool(dgtd_cfg.get("use_online_teacher_data", False))
            and not bool(dgtd_cfg.get("disable_online_teacher", True))
            and (
                str(teacher_cfg.get("type", "")).lower() in {"diffusers_ddpm", "ddpm", "diffusers", "sampler"}
                or str(teacher_cfg.get("backend", "")).lower() in {"diffusers_ddpm", "ddpm", "diffusers"}
            )
        ):
            if teacher_cfg.get("clean_input_range") is None:
                diagnoses.append(
                    "high-priority code audit flag: online teacher data appears to use dataloader images, but "
                    "teacher.clean_input_range is not set. Verify that clean images are forward-noised before "
                    "building the u=0 -> u=1 trajectory."
                )
            else:
                diagnoses.append(
                    "online teacher is configured for clean-image input. Verify endpoint stats: u=0 should be "
                    "noise-like and u=1 should be image-like, not identical to the dataloader image at u=0."
                )

    if not diagnoses:
        diagnoses.append("no obvious blocker detected from available logs; provide samples/eval metrics for stronger diagnosis.")
    return diagnoses


def _markdown_report(
    *,
    log_path: Path | None,
    train_info: dict[str, Any],
    eval_info: dict[str, Any],
    sample_info: dict[str, Any],
    checkpoint_info: dict[str, Any],
    gate_verdict: dict[str, Any],
    diagnoses: list[str],
) -> str:
    rows = []
    rows.append("# DGTD Run Analysis")
    rows.append("")
    rows.append("## Inputs")
    rows.append("")
    rows.append(f"- `train_log`: `{log_path}`" if log_path else "- `train_log`: n/a")
    rows.append(f"- `eval_available`: {bool(eval_info.get('available'))}")
    rows.append(f"- `sample_available`: {bool(sample_info.get('available'))}")
    rows.append(f"- `checkpoint_available`: {bool(checkpoint_info.get('available'))}")
    rows.append("")
    rows.append("## High-Priority Diagnoses")
    rows.append("")
    for item in diagnoses:
        rows.append(f"- {item}")
    rows.append("")
    rows.append("## Gate Verdict")
    rows.append("")
    rows.append(f"- `status`: `{gate_verdict.get('status')}`")
    rows.append(f"- `failed`: `{gate_verdict.get('failed')}`")
    rows.append(f"- `unknown`: `{gate_verdict.get('unknown')}`")
    rows.append(f"- `next_action`: {gate_verdict.get('next_action')}")
    checks = gate_verdict.get("checks", {})
    if isinstance(checks, dict):
        for name, item in checks.items():
            rows.append(f"- `{name}`: {item.get('status')} `{item.get('messages')}`")
    rows.append("")
    rows.append("## Training Trends")
    rows.append("")
    rows.append(f"- `num_rows`: {train_info.get('num_rows')}")
    rows.append(f"- `epoch_range`: {train_info.get('first_epoch')} -> {train_info.get('last_epoch')}")
    rows.append(f"- `stage_counts`: `{train_info.get('stage_counts')}`")
    rows.append(f"- `last_continuation_sources`: `{train_info.get('last_continuation_sources')}`")
    rows.append("")
    for key in (
        "train_loss",
        "val_loss",
        "train_defect",
        "val_defect",
        "train_direct_bridge_gap",
        "val_direct_bridge_gap",
        "train_noisy_endpoint_error",
        "val_noisy_endpoint_error",
        "train_endpoint_anchor_loss",
        "val_endpoint_anchor_loss",
        "train_online_teacher_traj_sec",
        "train_target_build_sec",
        "train_forward_sec",
        "entropy_q_phi",
        "kl_qD_qphi",
    ):
        trend = train_info.get("trends", {}).get(key, {})
        rows.append(
            f"- `{key}`: first={_fmt(trend.get('first'))}, last={_fmt(trend.get('last'))}, "
            f"min={_fmt(trend.get('min'))}, max={_fmt(trend.get('max'))}, ratio={_fmt(trend.get('ratio'))}"
        )
    rows.append("")
    rows.append("## Best Checkpoints From Log")
    rows.append("")
    rows.append(f"- `best_val_loss_row`: `{train_info.get('best_val_loss')}`")
    rows.append(f"- `best_val_defect_row`: `{train_info.get('best_val_defect')}`")
    rows.append("")
    rows.append("## Eval Summary")
    rows.append("")
    fids = _step_fid_table(eval_info)
    if fids:
        for step, fid in fids:
            rows.append(f"- `steps{step}` fid={fid:.6g}")
    else:
        rows.append("- no step-wise FID metrics found")
    rows.append("")
    rows.append("## Sample Tensor Summary")
    rows.append("")
    if sample_info.get("available"):
        for key in ("sample_path", "shape", "mean", "std", "min", "max", "saturation_0_1", "neighbor_corr_h", "neighbor_corr_v"):
            rows.append(f"- `{key}`: `{sample_info.get(key)}`")
    else:
        rows.append(f"- unavailable: `{sample_info}`")
    rows.append("")
    rows.append("## Checkpoint Summary")
    rows.append("")
    rows.append(f"- `{checkpoint_info}`")
    rows.append("")
    rows.append("## Next Evidence To Return")
    rows.append("")
    rows.append("- sample grid image for the checkpoint used in eval")
    rows.append("- one batch teacher trajectory endpoint stats: state at `u=0`, state at `u=1`, and dataloader image stats")
    rows.append("- eval metrics for `1 2 4 8` on both `best.pt` and the epoch with minimum `val_loss` if different")
    rows.append("- short run after fixing teacher initial-state semantics, if the audit flag applies")
    rows.append("")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DGTD v3 run logs into actionable experiment diagnostics.")
    parser.add_argument("--run-root", default=None, help="Run root containing logs/checkpoints/samples.")
    parser.add_argument("--log", default=None, help="Explicit train.jsonl path.")
    parser.add_argument("--eval-root", default=None, help="Explicit eval root.")
    parser.add_argument("--sample-root", default=None, help="Explicit sample root.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to inspect key structure.")
    parser.add_argument("--config", default=None, help="Optional resolved or experiment config for static diagnosis.")
    parser.add_argument("--teacher-endpoint-report", default=None, help="Optional JSON from scripts/diagnose_teacher_endpoints.py.")
    parser.add_argument("--output", default=None, help="Markdown report path. Defaults to stdout.")
    parser.add_argument("--json-output", default=None, help="Optional machine-readable JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root) if args.run_root else None
    defaults = _find_default_run_paths(run_root)
    log_path = Path(args.log) if args.log else defaults["log"]
    eval_root = Path(args.eval_root) if args.eval_root else defaults["eval_root"]
    sample_root = Path(args.sample_root) if args.sample_root else defaults["sample_root"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else defaults["checkpoint"]
    config = _load_config(Path(args.config)) if args.config else None

    if log_path is None or not log_path.exists():
        raise FileNotFoundError(f"train.jsonl not found. Pass --log or --run-root. Got: {log_path}")
    rows = _load_jsonl(log_path)
    if not rows:
        raise ValueError(f"No JSON rows found in {log_path}")

    train_info = _training_summary(rows)
    eval_info = _eval_summary(eval_root)
    sample_info = _sample_tensor_stats(sample_root)
    checkpoint_info = _checkpoint_keys(checkpoint)
    teacher_endpoint_report = _load_optional_json(Path(args.teacher_endpoint_report)) if args.teacher_endpoint_report else None
    gate_verdict = _gate_verdict(
        rows,
        train_info,
        eval_info,
        sample_info,
        checkpoint_info,
        config,
        teacher_endpoint_report,
    )
    diagnoses = _diagnoses(rows, train_info, eval_info, sample_info, config)
    report = _markdown_report(
        log_path=log_path,
        train_info=train_info,
        eval_info=eval_info,
        sample_info=sample_info,
        checkpoint_info=checkpoint_info,
        gate_verdict=gate_verdict,
        diagnoses=diagnoses,
    )

    payload = {
        "train_log": str(log_path),
        "train": train_info,
        "eval": eval_info,
        "sample": sample_info,
        "checkpoint": checkpoint_info,
        "teacher_endpoint": teacher_endpoint_report,
        "gate_verdict": gate_verdict,
        "diagnoses": diagnoses,
    }
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
