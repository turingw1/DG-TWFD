from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DGTD training diagnostics from train.jsonl")
    parser.add_argument("--history", required=True, help="Path to logs/train.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots and summaries")
    return parser.parse_args()


def _load_history(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No history rows found in {path}")
    return rows


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_plot_series(out_path: Path, title: str, values: dict[str, list[float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 4))
    for label, series in values.items():
        plt.plot(series, label=label)
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _maybe_plot_bins(out_path: Path, title: str, values: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(list(range(len(values))), values)
    plt.title(title)
    plt.xlabel("time bin")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    history = _load_history(Path(args.history))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest = history[-1]
    summary = {
        "epochs": len(history),
        "latest_epoch": latest["epoch"],
        "latest_stage": latest.get("stage"),
        "latest_train_loss": latest.get("train_loss"),
        "latest_val_loss": latest.get("val_loss"),
        "latest_train_defect": latest.get("train_defect"),
        "latest_val_defect": latest.get("val_defect"),
        "latest_eta": latest.get("eta"),
        "latest_beta": latest.get("beta"),
        "latest_time_grid": latest.get("time_grid"),
    }
    _save_json(output_dir / "summary.json", summary)
    _save_json(
        output_dir / "latest_bins.json",
        {
            "q_phi": latest.get("q_phi", []),
            "q_D": latest.get("q_D", []),
            "D_bar": latest.get("D_bar", []),
            "K_bar": latest.get("K_bar", []),
            "HF_bar": latest.get("HF_bar", []),
        },
    )

    _maybe_plot_series(
        output_dir / "loss_curve.png",
        "DGTD Loss",
        {
            "train_loss": [float(row.get("train_loss", 0.0)) for row in history],
            "val_loss": [float(row.get("val_loss", 0.0)) for row in history],
        },
    )
    _maybe_plot_series(
        output_dir / "defect_curve.png",
        "DGTD Defect",
        {
            "train_defect": [float(row.get("train_defect", 0.0)) for row in history],
            "val_defect": [float(row.get("val_defect", 0.0)) for row in history],
            "train_low_sigma_hf": [float(row.get("train_low_sigma_hf", 0.0)) for row in history],
        },
    )
    for key in ("q_phi", "q_D", "D_bar", "K_bar", "HF_bar"):
        _maybe_plot_bins(output_dir / f"{key}.png", key, [float(x) for x in latest.get(key, [])])

    print("dgtd diagnostics prepared")
    print(f"history: {args.history}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
