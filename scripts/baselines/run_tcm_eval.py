from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TCM_ROOT = ROOT / "refs" / "tcm"
EDM_ROOT = ROOT / "refs" / "edm"
FID_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")

FIELDNAMES = [
    "dataset",
    "method",
    "step",
    "fid",
    "is",
    "recall",
    "checkpoint",
    "eval_script",
    "notes",
]

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "checkpoint": "/cache/Zhengwei/DG-TWFD/checkpoints/baselines/tcm/tcm_cifar10_ddpmpp.pkl",
        "method": "TCM-official-DDPM++",
        "sample_root": "runs/tcm_cifar10_5k/samples",
        "eval_root": "eval/tcm_cifar10_5k",
        "csv_out": "results/baselines/baseline_tcm_cifar10.csv",
        "fid_ref": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
        "arch": "ddpmpp",
        "cond": False,
        "fp16": False,
        "resolution": 32,
        "num_classes": 0,
        "batch": 512,
    },
    "imagenet64": {
        "checkpoint": "/cache/Zhengwei/DG-TWFD/checkpoints/baselines/tcm/tcm_imgnet64_edm2s.pkl",
        "method": "TCM-official-EDM2-S",
        "sample_root": "runs/tcm_imagenet64_5k/samples",
        "eval_root": "eval/tcm_imagenet64_5k",
        "csv_out": "results/baselines/baseline_tcm_imagenet64.csv",
        "fid_ref": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz",
        "arch": "edm2-img64-s",
        "cond": True,
        "fp16": True,
        "resolution": 64,
        "num_classes": 1000,
        "batch": 128,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TCM official checkpoint generation and EDM FID.")
    subparsers = parser.add_subparsers(dest="command")

    main = subparsers.add_parser("run")
    main.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    main.add_argument("--checkpoint", default=None)
    main.add_argument("--method", default=None)
    main.add_argument("--sample-root", default=None)
    main.add_argument("--eval-root", default=None)
    main.add_argument("--csv-out", default=None)
    main.add_argument("--fid-ref", default=None)
    main.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    main.add_argument("--num-samples", type=int, default=5000)
    main.add_argument("--batch", type=int, default=None)
    main.add_argument("--fid-batch", type=int, default=512)
    main.add_argument("--seed", type=int, default=42)
    main.add_argument("--skip-sample", action="store_true")
    main.add_argument("--skip-fid", action="store_true")
    main.add_argument("--force", action="store_true")
    main.add_argument(
        "--stable-report-root",
        default="/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines",
    )

    worker = subparsers.add_parser("_generate")
    worker.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    worker.add_argument("--data", required=True)
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--outdir", required=True)
    worker.add_argument("--arch", required=True)
    worker.add_argument("--cond", action=argparse.BooleanOptionalAction, default=False)
    worker.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    worker.add_argument("--batch", type=int, required=True)
    worker.add_argument("--num-samples", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--mid-t", nargs="*", type=float, default=[])
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _dataset_defaults(args: argparse.Namespace) -> dict[str, Any]:
    defaults = dict(DATASET_DEFAULTS[args.dataset])
    defaults["checkpoint"] = args.checkpoint or defaults["checkpoint"]
    defaults["method"] = args.method or defaults["method"]
    defaults["sample_root"] = args.sample_root or defaults["sample_root"]
    defaults["eval_root"] = args.eval_root or defaults["eval_root"]
    defaults["csv_out"] = args.csv_out or defaults["csv_out"]
    defaults["fid_ref"] = args.fid_ref or defaults["fid_ref"]
    defaults["batch"] = args.batch or defaults["batch"]
    return defaults


def _ensure_schema_dataset(dataset: str, *, resolution: int, num_classes: int) -> Path:
    root = Path("/cache/Zhengwei/DG-TWFD/datasets/tcm_schema") / dataset
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "000000.png"
    if not image_path.exists():
        arr = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_path)
    json_path = root / "dataset.json"
    if num_classes > 0:
        label = num_classes - 1
        payload = {"labels": [["000000.png", label]]}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
    elif json_path.exists():
        json_path.unlink()
    return root


def _mid_t_for_step_count(step_count: int) -> list[float]:
    if step_count < 1:
        raise ValueError(f"step count must be >= 1, got {step_count}")
    if step_count == 1:
        return []
    official_two_step_mid = 0.821
    if step_count == 2:
        return [official_two_step_mid]
    values = np.geomspace(official_two_step_mid, 0.002, step_count - 1)
    return [float(f"{value:.8g}") for value in values]


def _run_and_tee(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as handle:
        for line in proc.stdout:
            captured.append(line)
            handle.write(line)
            handle.flush()
            print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command, output="".join(captured))
    return "".join(captured)


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("sample_*.png") if item.is_file())


def _parse_fid(stdout: str) -> float | None:
    for line in reversed(stdout.splitlines()):
        match = FID_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def _load_completed_metric(metrics_path: Path, image_dir: Path, *, expected: int) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    try:
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("fid") is None:
        return None
    if int(row.get("num_fid_samples", 0) or 0) != expected:
        return None
    if _count_pngs(image_dir) < expected:
        return None
    return row


def _write_preview(image_dir: Path, preview_path: Path, *, limit: int = 64) -> None:
    images = sorted(image_dir.glob("sample_*.png"))[:limit]
    if not images:
        return
    thumbs = []
    for path in images:
        image = Image.open(path).convert("RGB")
        image.thumbnail((64, 64))
        thumbs.append(image.copy())
    side = int(np.ceil(np.sqrt(len(thumbs))))
    canvas = Image.new("RGB", (side * 64, side * 64), "white")
    for idx, image in enumerate(thumbs):
        canvas.paste(image, ((idx % side) * 64, (idx // side) * 64))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path)


def _write_outputs(*, rows: list[dict[str, Any]], eval_root: Path, csv_out: Path, stable_report_root: Path) -> None:
    report_dir = eval_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": rows}, handle, indent=2)
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "step": row["step_count"],
                    "fid": "" if row["fid"] is None else f"{float(row['fid']):.6f}",
                    "is": "",
                    "recall": "",
                    "checkpoint": row["checkpoint"],
                    "eval_script": "scripts/baselines/run_tcm_eval.py",
                    "notes": row["notes"],
                }
            )

    stable_report_root.mkdir(parents=True, exist_ok=True)
    stem = csv_out.stem
    for source in [report_dir / "summary.json", report_dir / "summary.csv", csv_out]:
        target = stable_report_root / f"{stem}_{source.name}"
        target.write_bytes(source.read_bytes())


def _run_worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(TCM_ROOT))

    import torch
    import dnnlib
    from generate import evaluation
    from torch_utils import distributed as dist

    torch.multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 20000))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init()

    network_kwargs = dnnlib.EasyDict()
    if args.arch == "ddpmpp":
        network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
            attn_type="dot",
            emb_norm=False,
        )
    elif args.arch == "edm2-img64-s":
        network_kwargs.update(model_type="EDM2UNet", model_channels=192)
        network_kwargs.update(scale=1.0, emb_norm=False, learnable_scale=False)
    else:
        raise ValueError(f"Unsupported TCM arch: {args.arch}")
    network_kwargs.class_name = "training.networks.ECMPrecond"
    network_kwargs.update(dropout=0.13, use_fp16=args.fp16)

    try:
        evaluation(
            run_dir=args.outdir,
            dataset_kwargs=dnnlib.EasyDict(
                class_name="training.dataset.ImageFolderDataset",
                path=args.data,
                use_labels=args.cond,
                xflip=False,
                cache=False,
            ),
            network_kwargs=network_kwargs,
            batch_size=args.batch,
            seed=args.seed,
            resume_pkl=args.checkpoint,
            mid_t=list(args.mid_t),
            metrics=[],
            cudnn_benchmark=True,
            num_samples=args.num_samples,
            class_label=None,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_main(args: argparse.Namespace) -> None:
    defaults = _dataset_defaults(args)
    checkpoint = _resolve(defaults["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not TCM_ROOT.exists():
        raise FileNotFoundError(TCM_ROOT)
    if not EDM_ROOT.exists():
        raise FileNotFoundError(EDM_ROOT)

    schema_data = _ensure_schema_dataset(
        args.dataset,
        resolution=int(defaults["resolution"]),
        num_classes=int(defaults["num_classes"]),
    )
    sample_root = _resolve(defaults["sample_root"])
    eval_root = _resolve(defaults["eval_root"])
    csv_out = _resolve(defaults["csv_out"])
    stable_report_root = _resolve(args.stable_report_root)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(EDM_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("DNNLIB_CACHE_DIR", str(ROOT / ".torch" / "dnnlib"))
    env.setdefault("OMPI_MCA_btl", "^openib")
    env.setdefault("OMPI_MCA_btl_openib_warn_no_device_params_found", "0")
    env.setdefault("NCCL_DEBUG", "WARN")

    rows: list[dict[str, Any]] = []
    for step_count in args.steps:
        step_dir = eval_root / f"steps{step_count}"
        image_dir = sample_root / f"steps{step_count}"
        metrics_path = step_dir / "metrics.json"
        step_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        completed = None if args.force else _load_completed_metric(
            metrics_path,
            image_dir,
            expected=args.num_samples,
        )
        if completed is not None and not args.skip_fid:
            rows.append(completed)
            _write_outputs(
                rows=rows,
                eval_root=eval_root,
                csv_out=csv_out,
                stable_report_root=stable_report_root,
            )
            print(f"reuse tcm {args.dataset} step_count={step_count} fid={completed['fid']}", flush=True)
            continue

        mid_t = _mid_t_for_step_count(step_count)
        if not args.skip_sample and _count_pngs(image_dir) < args.num_samples:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_generate",
                "--dataset",
                args.dataset,
                "--data",
                str(schema_data),
                "--checkpoint",
                str(checkpoint),
                "--outdir",
                str(image_dir),
                "--arch",
                str(defaults["arch"]),
                "--batch",
                str(defaults["batch"]),
                "--num-samples",
                str(args.num_samples),
                "--seed",
                str(args.seed + 1000 * step_count),
            ]
            command.append("--cond" if defaults["cond"] else "--no-cond")
            command.append("--fp16" if defaults["fp16"] else "--no-fp16")
            if mid_t:
                command.append("--mid-t")
                command.extend(str(value) for value in mid_t)
            _run_and_tee(command, cwd=ROOT, env=env, log_path=step_dir / "sample.stdout_stderr.txt")

        png_count = _count_pngs(image_dir)
        if png_count < args.num_samples:
            raise RuntimeError(f"Only found {png_count}/{args.num_samples} images for step_count={step_count}")

        _write_preview(image_dir, step_dir / "preview_first64.png")
        fid = None
        if not args.skip_fid:
            fid_stdout = _run_and_tee(
                [
                    sys.executable,
                    str(EDM_ROOT / "fid.py"),
                    "calc",
                    f"--images={image_dir}",
                    f"--ref={defaults['fid_ref']}",
                    f"--num={args.num_samples}",
                    f"--batch={args.fid_batch}",
                ],
                cwd=EDM_ROOT,
                env=env,
                log_path=step_dir / "fid.stdout_stderr.txt",
            )
            fid = _parse_fid(fid_stdout)

        schedule_note = (
            "official 1-step path"
            if step_count == 1
            else "official README 2-step mid_t=0.821"
            if step_count == 2
            else "geometric extension from official mid_t=0.821 to sigma_min=0.002"
        )
        row = {
            "dataset": args.dataset,
            "step_count": int(step_count),
            "fid": fid,
            "num_fid_samples": int(args.num_samples),
            "checkpoint": str(checkpoint),
            "method": defaults["method"],
            "image_dir": str(image_dir),
            "mid_t": mid_t,
            "elapsed_sec": time.time() - t0,
            "notes": "; ".join(
                [
                    "official TCM checkpoint and sampling code",
                    schedule_note,
                    f"num_fid_samples={args.num_samples}",
                    f"png_count={png_count}",
                    "recall pending reference sample batch",
                ]
            ),
        }
        metrics_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        _write_outputs(
            rows=rows,
            eval_root=eval_root,
            csv_out=csv_out,
            stable_report_root=stable_report_root,
        )
        print(f"tcm {args.dataset} step_count={step_count} fid={fid} elapsed_sec={row['elapsed_sec']:.2f}", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "_generate":
        _run_worker(args)
    elif args.command == "run":
        _run_main(args)
    else:
        raise SystemExit("Choose a command: run or _generate")


if __name__ == "__main__":
    main()
