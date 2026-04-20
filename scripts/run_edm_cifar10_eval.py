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
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EDM_ROOT = ROOT / "refs" / "edm"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config


FID_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public EDM CIFAR-10 inference and official FID eval")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--sample-root", required=True, help="Directory for generated EDM samples")
    parser.add_argument("--eval-root", required=True, help="Directory for EDM metric reports")
    parser.add_argument("--steps", nargs="+", type=int, default=None, help="EDM sampler step counts")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of generated images for FID")
    parser.add_argument("--batch", type=int, default=None, help="Max batch size for EDM generate.py and fid.py")
    parser.add_argument("--network", default=None, help="EDM network pkl path or URL")
    parser.add_argument("--fid-ref", default=None, help="EDM FID reference npz path or URL")
    parser.add_argument("--seeds-start", type=int, default=None, help="First seed for generated images")
    parser.add_argument("--nproc-per-node", type=int, default=None, help="torchrun process count")
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing generated images")
    parser.add_argument("--skip-fid", action="store_true", help="Only generate images")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def _as_int_list(values: Iterable[int] | None, fallback: list[int]) -> list[int]:
    if values is None:
        return fallback
    return [int(value) for value in values]


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
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output="".join(captured))
    return "".join(captured)


def _torchrun_command(nproc_per_node: int, script: Path, args: list[str]) -> list[str]:
    if int(nproc_per_node) <= 1:
        return [sys.executable, str(script), *args]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        str(script),
        *args,
    ]


def _parse_fid(stdout: str) -> float | None:
    for line in reversed(stdout.splitlines()):
        match = FID_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def _nfe_for_edm_steps(step_count: int, sampler: str) -> int:
    if sampler == "euler":
        return int(step_count)
    return max(1, 2 * int(step_count) - 1)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    eval_cfg = config.get("eval", {})
    edm_cfg = eval_cfg.get("edm", {})

    steps = _as_int_list(args.steps, [int(item) for item in eval_cfg.get("step_counts", [18])])
    num_samples = int(args.num_samples or eval_cfg.get("num_fid_samples", 50000))
    batch = int(args.batch or eval_cfg.get("fid_batch_size", eval_cfg.get("sample_batch_size", 64)))
    network = str(args.network or edm_cfg.get("network") or os.environ.get("EDM_CIFAR10_NETWORK", ""))
    fid_ref = str(args.fid_ref or edm_cfg.get("fid_ref") or os.environ.get("EDM_CIFAR10_FID_REF", ""))
    seeds_start = int(args.seeds_start if args.seeds_start is not None else edm_cfg.get("seeds_start", 0))
    nproc_per_node = int(args.nproc_per_node or edm_cfg.get("nproc_per_node", os.environ.get("EDM_NPROC_PER_NODE", 1)))
    sampler = str(edm_cfg.get("sampler", "edm"))
    subdirs = bool(edm_cfg.get("generate_subdirs", True))

    if not network:
        raise ValueError("EDM network is required via eval.edm.network, --network, or EDM_CIFAR10_NETWORK")
    if not fid_ref and not args.skip_fid:
        raise ValueError("EDM FID reference is required via eval.edm.fid_ref, --fid-ref, or EDM_CIFAR10_FID_REF")
    if not EDM_ROOT.exists():
        raise FileNotFoundError(f"refs/edm not found: {EDM_ROOT}")

    sample_root = Path(args.sample_root)
    eval_root = Path(args.eval_root)
    report_dir = eval_root / "reports"
    sample_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("DNNLIB_CACHE_DIR", str(Path(env.get("TORCH_HOME", str(ROOT / ".torch"))) / "dnnlib"))
    env["PYTHONPATH"] = str(EDM_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    records: list[dict[str, object]] = []
    for step_count in steps:
        step_sample_dir = sample_root / f"steps{step_count}" / "images"
        step_eval_dir = eval_root / f"steps{step_count}"
        step_eval_dir.mkdir(parents=True, exist_ok=True)
        seed_end = seeds_start + num_samples - 1
        t0 = time.time()

        if not args.skip_generate:
            generate_args = [
                f"--outdir={step_sample_dir}",
                f"--seeds={seeds_start}-{seed_end}",
                f"--batch={batch}",
                f"--steps={step_count}",
                f"--network={network}",
            ]
            if subdirs:
                generate_args.append("--subdirs")
            generate_cmd = _torchrun_command(nproc_per_node, EDM_ROOT / "generate.py", generate_args)
            _run_and_tee(
                generate_cmd,
                cwd=EDM_ROOT,
                env=env,
                log_path=step_eval_dir / "generate.stdout_stderr.txt",
            )

        fid = None
        if not args.skip_fid:
            fid_cmd = _torchrun_command(
                nproc_per_node,
                EDM_ROOT / "fid.py",
                [
                    "calc",
                    f"--images={step_sample_dir}",
                    f"--ref={fid_ref}",
                    f"--num={num_samples}",
                    f"--batch={batch}",
                ],
            )
            fid_stdout = _run_and_tee(
                fid_cmd,
                cwd=EDM_ROOT,
                env=env,
                log_path=step_eval_dir / "fid.stdout_stderr.txt",
            )
            fid = _parse_fid(fid_stdout)

        elapsed = time.time() - t0
        record = {
            "step_count": int(step_count),
            "nfe": _nfe_for_edm_steps(step_count, sampler=sampler),
            "fid": fid,
            "num_fid_samples": num_samples,
            "batch": batch,
            "network": network,
            "fid_ref": fid_ref,
            "sample_dir": str(step_sample_dir),
            "sampler": sampler,
            "nproc_per_node": nproc_per_node,
            "seed_start": seeds_start,
            "seed_end": seed_end,
            "elapsed_sec": elapsed,
            "samples_per_sec": num_samples / max(elapsed, 1.0e-8),
        }
        with (step_eval_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        records.append(record)
        print(
            f"edm eval step_count={step_count} nfe={record['nfe']} fid={fid} "
            f"num_samples={num_samples} elapsed_sec={elapsed:.2f}",
            flush=True,
        )

    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    valid_fid = [item for item in records if item["fid"] is not None]
    if valid_fid:
        best = min(valid_fid, key=lambda item: float(item["fid"]))
        with (report_dir / "best.json").open("w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)
    print("edm cifar10 evaluation completed")
    print(f"sample_root: {sample_root}")
    print(f"eval_root: {eval_root}")


if __name__ == "__main__":
    main()
