from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "public_repos" / "dgfm-map-branch-public"

CONFIG_PATHS = [
    "configs/base.yaml",
    "configs/dataset/cifar10.yaml",
    "configs/dataset/imagenet64.yaml",
    "configs/eval/baseline.yaml",
    "configs/eval/map_branch.yaml",
    "configs/loss/map_ctm_like.yaml",
    "configs/model/map_unet.yaml",
    "configs/model/unet_fm.yaml",
    "configs/path/condot.yaml",
    "configs/path/ot.yaml",
    "configs/scheduler/condot.yaml",
    "configs/scheduler/polynomial.yaml",
    "configs/scheduler/timewarp_hook.yaml",
    "configs/target/teacher_sampler_online.yaml",
    "configs/teacher/none.yaml",
    "configs/teacher/sampler.yaml",
    "configs/teacher/edm_imagenet64.yaml",
    "configs/experiment/fm_cifar10_map_branch.yaml",
    "configs/experiment/fm_cifar10_map_branch_quick.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_smoke_base.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e1_traj_reg.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_teacher.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_ema.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_current.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e2_defect_probe.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e3_pred_residual.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e3_pred_direct.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e4_aux_endpoint_on.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e5_warp_identity.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e5_warp_data_dense.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e5_warp_source_dense.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e5_warp_learned.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e5_warp_spline.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e6_budget_quick.yaml",
    "configs/experiment/fm_cifar10_map_branch_s1_e6_budget_full.yaml",
    "configs/experiment/fm_cifar10_map_branch_s2_official_metrics.yaml",
    "configs/experiment/fm_cifar10_map_branch_s2_defect_eval.yaml",
    "configs/experiment/fm_imagenet64_baseline_smoke.yaml",
]

FILES_TO_COPY = CONFIG_PATHS + [
    "scripts/build_dataset.py",
    "scripts/prepare_imagenet64.py",
    "scripts/run_train.py",
    "scripts/run_eval.py",
    "scripts/run_multistep_panel.py",
    "scripts/run_export_samples_npz.py",
    "scripts/run_evaluate_metrics.py",
    "scripts/run_evaluate_defect.py",
    "scripts/experiments/activate_fm_cifar10.sh",
    "scripts/experiments/create_map_branch_env.sh",
    "src/dgfm",
]

FLOW_MATCHING_RUNTIME_PATHS = [
    "flow_matching/flow_matching",
    "flow_matching/examples/image/models",
]

PUBLIC_DOCS = [
    "docs/experiments/map_branch/A100_PIPELINE.md",
    "docs/experiments/map_branch/EXPERIMENT_LOG.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a minimal standalone public repo for EXPERIMENT_LOG runs")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output directory for the public repo")
    parser.add_argument("--force", action="store_true", help="Remove the output directory first if it exists")
    parser.add_argument("--init-git", action="store_true", help="Initialize git and create an initial commit")
    parser.add_argument("--commit-message", default="Initial public release", help="Initial commit message")
    return parser.parse_args()


def _copy_path(rel_path: str, output_root: Path) -> None:
    src = ROOT / rel_path
    dst = output_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pt", "*.ckpt"))
    else:
        shutil.copy2(src, dst)


def _relative_markdown_links(text: str, *, source_file: Path, output_root: Path) -> str:
    pattern = re.compile(r"\[([^\]]+)\]\((/home/gzwlinux/vscode/gitProject/DG-TWFD/[^)]+)\)")

    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        abs_target = Path(match.group(2))
        repo_rel_target = abs_target.relative_to(ROOT)
        rendered_target = output_root / repo_rel_target
        rel_target = os.path.relpath(rendered_target, start=source_file.parent)
        return f"[{label}]({rel_target})"

    return pattern.sub(replace, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rewrite_public_markdown(text: str) -> str:
    replacements = {
        "cd ~/workspace/Zhengwei/DG-TWFD": "cd <public-repo-root>",
        "git checkout map_branch_ctm_explicit_map\n": "",
        "git pull --ff-only\n": "",
        "current systematic\nexperiment phase": "public experiment phase",
        "current systematic experiment phase": "public experiment phase",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _rewrite_public_activate_script(text: str) -> str:
    replacements = {
        'echo "Example: source scripts/experiments/activate_fm_cifar10.sh baseline v1" >&2':
        'echo "Example: source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a" >&2',
        'variant="${1:-baseline}"': 'variant="${1:-fm_cifar10_map_branch_s1_e6_budget_full}"',
        'tag="${2:-v1}"': 'tag="${2:-dev}"',
        'export PROJ="${PROJ:-$HOME/workspace/Zhengwei/DG-TWFD}"':
        'export PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"',
        'echo "Expected one of: baseline map_branch map_branch_quick map_branch_timewarp_probe map_branch_timewarp_smoke stable or any config stem under configs/experiment/" >&2':
        'echo "Expected any config stem under configs/experiment/ that is listed in docs/experiments/map_branch/EXPERIMENT_LOG.md" >&2',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _rewrite_public_create_env_script(text: str) -> str:
    old = 'echo "  pytest tests/test_dgfm_map_branch.py tests/test_dgfm_teacher_trajectory.py tests/test_dgfm_velocity_model.py tests/test_dgfm_config.py tests/test_dgfm_overrides.py -q"\n'
    new = 'echo "  source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a"\n'
    return text.replace(old, new)


def _write_public_readme(output_root: Path) -> None:
    text = textwrap.dedent(
        """
        # DGFM Map Branch Public

        This repository is the stripped public release of the explicit-map `dgfm`
        experiment system. It keeps only the files needed to run the formal
        experiments recorded in:

        - [docs/experiments/map_branch/EXPERIMENT_LOG.md](docs/experiments/map_branch/EXPERIMENT_LOG.md)

        The stable operational procedure lives in:

        - [docs/experiments/map_branch/A100_PIPELINE.md](docs/experiments/map_branch/A100_PIPELINE.md)

        Public scope:

        - committed experiment configs
        - train / eval / multi-step panel entrypoints
        - official `.npz` sample export and metrics bridge
        - held-out defect evaluator
        - ImageNet64 preprocessing and baseline smoke path

        Omitted from the public release:

        - private development notes and planning docs
        - local artifacts, checkpoints, and one-off diagnostics
        - unused research snapshots not required by `EXPERIMENT_LOG`

        ## Environment

        Two installation paths are included:

        1. Exact project script:
           - [scripts/experiments/create_map_branch_env.sh](scripts/experiments/create_map_branch_env.sh)
        2. Reviewable conda spec:
           - [environment.yml](environment.yml)

        The script is the recommended path for CUDA servers because it installs
        the expected torch/torchvision wheels and project dependencies.

        ## Main code entry points

        Core runtime scripts:

        - train: [scripts/run_train.py](scripts/run_train.py)
        - eval: [scripts/run_eval.py](scripts/run_eval.py)
        - qualitative multi-step panel: [scripts/run_multistep_panel.py](scripts/run_multistep_panel.py)
        - official sample export: [scripts/run_export_samples_npz.py](scripts/run_export_samples_npz.py)
        - official metrics: [scripts/run_evaluate_metrics.py](scripts/run_evaluate_metrics.py)
        - held-out defect: [scripts/run_evaluate_defect.py](scripts/run_evaluate_defect.py)
        - experiment activation: [scripts/experiments/activate_fm_cifar10.sh](scripts/experiments/activate_fm_cifar10.sh)

        Main source modules:

        - config loader: [src/dgfm/config/loader.py](src/dgfm/config/loader.py)
        - map model: [src/dgfm/models/map.py](src/dgfm/models/map.py)
        - map trainer: [src/dgfm/trainers/map.py](src/dgfm/trainers/map.py)
        - target construction: [src/dgfm/targets/builder.py](src/dgfm/targets/builder.py)
        - map sampler: [src/dgfm/samplers/map_sampler.py](src/dgfm/samplers/map_sampler.py)
        - evaluation runner: [src/dgfm/evaluators/runner.py](src/dgfm/evaluators/runner.py)

        ## Quick start

        ```bash
        bash scripts/experiments/create_map_branch_env.sh dgfm_map
        conda activate /cache/Zhengwei/conda_envs/dgfm_map
        source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a
        python scripts/run_train.py --config $FM_CONFIG --run-root $RUN_ROOT --verbose
        ```
        """
    ).strip() + "\n"
    _write_text(output_root / "README.md", text)


def _write_public_environment_yaml(output_root: Path) -> None:
    text = textwrap.dedent(
        """
        name: dgfm_map_public
        channels:
          - defaults
        dependencies:
          - python=3.10
          - pip
          - pip:
              - PyYAML==6.0.3
              - numpy==2.2.3
              - scipy==1.15.3
              - torch-fidelity==0.4.0
              - diffusers>=0.30
              - transformers>=4.40
              - accelerate>=0.30
              - safetensors>=0.4
              - piq>=0.8
              - matplotlib
              - pillow
              - pytest
              - -e .
        """
    ).strip() + "\n"
    _write_text(output_root / "environment.yml", text)


def _write_public_gitignore(output_root: Path) -> None:
    text = textwrap.dedent(
        """
        __pycache__/
        *.py[cod]
        .pytest_cache/
        .mypy_cache/
        .ruff_cache/
        .cache/
        .venv/
        venv/
        env/
        .vscode/
        .idea/
        build/
        dist/
        *.egg-info/
        artifacts/
        checkpoints/
        checkpoints_*/
        outputs/
        logs/
        wandb/
        tensorboard/
        .DS_Store
        """
    ).strip() + "\n"
    _write_text(output_root / ".gitignore", text)


def _write_public_pyproject(output_root: Path) -> None:
    text = textwrap.dedent(
        """
        [build-system]
        requires = ["setuptools>=68", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "dgfm-map-branch-public"
        version = "0.1.0"
        description = "Public minimal release for dgfm explicit-map experiments"
        readme = "README.md"
        requires-python = ">=3.10"
        dependencies = [
          "PyYAML>=6.0",
          "numpy>=2.0",
          "scipy>=1.15",
          "torch>=2.1",
          "torchvision>=0.16",
          "torch-fidelity>=0.4.0",
          "pillow",
          "matplotlib",
        ]

        [project.optional-dependencies]
        teacher = [
          "diffusers>=0.30",
          "transformers>=4.40",
          "accelerate>=0.30",
          "safetensors>=0.4",
          "piq>=0.8",
        ]
        dev = [
          "pytest>=8.0",
        ]

        [tool.setuptools]
        package-dir = {"" = "src"}

        [tool.setuptools.packages.find]
        where = ["src"]
        """
    ).strip() + "\n"
    _write_text(output_root / "pyproject.toml", text)


def _write_minimal_dg_twfd(output_root: Path) -> None:
    _write_text(output_root / "src/dg_twfd/__init__.py", '"""Minimal dg_twfd subset for public dgfm experiments."""\n')
    _write_text(output_root / "src/dg_twfd/models/__init__.py", "from .embeddings import PairTimeConditioner, TimeEmbedding\n__all__ = ['TimeEmbedding', 'PairTimeConditioner']\n")
    src = ROOT / "src/dg_twfd/models/embeddings.py"
    dst = output_root / "src/dg_twfd/models/embeddings.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_public_docs(output_root: Path) -> None:
    for rel_path in PUBLIC_DOCS:
        src = ROOT / rel_path
        dst = output_root / rel_path
        text = src.read_text(encoding="utf-8")
        text = _relative_markdown_links(text, source_file=dst, output_root=output_root)
        text = _rewrite_public_markdown(text)
        _write_text(dst, text)


def _rewrite_public_scripts(output_root: Path) -> None:
    activate_path = output_root / "scripts/experiments/activate_fm_cifar10.sh"
    activate_text = activate_path.read_text(encoding="utf-8")
    _write_text(activate_path, _rewrite_public_activate_script(activate_text))

    create_env_path = output_root / "scripts/experiments/create_map_branch_env.sh"
    create_env_text = create_env_path.read_text(encoding="utf-8")
    _write_text(create_env_path, _rewrite_public_create_env_script(create_env_text))


def _remove_private_docs(output_root: Path) -> None:
    for rel_path in [
        "docs/experiments/map_branch/MASTER_PLAN.md",
        "docs/experiments/map_branch/TIMEWARP_CTM_FINALIZATION_PLAN.md",
        "docs/experiments/map_branch/TECHNICAL_REPORT.md",
        "docs/experiments/map_branch/ACCEPTANCE_CHECKLIST.md",
        "docs/experiments/map_branch/ENVIRONMENT.md",
        "docs/experiments/map_branch/README.md",
        "required_experiments.md",
        "AGENTS.md",
    ]:
        path = output_root / rel_path
        if path.is_file():
            path.unlink()


def _init_git_repo(output_root: Path, commit_message: str) -> None:
    subprocess.run(["git", "init"], cwd=output_root, check=True)
    subprocess.run(["git", "branch", "-m", "main"], cwd=output_root, check=True)
    subprocess.run(["git", "add", "."], cwd=output_root, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=output_root, check=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists; rerun with --force to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for rel_path in FILES_TO_COPY:
        _copy_path(rel_path, output_root)
    for rel_path in FLOW_MATCHING_RUNTIME_PATHS:
        _copy_path(rel_path, output_root)

    _write_public_readme(output_root)
    _write_public_environment_yaml(output_root)
    _write_public_gitignore(output_root)
    _write_public_pyproject(output_root)
    _write_minimal_dg_twfd(output_root)
    _copy_public_docs(output_root)
    _rewrite_public_scripts(output_root)
    _remove_private_docs(output_root)
    _write_text(output_root / "flow_matching/__init__.py", "")

    if args.init_git:
        _init_git_repo(output_root, args.commit_message)

    print("public repo export completed")
    print(f"output_dir: {output_root}")
    print("included: whitelisted configs, runtime scripts, src/dgfm, minimal src/dg_twfd, minimal flow_matching runtime subset, public docs")
    if args.init_git:
        print("git: initialized with initial commit")


if __name__ == "__main__":
    main()
