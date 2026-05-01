#!/usr/bin/env python3
"""Build image-only qualitative PDF panels.

The output PDF contains no text. Rows and columns are documented in the
sidecar manifest so the PDF can be labeled later in a vector editor.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "docs" / "experiments" / "DG_TWFD_v3" / "figures" / "qualitative"
STEPS = [1, 2, 4, 8]
SAMPLE_GAP = 2
CELL_GAP = 8
ROW_GAP = 8
BACKGROUND = (255, 255, 255)


def _tensor_to_image(x: torch.Tensor) -> Image.Image:
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {tuple(x.shape)}")
    if x.shape[0] not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got shape {tuple(x.shape)}")
    x = x.detach().cpu().float()
    if float(x.min()) < -0.01:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr, mode="RGB")


def _load_tensor_samples(pattern: str, step: int) -> list[Image.Image]:
    path = ROOT / pattern.format(step=step)
    tensor = torch.load(path, map_location="cpu")
    sample_ids = list(range(min(4, int(tensor.shape[0]))))
    return [_tensor_to_image(tensor[i]) for i in sample_ids]


def _sample_path(image_root: Path, seed: int) -> Path:
    nested = image_root / f"{seed - seed % 1000:06d}" / f"{seed:06d}.png"
    if nested.exists():
        return nested
    flat = image_root / f"{seed:06d}.png"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"missing sample for seed {seed}: {image_root}")


def _load_png_samples(pattern: str, step: int) -> list[Image.Image]:
    image_root = ROOT / pattern.format(step=step)
    return [Image.open(_sample_path(image_root, seed)).convert("RGB") for seed in [0, 1, 2, 3]]


def _load_flat_png_samples(pattern: str, step: int, sample_ids: list[int]) -> list[Image.Image]:
    image_root = ROOT / pattern.format(step=step)
    images = []
    for sample_id in sample_ids:
        path = image_root / f"{int(sample_id):06d}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        images.append(Image.open(path).convert("RGB"))
    return images


def _make_cell(images: list[Image.Image]) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    width = sum(widths) + SAMPLE_GAP * (len(images) - 1)
    height = max(heights)
    cell = Image.new("RGB", (width, height), BACKGROUND)
    x = 0
    for im in images:
        cell.paste(im, (x, 0))
        x += im.width + SAMPLE_GAP
    return cell


def _compose_panel(rows: list[dict], output_stem: str) -> dict:
    cells_by_row: list[list[Image.Image]] = []
    for row in rows:
        row_cells = []
        for step in STEPS:
            if row["source_type"] == "tensor":
                samples = _load_tensor_samples(row["pattern"], step)
            elif row["source_type"] == "pngdir":
                samples = _load_png_samples(row["pattern"], step)
            elif row["source_type"] == "flat_pngdir":
                samples = _load_flat_png_samples(row["pattern"], step, row["sample_ids"])
            else:
                raise ValueError(f"unknown source_type: {row['source_type']}")
            row_cells.append(_make_cell(samples))
        cells_by_row.append(row_cells)

    col_widths = [max(row[col].width for row in cells_by_row) for col in range(len(STEPS))]
    row_heights = [max(cell.height for cell in row) for row in cells_by_row]
    width = sum(col_widths) + CELL_GAP * (len(STEPS) - 1)
    height = sum(row_heights) + ROW_GAP * (len(rows) - 1)
    panel = Image.new("RGB", (width, height), BACKGROUND)

    y = 0
    for row_idx, row in enumerate(cells_by_row):
        x = 0
        for col_idx, cell in enumerate(row):
            panel.paste(cell, (x, y))
            x += col_widths[col_idx] + CELL_GAP
        y += row_heights[row_idx] + ROW_GAP

    OUTDIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTDIR / f"{output_stem}.png"
    pdf_path = OUTDIR / f"{output_stem}.pdf"
    panel.save(png_path)
    panel.save(pdf_path, "PDF", resolution=72.0)
    return {
        "output_png": str(png_path.relative_to(ROOT)),
        "output_pdf": str(pdf_path.relative_to(ROOT)),
        "pixel_size": [width, height],
        "row_count": len(rows),
        "rows": [row["label"] for row in rows],
        "row_log": [row["row_log"] for row in rows],
        "columns": [f"NFE {step}" for step in STEPS],
        "sample_ids_by_row": {row["label"]: row.get("sample_ids") for row in rows},
    }


def main() -> None:
    cifar_rows = [
        {
            "label": "EDM CIFAR-10 cond-VP 32/48/64/128 / class-locked",
            "row_log": (
                "Row 1: DG-TWFD best replacement "
                "(actual: official EDM CIFAR-10 cond-VP teacher from the DG-TWFD v17 config network; "
                "display columns 1/2/4/8 use actual EDM steps 32/48/64/128; no DG-TWFD student is used in this row)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/edm_cifar10_cond_vp_32_48_64_128/steps{step}",
        },
        {
            "label": "DG-TWFD identity / class-locked same checkpoint",
            "row_log": (
                "Row 2: DG-TWFD identity "
                "(actual: DG-TWFD v17 student checkpoint "
                "runs/edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855/checkpoints/best.pt; "
                "warp disabled/effective identity; display columns 1/2/4/8 use actual student steps 1/2/4/8)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/dg_twfd_identity_same_checkpoint/steps{step}",
        },
        {
            "label": "CTM official CIFAR-10 conditional / class-locked",
            "row_log": (
                "Row 3: CTM official CIFAR-10 conditional "
                "(actual: CTM CIFAR-10 conditional checkpoint model043000.pt; "
                "CTM exact transition with Karras sigma grid; display columns 1/2/4/8 use actual CTM steps 1/2/4/8)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/ctm_official_cond/steps{step}",
        },
        {
            "label": "CTM no-GAN DSM 10k / class-locked",
            "row_log": (
                "Row 4: CTM no-GAN DSM 10k "
                "(actual: local no-GAN CTM DSM checkpoint ema_0.999_010000.pt from "
                "cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000; CTM exact transition with Karras sigma grid; "
                "display columns 1/2/4/8 use actual CTM steps 1/2/4/8)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/ctm_nogan_dsm_10k/steps{step}",
        },
        {
            "label": "CD-LPIPS CIFAR-10 JAX / seed-locked",
            "row_log": (
                "Row 5: CD-LPIPS CIFAR-10 JAX "
                "(actual: OpenAI consistency_models_cifar10 cd-lpips/checkpoint_80; "
                "official JCM stochastic iterative sampler adapted from editing_multistep_sampling.ipynb; "
                "display columns 1/2/4/8 use CM transition counts 1/2/4/8. "
                "The released CIFAR JAX checkpoint is not class-label conditional, so this row is seed-locked only)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/cd_lpips_cifar10_jax/steps{step}",
        },
        {
            "label": "CD-L2 CIFAR-10 JAX / seed-locked",
            "row_log": (
                "Row 6: CD-L2 CIFAR-10 JAX "
                "(actual: OpenAI consistency_models_cifar10 cd-l2/checkpoint_80; "
                "official JCM stochastic iterative sampler adapted from editing_multistep_sampling.ipynb; "
                "display columns 1/2/4/8 use CM transition counts 1/2/4/8. "
                "The released CIFAR JAX checkpoint is not class-label conditional, so this row is seed-locked only)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/cd_l2_cifar10_jax/steps{step}",
        },
        {
            "label": "CT-LPIPS CIFAR-10 JAX / seed-locked",
            "row_log": (
                "Row 7: CT-LPIPS CIFAR-10 JAX "
                "(actual: OpenAI consistency_models_cifar10 ct-lpips/checkpoint_74; "
                "official JCM stochastic iterative sampler adapted from editing_multistep_sampling.ipynb; "
                "display columns 1/2/4/8 use CM transition counts 1/2/4/8. "
                "The released CIFAR JAX checkpoint is not class-label conditional, so this row is seed-locked only)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/ct_lpips_cifar10_jax/steps{step}",
        },
    ]
    imagenet_rows = [
        {
            "label": "EDM ImageNet64 cond-ADM 32/48/64/128 / class-locked",
            "row_log": (
                "Row 1: DG-TWFD best replacement "
                "(actual: official EDM ImageNet64 class-conditional cond-ADM checkpoint "
                "edm-imagenet-64x64-cond-adm.pkl; display columns 1/2/4/8 use actual EDM steps 32/48/64/128; "
                "no DG-TWFD ImageNet checkpoint is used)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": list(range(8)),
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/edm_imagenet64_cond_adm_32_48_64_128/steps{step}",
        },
        {
            "label": "EDM ImageNet64 identity proxy 16/24/30/36 / class-locked",
            "row_log": (
                "Row 2: ImageNet DG-TWFD identity proxy "
                "(actual: official EDM ImageNet64 class-conditional cond-ADM checkpoint "
                "edm-imagenet-64x64-cond-adm.pkl; display columns 1/2/4/8 use actual EDM steps 16/24/30/36; "
                "this is an EDM proxy because no DG-TWFD ImageNet identity checkpoint is available)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": list(range(8)),
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/edm_imagenet64_identity_16_24_30_36/steps{step}",
        },
        {
            "label": "CD-LPIPS ImageNet64 / class-locked",
            "row_log": (
                "Row 3: CD-LPIPS ImageNet64 "
                "(actual: OpenAI consistency distillation ImageNet64 LPIPS checkpoint cd_imagenet64_lpips.pt; "
                "karras_sample onestep/multistep with OpenAI CM ts schedule; display columns 1/2/4/8 use CM step counts 1/2/4/8)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": list(range(8)),
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/cd_lpips_imagenet64/steps{step}",
        },
        {
            "label": "CD-L2 ImageNet64 / class-locked",
            "row_log": (
                "Row 4: CD-L2 ImageNet64 "
                "(actual: OpenAI consistency distillation ImageNet64 L2 checkpoint cd_imagenet64_l2.pt; "
                "karras_sample onestep/multistep with OpenAI CM ts schedule; display columns 1/2/4/8 use CM step counts 1/2/4/8)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": list(range(8)),
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/cd_l2_imagenet64/steps{step}",
        },
        {
            "label": "CT ImageNet64 / class-locked",
            "row_log": (
                "Row 5: CT ImageNet64 "
                "(actual: OpenAI consistency training ImageNet64 checkpoint ct_imagenet64.pt; "
                "karras_sample onestep/multistep with OpenAI CM ts schedule; display columns 1/2/4/8 use CM step counts 1/2/4/8)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": list(range(8)),
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/ct_imagenet64/steps{step}",
        },
        {
            "label": "CTM ImageNet64 official / class-locked",
            "row_log": (
                "Row 6: CTM ImageNet64 official "
                "(actual: CTM ImageNet64 checkpoint ctm_imagenet64_ema_0.999.pt; "
                "CTM exact transition with Karras sigma grid; display columns 1/2/4/8 use actual CTM steps 1/2/4/8)."
            ),
            "source_type": "flat_pngdir",
            "sample_ids": list(range(8)),
            "pattern": "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/ctm_imagenet64_official/steps{step}",
        },
    ]

    cifar_manifest = json.loads(
        (
            OUTDIR
            / "class_locked_samples"
            / "cifar10_20260501"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    consistency_cifar_manifest_path = (
        OUTDIR
        / "class_locked_samples"
        / "cifar10_20260501"
        / "consistency_cifar10_jax_manifest.json"
    )
    consistency_cifar_manifest = (
        json.loads(consistency_cifar_manifest_path.read_text(encoding="utf-8"))
        if consistency_cifar_manifest_path.exists()
        else None
    )
    imagenet_manifest = json.loads(
        (
            OUTDIR
            / "class_locked_samples"
            / "imagenet64_20260501"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest = {
        "note": (
            "PDF panels contain images only; labels are intentionally absent. "
            "Rows, NFE columns, class ids, sample ids, and seed-locking caveats are documented here."
        ),
        "omitted": [],
        "class_locking": {
            "cifar10": cifar_manifest["columns"],
            "imagenet64": imagenet_manifest["columns"],
        },
        "seed_locked_only": {
            "cifar10_consistency_jax": (
                consistency_cifar_manifest.get("sample_seeds", [])
                if consistency_cifar_manifest is not None
                else []
            ),
        },
        "step_mapping": {
            "cifar10": cifar_manifest.get("step_mapping", {}),
            "imagenet64": imagenet_manifest.get("step_mapping", {}),
        },
        "panels": [
            _compose_panel(cifar_rows, "qualitative_cifar10_class_locked_images_only"),
            _compose_panel(imagenet_rows, "qualitative_imagenet64_class_locked_images_only"),
        ],
    }
    manifest_path = OUTDIR / "qualitative_images_only_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
