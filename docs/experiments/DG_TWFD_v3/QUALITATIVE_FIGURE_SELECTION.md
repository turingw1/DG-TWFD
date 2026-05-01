# DG-TWFD v3 Qualitative Figure Selection Draft

Last updated: 2026-05-01 Asia/Shanghai

本文档是定性展示图的可编辑配置草案。默认展示步数为 `1 / 2 / 4 / 8`。新版主图采用 class-locked 协议：同一列在所有条件模型中固定同一个类别条件和 latent/noise seed，避免“同 seed 但语义不一致”的定性误差。你可以直接修改 `Include`、`Figure split`、`Step 1/2/4/8`、`Display label` 和 `Notes` 列。

## Display Policy

| Field | Default |
|---|---|
| Main datasets | CIFAR-10, ImageNet64 |
| Main steps | 1, 2, 4, 8 |
| Per-cell visual | class-locked sample strip; currently 8 images per cell |
| Metric label | FID-5k or FID-50k, exactly matching the source report |
| Missing result policy | mark as `N/A`; do not regenerate silently |
| Invalid/smoke result policy | appendix only unless manually promoted |

## Generated Class-Locked Panels

当前已经生成一版无文字 PDF/PNG，图中只包含图片，行列标签留给后续 LaTeX/矢量编辑阶段添加。

| Dataset | Output PDF | Output PNG | Rows | Columns |
|---|---|---|---|---|
| CIFAR-10 | `docs/experiments/DG_TWFD_v3/figures/qualitative/qualitative_cifar10_class_locked_images_only.pdf` | `docs/experiments/DG_TWFD_v3/figures/qualitative/qualitative_cifar10_class_locked_images_only.png` | DG-TWFD best, DG-TWFD identity, EDM teacher, CTM official, CTM no-GAN | NFE 1/2/4/8, each cell has 8 class-locked samples |
| ImageNet64 | `docs/experiments/DG_TWFD_v3/figures/qualitative/qualitative_imagenet64_class_locked_images_only.pdf` | `docs/experiments/DG_TWFD_v3/figures/qualitative/qualitative_imagenet64_class_locked_images_only.png` | EDM, CD-LPIPS, CD-L2, CT | NFE 1/2/4/8, each cell has 8 class-locked samples |

Sidecar metadata:

```text
docs/experiments/DG_TWFD_v3/figures/qualitative/qualitative_images_only_manifest.json
docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/manifest.json
docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/manifest.json
```

CIFAR-10 columns use class ids `0..7` (`airplane, automobile, bird, cat, deer, dog, frog, horse`) with latent seeds `1000..1007`. ImageNet64 columns use class ids `[8, 22, 207, 281, 404, 555, 751, 817]` with deterministic seed `31`.

## New Repository Audit

| Repo | Dataset coverage | Public pretrained sampleable model? | Decision |
|---|---|---|---|
| `openai/consistency_models` | ImageNet64, LSUN | yes: EDM, CD-L2, CD-LPIPS, CT `.pt` checkpoints | already included for ImageNet64. |
| `openai/consistency_models_cifar10` | CIFAR-10 | yes: EDM, CD-L1, CD-L2, CD-LPIPS, CT-LPIPS, continuous CD/CT JAX checkpoints | candidate only for a separate seed-only panel; active env lacks JAX/Flax and these released CIFAR checkpoints are not class-label conditional like our class-locked panel. |
| `pkulwj1994/diff_instruct` | CIFAR-10, ImageNet64 training recipes | no directly released distilled student checkpoint found; README points to EDM teacher checkpoints and training command | exclude until a trained DI checkpoint is available. |
| `neuraloperator/DSNO` | CIFAR-10, ImageNet64 training/eval code | no final DSNO `solver-model_*.pt` checkpoint release found; README provides trajectory data and training recipe | exclude until a trained DSNO checkpoint is available. |

## Editable Final-Style Effect Table

这部分是你主要需要改的表。表格里的 `[image: ...]` 只是最终图片占位文字；真正生成时，每个 cell 会替换成对应模型在该 NFE 下的固定 seed 样本图。默认每个 cell 放同一组 seeds 的小图条，保证横向 NFE 和纵向模型对比都是同一批样本。

### Figure A: CIFAR-10 Main Qualitative Panel

| Sample set | Model | NFE 1 | NFE 2 | NFE 4 | NFE 8 | Keep? | Notes |
|---|---|---|---|---|---|---|---|
| sample A | DG-TWFD best | `[image: A / ours-best / 1]` | `[image: A / ours-best / 2]` | `[image: A / ours-best / 4]` | `[image: A / ours-best / 8]` | yes | 当前用 v17 step7750 auto warp。 |
| sample A | DG-TWFD identity | `[image: A / ours-identity / 1]` | `[image: A / ours-identity / 2]` | `[image: A / ours-identity / 4]` | `[image: A / ours-identity / 8]` | yes | 同 checkpoint，identity eval。 |
| sample A | CTM no-GAN | `[image: A / ctm-nogan / 1]` | `[image: A / ctm-nogan / 2]` | `[image: A / ctm-nogan / 4]` | `[image: A / ctm-nogan / 8]` | yes | CIFAR-10 no-GAN DSM 10k, 50k FID eval。 |
| sample A | EDM | `[image: A / edm / 1]` | `[image: A / edm / 2]` | `[image: A / edm / 4]` | `[image: A / edm / 8]` | yes | EDM teacher / sampler baseline；CIFAR 现有行需要标注 smoke 或重新生成。 |
| sample A | CD-LPIPS | `[image: A / cm-cd-lpips / 1]` | `[image: A / cm-cd-lpips / 2]` | `[image: A / cm-cd-lpips / 4]` | `[image: A / cm-cd-lpips / 8]` | yes-pending | OpenAI CIFAR-10 JAX checkpoint exists; samples need to be generated. |
| sample A | CD-L2 | `[image: A / cm-cd-l2 / 1]` | `[image: A / cm-cd-l2 / 2]` | `[image: A / cm-cd-l2 / 4]` | `[image: A / cm-cd-l2 / 8]` | yes-pending | OpenAI CIFAR-10 JAX checkpoint exists; samples need to be generated. |
| sample A | CT-LPIPS | `[image: A / cm-ct-lpips / 1]` | `[image: A / cm-ct-lpips / 2]` | `[image: A / cm-ct-lpips / 4]` | `[image: A / cm-ct-lpips / 8]` | yes-pending | OpenAI CIFAR-10 JAX checkpoint exists; samples need to be generated. |
| sample B | DG-TWFD best | `[image: B / ours-best / 1]` | `[image: B / ours-best / 2]` | `[image: B / ours-best / 4]` | `[image: B / ours-best / 8]` | optional | 第二组样本，可删。 |
| sample B | DG-TWFD identity | `[image: B / ours-identity / 1]` | `[image: B / ours-identity / 2]` | `[image: B / ours-identity / 4]` | `[image: B / ours-identity / 8]` | optional | 第二组样本，可删。 |
| sample B | CTM no-GAN | `[image: B / ctm-nogan / 1]` | `[image: B / ctm-nogan / 2]` | `[image: B / ctm-nogan / 4]` | `[image: B / ctm-nogan / 8]` | optional | 第二组样本，可删。 |
| sample B | EDM | `[image: B / edm / 1]` | `[image: B / edm / 2]` | `[image: B / edm / 4]` | `[image: B / edm / 8]` | optional | 第二组样本，可删。 |
| sample B | CD-LPIPS | `[image: B / cm-cd-lpips / 1]` | `[image: B / cm-cd-lpips / 2]` | `[image: B / cm-cd-lpips / 4]` | `[image: B / cm-cd-lpips / 8]` | optional-pending | 第二组样本，可删；需要先生成。 |
| sample B | CD-L2 | `[image: B / cm-cd-l2 / 1]` | `[image: B / cm-cd-l2 / 2]` | `[image: B / cm-cd-l2 / 4]` | `[image: B / cm-cd-l2 / 8]` | optional-pending | 第二组样本，可删；需要先生成。 |
| sample B | CT-LPIPS | `[image: B / cm-ct-lpips / 1]` | `[image: B / cm-ct-lpips / 2]` | `[image: B / cm-ct-lpips / 4]` | `[image: B / cm-ct-lpips / 8]` | optional-pending | 第二组样本，可删；需要先生成。 |

### Figure B: ImageNet64 Main Qualitative Panel

| Sample set | Model | NFE 1 | NFE 2 | NFE 4 | NFE 8 | Keep? | Notes |
|---|---|---|---|---|---|---|---|
| sample A | EDM | `[image: A / edm / 1]` | `[image: A / edm / 2]` | `[image: A / edm / 4]` | `[image: A / edm / 8]` | yes | ImageNet64 teacher / sampler baseline。 |
| sample A | CD-LPIPS | `[image: A / cd-lpips / 1]` | `[image: A / cd-lpips / 2]` | `[image: A / cd-lpips / 4]` | `[image: A / cd-lpips / 8]` | yes | Consistency distillation, LPIPS loss。 |
| sample A | CD-L2 | `[image: A / cd-l2 / 1]` | `[image: A / cd-l2 / 2]` | `[image: A / cd-l2 / 4]` | `[image: A / cd-l2 / 8]` | yes | Consistency distillation, L2 loss。 |
| sample A | CT | `[image: A / ct / 1]` | `[image: A / ct / 2]` | `[image: A / ct / 4]` | `[image: A / ct / 8]` | yes | Consistency training baseline。 |
| sample B | EDM | `[image: B / edm / 1]` | `[image: B / edm / 2]` | `[image: B / edm / 4]` | `[image: B / edm / 8]` | optional | 第二组样本，可删。 |
| sample B | CD-LPIPS | `[image: B / cd-lpips / 1]` | `[image: B / cd-lpips / 2]` | `[image: B / cd-lpips / 4]` | `[image: B / cd-lpips / 8]` | optional | 第二组样本，可删。 |
| sample B | CD-L2 | `[image: B / cd-l2 / 1]` | `[image: B / cd-l2 / 2]` | `[image: B / cd-l2 / 4]` | `[image: B / cd-l2 / 8]` | optional | 第二组样本，可删。 |
| sample B | CT | `[image: B / ct / 1]` | `[image: B / ct / 2]` | `[image: B / ct / 4]` | `[image: B / ct / 8]` | optional | 第二组样本，可删。 |

### Compact Alternative: One Row Per Model

如果你觉得按 `sample A / sample B` 分组太长，可以用下面这个压缩版。最终每个 cell 内部放 4 个固定 seed，而不是把 sample 分组写成多段。

| Dataset | Model | NFE 1 | NFE 2 | NFE 4 | NFE 8 | Keep? |
|---|---|---|---|---|---|---|
| CIFAR-10 | DG-TWFD best | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |
| CIFAR-10 | DG-TWFD identity | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |
| CIFAR-10 | CTM no-GAN | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |
| CIFAR-10 | EDM | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |
| CIFAR-10 | CD-LPIPS | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes-pending |
| CIFAR-10 | CD-L2 | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes-pending |
| CIFAR-10 | CT-LPIPS | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes-pending |
| ImageNet64 | EDM | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |
| ImageNet64 | CD-LPIPS | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |
| ImageNet64 | CD-L2 | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |
| ImageNet64 | CT | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | `[4 seeds]` | yes |

## CIFAR-10 Figure Matrix

| Include | Figure split | Block | Display label | Method / run id | Step 1 | Step 2 | Step 4 | Step 8 | Metric budget | Sample source | Metric source | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yes | main | Ours | DG-TWFD v17 auto warp | `edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750` | yes | yes | yes | yes | FID-2048 | `eval/...step7750/steps{1,2,4,8}/fixed_seed_grid.png` | `eval/...step7750/reports/summary.csv` | Current learned-warp eval; step16 exists but excluded by default. |
| yes | main | Ours-control | DG-TWFD v17 identity | `edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750_identity` | yes | yes | yes | yes | FID-2048 | `eval/...step7750_identity/steps{1,2,4,8}/fixed_seed_grid.png` | `eval/...step7750_identity/reports/summary.csv` | Same checkpoint, warp disabled/effective identity. |
| yes | main | Ours-control | DG-TWFD v17 budget warp | `edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750_budget` | yes | yes | yes | yes | FID-2048 | `eval/...step7750_budget/steps{1,2,4,8}/fixed_seed_grid.png` | `eval/...step7750_budget/reports/summary.csv` | Budget rule: step 1 identity, step 2 fixed, >=4 auto. |
| optional | appendix | Ours-legacy | DGTD v3 probe | `dgtd_cifar10_v3_probe_anchor1_long_e407a` | yes | yes | yes | yes | mixed | `eval/dgtd_cifar10_v3_probe_anchor1_long_e407a/steps*/fixed_seed_grid.png` | `eval/dgtd_cifar10_v3_probe_anchor1_long_e407a/reports/summary.csv` | Older diagnostic/probe run; include only if needed. |
| optional | appendix | Teacher | EDM official | `edm_cifar10_public_eval_e501ref` | yes | yes | yes | yes | smoke / check source | `eval/edm_cifar10_public_eval_e501ref/steps*/metrics.json` | `eval/edm_cifar10_public_eval_e501ref/reports/summary.csv` | CIFAR EDM row is not a final full 5k result in current report. |
| yes | main | Official baseline | CTM official cond | `ctm_cifar10_50k` | yes | yes | yes | yes | FID-50k | `runs/baselines_revalidated_20260428/ctm_cifar10_50k/samples/steps*/images` | `eval/baselines_revalidated_20260428/ctm_cifar10_50k/reports/summary.csv` | Main CTM CIFAR baseline. |
| yes | appendix | Diagnostic baseline | CTM no-GAN DSM 10k | `ctm_nogan_20260429/cifar10_ema010000_50k` | yes | yes | yes | yes | FID-50k | `runs/ctm_nogan_20260429/cifar10_ema010000_50k/samples/steps*/images` | `eval/ctm_nogan_20260429/cifar10_ema010000_50k/reports/summary.csv` | Diagnostic no-GAN retrain, not fully converged official baseline. |
| yes-pending | main | Official consistency baseline | OpenAI CM CIFAR CD-LPIPS | `refs/consistency_models_cifar10: cd-lpips checkpoint_80` | yes | yes | yes | yes | pending | pending generation from `https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cd-lpips/checkpoints/checkpoint_80` | pending | Public JAX checkpoint exists; sample via `python -m jcm.main ... --mode eval`. |
| yes-pending | main | Official consistency baseline | OpenAI CM CIFAR CD-L2 | `refs/consistency_models_cifar10: cd-l2 checkpoint_80` | yes | yes | yes | yes | pending | pending generation from `https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cd-l2/checkpoints/checkpoint_80` | pending | Public JAX checkpoint exists; sample via `python -m jcm.main ... --mode eval`. |
| yes-pending | main | Official consistency baseline | OpenAI CM CIFAR CT-LPIPS | `refs/consistency_models_cifar10: ct-lpips checkpoint_74` | yes | yes | yes | yes | pending | pending generation from `https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/ct-lpips/checkpoints/checkpoint_74` | pending | Public JAX checkpoint exists; sample via `python -m jcm.main ... --mode eval`. |
| optional-pending | appendix | Official consistency baseline | OpenAI CM CIFAR CD-L1 | `refs/consistency_models_cifar10: cd-l1 checkpoint_80` | yes | yes | yes | yes | pending | pending generation from `https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cd-l1/checkpoints/checkpoint_80` | pending | Public JAX checkpoint exists; optional because main comparison already keeps CD-L2/CD-LPIPS. |
| yes | main | Official baseline | TCM official | `tcm_cifar10_5k` | yes | yes | yes | yes | FID-5k | `eval/tcm_cifar10_5k/steps*/preview_first64.png` | `eval/tcm_cifar10_5k/reports/summary.csv` | Step 4/8 are engineering extensions under our grid. |
| yes | main | Schedule baseline | Entropic + SDDIM | `entropic_cifar10_5k` | yes | yes | yes | yes | FID-5k | `eval/entropic_cifar10_5k/steps*/preview_first64.png` | `eval/entropic_cifar10_5k/reports/summary.csv` | Local SDDIM path; not official full method reproduction. |
| yes | appendix | EDM schedule-only | EDM + OptimalSteps schedule | `edm_schedule_warp_5k_20260428/optimalsteps_edm` | yes | yes | yes | yes | FID-5k | `eval/edm_schedule_warp_5k_20260428/previews/optimalsteps_edm_steps*.png` | `eval/edm_schedule_warp_5k_20260428/reports/summary.csv` | Schedule-only diagnostic on EDM checkpoint. |
| yes | appendix | EDM schedule-only | EDM + Entropic schedule | `edm_schedule_warp_5k_20260428/entropic_edm` | yes | yes | yes | yes | FID-5k | `eval/edm_schedule_warp_5k_20260428/previews/entropic_edm_steps*.png` | `eval/edm_schedule_warp_5k_20260428/reports/summary.csv` | Entropic schedule transferred to EDM cond-VP checkpoint. |
| yes | appendix | EDM schedule-only | EDM + piecewise-linear warp | `edm_schedule_warp_5k_20260428/piecewise_linear` | yes | yes | yes | yes | FID-5k | `eval/edm_schedule_warp_5k_20260428/previews/piecewise_linear_steps*.png` | `eval/edm_schedule_warp_5k_20260428/reports/summary.csv` | Negative/control schedule-only warp. |
| yes | appendix | EDM schedule-only | EDM + spline warp | `edm_schedule_warp_5k_20260428/spline_warp` | yes | yes | yes | yes | FID-5k | `eval/edm_schedule_warp_5k_20260428/previews/spline_warp_steps*.png` | `eval/edm_schedule_warp_5k_20260428/reports/summary.csv` | Negative/control schedule-only warp. |
| yes | appendix | CTM schedule-only | CTM + OptimalSteps schedule | `ctm_schedule_warp_5k_20260428/cifar10/optimalsteps_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/cifar10/previews/optimalsteps_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/cifar10/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| yes | appendix | CTM schedule-only | CTM + Entropic schedule | `ctm_schedule_warp_5k_20260428/cifar10/entropic_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/cifar10/previews/entropic_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/cifar10/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| yes | appendix | CTM schedule-only | CTM + piecewise-linear warp | `ctm_schedule_warp_5k_20260428/cifar10/piecewise_linear_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/cifar10/previews/piecewise_linear_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/cifar10/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| yes | appendix | CTM schedule-only | CTM + spline warp | `ctm_schedule_warp_5k_20260428/cifar10/spline_warp_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/cifar10/previews/spline_warp_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/cifar10/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| no | excluded | Invalid / infra | OptimalSteps-like failed checkpoint | `schedule_optimalsteps_cifar10.csv` | no | no | no | no | invalid | N/A | `results/baselines/schedule_optimalsteps_cifar10.csv` | Old failed checkpoint infrastructure validation; not a valid qualitative row by default. |

## ImageNet64 Figure Matrix

| Include | Figure split | Block | Display label | Method / run id | Step 1 | Step 2 | Step 4 | Step 8 | Metric budget | Sample source | Metric source | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yes | main | Teacher | EDM official | `edm_imagenet64_public_eval_full` | yes | yes | yes | yes | FID-5k | `runs/edm_imagenet64_public_eval_full/samples/steps*/images` | `eval/edm_imagenet64_public_eval_full/reports/summary.csv` | Teacher/sampler anchor. |
| yes | main | Official baseline | CD-LPIPS official | `cd_imagenet64_lpips_5k` | yes | yes | yes | yes | FID-5k | `runs/cd_imagenet64_lpips_5k/samples/steps*/images` | `eval/cd_imagenet64_lpips_5k/reports/summary.csv` | OpenAI consistency distillation baseline. |
| yes | main | Official baseline | CD-L2 official | `cd_imagenet64_l2_5k` | yes | yes | yes | yes | FID-5k | `runs/cd_imagenet64_l2_5k/samples/steps*/images` | `eval/cd_imagenet64_l2_5k/reports/summary.csv` | OpenAI consistency distillation baseline. |
| yes | main | Official baseline | CT official | `ct_imagenet64_5k` | yes | yes | yes | yes | FID-5k | `runs/ct_imagenet64_5k/samples/steps*/images` | `eval/ct_imagenet64_5k/reports/summary.csv` | OpenAI consistency training baseline. |
| yes | main | Official baseline | CTM official | `ctm_imagenet64_50k` | yes | yes | yes | yes | FID-50k | `runs/baselines_revalidated_20260428*/ctm_imagenet64_50k*/samples/steps*/images` | `eval/baselines_revalidated_20260428*/ctm_imagenet64_50k*/reports/summary.csv` | Step 4/8 come from recovery root. |
| yes | main | Official baseline | TCM official | `tcm_imagenet64_5k` | yes | yes | yes | yes | FID-5k | `eval/tcm_imagenet64_5k/steps*/preview_first64.png` | `eval/tcm_imagenet64_5k/reports/summary.csv` | Step 4/8 are engineering extensions under our grid. |
| yes | main | Schedule baseline | Entropic + SDDIM | `entropic_imagenet64_5k` | yes | yes | yes | yes | FID-5k | `eval/entropic_imagenet64_5k/steps*/preview_first64.png` | `eval/entropic_imagenet64_5k/reports/summary.csv` | Local SDDIM path; not official full method reproduction. |
| yes | appendix | CTM schedule-only | CTM + OptimalSteps schedule | `ctm_schedule_warp_5k_20260428/imagenet64/optimalsteps_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/imagenet64/previews/optimalsteps_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/imagenet64/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| yes | appendix | CTM schedule-only | CTM + Entropic schedule | `ctm_schedule_warp_5k_20260428/imagenet64/entropic_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/imagenet64/previews/entropic_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/imagenet64/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| yes | appendix | CTM schedule-only | CTM + piecewise-linear warp | `ctm_schedule_warp_5k_20260428/imagenet64/piecewise_linear_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/imagenet64/previews/piecewise_linear_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/imagenet64/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| yes | appendix | CTM schedule-only | CTM + spline warp | `ctm_schedule_warp_5k_20260428/imagenet64/spline_warp_ctm` | yes | yes | yes | yes | FID-5k | `eval/ctm_schedule_warp_5k_20260428/imagenet64/previews/spline_warp_ctm_steps*.png` | `eval/ctm_schedule_warp_5k_20260428/imagenet64/reports/summary.csv` | Directly changes CTM exact-transition sigma nodes. |
| no | excluded | Missing | DG-TWFD ImageNet64 | N/A | no | no | no | no | N/A | N/A | N/A | No current ImageNet64 DG-TWFD sample/eval result. |
| no | excluded | Missing | CTM no-GAN ImageNet64 | N/A | no | no | no | no | N/A | N/A | N/A | Training blocked by missing ImageNet64 train data / verified teacher asset. |

## Planned Output Files

| Include | Output | Dataset | Step | Content |
|---|---|---|---:|---|
| yes | `qualitative_cifar10_steps1.png` | CIFAR-10 | 1 | Selected rows from CIFAR-10 matrix. |
| yes | `qualitative_cifar10_steps2.png` | CIFAR-10 | 2 | Selected rows from CIFAR-10 matrix. |
| yes | `qualitative_cifar10_steps4.png` | CIFAR-10 | 4 | Selected rows from CIFAR-10 matrix. |
| yes | `qualitative_cifar10_steps8.png` | CIFAR-10 | 8 | Selected rows from CIFAR-10 matrix. |
| yes | `qualitative_imagenet64_steps1.png` | ImageNet64 | 1 | Selected rows from ImageNet64 matrix. |
| yes | `qualitative_imagenet64_steps2.png` | ImageNet64 | 2 | Selected rows from ImageNet64 matrix. |
| yes | `qualitative_imagenet64_steps4.png` | ImageNet64 | 4 | Selected rows from ImageNet64 matrix. |
| yes | `qualitative_imagenet64_steps8.png` | ImageNet64 | 8 | Selected rows from ImageNet64 matrix. |
| yes | `qualitative_availability_matrix.png` | both | all | Availability + FID summary matrix. |
