# Checkpoint Retention

Last cleanup: 2026-05-01.

This project keeps training logs, eval reports, configs, analysis docs, and
small eval artifacts. Large training checkpoints are retained only when they
are decision nodes, branch points, final audited artifacts, or baseline
checkpoints needed to rerun a reported comparison.

Cleanup script:

```bash
python scripts/server/cleanup_redundant_checkpoints.py --apply
```

The script first copies all retained checkpoints to:

```text
/temp/Zhengwei/projects/DG-TWFD/critical/ckpt/key_nodes/
```

It then removes redundant checkpoint-like files from:

```text
/cache/Zhengwei/DG-TWFD-runtime/runs/
/temp/Zhengwei/projects/DG-TWFD/critical/runs/
/temp/Zhengwei/projects/DG-TWFD/critical/ckpt/
```

It does not delete eval reports, training logs, configs, metrics, summaries, or
analysis documents.

## Main EDM-First Retained Nodes

| run | checkpoint | purpose |
|---|---|---|
| `edm_first_cifar10_warp_e501a` | `best.pt` | first EDM-map learned-warp viability checkpoint |
| `edm_first_cifar10_warp_e502a` | `last.pt` | stopped continuation checkpoint for early EDM-map diagnosis |
| `edm_first_cifar10_onestep_prior_e503c` | `step2000.pt` | prior one-step endpoint reference |
| `edm_first_cifar10_onestep_msdefect_e504a` | `step250.pt` | diagnostic threshold reference |
| `edm_first_cifar10_onestep_msdefect_e504a` | `step1250.pt` | restored endpoint checkpoint before resume |
| `edm_first_cifar10_onestep_msdefect_e504a_resume_from1250` | `step1750.pt` | handoff from endpoint-only training to full-stack v11a |
| `edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750` | `step6750.pt` | best composition checkpoint and v12a branch point |
| `edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750` | `step8750.pt` | endpoint-specialized final evaluated checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v12a_from_step6750` | `step10500.pt` | best endpoint/budget-policy checkpoint and v13 branch point |
| `edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500` | `step3500.pt` | first all-budget-improving diagnostic checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500` | `step4250.pt` | first strong all-budget-improving checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500` | `step7500.pt` | best isolated FID@4 checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500` | `step7750.pt` | best all-around checkpoint and v14 branch point |
| `edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750` | `step10000.pt` | best FID@1/FID@2 checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750` | `step10500.pt` | best FID@4 checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750` | `step10750.pt` | best overall multi-step checkpoint and v15 branch point |
| `edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750` | `step10801.pt` | final-checkpoint guard artifact |
| `edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750` | `step10000.pt` | v15 positive signal checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750` | `step11500.pt` | v15 best balanced checkpoint |
| `edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750` | `step11855.pt` | v15 best high-budget and final evaluated checkpoint |

## Side And Baseline Retained Nodes

| run | checkpoint | purpose |
|---|---|---|
| `fm_cifar10_map_branch_v1` | `best.pt` | early FM map branch best checkpoint |
| `fm_cifar10_map_branch_timewarp_smoke_tws02` | `best.pt` | timewarp smoke checkpoint |
| `fm_cifar10_map_branch_timewarp_smoke_tws03` | `best.pt` | timewarp smoke checkpoint |
| `fm_cifar10_map_branch_timewarp_probe_tw001` | `best.pt` | timewarp probe checkpoint |
| `fm_cifar10_map_branch_quick_v1` | `best.pt` | quick branch best checkpoint |
| `ctm_nogan_20260429/cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000` | `ema_0.999_010000.pt` | evaluated CTM no-GAN diagnostic EMA checkpoint |

## Cleanup Result

The 2026-05-01 cleanup removed 478 redundant checkpoint files, approximately
265 GiB by file size:

- runtime runs: 280 files, about 155 GiB
- temp critical runs: 160 files, about 100 GiB
- temp critical checkpoint duplicates: 38 files, about 10 GiB

Verification after cleanup:

- `python scripts/server/cleanup_redundant_checkpoints.py` reports zero
  remaining redundant checkpoint candidates.
- `/temp/Zhengwei/projects/DG-TWFD/critical/ckpt/key_nodes/` contains 26
  retained checkpoint files.
- v15 training logs, resolved config, and final eval summaries remain present.
