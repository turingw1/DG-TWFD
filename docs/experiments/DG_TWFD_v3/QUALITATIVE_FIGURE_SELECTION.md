# DG-TWFD v3 Qualitative Figure Selection Draft

Last updated: 2026-05-02 Asia/Shanghai

本文档是定性展示图的可编辑配置草案。默认展示步数为 `1 / 2 / 4 / 8`。新版主图对条件模型采用 class-locked 协议：同一列在所有条件模型中固定同一个类别条件和 latent/noise seed，避免“同 seed 但语义不一致”的定性误差。OpenAI CIFAR-10 JAX consistency checkpoints 不支持类别标签条件，因此 CIFAR CD/CT 行只能 seed-locked，不能写成 class-locked。你可以直接修改 `Include`、`Figure split`、`Step 1/2/4/8`、`Display label` 和 `Notes` 列。

## Display Policy

| Field | Default |
|---|---|
| Main datasets | CIFAR-10, ImageNet64 |
| Main steps | 1, 2, 4, 8 |
| Per-cell visual | Paper-ready per-sample panels; each file is one seed/class, model rows x `1/2/4/8` step columns |
| Metric label | FID-5k or FID-50k, exactly matching the source report |
| Missing result policy | mark as `N/A`; do not regenerate silently |
| Invalid/smoke result policy | appendix only unless manually promoted |

## Recommended Paper Panels

当前推荐使用按 seed/class 拆分的 paper-ready 面板，而不是旧的长条总图。每个文件只包含一个 seed/class；面板内部行是模型，列是 `1 / 2 / 4 / 8` display steps。Python 只生成纯图像 grid，不内嵌文字；行名、列名、FID/NFE 等文字应在 LaTeX 或矢量编辑器中排版。

导出规则：

| Rule | Value |
|---|---|
| Sample upscaling | nearest-neighbor to `256 x 256` before tiling |
| Internal gap / border | `0 px` |
| PDF image encoding | lossless `FlateDecode`, no JPEG/DCT |
| Panel width | `1024 px` |
| Recommended max printed width | `3.413 in` for `>=300 dpi` |
| Text labels | not embedded; add in LaTeX |

推荐文件夹：

```text
docs/experiments/DG_TWFD_v3/figures/qualitative/paper_panels_20260502/
├── README.md
├── manifest.json
├── cifar10/
│   ├── pdf_seperatedbyseed/
│   └── png/
└── imagenet64/
    ├── pdf_imnt_sepbyseed/
    └── png/
```

| Dataset | Panels | Panel size | Rows x Columns | Notes |
|---|---:|---|---|---|
| CIFAR-10 | 18 | `1024 x 1792` | 7 models x 4 steps | 18 class/seed pairs; conditional rows are class-locked; JAX CD/CT rows are seed-locked only |
| ImageNet64 | 18 | `1024 x 1536` | 6 models x 4 steps | 18 class/seed pairs; all rows are class-conditional |

Sidecar metadata:

```text
docs/experiments/DG_TWFD_v3/figures/qualitative/paper_panels_20260502/manifest.json
docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260502_paper/manifest.json
docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260502_paper/consistency_cifar10_jax_manifest.json
docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260502_paper/manifest.json
```

CIFAR-10 conditional rows use 18 class/seed pairs: seeds `1000..1017`, with class ids `[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7]`. CIFAR-10 JAX CD/CT rows reuse sample seeds `1000..1017` but have no class labels. ImageNet64 uses 18 class/seed pairs with class ids `[8,22,207,281,404,555,751,817,130,145,292,340,407,444,569,701,779,980]` and seeds `31..48`.

## NeurIPS Main-Text Panel 2026-05-02

这版是按 NeurIPS 主文要求重新生成的紧凑 qualitative figure。目标是展示 `1 / 2 / 4 / 8` 少步采样下的 class preservation 和视觉一致性。样本选择规则是：先按语义类别预选固定 class/seed pairs，再生成所有方法输出；主文每个数据集展示 4 组样本，appendix 保留 8 组完整样本。当前导出的 PDF/PNG 是 image-only 版本，不在图中渲染任何文字；dataset、method、sample、step 标签只保留在 manifest 中，后续由 LaTeX 或矢量编辑器排版。

输出目录：

```text
docs/experiments/DG_TWFD_v3/figures/qualitative/neurips_main_20260502_identity_4_10_18_30/
├── main_text/qualitative_neurips_main.pdf
├── main_text/qualitative_neurips_main.png
├── appendix/cifar10_appendix_all_samples.pdf
├── appendix/imagenet64_appendix_all_samples.pdf
├── appendix/cifar10_seed_only_reference.pdf
├── caption.txt
└── manifest.json
```

样本目录：

```text
docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_neurips_main_20260502/
docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_neurips_main_20260502_identity_4_10_18_30/
```

导出规则：

| Rule | Value |
|---|---|
| Display steps | `1 / 2 / 4 / 8` |
| CIFAR-10 samples | 8 preselected class/seed pairs: airplane, automobile, cat, horse, ship, truck, bird, deer |
| ImageNet64 samples | 8 preselected class/seed pairs: bird, dog, cat-like, vehicle, sports car, clutter, object, bird detail |
| Main paper samples | CIFAR: plane/car/cat/truck; ImageNet64: bird/dog/vehicle/clutter |
| Raster handling | samples are nearest-neighbor upscaled before PDF embedding |
| Text handling | no rendered text in exported PDF/PNG; labels are manifest-only |
| Main figure size | generated as a compact single PDF intended for `<=0.98\linewidth` and `<0.55` page height |

主文行解释和 row notes：

| Dataset | Display row | Row note | Actual generator and parameters |
|---|---|---|---|
| CIFAR-10 | DG-TWFD full | class-locked | 使用 DG-TWFD CIFAR-10 best student checkpoint；learned/full clock；显示列 `1/2/4/8` 对应 student `1/2/4/8` steps。 |
| CIFAR-10 | DG-TWFD identity | class-locked | 使用同一个 DG-TWFD CIFAR-10 best student checkpoint；identity clock；显示列 `1/2/4/8` 对应 student `1/2/4/8` steps。 |
| CIFAR-10 | CTM official CIFAR-10 conditional | class-locked | 使用 CTM CIFAR-10 official conditional checkpoint `model043000.pt`；CTM exact transition；显示列 `1/2/4/8` 对应 CTM `1/2/4/8` steps。 |
| CIFAR-10 | CTM no-GAN DSM 10k | class-locked | 使用本地 no-GAN CTM DSM checkpoint `ema_0.999_010000.pt`；CTM exact transition；显示列 `1/2/4/8` 对应 CTM `1/2/4/8` steps。 |
| CIFAR-10 | CD-LPIPS CIFAR-10 JAX | seed-only | 使用 OpenAI `consistency_models_cifar10` 的 `cd-lpips/checkpoint_80`；公开 JAX checkpoint 无 class label 条件，因此只作为 seed-only reference。 |
| CIFAR-10 | CD-L2 CIFAR-10 JAX | seed-only | 使用 OpenAI `consistency_models_cifar10` 的 `cd-l2/checkpoint_80`；公开 JAX checkpoint 无 class label 条件，因此只作为 seed-only reference。 |
| CIFAR-10 | CT-LPIPS CIFAR-10 JAX | seed-only | 使用 OpenAI `consistency_models_cifar10` 的 `ct-lpips/checkpoint_74`；公开 JAX checkpoint 无 class label 条件，因此只作为 seed-only reference。 |
| ImageNet64 | DG-TWFD full | class-locked proxy | 当前无 ImageNet64 DG-TWFD checkpoint；展示名背后按既定替代规则使用官方 EDM ImageNet64 cond-ADM checkpoint，显示列 `1/2/4/8` 实际对应 EDM `32/48/64/128` steps。 |
| ImageNet64 | DG-TWFD identity | class-locked proxy | 当前无 ImageNet64 DG-TWFD checkpoint；展示名背后按既定替代规则使用官方 EDM ImageNet64 cond-ADM checkpoint，显示列 `1/2/4/8` 实际对应 EDM `4/10/18/30` steps。 |
| ImageNet64 | CD-LPIPS ImageNet64 | class-locked | 使用 OpenAI consistency distillation ImageNet64 LPIPS checkpoint；显示列 `1/2/4/8` 对应 CM `1/2/4/8` steps。 |
| ImageNet64 | CD-L2 ImageNet64 | class-locked | 使用 OpenAI consistency distillation ImageNet64 L2 checkpoint；显示列 `1/2/4/8` 对应 CM `1/2/4/8` steps。 |
| ImageNet64 | CT ImageNet64 | class-locked | 使用 OpenAI consistency training ImageNet64 checkpoint；显示列 `1/2/4/8` 对应 CM `1/2/4/8` steps。 |
| ImageNet64 | CTM ImageNet64 official | class-locked | 使用 CTM ImageNet64 official checkpoint `ctm_imagenet64_ema_0.999.pt`；CTM exact transition；显示列 `1/2/4/8` 对应 CTM `1/2/4/8` steps。 |

重要 caveat：

| Item | Policy |
|---|---|
| ImageNet64 DG-TWFD | 当前 workspace 没有可用 ImageNet64 DG-TWFD checkpoint，因此不能把 ImageNet64 的 EDM proxy 写成 DG-TWFD。 |
| CIFAR-10 OpenAI JAX CD/CT | 这些公开 CIFAR-10 checkpoint 是 unconditional，不支持 class label；当前按你的 row layout 放入 image-only grid，但必须标注为 seed-only reference，不能写成 class-preservation evidence。 |
| Appendix | CIFAR 和 ImageNet appendix 均保留全部 8 个预选 class/seed pairs。 |

建议 caption：

```text
Qualitative samples under 1, 2, 4, and 8 sampling steps test low-step class preservation and visual coherence. Rows use matched class labels and initial noise seeds for class-conditional methods; unconditional CIFAR-10 consistency checkpoints are marked as seed-only references.
```

## Previous Paper Panels Row Log

这部分是当前 PDF 的严格行解释，不是候选模型列表。CIFAR-10 当前有 7 行；ImageNet64 当前有 6 行。每一行都用“展示行名（实际使用的生成模型和参数）”记录，避免把已生成图片和待定候选 baseline 混在一起。

### CIFAR-10 Current Rows

| Row | Display row | Actual generator and parameters |
|---:|---|---|
| 1 | DG-TWFD best replacement / EDM reference | 使用官方 EDM CIFAR-10 cond-VP teacher；checkpoint 来自 DG-TWFD v17 config 的 `paths.network`；显示列 `1/2/4/8` 实际对应 EDM `32/48/64/128` steps；这一行不使用 DG-TWFD student。 |
| 2 | DG-TWFD identity | 使用 DG-TWFD v17 student checkpoint `runs/edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855/checkpoints/best.pt`；warp disabled/effective identity；显示列 `1/2/4/8` 实际对应 student `1/2/4/8` steps。 |
| 3 | CTM official CIFAR-10 conditional | 使用 CTM CIFAR-10 conditional checkpoint `model043000.pt`；CTM exact transition；Karras sigma grid；显示列 `1/2/4/8` 实际对应 CTM `1/2/4/8` steps。 |
| 4 | CTM no-GAN DSM 10k | 使用本地 no-GAN CTM DSM checkpoint `ema_0.999_010000.pt`；CTM exact transition；Karras sigma grid；显示列 `1/2/4/8` 实际对应 CTM `1/2/4/8` steps。 |
| 5 | CD-LPIPS CIFAR-10 JAX | 使用 OpenAI `consistency_models_cifar10` 的 `cd-lpips/checkpoint_80`；官方 JCM stochastic iterative sampler，按 `editing_multistep_sampling.ipynb` 适配；显示列 `1/2/4/8` 实际对应 CM transition `1/2/4/8`；该 checkpoint 无类别标签条件，所以只 seed-locked。 |
| 6 | CD-L2 CIFAR-10 JAX | 使用 OpenAI `consistency_models_cifar10` 的 `cd-l2/checkpoint_80`；官方 JCM stochastic iterative sampler，按 `editing_multistep_sampling.ipynb` 适配；显示列 `1/2/4/8` 实际对应 CM transition `1/2/4/8`；该 checkpoint 无类别标签条件，所以只 seed-locked。 |
| 7 | CT-LPIPS CIFAR-10 JAX | 使用 OpenAI `consistency_models_cifar10` 的 `ct-lpips/checkpoint_74`；官方 JCM stochastic iterative sampler，按 `editing_multistep_sampling.ipynb` 适配；显示列 `1/2/4/8` 实际对应 CM transition `1/2/4/8`；该 checkpoint 无类别标签条件，所以只 seed-locked。 |

### ImageNet64 Current Rows

| Row | Display row | Actual generator and parameters |
|---:|---|---|
| 1 | DG-TWFD best replacement / EDM reference | 使用官方 EDM ImageNet64 class-conditional cond-ADM checkpoint `edm-imagenet-64x64-cond-adm.pkl`；显示列 `1/2/4/8` 实际对应 EDM `32/48/64/128` steps；这一行不使用 DG-TWFD ImageNet checkpoint。 |
| 2 | ImageNet DG-TWFD identity proxy / EDM proxy | 使用官方 EDM ImageNet64 class-conditional cond-ADM checkpoint `edm-imagenet-64x64-cond-adm.pkl`；显示列 `1/2/4/8` 实际对应 EDM `6/8/16/24` steps；由于当前没有 DG-TWFD ImageNet identity checkpoint，这一行是 EDM proxy。 |
| 3 | CD-LPIPS ImageNet64 | 使用 OpenAI consistency distillation ImageNet64 LPIPS checkpoint `cd_imagenet64_lpips.pt`；`karras_sample` onestep/multistep；OpenAI CM ts schedule；显示列 `1/2/4/8` 实际对应 CM `1/2/4/8` steps。 |
| 4 | CD-L2 ImageNet64 | 使用 OpenAI consistency distillation ImageNet64 L2 checkpoint `cd_imagenet64_l2.pt`；`karras_sample` onestep/multistep；OpenAI CM ts schedule；显示列 `1/2/4/8` 实际对应 CM `1/2/4/8` steps。 |
| 5 | CT ImageNet64 | 使用 OpenAI consistency training ImageNet64 checkpoint `ct_imagenet64.pt`；`karras_sample` onestep/multistep；OpenAI CM ts schedule；显示列 `1/2/4/8` 实际对应 CM `1/2/4/8` steps。 |
| 6 | CTM ImageNet64 official | 使用 CTM ImageNet64 checkpoint `ctm_imagenet64_ema_0.999.pt`；CTM exact transition；Karras sigma grid；显示列 `1/2/4/8` 实际对应 CTM `1/2/4/8` steps。 |

## New Repository Audit

| Repo | Dataset coverage | Public pretrained sampleable model? | Decision |
|---|---|---|---|
| `openai/consistency_models` | ImageNet64, LSUN | yes: EDM, CD-L2, CD-LPIPS, CT `.pt` checkpoints | already included for ImageNet64. |
| `openai/consistency_models_cifar10` | CIFAR-10 | yes: EDM, CD-L1, CD-L2, CD-LPIPS, CT-LPIPS, continuous CD/CT JAX checkpoints | CD-LPIPS, CD-L2, and CT-LPIPS are included in the CIFAR panel as seed-locked rows; they are not class-label conditional. |
| `pkulwj1994/diff_instruct` | CIFAR-10, ImageNet64 training recipes | no directly released distilled student checkpoint found; README points to EDM teacher checkpoints and training command | exclude until a trained DI checkpoint is available. |
| `neuraloperator/DSNO` | CIFAR-10, ImageNet64 training/eval code | no final DSNO `solver-model_*.pt` checkpoint release found; README provides trajectory data and training recipe | exclude until a trained DSNO checkpoint is available. |

## Editable Final-Style Effect Table

这部分是你主要需要改的表。表格里的 `[image: ...]` 只是最终图片占位文字；真正生成时，每个 cell 会替换成对应模型在该 NFE 下的固定 seed 样本图。默认每个 cell 放同一组 seeds 的小图条，保证横向 NFE 和纵向模型对比都是同一批样本。

### Figure A: CIFAR-10 Main Qualitative Panel

| Sample set | Model | NFE 1 | NFE 2 | NFE 4 | NFE 8 | Keep? | Notes |
|---|---|---|---|---|---|---|---|
| sample A | EDM 32/48/64/128 | `[image: A / edm-32 / display-1]` | `[image: A / edm-48 / display-2]` | `[image: A / edm-64 / display-4]` | `[image: A / edm-128 / display-8]` | yes | 替换原 DG-TWFD best 行；使用官方 EDM CIFAR-10 cond-VP checkpoint。 |
| sample A | DG-TWFD identity | `[image: A / ours-identity / 1]` | `[image: A / ours-identity / 2]` | `[image: A / ours-identity / 4]` | `[image: A / ours-identity / 8]` | yes | 同 checkpoint，identity eval。 |
| sample A | CTM official | `[image: A / ctm-official / 1]` | `[image: A / ctm-official / 2]` | `[image: A / ctm-official / 4]` | `[image: A / ctm-official / 8]` | yes | CIFAR-10 official conditional CTM。 |
| sample A | CTM no-GAN | `[image: A / ctm-nogan / 1]` | `[image: A / ctm-nogan / 2]` | `[image: A / ctm-nogan / 4]` | `[image: A / ctm-nogan / 8]` | yes | CIFAR-10 no-GAN DSM 10k, 50k FID eval。 |
| sample A | CD-LPIPS | `[image: A / cm-cd-lpips / 1]` | `[image: A / cm-cd-lpips / 2]` | `[image: A / cm-cd-lpips / 4]` | `[image: A / cm-cd-lpips / 8]` | yes | OpenAI CIFAR-10 JAX `cd-lpips/checkpoint_80`; seed-locked only, not class-locked. |
| sample A | CD-L2 | `[image: A / cm-cd-l2 / 1]` | `[image: A / cm-cd-l2 / 2]` | `[image: A / cm-cd-l2 / 4]` | `[image: A / cm-cd-l2 / 8]` | yes | OpenAI CIFAR-10 JAX `cd-l2/checkpoint_80`; seed-locked only, not class-locked. |
| sample A | CT-LPIPS | `[image: A / cm-ct-lpips / 1]` | `[image: A / cm-ct-lpips / 2]` | `[image: A / cm-ct-lpips / 4]` | `[image: A / cm-ct-lpips / 8]` | yes | OpenAI CIFAR-10 JAX `ct-lpips/checkpoint_74`; seed-locked only, not class-locked. |
| sample B | EDM 32/48/64/128 | `[image: B / edm-32 / display-1]` | `[image: B / edm-48 / display-2]` | `[image: B / edm-64 / display-4]` | `[image: B / edm-128 / display-8]` | optional | 第二组样本，可删；当前最新 PDF 只使用 sample A 的 8 类 class-locked strip。 |
| sample B | DG-TWFD identity | `[image: B / ours-identity / 1]` | `[image: B / ours-identity / 2]` | `[image: B / ours-identity / 4]` | `[image: B / ours-identity / 8]` | optional | 第二组样本，可删。 |
| sample B | CTM official | `[image: B / ctm-official / 1]` | `[image: B / ctm-official / 2]` | `[image: B / ctm-official / 4]` | `[image: B / ctm-official / 8]` | optional | 第二组样本，可删。 |
| sample B | CTM no-GAN | `[image: B / ctm-nogan / 1]` | `[image: B / ctm-nogan / 2]` | `[image: B / ctm-nogan / 4]` | `[image: B / ctm-nogan / 8]` | optional | 第二组样本，可删。 |
| sample B | CD-LPIPS | `[image: B / cm-cd-lpips / 1]` | `[image: B / cm-cd-lpips / 2]` | `[image: B / cm-cd-lpips / 4]` | `[image: B / cm-cd-lpips / 8]` | optional | 第二组样本，可删；CIFAR JAX 行只能 seed-locked。 |
| sample B | CD-L2 | `[image: B / cm-cd-l2 / 1]` | `[image: B / cm-cd-l2 / 2]` | `[image: B / cm-cd-l2 / 4]` | `[image: B / cm-cd-l2 / 8]` | optional | 第二组样本，可删；CIFAR JAX 行只能 seed-locked。 |
| sample B | CT-LPIPS | `[image: B / cm-ct-lpips / 1]` | `[image: B / cm-ct-lpips / 2]` | `[image: B / cm-ct-lpips / 4]` | `[image: B / cm-ct-lpips / 8]` | optional | 第二组样本，可删；CIFAR JAX 行只能 seed-locked。 |

### Figure B: ImageNet64 Main Qualitative Panel

| Sample set | Model | NFE 1 | NFE 2 | NFE 4 | NFE 8 | Keep? | Notes |
|---|---|---|---|---|---|---|---|
| sample A | EDM 32/48/64/128 | `[image: A / edm-32 / display-1]` | `[image: A / edm-48 / display-2]` | `[image: A / edm-64 / display-4]` | `[image: A / edm-128 / display-8]` | yes | 替换原 DG-TWFD best 行；使用官方 EDM ImageNet64 cond-ADM checkpoint。 |
| sample A | EDM identity proxy 6/8/16/24 | `[image: A / edm-6 / display-1]` | `[image: A / edm-8 / display-2]` | `[image: A / edm-16 / display-4]` | `[image: A / edm-24 / display-8]` | yes | 按要求作为 ImageNet DG-TWFD identity proxy；当前无 ImageNet DG-TWFD checkpoint。 |
| sample A | CD-LPIPS | `[image: A / cd-lpips / 1]` | `[image: A / cd-lpips / 2]` | `[image: A / cd-lpips / 4]` | `[image: A / cd-lpips / 8]` | yes | Consistency distillation, LPIPS loss。 |
| sample A | CD-L2 | `[image: A / cd-l2 / 1]` | `[image: A / cd-l2 / 2]` | `[image: A / cd-l2 / 4]` | `[image: A / cd-l2 / 8]` | yes | Consistency distillation, L2 loss。 |
| sample A | CT | `[image: A / ct / 1]` | `[image: A / ct / 2]` | `[image: A / ct / 4]` | `[image: A / ct / 8]` | yes | Consistency training baseline。 |
| sample A | CTM official | `[image: A / ctm / 1]` | `[image: A / ctm / 2]` | `[image: A / ctm / 4]` | `[image: A / ctm / 8]` | yes | 已使用 CTM ImageNet64 official checkpoint 和 exact Karras grid 生成，纳入最新 PDF。 |
| sample B | EDM 32/48/64/128 | `[image: B / edm-32 / display-1]` | `[image: B / edm-48 / display-2]` | `[image: B / edm-64 / display-4]` | `[image: B / edm-128 / display-8]` | optional | 第二组样本，可删；当前最新 PDF 只使用 sample A 的 8 类 class-locked strip。 |
| sample B | EDM identity proxy 6/8/16/24 | `[image: B / edm-6 / display-1]` | `[image: B / edm-8 / display-2]` | `[image: B / edm-16 / display-4]` | `[image: B / edm-24 / display-8]` | optional | 第二组样本，可删。 |
| sample B | CD-LPIPS | `[image: B / cd-lpips / 1]` | `[image: B / cd-lpips / 2]` | `[image: B / cd-lpips / 4]` | `[image: B / cd-lpips / 8]` | optional | 第二组样本，可删。 |
| sample B | CD-L2 | `[image: B / cd-l2 / 1]` | `[image: B / cd-l2 / 2]` | `[image: B / cd-l2 / 4]` | `[image: B / cd-l2 / 8]` | optional | 第二组样本，可删。 |
| sample B | CT | `[image: B / ct / 1]` | `[image: B / ct / 2]` | `[image: B / ct / 4]` | `[image: B / ct / 8]` | optional | 第二组样本，可删。 |
| sample B | CTM official | `[image: B / ctm / 1]` | `[image: B / ctm / 2]` | `[image: B / ctm / 4]` | `[image: B / ctm / 8]` | optional | 第二组样本，可删。 |

### Compact Alternative: One Row Per Model

如果你觉得按 `sample A / sample B` 分组太长，可以用下面这个压缩版。当前最新 PDF 每个 cell 内部放 8 个 class-locked samples，而不是把 sample 分组写成多段。

| Dataset | Model | NFE 1 | NFE 2 | NFE 4 | NFE 8 | Keep? |
|---|---|---|---|---|---|---|
| CIFAR-10 | EDM 32/48/64/128 | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| CIFAR-10 | DG-TWFD identity | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| CIFAR-10 | CTM official | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| CIFAR-10 | CTM no-GAN | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| CIFAR-10 | CD-LPIPS | `[8 seed-locked samples]` | `[8 seed-locked samples]` | `[8 seed-locked samples]` | `[8 seed-locked samples]` | yes |
| CIFAR-10 | CD-L2 | `[8 seed-locked samples]` | `[8 seed-locked samples]` | `[8 seed-locked samples]` | `[8 seed-locked samples]` | yes |
| CIFAR-10 | CT-LPIPS | `[8 seed-locked samples]` | `[8 seed-locked samples]` | `[8 seed-locked samples]` | `[8 seed-locked samples]` | yes |
| ImageNet64 | EDM 32/48/64/128 | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| ImageNet64 | EDM identity proxy 6/8/16/24 | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| ImageNet64 | CD-LPIPS | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| ImageNet64 | CD-L2 | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| ImageNet64 | CT | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |
| ImageNet64 | CTM official | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | `[8 class-locked samples]` | yes |

## CIFAR-10 Figure Matrix

| Include | Figure split | Block | Display label | Method / run id | Step 1 | Step 2 | Step 4 | Step 8 | Metric budget | Sample source | Metric source | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yes | main | Teacher reference | EDM CIFAR-10 cond-VP 32/48/64/128 | `class_locked_samples/cifar10_20260501/edm_cifar10_cond_vp_32_48_64_128` | display 1 -> actual 32 | display 2 -> actual 48 | display 4 -> actual 64 | display 8 -> actual 128 | qualitative only | `docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/edm_cifar10_cond_vp_32_48_64_128/steps*` | `qualitative_images_only_manifest.json` | Latest panel row replacing former DG-TWFD best row. |
| no | archived | Ours | DG-TWFD v17 auto warp | `edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750` | no | no | no | no | FID-2048 | `eval/...step7750/steps{1,2,4,8}/fixed_seed_grid.png` | `eval/...step7750/reports/summary.csv` | Superseded in the current qualitative image-only panel by EDM 32/48/64/128, per latest figure request. |
| yes | main | Ours-control | DG-TWFD v17 identity | `edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750_identity` | yes | yes | yes | yes | FID-2048 | `eval/...step7750_identity/steps{1,2,4,8}/fixed_seed_grid.png` | `eval/...step7750_identity/reports/summary.csv` | Same checkpoint, warp disabled/effective identity. |
| yes | main | Ours-control | DG-TWFD v17 budget warp | `edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750_budget` | yes | yes | yes | yes | FID-2048 | `eval/...step7750_budget/steps{1,2,4,8}/fixed_seed_grid.png` | `eval/...step7750_budget/reports/summary.csv` | Budget rule: step 1 identity, step 2 fixed, >=4 auto. |
| optional | appendix | Ours-legacy | DGTD v3 probe | `dgtd_cifar10_v3_probe_anchor1_long_e407a` | yes | yes | yes | yes | mixed | `eval/dgtd_cifar10_v3_probe_anchor1_long_e407a/steps*/fixed_seed_grid.png` | `eval/dgtd_cifar10_v3_probe_anchor1_long_e407a/reports/summary.csv` | Older diagnostic/probe run; include only if needed. |
| no | archived | Teacher | EDM old low-step/smoke | `edm_cifar10_public_eval_e501ref` | no | no | no | no | smoke / check source | `eval/edm_cifar10_public_eval_e501ref/steps*/metrics.json` | `eval/edm_cifar10_public_eval_e501ref/reports/summary.csv` | Replaced in latest qualitative panel by official EDM 32/48/64/128 samples. |
| yes | main | Official baseline | CTM official cond | `ctm_cifar10_50k` | yes | yes | yes | yes | FID-50k | `runs/baselines_revalidated_20260428/ctm_cifar10_50k/samples/steps*/images` | `eval/baselines_revalidated_20260428/ctm_cifar10_50k/reports/summary.csv` | Main CTM CIFAR baseline. |
| yes | appendix | Diagnostic baseline | CTM no-GAN DSM 10k | `ctm_nogan_20260429/cifar10_ema010000_50k` | yes | yes | yes | yes | FID-50k | `runs/ctm_nogan_20260429/cifar10_ema010000_50k/samples/steps*/images` | `eval/ctm_nogan_20260429/cifar10_ema010000_50k/reports/summary.csv` | Diagnostic no-GAN retrain, not fully converged official baseline. |
| yes | main | Official consistency baseline | OpenAI CM CIFAR CD-LPIPS | `refs/consistency_models_cifar10: cd-lpips checkpoint_80` | yes | yes | yes | yes | qualitative only | `docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/cd_lpips_cifar10_jax/steps*` | `consistency_cifar10_jax_manifest.json` | Public JAX checkpoint; generated with dedicated `scripts/figures/generate_cifar10_consistency_jax_qualitative.py`; seed-locked only because no class labels. |
| yes | main | Official consistency baseline | OpenAI CM CIFAR CD-L2 | `refs/consistency_models_cifar10: cd-l2 checkpoint_80` | yes | yes | yes | yes | qualitative only | `docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/cd_l2_cifar10_jax/steps*` | `consistency_cifar10_jax_manifest.json` | Public JAX checkpoint; generated with dedicated `scripts/figures/generate_cifar10_consistency_jax_qualitative.py`; seed-locked only because no class labels. |
| yes | main | Official consistency baseline | OpenAI CM CIFAR CT-LPIPS | `refs/consistency_models_cifar10: ct-lpips checkpoint_74` | yes | yes | yes | yes | qualitative only | `docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501/ct_lpips_cifar10_jax/steps*` | `consistency_cifar10_jax_manifest.json` | Public JAX checkpoint; generated with dedicated `scripts/figures/generate_cifar10_consistency_jax_qualitative.py`; seed-locked only because no class labels. |
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
| yes | main | Teacher reference | EDM ImageNet64 cond-ADM 32/48/64/128 | `class_locked_samples/imagenet64_20260501/edm_imagenet64_cond_adm_32_48_64_128` | display 1 -> actual 32 | display 2 -> actual 48 | display 4 -> actual 64 | display 8 -> actual 128 | qualitative only | `docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501/edm_imagenet64_cond_adm_32_48_64_128/steps*` | `qualitative_images_only_manifest.json` | Latest panel row replacing former DG-TWFD best slot. |
| yes | main | Identity proxy | EDM ImageNet64 6/8/16/24 | `class_locked_samples/imagenet64_20260502_paper/edm_imagenet64_identity_8_16_24_30` | display 1 -> actual 6 | display 2 -> actual 8 | display 4 -> actual 16 | display 8 -> actual 24 | qualitative only | `docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260502_paper/edm_imagenet64_identity_8_16_24_30/steps*` | `manifest.json` | Requested ImageNet identity/proxy row; no ImageNet DG-TWFD checkpoint is currently available. Directory name is retained to replace the current files in-place. |
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
