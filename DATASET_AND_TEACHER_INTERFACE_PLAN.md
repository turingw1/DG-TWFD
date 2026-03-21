# Dataset and Teacher Interface Plan

## 1. Dataset System

### Goals
- CIFAR-10 first-class support
- ImageNet32/ImageNet64 ready in the same framework
- future dataset extensibility without trainer rewrite
- consistent data-root conventions across local and A100 environments

## 2. Dataset Module Layout

```text
src/dgfm/datasets/
├── registry.py
├── base.py
├── cifar10.py
├── imagenet.py
├── image_folder.py
├── teacher_shards.py
├── transforms.py
├── splits.py
└── loaders.py
```

## 3. Dataset Conventions

### Data root conventions
- local: `~/workspace/Zhengwei/DG-TWFD/data`
- large-scale / shared: `/cache/Zhengwei/datasets`

### Preprocessing conventions
- store image tensors in `[0,1]` before model-side normalization
- baseline continuous FM converts to `[-1,1]` in the training step
- evaluator uses the same inverse normalization before metric/image export

### Split handling
- CIFAR-10:
  - train: torchvision train split
  - val: either held-out subset from train or test split by config
  - test: torchvision test split
- ImageNet:
  - train/val follow directory layout
  - no hidden split logic in trainer

### Distributed dataloader expectations
- dataset module owns sampler creation
- trainer receives ready-made loaders
- all loaders expose split metadata and dataset size

## 4. Dataset Types Needed
- `cifar10_image`
- `imagenet_image`
- `teacher_shards`
- future `webdataset_image`

## 5. Teacher Interface

### Core API

```python
class Teacher(Protocol):
    def is_enabled(self) -> bool: ...
    def prepare(self, device: torch.device) -> None: ...
    def sample_source(self, batch_shape, device): ...
    def target(self, *, x_t, t, batch, model=None, extras=None): ...
    def state_dict(self) -> dict: ...
```

Phase 1 baseline uses `NullTeacher`.

## 6. Teacher Types

### 6.1 Null Teacher
Use case:
- pure baseline flow matching

Inputs:
- batch data only

Outputs:
- no teacher targets

Integration point:
- trainer sees `teacher.is_enabled() == False`

Checkpointing:
- no teacher weights

Config:
- `teacher.type: none`

### 6.2 Pretrained Teacher
Use case:
- future pretrained FM/diffusion teacher supervision

Inputs:
- `x_t`, `t`, optional conditioning

Outputs:
- target velocity, target `x_s`, or endpoint target depending on method

Integration point:
- optional auxiliary loss branch in trainer

Checkpointing:
- load from explicit external checkpoint path; do not bundle teacher weights into student checkpoint unless requested

Config:
- `teacher.type: pretrained`
- `teacher.name_or_path: ...`
- `teacher.target_type: velocity|endpoint|map`

### 6.3 Sampler-Based Teacher
Use case:
- external diffusion/sampler teacher, including current DDPM teacher logic

Inputs:
- `x_t`, `t`, optional `s`, condition

Outputs:
- teacher-generated target trajectory point or endpoint

Integration point:
- teacher adapter wraps external pipeline/sampler runtime

Checkpointing:
- sampler config + external model path only

Config:
- `teacher.type: sampler`
- `teacher.backend: diffusers_ddpm|custom`

### 6.4 Future Rectified Teacher
Use case:
- future rectified/pre-rectified supervision

Inputs/outputs:
- same as pretrained teacher, but target semantics differ

Requirement:
- no trainer rewrite; only a new teacher adapter

## 7. Where Teacher Plugs into Training
Trainer phases should expose hooks:
- `before_batch`
- `build_training_targets`
- `compute_auxiliary_losses`
- `after_step`

Baseline path:
- teacher hook is a no-op

Teacher path:
- teacher can add targets or losses without changing dataloader/model/solver APIs

## 8. Config Selection Model
Dataset, model, path, teacher, eval, and train strategy should be compositional.

### Example: baseline CIFAR-10 FM

```yaml
# configs/experiment/fm_cifar10_baseline.yaml
base: configs/base.yaml
includes:
  - dataset/cifar10.yaml
  - model/unet_fm.yaml
  - path/condot.yaml
  - scheduler/condot.yaml
  - teacher/none.yaml
  - eval/baseline.yaml
train:
  objective: flow_matching_velocity
```

### Example: baseline ImageNet FM

```yaml
base: configs/base.yaml
includes:
  - dataset/imagenet32.yaml
  - model/unet_fm.yaml
  - path/ot.yaml
  - scheduler/polynomial.yaml
  - teacher/none.yaml
  - eval/baseline.yaml
train:
  objective: flow_matching_velocity
```

### Example: future teacher-based experiment

```yaml
base: configs/base.yaml
includes:
  - dataset/cifar10.yaml
  - model/unet_fm.yaml
  - path/condot.yaml
  - scheduler/timewarp_hook.yaml
  - teacher/pretrained.yaml
  - eval/few_step.yaml
train:
  objective: flow_matching_plus_teacher
teacher:
  name_or_path: /cache/Zhengwei/teachers/some_teacher.pt
  target_type: endpoint
```

## 9. Practical Dataset/Teacher Decisions
- Keep `src/dg_twfd/data/teacher.py` as a migration seed, not the final teacher boundary.
- Keep trajectory shards as an optional dataset type, not the default baseline format.
- Make baseline FM training independent from teacher availability.
