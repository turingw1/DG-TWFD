# DG-TWFD v3 Documentation Registry

Last updated: 2026-04-26

This registry classifies active documentation by purpose and lifecycle. Do not
use filename dates as the primary source of truth; dated names are evidence
snapshots only.

## Lifecycle Labels

```text
active      read during normal development
reference   read when touching the related subsystem
evidence    read when interpreting a specific experiment/result
superseded  keep for traceability, do not use for current decisions
archive     move to docs/archive when no active/reference/evidence role remains
```

## Current Reading Set

Read these first for current EDM-first work:

| order | role | document |
|---:|---|---|
| 1 | active overview | [README.md](README.md) |
| 2 | active experiment supervision | [EDM_FIRST_SUPERVISION.md](EDM_FIRST_SUPERVISION.md) |
| 3 | server/network/recovery | [NETWORK_AND_RECOVERY.md](NETWORK_AND_RECOVERY.md) |
| 4 | baseline tracking | [BASELINE_STATUS.md](BASELINE_STATUS.md) |
| 5 | baseline interpretation | [BASELINE_COMPARISON_GUIDE.md](BASELINE_COMPARISON_GUIDE.md) |
| 6 | active context summary | [../../ACTIVE_CONTEXT.md](../../ACTIVE_CONTEXT.md) |

Read these only when modifying the older DDPM/DGTD path or comparing against
historical failure cases:

| role | document |
|---|---|
| DDPM route decision evidence | [DDPM_TEACHER_SUITABILITY_2026-04-26.md](DDPM_TEACHER_SUITABILITY_2026-04-26.md) |
| DGTD architecture reference | [ARCHITECTURE_AND_IMPLEMENTATION.md](ARCHITECTURE_AND_IMPLEMENTATION.md) |
| DGTD pipeline commands | [PIPELINE.md](PIPELINE.md) |
| DGTD experiment ledger | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| time-coordinate ablation evidence | [TIME_COORDINATE_DESIGN_ABLATION.md](TIME_COORDINATE_DESIGN_ABLATION.md) |

## Categories

### Status Dashboards

| lifecycle | document | owner/use |
|---|---|---|
| active | [EDM_FIRST_SUPERVISION.md](EDM_FIRST_SUPERVISION.md) | current e504a training/eval supervision, thresholds, commands |
| active | [BASELINE_STATUS.md](BASELINE_STATUS.md) | external baseline queue, blockers, generated CSV status |
| reference | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | legacy DGTD run ledger; update only if DDPM/DGTD route is resumed |

Active supervision scripts:

| lifecycle | path | owner/use |
|---|---|---|
| active | `experiments/edm_first/scripts/hourly_supervise_edm_first.sh` | hourly train/eval supervisor and success/failure action |
| active | `experiments/edm_first/scripts/analyze_hourly_supervision.py` | 7-hour blocker report generator |
| active | `experiments/edm_first/scripts/launch_timewarp_followup.sh` | threshold-triggered timewarp follow-up launcher |

### Operations And Recovery

| lifecycle | document | owner/use |
|---|---|---|
| active | [NETWORK_AND_RECOVERY.md](NETWORK_AND_RECOVERY.md) | proxy, heavy-download policy, `/temp` recovery |
| reference | [A100_SERVER_DEPLOYMENT_2026-04-25.md](A100_SERVER_DEPLOYMENT_2026-04-25.md) | server layout, cache/workspace/temp conventions |

### Algorithm And Implementation

| lifecycle | document | owner/use |
|---|---|---|
| reference | [ARCHITECTURE_AND_IMPLEMENTATION.md](ARCHITECTURE_AND_IMPLEMENTATION.md) | DGTD implementation details, not EDM-first primary route |
| evidence | [DDPM_TEACHER_SUITABILITY_2026-04-26.md](DDPM_TEACHER_SUITABILITY_2026-04-26.md) | why DDPM/discrete teacher is paused |
| evidence | [TIME_COORDINATE_DESIGN_ABLATION.md](TIME_COORDINATE_DESIGN_ABLATION.md) | time-coordinate and OSS-like ablation evidence |

### Experiment Planning

| lifecycle | document | owner/use |
|---|---|---|
| active | [PAPER_EXPERIMENT_TARGETS.md](PAPER_EXPERIMENT_TARGETS.md) | paper-facing target table and baseline scope |
| reference | [PLAN/impreved_to_cinsistent.md](PLAN/impreved_to_cinsistent.md) | historical plan that led to EDM-first; do not treat as live checklist |
| reference | [PIPELINE.md](PIPELINE.md) | old DGTD command families |

### Handoff And Historical State

| lifecycle | document | owner/use |
|---|---|---|
| superseded | [HANDOFF_2026-04-20.md](HANDOFF_2026-04-20.md) | earlier DGTD branch handoff; keep only for lineage |

## Repo Management Rules

- Every active doc must have one primary category in this registry.
- Every new experiment-status doc must record `run tag`, `config`, `checkpoint`,
  `eval root`, `backup root`, and the decision threshold it supports.
- Dated filenames are allowed only for evidence snapshots or handoffs; they are
  not versioning. Current state belongs in status docs and this registry.
- When a doc is no longer used in the current reading set, change its lifecycle
  to `reference`, `evidence`, or `superseded` before archiving it.
- Move docs to `docs/archive/` only after updating this registry and
  `docs/ACTIVE_CONTEXT.md`.
- Keep large result files out of docs. Store only paths, metrics tables, and
  analysis; use `/temp/Zhengwei/DG-TWFD-backups` for durable result evidence.
- Prefer one status document per active experiment track. Do not create new
  versioned status files unless they are immutable evidence snapshots.
