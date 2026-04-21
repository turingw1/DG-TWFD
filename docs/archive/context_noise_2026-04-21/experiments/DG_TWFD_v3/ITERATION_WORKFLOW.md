# DGTD v3 Iteration Workflow

This note summarizes the workflow actually used so far on the `DG_TWFD_v3`
branch.

## 1. Working model

The project is currently developed in an iterative loop:

1. define the next technical target
2. inspect the current code and existing experiment evidence
3. modify the code on `DG_TWFD_v3` only
4. write the corresponding patch / audit / pipeline notes
5. commit each completed round
6. run smoke or short experiments on the server
7. return logs, tracebacks, metrics, and file outputs
8. use those results to decide the next patch

This means the branch is advanced through repeated:

- local implementation
- local static/unit validation
- server execution
- evidence-driven correction

## 2. Document roles

The documentation is split by function.

Core design / reconstruction:

- `docs/experiments/DG_TWFD_v3/reconstruction_v3.md`
- `docs/experiments/DG_TWFD_v3/ARCHITECTURE_AND_IMPLEMENTATION.md`

Planning / completion tracking:

- `docs/experiments/DG_TWFD_v3/CHECKLIST.md`

Intermediate validation and server operations:

- `docs/experiments/DG_TWFD_v3/DEVELOPMENT_VALIDATION.md`
- round-specific smoke instruction docs such as
  `docs/dgtd_v3_round*_server_smoke_instructions.md`

Round outputs and technical decisions:

- round audit docs
- round patch notes
- round verification docs

The intended separation is:

- final architecture / pipeline docs describe the stable experiment plan
- validation docs record temporary deployment, smoke, and debugging procedure
- verification docs record what the server actually proved

## 3. Practical execution flow

The actual execution flow used so far is:

1. I read the current code, configs, and prior verification materials.
2. I produce an audit or implementation patch for one focused objective.
3. I add or update the relevant documentation for that round.
4. I run local checks when possible, usually:
   - `py_compile`
   - targeted `pytest`
5. I commit the round immediately on `DG_TWFD_v3`.
6. You pull the branch and run the provided server commands.
7. You send back:
   - stdout/stderr
   - traceback if failed
   - `train.jsonl` tail
   - sample/eval output lists when needed
8. I use that server evidence to:
   - identify the next blocker
   - patch the code
   - update the docs
   - commit again

## 4. Current pattern of experiment advancement

So far, the branch has progressed in this order:

1. baseline reconstruction of DGTD v3 on top of the current dgfm line
2. architecture and loss-path audit
3. symmetric residual / sigma / diagnostics patch
4. online-teacher data-path enablement
5. removal of untracked external runtime dependencies
6. online-teacher continuation audit
7. promotion of online trajectory anchors into the DGTD continuation mainline

This confirms that the project is not being advanced by a single large rewrite.
Instead, it is being advanced through small, reviewable, evidence-backed rounds.

## 5. What counts as a complete round

A round is considered complete only when all of the following exist:

- code changes for the target issue
- a short written explanation of the change
- a commit on `DG_TWFD_v3`
- explicit server instructions when runtime validation is needed
- returned server evidence
- a verification or follow-up judgment based on that evidence

If server evidence is missing, the round is only partially complete even if the
code compiles locally.

## 6. Current collaboration contract

The effective collaboration contract at this point is:

- I handle code inspection, implementation, local validation, and technical
  documentation.
- You handle server-side execution and return the real runtime evidence.
- New patches are based on returned evidence, not only on static reasoning.
- Experimental claims are only upgraded after server verification confirms them.

In short:

- patch locally
- document clearly
- run on server
- return evidence
- refine again

That is the current system-level workflow of this project.
