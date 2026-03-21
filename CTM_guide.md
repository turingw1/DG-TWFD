You are an expert research engineer in diffusion, consistency distillation, trajectory modeling, and generative model systems.

Your task is to deeply inspect the public CTM project and repository:
- Project page: https://consistencytrajectorymodel.github.io/CTM/
- Codebase: https://github.com/sony/ctm

You must understand its technical design in detail and produce a concise, implementation-oriented report saved to `technical_report.md`.

The goal is NOT just to summarize CTM. The goal is to evaluate whether CTM can serve as a base architecture, or a partial technical reference, for my future research pipeline, whose intended extensions include:
1. semigroup-defect-based flow-map training,
2. monotonic time-warp / time reparameterization,
3. optional boundary correction near the high-noise end,
4. step-robust 1/2/4/8/16-step generation,
5. migration into my own cleaner research framework.

Write the report in English.
Be concise, technically precise, and remove redundant explanation.

Required output structure in `technical_report.md`:

# Technical Report: Assessing CTM as a Base Architecture

## 1. Executive Verdict
Give a short verdict:
- Is CTM a good direct base architecture?
- Is it better used as a partial reference rather than a full base?
- What are its main architectural strengths and liabilities?

## 2. Project-Level Technical Claim
Explain CTM’s core claim at the mathematical-object level:
- what CTM is trying to learn,
- how it differs from standard Consistency Models,
- how it differs from score-based diffusion training,
- why it supports arbitrary traversal from time t to time s.

Do this in implementation-relevant language, not just paper-style summary.

## 3. Repository Architecture
Explain the actual repository structure and interaction of components:
- model definition,
- training entrypoints,
- sampling entrypoints,
- command scripts,
- teacher model dependency,
- dataset handling,
- evaluation flow,
- checkpoint flow,
- config / argument handling.

Do not merely list files. Explain how the system is wired.

## 4. Learned Object and Parameterization
Identify exactly what the CTM code is learning:
- score,
- consistency projection,
- trajectory map,
- dual-purpose parameterization,
- or another object.

Clarify:
- model inputs,
- model outputs,
- time parameterization,
- how t and s are represented,
- where the “trajectory model” abstraction actually appears in code.

## 5. Training Mechanics
Explain the exact training flow:
- what targets are constructed,
- how teacher supervision is used,
- whether DSM and consistency-style losses are combined,
- whether GAN training is optional or entangled,
- what parts of the method are essential vs optional.

Distinguish clearly between CTM+DSM and CTM+DSM+GAN.

## 6. Sampling Mechanics
Explain:
- how sampling is implemented,
- how gamma-sampling is represented,
- how deterministic vs stochastic modes are handled,
- whether long jumps are natively supported in code,
- whether the implementation can naturally support unified few-step evaluation.

## 7. Compatibility with My Planned Method
Assess compatibility with the following additions:

### 7.1 Semigroup Defect
Can the code naturally support a loss of the form:
D(t,s,u,x) = ||M(t,u,x) - M(s,u, M(t,s,x))||^2
If yes, explain where.
If no, state exactly what abstraction is missing.

### 7.2 Time-Warp
Can a learnable monotonic reparameterization g_phi(t) be inserted cleanly?
Identify the most appropriate insertion point:
- sampling of time pairs,
- trajectory parameterization,
- model conditioning,
- scheduler,
- or training loop.

### 7.3 Boundary Correction
Could a high-noise-end correction module be added without disrupting the main CTM formulation?
Explain the best insertion point.

### 7.4 Step-Robust Evaluation
Can CTM be cleanly extended to benchmark 1/2/4/8/16-step generation under a single unified interface?
Identify what should be refactored.

## 8. Architectural Friction and Refactor Burden
Assess how difficult it would be to migrate CTM into a cleaner personal research codebase.
Focus on:
- hidden coupling,
- hard-coded assumptions,
- dataset-specific logic,
- command-script dependence,
- poor abstraction boundaries,
- teacher-path assumptions,
- training/sampling entanglement.

## 9. Migration Plan into My Framework
Provide a practical migration roadmap:
- what should be reused directly,
- what should be wrapped,
- what should be rewritten,
- what should be discarded.

Prioritize minimal engineering pain.

## 10. Minimal Refactor Plan
Propose the smallest set of changes needed to convert CTM into a research-friendly base for my future method.
Be specific about:
- modules to isolate,
- interfaces to generalize,
- hard-coded logic to remove,
- time-conditioning APIs to expose,
- sampling APIs to standardize.

## 11. Comparative Judgment: CTM vs flow_matching
Give a short direct comparison:
- which repo is the better engineering base,
- which repo is the better conceptual reference,
- which repo is easier for adding semigroup defect,
- which repo is easier for adding time-warp,
- which repo is safer for long-term research iteration.

## 12. Final Recommendation
End with one of the following recommendations:
- use CTM directly as the base,
- use CTM partially and refactor heavily,
- use CTM only as a conceptual/algorithmic reference,
- or do not use CTM as a base.

Important constraints:
- Be concise and technical.
- Avoid generic explanations.
- Ground every conclusion in actual code structure and behavior.
- Explicitly flag uncertainty when code paths are unclear.
- If useful, include a compact compatibility table.

Before writing the report, make sure to:
1. inspect the repository tree,
2. find the main training script(s),
3. find the main sampling script(s),
4. identify where t and s are represented,
5. identify where new losses can be inserted,
6. identify whether trajectory traversal is an explicit abstraction or only implicit in training logic.

Save the final result to `technical_report.md`.