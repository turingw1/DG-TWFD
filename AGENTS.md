# Repository Guidelines

## Project Structure & Module Organization
This repository is documentation-first. The root contains five phase briefs: `codex_p1.md` through `codex_p5.md`. Each file defines a sequential implementation stage for the DG-TWFD project, from initialization and data pipelines to inference and final documentation. Keep new contributor-facing material in the repository root unless a clear subdirectory structure emerges.

When extending the prompt set, preserve the existing phase-based naming pattern such as `codex_p6.md`. If you add support files, group them by purpose, for example `examples/`, `references/`, or `templates/`.

## Build, Test, and Development Commands
There is no build system or runnable application in the current tree. Development is limited to editing and reviewing Markdown files. Use the `consistency` conda environment before running repository commands.

- `conda activate consistency` selects the required local environment.
- `ls -1` lists the active phase documents.
- `sed -n '1,120p' codex_p3.md` previews a prompt without opening an editor.
- `wc -w codex_p*.md` checks document length and helps keep prompts scoped.
- `markdownlint AGENTS.md codex_p*.md` is recommended if `markdownlint` is installed locally.

## Coding Style & Naming Conventions
Use Markdown with concise sections, ordered headings, and task-oriented bullet lists. Keep filenames lowercase with underscores and phase suffixes, matching the current `codex_pN.md` convention. Prefer direct instructional language, explicit file paths, and inline code formatting for commands, symbols, and required outputs.

Wrap lines consistently, avoid mixed Chinese and English terminology unless the document already requires it, and keep each phase self-contained so reviewers can read files independently.

## Testing Guidelines
There is no automated test suite in this repository. Validation is editorial:

- confirm headings render correctly,
- verify commands and file paths are plausible,
- check that each phase stops at the requested boundary,
- compare cross-phase terminology so symbols and filenames stay consistent.

Before submitting changes, re-read the full modified document once for clarity and once for instruction fidelity.
If you run any local checks, do so from `conda activate consistency`.

## Commit & Pull Request Guidelines
Git history is not available from this directory, so no local commit convention can be inferred. Use short imperative commit subjects such as `Add contributor guide for phase prompts`.

Pull requests should include a brief summary, list of affected prompt files, and note any changes to phase sequencing, filenames, or required outputs. Include rendered screenshots only if formatting changes are substantial.
