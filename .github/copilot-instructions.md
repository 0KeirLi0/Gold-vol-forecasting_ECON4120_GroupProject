# Copilot instructions

## Project and workflow

This repository supports an Applied Forecasting Methods group project on gold-volatility forecasting. Work proceeds iteratively:

1. **Infrastructure** — acquire and prepare data, maintain the repository, and prepare presentation materials (Keir).
2. **Model development** — implement forecasting approaches and investigate relevant variables (Enoch).
3. **Model analysis and decision** — compare candidate models, including information-criterion selection where appropriate, explain limitations, and recommend the next step (Samuel).
4. Feed that decision into the next iteration.

Treat these as areas of focus, not exclusive ownership: coordinate changes that cross stages and keep outputs usable by the next stage.
For the initial data pipeline, wait until model development has identified the data needed; then coordinate the pipeline design and implementation with Keir. Avoid locking in data sources or schemas before those requirements are known.

Early iterations may explore methods from the course, for example AR, MA, ARMA, ARCH, GARCH, or EGARCH. Later work may investigate methods such as quantile regression or shock analysis when project questions and findings motivate them. These are examples, not a prescribed or exhaustive list; do not assume any method is required or suitable merely because it is named here.

## Collaboration and project state

- Make each iteration reproducible: preserve the data inputs and transformations, model specification, evaluation evidence, and the reasoning behind the next-step decision in the project’s existing artifacts.
- When comparing models, explain the criteria and evidence used, their limitations, and any unresolved issues; do not prescribe a model-selection method on behalf of the team.
- Keep shared interfaces and artifacts clear between data preparation, model development, and model analysis. If a change affects another stage, identify the affected inputs or outputs in the change.
- Update the project’s existing status or decision-tracking artifact when one is present. If none exists, ask Keir before introducing a new source of truth for project status.
- Do not invent project decisions, data details, or results. Ask the team when missing information would change an implementation or interpretation.
- Follow [`GIT_WORKFLOW.md`](../GIT_WORKFLOW.md): work on focused topic branches, open PRs for shared changes, and merge only after review and applicable validation.

## Repository commands and structure

The repository currently contains project documentation only; implementation files and manifests have not yet been established. No build, test, or lint commands, application architecture, or code conventions are known. Inspect the project files and documentation as they are added before choosing commands or extending the architecture; do not assume a language or toolchain.
