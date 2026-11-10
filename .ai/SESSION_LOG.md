# AI and team handoff log

This is an append-only record of project context that may help future sessions. Confirm current implementation and Git state before acting on entries.

## 2026-11-10 — Added return-data assumption diagnostics
- Author: Copilot
- Type: implementation
- Context: The return-only baseline models needed a repeatable way to inspect whether prepared returns display properties relevant to their assumptions.
- Outcome: Added `gold_vol_pipeline.diagnostics` with checks for mean-zero evidence, return/squared-return stationarity, serial dependence, ARCH effects, and normality. The tests are exploratory diagnostics; they do not establish assumptions or select a model.
- Artifacts: `src/gold_vol_pipeline/diagnostics.py`, `tests/test_diagnostics.py`, `pyproject.toml`, `README.md`
- Next: Run the report on each analysis dataset and interpret results alongside chronological out-of-sample comparisons.

## 2026-11-10 — Retired superseded branch refs
- Author: Copilot
- Type: handoff
- Context: After the data-pipeline and model-development work was pushed to `main`, the remaining remote branch tips were checked against `origin/main`.
- Outcome: Confirmed each non-main branch tip was an ancestor of `origin/main`, deleted the `data/interactive-pipeline-review`, `feat/model-development`, and `docs/project-guides` remote refs, and removed the local data branch. The remote advertises `main` as HEAD. GitHub API branch and PR listing remained unavailable (404), so PR state or review status is not claimed.
- Artifacts: `main`, `GIT_WORKFLOW.md`, `presentations/project-progress.qmd`
- Next: Work directly on `main`; the progress deck's contribution instructions now fetch `origin main`.

## 2026-11-10 — Consolidated project work on main
- Author: Keir with Copilot
- Type: decision
- Context: Keir directed that all project work be consolidated into `main` and that contributors and AI agents use `main` only.
- Outcome: Merged the Yahoo futures data pipeline and return-only model baselines into `main`; the package retains both `yfinance` and `scipy`. Confirmed the remote's advertised HEAD is `main`. Updated contributor and AI instructions to prohibit topic branches and PRs. GitHub API branch and PR listing returned 404, so PR state or review status could not be verified.
- Artifacts: `src/gold_vol_pipeline/`, `tests/`, `README.md`, `GIT_WORKFLOW.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `llms.txt`
- Next: Continue project changes directly on `main`; do not bypass repository protection if a direct push is rejected.

## 2026-11-10 — Verified end-to-end Yahoo futures CSV download
- Author: Copilot
- Type: finding
- Context: Ran the default pipeline using the Yahoo Finance `GC=F` data endpoint.
- Outcome: The API returned 2,981 daily closes from 2015-01-02 through 2026-11-09. The pipeline wrote `data/raw/gold_futures.csv` and 2,975 prepared rows to `data/processed/gold_futures.csv`; the prepared file includes Yahoo/instrument metadata and a forward five-observation realized-variance label. These data files remain local and Git-ignored.
- Artifacts: `src/gold_vol_pipeline/`, `data/raw/gold_futures.csv`, `data/processed/gold_futures.csv`
- Next: Inspect source quality/calendar and confirm the futures series remains the team's intended instrument before formal comparisons.

## 2026-11-10 — Use Yahoo gold futures as the default downloadable series
- Author: Keir
- Type: decision
- Context: The pipeline needed a runnable datasource API path and the team had not finalized spot versus futures.
- Outcome: Use Yahoo Finance `GC=F` (COMEX gold futures continuous series) as the current default because the spot ticker `XAUUSD=X` was unavailable. The CLI and notebook should download from Yahoo by default and save raw and prepared CSVs. This is a working default, not a claim that futures and spot are interchangeable; keep any future instrument change explicit.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`, `presentations/project-progress.qmd`
- Next: Team may revisit instrument choice; if changed, verify source, instrument and pipeline behavior before using or comparing series.

## 2026-11-10 — Added return-only baseline forecast functions
- Author: Enoch with Copilot
- Type: implementation
- Context: Model-development work requested baseline functions that can accept future gold-return inputs; the project checklist calls for return-only ARCH/GARCH models and simple historical-variance benchmarks.
- Outcome: Added expanding and rolling historical variance, EWMA, AR(1)/MA(1) squared-return models, and ARCH(1)/GARCH(1,1) forecasts. Functions forecast summed variance over a configurable future horizon and require returns ending at the forecast origin. No actual gold data or model-selection result is included.
- Artifacts: `src/gold_vol_pipeline/models.py`, `tests/test_models.py`, `README.md`, `pyproject.toml`
- Next: Confirm the instrument, target horizon, and available data with the team; then build chronological out-of-sample comparisons using the selected series.

## 2026-11-10 — Spot versus futures decision deferred to team meeting
- Author: Keir
- Type: open question
- Context: Keir raised that the team may want to discuss whether the project should forecast spot-gold or gold-futures volatility in a meeting.
- Outcome: Treat the instrument choice as pending team confirmation; the earlier spot selection is not final. No Bloomberg data has been requested or downloaded.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`
- Next: After the meeting, confirm spot or futures and the provider. Request a Bloomberg export only if the team selects spot and no suitable accessible source is available, or if Bloomberg is otherwise chosen.

## 2026-11-10 — Working instrument changed to spot gold
- Author: Keir
- Type: decision
- Context: After reviewing the distinction between spot and futures volatility, Keir selected spot-gold volatility as the working project instrument.
- Outcome: Use one consistent daily gold spot series. Yahoo `XAUUSD=X` was previously checked and returned no data; the existing `GC=F` futures default is not an acceptable substitute. Pipeline ingestion is CSV-based until a suitable spot provider is selected.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`, `llms.txt`, `.github/copilot-instructions.md`
- Next: Keir and the team should select/confirm a spot-price source and verify access, licensing, currency, units, calendar, and available history before data collection.

## 2026-11-06 — Default data series set to COMEX gold futures
- Author: Keir and Copilot
- Type: decision
- Context: Yahoo Finance returned no data for `XAUUSD=X`; Yahoo's `GC=F` returned daily prices.
- Outcome: The default series is Yahoo Finance `GC=F`, a COMEX gold futures continuous series. It is not spot gold or a single fixed-expiry contract. Keep its identity explicit and note potential roll effects; never silently combine it with spot observations.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`, `llms.txt`, `.github/copilot-instructions.md`
- Next: Revisit the series definition if the research question or data availability changes. Preserve source, instrument, currency, units, and dates for any alternate CSV input.

## 2026-11-06 — Initial preparation choices and pipeline behavior
- Author: Keir and Copilot
- Type: decision
- Context: Enoch's data checklist specifies daily actual/realized volatility, while leaving the horizon and exact realized-volatility measure for the team to choose.
- Outcome: The implemented default horizon is five observed price sessions. The default target is forward realized variance (sum of squared close-to-close log returns); forward realized volatility (its square root) is selectable. The future window is the label for each origin, not an available predictor.
- Artifacts: `src/gold_vol_pipeline/pipeline.py`, `tests/test_pipeline.py`, `notebooks/01_inspect_gold_data.ipynb`
- Next: Revisit the target definition if the model-development requirements or research question call for a different measure or horizon.

## 2026-11-06 — Data source and interactive workflow
- Author: Copilot
- Type: implementation
- Context: The Yahoo spot symbol was unavailable, and switching to a futures symbol changes the instrument.
- Outcome: Implemented Yahoo `GC=F` download, explicitly labeled local CSV ingestion, a command-line pipeline, and a notebook for inspecting prices, target alignment, and saved run metadata. Raw/processed data under `data/` are excluded from Git.
- Artifacts: `pyproject.toml`, `src/gold_vol_pipeline/`, `tests/test_pipeline.py`, `notebooks/01_inspect_gold_data.ipynb`, `.gitignore`
- Next: Review source availability and terms for the team's intended use. The current pipeline has not committed downloaded price data.
