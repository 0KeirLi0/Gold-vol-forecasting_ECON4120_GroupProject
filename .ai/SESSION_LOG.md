# AI and team handoff log

This is an append-only record of project context that may help future sessions. Confirm current implementation and Git state before acting on entries.

## 2026-11-11 — Added optional LSEG Workspace export helper
- Author: Keir's request; implementation by Copilot
- Type: implementation
- Context: Keir requested one Python downloader combining his LSEG gold futures, spot/fixing, and implied-volatility examples with the project's selected Stage 2 daily predictor candidates.
- Outcome: Added a standalone script that saves per-RIC CSVs, a combined workbook, and a manifest. Candidate Stage 2 RIC/field mappings are explicitly provisional and need validation in Workspace; exports preserve series separately and do not claim point-in-time release availability. The existing Yahoo/FRED panel and working futures-series decision are unchanged.
- Artifacts: `scripts/download_lseg_data.py`, `README.md`
- Next: Keir to run on the Workspace computer, verify RICs/fields/units/history and data-use terms, and coordinate any later conversion into the model panel with Enoch.

## 2026-11-11 — Added exploratory multivariate model notebook
- Author: User direction; implementation by Copilot
- Type: finding
- Context: The Stage 2 model request asked to include all candidate predictors and to assess predictor significance before modeling. After clarification, the requested significance target was predictor associations, for which a paired t-test is not appropriate.
- Outcome: On the first 80% of 2,976 complete observations (2015-01-05 through 2024-06-21), OLS of log forward realized variance on seven standardized predictors with HAC lags=4 and Holm correction found VIX significant (adjusted p=0.000308); other predictors were not. On the final 596 origins, HistGradientBoosting RMSE was 0.001883, versus 0.001833 for the expanding-variance benchmark and 0.001855 for the training-mean benchmark. The notebook runs end to end; all project tests pass (50 passed). Results are exploratory and do not imply a final model choice.
- Artifacts: `notebooks/05_advanced_models.ipynb`, `pyproject.toml`, `README.md`, `presentations/project-progress.qmd`
- Next: Samuel to review the predictor screen and comparable out-of-sample evidence; Enoch to review data timing and candidate inputs before any model-selection recommendation.

## 2026-11-11 — Executed Stage 2 predictor notebook without a key
- Author: Copilot
- Type: finding
- Context: The public FRED CSV downloader replaced the API-key-gated requests in the Stage 2 workflow.
- Outcome: Executed `notebooks/04_predictors.ipynb`; it downloaded FRED and Yahoo data and produced 2,976 forecast rows (2015-01-05 through 2026-11-03). All six selected predictor values are non-missing across those rows. The focused predictor tests pass. Execution also exposed and fixed incompatible pandas datetime units at the gold/predictor join. Existing yfinance deprecation warnings remain.
- Artifacts: `notebooks/04_predictors.ipynb`, `data/processed/stage2_multivariate.csv`, `data/processed/stage2_multivariate_manifest.json`, `src/gold_vol_pipeline/predictors.py`, `tests/test_predictors.py`
- Next: Enoch to review source freshness and timing assumptions before model fitting.

## 2026-11-11 — Use public FRED CSV downloads
- Author: User direction; implementation by Copilot
- Type: implementation
- Context: The Stage 2 notebook should not depend on `FRED_API_KEY`; use the downloadable series data from FRED's website instead.
- Outcome: Replaced keyed API observation requests with the public FRED graph CSV endpoint using explicit inclusive date bounds, and removed the API-key gate from the downloader and notebook. The existing one-weekday availability proxy and current-vintage limitation are unchanged.
- Artifacts: `src/gold_vol_pipeline/predictors.py`, `tests/test_predictors.py`, `notebooks/04_predictors.ipynb`, `README.md`
- Next: Run the focused tests and execute the notebook download; inspect data freshness and missingness before modeling.

## 2026-11-11 — Added Stage 2 predictor download notebook
- Author: Copilot
- Type: implementation
- Context: The Stage 2 predictor panel needed a notebook entry point for running the existing acquisition and alignment CLI.
- Outcome: Added `notebooks/04_predictors.ipynb` to invoke `python -m gold_vol_pipeline.predictors`, preview the generated panel and manifest, and stop before network requests when `FRED_API_KEY` is absent. The key is read from the Jupyter kernel environment and is never embedded in the notebook. The notebook and code-cell syntax were validated, but the API download could not be run because this environment has no configured FRED key.
- Artifacts: `notebooks/04_predictors.ipynb`, `README.md`
- Next: Set `FRED_API_KEY` before starting Jupyter, run the notebook, then review source-date freshness and missingness before model fitting.

## 2026-11-11 — Validated and refreshed Stage 2 handoff
- Author: Copilot
- Type: finding
- Context: The Stage 2 predictor panel was implemented while the initial model report still described predictors and pipeline work as pending.
- Outcome: Updated the report and progress tracker to reflect the selected candidate panel, source mapping, and timing limitations. Rendered both presentations successfully and ran the full test suite: 50 passed. Tests emitted existing statistical-library/deprecation warnings; the new FRED date-lag warning was removed.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`, `presentations/project-progress.qmd`, `src/gold_vol_pipeline/`
- Next: Enoch to review predictor transformations, source-date freshness, and missing-data handling before fitting multivariate candidates.

## 2026-11-11 — Added Stage 2 daily predictor pipeline
- Author: Keir's requirements; implementation by Copilot
- Type: implementation
- Context: Enoch identified USD strength, rates, equity, inflation/oil, optional gold volume, and conditional central-bank demand as Stage 2 candidate data. Keir selected futures, DXY, nominal yields, both equity returns and volatility, both inflation expectations and oil, and asked for API support with CSV input where suitable.
- Outcome: Added an API-backed daily panel for Yahoo `DX-Y.NYB`, `^GSPC`, `^VIX`, and optional `GC=F` volume, plus FRED `DGS10`, `T10YIE`, and `DCOILWTICO`. S&P 500 close-to-close log returns are derived. Yahoo values align on same-day availability after market close; FRED values use a one-business-day availability proxy, not historical release timestamps or vintages, with observation dates retained and only past-available values joined to gold origins. Central-bank demand is deferred until a source and release schedule can support point-in-time alignment. FRED requires a local API key. Raw observations remain Git-ignored; no data are committed because source-specific redistribution rights must be checked, regardless of file size. Targeted pytest execution was denied by the current environment, so test status is unverified.
- Artifacts: `src/gold_vol_pipeline/predictors.py`, `src/gold_vol_pipeline/pipeline.py`, `tests/test_predictors.py`, `tests/test_pipeline.py`, `README.md`, `presentations/project-progress.qmd`
- Next: Run the targeted tests when command execution is available; Enoch to review predictor transformations, missingness/freshness, and model inputs; Keir to check volume coverage and central-bank-demand publication timing before extending the panel.

## 2026-11-11 — Framed next stage as multivariate modeling
- Author: User direction, recorded by Copilot
- Type: decision
- Context: The project clarified that the current stage applies traditional time-series models to one variable, with the next stage intended to use multiple variables.
- Outcome: Reframed the report and project tracker as Stage 1 univariate modeling followed by planned multivariate development. No additional predictors or multivariate method were selected; variable rationale, source, timing, and availability at the forecast origin remain prerequisites.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`, `presentations/project-progress.qmd`, `README.md`
- Next: Enoch identifies candidate variables and timing; Keir coordinates pipeline changes after those requirements are defined; Samuel compares Stage 2 evidence with Stage 1 baselines.

## 2026-11-11 — Added current-project map to model report
- Author: Copilot
- Type: implementation
- Context: The model-development report needed a visual map of current project status and handoffs.
- Outcome: Added a Metropolis-style workflow map showing open model/data requirements, the pipeline prototype, current model-development work, and the next analysis handoff, with the team decision feeding the next iteration.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`
- Next: Follow the map's requirements-to-pipeline-to-analysis handoffs; no instrument or model decision is implied by the diagram.

## 2026-11-10 — Reported initial model-development results
- Author: Copilot
- Type: finding
- Context: The first-iteration deck was revised to report quantitative model evidence instead of Git contributions.
- Outcome: On the local prepared Yahoo `GC=F` snapshot (2,975 origins through 2026-11-02), GARCH(1,1) had the lowest point-estimate errors over 496 completed recent forecast origins (MAE 0.000776, RMSE 0.001879); its RMSE was 6.1% below expanding variance and 0.8% below EWMA. EGARCH(2,1) had the lowest in-sample AIC/BIC among 21 return-series candidates, but 4 candidates did not converge; the current EGARCH comparison leaves asymmetry order at zero. EGARCH residual Ljung-Box p-value was 0.020 at lag 20. A current squared-return HK rerun selected ARIMA(3,1,2), conflicting with the earlier logged ARIMA(0,1,4); neither should be relied on until the run/input discrepancy is reconciled. These are exploratory futures-series results, not a model or instrument decision.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`, `presentations/project-progress.qmd`
- Next: Enoch and Samuel should standardize the data snapshot and rolling-origin design, investigate non-convergence/specification, and include automatic order selection inside training windows before comparing forecasts.

## 2026-11-10 — Prepared first-iteration progress handoff
- Author: Copilot
- Type: handoff
- Context: The project needed a concise presentation to close the initial implementation phase and identify the next stage.
- Outcome: Created a Beamer PDF and reproducible Quarto source covering implemented pipeline/model tools, the exploratory status and limitations of current evidence, open instrument/target/evaluation choices, and owner-specific next actions. Clarified in the progress tracker that the available futures and CSV ingestion paths do not settle the team instrument decision.
- Artifacts: `presentations/iteration-1-summary.pdf`, `presentations/iteration-1-summary.qmd`, `presentations/project-progress.qmd`, `README.md`
- Next: Model development should specify data requirements; the team should confirm the research choices before the next pipeline design and out-of-sample evaluation.

## 2026-11-10 — Fixed notebook figure display
- Author: Copilot
- Type: finding
- Context: Notebook plots were not rendering because importing `gold_vol_pipeline` imported `models.py`, which globally switched Matplotlib to the non-interactive `Agg` backend.
- Outcome: Removed the import-time backend switch and load pyplot only when chart generation is requested, preserving the notebook's selected backend. Added a regression test for backend preservation.
- Artifacts: `src/gold_vol_pipeline/models.py`, `tests/test_models.py`
- Next: Rerun the affected notebook with its intended Jupyter kernel.

## 2026-11-10 — Verified notebook execution
- Author: Copilot
- Type: finding
- Context: The project requested an end-to-end execution check for all notebooks. The data-inspection notebook referenced an unavailable `deeplob` kernel, so execution initially stopped before running any cells.
- Outcome: Updated its kernel metadata to the standard `python3` kernel. All three notebooks then executed without cell errors. The Yahoo download notebook validated 2,981 closes and 2,975 prepared target rows; the baseline and evaluation notebooks ran on the existing prepared dataset. Statistical-library warnings were emitted during model fitting but did not prevent execution.
- Artifacts: `notebooks/01_inspect_gold_data.ipynb`, `notebooks/02_models.ipynb`, `notebooks/03_model_evaluation.ipynb`
- Next: None.

## 2026-11-10 — Added sequential baseline forecast comparison
- Author: Enoch
- Type: implementation
- Context: Baseline variance forecasts needed a time-ordered comparison with realized-variance targets.
- Outcome: `prediction_history` now supports evaluating a recent window of forecast origins while using all prior returns at each origin. The model notebook summarizes MAE/RMSE and plots actual versus forecast values by baseline. This is exploratory evidence, not a final model-selection decision.
- Artifacts: `src/gold_vol_pipeline/models.py`, `notebooks/02_models.ipynb`, `plots/prediction_vs_actual.png`, `tests/test_models.py`
- Next: Confirm the evaluation window, target, and selection criteria with the team before interpreting results.

## 2026-11-10 — Added model-evaluation workbench
- Author: Copilot
- Type: implementation
- Context: The project needed reusable time-series transformations, candidate-model comparison, residual diagnostics, and an interactive review notebook.
- Outcome: Added price/log/difference/return transformations; ADF unit-root tests; ACF/PACF and transformation plots; configurable AR, MA, ARMA, ARCH, GARCH, and EGARCH fits with AIC/BIC; nonseasonal Hyndman-Khandakar stepwise ARIMA selection; residual export/panels; and Ljung-Box tests. Added `notebooks/03_model_evaluation.ipynb`. These tools provide exploratory/in-sample evidence, not a model-selection decision.
- Artifacts: `src/gold_vol_pipeline/model_evaluation.py`, `notebooks/03_model_evaluation.ipynb`, `tests/test_model_evaluation.py`, `README.md`, `presentations/project-progress.qmd`, `pyproject.toml`
- Next: Model analysis should review the diagnostics and information-criterion limitations, then use an agreed chronological out-of-sample evaluation before recommending a model.

## 2026-11-10 — Added Hyndman-Khandakar ARIMA selection
- Author: Copilot
- Type: implementation
- Context: Enoch requested automatic ARIMA parameter selection for the return-only volatility work.
- Outcome: Added a nonseasonal Hyndman-Khandakar selector using KPSS differencing and bounded stepwise AICc search, with a CLI that fits squared returns by default and prints order, coefficients, candidate ranking, and a five-step variance-proxy forecast. On the local prepared Yahoo `GC=F` data (2,975 observations), the exploratory full-sample fit selected ARIMA(0,1,4), AICc -39098.363176; estimated trend/drift (`x1`) 3.122375346e-08, MA coefficients (-0.9178895503, 0.1926706782, -0.2153530483, 0.03742078706), and sigma2 1.183282999e-07. This is a full-sample fit, not a held-out evaluation or model-selection decision.
- Artifacts: `src/gold_vol_pipeline/auto_arima.py`, `tests/test_auto_arima.py`, `README.md`, `presentations/project-progress.qmd`
- Next: Include the selector in a chronological rolling-origin evaluation before comparing it with the existing baselines; confirm the target/instrument choices with the team.

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
