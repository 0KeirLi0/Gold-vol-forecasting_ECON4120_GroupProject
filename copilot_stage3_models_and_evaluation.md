# Copilot task: Stage 3 — regression/penalized models, volatility extensions, and a unified out-of-sample evaluation

You are working in the `gold-volatility-pipeline` repository (ECON 4120 Applied Forecasting Methods group project). Read `README.md`, `AGENTS.md`, `llms.txt`, `pyproject.toml`, `src/gold_vol_pipeline/models.py`, `src/gold_vol_pipeline/predictors.py`, `notebooks/02_models.ipynb`, and `notebooks/05_advanced_models.ipynb` before editing anything. Match the existing code style: `from __future__ import annotations`, type hints, strict input validation raising `ValueError`, NumPy-style short docstrings, and pytest tests in `tests/`.

Do NOT commit, push, or create branches. Do NOT add or modify files under `data/`. Do NOT invent results in docs; write placeholders like `<fill after running>` where numbers are needed.

## Fixed project conventions (do not change)

- Instrument: Yahoo `GC=F` continuous COMEX gold futures. Never mix with spot.
- Daily close-to-close log returns `r_t`. Forecast origin `t` = after day-t close.
- Target: `realized_variance` at origin t = sum of `r_{t+1..t+h}^2`, default `h = 5` (column already in `data/processed/gold_futures.csv` and `data/processed/stage2_multivariate.csv`).
- Stage 2 predictor columns: `dxy_close`, `sp500_log_return`, `vix_close`, `treasury_10y_yield`, `breakeven_inflation_10y`, `wti_spot`, `gold_volume`. Use only values dated on or before the origin (already aligned by `prepare_multivariate_dataset`).

## No-look-ahead rule (critical — current notebooks partly violate it)

The target at origin s uses returns through s+h. When fitting any model that uses the target as the dependent variable at forecast origin t, the training set may only contain rows with `target_end_date <= date_t` (i.e. origins s <= t − h). Implement this as a purge/embargo of h observations in every rolling/expanding fit and in any train/test split. Add a test that fails if a training row's `target_end_date` is after the forecast origin.

## Task 1 — New module `src/gold_vol_pipeline/regression_models.py`

Implement origin-wise forecast functions, each returning the forecast of h-step summed realized variance (non-negative float) given only data available at the origin:

1. `har_features(returns, *, horizon)` — build HAR regressors from daily squared returns: `RV_d` (1-day), `RV_w` (5-day mean), `RV_m` (22-day mean). Work in logs (`log(RV + eps)`) as an option `log_space: bool = True`; when back-transforming from logs apply the log-normal bias correction `exp(mu + 0.5*sigma^2)` using the training residual variance.
2. `har_forecast(...)` — OLS HAR-RV (Corsi 2009) for the h-step target.
3. `harx_forecast(..., exog=["vix_close"])` — HAR plus lagged exogenous predictors (default VIX only, since it was the only predictor significant after Holm correction; accept any subset). This is the "dynamic regression" model in the course list.
4. `penalized_forecast(..., method="lasso"|"ridge")` — HAR terms + all seven predictors (+ 1-day lag of each); standardize using training-window statistics only; choose alpha with `sklearn.model_selection.TimeSeriesSplit` (with `gap=horizon`) on the training window only. Return the selected alpha and non-zero coefficients for reporting.
5. `var_forecast(...)` — small VAR on `[log RV_d, log vix_close]` (statsmodels `VAR`, lag order by BIC, max 5 lags), iterate h steps, sum the implied daily variances. Mark as optional/secondary in the docstring.

Add `scikit-learn>=1.2` to core `dependencies` in `pyproject.toml` (it is currently only in `interactive`).

## Task 2 — Extend volatility models in `src/gold_vol_pipeline/models.py` (use the `arch` package, already a dependency)

Add, without changing the existing seven baselines' behaviour:

- `gjr_garch_forecast` — GJR-GARCH(1,1,1) (`arch_model(..., p=1, o=1, q=1)`), the asymmetric model missing from the current EGARCH setup.
- `egarch_asym_forecast` — EGARCH(1,1,1) with the asymmetry term (`o=1`); use simulation or bootstrap multi-step forecasts as `arch` requires for EGARCH horizons > 1.
- `garch_t_forecast` — GARCH(1,1) with Student-t errors (`dist="t"`), justified by the Jarque-Bera rejection.
- `garchx_forecast` — GARCH(1,1) with lagged VIX in the variance equation is not directly supported by `arch`; instead implement it as a two-step model: GARCH(1,1) forecast as a regressor alongside lagged `vix_close` in an OLS for the h-step target (document this clearly), OR skip with a clear `NotImplementedError` note if it cannot be done cleanly.
- `ewma_estimated_forecast` — EWMA with λ chosen on the training window by minimizing QLIKE over a grid (0.80–0.99), instead of the fixed 0.94 (formal exponential-smoothing variant).

Rescale returns by 100 before `arch` fits (and back-scale the variance by 1e-4) to avoid convergence warnings. Record convergence status; never silently return garbage — raise or return NaN and let the evaluator log it.

## Task 3 — New module `src/gold_vol_pipeline/evaluation.py`

1. Loss functions: `mse`, `mae`, `qlike(actual, forecast) = actual/forecast − log(actual/forecast) − 1` (Patton 2011 robust form), each returning per-origin losses and means.
2. `diebold_mariano(loss_a, loss_b, *, horizon)` — DM test with Newey-West HAC variance using `horizon − 1` lags (overlapping targets) and the Harvey-Leybourne-Newbold small-sample correction; return statistic, p-value, mean loss difference.
3. `mincer_zarnowitz(actual, forecast, *, horizon)` — OLS `actual = a + b·forecast`, HAC (horizon−1 lags), joint Wald test of `a=0, b=1`.
4. `forecast_combination(forecasts: dict)` — equal-weight average of chosen models (default GARCH(1,1), EWMA, HAR).
5. `rolling_origin_evaluation(data, models: dict[str, callable], *, horizon, min_train=750, step=1, window="expanding")` — one shared loop for ALL models (univariate baselines, HK auto-ARIMA, HAR family, penalized, VAR, GARCH extensions, combination) on the SAME set of origins, enforcing the purge rule above. Return a long DataFrame `origin_date, model, forecast, actual, converged`. Allow saving to `data/processed/evaluation_forecasts.csv` (git-ignored) plus a JSON manifest (data range, horizon, min_train, step, package versions, run timestamp).
6. `summary_table(results, benchmark="garch11")` — per model: n, MSE, MAE, QLIKE, ratio to benchmark, DM p-value vs benchmark (MSE and QLIKE), MZ a, b and Wald p-value. Also a DM p-value matrix for the top models. Apply Holm correction across the DM p-values vs benchmark and report both raw and adjusted.

Use the full available sample (≈2015 onward), not the latest 20 origins.

## Task 4 — Resolve the auto-ARIMA discrepancy

The repo log records ARIMA(0,1,4) for HK selection on squared returns, but a rerun selected ARIMA(3,1,2). Write `tests/test_auto_arima_reproducibility.py` or a short script that: fixes the input file and date range, prints the input row count/date range/hash, runs `hyndman_khandakar` twice, and asserts identical output. In a short markdown note `docs/arima_discrepancy.md`, explain the likely cause (different data snapshot/end date, KPSS differencing decision, or stepwise search path) and which order the final evaluation uses (the one selected per origin inside the rolling loop). Do not fabricate which one is "correct".

## Task 5 — Notebook `notebooks/06_final_evaluation.ipynb`

- Loads the prepared Stage 2 CSV, runs `rolling_origin_evaluation` for all models, and displays `summary_table`.
- Plots: (a) actual vs forecast for the best 3 models + benchmark (label: "5-session forward realized variance, observed vs forecast"), (b) cumulative QLIKE loss difference vs GARCH(1,1), (c) lasso coefficient path / selection frequency over origins, (d) DM p-value heatmap.
- Saves figures to `plots/final_evaluation/` as PNG at 200 dpi with labeled units and horizon (these will be used in the report and slides).
- Ends with a markdown cell summarizing which models beat the benchmark at the 5% level under MSE and QLIKE, with placeholders for numbers.

## Task 6 — Tests

Add `tests/test_regression_models.py` and `tests/test_evaluation.py` covering: input validation; non-negative forecasts; the purge rule (no training row with `target_end_date` after origin); QLIKE equals 0 when forecast == actual; DM statistic ≈ 0 for identical losses and correct sign when model A is clearly better; HAR recovers known coefficients on simulated HAR data; lasso uses only training-window standardization. Run `python -m pytest` and make all tests pass, including the existing ones.

## Task 7 — Repository hygiene and docs

1. Create `.gitignore` with `data/`, `__pycache__/`, `*.pyc`, `*.egg-info/`, `.venv/`, `.ipynb_checkpoints/`, `presentations/*_files/`. Then print (do not run) the command the team should run to untrack already-committed data: `git rm -r --cached data/ src/**/__pycache__ tests/__pycache__ src/*.egg-info`.
2. Update `README.md` "Repository status" and `llms.txt` to list the new modules and notebook; remove or fix links to the missing `.ai/` and `.github/` folders (create `.ai/README.md` and `.ai/SESSION_LOG.md` with a dated Stage 3 entry labeled "proposal", and `.github/copilot-instructions.md` summarizing the conventions above).
3. Add `REPRODUCE.md`: environment setup, exact commands in order (data download → predictors → `06_final_evaluation`), expected runtime, and how to obtain data that cannot be redistributed (Yahoo `GC=F`, `DX-Y.NYB`, `^GSPC`, `^VIX`; FRED `DGS10`, `T10YIE`, `DCOILWTICO`).

## Deliverable checklist (report back when done)

- List of files created/changed.
- `pytest` output summary.
- Any model that failed to converge or was skipped, with the reason.
- Remaining TODOs that need a team decision (e.g. final benchmark choice, min_train, whether to keep VAR).
