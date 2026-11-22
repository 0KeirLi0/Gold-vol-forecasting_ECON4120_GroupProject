# Group 2 report and presentation

The submitted report and 20-minute Beamer are the LSEG `GCc1` next-session
Yang–Zhang analysis. They do not use the Yahoo five-session GARCH-X comparison.
Render the report and slides from separate Quarto sources:

```bash
quarto render report_group2/Group2_Report.qmd --to pdf
quarto render presentations/Group2_Slides.qmd --to beamer
cp presentations/Group2_Slides.pdf report_group2/Group2_Slides.pdf
```

Figures already in `report_group2/figures/` are used as-is; the render does
not refit models. The story outline is `report_group2/narrative-draft.md`.
The earlier Yahoo/FRED GARCH-X report is preserved under
`archive/model_exploration_2026-11-15/`.

Rebuild the report diagnostics (ADF, ACF/PACF, price/vol plots, sequential
forecast panels, residual-time plots, training AIC/BIC, Hyndman–Khandakar
re-run) from saved Stage 2 forecasts without refitting the holdout:

```bash
python scripts/report_yz_diagnostics.py
```

Outputs go to `report_group2/diagnostics/` and `report_group2/figures/`.

Rebuild the reviewer-feedback robustness tables (Q5 error influence,
2023–24 / 2025–26 split, $n=5$ and $n=10$ windows, HAR-X month dummies)
from the saved shortlist errors plus sequential refits:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python3 scripts/report_feedback_robustness.py --workers 2
```

Cheap tables only (no sequential window or month-dummy refits):

```bash
PYTHONPATH=src python3 scripts/report_feedback_robustness.py --skip-sequential
```

Six-model Normal versus Student-$t$ sequential holdout (slow; GARCH-X/EGARCH-X
use a custom likelihood). Copy the summary and quintile CSVs into
`report_group2/diagnostics/` after the run because
`data/processed/stage2_yz_distributions/` is gitignored:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python3 scripts/compare_volatility_distributions.py \
  --workers 6 --output-dir data/processed/stage2_yz_distributions
```

# Reproducing segmented Yang-Zhang estimation

The segmented experiment estimates separate normal/extreme parameter sets for
GARCH, EGARCH, GARCH-X, EGARCH-X, HAR-X, and histogram gradient boosting. It
uses the prepared LSEG `GCc1` Yang-Zhang input and selected DXY, gold 1M IV,
and gold 1Y IV columns. The normal/extreme threshold is the expanding training
history's 80th percentile of realized volatility. Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/evaluate_segmented_yang_zhang.py \
  --input data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv \
  --output-dir data/processed/stage2_yz_segmented --workers 8
```

The `forecast_trigger` experiment fits a preliminary full-history forecast and
selects the extreme parameter set when that forecast exceeds the threshold.
The `lagged_target_trigger` experiment selects the set using the latest
observed target `y_(t-1)`. Each selected parameter set is estimated only from
prior observations whose lagged target is in the selected normal/extreme
regime. Outputs are `segmented_yz_forecasts.csv`,
`segmented_yz_summary.csv`, and `segmented_yz_manifest.json`. These are
exploratory holdout diagnostics, not a final model choice: the targets
overlap, the holdout was used in earlier comparisons, and forecast-trigger
regime selection is paired with lagged-target-defined estimation samples.

# Reproducing the integrated Stage 3 evaluation

Run the evaluation from the repository root. It refits seven candidates on all
requested origins and reuses the other 18 models from the existing aligned
forecast file. It does not download data. Preserve the Stage 2 panel and the
aligned forecast input; both hashes are recorded in the output manifest.

## Environment and run

```bash
python -m pip install -e ".[dev,interactive]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m nbconvert --to notebook --execute notebooks/06_final_evaluation.ipynb --ExecutePreprocessor.timeout=1800 --output 06_integrated.executed.ipynb --output-dir data/processed
```

The notebook refits fixed-order HK auto-ARIMA, BIC-selected VAR, GJR-GARCH,
asymmetric EGARCH, Student-t GARCH, VIX-GARCH-X, and quantile GARCH. It checks
dates and actuals and saves all attempted forecasts, a balanced common-origin
sample, a loss summary and a manifest as `stage3_integrated_*.csv/json` under
`data/processed/`. Fit failures remain explicit. It also updates the
predictor-set comparison and figures under `plots/final_evaluation/`.
The run can take several minutes; runtime depends on hardware. The GARCH-X
model holds the origin's VIX constant in the forecast horizon. The HK order
is selected at the first origin and retained thereafter. See the report for
the equations and limitations.

## Earlier 20-model comparison

The following legacy commands reproduce the earlier 20-model comparison,
not the integrated results; run them only when intentionally rebuilding the
historical cached input and note that they overwrite existing comparison files:

```bash
python scripts/evaluate_stage1_on_stage3_origins.py
python scripts/evaluate_stage3_var.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/evaluate_stage3_auto_arima.py --workers 4
python scripts/plot_stage3_all_model_forecasts.py
quarto render archive/model_exploration_2026-11-15/model-performance-stage2-stage3.qmd --to pdf
```

The HK run selects one nonseasonal order on squared returns available at the
first origin, using KPSS differencing and stepwise AICc with the existing
selector's default search bounds. It then refits that fixed order using
expanding data at each origin and sums five individually nonnegative daily
forecasts. Its CSV records the fixed order and explicit fit errors; its manifest
records the selection origin, AICc, KPSS p-values, input-panel hash and fit
counts. The 445 fits can take considerably longer than the cached notebook;
`--workers` controls parallel processes. The combined comparison uses only
dates with valid forecasts for every model, while the appendix charts retain
all attempts. Running the baseline script replaces the combined files, so run
the VAR and HK scripts afterward to restore those models.

## Refit scaled GARCH only

To reproduce the 445-origin GARCH update while retaining the other completed
forecasts, run the following against the existing panel and full 20-model
comparison. This refits every origin, checks convergence and parameter
constraints, requires all 20 models to have forecasts at every origin, and
refreshes the saved comparisons and baseline GARCH figure.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/refit_stage3_garch.py
python scripts/plot_stage3_all_model_forecasts.py
quarto render archive/model_exploration_2026-11-15/model-performance-stage2-stage3.qmd --to pdf
```

The fit scales returns by their root mean square at each origin and rescales
forecast variance to original units. The run used Python 3.13, NumPy 2.4.6,
pandas 2.3.3, and SciPy 1.15.2. Intermediate bounds-clipping warnings were
observed; a successful fit and constraint checks do not prove global optimality.
As an additional sensitivity check, scaled initial parameters
`(omega, alpha, beta) = (0.05, 0.05, 0.90)` and `(0.20, 0.05, 0.75)` yielded
negative log-likelihoods within 1e-8 at origins 2018-01-05, 2018-01-22,
2018-04-05, 2018-11-13, 2019-02-12, 2023-06-20, 2024-11-06, and 2026-11-03.
This limited check is not proof of global optimality.

To rebuild inputs from raw sources instead, use the pipeline commands in
[`README.md`](README.md) and verify the fixed end date, data vintage, and
market-data usage terms. Rebuilding the historical forecast files changes the
cached input to the integrated run.

## Exploratory quantile GARCH calibration

Run the following to reproduce the single-origin calibration, without changing
the 445-origin comparison or downloading data:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/calibrate_quantile_garch.py --origin 2026-11-03 --quantile 0.95 --horizon 5 --simulations 20000 --seed 42
```

The JSON output includes the input hash, origin, fitted coefficients, threshold,
observation counts, and forecast. Use `--input` to select another compatible
panel. The threshold is recomputed using only returns through the specified
origin and held fixed during simulation. The optimizer fits scaled returns
from two starting points; bounds-clipping warnings can occur. The output is a
single fitted case, not an out-of-sample comparison or significance test.
The reported case used a panel with SHA-256
`a2f957950e4df6d445271b720ac758a4eb777d998e4e37e0f1195229c515afcd`
and Python 3.13, NumPy 2.4.6, pandas 2.3.3, and SciPy 1.15.2.

## Obtaining source data

The pipeline downloads Yahoo Finance `GC=F` gold futures, `DX-Y.NYB` U.S.
Dollar Index, `^GSPC` S&P 500 price index, and `^VIX` observations. The
predictor pipeline obtains FRED `DGS10`, `T10YIE`, and `DCOILWTICO` as public
CSV downloads; no FRED API key is required. Yahoo observations may be
unavailable or revised, and FRED downloads are current-vintage data, not
historical real-time vintages. The one-weekday FRED availability date is an
approximation, not a series-specific release timestamp.

Installations and downloads do not grant rights to redistribute market data.
Check the applicable Yahoo Finance and FRED series terms before sharing or
committing observations. Keep downloaded data and manifests local under
`data/`; do not combine gold spot and gold futures into one series.
