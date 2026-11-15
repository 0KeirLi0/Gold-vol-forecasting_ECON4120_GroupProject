# Current Yang-Zhang daily-volatility comparison

The current model comparison uses LSEG `GCc1.csv` OHLC. It computes
unannualized 2-observation Yang-Zhang volatility, leaves windows with
incomplete OHLC unscored, and evaluates sequential one-step forecasts after a
chronological 80/20 split of available target observations. Each model is
re-estimated from the initial training data plus prior observed test targets.
One step advances to the next valid target date, which may skip dates after an
incomplete OHLC window. The Yang-Zhang persistence forecast uses the latest
available target. Candidate failures are retained;
the rankings use the common successful test observations and report both MAE
and RMSE. The overlapping 2-observation targets make the loss sequence
dependent; the current output is descriptive and does not apply significance
tests.

Run from the repository root:

```bash
PYTHONPATH=src python scripts/evaluate_yang_zhang_volatility.py
```

The script uses four workers by default; pass `--workers 1` for sequential
execution or another positive integer for a different worker count.

For the user-requested VIX-augmented variant, keep the original no-VIX
artifacts separate and run:

```bash
PYTHONPATH=src python scripts/evaluate_yang_zhang_volatility.py \
  --vix-input data/refinitiv/VIX.csv --output-dir data/processed/stage4_yz_vix
```

The latest VIX close available by each origin under the next-weekday proxy is
added to HARX, Ridge, Lasso, gradient boosting, range GARCH-X, and VAR.
VIX source and availability dates are retained in the prepared file, and the
input hash is included in the manifest. The proxy does not reconstruct
publication times. The earlier significance finding used a different target;
it does not establish significance for Yang-Zhang volatility, and the screen
training sample extended into the start of this holdout. The completed VIX
run has 761 common successful dates and zero recorded failures; gradient
boosting has the lowest MAE (0.00242640) and RMSE (0.00478093).

The run reads `data/refinitiv/GCc1.csv` and writes the prepared target,
forecast attempts, summary, and source-hash manifest under `data/processed/`:

- `GCc1_yang_zhang_volatility.csv`
- `stage4_yz_forecasts.csv`
- `stage4_yz_summary.csv`
- `stage4_yz_manifest.json`

The original no-VIX n=2 run covered all 761 test observations for all 20 candidates,
with zero recorded fit failures. HistGradientBoosting had the lowest MAE
(0.00241765) and RMSE (0.00476724). Optimizer and convergence warnings were
emitted; because target windows overlap, these scores are descriptive and no
significance tests were run.

The estimator and candidate set are documented in the current-model section
of [`README.md`](README.md). Rows without complete OHLC are not imputed, so
their Yang-Zhang windows are unavailable and do not enter the 80/20 target
split. The holdout advances to the next valid target observation at each
one-step origin; check the saved target dates and manifest when interpreting
results around missing source bars. This specification is a user-selected
working analysis, not a team-approved model decision.

## Yang-Zhang report and forecast figures

The full formula and model-specification report is available as
[`reports/yang-zhang-volatility.pdf`](reports/yang-zhang-volatility.pdf), with
its editable Quarto source at
[`reports/yang-zhang-volatility.qmd`](reports/yang-zhang-volatility.qmd).
The report includes the full MAE/RMSE comparison and a timeline for each
candidate; the chronological test interval is shaded in the plots.

Regenerate the figures and render the PDF after running the holdout:

```bash
python scripts/plot_yang_zhang_report.py \
  --processed-dir data/processed/stage4_yz_vix \
  --output-dir reports/figures/stage4_yz_vix
quarto render reports/yang-zhang-volatility.qmd
```

Figures are saved under `reports/figures/stage4_yz_vix/`. The plotting command
requires Matplotlib. The report's numerical table and narrative are a
snapshot; update them from the new summary before rendering a different run.

## Historical integrated Stage 3 evaluation

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
quarto render reports/model-performance-stage2-stage3.qmd --to pdf
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
quarto render reports/model-performance-stage2-stage3.qmd --to pdf
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
