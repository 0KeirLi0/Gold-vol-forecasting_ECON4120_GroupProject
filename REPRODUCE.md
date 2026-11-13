# Reproducing the short Stage 3 evaluation

Run the evaluation from the repository root. It reuses the already completed
forecast files in `data/processed/evaluation_forecasts_by_model/` and
`data/processed/predictor_set_variant_forecasts.csv`; it does not download data
or refit forecasts. The Stage 2 panel is still read to verify the target and
horizon.

## Environment and run

```bash
python -m pip install -e ".[dev,interactive]"
python -m nbconvert --to notebook --execute notebooks/06_final_evaluation.ipynb --output 06_final_evaluation.executed.ipynb --output-dir data/processed
```

The evaluation is designed to complete in approximately 10 minutes or less on
a typical laptop; runtime depends on the environment. It uses saved return-only
baselines, HAR/HARX forecasts, and the nine predictor-set variants. The
notebook checks the common origin dates and actual targets before comparing
models. The earlier GARCH(1,1) and 500-origin results in
`cross_model_comparison_with_baselines.csv` are displayed as context only,
because their sample is not the same as the 445 weekly origins used by the
Stage 3 comparison.

The notebook writes `data/processed/evaluation_forecasts.csv`, its run
manifest, `data/processed/evaluation_summary.csv`, and
`data/processed/predictor_set_comparison.csv`. It also refreshes 200-dpi
figures under `plots/final_evaluation/`. The
executed notebook is stored under `data/processed/`, leaving the source
notebook unchanged.

To refit the original seven Stage 1 baselines, the BIC-selected VAR and the
HK auto-ARIMA at the exact same 445 weekly origin dates without refitting the
other completed Stage 3 models, run these commands in order:

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

The cached notebook itself still omits repeated auto-ARIMA, VAR, GJR-GARCH,
asymmetric EGARCH, Student-t GARCH, and GARCH-X fits. The first two are added
by the separate scripts above. The other code remains available
where implemented, but no new performance evidence for them is claimed here.
This runtime-based scope is not a model-selection decision. To rebuild inputs
from raw sources instead, use the pipeline commands in [`README.md`](README.md)
and verify the fixed end date, data vintage, and market-data usage terms.

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
