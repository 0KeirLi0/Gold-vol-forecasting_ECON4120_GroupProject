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

Repeated auto-ARIMA, VAR, GJR-GARCH, asymmetric EGARCH, Student-t GARCH, and
GARCH-X fits are omitted from this short run. Their code remains available
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
