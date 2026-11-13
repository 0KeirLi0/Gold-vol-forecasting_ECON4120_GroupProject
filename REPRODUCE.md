# Reproducing the unified evaluation

Run these commands from the repository root. The default instrument is Yahoo
Finance `GC=F`, a continuous COMEX gold futures series (not spot gold). The
end date is exclusive; use a fixed end date to reproduce a particular data
snapshot.

## Environment and ordered run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,interactive]"
python -m gold_vol_pipeline --start 2015-01-01 --end 2026-11-04
python -m gold_vol_pipeline.predictors --start 2015-01-01 --end 2026-11-04
jupyter nbconvert --to notebook --execute notebooks/06_final_evaluation.ipynb --output 06_final_evaluation.executed.ipynb --output-dir data/processed
```

The evaluation notebook writes `data/processed/evaluation_forecasts.csv`, a
JSON run manifest beside it, and 200-dpi PNG figures under
`plots/final_evaluation/`. It uses the full available date span, a 750-origin
minimum training window, and weekly forecast origins (`step=5`). Repeated
Hyndman-Khandakar, GARCH, lasso, and ridge fits make the run computationally
intensive; plan for several hours on a typical laptop. Actual runtime depends
on hardware and the input snapshot. The final output notebook is stored under
`data/processed/` so it does not overwrite the source notebook.

The same run compares the seven-predictor specification with reduced
VIX-only and DXY-plus-VIX specifications for HARX, lasso, and ridge, using the
same forecast origins. It writes the focused loss and Diebold-Mariano
comparison table to `data/processed/predictor_set_comparison.csv`. DXY-plus-VIX
is a sensitivity set informed by the separate LSEG screen; it should not be
interpreted as a Yahoo/FRED significance finding.

The first command prepares the gold futures series. The second downloads and
aligns the Stage 2 predictor panel and writes
`data/processed/stage2_multivariate.csv`. The final command runs all cells in
the evaluation notebook in order.

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
