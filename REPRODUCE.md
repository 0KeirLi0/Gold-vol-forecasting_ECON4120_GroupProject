# Reproducing the two-stage analysis

## Stage 1: Training-only Yang-Zhang predictor screen

```bash
PYTHONPATH=src python scripts/screen_yang_zhang_predictors.py
```

The script reads `GCc1.csv` and the 11 LSEG predictor sources from
`data/refinitiv/`; all are required. It retains the existing chronological
80/20 split before pairing consecutive valid training targets. Predictors
available by each origin use the one-weekday proxy and backward as-of
matching. Using these same complete training pairs, it calculates VIFs
including the current-volatility control, iteratively removes the candidate
with the highest VIF until every remaining VIF is at most 5, and retains the
control throughout. Ties follow the original predictor order. It regresses
next-valid Yang-Zhang volatility on the control and remaining candidates,
standardizing regressors on the training pairs. The screen uses HAC(4)
standard errors and Holm correction across retained predictor coefficients
at 5%; it does not use holdout responses.

The current 3,039-pair screen removes gold IV 6M, 3M and 2M; eight predictors
remain, and all final VIFs are at most 4.31. Gold IV 1M, gold IV 1Y and DXY
are significant after correction; VIX is not. These are exploratory
post-selection conditional associations; remaining correlation, selection
on the same sample, and potentially longer dependence limit interpretation. The procedure does not
reselect or rerun forecast models. Outputs under
`data/processed/stage1_predictor_significance/` are the source-hash
`manifest.json`, `training_pairs.csv`, `vif_history.csv`, and
`predictor_significance.csv`.

## Stage 2: Yang-Zhang daily-volatility comparison

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
dependent; the forecast comparison is descriptive and does not apply
forecast-loss significance tests.

Run from the repository root:

```bash
PYTHONPATH=src python scripts/evaluate_yang_zhang_volatility.py
```

The script uses four workers by default; pass `--workers 1` for sequential
execution or another positive integer for a different worker count.

For the current joint-predictor experiment, keep the original
no-external-predictor artifacts separate and run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/evaluate_yang_zhang_volatility.py --predictor-dir data/refinitiv \
  --external-models-only --output-dir data/processed/stage2_yz_selected
PYTHONPATH=src python scripts/compare_selected_predictors.py
```

The latest DXY, gold IV 1M and gold IV 1Y available by each origin under
the next-weekday proxy are added jointly to HARX, Ridge, Lasso, gradient
boosting, range GARCH-X, and VAR; VIX is not used. Source and availability
dates are retained in the prepared file, and input hashes are in the
manifest. The proxy does not reconstruct publication times. The completed run
has 761 common successful dates and zero recorded failures. Selection occurred
after earlier holdout results had been observed, so the comparison is not a
fully untouched holdout. Differences are joint descriptive comparisons, not
isolated predictor effects or forecast-loss significance tests.

The run reads `data/refinitiv/GCc1.csv` and writes the prepared target,
forecast attempts, summary, and source-hash manifest under `data/processed/`:

- `GCc1_yang_zhang_volatility.csv`
- `stage2_yz_forecasts.csv`
- `stage2_yz_summary.csv`
- `stage2_yz_manifest.json`

The original no-external-predictor n=2 run covered all 761 test observations for all 20 candidates,
with zero recorded fit failures. HistGradientBoosting had the lowest MAE
(0.00241765) and RMSE (0.00476724). Optimizer and convergence warnings were
emitted; because target windows overlap, these scores are descriptive and no
forecast-comparison significance tests were run.

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
  --processed-dir data/processed/stage2_yz_selected \
  --output-dir reports/figures/stage2_yz_selected
quarto render reports/yang-zhang-volatility.qmd
```

Figures are saved under `reports/figures/stage2_yz_selected/`. The plotting command
requires Matplotlib. The report's numerical table and narrative are a
snapshot; update them from the new summary before rendering a different run.
