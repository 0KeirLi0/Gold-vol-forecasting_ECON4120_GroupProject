# Gold volatility forecasting

This Applied Forecasting Methods group project uses LSEG Workspace `GCc1`,
a generic COMEX gold futures series (not spot gold or a fixed-expiry
contract). The current analysis has two stages:

1. **Predictor significance:** test 11 LSEG predictors against the next
   valid Yang-Zhang daily volatility, controlling for current volatility.
2. **Yang-Zhang volatility forecasting:** compare one-step volatility
   forecasts on a chronological holdout.

The stages share target preparation and source data. The screen does not
automatically select predictors or alter the saved forecasts.

## Setup and inputs

Install with `python -m pip install -e ".[dev,interactive]"`; run tests with
`python -m pytest`. Inputs are LSEG exports in `data/refinitiv/`. Stage 1
requires `GCc1.csv`, DXY, SPX, VIX, US10YT, five gold implied-volatility
tenors, risk reversal and butterfly exports. The current Stage 2 comparison
uses GCc1 OHLC, DXY, gold IV 1M and gold IV 1Y; VIX is not included in its
forecast models. The input file names and SHA-256 hashes are
recorded in each manifest. `scripts/download_lseg_data.py` requires an
authenticated LSEG Workspace environment; verify data rights, quote units
and publication timing before use.

## Stage 1: Predictor significance

```bash
PYTHONPATH=src python scripts/screen_yang_zhang_predictors.py
```

The screen begins with 11 predictors and the current-volatility control,
using the same complete training pairs for all iterations. It calculates
VIFs including the control and iteratively removes the highest-VIF candidate
until all remaining VIFs are at most 5. The control is never removed.
The 80/20 split is
made before pairing; 3,039 training pairs end on 2023-06-01, before the
2023-06-02 holdout. Predictors use the latest value available at each origin
under a one-weekday proxy. Missing target labels can mean that the next valid
observation skips market sessions.

The 6M, 3M, then 2M gold implied-volatility predictors were removed; eight
remain, with a maximum VIF of 4.31 including the control. Next-valid
volatility is regressed on these predictors and the control, standardized on
the training pairs. HAC(4) standard errors and Holm correction across the
eight retained predictor tests yield adjusted p-values below 5% for 1M gold
IV, 1Y gold IV, and DXY; VIX's adjusted p-value is 0.587.
These post-selection conditional associations are exploratory, not causal
findings or evidence that a predictor has or lacks forecasting value. VIF
selection does not eliminate all collinearity; unverified publication timing,
finite HAC lag, and selection on the same sample limit interpretation.
Results, the full VIF history, aligned pairs, and the input-hash manifest are in
`data/processed/stage1_predictor_significance/`.

## Stage 2: Yang-Zhang volatility

The target is the unannualized square root of the two-observation
Yang-Zhang variance computed from OHLC; incomplete bars are not imputed.
There are 3,040 training and 761 test targets. Models are refit
sequentially using only targets observed before each forecast. Adjacent
target windows overlap. The current experiment compares six models with the
three predictors jointly against their saved no-external-predictor baseline.
MAE and RMSE rankings are descriptive, not forecast-comparison significance
tests or a team model-selection decision.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/evaluate_yang_zhang_volatility.py --predictor-dir data/refinitiv \
  --external-models-only --output-dir data/processed/stage2_yz_selected
PYTHONPATH=src python scripts/compare_selected_predictors.py
```

All six models completed 761 test dates. Adding the three predictors jointly
improved both MAE and RMSE for Ridge, Lasso and VAR, worsened both for HARX,
and produced mixed changes for gradient boosting and GARCH-X. This comparison
does not identify individual predictor effects. Some optimizer warnings
occurred. Stage 1 used training data only, but the predictors were chosen
after earlier holdout results were observed, so this is not a fully untouched
holdout.

See the [report](reports/yang-zhang-volatility.pdf) and its
[editable source](reports/yang-zhang-volatility.qmd) for formulas, VIF and
screen results, model specifications, metrics, and forecast graphs with the
test period shaded. See [REPRODUCE.md](REPRODUCE.md) for reproduction steps.
The modern [Group 2 Beamer presentation](presentations/Group2_Slides.pdf) and
[editable slide source](presentations/Group2_Slides.qmd) summarize the same
Yang-Zhang analysis.
The [test-vs-main comparison](reports/test-vs-main-model-comparison.pdf)
summarizes branch changes and explains why model scores from the two targets
cannot be ranked directly.
The [Yang-Zhang seasonality report](reports/yang-zhang-seasonality.pdf)
checks month and weekday patterns in the original LSEG/Refinitiv GCc1
two-observation volatility series. It finds a descriptive March/April/February
elevation and no robust weekday effect, but HAC month-test p-values are
lag-sensitive and do not support a stable monthly seasonality claim.
The [Stage 2 forecast-error report](reports/yang-zhang-residual-diagnostics.pdf)
summarizes holdout error bias, tails, serial dependence, extreme misses, and
retrospective volatility-regime performance. It analyzes out-of-sample
forecast errors, not fitted GARCH innovations.

The [selected-model residual report](reports/yang-zhang-selected-residual-diagnostics.pdf)
adds GARCH(1,1), EGARCH(1,1), EGARCH-X, range GARCH-X, HAR-X and histogram
gradient boosting. EGARCH-X uses the selected DXY and gold implied-volatility
predictors; the other five models use the no-external-predictor run. These too
are out-of-sample forecast errors, not fitted GARCH innovations; see
[REPRODUCE.md](REPRODUCE.md) for the command to regenerate the tables and
figures.
The selected-model report now also includes training-origin AIC/BIC where a
likelihood is defined; mixed target transformations and non-likelihood models
are explicitly flagged.

## Collaboration

Keir focuses on infrastructure and data, Enoch on model development and
variables, and Samuel on model analysis and recommendations; coordinate
cross-stage changes. Follow [GIT_WORKFLOW.md](GIT_WORKFLOW.md) and the shared
[AI handoff protocol](.ai/README.md). Historical context is retained in the
append-only [session log](.ai/SESSION_LOG.md), not as an active analysis.
