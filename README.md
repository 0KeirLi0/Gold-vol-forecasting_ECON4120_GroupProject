# Gold Volatility Forecasting

An Applied Forecasting Methods group project exploring forecasting approaches for gold-market volatility. The project will develop iteratively, starting with methods from the course and extending the methodology as model comparisons and analysis motivate further work.

## Project workflow

Each iteration follows a shared loop:

1. **Infrastructure** — prepare data and repository artifacts, and support presentations.
2. **Model development** — identify relevant variables and develop a forecasting model.
3. **Model analysis** — compare candidate models, explain results and limitations, and recommend the next step.
4. **Next iteration** — use that recommendation to guide the next round.

```mermaid
flowchart LR
    I["Infrastructure<br/>Keir"] --> D["Model development<br/>Enoch"]
    D --> A["Model analysis<br/>Samuel"]
    A --> Q{"Next-step decision"}
    Q -->|"Refine or compare models"| I
    Q -->|"Add a method or investigate a new factor"| I
    D -. "Identify data requirements" .-> P["Build or update data pipeline<br/>Keir + model development"]
    P --> D
```

The analysis decision closes one round and sets the question for the next. If the next step needs new or changed data, first clarify the model-development requirements, then update the pipeline with Keir. Keep handoffs explicit: model development should share the model specification and inputs with analysis; analysis should share its evidence, limitations, and recommended next step with the team.

Early iterations may explore course methods such as AR, MA, ARMA, ARCH, GARCH, or EGARCH. As the project develops, later iterations may investigate extensions such as quantile regression or shock analysis. These are examples, not a prescribed or exhaustive method list; the team will choose methods based on project questions and findings.

## Team

| Member | Primary focus |
|---|---|
| Keir | Infrastructure, data acquisition, repository management, and presentation materials |
| Enoch | Model development and identifying relevant variables |
| Samuel | Model analysis, evaluating and comparing candidate models, discussing limitations, and recommending next steps |

These are areas of focus rather than exclusive ownership. Coordinate changes and handoffs when work crosses stages.

## Data and reproducibility

Keir selected gold futures as the Stage 2 working instrument; the current
downloadable series is Yahoo Finance `GC=F`, a COMEX gold futures continuous
series. It is not spot gold or a single fixed-expiry contract, and its history
may reflect contract rolls. The CLI and notebook download daily data from
Yahoo Finance by default; local CSV input is also supported. Revisit the series
or instrument only by an explicit team choice, and never mix spot and futures.
Yahoo's `XAUUSD=X` spot ticker returned “No data found” when checked on
2026-11-06.

The pipeline computes close-to-close log returns and a forward target over the next five observed price sessions by default. The default target is realized variance, calculated as the sum of squared daily log returns; `realized_volatility` is also available as the square root of that sum. Horizon and target are configurable. The target uses future observations only as the label for each forecast origin; predictors in this baseline contain only information available through that origin.

Raw and prepared downloads are written under `data/` and excluded from Git. Preserve them for each analysis run and record their provenance, observation date range, and pipeline options. The prepared CSV includes the forecast-origin date, close, log return, target-window dates, target value, and provider/instrument metadata. Do not mix futures and spot observations. Do not share Bloomberg or other restricted exports unless the applicable license permits it.

### Current daily-volatility model comparison

The current model-development comparison uses only LSEG Workspace
`data/refinitiv/GCc1.csv`, a generic COMEX gold futures series that is distinct
from Yahoo `GC=F`. It constructs unannualized daily volatility as the square
root of the 2-observation Yang-Zhang variance estimator from overnight
returns $r_o=\log(O_t/C_{t-1})$, open-to-close returns
$r_c=\log(C_t/O_t)$, and Rogers-Satchell terms
$RS_t=\log(H_t/O_t)\log(H_t/C_t)+\log(L_t/O_t)\log(L_t/C_t)$:

$$
\widehat{\sigma}_{YZ,t}^2 =
s_o^2 + k s_c^2 + (1-k)\overline{RS},\qquad
k=\frac{0.34}{1.34+(n+1)/(n-1)},\quad n=2.
$$

The target is $\widehat{\sigma}_{YZ,t}=\sqrt{\widehat{\sigma}_{YZ,t}^2}$;
`s_o^2` and `s_c^2` are sample variances. It is not annualized. Rows with
incomplete OHLC are not imputed; estimator windows containing incomplete
observations have no target and are excluded from model fitting and scoring.
The 2-observation target windows overlap, so the resulting loss series is
dependent; reported MAE/RMSE rankings are descriptive and no significance
tests are run.

Run the end-to-end workflow from the repository root:

```bash
PYTHONPATH=src python scripts/evaluate_yang_zhang_volatility.py
```

The prepared target is saved to
`data/processed/GCc1_yang_zhang_volatility.csv`. Forecasts, the out-of-sample
summary, and a provenance manifest are saved as `stage4_yz_forecasts.csv`,
`stage4_yz_summary.csv`, and `stage4_yz_manifest.json` in the same directory.
The evaluation uses a chronological 80/20 split of available target
observations and sequential one-step forecasts over valid target dates: at
each test origin, models are refit on the original training period and actual
target observations available before that origin. Because incomplete OHLC
windows have no label, one step means the next valid target observation, not
necessarily the next market calendar session. The Yang-Zhang persistence
forecast uses the latest available target. The script uses four workers by
default; adjust with `--workers N`. Candidate families include expanding-mean and EWMA
baselines; AR, MA, ARMA, and fixed-order HK auto-ARIMA; ARCH/GARCH,
GJR-GARCH, EGARCH, Student-t GARCH, GARCH-X, and threshold-GARCH; HAR/HARX;
Ridge, Lasso, HistGradientBoosting; and VAR. The dedicated rolling-variance
baseline is omitted. Results rank MAE and RMSE on shared successful test
dates; failures remain in the forecast file with error messages. These are
exploratory user-selected specifications, not a team-approved model decision.

The completed two-observation run evaluated all 20 candidates on the same
761 test dates with zero recorded fit failures. HistGradientBoosting had the
lowest MAE (0.00241765) and RMSE (0.00476724). Optimizer and convergence
warnings were emitted during fitting, so the ranking is descriptive evidence,
not a validated model recommendation; no significance tests were run.
This is the original no-VIX run. A separate VIX-augmented rerun is complete:
VIX is added to HARX, Ridge, Lasso, gradient boosting, range GARCH-X, and VAR
using the latest observation available by the forecast origin under the
next-weekday proxy. The earlier predictor significance finding concerned a
different five-session variance target and does not establish significance
for this Yang-Zhang target; that screen's training interval also overlaps
the start of this holdout. In the VIX-augmented run, all 20 models had 761
common successful dates and no recorded failures; gradient boosting had the
lowest MAE (0.00242640) and RMSE (0.00478093). These are descriptive
comparisons, not significance tests or a team recommendation. See
[`REPRODUCE.md`](REPRODUCE.md) for the command.

The existing Yahoo/CSV forward-realized-variance pipeline, notebooks, and
Stage 1–3 outputs are retained as historical analyses; they are not the target
or holdout used by this current comparison. Do not compare their metrics
directly with the Yang-Zhang holdout.

The standalone analysis report gives the target and all candidate-model
formulas, the full MAE/RMSE table, and forecast graphs with the chronological
test period shaded. See
[`reports/yang-zhang-volatility.qmd`](reports/yang-zhang-volatility.qmd) and
its rendered
[`PDF`](reports/yang-zhang-volatility.pdf). Regenerate figures with
`python scripts/plot_yang_zhang_report.py --processed-dir data/processed/stage4_yz_vix --output-dir reports/figures/stage4_yz_vix`,
then render with `quarto render reports/yang-zhang-volatility.qmd`.

### Stage 2 daily predictor panel

Keir selected the following daily candidates from Enoch's requirements:

| Candidate | Source and series | Frequency and units |
|---|---|---|
| USD strength | Yahoo Finance `DX-Y.NYB`, ICE U.S. Dollar Index | Daily close, index points |
| Equity return | Yahoo Finance `^GSPC`, S&P 500 price index | Daily close-to-close log return (derived from index points) |
| Equity volatility | Yahoo Finance `^VIX`, CBOE VIX | Daily close, index points |
| Nominal rate | FRED [`DGS10`](https://fred.stlouisfed.org/series/DGS10), 10-year Treasury constant-maturity yield | Daily, percent |
| Inflation expectations | FRED [`T10YIE`](https://fred.stlouisfed.org/series/T10YIE), 10-year breakeven inflation rate | Daily, percent |
| Oil price | FRED [`DCOILWTICO`](https://fred.stlouisfed.org/series/DCOILWTICO), WTI Cushing spot | Daily, U.S. dollars per barrel |
| Gold volume | Yahoo Finance `GC=F` | Daily provider-reported volume, when returned |

The pipeline preserves raw closes, rates, and volume; Enoch can choose and
document further transformations for modeling. The S&P 500 input is a price
index, not a total-return index. Breakeven inflation is market-implied
inflation compensation, not a survey measure. Futures volume is an optional
continuous-series measure, not a fixed-contract or exchange-wide volume
series. Central-bank gold demand is deferred until a specific source and
publication schedule support point-in-time alignment.

Run the daily predictor panel; FRED observations are downloaded as CSV directly
from the public FRED website and do not require an API key:

```bash
python -m gold_vol_pipeline.predictors --start 2016-01-01 --end 2026-11-01
```

[`notebooks/04_predictors.ipynb`](notebooks/04_predictors.ipynb) provides a
notebook interface for this CLI and previews the saved panel and manifest.

Yahoo daily closes are treated as available on their observation date because
the forecast origin is after that date's market data have been published. FRED
series are conservatively treated as available on the next weekday; the
prepared dataset uses the latest value available on or before each gold
forecast origin and retains each source observation date. Missing predictor
values remain missing; Enoch should inspect predictor freshness and choose a
missing-data policy before fitting models.
The one-business-day FRED lag is an availability-date proxy, not a
series-specific release timestamp, and the website download returns current
observations rather than historical real-time vintages. Verify release schedules
and revisions before using this panel for a strict real-time backtest.

The raw downloads, prepared dataset, and run manifest are written beneath the
Git-ignored `data/` directory. A local normalized CSV can be used instead with
`--predictor-input`; its `date` column must be the value's availability date,
with optional `{predictor}_observation_date` columns to retain source dates.
For example:

```bash
python -m gold_vol_pipeline.predictors \
  --gold-input data/raw/gold_futures.csv \
  --gold-source "Yahoo Finance" \
  --gold-instrument "COMEX Gold Futures continuous series (GC=F)" \
  --predictor-input data/raw/stage2_predictors.csv
```

The selected predictor columns are `dxy_close`, `sp500_close`, `sp500_log_return`
(close-to-close log difference), `vix_close`, `treasury_10y_yield`,
`breakeven_inflation_10y`, and `wti_spot`. A gold CSV passed with `--gold-input`
may also contain `volume`.

### Optional LSEG Workspace export

For a computer with LSEG Workspace, [`scripts/download_lseg_data.py`](scripts/download_lseg_data.py)
exports the supplied `GCc1`–`GCc3`, gold spot/fixing, and implied-volatility
examples, plus editable candidate RIC mappings for the Stage 2 variables.
It writes one CSV per RIC, a combined `ALL_LSEG_DATA.xlsx`, and a JSON
download manifest under `data/refinitiv/`:

```bash
python -m pip install lseg-data pandas openpyxl
python scripts/download_lseg_data.py --start 2010-01-01
```

Workspace must be installed, signed in, and open. Review
`PROJECT_PANEL_REQUESTS` in the script and confirm each RIC, field, units, and
history in Workspace before relying on the data; provider symbology and field
entitlements may differ. The script preserves series separately: do not mix
spot/fixing with the working gold-futures series, or treat `GCc1`–`GCc3` as
equivalent to Yahoo `GC=F`. These exports retain observation dates and do not
reconstruct point-in-time release availability, so they are not automatically
a leakage-safe `--predictor-input` panel. Keep licensed data local unless
redistribution is explicitly permitted.

#### LSEG export file and column schema

Each successful RIC request produces one CSV. The filename is the RIC with
characters other than letters, digits, `.`, `_`, and `-` replaced by `_`, then
leading/trailing dots and underscores removed. The first column is always
`Date` (the LSEG history index); the remaining columns are the requested field
codes returned by Workspace. A RIC with no data or an error has no CSV; check
`download_manifest.json` for its status and error message.

| CSV file | RIC | Series meaning and requested fields |
|---|---|---|
| `GCc1.csv` | `GCc1` | First generic COMEX gold futures contract: `TRDPRC_1`, `OPEN_PRC`, `HIGH_1`, `LOW_1`, `ACVOL_1`, `OPINT_1` |
| `GCc2.csv` | `GCc2` | Second generic gold futures contract; same fields as `GCc1` |
| `GCc3.csv` | `GCc3` | Third generic gold futures contract; same fields as `GCc1` |
| `XAUFIX.csv` | `XAUFIX=` | Gold fixing series: `TRDPRC_1` |
| `XAU.csv` | `XAU=` | Gold spot series: `TRDPRC_1` |
| `XAU1MO_R.csv`, `XAU2MO_R.csv`, `XAU3MO_R.csv`, `XAU6MO_R.csv`, `XAU1YO_R.csv` | Corresponding `XAU{tenor}=R` RIC | Gold implied-volatility tenors 1M, 2M, 3M, 6M, and 1Y: `MID_PRICE` |
| `XAU1MRR.csv` | `XAU1MRR=` | 1M gold risk-reversal quote: `MID_PRICE` |
| `XAU1MBF.csv` | `XAU1MBF=` | 1M gold butterfly quote: `MID_PRICE` |
| `DXY.csv` | `.DXY` | Candidate USD-strength index: `TRDPRC_1` |
| `SPX.csv` | `.SPX` | Candidate S&P 500 price index: `TRDPRC_1` |
| `VIX.csv` | `.VIX` | Candidate VIX level: `TRDPRC_1` |
| `US10YT_RR.csv` | `US10YT=RR` | Candidate 10-year US Treasury yield: `BID`, `ASK` |
| `USBEI10Y_RR.csv` | `USBEI10Y=RR` | Candidate 10-year breakeven inflation quote: `BID`, `ASK` |
| `USCRWT_ECI.csv` | `USCRWT=ECI` | Candidate WTI spot price: `TRDPRC_1` |

Field-code meanings: `TRDPRC_1` is the vendor's last-traded price field;
`OPEN_PRC`, `HIGH_1`, and `LOW_1` are the session open, high, and low;
`ACVOL_1` is accumulated trading volume; `OPINT_1` is open interest;
`MID_PRICE` is a midpoint quote; `BID` and `ASK` are bid and ask quotes.
Confirm the instrument-specific field description, units, quote convention,
and availability in Workspace. In particular, `MID_PRICE` for volatility
instruments may be quoted in volatility points, while risk-reversal and
butterfly signs/conventions are instrument-specific.

`ALL_LSEG_DATA.xlsx` has one worksheet per successful RIC, with the same
`Date` and field columns as that RIC's CSV. `download_manifest.json` records
the provider, requested date range, availability caveat, and for each RIC its
category, requested fields, status, returned columns, row count, and CSV name
(or an error/no-data message). This export does not calculate S&P 500 returns,
Treasury bid/ask midpoints, or predictor availability dates; those are later
transformations and must be documented before modeling. For exact returned
headers, consult the manifest because Workspace entitlements and instrument
definitions can differ.

#### Import LSEG exports into the forecasting pipeline

The pipeline can normalize the local exports and prepare a separate LSEG
dataset without overwriting the existing Yahoo/FRED Stage 2 output:

```bash
python -m gold_vol_pipeline.predictors --lseg-input-dir data/refinitiv
```

By default this writes normalized gold and predictor CSVs under `data/raw/`,
plus `data/processed/stage2_lseg_multivariate.csv` and its manifest. The
importer uses only `GCc1.csv` as the gold target; it never substitutes spot,
fixing, `GCc2`, or `GCc3`. It drops `GCc1` rows whose `TRDPRC_1` close is
missing, uses `ACVOL_1` as volume only when that field is actually present,
derives `sp500_log_return` from the S&P 500 closes, and derives the 10-year
yield as the midpoint of `BID` and `ASK` when the Treasury file is present.
Available gold implied-volatility, risk-reversal, and butterfly quotes are
preserved as additional columns (`gold_iv_1m`, `gold_iv_2m`, `gold_iv_3m`,
`gold_iv_6m`, `gold_iv_1y`, `gold_1m_risk_reversal`, and
`gold_1m_butterfly`). These additional option features are not part of the
existing seven-variable model specification.

All imported predictors are assigned an availability date one weekday after
their observation date as a conservative proxy; both dates are retained.
This does not reconstruct exact publication times or historical vintages.
Files absent from the export are skipped and listed in the generated manifest.
In the current local export, `GCc1.csv` has no `ACVOL_1`; breakeven-inflation
and WTI files are absent, and LSEG spot/fixing requests failed. Consequently
the prepared LSEG dataset is a partial predictor panel, not a replacement for
the seven-variable Yahoo/FRED dataset or a drop-in input for the existing
seven-variable Stage 2 model notebook. Review `data/refinitiv/download_manifest.json`
and coordinate feature changes with model development before estimating models.

#### Exploratory LSEG predictor significance

The LSEG predictor screen is in
[`notebooks/05_advanced_models.ipynb`](notebooks/05_advanced_models.ipynb).
It uses the current prepared `GCc1` panel, an 80/20 chronological split, and
purges training rows whose five-session target overlaps the first holdout
origin. The 4,057 complete observations span 2010-01-06 through 2026-11-03;
3,241 training rows remain after purging four rows. The model regresses
`log1p` five-session forward realized variance on standardized predictors,
uses HAC standard errors with four lags, and applies Holm correction across
the 11 individual predictor tests.

| Predictor | Direction in fitted model | Holm-adjusted p-value | Significant at 5% |
|---|---:|---:|:---:|
| DXY close | Negative | 0.00103 | Yes |
| VIX close | Positive | 0.0124 | Yes |
| 2M gold implied volatility | Positive | 0.140 | No |
| S&P 500 log return | Negative | 1.000 | No |
| 10-year Treasury yield | Positive | 1.000 | No |
| 1M gold implied volatility | Positive | 1.000 | No |
| 3M gold implied volatility | Positive | 1.000 | No |
| 6M gold implied volatility | Negative | 1.000 | No |
| 1Y gold implied volatility | Negative | 1.000 | No |
| 1M gold risk reversal | Positive | 1.000 | No |
| 1M gold butterfly | Positive | 1.000 | No |

The five implied-volatility tenors are significant as a joint block (HAC
F=17.42; Holm-adjusted p=9.26e-17 across three group tests), while the 1M risk
reversal and butterfly are not jointly significant (adjusted p=0.386).
Individual IV-tenor coefficients do not identify a preferred tenor: their
maximum absolute pairwise training correlation is 0.991. These are exploratory
conditional associations, not proof of out-of-sample predictive value or a
final feature-selection decision. The dataset uses a one-weekday availability
proxy, not verified historical publication times; confirm LSEG field units and
quote conventions before interpreting the option results. Available results
cover only the successful exports listed in the manifest.

API access and small file size do not grant redistribution rights. FRED's
series notes identify third-party copyright restrictions for some series
(including S&P 500, VIX, and WTI); check each series' terms before sharing or
committing data. Keep downloaded market data local unless redistribution is
explicitly permitted, and cite the source in any analysis.

### Setup and commands

Install the package and test dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For the interactive inspection notebook, install JupyterLab and plotting support, then launch JupyterLab from the repository root:

```bash
python -m pip install -e ".[interactive]"
python -m jupyter lab
```

Open [`notebooks/01_inspect_gold_data.ipynb`](notebooks/01_inspect_gold_data.ipynb). It downloads Yahoo `GC=F` by default, and lets you choose a date range, horizon, and target; inspect and plot data; check target dates and values; and save raw/prepared CSVs with a JSON manifest. Set `DATA_SOURCE = "csv"` to use a local file instead.

Download and prepare Yahoo futures data directly (the end date is exclusive; by default it is today, avoiding a partial current-day bar):

```bash
python -m gold_vol_pipeline
```

By default this downloads from 2015-01-01 to the current date (exclusive), saving the raw Yahoo prices to `data/raw/gold_futures.csv` and prepared data to `data/processed/gold_futures.csv`. Choose a range with `--start YYYY-MM-DD --end YYYY-MM-DD`, or change paths with `--raw-output` and `--output`. Set `--horizon` or `--target realized_variance|realized_volatility` to change target construction.

To prepare a local CSV instead, provide `--input data/raw/your_gold_series.csv`, its `--date-column` and `--close-column`, and accurate `--source` and `--instrument` labels. The original input is not modified. The data source must represent one consistent instrument and series.

Run the full test suite with `python -m pytest`. Run one test with:

```bash
python -m pytest tests/test_pipeline.py::test_default_target_is_forward_five_session_realized_variance
```

There is no configured linter yet.

### Return-only baseline forecasts

The `gold_vol_pipeline` package also provides return-only variance forecasts:
expanding historical variance, rolling historical variance, EWMA, AR(1) and
MA(1) models of squared returns, ARCH(1), and GARCH(1,1). Each forecast
function accepts returns available through a forecast origin and a configurable
horizon; it returns the expected variance summed over the next horizon
observations, matching the pipeline's realized-variance target definition.
GARCH(1,1) scales the returns by their root mean square during fitting and
converts the forecast variance back to the original units.
For example:

```python
from gold_vol_pipeline import baseline_forecasts, prepare_dataset

dataset = prepare_dataset(prices, horizon=5)
origin_returns = dataset["log_return"].dropna()
forecasts = baseline_forecasts(origin_returns, horizon=5)
```

Only pass returns known at the forecast origin. These are candidate baseline
forecasts, not a backtest by themselves. `notebooks/02_models.ipynb` performs
rolling-origin comparisons against prepared targets over the latest 20 common
origins and reports MAE/RMSE. The Hyndman-Khandakar order is reselected using
only returns available at each origin. The current comparison uses the working
`GC=F` series; target windows overlap and the evaluation window is small, so
the results are exploratory and do not constitute a final model-selection
decision.

Run exploratory diagnostics on the prepared return series with:

```bash
python -m gold_vol_pipeline.diagnostics --input data/processed/gold_futures.csv
```

The report includes tests for zero mean, stationarity, serial dependence,
ARCH effects, and return normality. These are diagnostic checks, not proof
that a model's assumptions hold or a substitute for out-of-sample evaluation.

### Model-evaluation workbench

Open [`notebooks/03_model_evaluation.ipynb`](notebooks/03_model_evaluation.ipynb)
to inspect price levels and transformations, ADF unit-root tests, ACF/PACF
plots, and return diagnostics. It fits configurable AR, MA, ARMA, ARCH, GARCH,
and EGARCH candidate orders and displays their model names, AIC, BIC, and
convergence status in a readable table. These are in-sample criteria for the
same input series; they do not choose a final project model.

The notebook also saves raw and standardized residual series under the
git-ignored `data/processed/` directory, plots residual panels, and applies
Ljung-Box tests to residuals and squared standardized residuals.
The evaluation helpers are in
[`src/gold_vol_pipeline/model_evaluation.py`](src/gold_vol_pipeline/model_evaluation.py).

Select a nonseasonal ARIMA model for squared returns (the volatility proxy)
with the Hyndman-Khandakar stepwise procedure:

```bash
python -m gold_vol_pipeline.auto_arima --input data/processed/gold_futures.csv
```

The implementation uses KPSS tests to choose the differencing order and
stepwise AICc search for AR/MA orders. It prints the selected order, fitted
coefficients, candidate rankings, and a five-session variance-proxy forecast.
Selection on the full sample is exploratory; use training-only selection
inside a chronological backtest before comparing forecast performance.

As the project develops, record enough information to reproduce and interpret each iteration: data inputs and transformations, model specification, evaluation evidence, selection criteria, limitations, and the rationale for the next step. Do not publish sensitive or restricted data; document its provenance and access requirements instead.

## Repository status

The first modeling stage uses traditional univariate time-series methods on one
gold-derived series: return-only candidate forecasts, sequential rolling-origin
comparisons with MAE/RMSE, exploratory diagnostics, configurable in-sample
candidate comparisons, and a nonseasonal Hyndman-Khandakar ARIMA selector.
The current provisional recommendation is GARCH-X with lagged VIX, pending team
review as described below. The automatic ARIMA selector
is included in an exploratory 20-origin rolling comparison. The integrated Stage 3
evaluation selects one HK order using data at its first origin and
refits that fixed order at each subsequent origin.
For Stage 2, Keir selected daily DXY,
nominal 10-year Treasury yield, S&P 500 returns, VIX, 10-year breakeven
inflation, WTI spot, and optional gold-futures volume. The pipeline is
implemented in `gold_vol_pipeline.predictors`; it derives close-to-close
log-returns for the S&P 500. An optional LSEG importer can prepare a separate
dataset from `data/refinitiv/`; the available export includes a partial
predictor panel plus gold-option quotes, while breakeven inflation, WTI, spot,
and gold volume are unavailable in that snapshot. It does not replace the
Yahoo/FRED panel or the current seven-feature model input.
`notebooks/05_advanced_models.ipynb` provides an exploratory all-predictor
gradient-boosting forecast with a training-only HAC-robust coefficient
significance screen. In the current snapshot, only VIX remains significant
after Holm correction, and the gradient-boosting model underperforms the
expanding-variance benchmark on the holdout; no final multivariate model or
variable-selection decision is recorded. On the separate LSEG `GCc1` panel,
the training-only HAC screen found DXY and VIX individually significant after
Holm correction; the gold implied-volatility tenors were significant jointly
but highly collinear (maximum pairwise correlation 0.991), so their individual
importance is unresolved. The LSEG panel is partial, and its quote units and
one-weekday availability proxy require verification. These association results
do not establish out-of-sample predictive value or a final variable-selection
decision. Central-bank demand remains deferred.

The integrated Stage 3 evaluation in `notebooks/06_final_evaluation.ipynb`
refits auto-ARIMA, VAR, GJR-GARCH, asymmetric EGARCH, Student-t GARCH,
VIX-GARCH-X, and quantile GARCH on the 445 weekly origins, alongside 18 cached
baseline and predictor-model forecasts. Two quantile-GARCH fits failed; the
25-model common-origin comparison uses 443 observations per model. The
provisional recommendation is **GARCH-X with lagged VIX**, using MSE as the
primary criterion: MSE is 9.492e-7 (RMSE 0.000974), lowest in that sample.
Asymmetric EGARCH has the lowest MAE and QLIKE; no candidate significantly
beats historical variance after Holm adjustment. GARCH-X holds VIX fixed at
its origin value during the forecast horizon. This is an exploratory
recommendation, not a confirmed team decision; optimizer warnings, failed
fits, source timing and overlapping targets limit the evidence. See the
[Stage 2–3 report](reports/model-performance-stage2-stage3.qmd) for the
equations, full comparison and limitations, and [`REPRODUCE.md`](REPRODUCE.md)
for the integrated workflow.

For AI-agent collaboration guidance, see [`AGENTS.md`](AGENTS.md) and [`.github/copilot-instructions.md`](.github/copilot-instructions.md). Shared AI/team decisions and handoffs are recorded in [`.ai/SESSION_LOG.md`](.ai/SESSION_LOG.md), using the protocol in [`.ai/README.md`](.ai/README.md). The project uses a main-only Git workflow described in [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md). For a compact project context and pointers, see [`llms.txt`](llms.txt).

The presentation-ready progress tracker is [`presentations/project-progress.qmd`](presentations/project-progress.qmd). It summarizes stage status and reads Git submission history when rendered. The model-development report is available as a [Beamer PDF](presentations/iteration-1-summary.pdf), with its [Quarto source](presentations/iteration-1-summary.qmd). Fetch the latest `main` with `git fetch origin main` before rendering updated presentations.

The draft [Stage 2 and Stage 3 model-performance report](reports/model-performance-stage2-stage3.qmd) summarizes baseline errors, predictor screens, and advanced-model comparisons with tables and figures; its rendered PDF is available alongside the source.
