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

The current working default is Yahoo Finance `GC=F`, a COMEX gold futures continuous series. It is not spot gold or a single fixed-expiry contract, and its history may reflect contract rolls. The CLI and notebook download daily data from Yahoo Finance by default; local CSV input is also supported. The team may revisit spot versus futures, but any change should be explicit and must not mix instrument types. Yahoo's `XAUUSD=X` spot ticker returned “No data found” when checked on 2026-11-06.

The pipeline computes close-to-close log returns and a forward target over the next five observed price sessions by default. The default target is realized variance, calculated as the sum of squared daily log returns; `realized_volatility` is also available as the square root of that sum. Horizon and target are configurable. The target uses future observations only as the label for each forecast origin; predictors in this baseline contain only information available through that origin.

Raw and prepared downloads are written under `data/` and excluded from Git. Preserve them for each analysis run and record their provenance, observation date range, and pipeline options. The prepared CSV includes the forecast-origin date, close, log return, target-window dates, target value, and provider/instrument metadata. Do not mix futures and spot observations. Do not share Bloomberg or other restricted exports unless the applicable license permits it.

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
For example:

```python
from gold_vol_pipeline import baseline_forecasts, prepare_dataset

dataset = prepare_dataset(prices, horizon=5)
origin_returns = dataset["log_return"].dropna()
forecasts = baseline_forecasts(origin_returns, horizon=5)
```

Only pass returns known at the forecast origin. A single-origin forecast is not
a backtest or model-selection result. `prediction_history` and
[`notebooks/02_models.ipynb`](notebooks/02_models.ipynb) provide sequential
forecast/actual series and MAE/RMSE summaries for the selected origin window.
These are exploratory baseline comparisons, not a final model-selection result
or an evaluation on a team-confirmed instrument.

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

The notebook also runs a nonseasonal Hyndman-Khandakar stepwise ARIMA search
using KPSS differencing tests and AICc, saves raw and standardized residual
series under the git-ignored `data/processed/` directory, plots residual panels,
and applies Ljung-Box tests to residuals and squared standardized residuals.
The evaluation helpers are in
[`src/gold_vol_pipeline/model_evaluation.py`](src/gold_vol_pipeline/model_evaluation.py).

As the project develops, record enough information to reproduce and interpret each iteration: data inputs and transformations, model specification, evaluation evidence, selection criteria, limitations, and the rationale for the next step. Do not publish sensitive or restricted data; document its provenance and access requirements instead.

## Repository status

The repository includes a Python data-preparation module, return-only candidate
forecasts, sequential forecast histories, exploratory model diagnostics, and
configurable in-sample candidate comparisons. The team has not yet recorded an
interpretation or model-selection decision based on these tools.

For AI-agent collaboration guidance, see [`AGENTS.md`](AGENTS.md) and [`.github/copilot-instructions.md`](.github/copilot-instructions.md). Shared AI/team decisions and handoffs are recorded in [`.ai/SESSION_LOG.md`](.ai/SESSION_LOG.md), using the protocol in [`.ai/README.md`](.ai/README.md). The project uses a main-only Git workflow described in [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md). For a compact project context and pointers, see [`llms.txt`](llms.txt).

The presentation-ready progress tracker is [`presentations/project-progress.qmd`](presentations/project-progress.qmd). It summarizes stage status and reads Git submission history when rendered. Fetch the latest `main` with `git fetch origin main`, then render with `quarto render presentations/project-progress.qmd`.
