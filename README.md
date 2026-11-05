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

The default series is Yahoo Finance's `GC=F`, a COMEX gold futures continuous series (typically a front-month series), not spot gold or a single fixed-expiry contract. Its returns may reflect contract rolls and futures-market effects such as contango or backwardation; record this when interpreting results. The pipeline computes close-to-close log returns and a forward target over the next five observed price sessions by default. The default target is realized variance, calculated as the sum of squared daily log returns; `realized_volatility` is also available as the square root of that sum. Horizon and target are configurable. The target uses future observations only as the label for each forecast origin; predictors in this baseline contain only information available through that origin.

Yahoo Finance's `XAUUSD=X` spot ticker returned “No data found” when checked on 2026-11-06. The project therefore uses the available `GC=F` futures series as its default, with the instrument change made explicit. If the project later needs spot, select a provider and use a spot CSV; do not combine spot and futures observations in one series. The CSV import accepts explicit date and close column names and records source/instrument metadata.

Raw prices and prepared outputs are written under `data/` and are intentionally excluded from Git. Preserve these files for each analysis run and record the date range and pipeline options used. The prepared CSV includes the forecast-origin date, close, log return, target window dates, target value, and provider/instrument metadata. For reproducible comparisons, retain the exact raw input used rather than relying only on redownloading a potentially revised provider series.

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

Open [`notebooks/01_inspect_gold_data.ipynb`](notebooks/01_inspect_gold_data.ipynb). The notebook lets you set the date range, horizon, and target; inspect and plot raw/processed futures data; check target dates and values; and save run-specific CSVs with a JSON manifest.

Prepare a gold-price CSV with `date` and `close` columns:

```bash
python -m gold_vol_pipeline --input data/raw/gold_futures.csv \
  --source "Provider name" --instrument "COMEX Gold Futures (continuous series)" \
  --currency USD --price-unit "USD per troy ounce"
```

To download the default Yahoo continuous futures series directly, use:

```bash
python -m gold_vol_pipeline --start 2015-01-01 --end 2025-01-01
```

For alternate CSV headers, set `--date-column` and `--close-column`. The original input remains untouched; prepared data are saved to `data/processed/gold_futures.csv`. Override output paths with `--raw-output` and `--output`, and target settings with `--horizon` or `--target realized_variance|realized_volatility`. Yahoo downloads use `GC=F` and require `--start YYYY-MM-DD --end YYYY-MM-DD`; the end date is exclusive.

Run the full test suite with `python -m pytest`. Run one test with:

```bash
python -m pytest tests/test_pipeline.py::test_default_target_is_forward_five_session_realized_variance
```

There is no configured linter yet.

As the project develops, record enough information to reproduce and interpret each iteration: data inputs and transformations, model specification, evaluation evidence, selection criteria, limitations, and the rationale for the next step. Do not publish sensitive or restricted data; document its provenance and access requirements instead.

## Repository status

The repository now includes a Python data-preparation module and focused tests. Model training, model comparison, and project-specific forecast evaluation are not implemented yet.

For AI-agent collaboration guidance, see [`AGENTS.md`](AGENTS.md) and [`.github/copilot-instructions.md`](.github/copilot-instructions.md). Shared AI/team decisions and handoffs are recorded in [`.ai/SESSION_LOG.md`](.ai/SESSION_LOG.md), using the protocol in [`.ai/README.md`](.ai/README.md). For the team’s branch, pull request, and merge process, see [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md). For a compact project context and pointers, see [`llms.txt`](llms.txt).
