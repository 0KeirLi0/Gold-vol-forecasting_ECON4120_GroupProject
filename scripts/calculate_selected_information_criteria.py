"""Calculate training-origin information criteria for selected models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from gold_vol_pipeline.volatility_forecasting import (
    SELECTED_PREDICTORS,
    _lagged_design,
    fit_conditional_variance,
)


def _row(
    model: str,
    *,
    n: int | None,
    log_likelihood: float | None,
    parameters: int | None,
    aic: float | None,
    bic: float | None,
    method: str,
) -> dict[str, object]:
    return {
        "model": model,
        "n_training_observations": n,
        "log_likelihood": log_likelihood,
        "n_parameters": parameters,
        "AIC": aic,
        "BIC": bic,
        "method": method,
    }


def _arch_row(
    model: str, values: np.ndarray, *, x: np.ndarray | None = None
) -> dict[str, object]:
    fit = arch_model(
        values * 100,
        mean="ARX" if x is not None else "AR",
        lags=1,
        x=x,
        vol="EGARCH" if model in {"egarch", "egarchx"} else "GARCH",
        p=1,
        o=1 if model in {"egarch", "egarchx"} else 0,
        q=1,
        rescale=False,
    ).fit(disp="off", show_warning=False)
    if int(getattr(fit, "convergence_flag", 1)) != 0:
        raise RuntimeError(f"{model} fit did not converge.")
    return _row(
        model,
        n=len(values),
        log_likelihood=float(fit.loglikelihood),
        parameters=len(fit.params),
        aic=float(fit.aic),
        bic=float(fit.bic),
        method="arch native likelihood",
    )


def _harx_row(history: pd.DataFrame) -> dict[str, object]:
    features, targets, _ = _lagged_design(history, include_ohlc=True)
    design = np.column_stack((np.ones(len(features)), features))
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    residuals = targets - design @ coefficients
    variance = float(np.mean(np.square(residuals)))
    log_likelihood = -0.5 * len(residuals) * (
        np.log(2 * np.pi) + 1 + np.log(variance)
    )
    parameters = design.shape[1] + 1
    aic = 2 * parameters - 2 * log_likelihood
    bic = parameters * np.log(len(residuals)) - 2 * log_likelihood
    return _row(
        "harx_ohlc",
        n=len(residuals),
        log_likelihood=float(log_likelihood),
        parameters=parameters,
        aic=float(aic),
        bic=float(bic),
        method="Gaussian OLS likelihood on log-volatility target",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-input",
        type=Path,
        default=Path("data/processed/GCc1_yang_zhang_volatility.csv"),
    )
    parser.add_argument(
        "--egarchx-input",
        type=Path,
        default=Path(
            "data/processed/stage2_yz_egarchx/GCc1_yang_zhang_volatility.csv"
        ),
    )
    parser.add_argument(
        "--range-input",
        type=Path,
        default=Path(
            "data/processed/stage2_yz_selected/GCc1_yang_zhang_volatility.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/stage2_yz_diagnostics/model_residuals/"
            "information_criteria.csv"
        ),
    )
    args = parser.parse_args()
    baseline = pd.read_csv(args.baseline_input)
    egarchx = pd.read_csv(args.egarchx_input)
    range_data = pd.read_csv(args.range_input)
    baseline_train = baseline.loc[baseline["split"].eq("train")].copy()
    egarchx_train = egarchx.loc[egarchx["split"].eq("train")].copy()
    range_train = range_data.loc[range_data["split"].eq("train")].copy()
    values = pd.to_numeric(
        baseline_train["daily_volatility"], errors="raise"
    ).to_numpy(dtype=float)
    egarchx_values = pd.to_numeric(
        egarchx_train["daily_volatility"], errors="raise"
    ).to_numpy(dtype=float)
    rows = [
        _arch_row("garch11", values),
        _arch_row("egarch", values),
    ]
    features = egarchx_train.loc[:, SELECTED_PREDICTORS].to_numpy(dtype=float)
    features = features / features.mean(axis=0)
    lagged_features = features.copy()
    lagged_features[1:] = features[:-1]
    rows.append(_arch_row("egarchx", egarchx_values, x=lagged_features))
    rows.append(_harx_row(baseline_train))
    range_fit = fit_conditional_variance(
        range_train,
        tail_threshold=None,
        external_predictors=SELECTED_PREDICTORS,
    )
    range_aic = (
        2 * range_fit.n_parameters - 2 * range_fit.log_likelihood
    )
    range_bic = (
        range_fit.n_parameters * np.log(range_fit.n_observations)
        - 2 * range_fit.log_likelihood
    )
    rows.extend(
        [
            _row(
                "garchx_ohlc_range",
                n=range_fit.n_observations,
                log_likelihood=range_fit.log_likelihood,
                parameters=range_fit.n_parameters,
                aic=float(range_aic),
                bic=float(range_bic),
                method="custom constrained Gaussian quasi-likelihood",
            ),
            _row(
                "hist_gradient_boosting_ohlc",
                n=None,
                log_likelihood=None,
                parameters=None,
                aic=None,
                bic=None,
                method="not applicable: no specified likelihood",
            ),
        ]
    )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
