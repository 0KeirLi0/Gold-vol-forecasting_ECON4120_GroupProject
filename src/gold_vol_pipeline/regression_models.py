"""Legacy HAR, exogenous, penalized, and VAR realized-variance forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

EPSILON = 1e-12
PREDICTOR_COLUMNS = (
    "dxy_close",
    "sp500_log_return",
    "vix_close",
    "treasury_10y_yield",
    "breakeven_inflation_10y",
    "wti_spot",
    "gold_volume",
)


@dataclass(frozen=True)
class PenalizedForecast:
    """Penalized forecast and training-window selection details."""

    forecast: float
    alpha: float
    coefficients: pd.Series


def _validated_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of observations.")
    return horizon


def _validated_returns(
    returns: Sequence[float] | pd.Series,
) -> pd.Series:
    try:
        series = returns.copy() if isinstance(returns, pd.Series) else pd.Series(returns)
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("returns must contain finite numeric values.") from error
    if values.ndim != 1 or values.size < 1:
        raise ValueError("returns must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain finite numeric values.")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("returns must have a unique, increasing index.")
    series = pd.Series(values, index=series.index, name="log_return")
    return series


def _validated_predictors(
    predictors: pd.DataFrame,
    index: pd.Index,
    columns: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(predictors, pd.DataFrame):
        raise ValueError("predictors must be a pandas DataFrame.")
    missing = sorted(set(columns) - set(predictors.columns))
    if missing:
        raise ValueError(f"predictors are missing required columns: {missing}")
    if not predictors.index.equals(index):
        raise ValueError("predictors must have the same index as returns.")
    frame = predictors.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictor values must be finite numeric values.")
    return frame


def har_features(
    returns: Sequence[float] | pd.Series,
    *,
    horizon: int,
    log_space: bool = True,
) -> pd.DataFrame:
    """Build daily, weekly, and monthly HAR features and aligned forward targets.

    Rows are indexed by forecast origin. ``target_end_date`` identifies the last
    return used by each forward target; targets extending beyond the supplied
    returns are missing.
    """
    series = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if not isinstance(log_space, bool):
        raise ValueError("log_space must be a boolean.")

    squared = np.square(series.to_numpy(dtype=float))
    features = pd.DataFrame(
        {
            "rv_d": squared,
            "rv_w": pd.Series(squared, index=series.index).rolling(5).mean(),
            "rv_m": pd.Series(squared, index=series.index).rolling(22).mean(),
        },
        index=series.index,
    )
    if log_space:
        features = np.log(features + EPSILON)

    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    targets = np.full(len(series), np.nan, dtype=float)
    end_dates: list[object] = [pd.NaT] * len(series)
    for origin in range(len(series) - horizon):
        targets[origin] = (
            cumulative[origin + horizon + 1] - cumulative[origin + 1]
        )
        end_dates[origin] = series.index[origin + horizon]
    features["target"] = targets
    features["target_end_date"] = end_dates
    return features


def fit_har_coefficients(
    features: pd.DataFrame,
    target: Sequence[float] | pd.Series,
    *,
    log_space: bool = True,
) -> pd.Series:
    """Fit HAR coefficients from complete RV_d, RV_w, RV_m features and targets."""
    required = ("rv_d", "rv_w", "rv_m")
    if not isinstance(features, pd.DataFrame) or not set(required).issubset(
        features.columns
    ):
        raise ValueError(f"features must include columns {list(required)}.")
    if not isinstance(log_space, bool):
        raise ValueError("log_space must be a boolean.")
    try:
        y = np.asarray(target, dtype=float)
        x = features.loc[:, list(required)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("features and target must contain finite numeric values.") from error
    if (
        y.ndim != 1
        or len(y) != len(x)
        or len(y) <= len(required) + 1
        or not np.isfinite(y).all()
        or not np.isfinite(x).all()
    ):
        raise ValueError("features and target must be aligned with enough finite rows.")
    if log_space:
        if (y < 0).any():
            raise ValueError("variance targets must be non-negative.")
        y = np.log(y + EPSILON)
    fit = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
    return pd.Series(fit.params, index=("const", *required), dtype=float)


def _forecast_design(
    returns: pd.Series,
    horizon: int,
    *,
    log_space: bool,
    exog: Sequence[str] = (),
    predictors: pd.DataFrame | None = None,
    include_lags: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    features = har_features(returns, horizon=horizon, log_space=log_space)
    design = features[["rv_d", "rv_w", "rv_m"]].copy()
    if exog:
        if predictors is None:
            raise ValueError("predictors are required when exogenous columns are used.")
        exogenous = _validated_predictors(predictors, returns.index, exog)
        design = design.join(exogenous)
        if include_lags:
            lagged = exogenous.shift(1)
            lagged.columns = [f"{column}_lag1" for column in exog]
            design = design.join(lagged)

    origin = returns.index[-1]
    end_dates = features["target_end_date"]
    if isinstance(returns.index, pd.DatetimeIndex):
        available_labels = pd.to_datetime(end_dates, errors="coerce")
        observed_origin = pd.Timestamp(origin)
        train_mask = available_labels.le(observed_origin)
    else:
        train_mask = pd.Series(
            [position + horizon < len(returns) for position in range(len(returns))],
            index=returns.index,
        )
    train_mask &= features["target"].notna()
    train_mask &= design.notna().all(axis=1)
    forecast_mask = design.notna().all(axis=1)
    train_x = design.loc[train_mask]
    train_y = features.loc[train_mask, "target"]
    if log_space:
        train_y = np.log(train_y + EPSILON)
    forecast_x = design.loc[forecast_mask].iloc[[-1]]
    residual_scale = pd.Series(features.loc[train_mask, "target"], index=train_x.index)
    return train_x, train_y, forecast_x, residual_scale


def _linear_forecast(
    returns: pd.Series,
    *,
    horizon: int,
    log_space: bool,
    exog: Sequence[str] = (),
    predictors: pd.DataFrame | None = None,
) -> float:
    train_x, train_y, forecast_x, _ = _forecast_design(
        returns,
        horizon,
        log_space=log_space,
        exog=exog,
        predictors=predictors,
    )
    if len(train_x) <= train_x.shape[1]:
        raise ValueError("Not enough complete, purged observations to fit the model.")
    train_design = sm.add_constant(train_x, has_constant="add")
    forecast_design = sm.add_constant(forecast_x, has_constant="add")
    fit = sm.OLS(train_y, train_design).fit()
    predicted = float(fit.predict(forecast_design).iloc[0])
    if log_space:
        residual_variance = float(np.var(fit.resid, ddof=train_design.shape[1]))
        predicted = float(np.exp(predicted + 0.5 * max(residual_variance, 0.0)))
    return max(predicted, 0.0) if np.isfinite(predicted) else float("nan")


def har_forecast(
    returns: Sequence[float] | pd.Series,
    *,
    horizon: int,
    log_space: bool = True,
) -> float:
    """Fit a HAR-RV regression using only targets observable at the origin."""
    series = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    return _linear_forecast(series, horizon=horizon, log_space=log_space)


def harx_forecast(
    returns: Sequence[float] | pd.Series,
    predictors: pd.DataFrame,
    *,
    horizon: int,
    exog: Sequence[str] = ("vix_close",),
    log_space: bool = True,
) -> float:
    """Fit HAR-RV with predictors observed on or before each forecast origin."""
    series = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if not exog or len(set(exog)) != len(exog):
        raise ValueError("exog must contain unique predictor column names.")
    return _linear_forecast(
        series,
        horizon=horizon,
        log_space=log_space,
        exog=exog,
        predictors=predictors,
    )


def penalized_forecast(
    returns: Sequence[float] | pd.Series,
    predictors: pd.DataFrame,
    *,
    horizon: int,
    method: str = "lasso",
    exog: Sequence[str] = PREDICTOR_COLUMNS,
) -> PenalizedForecast:
    """Tune a standardized HAR-plus-predictor Lasso or Ridge using purged CV."""
    series = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if method not in {"lasso", "ridge"}:
        raise ValueError("method must be 'lasso' or 'ridge'.")
    if not exog or len(set(exog)) != len(exog):
        raise ValueError("exog must contain unique predictor column names.")
    train_x, train_y, forecast_x, _ = _forecast_design(
        series,
        horizon,
        log_space=True,
        exog=exog,
        predictors=predictors,
        include_lags=True,
    )
    if len(train_x) <= horizon + 2:
        raise ValueError("Not enough complete, purged observations for time-series CV.")
    splits = min(5, len(train_x) - horizon - 1)
    if splits < 2:
        raise ValueError("Not enough observations for TimeSeriesSplit cross-validation.")
    estimator = Lasso(max_iter=50_000, tol=1e-5) if method == "lasso" else Ridge()
    pipeline = make_pipeline(StandardScaler(), estimator)
    model_name = "lasso" if method == "lasso" else "ridge"
    search = GridSearchCV(
        pipeline,
        {f"{model_name}__alpha": np.logspace(-5, 2, 30)},
        cv=TimeSeriesSplit(n_splits=splits, gap=horizon),
        scoring="neg_mean_squared_error",
        error_score="raise",
    )
    search.fit(train_x, train_y)
    prediction = float(search.predict(forecast_x)[0])
    residuals = train_y.to_numpy(dtype=float) - search.predict(train_x)
    residual_variance = float(np.var(residuals, ddof=1))
    if not np.isfinite(residual_variance):
        raise RuntimeError("Penalized regression residual variance is invalid.")
    coefficients = pd.Series(
        search.best_estimator_.named_steps[model_name].coef_,
        index=train_x.columns,
        dtype=float,
    )
    coefficients = coefficients.loc[coefficients.ne(0.0)].sort_values(
        key=np.abs, ascending=False
    )
    forecast = float(np.exp(prediction + 0.5 * max(residual_variance, 0.0)))
    return PenalizedForecast(
        forecast=max(forecast, 0.0),
        alpha=float(search.best_params_[f"{model_name}__alpha"]),
        coefficients=coefficients,
    )


def var_forecast(
    returns: Sequence[float] | pd.Series,
    vix_close: Sequence[float] | pd.Series,
    *,
    horizon: int,
    max_lags: int = 5,
) -> float:
    """Optionally fit a small BIC-selected VAR and sum its daily variance forecasts."""
    series = _validated_returns(returns)
    horizon = _validated_horizon(horizon)
    if isinstance(max_lags, bool) or not isinstance(max_lags, int) or not 1 <= max_lags <= 5:
        raise ValueError("max_lags must be an integer from 1 to 5.")
    try:
        vix = vix_close.copy() if isinstance(vix_close, pd.Series) else pd.Series(vix_close)
        vix_values = pd.to_numeric(vix, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("vix_close must contain finite numeric values.") from error
    if len(vix_values) != len(series) or not np.isfinite(vix_values).all():
        raise ValueError("vix_close must be finite and aligned with returns.")
    if isinstance(vix_close, pd.Series) and not vix_close.index.equals(series.index):
        raise ValueError("vix_close must have the same index as returns.")
    if (vix_values <= 0).any():
        raise ValueError("vix_close values must be strictly positive.")
    daily_variance = np.square(series.to_numpy(dtype=float))
    if not np.any(daily_variance):
        return 0.0
    observations = pd.DataFrame(
        {
            "log_rv_d": np.log(daily_variance + EPSILON),
            "log_vix": np.log(vix_values),
        }
    )
    if len(observations) < 3 * max_lags + 5:
        raise ValueError("VAR forecasts require a longer aligned training history.")
    model = VAR(observations)
    selection = model.select_order(maxlags=max_lags)
    lag_order = selection.selected_orders["bic"]
    if lag_order is None:
        raise RuntimeError("BIC failed to select a VAR lag order.")
    lag_order = int(lag_order)
    if lag_order == 0:
        return float(
            horizon
            * np.exp(float(observations["log_rv_d"].mean()))
        )
    fit = model.fit(lag_order)
    forecast = fit.forecast(observations.to_numpy()[-lag_order:], steps=horizon)
    return float(np.exp(forecast[:, 0]).sum())
