"""Training-only predictor screen for next-observation Yang-Zhang volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

PREDICTORS = (
    "dxy_close",
    "sp500_log_return",
    "vix_close",
    "treasury_10y_yield",
    "gold_iv_1m",
    "gold_iv_2m",
    "gold_iv_3m",
    "gold_iv_6m",
    "gold_iv_1y",
    "gold_1m_risk_reversal",
    "gold_1m_butterfly",
)


def build_training_screen(
    targets: pd.DataFrame, predictors: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pair training targets with preceding valid origins and as-of predictors."""
    required_target = {"date", "daily_volatility", "split"}
    required_predictors = {"date", *PREDICTORS}
    for frame, required, label in (
        (targets, required_target, "target"),
        (predictors, required_predictors, "predictor"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} data missing columns: {missing}")
    valid = targets.copy()
    valid["date"] = pd.to_datetime(valid["date"], errors="raise")
    if valid["date"].isna().any() or valid["date"].duplicated().any():
        raise ValueError("Target dates must be nonmissing and unique.")
    valid = valid.sort_values("date").dropna(subset=["daily_volatility"])
    if not valid["split"].isin(["train", "test"]).all():
        raise ValueError("Every available target must have a train or test label.")
    if not valid["split"].eq("train").any() or not valid["split"].eq("test").any():
        raise ValueError("Target data must contain training and test observations.")
    first_test = valid.loc[valid["split"].eq("test"), "date"].min()
    if (valid.loc[valid["split"].eq("train"), "date"] >= first_test).any():
        raise ValueError("The training split must precede the test split.")

    train = valid.loc[valid["split"].eq("train"), ["date", "daily_volatility"]].copy()
    if len(train) < 3:
        raise ValueError("At least three training targets are required.")
    pairs = pd.DataFrame(
        {
            "origin_date": train["date"].iloc[:-1].to_numpy(),
            "target_date": train["date"].iloc[1:].to_numpy(),
            "origin_volatility": train["daily_volatility"].iloc[:-1].to_numpy(),
            "next_volatility": train["daily_volatility"].iloc[1:].to_numpy(),
        }
    )
    panel = predictors.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    if panel["date"].isna().any() or panel["date"].duplicated().any():
        raise ValueError("Predictor availability dates must be nonmissing and unique.")
    observation_columns = [f"{name}_observation_date" for name in PREDICTORS]
    missing_observation_columns = sorted(set(observation_columns) - set(panel.columns))
    if missing_observation_columns:
        raise ValueError(f"Predictor observation dates missing: {missing_observation_columns}")
    for column in observation_columns:
        panel[column] = pd.to_datetime(panel[column], errors="raise")
    panel = panel[["date", *PREDICTORS, *observation_columns]].rename(
        columns={"date": "panel_available_date"}
    )
    aligned = pd.merge_asof(
        pairs.sort_values("origin_date"),
        panel.sort_values("panel_available_date"),
        left_on="origin_date",
        right_on="panel_available_date",
        direction="backward",
    )
    numerical = ["origin_volatility", "next_volatility", *PREDICTORS]
    for column in numerical:
        aligned[column] = pd.to_numeric(aligned[column], errors="raise")
    finite = np.isfinite(aligned[numerical].to_numpy(dtype=float)).all(axis=1)
    observed = aligned[observation_columns].notna().all(axis=1).to_numpy()
    dropped = int((~(finite & observed)).sum())
    aligned = aligned.loc[finite & observed].copy()
    for column in observation_columns:
        availability = aligned[column] + pd.offsets.BDay(1)
        if (availability > aligned["origin_date"]).any():
            raise ValueError(f"{column} is not available at the forecast origin.")
    if len(aligned) <= len(PREDICTORS) + 2:
        raise ValueError("Too few complete training pairs for the predictor screen.")
    if (aligned["target_date"] >= first_test).any():
        raise ValueError("A screen response reaches the holdout.")
    metadata: dict[str, object] = {
        "first_test_target_date": first_test.date().isoformat(),
        "training_target_count": len(train),
        "candidate_training_pairs": len(pairs),
        "complete_training_pairs": len(aligned),
        "incomplete_training_pairs_dropped": dropped,
        "first_origin_date": aligned["origin_date"].min().date().isoformat(),
        "last_origin_date": aligned["origin_date"].max().date().isoformat(),
        "last_response_date": aligned["target_date"].max().date().isoformat(),
        "availability": "latest available at origin; observation date plus one weekday",
    }
    return aligned.reset_index(drop=True), metadata


def select_predictors_by_vif(
    screen: pd.DataFrame, *, threshold: float = 5.0
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Remove highest-VIF candidates while retaining the volatility control."""
    if not np.isfinite(threshold) or threshold <= 1:
        raise ValueError("VIF threshold must be finite and greater than 1.")
    columns = ["origin_volatility", *PREDICTORS]
    missing = sorted(set(columns) - set(screen.columns))
    if missing:
        raise ValueError(f"Screen data missing columns: {missing}")
    x = screen[columns].apply(pd.to_numeric, errors="raise")
    if len(x) <= len(columns) + 1 or not np.isfinite(x.to_numpy()).all():
        raise ValueError("VIF data must be finite and have enough observations.")
    scales = x.std(ddof=0)
    if (scales <= 0).any():
        raise ValueError(f"Constant screen predictors: {scales.index[scales <= 0].tolist()}")
    x = (x - x.mean()) / scales
    remaining = list(PREDICTORS)
    records: list[dict[str, object]] = []
    iteration = 1
    while True:
        active = ["origin_volatility", *remaining]
        vifs: dict[str, float] = {}
        for column in active:
            others = [name for name in active if name != column]
            y = x[column].to_numpy()
            design = np.column_stack((np.ones(len(x)), x[others].to_numpy()))
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coefficients
            unexplained = float(residual @ residual / (y @ y))
            vifs[column] = float("inf") if unexplained <= 1e-12 else 1.0 / unexplained
        max_vif = max(vifs.values())
        removed = ""
        if max_vif > threshold:
            if not remaining:
                raise ValueError("VIF threshold cannot be met while retaining the control.")
            removed = max(remaining, key=lambda name: vifs[name])
        for name, value in vifs.items():
            records.append({
                "iteration": iteration,
                "variable": name,
                "vif": value,
                "removed": name == removed,
            })
        if not removed:
            break
        remaining.remove(removed)
        iteration += 1
    if not remaining:
        raise ValueError("VIF screening removed every candidate predictor.")
    return tuple(remaining), pd.DataFrame(records)


def test_predictor_significance(
    screen: pd.DataFrame, *, hac_lags: int = 4,
    predictors: tuple[str, ...] = PREDICTORS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate a joint OLS screen with a volatility control and Holm correction."""
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a nonnegative integer.")
    if not predictors or len(set(predictors)) != len(predictors) or not set(predictors) <= set(PREDICTORS):
        raise ValueError("Predictors must be a nonempty, unique subset of the candidates.")
    columns = ["origin_volatility", *predictors]
    missing = sorted(set(columns + ["next_volatility"]) - set(screen.columns))
    if missing:
        raise ValueError(f"Screen data missing columns: {missing}")
    x = screen[columns].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(screen["next_volatility"], errors="raise")
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Screen variables must be finite.")
    scales = x.std(ddof=0)
    if (scales <= 0).any():
        raise ValueError(f"Constant screen predictors: {scales.index[scales <= 0].tolist()}")
    standardized = (x - x.mean()) / scales
    design = sm.add_constant(standardized, has_constant="add")
    if len(design) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Screen design has insufficient observations or deficient rank.")
    fit = sm.OLS(y, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}, use_t=True
    )
    pvalues = fit.pvalues.loc[list(predictors)]
    if not np.isfinite(pvalues.to_numpy()).all():
        raise ValueError("Predictor inference produced non-finite p-values.")
    result = pd.DataFrame(
        {
            "predictor": predictors,
            "coefficient_per_training_sd": fit.params.loc[list(predictors)].to_numpy(),
            "hac_standard_error": fit.bse.loc[list(predictors)].to_numpy(),
            "hac_t_statistic": fit.tvalues.loc[list(predictors)].to_numpy(),
            "p_value": pvalues.to_numpy(),
            "holm_p_value": multipletests(pvalues, method="holm")[1],
        }
    )
    result["significant_5pct"] = result["holm_p_value"] <= 0.05
    result = result.sort_values("holm_p_value", kind="stable").reset_index(drop=True)
    metadata = {
        "response": "next valid Yang-Zhang daily volatility (unannualized)",
        "control": "current Yang-Zhang daily volatility",
        "model": "OLS with intercept; training-standardized predictors and control",
        "hac_maxlags": hac_lags,
        "multiple_testing": f"Holm across {len(predictors)} retained predictor coefficients, excluding control",
        "tested_predictors": list(predictors),
        "alpha": 0.05,
        "training_r_squared": float(fit.rsquared),
        "design_condition_number": float(np.linalg.cond(design)),
    }
    return result, metadata
