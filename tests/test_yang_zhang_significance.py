import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gold_vol_pipeline.yang_zhang_significance import (
    PREDICTORS,
    build_training_screen,
    select_predictors_by_vif,
    test_predictor_significance as run_significance,
)


def _sample():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    targets = pd.DataFrame(
        {
            "date": dates,
            "daily_volatility": rng.uniform(0.005, 0.02, len(dates)),
            "split": ["train"] * 80 + ["test"] * 20,
        }
    )
    targets.loc[20, "daily_volatility"] = np.nan
    predictors = pd.DataFrame({"date": dates})
    for name in PREDICTORS:
        predictors[name] = rng.normal(size=len(dates))
        predictors[f"{name}_observation_date"] = dates - pd.offsets.BDay(1)
    return targets, predictors


def test_training_pairs_use_previous_valid_target_and_exclude_holdout():
    targets, predictors = _sample()
    screen, metadata = build_training_screen(targets, predictors)
    assert len(screen) == 78
    assert metadata["first_test_target_date"] == "2020-04-22"
    assert screen["target_date"].max() < pd.Timestamp(metadata["first_test_target_date"])
    row = screen.loc[screen["target_date"].eq(targets.loc[21, "date"])].iloc[0]
    assert row["origin_date"] == targets.loc[19, "date"]
    assert row["origin_volatility"] == targets.loc[19, "daily_volatility"]
    assert row["vix_close"] == predictors.loc[19, "vix_close"]


def test_holdout_values_do_not_change_inference():
    targets, predictors = _sample()
    original, _ = build_training_screen(targets, predictors)
    changed = targets.copy()
    changed.loc[changed["split"].eq("test"), "daily_volatility"] = 1_000.0
    predictors.loc[predictors["date"] >= changed.loc[80, "date"], list(PREDICTORS)] = 999.0
    altered, _ = build_training_screen(changed, predictors)
    pd.testing.assert_frame_equal(original, altered)
    pd.testing.assert_frame_equal(run_significance(original)[0], run_significance(altered)[0])
    selected, history = select_predictors_by_vif(original)
    altered_selected, altered_history = select_predictors_by_vif(altered)
    assert selected == altered_selected
    pd.testing.assert_frame_equal(history, altered_history)


def test_screen_matches_hac_ols_and_holm_calculation():
    targets, predictors = _sample()
    screen, _ = build_training_screen(targets, predictors)
    results, metadata = run_significance(screen)
    columns = ["origin_volatility", *PREDICTORS]
    x = screen[columns]
    x = (x - x.mean()) / x.std(ddof=0)
    fit = sm.OLS(screen["next_volatility"], sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4}, use_t=True
    )
    expected = multipletests(fit.pvalues.loc[list(PREDICTORS)], method="holm")[1]
    expected_by_name = dict(zip(PREDICTORS, expected, strict=True))
    for row in results.itertuples():
        assert row.holm_p_value == pytest.approx(expected_by_name[row.predictor])
        assert row.coefficient_per_training_sd == pytest.approx(fit.params[row.predictor])
    assert metadata["hac_maxlags"] == 4
    assert len(results) == 11


def test_screen_rejects_predictor_from_future():
    targets, predictors = _sample()
    predictors["vix_close_observation_date"] = predictors["date"]
    with pytest.raises(ValueError, match="not available"):
        build_training_screen(targets, predictors)


def test_screen_drops_missing_predictors_and_rejects_rank_deficiency():
    targets, predictors = _sample()
    predictors.loc[10, "vix_close"] = np.nan
    screen, metadata = build_training_screen(targets, predictors)
    assert metadata["incomplete_training_pairs_dropped"] == 1
    screen["vix_close"] = screen["dxy_close"]
    with pytest.raises(ValueError, match="deficient rank"):
        run_significance(screen)


def test_vif_iteratively_removes_collinear_predictors_and_keeps_control():
    targets, predictors = _sample()
    screen, _ = build_training_screen(targets, predictors)
    screen["gold_iv_2m"] = screen["gold_iv_1m"]
    retained, history = select_predictors_by_vif(screen, threshold=10)
    assert "gold_iv_1m" not in retained
    assert "gold_iv_2m" in retained
    assert history.loc[history["removed"], "variable"].tolist() == ["gold_iv_1m"]
    assert np.isinf(history.loc[history["removed"], "vif"].iloc[0])
    final = history.loc[history["iteration"].eq(history["iteration"].max())]
    assert "origin_volatility" in final["variable"].tolist()
    assert (final["vif"] <= 10).all()
    results, metadata = run_significance(screen, predictors=retained)
    assert set(results["predictor"]) == set(retained)
    assert metadata["tested_predictors"] == list(retained)


def test_vif_removes_candidate_collinear_with_control_and_ignores_response():
    targets, predictors = _sample()
    screen, _ = build_training_screen(targets, predictors)
    screen["dxy_close"] = screen["origin_volatility"]
    retained, history = select_predictors_by_vif(screen)
    assert history.loc[history["removed"], "variable"].tolist() == ["dxy_close"]
    assert "dxy_close" not in retained
    screen["next_volatility"] = np.arange(len(screen))
    changed_retained, changed_history = select_predictors_by_vif(screen)
    assert retained == changed_retained
    pd.testing.assert_frame_equal(history, changed_history)


def test_vif_rejects_invalid_threshold_and_constant_columns():
    targets, predictors = _sample()
    screen, _ = build_training_screen(targets, predictors)
    with pytest.raises(ValueError, match="threshold"):
        select_predictors_by_vif(screen, threshold=1)
    screen["vix_close"] = 1.0
    with pytest.raises(ValueError, match="Constant"):
        select_predictors_by_vif(screen)


def test_default_cutoff_is_five_not_ten():
    targets, predictors = _sample()
    screen, _ = build_training_screen(targets, predictors)
    screen["gold_iv_2m"] = (
        screen["gold_iv_1m"] + 0.4 * screen["gold_iv_2m"]
    )
    strict, history = select_predictors_by_vif(screen)
    loose, _ = select_predictors_by_vif(screen, threshold=10)
    assert len(strict) == len(loose) - 1
    assert history.loc[history["iteration"].eq(history["iteration"].max()), "vif"].max() <= 5


def test_vif_matches_statsmodels_and_selected_holm_values():
    targets, predictors = _sample()
    screen, _ = build_training_screen(targets, predictors)
    retained, history = select_predictors_by_vif(screen)
    initial = history.loc[history["iteration"].eq(1)].set_index("variable")
    columns = ["origin_volatility", *PREDICTORS]
    design = sm.add_constant(screen[columns])
    for i, column in enumerate(columns, start=1):
        assert initial.loc[column, "vif"] == pytest.approx(
            variance_inflation_factor(design.to_numpy(), i)
        )
    results, _ = run_significance(screen, predictors=retained)
    x = screen[["origin_volatility", *retained]]
    x = (x - x.mean()) / x.std(ddof=0)
    fit = sm.OLS(screen["next_volatility"], sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4}, use_t=True
    )
    adjusted = multipletests(fit.pvalues.loc[list(retained)], method="holm")[1]
    expected = dict(zip(retained, adjusted, strict=True))
    for row in results.itertuples():
        assert row.holm_p_value == pytest.approx(expected[row.predictor])
