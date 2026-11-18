import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from gold_vol_pipeline.distribution_comparison import (
    _fit_hgb_locations,
    _fit_x_arch_model,
    _fit_zero_centered_density,
    _forecast_origin,
    _interval,
    _standard_logpdf,
    evaluate_distribution_comparison,
    save_distribution_comparison,
    summarize_distribution_comparison,
    summarize_forecast_quintiles,
)


def _prediction(forecast, distribution, *, density_space="level", scale=0.002):
    log_location = np.nan
    if density_space == "log":
        log_location = float(np.log(forecast))
        scale = 0.15
    return {
        "forecast": forecast,
        "scale": scale,
        "degrees_freedom": 6.0 if distribution == "student_t" else None,
        "density_space": density_space,
        "log_location": log_location,
    }


def test_standardized_log_density_matches_scipy():
    errors = np.array([-2.0, -0.2, 0.0, 1.3])
    np.testing.assert_allclose(_standard_logpdf(errors, "normal"), norm.logpdf(errors))
    degrees = 7.0
    expected = t.logpdf(errors, degrees, scale=np.sqrt((degrees - 2) / degrees))
    np.testing.assert_allclose(
        _standard_logpdf(errors, "student_t", degrees), expected, rtol=1e-10
    )


def test_normal_interval_uses_the_forecast_scale():
    row = pd.Series(
        {
            "distribution": "normal",
            "density_space": "level",
            "forecast": 0.01,
            "scale": 0.002,
            "degrees_freedom": np.nan,
            "actual": 0.01,
        }
    )
    lower, upper = _interval(row, 0.9)
    np.testing.assert_allclose(upper - lower, 2 * norm.ppf(0.95) * 0.002)
    assert lower < 0.01 < upper


def test_student_t_calibration_allows_heavy_tails():
    draws = t.rvs(
        5,
        scale=np.sqrt(3 / 5),
        size=800,
        random_state=7,
    )
    scale, degrees = _fit_zero_centered_density(draws, "student_t")
    assert 2.05 < degrees < 12
    assert 0.7 < scale < 1.3


def test_hgb_point_forecast_does_not_depend_on_the_density():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(90, 4))
    targets = features @ np.array([0.1, -0.2, 0.05, 0.0]) - 4.5
    targets += rng.normal(0, 0.05, len(targets))
    fitted = _fit_hgb_locations(features, targets, rng.normal(size=(1, 4)))
    assert fitted["normal"]["forecast"] == fitted["student_t"]["forecast"]
    assert fitted["normal"]["log_location"] == fitted["student_t"]["log_location"]
    assert fitted["student_t"]["degrees_freedom"] > 2
    assert fitted["normal"]["degrees_freedom"] is None


def test_one_failed_candidate_does_not_discard_the_others(monkeypatch):
    def fake_arch(history, model, distribution):
        if model == "egarch":
            raise RuntimeError("forced EGARCH failure")
        return _prediction(0.01, distribution)

    def fake_x(history, model, distribution):
        return _prediction(0.011, distribution)

    def fake_linear(features, targets, forecast_features, distribution):
        return _prediction(0.012, distribution, density_space="log")

    def fake_hgb(features, targets, forecast_features):
        shared = _prediction(0.013, "normal", density_space="log")
        student = _prediction(0.013, "student_t", density_space="log")
        student["log_location"] = shared["log_location"]
        return {"normal": shared, "student_t": student}

    monkeypatch.setattr(
        "gold_vol_pipeline.distribution_comparison._fit_arch_model", fake_arch
    )
    monkeypatch.setattr(
        "gold_vol_pipeline.distribution_comparison._fit_x_arch_model", fake_x
    )
    monkeypatch.setattr(
        "gold_vol_pipeline.distribution_comparison._fit_linear_location", fake_linear
    )
    monkeypatch.setattr(
        "gold_vol_pipeline.distribution_comparison._fit_hgb_locations", fake_hgb
    )
    dates = pd.bdate_range("2020-01-02", periods=40)
    frame = pd.DataFrame(
        {
            "date": dates,
            "daily_volatility": np.full(len(dates), 0.01),
            "close_return": 0.0,
            "overnight_log_return": 0.0,
            "open_close_log_return": 0.0,
            "high_low_log_range": 0.008,
            "dxy_close": 100.0,
            "gold_iv_1m": 15.0,
            "gold_iv_1y": 16.0,
        }
    )
    rows = _forecast_origin(
        (
            30,
            frame.iloc[:30].copy(),
            pd.Timestamp(frame.iloc[29]["date"]),
            pd.Timestamp(frame.iloc[30]["date"]),
            0.0105,
        )
    )
    by_key = {(row["model"], row["distribution"]): row for row in rows}
    assert len(by_key) == 12
    assert by_key[("egarch", "normal")]["converged"] is False
    assert "forced EGARCH failure" in by_key[("egarch", "student_t")]["error"]
    assert by_key[("garch", "normal")]["converged"] is True
    assert by_key[("garch_x", "student_t")]["converged"] is True
    assert by_key[("hgb", "normal")]["forecast"] == by_key[("hgb", "student_t")]["forecast"]


def test_garch_x_estimates_both_likelihoods():
    rng = np.random.default_rng(4)
    size = 140
    volatility = np.empty(size)
    volatility[0] = 0.008
    for position in range(1, size):
        shock = rng.standard_t(5) * np.sqrt(3 / 5)
        variance = 1e-6 + 0.08 * shock**2 * volatility[position - 1] ** 2
        variance += 0.85 * (volatility[position - 1] ** 2)
        volatility[position] = abs(0.002 + 0.4 * volatility[position - 1] + np.sqrt(variance) * shock)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2016-01-04", periods=size),
            "daily_volatility": volatility,
            "high_low_log_range": np.abs(rng.normal(0.008, 0.001, size)),
            "dxy_close": np.exp(rng.normal(4.6, 0.01, size)),
            "gold_iv_1m": np.exp(rng.normal(2.7, 0.02, size)),
            "gold_iv_1y": np.exp(rng.normal(2.8, 0.02, size)),
        }
    )
    cap = 10 * float(np.max(volatility))
    for model in ("garch_x", "egarch_x"):
        normal = _fit_x_arch_model(frame, model, "normal")
        student = _fit_x_arch_model(frame, model, "student_t")
        assert 0 < normal["forecast"] <= cap and 0 < normal["scale"] <= cap
        assert normal["degrees_freedom"] is None
        assert 0 < student["forecast"] <= cap and 0 < student["scale"] <= cap
        assert 2.05 <= student["degrees_freedom"] <= 100


def test_summary_uses_common_dates_and_counts_failures():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    rows = []
    for model, distribution in (
        ("garch", "normal"),
        ("garch", "student_t"),
    ):
        for date, actual, forecast, converged in (
            (dates[0], 0.010, 0.012, True),
            (dates[1], 0.011, 0.010, model != "garch" or distribution != "student_t"),
        ):
            rows.append(
                {
                    "model": model,
                    "distribution": distribution,
                    "target_date": date,
                    "actual": actual,
                    "forecast": forecast if converged else np.nan,
                    "negative_log_score": 1.5 if converged else np.nan,
                    "degrees_freedom": np.nan if distribution == "normal" else 6,
                    "interval_80_covered": True,
                    "interval_80_lower": 0.005,
                    "interval_80_upper": 0.015,
                    "interval_90_covered": True,
                    "interval_90_lower": 0.004,
                    "interval_90_upper": 0.016,
                    "interval_95_covered": False,
                    "interval_95_lower": 0.003,
                    "interval_95_upper": 0.017,
                    "converged": converged,
                }
            )
    summary = summarize_distribution_comparison(pd.DataFrame(rows))
    assert set(summary["n_common"]) == {1}
    student = summary.loc[summary["distribution"].eq("student_t")].iloc[0]
    assert student["failure_count"] == 1
    assert student["attempts"] == 2
    assert summary["MAE"].notna().all()


def test_quintiles_describe_error_dispersion_by_forecast_level():
    forecasts = np.linspace(0.004, 0.012, 10)
    errors = np.array(
        [0.0001, -0.0001, 0.0002, -0.0002, 0.0004, -0.0004, 0.001, -0.001, 0.003, -0.003]
    )
    frame = pd.DataFrame(
        {
            "model": "garch",
            "distribution": "normal",
            "forecast": forecasts,
            "actual": forecasts + errors,
            "converged": True,
        }
    )
    quintiles = summarize_forecast_quintiles(frame)
    assert list(quintiles["forecast_quintile"]) == [1, 2, 3, 4, 5]
    assert quintiles["error_variance"].iloc[-1] > quintiles["error_variance"].iloc[0]


def test_saved_distribution_outputs_record_the_input_hash(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("source", encoding="utf-8")
    output = tmp_path / "out"
    paths = save_distribution_comparison(
        pd.DataFrame({"forecast": [0.01]}),
        pd.DataFrame({"MAE": [0.001]}),
        pd.DataFrame({"forecast_quintile": [1]}),
        {"input_path": str(source), "outputs": []},
        output,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["input_sha256"]
    assert len(manifest["input_sha256"]) == 64
    assert {path.name for path in output.iterdir()} == {
        "distribution_forecasts.csv",
        "distribution_summary.csv",
        "forecast_error_quintiles.csv",
        "distribution_manifest.json",
    }


def test_evaluation_rejects_an_invalid_split():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-02", periods=5),
            "daily_volatility": np.linspace(0.01, 0.02, 5),
            "close_return": 0.0,
            "overnight_log_return": 0.0,
            "open_close_log_return": 0.0,
            "high_low_log_range": 0.01,
            "dxy_close": 100.0,
            "gold_iv_1m": 15.0,
            "gold_iv_1y": 16.0,
        }
    )
    with pytest.raises(ValueError, match="train_fraction"):
        evaluate_distribution_comparison(frame, train_fraction=1)
