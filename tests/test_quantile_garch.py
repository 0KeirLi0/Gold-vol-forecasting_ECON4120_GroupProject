from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gold_vol_pipeline import QuantileGarchFit, fit_quantile_garch


def test_extreme_shock_adds_gamma_to_the_shock_coefficient():
    fit = QuantileGarchFit(
        omega=0.1, alpha=0.2, beta=0.3, gamma=0.4,
        threshold=1.0, last_return=-2.0, last_variance=0.5,
        observations=100, extreme_observations=5, negative_log_likelihood_scaled=1.0,
    )
    assert fit.forecast() == pytest.approx(0.1 + (0.2 + 0.4) * 4 + 0.3 * 0.5)
    ordinary = replace(fit, last_return=1.0)
    assert ordinary.forecast() == pytest.approx(0.1 + 0.2 + 0.3 * 0.5)
    assert replace(fit, last_return=2.0).forecast() == pytest.approx(fit.forecast())


def test_simulation_is_reproducible_and_positive():
    returns = np.random.default_rng(42).normal(0, 0.01, 200)
    fit = fit_quantile_garch(returns)
    assert fit.extreme_observations == 10
    assert fit.alpha + fit.beta + fit.gamma <= 0.999 + 1e-8
    assert fit.forecast(horizon=5, simulations=500) > 0
    assert fit.forecast(horizon=5, simulations=500) == fit.forecast(
        horizon=5, simulations=500
    )


def test_simulation_matches_garch_expectation_when_gamma_is_zero():
    fit = QuantileGarchFit(
        omega=0.1, alpha=0.2, beta=0.3, gamma=0.0,
        threshold=1.0, last_return=2.0, last_variance=0.5,
        observations=100, extreme_observations=5, negative_log_likelihood_scaled=1.0,
    )
    first = fit.forecast()
    expected = first + 0.1 + (0.2 + 0.3) * first
    assert fit.forecast(horizon=2, simulations=200_000) == pytest.approx(
        expected, rel=0.005
    )


def test_fit_and_forecast_are_equivariant_to_return_scale():
    returns = np.random.default_rng(21).normal(0, 0.01, 150)
    original = fit_quantile_garch(returns)
    scaled = fit_quantile_garch(returns * 100)
    assert scaled.threshold == pytest.approx(original.threshold * 100)
    assert scaled.gamma == pytest.approx(original.gamma, abs=1e-5)
    assert scaled.forecast(horizon=5) == pytest.approx(
        original.forecast(horizon=5) * 100**2, rel=1e-5
    )


def test_invalid_inputs_are_explicit():
    with pytest.raises(ValueError, match="at least 100"):
        fit_quantile_garch([0.01] * 99)
    with pytest.raises(ValueError, match="nonzero"):
        fit_quantile_garch(np.zeros(100))
    with pytest.raises(ValueError, match="quantile"):
        fit_quantile_garch(np.ones(100), quantile=1)
    with pytest.raises(ValueError, match="extreme lagged return"):
        fit_quantile_garch(np.ones(100))
    fit = fit_quantile_garch(np.random.default_rng(2).normal(size=100))
    with pytest.raises(ValueError, match="simulations"):
        fit.forecast(horizon=5, simulations=0)


def test_calibration_script_does_not_use_returns_after_origin(tmp_path):
    root = Path(__file__).resolve().parents[1]
    returns = np.random.default_rng(42).normal(0, 0.01, 130)
    dates = pd.date_range("2020-01-01", periods=len(returns))
    panel = pd.DataFrame({"date": dates, "log_return": returns})
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)
    origin = dates[119].strftime("%Y-%m-%d")
    command = [
        sys.executable, str(root / "scripts/calibrate_quantile_garch.py"),
        "--input", str(path), "--origin", origin, "--simulations", "100",
    ]
    first = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    panel.loc[120:, "log_return"] = 100.0
    panel.to_csv(path, index=False)
    second = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    assert first["fit"] == second["fit"]
    assert first["forecast"] == second["forecast"]
    assert first["input_sha256"] != second["input_sha256"]
