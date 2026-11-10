import numpy as np
import pytest

from gold_vol_pipeline.auto_arima import hyndman_khandakar


def test_hyndman_khandakar_selects_best_converged_aicc_candidate():
    rng = np.random.default_rng(42)
    innovations = rng.normal(0, 0.1, size=180)
    series = np.zeros_like(innovations)
    for index in range(1, len(series)):
        series[index] = 0.65 * series[index - 1] + innovations[index]

    selected = hyndman_khandakar(series, max_p=3, max_q=3, max_order=3)

    successful = selected.candidates.loc[selected.candidates["status"] == "ok"]
    assert selected.order[0] <= 3
    assert selected.order[1] == 0
    assert selected.order[2] <= 3
    assert selected.aicc == pytest.approx(successful["aicc"].min())
    assert np.isfinite(selected.fit.params).all()
    assert selected.forecast(5).shape == (5,)
    assert np.all(selected.forecast(5) >= 0)


def test_hyndman_khandakar_differences_a_random_walk():
    rng = np.random.default_rng(123)
    series = np.cumsum(rng.normal(size=220))

    selected = hyndman_khandakar(series, max_p=2, max_q=2, max_order=2)

    assert selected.order[1] >= 1
    assert len(selected.differencing_p_values) == selected.order[1] + 1


@pytest.mark.parametrize(
    ("series", "message"),
    [
        ([1.0] * 30, "non-zero variance"),
        ([float(index) for index in range(29)], "at least 30"),
        ([float(index) for index in range(29)] + [np.nan], "finite numeric"),
    ],
)
def test_hyndman_khandakar_rejects_invalid_series(series, message):
    with pytest.raises(ValueError, match=message):
        hyndman_khandakar(series)
