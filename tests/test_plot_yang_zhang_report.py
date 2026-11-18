import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.axes import Axes

from scripts.plot_yang_zhang_report import create_paired_forecast_figures


def _inputs(tmp_path):
    baseline = pd.DataFrame([
        {
            "model": "model_a", "origin_date": "2024-01-01",
            "target_date": "2024-01-02", "actual": 2.0,
            "forecast": 1.0, "converged": True,
        },
        {
            "model": "model_a", "origin_date": "2024-01-02",
            "target_date": "2024-01-03", "actual": 3.0,
            "forecast": 2.0, "converged": True,
        },
    ])
    selected = baseline.assign(forecast=[1.5, 2.5])
    summary = pd.DataFrame([{
        "model": "model_a", "rank_common_MAE": 1, "rank_common_RMSE": 1,
    }])
    paths = [tmp_path / name for name in ("baseline.csv", "selected.csv", "summary.csv")]
    for frame, path in zip((baseline, selected, summary), paths):
        frame.to_csv(path, index=False)
    return paths


def test_paired_figure_plots_actual_and_both_forecasts(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    lines = {}
    original = Axes.plot

    def capture(axis, x, y, *args, **kwargs):
        lines[kwargs["label"]] = list(y)
        return original(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    figures = create_paired_forecast_figures(*paths, tmp_path / "figures")
    assert len(figures) == 1 and figures[0].is_file()
    assert lines == {
        "Actual target": [2.0, 3.0],
        "No external predictors": [1.0, 2.0],
        "Joint predictors": [1.5, 2.5],
    }


@pytest.mark.parametrize("column,value,message", [
    ("target_date", "2024-01-04", "target dates differ"),
    ("origin_date", "2023-12-31", "origins differ"),
    ("actual", 4.0, "actuals differ"),
    ("converged", False, "Unsuccessful"),
])
def test_paired_figure_rejects_mismatched_or_failed_data(tmp_path, column, value, message):
    paths = _inputs(tmp_path)
    selected = pd.read_csv(paths[1])
    selected.loc[0, column] = value
    selected.to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match=message):
        create_paired_forecast_figures(*paths, tmp_path / "figures")
