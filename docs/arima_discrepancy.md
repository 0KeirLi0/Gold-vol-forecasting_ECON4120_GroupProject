# Auto-ARIMA selection discrepancy

The project log records an exploratory ARIMA(0,1,4) selection on squared
returns, while a later rerun selected ARIMA(3,1,2). These are observations
from different runs, not evidence that either order is correct. The input
snapshot and inclusive end date may have differed; the KPSS decision can
change the differencing order; and a stepwise AICc search can follow a
different candidate path when its input or search settings change.

Use `python scripts/check_auto_arima_reproducibility.py` to print the fixed
input slice's row count, date range, and file hash, then run the selection
twice and assert identical orders, fitted parameters, AICc, and forecasts.
This only checks reproducibility for the local snapshot; it does not resolve
which historical result used the intended data.

The aligned Stage 3 evaluation uses a different, user-requested design:
select the Hyndman-Khandakar order once using only returns available at its
first forecast origin, then keep that order fixed while refitting coefficients
at each rolling origin. On the recorded snapshot this selected ARIMA(3,1,1);
the selection date, data hash and diagnostics are in
`data/processed/stage3_auto_arima_manifest.json`. This does not resolve the
earlier full-sample discrepancy. Re-selecting at every origin would be a
different evaluation design.
