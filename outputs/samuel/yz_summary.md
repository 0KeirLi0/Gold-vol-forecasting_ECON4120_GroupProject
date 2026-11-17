# Samuel — Yang-Zhang model evaluation

## Evaluation design

- Target: unannualized daily volatility from a two-observation Yang-Zhang estimator on LSEG GCc1 OHLC.
- Source: read-only forecast artifacts from test-branch commit `988adfcdaf80015cce7b2a2d15fccf5dfbf9d8a4`; main was not switched.
- Sequential one-step test dates: 761; 2023-06-02 to 2026-11-10.
- All 20 baseline models use the same test targets; MAE is the primary ranking metric and RMSE is a complementary outlier-sensitive metric.
- Recomputed MAE/RMSE match the saved Stage 2 common-sample summary: True.
- DM tests use absolute and squared errors, Newey-West bandwidths 1 and 4, one-step HLN correction, and Holm adjustment within each comparison family.

## Main results

- Lowest MAE: **hist_gradient_boosting_ohlc** (MAE 0.00241765; RMSE 0.00476724).
- Lowest RMSE: **hist_gradient_boosting_ohlc** (RMSE 0.00476724; MAE 0.00241765).
- The MAE leader has significantly lower absolute loss than these fixed benchmarks after Holm correction (bandwidth 1): garch11, har, yang_zhang_persistence.
- Other statistically significant paired results at bandwidth 1, absolute loss (Holm-adjusted): yang_zhang_persistence: lower loss [arch1, egarch, garch_t, garchx_ohlc_range, gjr_garch, harx_ohlc, hist_gradient_boosting_ohlc, quantile_garch], higher loss [expanding_mean]; har: lower loss [harx_ohlc, hist_gradient_boosting_ohlc, ridge_ohlc], higher loss [ewma, expanding_mean, ma1]; garch11: lower loss [hist_gradient_boosting_ohlc], higher loss [ewma, expanding_mean, ma1].
- For squared loss at bandwidth 1, significant lower/higher-loss models are: yang_zhang_persistence: lower loss [none], higher loss [none]; har: lower loss [hist_gradient_boosting_ohlc], higher loss [ewma, expanding_mean, ma1]; garch11: lower loss [gjr_garch], higher loss [expanding_mean, garch_t]. Bandwidth 4 sensitivity results are in `yz_dm_results.csv`.
- MAE and RMSE are in daily-volatility units. Bias is reported as actual minus forecast; a positive value means average under-forecasting.
- The HAC mean-error test rejects zero bias for 18 of 20 base models; the non-rejections are ewma, yang_zhang_persistence. Positive errors indicate under-forecasting.
- The MAE leader under-forecasts on average (HAC p=6.491e-05); its Mincer-Zarnowitz joint-calibration p=2.02e-06. The joint test fails to reject for: ewma. A zero mean error alone does not establish calibration.

## Predictor experiments

- Adding DXY and gold 1M/1Y IV changed MAE for 4 of 6 paired models in the improving direction; it changed RMSE for 4 of 6.
- Adding VIX changed MAE for 2 of 6 paired models in the improving direction; it changed RMSE for 2 of 6.
- After Holm adjustment for paired absolute-loss tests, significant selected-predictor changes: var_ohlc [selected_external_predictors_vs_base]; significant VIX changes: none.
- Predictor additions are judged on paired out-of-sample losses. In-sample predictor significance alone is not evidence of better forecasting.

## Gold implied-volatility reference

- The 1M IV was converted to daily unannualized volatility as `MID_PRICE / 100 / sqrt(252)` and aligned to the prior quote date.
- The scale factor 0.801195 was estimated using 3040 training targets only (2010-01-06 to 2023-06-01).
- Unscaled IV MAE/RMSE: 0.00393484/0.00575257; training-scaled IV MAE/RMSE: 0.00324413/0.00558733.
- Both unscaled and training-scaled IV have higher absolute loss than the test-sample MAE leader (Holm-adjusted p=2.083e-32 and 1.565e-16, respectively). Scaled IV is not significantly different from persistence (Holm p=0.664). These tests are exploratory because the leader was selected on this same test set.
- IV maturity is one month while the target is next-session volatility, so this is a market reference, not a maturity-matched forecast.
- VIX measures U.S. equity implied volatility. It is assessed as an input to selected gold-volatility models, not as a standalone gold-volatility forecast.

## Calibration and residual checks

- Residual check: At least one selected model has forecast-error autocorrelation at a tested lag; overlapping targets can contribute to this result.
- A changing residual variance across volatility regimes is not automatically a model failure; forecast errors can naturally widen when actual volatility is high. Check conditional bias and forecast calibration rather than demanding constant error variance.
- The calibration regression uses HAC standard errors; any intercept correction must be estimated on training/validation data and then tested on untouched test data.

## Limitations

1. The Yang-Zhang target is an estimated two-observation volatility measure, not directly observed latent volatility; adjacent targets overlap.
2. The comparison covers one continuous COMEX gold futures series and one chronological holdout; it does not establish performance in every market regime.
3. Choosing a winner using this test sample creates selection uncertainty; Holm correction controls the specified test families but does not replace a later holdout.
4. One-month gold IV and one-day realized volatility have different maturities; the conversion and training-period scale factor may not transfer across periods.
5. VIX is a cross-market predictor, and the availability proxy for external predictors may not equal the true historical publication time.

## Reproducibility

- Random seed: 42. Package versions: Python 3.14.6, numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, statsmodels 0.14.6, matplotlib 3.11.0.
- Forecast inputs were read from a pinned Git object with `git show`; no branch checkout or source-data edits were made.
- Outputs are written under `outputs/samuel/`.
