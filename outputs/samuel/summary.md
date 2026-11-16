# Samuel — model comparison

## Scope

**This analysis uses the Stage 3 forecasts currently in the workspace, not the Yang–Zhang Stage 4 files.**
The available target is five-session realized variance from the Yahoo `GC=F` continuous gold-futures series.
The requested Stage 4 Yang–Zhang forecast files are not present in this checkout, so no result below should be presented as a Yang–Zhang result.

- Models: 25; common forecast origins: 443.
- Evaluation period: 2018-01-22 to 2026-11-03.
- Horizon: 5 observed sessions; target windows overlap.
- Primary ranking: RMSE (equivalent ordering to MSE); QLIKE and MAE are additional criteria.
- HAC lag/bandwidth for this five-session target: 4.

## Results

- Lowest RMSE: **garchx_vix** (RMSE 0.000974271; MAE 0.000420051; QLIKE 0.4552).
- Lowest MAE: **egarch_asym** (MAE 0.000412775).
- Lowest QLIKE: **egarch_asym** (QLIKE 0.4432).
- Compared with expanding historical variance under squared-error loss, models with significantly lower loss after Holm correction: None after Holm correction.
- Models with significantly higher squared-error loss after Holm correction: var.
- Under QLIKE, significantly lower loss after Holm correction: None after Holm correction; significantly higher loss: var.
- Mean-error HAC bias tests (actual minus forecast; p-values are unadjusted): ar1_squared_return (mean error 0.000197, p=0.00481), arch1 (mean error 0.000208, p=0.00327), historical_variance (mean error 0.000203, p=0.00452), lasso_full (mean error 0.000141, p=0.0199), ma1_squared_return (mean error 0.000199, p=0.00483), var (mean error 0.000539, p=1.16e-13), xau_1m_iv_unscaled (mean error 0.000132, p=0.00944), xau_1m_iv_scaled (mean error 0.000143, p=0.00524)
- Mincer-Zarnowitz calibration: The HAC Mincer-Zarnowitz joint test rejects (intercept=0, slope=1) for 22 of 25 models at 5%: ar1_squared_return, arch1, auto_arima_hk, ewma, ewma_fixed, garch11, garch_t, gjr_garch, harx_dxy_vix, harx_full, harx_vix, historical_variance, lasso_dxy_vix, lasso_full, lasso_vix, ma1_squared_return, quantile_garch, ridge_dxy_vix, ridge_full, ridge_vix, rolling_variance, var. Failure to reject is not proof of perfect calibration.
- Residual checks: Some leading-model errors show Ljung-Box evidence of serial dependence at one or more tested lags; overlapping targets make this diagnostic exploratory.
- VIX role: Against GARCH(1,1), `garchx_vix` has mean squared-loss difference -2.48e-08 (HLN p=0.487). VIX is an input to this model, not a standalone gold-volatility forecast.

## Implied-volatility reference

- Gold 1-month implied volatility was converted using `five-session variance = 5 * (MID_PRICE / 100 / sqrt(252))^2`.
- Quotes were aligned using the previous available quote; same-day quotes were excluded.
- The scale factor (0.978031) uses 760 pre-test targets whose target windows had ended before the first test origin (2018-01-22).
- On 443 dates with a prior quote, unscaled IV RMSE/MAE were 0.000938294/0.000388812; training-scaled IV RMSE/MAE were 0.000941202/0.000387822.
- Nominal DM comparisons with the test-sample RMSE winner give Holm-adjusted squared-error p-values 0.069 (unscaled) and 0.069 (scaled); the winner was selected on this same test sample, so these are exploratory.
- The raw IV RMSE is lower than the top model's RMSE (0.000938294 versus 0.000974271), but the Holm-adjusted DM p-value is 0.069; this does not meet the 5% threshold.
- Quote quality flag: 1 test quote(s) had MID_PRICE below 1% ([{'origin_date': Timestamp('2021-12-28 00:00:00'), 'quote_date': Timestamp('2021-12-27 00:00:00'), 'MID_PRICE': 0.132}]). The raw unscaled IV QLIKE is 26.7150; excluding that flagged date only as a sensitivity check changes it to 0.3939. The original quote is retained; its source should be verified before treating IV QLIKE as reliable.
- This is only an exploratory reference: the implied-volatility quote is one-month, while the evaluated target is five-session futures realized variance; constant variance over the five sessions is assumed for conversion, and the implied-volatility underlying may not match the futures series.
- VIX measures U.S. equity implied volatility, not gold implied volatility. It is treated as a model predictor here, not as a direct gold forecast benchmark.

## Limitations

1. The available data do not match the proposed next-valid-day Yang–Zhang target; the requested YZ forecast artifacts are absent.
2. The target is a noisy five-session realized-variance proxy, and adjacent target windows overlap, creating dependent losses.
3. `GC=F` is a continuous futures series, not spot gold or a single fixed-expiry contract; results depend on that instrument and its roll behavior.
4. The implied-volatility benchmark has a one-month maturity and potentially different underlying/instrument; converting it to five-session variance requires a flat-variance assumption.
5. The same test sample is used to rank models and identify the RMSE winner; post-selection comparisons are exploratory, and the test period does not establish future stability or causality.

## Reproducibility

- Random seed set to 42.
- Package versions: Python 3.14.6, numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, statsmodels 0.14.6, matplotlib 3.11.0.
- Source forecasts and input data were read only; this analysis writes new files under `outputs/samuel/`.
- DM comparisons use Newey–West Bartlett long-run variance, Harvey–Leybourne–Newbold correction, and Holm correction separately by loss and bandwidth.
