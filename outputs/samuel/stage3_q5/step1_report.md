# Step 1 — Q5 forecast-error diagnosis

Signed errors use **actual minus forecast**; positive errors mean under-forecasting. HGB remains a point forecast and is not assigned a likelihood. Range GARCH-X is labeled Gaussian quasi-likelihood.

## Data and grouping

- Common test targets: 761 (2023-06-02 to 2026-11-10).
- Five model forecasts come from the tracked baseline 20-model CSV. EGARCH-X was regenerated sequentially; its selected predictors remain in the ARX mean equation.
- Forecast Q1–Q5 are equal-count holdout ranks separately within each model; tied forecasts are ordered by target date.
- Origin-volatility bins use the initial training sample's 20/40/60/80% cutoffs. The input is the last observed target at each forecast origin, available at that time. Test bin sizes need not be equal.
- Realized-volatility bins rank holdout actuals and are retrospective diagnostics.
- HAC regressions use four lags. The slope tests are associations, not causal tests.

## Existing-report comparison

The existing report gives HGB Q5 mean error 2.025 and skewness 4.01 (mean in 10^-3 units). This baseline run gives 2.214 and 4.07. The report's HGB is from its joint-predictor run, while this requested baseline CSV uses the no-external-predictor forecast, so a difference is not a same-specification mismatch. For regenerated EGARCH-X on 2026-02-02, actual is 77.521, forecast is 13.608, and absolute error is 63.912 in 10^-3 units. The selected-model report only publishes the cross-model error range 62.989–65.588; this regenerated EGARCH-X miss falls within that range, but no model-specific EGARCH-X value is given there.
No Q5 residual-variance summary CSV is present in the checkout, so there is no prior Q5 variance number to reproduce. The report does contain mean-error and skewness tables.

## Forecast-Q5 variance ratios

| Model | Raw variance Q5/Q1 | Relative variance Q5/Q1 | Log-error variance Q5/Q1 |
|---|---:|---:|---:|
| GARCH(1,1) | 9.187 | 1.035 | 0.8377 |
| EGARCH(1,1) | 9.194 | 1.029 | 0.8399 |
| Range GARCH-X | 9.329 | 1.072 | 0.8474 |
| EGARCH-X | 15.05 | 1.327 | 0.7856 |
| HAR-X (OHLC) | 11.55 | 1.283 | 0.477 |
| HGB (OHLC) | 12.28 | 0.9582 | 0.5078 |

## Evidence and hypotheses

- **Evidence:** Raw Q5 error variance exceeds Q1 for 6 of 6 models in this common-date sample.
- **Evidence:** Raw Q5/Q1 variance ratios are 9.2–15.1, while relative-error ratios are 0.96–1.33 and log-error ratios are 0.48–0.85. The raw-scale widening largely disappears after rescaling; it does not persist as larger Q5 variance in log errors.
- **Evidence:** Q5 errors remain right-skewed and heavy-tailed. Raw-error Q5 skewness is 3.36–4.07 and excess kurtosis is 21.9–25.0, so scale is not the only diagnostic concern.
- **Evidence:** All six models have successful forecasts on the 761 common targets used here.
- **Evidence:** 6 of the ten largest Q5 misses share the target date 2026-02-02, showing a common difficult episode across models.
- **Hypothesis:** If relative and log Q5/Q1 variance ratios are much closer to 1 than the raw ratio, much of the raw widening is a scale effect. Ratios that remain high are consistent with proportional misses, heavy tails, or changing regimes; these summaries cannot identify which mechanism caused them.
- **Hypothesis:** Extreme misses and shared miss dates may reflect abrupt volatility changes a one-step point forecast does not capture. This is a possible explanation, not a causal conclusion.

## HAC variance-trend regressions

Each model is fit separately. The raw regression uses squared actual-minus-forecast errors as the outcome and forecast level as the predictor. The second uses squared log errors and log forecast.

| Model | Regression | Slope sign | HAC slope | HAC SE | p-value | Lags |
|---|---|---|---:|---:|---:|---:|
| EGARCH(1,1) | raw | positive | 0.005661 | 0.00146 | 0.0001149 | 4 |
| EGARCH(1,1) | log | non-positive | -0.06918 | 0.04533 | 0.1274 | 4 |
| EGARCH-X | raw | positive | 0.006835 | 0.001901 | 0.0003449 | 4 |
| EGARCH-X | log | non-positive | -0.02345 | 0.04376 | 0.5922 | 4 |
| GARCH(1,1) | raw | positive | 0.005697 | 0.001475 | 0.0001211 | 4 |
| GARCH(1,1) | log | non-positive | -0.06967 | 0.04541 | 0.1254 | 4 |
| Range GARCH-X | raw | positive | 0.005765 | 0.001481 | 0.0001079 | 4 |
| Range GARCH-X | log | non-positive | -0.06712 | 0.0462 | 0.1467 | 4 |
| HAR-X (OHLC) | raw | positive | 0.006843 | 0.002389 | 0.004303 | 4 |
| HAR-X (OHLC) | log | non-positive | -0.1232 | 0.04222 | 0.003628 | 4 |
| HGB (OHLC) | raw | positive | 0.007083 | 0.003528 | 0.04501 | 4 |
| HGB (OHLC) | log | non-positive | -0.09561 | 0.03928 | 0.01514 | 4 |

## Ten largest absolute misses among forecast-Q5 rows

| Rank | Model | Target date | Actual | Forecast | Actual − forecast |
|---:|---|---|---:|---:|---:|
| 1 | HAR-X (OHLC) | 2026-02-02 | 0.07752 | 0.009748 | 0.06777 |
| 2 | Range GARCH-X | 2026-02-02 | 0.07752 | 0.01201 | 0.06551 |
| 3 | GARCH(1,1) | 2026-02-02 | 0.07752 | 0.01202 | 0.0655 |
| 4 | EGARCH(1,1) | 2026-02-02 | 0.07752 | 0.01205 | 0.06547 |
| 5 | EGARCH-X | 2026-02-02 | 0.07752 | 0.01361 | 0.06391 |
| 6 | HGB (OHLC) | 2026-02-02 | 0.07752 | 0.01453 | 0.06299 |
| 7 | HGB (OHLC) | 2026-03-23 | 0.06132 | 0.01451 | 0.04681 |
| 8 | HAR-X (OHLC) | 2026-03-23 | 0.06132 | 0.0148 | 0.04651 |
| 9 | Range GARCH-X | 2026-03-23 | 0.06132 | 0.01825 | 0.04307 |
| 10 | GARCH(1,1) | 2026-03-23 | 0.06132 | 0.01825 | 0.04307 |

## Fit attempts

| Model | Attempts | Failures |
|---|---:|---:|
| GARCH(1,1) | 761 | 0 |
| EGARCH(1,1) | 761 | 0 |
| Range GARCH-X | 761 | 0 |
| EGARCH-X | 761 | 0 |
| HAR-X (OHLC) | 761 | 0 |
| HGB (OHLC) | 761 | 0 |

Complete statistics for every model, grouping, quintile, and error measure are in `grouped_error_metrics.csv`. Row-level assignments and errors are in `classified_forecasts.csv`; regenerated EGARCH-X rows are explicitly labeled.
