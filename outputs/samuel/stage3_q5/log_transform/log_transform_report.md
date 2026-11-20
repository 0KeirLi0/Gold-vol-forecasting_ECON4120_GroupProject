# Log-target and bias-corrected forecast experiment

All six model specifications were fit to `log(daily_volatility)` using only observations available through each forecast origin. Forecasts are back-transformed as `exp(mu_hat + 0.5 * sigma_hat^2)`, the conditional-mean correction under a normal log-error assumption. It differs from `exp(mu_hat)`, which estimates the conditional median.

GARCH and EGARCH use their one-step conditional variance on the log target. Range GARCH-X keeps its range predictor in the conditional-variance equation and uses Gaussian quasi-likelihood. EGARCH-X keeps DXY and gold IV predictors in the ARX mean. HAR-X uses the existing OHLC and selected predictors in its log-scale OLS mean. HGB uses its existing log-scale features and fixed random seed 42. For HAR-X/HGB, `sigma_hat^2` is estimated from training residuals; HGB uses in-sample training residual variance, which may be optimistic.

## Sample and convergence

- Test targets: 761 (2023-06-02 through 2026-11-10).
- Forecast errors are actual minus forecast; positive mean error indicates under-forecasting.
- Each transformed model is compared with its existing saved forecast on the same target dates.

| model_label   |   attempts |   failures |
|:--------------|-----------:|-----------:|
| GARCH(1,1)    |        761 |          0 |
| EGARCH(1,1)   |        761 |          0 |
| Range GARCH-X |        761 |          0 |
| EGARCH-X      |        761 |          0 |
| HAR-X (OHLC)  |        761 |          0 |
| HGB (OHLC)    |        761 |          0 |

## Overall performance

| model_label   |   baseline_MAE |   logY_corrected_MAE |   baseline_RMSE |   logY_corrected_RMSE |   baseline_mean_error |   logY_corrected_mean_error |
|:--------------|---------------:|---------------------:|----------------:|----------------------:|----------------------:|----------------------------:|
| GARCH(1,1)    |    0.002911639 |          0.002930065 |     0.005024506 |           0.005035741 |          0.0006437899 |                0.0004198108 |
| EGARCH(1,1)   |    0.002912152 |          0.002945232 |     0.005021795 |           0.005114624 |          0.0006208026 |                0.0005400703 |
| Range GARCH-X |    0.002913623 |          0.002930073 |     0.00503715  |           0.005035752 |          0.0007163328 |                0.0004198098 |
| EGARCH-X      |    0.002833236 |          0.002854768 |     0.004859826 |           0.004902377 |          0.000320758  |                8.643825e-05 |
| HAR-X (OHLC)  |    0.002757858 |          0.002825097 |     0.005624986 |           0.005851882 |          0.0006232538 |               -5.972866e-05 |
| HGB (OHLC)    |    0.002423457 |          0.002436843 |     0.00473556  |           0.004656587 |          0.0008212024 |                0.0003858922 |

## Forecast-Q5 variance

The first two variance columns compare base and transformed errors on the exact same dates classified as baseline forecast Q5. The final column is also shown, but uses the transformed model's own holdout forecast ranking, so it can contain different dates.

| model                       | model_label   |   baseline_q5_n |   baseline_q5_variance |   transformed_error_variance_on_baseline_q5_dates |   same_dates_variance_reduction_pct |   transformed_own_q5_n |   transformed_own_q5_variance |
|:----------------------------|:--------------|----------------:|-----------------------:|--------------------------------------------------:|------------------------------------:|-----------------------:|------------------------------:|
| garch11                     | GARCH(1,1)    |             152 |           7.433159e-05 |                                      7.543317e-05 |                          -1.481969  |                    152 |                  7.681495e-05 |
| egarch                      | EGARCH(1,1)   |             152 |           7.436765e-05 |                                      7.839181e-05 |                          -5.411177  |                    152 |                  7.916328e-05 |
| garchx_ohlc_range           | Range GARCH-X |             152 |           7.440889e-05 |                                      7.543358e-05 |                          -1.37711   |                    152 |                  7.681534e-05 |
| egarchx                     | EGARCH-X      |             152 |           7.287578e-05 |                                      7.600157e-05 |                          -4.289197  |                    152 |                  7.769693e-05 |
| harx_ohlc                   | HAR-X (OHLC)  |             152 |           0.0001141299 |                                      0.000128808  |                         -12.86089   |                    152 |                  0.000128808  |
| hist_gradient_boosting_ohlc | HGB (OHLC)    |             152 |           7.038051e-05 |                                      7.02383e-05  |                           0.2020515 |                    152 |                  7.028285e-05 |

## Paired tests

Diebold-Mariano tests compare transformed minus baseline loss, use Newey-West bandwidths 1 and 4, the Harvey-Leybourne-Newbold small-sample correction with one-step horizon, and Holm adjustment across six models within each loss/bandwidth family. Negative mean loss differences favor the transformed forecast.

| model                       | model_label   | loss           |   bandwidth |   n |   mean_loss_difference_transformed_minus_baseline |   DM_HLN_statistic |      p_raw |    p_holm |
|:----------------------------|:--------------|:---------------|------------:|----:|--------------------------------------------------:|-------------------:|-----------:|----------:|
| garch11                     | GARCH(1,1)    | absolute_error |           1 | 761 |                                       1.84258e-05 |          0.804523  | 0.421347   | 1         |
| garch11                     | GARCH(1,1)    | absolute_error |           4 | 761 |                                       1.84258e-05 |          0.929013  | 0.353177   | 1         |
| garch11                     | GARCH(1,1)    | squared_error  |           1 | 761 |                                       1.13026e-07 |          0.241733  | 0.809052   | 1         |
| garch11                     | GARCH(1,1)    | squared_error  |           4 | 761 |                                       1.13026e-07 |          0.262983  | 0.792635   | 1         |
| egarch                      | EGARCH(1,1)   | absolute_error |           1 | 761 |                                       3.30802e-05 |          1.16312   | 0.245148   | 1         |
| egarch                      | EGARCH(1,1)   | absolute_error |           4 | 761 |                                       3.30802e-05 |          1.40499   | 0.160433   | 0.802165  |
| egarch                      | EGARCH(1,1)   | squared_error  |           1 | 761 |                                       9.40947e-07 |          1.15491   | 0.248492   | 1         |
| egarch                      | EGARCH(1,1)   | squared_error  |           4 | 761 |                                       9.40947e-07 |          1.19429   | 0.232739   | 1         |
| garchx_ohlc_range           | Range GARCH-X | absolute_error |           1 | 761 |                                       1.64504e-05 |          0.702446  | 0.482616   | 1         |
| garchx_ohlc_range           | Range GARCH-X | absolute_error |           4 | 761 |                                       1.64504e-05 |          0.795059  | 0.426828   | 1         |
| garchx_ohlc_range           | Range GARCH-X | squared_error  |           1 | 761 |                                      -1.40823e-08 |         -0.0315518 | 0.974838   | 1         |
| garchx_ohlc_range           | Range GARCH-X | squared_error  |           4 | 761 |                                      -1.40823e-08 |         -0.0346478 | 0.97237    | 1         |
| egarchx                     | EGARCH-X      | absolute_error |           1 | 761 |                                       2.15323e-05 |          0.723926  | 0.469334   | 1         |
| egarchx                     | EGARCH-X      | absolute_error |           4 | 761 |                                       2.15323e-05 |          0.865813  | 0.386866   | 1         |
| egarchx                     | EGARCH-X      | squared_error  |           1 | 761 |                                       4.15397e-07 |          0.596504  | 0.551016   | 1         |
| egarchx                     | EGARCH-X      | squared_error  |           4 | 761 |                                       4.15397e-07 |          0.878045  | 0.380197   | 1         |
| harx_ohlc                   | HAR-X (OHLC)  | absolute_error |           1 | 761 |                                       6.72395e-05 |          2.04554   | 0.0411455  | 0.246873  |
| harx_ohlc                   | HAR-X (OHLC)  | absolute_error |           4 | 761 |                                       6.72395e-05 |          1.88202   | 0.0602151  | 0.36129   |
| harx_ohlc                   | HAR-X (OHLC)  | squared_error  |           1 | 761 |                                       2.60405e-06 |          1.01503   | 0.310413   | 1         |
| harx_ohlc                   | HAR-X (OHLC)  | squared_error  |           4 | 761 |                                       2.60405e-06 |          0.967932  | 0.333386   | 1         |
| hist_gradient_boosting_ohlc | HGB (OHLC)    | absolute_error |           1 | 761 |                                       1.33863e-05 |          0.670635  | 0.502657   | 1         |
| hist_gradient_boosting_ohlc | HGB (OHLC)    | absolute_error |           4 | 761 |                                       1.33863e-05 |          0.65549   | 0.512351   | 1         |
| hist_gradient_boosting_ohlc | HGB (OHLC)    | squared_error  |           1 | 761 |                                      -7.41727e-07 |         -2.59276   | 0.00970401 | 0.0582241 |
| hist_gradient_boosting_ohlc | HGB (OHLC)    | squared_error  |           4 | 761 |                                      -7.41727e-07 |         -2.26623   | 0.023718   | 0.142308  |

## Interpretation limits

- The correction assumes conditionally normal log errors; HGB and HAR-X residuals need not be normal.
- The variance estimates use only information through the origin, but HAR-X/HGB use in-sample training residuals.
- Q5 membership is based on holdout forecast ranks and is retrospective; same-date Q5 variances are included to separate membership changes from forecast changes.
- This is a single holdout experiment, not evidence of causal effects or durable model superiority.
