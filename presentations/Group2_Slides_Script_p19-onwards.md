# Group 2 Presentation Script

## From PDF page 19 onwards

Student-style speaking script for the remaining slides.

---

## Page 19 - How much Q5 variance the largest misses carry

This figure summarises how sensitive the Q5 result is to the biggest residual misses.

In other words, we are asking: if we remove the very largest forecasting errors, how much of the extra variance in Q5 disappears? The chart shows that removing the top one, three, or five absolute errors reduces the high-volatility residual variance for every model.

The 2026-02-02 miss is the largest miss in every model, and 2026-03-23 is another major one. Even after we remove the top five, Q5 is still noisier than Q1, but the remaining ratio is much smaller. So, my interpretation is not simply “one bad date causes everything.” Instead, the extreme dates explain a large share of the headline problem, while a milder high-volatility problem remains.

**Transition:** Because the Student-t distribution is often used for heavy-tailed data, we also tested whether changing the error distribution would solve the Q5 issue.

---

## Page 20 - Patch 1: Student-t MLE

For our first patch, we ran a Normal-versus-Student-t experiment across 761 origins and all six models. We got 761 successful forecasts for each model and distribution, with no fitting failures.

The estimated degrees of freedom are roughly 4.1 to 4.4 for the GARCH-family models, and 4.51 for HGB. This means the residuals are heavier-tailed than a Normal distribution.

The Student-t model improves the density scores, so it is better at representing tail uncertainty. However, it does not really fix the point forecasts. The Q5-to-Q1 variance ratio remains almost unchanged for GARCH and HGB, and the HGB forecast is exactly unchanged because it is not a likelihood-based model.

So, we would say that Student-t is useful for uncertainty and tail density, but it is not a complete answer to the Q5 point-forecast problem.

**Transition:** We also tried a second idea: fitting separate parameters for normal and high-volatility regimes.

---

## Page 21 - Patch 2: segmented high-volatility parameters

For the segmented experiment, we defined high volatility using the 80th percentile of the training period. We then tried two ways to classify each origin: one used a pilot forecast and the other used the lagged realised target.

This gave us 12 model-and-trigger combinations, and all of them produced forecasts at all 761 origins without failures.

However, the results were worse. HGB MAE increased from 0.002418 to 0.002680 or 0.002760, while GARCH MAE increased from 0.002912 to around 0.00324. This held for both classification methods.

Therefore, we treat this as a failed patch rather than an improved model. It also uses the retrospective holdout that was already examined, so it should not be interpreted as a deployable regime model.

**Transition:** The third patch changed the target scale.

---

## Page 22 - Patch 3: log Y and Jensen reverse

In the log-target patch, we fit the models to log Yang-Zhang volatility and then reversed the transformation using the usual Jensen correction.

This did reduce the overall under-forecasting bias. It moved the bias closer to zero for all six models. The differences in MAE and RMSE were not statistically significant after the Holm adjustment.

But the patch did not solve the Q5 problem. On the original baseline Q5 dates, residual variance increased for five of the six models. HGB reduced variance by only about 0.2%.

So, the log transformation is useful for addressing some average bias, but it did not give us a robust fix for high-volatility forecast errors.

**Transition:** To check whether the ranking and Q5 result depend on one particular time period, we split the holdout into two subperiods.

---

## Page 23 - Confirmatory later-holdout split

We split the same 761 holdout observations at 31 December 2024. The earlier window has 347 dates, and the later window has 414.

The same main ranking appears in both slices. HGB has the lowest MAE in both periods, followed by HAR-X.

In 2023-24, the Q5-to-Q1 ratios are small: about 1.28 for HGB and 1.36 for HAR-X. In 2025-26, the ratios increase substantially to 9.29 and 5.54.

This confirms that the high-volatility problem becomes much more visible in the later higher-volatility period. We should be clear that this is a confirmatory split, not an untouched holdout, because the 2026-02-02 date had already been inspected.

**Transition:** The subperiod plot gives a similar picture.

---

## Page 24 - MAE on 2023-24 versus 2025-26

This plot compares MAE across the earlier and later windows.

HGB remains in first place in both periods. However, every model has higher MAE in the later window. This is consistent with the later period being a higher-volatility and more event-heavy regime.

The green line represents the full holdout ranking, while the orange results represent the later split. The key point is that HGB’s lead does not disappear over time, but the absolute level of forecasting error gets worse.

This result therefore supports two conclusions: the ranking is reasonably stable, but model performance is still regime-dependent.

**Transition:** Finally, we checked what happens if we use longer Yang-Zhang windows instead of the default n equal to 2.

---

## Page 25 - Window n equals 2, 5, or 10

As we increase the Yang-Zhang window from 2 to 5 or 10 sessions, the lag-1 autocorrelation rises from 0.65 to 0.90 and 0.96.

At the longer windows, persistence starts to perform better in MAE, while the Q5-to-Q1 ratio becomes much larger, increasing to roughly 30 to 63.

The reason is overlapping windows. With a longer window, many of the observations share the same underlying trading sessions. This smooths the target and also makes the Q1 variance very small, while extreme observations remain in Q5.

So, a longer window is not a valid fix for the Q5 problem. It reduces the apparent variation in many observations and makes the target process mechanically more persistent.

**Transition:** We now come to the limitations and overall interpretation.

---

## Page 26 - Limits

Before our conclusion, we should make the main limitations explicit.

First, Yang-Zhang windows overlap, especially for n equals 5 and 10. This creates serial dependence that is not the same as independent daily observations.

Second, we have only one chronological holdout, and it has already been used in model development. The 2025-26 split is therefore confirmatory, not a completely unused test set.

Third, the gold IV used in the predictors is a one-month maturity, while the target is next-session realised volatility. We also use a next-weekday LSEG proxy. These are not perfectly matched instruments.

Fourth, GCc1 is a generic gold futures continuation, and TRDPRC_1 is the last trade rather than settlement. It is therefore an available data-series proxy, not a spot series or a fixed-expiry contract.

Finally, the MAE-to-dollar illustration is only an illustration. It should not be interpreted as a trading gain or a full options-pricing result.

We do not claim that HGB is the true data-generating process, or that Student-t, segmentation, or the log transformation solved Q5.

**Transition:** We also need to explain the team and AI contribution accurately.

---

## Page 27 - Use of AI and contributions

For the use-of-AI section, we should keep the distinction clear.

Copilot and Grok assisted with code, diagnostics, and draft text. The team made the substantive choices, including using GCc1, using Yang-Zhang, choosing the three-part story, and deciding on the shortlist.

The contribution split is: Enoch was responsible for model development and the predictor screen, Keir handled infrastructure and the LSEG Yang-Zhang pipeline, and I handled the Q5 diagnosis and several robustness experiments.

One point to state clearly is that AI did not estimate the model results. The quoted figures come from saved output files, and AI did not estimate FIGARCH or invent the holdout results.

**Transition:** Our final recap will bring all of these findings together.

---

## Page 28 - Recap

To summarise, we used COMEX GCc1 gold futures, not spot gold, and we forecast next-trading-session Yang-Zhang volatility.

Across the six shortlisted models, HGB had the lowest MAE on the common holdout. That is a descriptive result, and we do not present it as a definitive model choice.

The main finding is that forecast-error variance increases with predicted volatility, especially in Q5. Most of the extreme Q5-to-Q1 ratio is concentrated in a few 2026 crash dates, but there is still a smaller high-volatility problem after removing those observations.

The 2026-02-02 event is not simply a clean market shock. It combines missing OHLC data and continuation-series effects with the Warsh and margin-hike context. The 23 March result is associated with US-Iran and risk-off news, but we cannot identify a unique causal effect from one daily observation.

Our three patches did not solve the problem. Student-t improved density, not point forecasts. Segmented parameters performed worse. The log transformation reduced average bias but did not fix Q5 variance.

In conclusion, HGB is a good MAE-based starting point, but high predicted volatility remains the unreliable region. The next sensible step would be to investigate event or jump components, rather than simply adding another GARCH specification.
