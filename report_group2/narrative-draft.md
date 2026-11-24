# Group 2 narrative draft

Talking outline for the report and 20-minute Beamer. Not the formatted QMD.
Edit this file until the story is right; then convert it.

**Forecast object.** Next trading-day unannualised Yang–Zhang volatility of continuous COMEX gold futures (`GCc1`). Not the gold price, not spot gold.

**Headline.** Among six shortlisted models, forecast-error variance rises when the predicted value is in the top 20% (Q5). Two patches (log-mean correction; segmented high-vol parameters) did not remove that problem. Student-t is coded but has no saved holdout CSV.

**Omit.** Yahoo `GC=F` five-session realized variance and GARCH-X-with-VIX as a working model.

**Placeholders.** AI use and contribution statement: `[team to complete]`.

---

## Part 1 — Why this object

### Beat 1. The file we have

LSEG Workspace (Refinitiv), `scripts/download_lseg_data.py`, 2010-01-01 to 2026-11-10.
Source: `data/refinitiv/GCc1.csv`. Generic first COMEX gold futures continuation. Not a single expiry, not spot.

| Column | Field | Role |
|---|---|---|
| `Date` | session date of the daily interday bar | trading day, not a filled calendar |
| `TRDPRC_1` | LAST | close proxy (not settlement) |
| `OPEN_PRC`, `HIGH_1`, `LOW_1` | OHLC | Yang–Zhang inputs |
| `OPINT_1` | open interest | downloaded, unused in YZ |

Volume `ACVOL_1` was requested and not returned. 4,241 rows; 4,064 complete OHLC; incomplete bars dropped, never filled.

**Slide / figure:** table of columns; maybe first few rows.

### Beat 2. Why futures, not spot

**Availability first.** Same LSEG run: `XAUFIX=` permission error; `XAU=` does not support `TRDPRC_1`. No long continuous spot OHLC from this account. `GCc1` did download.

**Then economics.** Spot and COMEX futures are different instruments. Futures embed carry. `GCc1` is the exchange-traded continuous contract. We forecast futures volatility. We never mix spot and futures.

### Beat 3. Why Yang–Zhang

We want **daily realised volatility**. Ideal: high-frequency RV

\[
RV_t=\sum_{i=1}^{M} r_{t,i}^{2},\qquad \sigma^{\mathrm{HF}}_t=\sqrt{RV_t}.
\]

No tick/5-minute bars in this project. One OHLC bar per `Date`.

Close-to-close uses only two points and ignores the range:

\[
r^{cc}_t=\log(C_t/C_{t-1}),\qquad \hat\sigma^{cc,2}_t=(r^{cc}_t)^2.
\]

**Two-path cartoon (must be on a slide).** Same close-to-close return:

1. Price walks almost straight from open to close.
2. Price swings; high–low is large; close equals yesterday’s close so \(r^{cc}_t=0\).

Close-to-close treats them as identical. Range estimators do not. Overnight gaps are also missed by close-to-close.

**Yang and Zhang (2000), DOI 10.1086/209650.** AI suggested an OHLC estimator; the team used the published paper. Drift-independent, handles opening jumps, more efficient than close-to-close. Implementation: `src/gold_vol_pipeline/yang_zhang.py`, \(n=2\), \(k\approx 0.078341\).

\[
\hat\sigma^{2}_{\mathrm{YZ},t}=s^{2}_{o,t}+k\,s^{2}_{c,t}+(1-k)\overline{RS}_t,\qquad
y_t=\sqrt{\max(0,\hat\sigma^{2}_{\mathrm{YZ},t})}.
\]

Unannualised. Adjacent windows overlap. Incomplete windows have no target.

**Horizon: one trading day.** LSEG interday `DATE` is the trading session of that daily bar (weekends/holidays skipped: 2010-01-08 then 2010-01-11). Forecast \(y_{t+1}\) at origin \(t\): next valid session, not next calendar day, not a five-session sum.

**Figure:** `figures/target_split.png`. 3,801 targets; 3,040 / 761 chronological; test 2023-06-02 to 2026-11-10. Sequential one-step; future \(y\) never a predictor.

### Beat 4. Seasonality (short)

Month means: Mar 0.00899 highest, Jul 0.00694 lowest; Feb/Apr also high.
Weekdays 0.00766–0.00819. No weekday HAC rejection. Month-only \(p=0.012\) at 5 lags, \(0.068\) at 10, \(0.122\) at 20.

**Line we will say:** descriptive February–April elevation; **not** a seasonal dummy in the forecasting models.

**Figures:** `month_means.png`, `weekday_means.png`.

**Part 1 will not claim:** futures vol = spot vol; YZ = tick RV; \(n=2\) optimal; AI invented the estimator; month means are a confirmed season.

---

## Part 2 — Ladder, then shortlist

Each step is motivated by a limit of the previous one.

1. **AR / MA / ARIMA** on \(z_t=\log y_t\). Course linear mean. Limit: constant innovation variance; gold vol clusters.
2. **ARCH / GARCH / EGARCH.** Conditional heteroskedasticity and (EGARCH) asymmetry. Limit: still only own shocks.
3. **-X.** Training VIF then Holm/HAC: keep gold IV 1M, gold IV 1Y, DXY; VIX drops. Range GARCH-X and EGARCH-X add them jointly.
4. **HAR / HAR-X.** Daily / ~weekly / ~monthly log-vol components plus OHLC.
5. **HGB.** Same features, trees. No likelihood, no AIC/BIC.

**Holdout:** 761 common dates. MAE headline. RMSE complementary. AIC/BIC only inside a likelihood family, training origin, not comparable across families.

| Model | Role | MAE |
|---|---|---:|
| HGB | ML | 0.002418 |
| HAR-X | multi-horizon | 0.002711 |
| EGARCH-X | -X | 0.002833 |
| GARCH(1,1) | persistent variance | 0.002912 |
| EGARCH(1,1) | asymmetry | 0.002912 |
| Range GARCH-X | GARCH-X | 0.002914 |
| AR/MA/ARIMA | linear mean | ≈ 0.0030–0.0033 |
| Persistence | naive | 0.003190 |

HGB vs persistence / HAR / GARCH absolute-loss Holm \(p\): \(7.95\times10^{-14}\), \(1.15\times10^{-13}\), \(1.78\times10^{-9}\). Ranking is descriptive; overlapping targets; leader chosen on this holdout.

**Shortlist for Part 3:** GARCH, EGARCH, range GARCH-X, EGARCH-X, HAR-X, HGB.

**Figures:** `top5_forecasts.png` (joint-predictor top five — caption it as such); `yz_top_models_actual_vs_forecast.png` (HGB vs persistence/HAR/GARCH).

**Part 2 will not claim:** HGB is the DGP; VIX or gold IV causes futures vol.

---

## Part 3 — Q5, then two failed patches

Residual \(e_t=y_t-\hat y_{t\mid t-1}\). Positive = under-forecast.

Shared pattern on the six:

- Positive mean error (under-forecast).
- Fan: dispersion rises with \(\hat y\). **Figure:** `forecast_errors_vs_forecast.png`.
- **Headline:** forecast Q5 (top 20% of \(\hat y\), known at the origin) has raw error variance about **9–15 times Q1**. Relative ratios ~0.96–1.33; log-error ratios ~0.48–0.85. Q5 still skewed and heavy-tailed. **Figure:** `forecast_error_by_forecast_quantile.png`.
- Shared miss **2026-02-02**: actual \(\approx 0.0775\), forecasts \(\approx 0.01\)–\(0.015\).

Calm-period point forecasts are more trustworthy. High predicted volatility is where they fail.

### Remedy A — Student-t MLE (design only)

Fat-tailed likelihood instead of Gaussian QMLE. Code: `distribution_comparison.py`. **No saved holdout summary** (`stage2_yz_distributions/` has the target file only). Do not quote t-vs-Normal MAE until that CSV exists.

### Remedy B — Segmented / Q5-regime parameters (ran; did not win)

Enoch, 21 Nov. `data/processed/stage2_yz_segmented/`. Training-history 80th percentile. Two triggers: pilot full-history forecast; lagged observed target. Regime samples defined by lagged realized vol. 12 combinations, 761 dates, 0 failures.

Segmented MAE is **worse** than unsegmented:

| | Unsegmented MAE | Segmented (forecast trigger) | Segmented (lagged target) |
|---|---:|---:|---:|
| HGB | 0.002418 | 0.002680 | 0.002760 |
| HAR-X | 0.002711 | 0.002883 | 0.002879 |
| GARCH | 0.002912 | 0.003238 | 0.003249 |

Tell as a **failed patch**. Same reused holdout; forecast-trigger mixes a pilot forecast with lagged-vol samples.

### Remedy C — Log \(Y\) + \(\exp(\hat\mu+\tfrac12\hat\sigma^2)\) (ran; bias down, Q5 not fixed)

Samuel, 20 Nov. `outputs/samuel/stage3_q5/log_transform/`. 761 dates, 0 failures. Mean bias moved toward zero for all six. MAE/RMSE vs baseline **not significant** after Holm. On baseline-Q5 dates, residual variance **rose** for five models and fell 0.2% for HGB.

Tell as a **failed Q5 patch** that did reduce average under-forecasting.

### Punchline

High predicted volatility is the unreliable region. Two implemented fixes did not remove Q5 variance inflation. Student-t is unevaluated. Do not say the problem is solved.

**Limits:** overlapping \(n=2\) targets; one chronological holdout reused in development; 1M IV vs next-session vol; LSEG next-weekday availability proxy; `GCc1` is a generic continuous futures series.

---

## Slide map (20 min)

| Min | Part | Beat |
|---|---|---|
| 0–1 | — | Title |
| 1–4 | 1 | `GCc1` columns; futures not spot; `Date` = trading session |
| 4–7 | 1 | HF RV → close-to-close cartoon → YZ formula; one trading day |
| 7–8 | 1 | Seasonality, no dummies |
| 8–12 | 2 | Ladder + MAE table + shortlist |
| 12–16 | 3 | Fan, Q5, 2026-02-02 |
| 16–19 | 3 | Student-t (no CSV); segmented worse MAE; log correction bias only |
| 19–20 | — | Limits; AI `[team to complete]` |

## Report map (≤15 pages)

1. Executive summary.
2. Part 1 (~4 pp).
3. Part 2 (~4 pp).
4. Part 3 (~4 pp).
5. Conclusion.
6. AI `[team to complete]`; contributions `[team to complete]`; Yang and Zhang (2000).

## Open for the group

- Approve or edit this story.
- Run Student-t holdout or leave it as design-only.
- Then convert to `Group2_Report.qmd` and `report_group2/presentations/Group2_Slides.qmd`.
