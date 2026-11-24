# Plan: LSEG Yang–Zhang report and 20-minute Beamer

Working copy of the Group 2 story plan. Saved 2026-11-20. This is not the formatted report.

## What we agreed

- **Confirmed analysis:** LSEG `GCc1` next-session Yang–Zhang volatility.
- **Three-part story:** (1) data, futures vs spot, Yang–Zhang, one-trading-day horizon, seasonality; (2) model ladder from AR/MA/ARIMA to GARCH/-X, HAR, HGB, then shortlist; (3) Q5 residual-variance problem and three attempted remedies.
- **Headline finding:** among the shortlisted models, forecast-error variance rises when the fitted value is in the top 20% (Q5).
- **Yahoo five-session GARCH-X:** omit from the submitted report and slides.
- **AI reflection and contribution statement:** placeholders only; the group fills them.
- **Process:** draft the **story in Markdown first**. After you approve the narrative, convert it into `Group2_Report.qmd` (15-page PDF) and `report_group2/presentations/Group2_Slides.qmd` (20-minute Beamer).

## Two-step process

### Step 1 — Story draft (this is the next action)

Write `report_group2/narrative-draft.md`: a talking outline, not the formatted report. We iterate on this file until the argument, shortlist, and wording feel right.

It will contain the three-part story below, the numbers we already have, and a clear list of Part 3 experiments that still need a holdout run before we quote results.

### Step 2 — After you approve the story

- Rewrite `report_group2/Group2_Report.qmd` from that draft (A4, 12 pt, 1.5 spacing, 2.5 cm, ≤15 pages, names + student IDs).
- Fill `report_group2/presentations/Group2_Slides.qmd` as the 20-minute Metropolis deck with speaker notes.
- Render `Group2_Report.pdf` and `report_group2/presentations/Group2_Slides.pdf`.
- Update `REPRODUCE.md`, `report_group2/presentations/project-progress.qmd`, and `.ai/SESSION_LOG.md`.
- Leave AI and contribution text as `[team to complete]`.

Do not mix the two instruments or targets. Do not invent results.

---

## Proposed story (for the Markdown draft)

### Part 1 — Why this forecast object (detailed; first section of report and talk)

**Forecast object, said in one sentence.** Next-session (daily) unannualised Yang–Zhang volatility of continuous COMEX gold futures (`GCc1`), not the gold price level and not spot gold.

This part starts from the file we actually have, then answers two design questions: why futures rather than spot, and why Yang–Zhang rather than high-frequency realised volatility or close-to-close volatility.

#### 1. Data source: LSEG / Refinitiv `GCc1`

Source: LSEG Workspace (Refinitiv), exported by `scripts/download_lseg_data.py` on an authenticated Workspace machine. History request: 2010-01-01 through 2026-11-10. The gold series used for the target is `data/refinitiv/GCc1.csv`.

`GCc1` is the generic first COMEX gold futures continuation (continuous contract), not a single fixed-expiry GC contract and not a stitched spot series.

Observed columns in the saved CSV:

| Column | LSEG field | Role |
|---|---|---|
| `Date` | history index | Session date |
| `TRDPRC_1` | LAST | Last trade; used as the close/last price |
| `OPEN_PRC` | Open | Session open |
| `HIGH_1`, `LOW_1` | High, low | Daily range |
| `OPINT_1` | Open interest | Downloaded with the bar; **not** used in Yang–Zhang |

The downloader also requested `ACVOL_1` (volume). It was not returned; the saved file has no volume column. Manifest: 4,241 dated rows; 4,064 complete OHLC bars; incomplete bars are dropped, never filled.

`TRDPRC_1` is a last-trade close proxy, not a settlement-price series. We keep that identity explicit.

#### 2. Why not spot, why forwards/futures

Two reasons, in this order.

**Availability (what the export actually returned).** The same LSEG run requested spot/fixing gold:

- `XAUFIX=` (fixing): permission error (`User does not have permission for this universe`).
- `XAU=` (spot): the universe does not support `TRDPRC_1`.

So a long, continuous **spot OHLC** series was not available from this Workspace account. `GCc1` was: daily open, high, low, last, and open interest from 2010.

**Economic identity.** Even if spot had downloaded, spot gold and COMEX futures are different instruments.

- Spot is OTC / locational; quotes can include premia and delivery frictions.
- COMEX GC is the exchange-traded benchmark; `GCc1` is a continuous generic contract, so we do not manually stitch expiries.
- Futures prices embed the cost of carry (rates, storage, convenience yield). The volatility of that price is the object faced by futures traders, hedgers, and options on futures.

We forecast **gold-forward/futures volatility**. We do not claim it equals spot-gold volatility. We never mix spot and futures in one series.

#### 3. Why Yang–Zhang: daily realised volatility, given only daily bars

The team’s object of interest is **daily realised volatility** of the futures price: how much the contract moved over a trading day, not a multi-day variance sum and not an annualised option quote.

**Ideal estimator: high-frequency realised variance.** If we observed \(M\) intraday log returns \(r_{t,i}=\log(P_{t,i}/P_{t,i-1})\) on day \(t\), the standard realised-variance target would be

\[
RV_t=\sum_{i=1}^{M} r_{t,i}^{2},\qquad
\sigma^{\mathrm{HF}}_t=\sqrt{RV_t}.
\]

That uses the whole path. We do **not** have tick or 5-minute bars from LSEG in this project. The source is one OHLC bar per date. High-frequency RV is therefore not feasible.

**Naive daily alternative: close-to-close.** With only closes \(C_t\),

\[
r^{cc}_t=\log(C_t/C_{t-1}),\qquad
\hat\sigma^{cc,2}_t=\bigl(r^{cc}_t\bigr)^{2}
\]

(or a sample variance of \(r^{cc}\) over several days). This uses two points on the path and **ignores the high–low range and the open**.

**Thought experiment (put this in the report and on a slide).** Two days can have the **same** close-to-close return and very different volatility:

1. The price barely moves: it opens and walks almost straight to the close. High and low sit near the open and close.
2. The price swings widely: the high–low gap is large, but the close ends at yesterday’s close, so \(C_t=C_{t-1}\) and \(r^{cc}_t=0\).

Close-to-close treats these days as identical (including a zero-vol day in case 2). Any estimator that uses the intraday range will record case 2 as much more volatile. Gold futures often gap overnight as well, so the open can differ from the previous close; close-to-close also misses that jump.

**Estimator we use: Yang–Zhang (2000).** AI suggested a daily OHLC estimator; we adopted the published version: Dennis Yang and Qiang Zhang, “Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices,” *Journal of Business* 73(3), 477–492, [https://doi.org/10.1086/209650](https://doi.org/10.1086/209650).

Properties we need, as stated in that paper: unbiased in the continuous limit; independent of drift; accounts for opening jumps; smallest variance among estimators with those properties; large accuracy gain over classical close-to-close on real series.

Implementation matches `src/gold_vol_pipeline/yang_zhang.py`, window \(n=2\). Let \(O_t,H_t,L_t,C_t\) be open, high, low, last. Overnight and open-to-close log returns and the Rogers–Satchell range term are

\[
r_{o,t}=\log(O_t/C_{t-1}),\qquad
r_{c,t}=\log(C_t/O_t),
\]

\[
RS_t=
\log(H_t/O_t)\log(H_t/C_t)
+\log(L_t/O_t)\log(L_t/C_t).
\]

Over a trailing window of \(n=2\) complete bars,

\[
s^{2}_{o,t}=\frac{1}{n-1}\sum_{j=t-n+1}^{t}(r_{o,j}-\bar r_{o,t})^{2},\qquad
s^{2}_{c,t}=\frac{1}{n-1}\sum_{j=t-n+1}^{t}(r_{c,j}-\bar r_{c,t})^{2},
\]

\[
\overline{RS}_t=\frac{1}{n}\sum_{j=t-n+1}^{t}RS_j,\qquad
k=\frac{0.34}{1.34+(n+1)/(n-1)}=\frac{0.34}{4.34}\approx 0.078341.
\]

Yang–Zhang **variance** and the **unannualised daily volatility** we forecast are

\[
\hat\sigma^{2}_{\mathrm{YZ},t}
=s^{2}_{o,t}+k\,s^{2}_{c,t}+(1-k)\overline{RS}_t,\qquad
y_t=\sqrt{\max(0,\hat\sigma^{2}_{\mathrm{YZ},t})}.
\]

Overnight variance uses the previous observed close. Incomplete OHLC windows have no target. Adjacent \(n=2\) windows overlap, so adjacent \(y_t\) and forecast losses are dependent. \(y_t\) is a proxy for latent daily volatility, not a high-frequency realised variance.

**Horizon: one trading day, not one calendar day.** LSEG historical pricing stores this series as **interday daily summaries**. The `Date` column is the date label of that daily bar — the **exchange trading session** for which open, high, low, and last are summarised — not a filled calendar including weekends. LSEG’s Historical Pricing summaries use interval `DAILY` / `P1D`; `DATE` labels the summary period (with `summaryTimestampLabel` start vs end of the bar). See the LSEG Data Guide and Historical Pricing summaries: [LSEG Data Guide](https://developers.lseg.com/content/dam/devportal/en_us/product-docs/lseg-data-guide/).

In `GCc1.csv` this is visible: 2010-01-08 is followed by 2010-01-11 (weekend skipped); holidays are skipped the same way. Incomplete OHLC rows have no Yang–Zhang target, so the next *valid* target can skip an extra session.

So the forecast horizon is **the next trading session’s unannualised daily volatility** \(y_{t+1}\): one COMEX trading day ahead. It is not a five-calendar-day or five-session sum, and it is not annualised.

**Pre-model diagnostics, including seasonality.** Before models, inspect the target:

- Path and clustering (target-split figure).
- Calendar month and weekday means (already computed on 3,801 targets, 2010-01-06 to 2026-11-10).
- Month means: March 0.00899 highest, July 0.00694 lowest; Feb and Apr also high. Weekday means only 0.00766 (Wed) to 0.00819 (Mon).
- HAC joint tests: no weekday rejection. Month-only rejects at 5 lags (\(p=0.012\)) and weakens at 10 and 20 lags (\(p=0.068\), \(0.122\)). Kruskal–Wallis flags months but ignores overlap.

**Seasonality conclusion we will state.** There is a descriptive February–April elevation. It is **not** a stable seasonal dummy we add to the forecasting models. Overlapping \(n=2\) windows make calendar tests lag-sensitive.

**Sample after this construction.** 3,801 valid targets; chronological 80/20: 3,040 training, 761 test (2023-06-02 to 2026-11-10). Sequential one-step refits; the future target is never a predictor at the origin.

**Part 1 claims we will not make.** That futures vol equals spot vol. That Yang–Zhang equals tick-based RV. That \(n=2\) is optimal. That AI “invented” the estimator. That month means are a confirmed seasonal factor.

---

### Part 2 — From linear time series to a shortlist

Pedagogical ladder. Each step is motivated by a limitation of the previous one. Then compare holdout metrics and keep several candidates.

**2.1 Linear models (course AR / MA / ARIMA).** On a transformed target (log volatility in the implemented ARIMA family):

\[
\phi(B)(1-B)^d z_t = c + \theta(B)\varepsilon_t,\qquad z_t=\log y_t.
\]

AR(1), MA(1), ARMA(1,1), and a Hyndman–Khandakar order selected on the training sample (saved run: ARIMA(1,1,3), coefficients refit each origin).

**Limitation.** These models a *conditional mean* of \(z_t\) with roughly constant innovation variance. Gold-futures volatility clusters: quiet days follow quiet days, and large moves cluster. Linear ARIMA does not have a changing conditional variance, so it is a weak description of \(y_t\) even if it can produce a point forecast.

**2.2 ARCH / GARCH / EGARCH.** Model the innovation variance explicitly. ARCH(1) uses only the last shock; GARCH(1,1) adds persistence \(\beta\sigma_{t-1}^2\); EGARCH models \(\log\sigma_t^2\) and allows asymmetric news. These are the course tools for conditional heteroskedasticity.

**Limitation.** They still use only the gold-volatility series (and its own shocks). They do not use the dollar, the options market, or multi-horizon realised features.

**2.3 Exogenous -X models.** After the training-only VIF + HAC/Holm screen, gold IV 1M, gold IV 1Y, and DXY are retained; VIX is not. Range GARCH-X puts the OHLC range and those predictors in the variance recursion. EGARCH-X puts the selected predictors in the ARX mean with EGARCH variance. Joint addition; we do not isolate each predictor.

**2.4 Multi-horizon HAR / HAR-X.** Corsi-style daily, weekly (\(\approx 5\)), and monthly (\(\approx 22\)) log-vol averages, plus OHLC features (and the three predictors in HAR-X). This is persistence at several time scales, not a GARCH recursion.

**2.5 Machine learning: histogram gradient boosting.** Same HAR-X features, additive trees, squared-error loss on log volatility, then exponentiate. No likelihood, so no AIC/BIC.

**2.6 Compare, then shortlist.** Sequential one-step holdout, 761 common dates. Metrics: MAE (headline, already in volatility units), RMSE (outlier-sensitive), signed mean error, and AIC/BIC only inside comparable likelihood families on the *training* origin.

Descriptive ranking we already have (no-external unless noted):

| Model | Role on the ladder | Holdout MAE |
|---|---|---:|
| HGB | ML | 0.002418 |
| HAR-X | multi-horizon | 0.002711 |
| EGARCH-X | -X (selected predictors) | 0.002833 |
| GARCH(1,1) | persistent variance | 0.002912 |
| EGARCH(1,1) | asymmetry | 0.002912 |
| Range GARCH-X | GARCH-X | 0.002914 |
| ARIMA family / AR / MA | linear mean | ≈ 0.0030–0.0033 |
| Persistence / expanding mean | naive | 0.00319 / 0.00391 |

**Shortlist for Part 3:** GARCH, EGARCH, range GARCH-X, EGARCH-X, HAR-X, HGB. Linear AR/ARIMA, plain HAR, Ridge, and Lasso are discussed as steps on the ladder and not retained. MAE ranking is descriptive; overlapping targets; leader chosen on this holdout.

---

### Part 3 — Residuals, Q5, and three remedies

**Diagnosis.** Residual \(e_t=y_t-\hat y_{t\mid t-1}\) (positive = under-forecast). Across the six candidates:

- Average under-forecasting (positive mean \(e_t\)).
- Fan shape: residual dispersion rises with \(\hat y\).
- **Headline:** when \(\hat y\) is in that model’s **top 20% (Q5)**, raw residual variance is about 9–15 times Q1. Relative and log errors do not inflate the same way, so much of the widening is scale, but Q5 errors stay right-skewed and heavy-tailed.
- Shared crash: 2026-02-02, actual \(y\approx 0.0775\), forecasts ≈ 0.01–0.015.

Q5 is a *predicted*-volatility bin, available at the origin. High-vol point forecasts are the least reliable.

**Three remedies (git as of 2026-11-21, `17b2978`):**

1. **Student-t MLE.** Code exists; **holdout summary still missing**. Quote design only until `distribution_summary.csv` exists.

2. **Q5-regime / segmented parameters** (Enoch, 21 Nov). Training 80th percentile; `forecast_trigger` and `lagged_target_trigger`; 12 fits, 761 dates, 0 failures. Segmented MAE is **worse** than the unsegmented six (HGB 0.002680 / 0.002760 vs 0.002418). Failed patch, not a new winner.

3. **Log target + \(\exp(\hat\mu+\tfrac12\hat\sigma^2)\)** (Samuel, 20 Nov). Bias improved; MAE/RMSE not significant after Holm; Q5 variance rose for five models, −0.2% for HGB. Failed Q5 patch that did reduce mean under-forecasting.

**Part 3 punchline.** High predicted volatility is where point forecasts fail. Two implemented fixes did not remove Q5 variance inflation. Student-t remains unevaluated. Do not claim any remedy solved it.

**Limits (say them out loud).** Overlapping two-observation targets; one chronological holdout reused in development; 1M IV vs next-session vol; one-weekday LSEG availability proxy; `GCc1` is a generic continuous futures series, not spot.

**What we will not claim.** That HGB is the true DGP. That VIX or gold IV “causes” gold-futures vol. That Student-t / segmented / Jensen reverse solved Q5. That the Yahoo five-session GARCH-X result still applies.

---

## 20-minute slide map

| Min | Part | Beat |
|---|---|---|
| 0–1 | — | Title: next trading-day unannualised YZ vol of `GCc1` |
| 1–4 | 1 | LSEG `GCc1` columns; why futures not spot; `Date` = trading session |
| 4–7 | 1 | HF RV (no ticks) → close-to-close two-path example → Yang–Zhang formula; horizon = one trading day |
| 7–8 | 1 | Seasonality: Feb–Apr descriptive; no weekday; no seasonal dummies |
| 8–12 | 2 | Ladder: AR/MA/ARIMA limits → GARCH/EGARCH → -X → HAR → HGB; MAE table; shortlist of six |
| 12–16 | 3 | Residual fan; Q5 variance; 2026-02-02 |
| 16–19 | 3 | Student-t (no CSV); segmented worse MAE; log correction bias only |
| 19–20 | — | Limits; AI `[team to complete]`; recap |

## 15-page report map

1. Executive summary.
2. **Part 1 (~4 pages):** LSEG data; futures vs spot; HF RV vs close-to-close vs Yang–Zhang formulas; one-trading-day horizon and LSEG `Date`; seasonality diagnostics.
3. **Part 2 (~4 pages):** AR/MA/ARIMA and their limits; ARCH/GARCH/EGARCH; predictors and -X; HAR; HGB; comparison table; shortlist.
4. **Part 3 (~4 pages):** residual definition; Q5 finding; segmented and log-transform as failed patches; Student-t design-only.
5. Conclusion and limitations.
6. AI `[team to complete]`; contributions `[team to complete]`; Yang and Zhang (2000) and other references.

## Evidence already in the repo vs still to run

**Ready to quote:** `GCc1.csv` structure; spot/fixing download failures; YZ formulas in `yang_zhang.py`; 3040/761 split; seasonality tables and HAC tests; 20-model MAE/RMSE; DM tests; Q5 residual diagnostics; 2026-02-02 misses.

**Git as of 2026-11-21:** log-transform and segmented holdouts exist. Student-t summary CSV still missing.

- Log-target: `outputs/samuel/stage3_q5/log_transform/`. Bias down; MAE/RMSE not significant; Q5 variance not fixed.
- Segmented: `data/processed/stage2_yz_segmented/`. MAE worse than unsegmented shortlist.
- Student-t: quote design only unless `distribution_summary.csv` is written.

**Figures (already in `report_group2/figures/`):** target split; month means; top-five or actual-vs-forecast; residuals vs forecast; error-by-forecast-quantile.

**Omit:** Yahoo `GC=F` five-session realized variance; 25-model Stage 3 table; GARCH-X-with-VIX as the working model.

## Numbers we will quote (from saved artifacts)

- Sample: 4,064 complete OHLC; 177 incomplete; 3,801 targets; 3,040 / 761; test 2023-06-02–2026-11-10.
- Screen: drop gold IV 6M/3M/2M; Holm-significant: gold IV 1M, gold IV 1Y, DXY; VIX not.
- No-external MAE: HGB 0.002418, HAR-X 0.002711, GARCH 0.002912, EGARCH 0.002912, persistence 0.003190.
- EGARCH-X MAE 0.002833 (selected predictors in the mean).
- HGB vs persistence/HAR/GARCH absolute-loss Holm p-values: \(7.95\times10^{-14}\), \(1.15\times10^{-13}\), \(1.78\times10^{-9}\).
- Q5/Q1 raw error-variance ratios ~9–15; relative ~0.96–1.33; 2026-02-02 actual 0.0775.
- Training AIC/BIC only as a labelled, non-comparable supporting table.

Caption every figure for the specification it actually shows (no-external vs joint-predictor vs selected six).

## Course constraints

- Report: `Group2_Report.qmd` + `.pdf`; A4, 12 pt, 1.5 line spacing, 2.5 cm; ≤15 pages including tables, figures, references.
- Names and student IDs: Luo Pui Ki Enoch (1155212655); Li Hung Kui (1155212707); Siriasavakul Hok Sum (1155179729).
- Slides: `Group2_Slides.pdf`; 20 minutes + 5 Q&A; Wed 25 Nov 2026, 8:55–9:20.
- Deadline: Tue 24 Nov 2026, 23:59.

## Out of scope until the group reviews the PDFs

- Filling AI or contribution text (`[team to complete]`).
- Running the Student-t holdout.
- Committing.

## Next action

Report and Beamer sources rewritten from `narrative-draft.md` (2026-11-22). Remaining: team fills AI and contribution placeholders; optional Student-t holdout; commit when the group is ready.
