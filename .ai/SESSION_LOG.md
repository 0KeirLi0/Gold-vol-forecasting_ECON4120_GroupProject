# AI and team handoff log

This is an append-only record of project context that may help future sessions. Confirm current implementation and Git state before acting on entries.

## 2026-11-22 — Updated Beamer with geopolitical event context
- Author: Copilot at Samuel's request
- Type: implementation
- Context: The report now distinguishes 2026-02-02 and 2026-03-23 event associations from causal claims.
- Outcome: Added a dedicated 23 March slide and expanded the 2 February slide with descriptive Warsh, CME-margin, missing-OHLC, and continuation-series context in `presentations/Group2_Slides.qmd`; strengthened the recap caveat.
- Artifacts: `presentations/Group2_Slides.qmd`, `presentations/Group2_Slides.pdf` (pending re-render).
- Next: Re-render the Beamer PDF and inspect it before any commit.

## 2026-11-22 — Added geopolitical context for the largest residual misses
- Author: Copilot at Samuel's request
- Type: finding
- Context: The report identified 2026-02-02 and 2026-03-23 as extreme residual dates but needed contemporaneous market-event context and a clear limit on causal interpretation.
- Outcome: Updated `report_group2/Group2_Report.qmd` and `report_group2/diagnostics/spike_market_events.csv`. Reuters reporting connects 2026-01-30/02-02 to the Kevin Warsh nomination and a CME gold-margin increase from 6% to 8%; the report retains the missing-OHLC and continuation-roll explanation. For 2026-03-23, Reuters coverage associates the move with US-Iran war uncertainty and risk-off trading, without claiming a proven unique causal effect.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/diagnostics/spike_market_events.csv`.
- Next: Re-render the report and inspect the PDF before any commit.

## 2026-11-22 — Updated 20-minute Beamer to match Student-t / robustness report
- Author: Grok with Keir
- Type: implementation
- Context: Report already had Student-t Q5, spike influence, 2025–26 split, n-window, month dummies, economic MAE, and filled AI/contributions. Slides were still the earlier 21-frame story.
- Outcome: Rewrote `presentations/Group2_Slides.qmd` (Metropolis 16:9, 27 frames). Headline is residual-first with three failed patches. New frames: GCc1 price regime, YZ series, spike influence, Student-t results, confirmatory split, window ACF. Copied PDF to `report_group2/Group2_Slides.pdf`.
- Artifacts: `presentations/Group2_Slides.qmd`, `presentations/Group2_Slides.pdf`, `report_group2/Group2_Slides.pdf`.
- Next: Team timing pass if 27 frames is tight for 20 minutes.

## 2026-11-22 — Ran Student-t Q5 holdout and remaining robustness checks
- Author: Grok with Keir
- Type: implementation | finding
- Context: Reviewer notes that Student-t was coded but unscored, the 761-day holdout was reused, 2026-02-02 lacked a quantitative influence/event write-up, n=2 and month effects were untested, and AI/contributions/economic MAE text were placeholders.
- Outcome: Completed the six-model Normal vs Student-t sequential holdout (761 origins, 0 failures). Student-t df medians are about 4.1–4.4 (HGB residual df 3.51). Density scores improve; GARCH-family MAE is slightly worse; HGB point MAE is unchanged; Q5/Q1 stays about 9.2–9.5 for GARCH, 15.76 for HAR-X (from 18.55), 13.39 for HGB. Dropping 2026-02-02 cuts Q5 variance by about one third; top five misses cut 59–72%; remaining Q5/Q1 is about 3.2–6.2. Reuters 2026-02-02: Warsh nomination, Friday crash, CME gold margins 6% to 8%; GCc1 27–28 Jan incomplete OHLC delay the first valid YZ to 2 Feb (y=0.077521). Confirmatory 2023–24 vs 2025–26 split: HGB then HAR-X remain ranks 1–2; Q5 is much weaker in 2023–24. Window n=5/10 raises lag-1 ACF to 0.90/0.96 and makes persistence the MAE winner; Q5/Q1 explodes. HAR-X month dummies slightly worsen MAE (0.002711 to 0.002739) and Q5/Q1. Economic illustration: HGB vs persistence MAE gap 0.000772 ≈ 1.23 annualised vol points ≈ $400 per 100 oz at mean holdout price. AI and contribution sections filled. Report PDF re-rendered. This is not a pristine unused holdout.
- Artifacts: `scripts/report_feedback_robustness.py`, `scripts/copy_distribution_report_tables.py`, `report_group2/diagnostics/` (q5_error_influence, holdout_subperiod_metrics, window_*, harx_month_dummy_*, distribution_*), `report_group2/figures/q5_error_influence.png`, `holdout_subperiod_mae.png`, `window_acf.png`, `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`.
- Next: Team review and page cut; optional Beamer update; commit when asked. Do not invent a later unused holdout.

## 2026-11-22 — Added report diagnostics pipeline (ADF, ACF/PACF, forecasts, residuals)
- Author: Copilot with Keir
- Type: implementation | finding
- Context: Report needed Yang–Zhang and gold-price time plots, ACF/PACF, residual-time plots, sequential one-step forecasts, Dickey–Fuller with the stationary alternative and default AIC lags, and an explicit parameter-selection write-up (AIC/BIC/Ljung–Box). Page limit deferred until after review.
- Outcome: Added `src/gold_vol_pipeline/report_diagnostics.py` and `scripts/report_yz_diagnostics.py`. Saved artifacts under `report_group2/diagnostics/` and new figures. ADF rejects a unit root in $y_t$ and $\log y_t$ (train $p\approx 3\times10^{-12}$ and $6\times10^{-11}$) and does not reject for the GCc1 last price (train $p=0.59$, full $p=0.97$). HK still selects ARIMA(1,1,3) because KPSS on $\log y_t$ rejects stationarity. Training $y_t$ ACF is 0.65 at lag 1 and remains about 0.14 at lag 30. Holdout Ljung–Box rejects white-noise forecast errors for all six shortlisted models. GARCH vs EGARCH training AIC is 1291.8 vs 1286.6; orders remain course defaults. Report now includes these plots, tables, interpretation, and next-step recommendations. Student-t holdout is still missing.
- Artifacts: `scripts/report_yz_diagnostics.py`, `src/gold_vol_pipeline/report_diagnostics.py`, `tests/test_report_diagnostics.py`, `report_group2/diagnostics/`, `report_group2/figures/gcc1_price_date.png`, `yz_volatility_date.png`, `yz_acf_pacf.png`, `log_yz_acf_pacf.png`, `shortlist_sequential_forecasts.png`, `shortlist_residuals_time.png`, `report_group2/Group2_Report.qmd`.
- Next: Team review of the expanded report (page count will exceed 15 until content is cut); optional Student-t holdout.

## 2026-11-22 — Expanded report bibliography after reviewer comment
- Author: Copilot with Keir
- Type: implementation
- Context: Reviewer note that the report listed only three references and missed ARCH/GARCH originals, gold FIGARCH/asymmetric GARCH papers, and HAR on gold/commodities.
- Outcome: In-text cites now cover Engle (1982), Bollerslev (1986), Nelson (1991), Tully and Lucey (2007), Baur (2012), Baillie–Bollerslev–Mikkelsen (1996), Bentes (2015), Chkili–Hammoudeh–Nguyen (2014), Corsi (2009), Todorova (2017), Diebold–Mariano (1995), Newey–West (1987), Harvey–Leybourne–Newbold (1997), Holm (1979), Hyndman–Khandakar (2008), Rogers–Satchell (1991), and Yang–Zhang (2000). The report states that FIGARCH/FIAPARCH are not estimated; the ladder stays with course EGARCH. References section has 17 entries. Rendered report is 15 pages (at the A4 12pt 1.5 limit). Beamer has a one-line literature note on the ladder slide (21 frames).
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`, `presentations/Group2_Slides.qmd`, `presentations/Group2_Slides.pdf`, `report_group2/Group2_Slides.pdf`.
- Next: Team review of the PDFs; fill AI and contribution text; optional Student-t holdout; commit when ready.

## 2026-11-22 — Converted Yang–Zhang narrative into report and Beamer
- Author: Copilot with Keir
- Type: implementation | handoff
- Context: After the three-part narrative in `report_group2/narrative-draft.md`, convert to the 15-page report and 20-minute slides. Yahoo five-session GARCH-X omitted. Student-t remains design-only (no `distribution_summary.csv`).
- Outcome: Rewrote `report_group2/Group2_Report.qmd` (PDF-only) and `presentations/Group2_Slides.qmd`. Rendered both PDFs; copied slides into `report_group2/`. Report is A4 12pt 1.5 spacing, 14 pages. Headline is Q5 residual-variance inflation and two failed patches, not HGB as winner. AI and contribution sections left as `[team to complete]`.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`, `presentations/Group2_Slides.qmd`, `presentations/Group2_Slides.pdf`, `report_group2/Group2_Slides.pdf`, `report_group2/figures/variance_by_forecast_quintile.png`, `REPRODUCE.md`, `presentations/project-progress.qmd`.
- Next: Team review of the PDFs; fill AI and contribution text; optional Student-t holdout; commit when ready. Do not mix Yahoo and LSEG targets.

## 2026-11-21 — Narrative draft for LSEG Yang–Zhang report/Beamer
- Author: Copilot with Keir, after team confirmed LSEG Yang–Zhang
- Type: handoff
- Context: Group 2 deliverable will use LSEG `GCc1` next-session Yang–Zhang, not Yahoo five-session GARCH-X. Story is three parts: object/data/YZ; model ladder; Q5 residuals and remedies.
- Outcome: Wrote `report_group2/narrative-draft.md` and updated `report_group2/plan.md`. Git `17b2978` already has log-transform (bias down, Q5 not fixed) and segmented-regime forecasts (MAE worse than unsegmented). Student-t holdout summary CSV is still missing; the draft treats it as design-only.
- Artifacts: `report_group2/narrative-draft.md`, `report_group2/plan.md`.
- Next: Team edit/approve the narrative; then convert to `Group2_Report.qmd` and `presentations/Group2_Slides.qmd`. Optional Student-t run before quoting t-vs-Normal numbers. AI and contribution text remain `[team to complete]`.

## 2026-11-21 — Refined the log-transformation report
- Author: Copilot at Samuel's request
- Type: implementation
- Context: Samuel asked to complete and refine the log-transformation report without changing existing results or statistics.
- Outcome: Expanded the report with the segmented-estimation proposal and results, exact baseline-versus-segmented MAE/RMSE comparison, segmented mean-error signs normalized to actual-minus-forecast, a residual timing/source-of-error diagnostic plan, and links to Student-t and residual reports. Existing experiment numbers and test statistics were preserved.
- Artifacts: `reports/yang-zhang-log-transformation.qmd`, `reports/yang-zhang-log-transformation.html`.
- Next: Complete residual plotting/source analysis and Normal-versus-Student-t review before a final model decision.

## 2026-11-21 — Added segmented Yang-Zhang parameter estimation
- Author: User-directed implementation by Copilot
- Type: implementation | finding
- Context: The user requested separate normal/extreme parameter sets for GARCH, EGARCH, GARCH-X, EGARCH-X, HAR-X, and histogram gradient boosting, selected either by an extreme preliminary forecast or by an extreme lagged target.
- Outcome: Added `segmented_forecasting.py` and `evaluate_segmented_yang_zhang.py`. The two triggers use the training-history 80th percentile: `forecast_trigger` fits a pilot full-history forecast and selects the extreme parameter set when the pilot exceeds the threshold; `lagged_target_trigger` uses the latest observed target. Each selected set is refit on prior observations whose lagged realized volatility belongs to the corresponding normal/extreme regime. All 12 model/trigger combinations completed 761 holdout dates with no fit failures.
- Artifacts: `src/gold_vol_pipeline/segmented_forecasting.py`, `scripts/evaluate_segmented_yang_zhang.py`, `data/processed/stage2_yz_segmented/`, `tests/test_yang_zhang.py`, `REPRODUCE.md`.
- Limitations: The result is exploratory, uses the previously examined holdout, overlapping targets, and a training-history quantile threshold. The forecast-trigger design uses a preliminary forecast to select a regime, while its regime-specific parameter samples are defined by lagged observed volatility; it is not evidence of a final deployable model.
- Next: Compare these segmented forecasts with the unsegmented six-model baselines on a later untouched holdout before recommending a model.

## 2026-11-20 — Published the log-transformation report
- Author: Copilot at Samuel's request
- Type: implementation
- Context: Samuel asked to save the six-model log-target experiment as a report under `reports/`.
- Outcome: Added a Quarto source report and rendered a self-contained HTML version. The report presents the bias-corrected retransformation, common-holdout metrics, Q5 variance comparisons, paired tests, and limitations; detailed per-origin outputs remain under `outputs/samuel/stage3_q5/log_transform/`.
- Artifacts: `reports/yang-zhang-log-transformation.qmd`, `reports/yang-zhang-log-transformation.html`.
- Next: Samuel to share/review the report with Keir and Enoch.

## 2026-11-20 — Log-target bias correction across six Yang-Zhang models
- Author: Copilot analysis at Samuel's request, based on suggestions from Keir and Enoch
- Type: finding
- Context: Samuel asked to fit all six shortlisted volatility models to log Yang-Zhang volatility and back-transform with `exp(mu_hat + 0.5 * sigma_hat^2)`.
- Outcome: Ran 761 sequential one-step forecasts per model with no fit failures, using only observations available through each origin. After Holm-adjusted paired tests, no model's overall MAE or RMSE difference from its saved baseline was significant at 5%; HGB's squared-error test was closest (Holm p=0.058 at Newey-West bandwidth 1; p=0.142 at bandwidth 4). On identical baseline-Q5 dates, residual variance rose for five models and fell 0.2% for HGB. The log-target Q5/Q1 raw variance ratios remained 9.65–20.93.
- Artifacts: `scripts/samuel_stage3_q5_log_transform.py`, `outputs/samuel/stage3_q5/log_transform/`.
- Limitations: The correction assumes normal log errors; HAR-X/HGB use training residual variance, with HGB variance estimated in-sample; Q5 and the model comparison use the same retrospective holdout.
- Next: Samuel to review with Keir and Enoch; no Student-t refit has been run.

## 2026-11-20 — Samuel's Q5 forecast-error diagnosis
- Author: Samuel analysis
- Type: finding
- Context: Samuel requested q5 diagnostics for the six-model LSEG `GCc1` Yang–Zhang shortlist, preserving current model definitions and actual-minus-forecast errors.
- Outcome: On 761 common holdout targets, all six models had larger raw forecast-error variance in forecast Q5 than Q1 (Q5/Q1 ratios 9.2–15.1). Relative-error ratios were 0.96–1.33 and log-error ratios 0.48–0.85, while Q5 errors remained strongly skewed and heavy-tailed. Four-lag HAC regressions found positive raw squared-error/forecast slopes for all six models. Regenerated the sequential ARX-mean EGARCH-X forecasts at all 761 origins with no fit failures. The 2026-02-02 EGARCH-X miss (63.912 in 10^-3 units) falls within the report's published six-model miss range; no model-specific EGARCH-X value is provided there. HGB Q5 mean/skew differ from the report because the report's HGB uses joint predictors while this run uses the baseline no-external-predictor forecast.
- Artifacts: `scripts/samuel_stage3_q5_diagnose.py`, `outputs/samuel/stage3_q5/`.
- Limitations: Q5 uses full-holdout forecast ranks; realized-volatility bins are retrospective; the origin-volatility bins use training-sample cutoffs. The single holdout has prior model-development use.
- Next: Samuel to review Step 1 results before any Student-t refit.

## 2026-11-16 — Added separate Yang-Zhang findings to the Group 2 report
- Author: Copilot report update at Samuel's request
- Type: implementation
- Context: Samuel asked to include the model-analysis evaluation in the main report source used for both the report PDF and Beamer slide PDF.
- Outcome: Added the LSEG `GCc1` next-session Yang-Zhang results as a supplementary, clearly separated evaluation; retained the existing five-session Yahoo/FRED GARCH-X conclusion. Updated the main report date, refreshed the report and slide PDFs, and clarified the PDF reproduction command.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`, `report_group2/Group2_Slides.pdf`, `REPRODUCE.md`, `presentations/project-progress.qmd`.
- Next: Samuel and team to review the distinction between targets before submission.

## 2026-11-16 — Samuel's Yang-Zhang model evaluation
- Author: Copilot analysis at Samuel's request
- Type: finding
- Context: Samuel requested a paired out-of-sample comparison, statistical tests, gold implied-volatility reference, calibration, and residual diagnostics for Enoch's Yang-Zhang forecasts.
- Outcome: Read forecast artifacts from pinned test-branch commit `988adfcdaf80015cce7b2a2d15fccf5dfbf9d8a4` without switching from main. Across 761 common test targets, histogram gradient boosting had the lowest MAE and RMSE; its absolute loss was significantly lower than persistence, HAR, and GARCH(1,1) after Holm correction. Raw and training-scaled 1M gold IV had higher MAE than the selected leader; scaled IV was not distinguished from persistence. Adding VIX did not significantly improve the six predictor models; selected external predictors significantly improved VAR absolute loss only. HAC mean-error tests found under-forecasting in 18 of 20 base models.
- Artifacts: `scripts/samuel_yz_evaluation.py`, `outputs/samuel/yz_*`.
- Limitations: The overlapping two-observation target, single holdout, same-test model selection, 1M-versus-1D IV maturity mismatch, and external-predictor availability proxy constrain interpretation.
- Next: Samuel to review the summary with Enoch and Keir; use a later untouched holdout before treating the winner as a durable model choice.

## 2026-11-16 — Samuel's Stage 3 model comparison
- Author: Copilot analysis at Samuel's request
- Type: finding
- Context: Samuel requested an out-of-sample comparison, benchmark analysis, calibration, residual checks, and workspace outputs using the available project data.
- Outcome: The requested Stage 4 Yang–Zhang forecast artifacts were absent, so the reproducible analysis used the existing Stage 3 five-session `GC=F` realized-variance forecasts instead. Across 25 models and 443 common origins, GARCH-X with VIX had the lowest RMSE; no model beat expanding historical variance under MSE after Holm correction, while VAR had significantly higher MSE. The LSEG 1M IV series includes one prior quote of 0.132% in the test period, which inflates its QLIKE; raw and sensitivity results are clearly distinguished in the output.
- Artifacts: `scripts/samuel_model_comparison.py`, `outputs/samuel/`.
- Next: Samuel/team to confirm the intended target and supply the missing Stage 4 Yang–Zhang forecasts before presenting any of these results as YZ evidence; verify the flagged IV observation with the data owner.

## 2026-11-15 — Focused the Group 2 deliverable on GARCH-X
- Author: User-directed implementation by Copilot
- Type: handoff
- Context: The user asked to reduce the report to GARCH-X and basic comparisons, archive other explorations, clarify project pointers for people and AI, and commit the finished work.
- Outcome: The active report and same-source Beamer deck now focus on lagged-VIX GARCH-X plus seven basic benchmarks. On the integrated 443-origin Yahoo/FRED sample, GARCH-X has the lowest MSE in the 25-model comparison but does not significantly beat expanding historical variance after Holm correction. This is a user-directed working focus, not a confirmed team-wide decision. The broad report and previous Group 2 outputs are preserved under `archive/model_exploration_2026-11-15/`; implementation code and local evaluation inputs remain in place.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`, `report_group2/Group2_Slides.pdf`, `report_group2/figures/core_forecasts.png`, `scripts/plot_group2_core_models.py`, `README.md`, `llms.txt`, `REPRODUCE.md`, `presentations/project-progress.qmd`, `archive/model_exploration_2026-11-15/`.
- Next: Group members to confirm the required agreed contribution statement; review GARCH-X's constant-future-VIX assumption and evaluate on a later holdout before treating the model choice as a team decision.

## 2026-11-15 — Added a Beamer deck to the Group 2 report source
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: The user asked for a modern Beamer presentation generated from the same source file as the report.
- Outcome: Added format-specific content to `Group2_Report.qmd`, using a Metropolis 16:9 theme for a 10-slide presentation while retaining the 11-page A4 report output. Rendered and visually checked both PDFs; report and slide content remain separated by output format.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`, `report_group2/Group2_Slides.pdf`

## 2026-11-15 — Added a predictor inventory to the Group 2 report
- Author: User-directed report update by Copilot
- Type: implementation
- Context: The user asked for a table listing all predictors used in the report.
- Outcome: Added a consolidated inventory of historical gold inputs and external predictor candidates, distinguishing the Yahoo/FRED integrated comparison from the separate, partial LSEG exploratory screen. Documented unavailable LSEG fields and emphasized that not every candidate enters every model.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`

## 2026-11-15 — Made the recommended GARCH-X forecast equation explicit
- Author: User-directed report update by Copilot
- Type: implementation
- Context: The user asked for the selected model to be written explicitly, in a form comparable to a regression equation.
- Outcome: Expanded the report's GARCH-X section to define the return and variance process, normalized lagged-VIX regressor, parameter constraints, constant-future-VIX assumption, and five-session realized-variance forecast. Clarified that GARCH-X is a conditional-variance model rather than an ordinary linear regression.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`

## 2026-11-15 — Updated Group 2 report with integrated volatility-model comparison
- Author: User-directed report update by Copilot
- Type: handoff
- Context: The user asked to review the latest integrated Stage 3 commit, add quantile GARCH and GARCH-X with lagged VIX to the Group 2 report, and identify the best model.
- Outcome: Updated the Group 2 report with the GARCH-X and quantile-GARCH specifications, a 443-common-origin metric comparison, and the criterion-dependent conclusion. GARCH-X has the lowest MSE; asymmetric EGARCH has the lowest MAE and QLIKE. Holm-adjusted tests do not establish that GARCH-X beats expanding historical variance. Clarified that the integrated ranking uses Yahoo/FRED data and is distinct from the exploratory Refinitiv/LSEG screen.
- Artifacts: `report_group2/Group2_Report.qmd`, `report_group2/Group2_Report.pdf`
- Next: Team to agree on the primary forecast loss and review GARCH-X's constant-future-VIX assumption before treating the MSE-based recommendation as final.

## 2026-11-15 — Integrated Stage 3 model comparison and recommendation
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested the previously excluded models in Stage 3 and a final model recommendation. GARCH-X is implemented as a direct zero-mean lagged-VIX variance equation with VIX held fixed over the forecast horizon, not the previously proposed two-step design.
- Outcome: On 443 shared successful origins, GARCH-X has lowest MSE (9.492e-7; RMSE 0.000974); asymmetric EGARCH has lowest MAE and QLIKE. GARCH-X is provisionally recommended under an MSE-first criterion, pending team review. It does not significantly beat historical variance after Holm correction. Two of 445 quantile-GARCH fits failed; other newly refitted candidates succeeded. Optimizer warnings and common-success selection limit the result.
- Artifacts: `notebooks/06_final_evaluation.ipynb`, `src/gold_vol_pipeline/stage3_evaluation.py`, `src/gold_vol_pipeline/models.py`, `data/processed/stage3_integrated_*`, `reports/model-performance-stage2-stage3.qmd`, `README.md`, `REPRODUCE.md`, `presentations/project-progress.qmd`.
- Next: Samuel to review the MSE criterion, EGARCH trade-off, inference, and later-holdout robustness; Enoch to review GARCH-X's constant-future-VIX assumption.

## 2026-11-15 — Exploratory quantile GARCH case
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested an explicit GARCH equation, calibration, and a quantile term for extreme lagged observations. With clarification unavailable, the implemented exploratory interpretation uses the 95th percentile of historical absolute daily `GC=F` returns and a nonnegative extra squared-shock coefficient.
- Outcome: At the 2026-11-03 origin, 2,976 returns yielded threshold 0.0215304 and constrained tail coefficient zero; the five-session simulated forecast was 0.00082648. This is one fitted case, not evidence of significance or a comparison with existing models. Optimizer clipping warnings remain; threshold estimation and coefficient-boundary inference need care.
- Artifacts: `src/gold_vol_pipeline/quantile_garch.py`, `scripts/calibrate_quantile_garch.py`, `tests/test_quantile_garch.py`, `reports/model-performance-stage2-stage3.qmd`, `REPRODUCE.md`, `presentations/project-progress.qmd`.
- Next: Enoch to confirm the intended quantile specification; Samuel to consider an appropriate common-origin evaluation and inference before interpreting model performance.

## 2026-11-13 — Refit GARCH on all aligned origins
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested GARCH forecasts on all 445 Stage 3 dates rather than the previous 435-date complete-case comparison.
- Outcome: The GARCH fit now scales historical returns by their root mean square, uses a consistent feasible initialization, checks optimizer success and parameter constraints, and converts forecast variance back to original units. All 445 fits passed; intermediate bounds-clipping warnings remain. On 445 common origins GARCH had lowest MSE (9.696e-7; RMSE 0.000985), HAR had lowest MAE (0.000416), and no model significantly beat historical variance after Holm correction. VAR had significantly larger MSE and QLIKE. Eight selected fits agreed in negative log-likelihood within 1e-8 across two starting values, not a proof of global optimality.
- Artifacts: `src/gold_vol_pipeline/models.py`, `scripts/refit_stage3_garch.py`, aligned forecasts, summaries, figures, report, presentation, and `REPRODUCE.md`.
- Next: Samuel to review the 445-origin loss and inference results; Enoch to review numerical warnings and source-date freshness.

## 2026-11-13 — Documented GARCH scaling investigation
- Author: User-directed documentation update by Copilot
- Type: handoff
- Context: The user requested updated reports after the exploratory GARCH scaling check.
- Outcome: The performance report and project progress presentation distinguish the 10-origin diagnostic from the unchanged 435-origin comparison; reproduction instructions describe the diagnostic and its limitations. The report PDF and presentation HTML and PDF were regenerated; the presentation PDF uses the existing Quarto document format rather than a browser slide export.
- Artifacts: `reports/model-performance-stage2-stage3.qmd` and PDF, `presentations/project-progress.qmd`, HTML and PDF, `REPRODUCE.md`.
- Next: Validate any proposed scaling change across all 445 origins and refresh comparisons before drawing new conclusions.

## 2026-11-13 — Investigated GARCH convergence failures
- Author: Copilot investigation
- Type: finding
- Context: The Stage 1 GARCH(1,1) fit failed on 10 of 445 Stage 3 origins; six reported incompatible inequality constraints and four reached the iteration limit.
- Outcome: In an exploratory runtime check, dividing each origin's historical returns by their root mean square, fitting the existing model, and converting its variance forecast back to original units produced finite forecasts at all 10 previously failed origins. The optimizer emitted bounds-clipping warnings. This suggests numerical scaling is a promising fix, not proof of optimal fits or an updated full-sample result.
- Artifacts: `src/gold_vol_pipeline/models.py`, `data/processed/stage1_on_stage3_origins_forecasts.csv`.
- Next: If adopted, validate constraints and fit quality, rerun all origins and aligned comparisons, and update the report and presentation; do not treat the exploratory forecasts as published results.

## 2026-11-13 — Added fixed-order HK auto-ARIMA to aligned comparison
- Author: User-directed implementation by Copilot
- Type: finding
- Context: Enoch requested the existing Hyndman-Khandakar algorithm as a return-only benchmark, with one parameter set across observations.
- Outcome: Selected ARIMA(3,1,1) by KPSS and stepwise AICc on squared returns available at the first forecast origin, 2018-01-05; kept the order fixed and refit coefficients using expanding data at each origin to avoid look-ahead. All 445 fits produced forecasts. The 20-model complete-case comparison remains 435 dates because GARCH(1,1) failed on 10; HK MSE was 1.099e-6, RMSE 0.001048, MAE 0.000483, and QLIKE 1.895. Holm-adjusted DM p-values against historical variance were 1.000 (MSE) and 0.943 (QLIKE). This is exploratory, not a model decision; the fixed-order interpretation does not freeze fitted coefficients.
- Artifacts: `scripts/evaluate_stage3_auto_arima.py`, `data/processed/stage3_auto_arima_forecasts.csv` and manifest, aligned comparison tables, `reports/model-performance-stage2-stage3.qmd`, `presentations/project-progress.qmd`, `REPRODUCE.md`.
- Next: Samuel to review the aligned loss, inference, and fixed-order sensitivity evidence; Enoch to confirm source-date freshness.

## 2026-11-13 — Added VAR to aligned Stage 3 comparison
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The Stage 3 report and aligned comparison had omitted VAR even though the modeling package contained a BIC-selected VAR implementation.
- Outcome: Refit the established VAR using log daily squared returns and log VIX, selecting up to five lags by BIC at each of the 445 forecast origins. VAR returned forecasts at all origins; the balanced comparison remains 435 origins because GARCH(1,1) failed at 10. The 19-model results show VAR MSE and QLIKE above expanding historical variance (ratios 1.229 and 7.259); Holm-adjusted DM p-values are 0.000071 and 1.42e-7. Added a reproducible evaluator, manifest, forecast-versus-actual chart, report entries, and reproduction instructions. Refreshed the report PDF and project progress HTML.
- Artifacts: `scripts/evaluate_stage3_var.py`, `data/processed/stage3_var_forecasts.csv`, `data/processed/stage3_var_manifest.json`, `scripts/plot_stage3_all_model_forecasts.py`, `reports/model-performance-stage2-stage3.qmd`, `reports/model-performance-stage2-stage3.pdf`, `presentations/project-progress.qmd`
- Next: Samuel to review the aligned loss and test evidence, including the VAR's higher losses, before making any model recommendation.

## 2026-11-13 — Added all Stage 3 model forecasts to appendix
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: The report appendix initially showed forecast-versus-actual charts only for the seven Stage 1 baselines, while the aligned Stage 3 comparison contains 18 models.
- Outcome: Added a reproducible plotting script that reads the saved combined forecasts without refitting models, generates one chart per model over the 445 requested origins, and validates all 18 model names and origin counts. Updated the appendix to include all charts and regenerated the 24-page PDF. GARCH(1,1) gaps remain visible for its 10 failed fits; the comparison table continues to use the 435-origin balanced subset.
- Artifacts: `scripts/plot_stage3_all_model_forecasts.py`, `reports/figures/stage3_all_model_forecasts/`, `reports/model-performance-stage2-stage3.qmd`, `reports/model-performance-stage2-stage3.pdf`
- Next: Samuel to review the aligned loss, DM, and Mincer-Zarnowitz evidence and recommend a next step without assuming a selection criterion.

## 2026-11-13 — Aligned Stage 1 baselines with Stage 3 evaluation
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The Stage 1 baselines were freshly rerun on Stage 3's 445 requested weekly dates so the final model comparison would use common origins.
- Outcome: Six baselines forecast all dates; GARCH(1,1) failed at 10. The combined comparison and report use 435 dates with complete forecasts for all 18 models. GARCH(1,1) had the lowest point MSE, RMSE, and MAE, but no candidate beat historical variance after Holm correction for MSE or QLIKE. Rendered and checked the 13-page PDF report, including all seven baseline forecast-versus-actual appendix charts, and refreshed the progress presentation HTML/source. Results remain exploratory; no model decision is asserted.
- Artifacts: `scripts/evaluate_stage1_on_stage3_origins.py`, `data/processed/stage1_on_stage3_origins_*`, `data/processed/evaluation_*_with_stage1*`, `reports/model-performance-stage2-stage3.qmd`, `reports/model-performance-stage2-stage3.pdf`, `reports/figures/stage1_on_stage3_origins/`, `presentations/project-progress.qmd`
- Next: Samuel to review the aligned loss, DM, and Mincer-Zarnowitz evidence and recommend a next step without assuming a selection criterion.

## 2026-11-13 — Drafted Stage 2 and Stage 3 performance report
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The team requested a report of baseline performance, predictor significance, and advanced-model performance with explanatory tables and charts.
- Outcome: Created a Quarto report that keeps Stage 1, Stage 2 Yahoo/FRED, separate LSEG, and Stage 3 common-origin evidence distinct. The report includes the Stage 2 training-only HAC/Holm screen, holdout HistGradientBoosting comparison, Stage 1 baseline errors, and the 17-model Stage 3 comparison. Results remain exploratory; no final model choice is asserted.
- Artifacts: `reports/model-performance-stage2-stage3.qmd`, `reports/model-performance-stage2-stage3.pdf`, `README.md`
- Next: Samuel to review the reported evidence, caveats, and next-step implications.

## 2026-11-13 — Rendered updated project progress presentation
- Author: User-directed execution by Copilot
- Type: finding
- Context: The Stage 3 fast common-origin evaluation was documented in the project progress source.
- Outcome: Rendered `presentations/project-progress.qmd` to RevealJS HTML and PDF; checked that the generated outputs contain the Stage 3 findings.
- Artifacts: `presentations/project-progress.qmd`, `presentations/project-progress.html`, `presentations/project-progress.pdf`
- Next: Samuel to review the Stage 3 loss, DM, and MZ evidence and recommend a next step without assuming a selection criterion.

## 2026-11-13 — Reduced and reran the Stage 3 evaluation package
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The original unified notebook configured repeated ARIMA, VAR, GARCH-extension, and rolling-CV fits; prior completed weekly forecasts were already available for fast baselines, HAR/HARX, and all nine predictor-set variants.
- Outcome: Reworked `notebooks/06_final_evaluation.ipynb` to reuse completed forecasts and compare 17 models on their 445 shared origins (2018-01-05 to 2026-11-03), without refitting. EWMA had the lowest MSE (9.76e-7), HAR the lowest MAE (0.000416), and VIX-only lasso the lowest MSE among predictor variants (9.85e-7). No candidate beat the historical-variance benchmark after Holm correction for MSE or QLIKE. Auto-ARIMA, VAR, GJR-GARCH, asymmetric EGARCH, Student-t GARCH, and GARCH-X were excluded from the short run for runtime reasons; this is not a model-selection judgment. The prior 500-origin GARCH(1,1) metrics are presented separately because their origins differ.
- Artifacts: `data/processed/evaluation_forecasts.csv` and manifest, `data/processed/evaluation_summary.csv`, `data/processed/predictor_set_comparison.csv`, `data/processed/06_final_evaluation.executed.ipynb`, `plots/final_evaluation/`, `README.md`, `REPRODUCE.md`, `presentations/project-progress.qmd`
- Next: Enoch to confirm panel source-date freshness; Samuel to review the loss, DM, and MZ evidence and recommend a next step without assuming a selection criterion.

## 2026-11-11 — Re-ran Stage 1 baselines and generated plots
- Author: Copilot at user's request
- Type: finding
- Context: Re-ran the seven return-only baselines on the 500 most recent origins from the prepared Yahoo `GC=F` snapshot, with the five-session forward realized-variance target.
- Outcome: The shared sample contains 488 forecasts from 2024-11-07 through 2026-11-02. GARCH(1,1) fitting failed at 12 origins; the existing comparison helper omits an origin when any baseline fails. GARCH(1,1) retained the lowest MAE (0.000783) and RMSE (0.001891), with RMSE 6.3% below expanding variance and 1.0% below EWMA. The sample size and point estimates differ slightly from the older 496-origin Stage 1 PDF, so treat this as a rerun finding, not a model decision.
- Artifacts: `plots/model_comparisons/` (seven per-model charts, an overlay, and an RMSE comparison), `data/processed/stage1_baseline_500_origin_forecasts.csv` and `data/processed/stage1_baseline_500_origin_metrics.csv` (local/ignored), `presentations/project-progress.qmd`
- Next: Enoch to review GARCH fitting failures and maintain the distinction between these point estimates and a model-selection conclusion.

## 2026-11-11 — Screened LSEG predictor importance
- Author: Copilot at Enoch's request
- Type: finding
- Context: Prepared the available LSEG Workspace exports as a separate `GCc1` dataset and repeated the Stage 2 training-only, standardized OLS screen for the 11 available predictors: DXY, S&P 500 log return, VIX, 10-year Treasury yield, five gold implied-volatility tenors, 1M risk reversal, and 1M butterfly. The 4,057 complete origins span 2010-01-06 through 2026-11-03; a chronological 80/20 split with four overlapping target rows purged leaves 3,241 training rows through 2023-06-09 and 812 holdout rows from 2023-06-16 onward. Inference uses `log1p` five-session forward realized variance and HAC lag 4.
- Outcome: DXY has a negative conditional coefficient per training SD (-0.000122; HAC p=0.000094; Holm p=0.00103) and VIX a positive coefficient (0.000137; HAC p=0.00124; Holm p=0.0124); these are the only individual predictors significant at 5% after Holm correction across 11 tests. No individual option feature survives correction. A secondary joint HAC test finds the five IV tenors significant as a block (F=17.42; Holm-adjusted p=9.26e-17 across three group tests), while risk reversal plus butterfly are not (adjusted p=0.386). IV tenors are highly collinear (maximum absolute pairwise correlation 0.991), so their individual coefficients do not identify a preferred tenor. This is exploratory conditional association, not evidence of out-of-sample importance or a final variable-selection decision. Availability is proxied as one weekday after observation; option quote units remain unverified. The current export has no gold volume, breakeven inflation, or WTI, and spot/fixing requests failed.
- Artifacts: `notebooks/05_advanced_models.ipynb`, `README.md`, `presentations/project-progress.qmd`, `data/processed/stage2_lseg_multivariate.csv` (local/ignored)
- Next: Enoch to verify LSEG field units and availability assumptions, then compare a reduced predictor set using the purged out-of-sample evaluation before selecting model inputs.

## 2026-11-11 — Started Stage 3 full-sample evaluation
- Author: User-directed execution by Copilot
- Type: finding
- Context: The Stage 3 notebook was launched against the prepared panel using the current working settings (`min_train=750`, expanding window, and five-session origin steps).
- Outcome: Notebook execution is still running; it has not produced a completed forecast table or performance results yet. No Stage 3 model ranking or selection conclusion is available.
- Artifacts: `notebooks/06_final_evaluation.ipynb`; ignored execution output under `data/processed/`
- Next: Enoch to review completed forecasts and convergence failures; Samuel to assess the reported loss and inference evidence before recommending a next step.

## 2026-11-11 — Rechecked Stage 2 predictor significance
- Author: Copilot at Enoch's request
- Type: finding
- Context: Re-screened the current 2,976-row Yahoo/FRED Stage 2 panel through 2026-11-03 using the existing training-only HAC regression and Holm correction across the seven core predictors.
- Outcome: The complete sample has 2,376 training rows (2015-01-05 through 2024-06-14) after purging 4 rows whose five-session targets overlap the holdout; the 596-row holdout starts 2024-06-24. With log1p forward realized variance, predictors standardized on training data, and HAC max lag 4, only VIX is significant at 5% after Holm correction (standardized coefficient 0.000268, HAC p=0.0000435, Holm-adjusted p=0.000305). Breakeven inflation is nominally significant before correction (p=0.0455) but not after (adjusted p=0.273); the other five predictors are not significant. This is an association screen, not evidence that excluded variables lack out-of-sample value or a final model-selection decision. The partial LSEG option panel was not included.
- Artifacts: `data/processed/stage2_multivariate.csv`, `notebooks/05_advanced_models.ipynb`
- Next: Enoch to treat VIX as the leading statistically supported candidate under this screen and assess incremental out-of-sample value before dropping other predictors.

## 2026-11-11 — Implemented Stage 3 forecasting and evaluation workflow
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: Stage 3 requested regression and volatility extensions, a common no-look-ahead evaluation, an auto-ARIMA reproducibility check, and updated notebook/docs.
- Outcome: Added HAR/HARX, penalized, and BIC-selected VAR functions; GJR-GARCH, asymmetric EGARCH, Student-t GARCH, and training-selected EWMA; a purged rolling-origin evaluator and inference summaries; and a full-sample evaluation notebook. GARCH-X is explicitly skipped because its proposed two-step regressors cannot be constructed without historical-origin leakage. The Stage 2 holdout notebook now purges training targets at the first test origin. The fixed local ARIMA slice (2,975 rows; 2015-01-05 through 2026-11-02) selected ARIMA(2,1,1) identically on two runs; this does not establish which earlier order is correct. The full evaluation notebook remains unrun, so no Stage 3 performance conclusion is recorded.
- Artifacts: `src/gold_vol_pipeline/regression_models.py`, `src/gold_vol_pipeline/models.py`, `src/gold_vol_pipeline/evaluation.py`, `notebooks/05_advanced_models.ipynb`, `notebooks/06_final_evaluation.ipynb`, `scripts/check_auto_arima_reproducibility.py`, `docs/arima_discrepancy.md`, `tests/`, `README.md`, `llms.txt`, `REPRODUCE.md`, `presentations/project-progress.qmd`
- Next: Enoch to run the full prepared-data evaluation; Samuel to review the resulting loss, DM, MZ, and convergence evidence before any model-selection recommendation.

## 2026-11-11 — Added Stage 3 regression forecasts
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: Stage 3 Task 1 adds HAR/HARX, penalized regression, and a secondary VAR for the five-session realized-variance target.
- Outcome: Added origin-wise regression forecast functions with target-end metadata and training-row purging at the supplied forecast origin; promoted scikit-learn to a core dependency. Module compilation and a synthetic smoke check of HAR, HARX, lasso, VAR, and target alignment passed. This does not constitute a backtest or model-selection decision.
- Artifacts: `src/gold_vol_pipeline/regression_models.py`, `pyproject.toml`
- Next: Continue with Stage 3 Task 2 after team review/continuation; unified evaluation and dedicated regression tests remain pending.

## 2026-11-11 — Added LSEG import to the Stage 2 pipeline
- Author: Keir's request; implementation by Copilot
- Type: implementation
- Context: Keir requested that the Stage 2 pipeline use the downloaded exports in `data/refinitiv/`.
- Outcome: Added `--lseg-input-dir` to normalize `GCc1`, available DXY/S&P 500/VIX/Treasury data, and available gold option quotes into a separate prepared dataset. Predictor values use a one-weekday availability proxy and retain observation dates; the importer drops missing gold closes and never substitutes spot or other futures ranks. On the local export it produced 4,058 origins through 2026-11-03 from 4,064 `GCc1` closes; `ACVOL_1` was not returned, while breakeven inflation, WTI, and spot/fixing exports were unavailable. The existing Yahoo/FRED panel and seven-feature model specification remain unchanged. Full test suite passed (51 tests).
- Artifacts: `src/gold_vol_pipeline/lseg.py`, `src/gold_vol_pipeline/predictors.py`, `tests/test_predictors.py`, `README.md`, `presentations/project-progress.qmd`, `data/processed/stage2_lseg_multivariate.csv` (local/ignored)
- Next: Enoch to review the partial LSEG input and decide whether the option quotes or alternate source mappings warrant model-development follow-up; Keir to resolve missing RIC permissions/fields and verify quote units before interpretation.

## 2026-11-11 — Documented LSEG export schema
- Author: Keir's request; implementation by Copilot
- Type: implementation
- Context: The LSEG Workspace downloader needed a handoff schema explaining per-RIC files, returned columns, manifest contents, and series meanings.
- Outcome: Documented expected CSV filenames, date/field columns, the combined workbook and manifest, and requested-series meanings in the README. CSV and workbook exports now label the history index `Date`; instrument-specific units and quote conventions remain subject to Workspace verification.
- Artifacts: `README.md`, `scripts/download_lseg_data.py`
- Next: Keir to validate vendor field definitions and candidate RIC mappings against his Workspace account.

## 2026-11-11 — Added optional LSEG Workspace export helper
- Author: Keir's request; implementation by Copilot
- Type: implementation
- Context: Keir requested one Python downloader combining his LSEG gold futures, spot/fixing, and implied-volatility examples with the project's selected Stage 2 daily predictor candidates.
- Outcome: Added a standalone script that saves per-RIC CSVs, a combined workbook, and a manifest. Candidate Stage 2 RIC/field mappings are explicitly provisional and need validation in Workspace; exports preserve series separately and do not claim point-in-time release availability. The existing Yahoo/FRED panel and working futures-series decision are unchanged.
- Artifacts: `scripts/download_lseg_data.py`, `README.md`
- Next: Keir to run on the Workspace computer, verify RICs/fields/units/history and data-use terms, and coordinate any later conversion into the model panel with Enoch.

## 2026-11-11 — Added exploratory multivariate model notebook
- Author: User direction; implementation by Copilot
- Type: finding
- Context: The Stage 2 model request asked to include all candidate predictors and to assess predictor significance before modeling. After clarification, the requested significance target was predictor associations, for which a paired t-test is not appropriate.
- Outcome: On the first 80% of 2,976 complete observations (2015-01-05 through 2024-06-21), OLS of log forward realized variance on seven standardized predictors with HAC lags=4 and Holm correction found VIX significant (adjusted p=0.000308); other predictors were not. On the final 596 origins, HistGradientBoosting RMSE was 0.001883, versus 0.001833 for the expanding-variance benchmark and 0.001855 for the training-mean benchmark. The notebook runs end to end; all project tests pass (50 passed). Results are exploratory and do not imply a final model choice.
- Artifacts: `notebooks/05_advanced_models.ipynb`, `pyproject.toml`, `README.md`, `presentations/project-progress.qmd`
- Next: Samuel to review the predictor screen and comparable out-of-sample evidence; Enoch to review data timing and candidate inputs before any model-selection recommendation.

## 2026-11-11 — Executed Stage 2 predictor notebook without a key
- Author: Copilot
- Type: finding
- Context: The public FRED CSV downloader replaced the API-key-gated requests in the Stage 2 workflow.
- Outcome: Executed `notebooks/04_predictors.ipynb`; it downloaded FRED and Yahoo data and produced 2,976 forecast rows (2015-01-05 through 2026-11-03). All six selected predictor values are non-missing across those rows. The focused predictor tests pass. Execution also exposed and fixed incompatible pandas datetime units at the gold/predictor join. Existing yfinance deprecation warnings remain.
- Artifacts: `notebooks/04_predictors.ipynb`, `data/processed/stage2_multivariate.csv`, `data/processed/stage2_multivariate_manifest.json`, `src/gold_vol_pipeline/predictors.py`, `tests/test_predictors.py`
- Next: Enoch to review source freshness and timing assumptions before model fitting.

## 2026-11-11 — Use public FRED CSV downloads
- Author: User direction; implementation by Copilot
- Type: implementation
- Context: The Stage 2 notebook should not depend on `FRED_API_KEY`; use the downloadable series data from FRED's website instead.
- Outcome: Replaced keyed API observation requests with the public FRED graph CSV endpoint using explicit inclusive date bounds, and removed the API-key gate from the downloader and notebook. The existing one-weekday availability proxy and current-vintage limitation are unchanged.
- Artifacts: `src/gold_vol_pipeline/predictors.py`, `tests/test_predictors.py`, `notebooks/04_predictors.ipynb`, `README.md`
- Next: Run the focused tests and execute the notebook download; inspect data freshness and missingness before modeling.

## 2026-11-11 — Added Stage 2 predictor download notebook
- Author: Copilot
- Type: implementation
- Context: The Stage 2 predictor panel needed a notebook entry point for running the existing acquisition and alignment CLI.
- Outcome: Added `notebooks/04_predictors.ipynb` to invoke `python -m gold_vol_pipeline.predictors`, preview the generated panel and manifest, and stop before network requests when `FRED_API_KEY` is absent. The key is read from the Jupyter kernel environment and is never embedded in the notebook. The notebook and code-cell syntax were validated, but the API download could not be run because this environment has no configured FRED key.
- Artifacts: `notebooks/04_predictors.ipynb`, `README.md`
- Next: Set `FRED_API_KEY` before starting Jupyter, run the notebook, then review source-date freshness and missingness before model fitting.

## 2026-11-11 — Validated and refreshed Stage 2 handoff
- Author: Copilot
- Type: finding
- Context: The Stage 2 predictor panel was implemented while the initial model report still described predictors and pipeline work as pending.
- Outcome: Updated the report and progress tracker to reflect the selected candidate panel, source mapping, and timing limitations. Rendered both presentations successfully and ran the full test suite: 50 passed. Tests emitted existing statistical-library/deprecation warnings; the new FRED date-lag warning was removed.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`, `presentations/project-progress.qmd`, `src/gold_vol_pipeline/`
- Next: Enoch to review predictor transformations, source-date freshness, and missing-data handling before fitting multivariate candidates.

## 2026-11-11 — Added Stage 2 daily predictor pipeline
- Author: Keir's requirements; implementation by Copilot
- Type: implementation
- Context: Enoch identified USD strength, rates, equity, inflation/oil, optional gold volume, and conditional central-bank demand as Stage 2 candidate data. Keir selected futures, DXY, nominal yields, both equity returns and volatility, both inflation expectations and oil, and asked for API support with CSV input where suitable.
- Outcome: Added an API-backed daily panel for Yahoo `DX-Y.NYB`, `^GSPC`, `^VIX`, and optional `GC=F` volume, plus FRED `DGS10`, `T10YIE`, and `DCOILWTICO`. S&P 500 close-to-close log returns are derived. Yahoo values align on same-day availability after market close; FRED values use a one-business-day availability proxy, not historical release timestamps or vintages, with observation dates retained and only past-available values joined to gold origins. Central-bank demand is deferred until a source and release schedule can support point-in-time alignment. FRED requires a local API key. Raw observations remain Git-ignored; no data are committed because source-specific redistribution rights must be checked, regardless of file size. Targeted pytest execution was denied by the current environment, so test status is unverified.
- Artifacts: `src/gold_vol_pipeline/predictors.py`, `src/gold_vol_pipeline/pipeline.py`, `tests/test_predictors.py`, `tests/test_pipeline.py`, `README.md`, `presentations/project-progress.qmd`
- Next: Run the targeted tests when command execution is available; Enoch to review predictor transformations, missingness/freshness, and model inputs; Keir to check volume coverage and central-bank-demand publication timing before extending the panel.

## 2026-11-11 — Framed next stage as multivariate modeling
- Author: User direction, recorded by Copilot
- Type: decision
- Context: The project clarified that the current stage applies traditional time-series models to one variable, with the next stage intended to use multiple variables.
- Outcome: Reframed the report and project tracker as Stage 1 univariate modeling followed by planned multivariate development. No additional predictors or multivariate method were selected; variable rationale, source, timing, and availability at the forecast origin remain prerequisites.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`, `presentations/project-progress.qmd`, `README.md`
- Next: Enoch identifies candidate variables and timing; Keir coordinates pipeline changes after those requirements are defined; Samuel compares Stage 2 evidence with Stage 1 baselines.

## 2026-11-11 — Added current-project map to model report
- Author: Copilot
- Type: implementation
- Context: The model-development report needed a visual map of current project status and handoffs.
- Outcome: Added a Metropolis-style workflow map showing open model/data requirements, the pipeline prototype, current model-development work, and the next analysis handoff, with the team decision feeding the next iteration.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`
- Next: Follow the map's requirements-to-pipeline-to-analysis handoffs; no instrument or model decision is implied by the diagram.

## 2026-11-10 — Reported initial model-development results
- Author: Copilot
- Type: finding
- Context: The first-iteration deck was revised to report quantitative model evidence instead of Git contributions.
- Outcome: On the local prepared Yahoo `GC=F` snapshot (2,975 origins through 2026-11-02), GARCH(1,1) had the lowest point-estimate errors over 496 completed recent forecast origins (MAE 0.000776, RMSE 0.001879); its RMSE was 6.1% below expanding variance and 0.8% below EWMA. EGARCH(2,1) had the lowest in-sample AIC/BIC among 21 return-series candidates, but 4 candidates did not converge; the current EGARCH comparison leaves asymmetry order at zero. EGARCH residual Ljung-Box p-value was 0.020 at lag 20. A current squared-return HK rerun selected ARIMA(3,1,2), conflicting with the earlier logged ARIMA(0,1,4); neither should be relied on until the run/input discrepancy is reconciled. These are exploratory futures-series results, not a model or instrument decision.
- Artifacts: `presentations/iteration-1-summary.qmd`, `presentations/iteration-1-summary.pdf`, `presentations/project-progress.qmd`
- Next: Enoch and Samuel should standardize the data snapshot and rolling-origin design, investigate non-convergence/specification, and include automatic order selection inside training windows before comparing forecasts.

## 2026-11-10 — Prepared first-iteration progress handoff
- Author: Copilot
- Type: handoff
- Context: The project needed a concise presentation to close the initial implementation phase and identify the next stage.
- Outcome: Created a Beamer PDF and reproducible Quarto source covering implemented pipeline/model tools, the exploratory status and limitations of current evidence, open instrument/target/evaluation choices, and owner-specific next actions. Clarified in the progress tracker that the available futures and CSV ingestion paths do not settle the team instrument decision.
- Artifacts: `presentations/iteration-1-summary.pdf`, `presentations/iteration-1-summary.qmd`, `presentations/project-progress.qmd`, `README.md`
- Next: Model development should specify data requirements; the team should confirm the research choices before the next pipeline design and out-of-sample evaluation.

## 2026-11-10 — Fixed notebook figure display
- Author: Copilot
- Type: finding
- Context: Notebook plots were not rendering because importing `gold_vol_pipeline` imported `models.py`, which globally switched Matplotlib to the non-interactive `Agg` backend.
- Outcome: Removed the import-time backend switch and load pyplot only when chart generation is requested, preserving the notebook's selected backend. Added a regression test for backend preservation.
- Artifacts: `src/gold_vol_pipeline/models.py`, `tests/test_models.py`
- Next: Rerun the affected notebook with its intended Jupyter kernel.

## 2026-11-10 — Verified notebook execution
- Author: Copilot
- Type: finding
- Context: The project requested an end-to-end execution check for all notebooks. The data-inspection notebook referenced an unavailable `deeplob` kernel, so execution initially stopped before running any cells.
- Outcome: Updated its kernel metadata to the standard `python3` kernel. All three notebooks then executed without cell errors. The Yahoo download notebook validated 2,981 closes and 2,975 prepared target rows; the baseline and evaluation notebooks ran on the existing prepared dataset. Statistical-library warnings were emitted during model fitting but did not prevent execution.
- Artifacts: `notebooks/01_inspect_gold_data.ipynb`, `notebooks/02_models.ipynb`, `notebooks/03_model_evaluation.ipynb`
- Next: None.

## 2026-11-10 — Added sequential baseline forecast comparison
- Author: Enoch
- Type: implementation
- Context: Baseline variance forecasts needed a time-ordered comparison with realized-variance targets.
- Outcome: `prediction_history` now supports evaluating a recent window of forecast origins while using all prior returns at each origin. The model notebook summarizes MAE/RMSE and plots actual versus forecast values by baseline. This is exploratory evidence, not a final model-selection decision.
- Artifacts: `src/gold_vol_pipeline/models.py`, `notebooks/02_models.ipynb`, `plots/prediction_vs_actual.png`, `tests/test_models.py`
- Next: Confirm the evaluation window, target, and selection criteria with the team before interpreting results.

## 2026-11-10 — Added model-evaluation workbench
- Author: Copilot
- Type: implementation
- Context: The project needed reusable time-series transformations, candidate-model comparison, residual diagnostics, and an interactive review notebook.
- Outcome: Added price/log/difference/return transformations; ADF unit-root tests; ACF/PACF and transformation plots; configurable AR, MA, ARMA, ARCH, GARCH, and EGARCH fits with AIC/BIC; nonseasonal Hyndman-Khandakar stepwise ARIMA selection; residual export/panels; and Ljung-Box tests. Added `notebooks/03_model_evaluation.ipynb`. These tools provide exploratory/in-sample evidence, not a model-selection decision.
- Artifacts: `src/gold_vol_pipeline/model_evaluation.py`, `notebooks/03_model_evaluation.ipynb`, `tests/test_model_evaluation.py`, `README.md`, `presentations/project-progress.qmd`, `pyproject.toml`
- Next: Model analysis should review the diagnostics and information-criterion limitations, then use an agreed chronological out-of-sample evaluation before recommending a model.

## 2026-11-10 — Added Hyndman-Khandakar ARIMA selection
- Author: Copilot
- Type: implementation
- Context: Enoch requested automatic ARIMA parameter selection for the return-only volatility work.
- Outcome: Added a nonseasonal Hyndman-Khandakar selector using KPSS differencing and bounded stepwise AICc search, with a CLI that fits squared returns by default and prints order, coefficients, candidate ranking, and a five-step variance-proxy forecast. On the local prepared Yahoo `GC=F` data (2,975 observations), the exploratory full-sample fit selected ARIMA(0,1,4), AICc -39098.363176; estimated trend/drift (`x1`) 3.122375346e-08, MA coefficients (-0.9178895503, 0.1926706782, -0.2153530483, 0.03742078706), and sigma2 1.183282999e-07. This is a full-sample fit, not a held-out evaluation or model-selection decision.
- Artifacts: `src/gold_vol_pipeline/auto_arima.py`, `tests/test_auto_arima.py`, `README.md`, `presentations/project-progress.qmd`
- Next: Include the selector in a chronological rolling-origin evaluation before comparing it with the existing baselines; confirm the target/instrument choices with the team.

## 2026-11-10 — Added return-data assumption diagnostics
- Author: Copilot
- Type: implementation
- Context: The return-only baseline models needed a repeatable way to inspect whether prepared returns display properties relevant to their assumptions.
- Outcome: Added `gold_vol_pipeline.diagnostics` with checks for mean-zero evidence, return/squared-return stationarity, serial dependence, ARCH effects, and normality. The tests are exploratory diagnostics; they do not establish assumptions or select a model.
- Artifacts: `src/gold_vol_pipeline/diagnostics.py`, `tests/test_diagnostics.py`, `pyproject.toml`, `README.md`
- Next: Run the report on each analysis dataset and interpret results alongside chronological out-of-sample comparisons.

## 2026-11-10 — Retired superseded branch refs
- Author: Copilot
- Type: handoff
- Context: After the data-pipeline and model-development work was pushed to `main`, the remaining remote branch tips were checked against `origin/main`.
- Outcome: Confirmed each non-main branch tip was an ancestor of `origin/main`, deleted the `data/interactive-pipeline-review`, `feat/model-development`, and `docs/project-guides` remote refs, and removed the local data branch. The remote advertises `main` as HEAD. GitHub API branch and PR listing remained unavailable (404), so PR state or review status is not claimed.
- Artifacts: `main`, `GIT_WORKFLOW.md`, `presentations/project-progress.qmd`
- Next: Work directly on `main`; the progress deck's contribution instructions now fetch `origin main`.

## 2026-11-10 — Consolidated project work on main
- Author: Keir with Copilot
- Type: decision
- Context: Keir directed that all project work be consolidated into `main` and that contributors and AI agents use `main` only.
- Outcome: Merged the Yahoo futures data pipeline and return-only model baselines into `main`; the package retains both `yfinance` and `scipy`. Confirmed the remote's advertised HEAD is `main`. Updated contributor and AI instructions to prohibit topic branches and PRs. GitHub API branch and PR listing returned 404, so PR state or review status could not be verified.
- Artifacts: `src/gold_vol_pipeline/`, `tests/`, `README.md`, `GIT_WORKFLOW.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `llms.txt`
- Next: Continue project changes directly on `main`; do not bypass repository protection if a direct push is rejected.

## 2026-11-10 — Verified end-to-end Yahoo futures CSV download
- Author: Copilot
- Type: finding
- Context: Ran the default pipeline using the Yahoo Finance `GC=F` data endpoint.
- Outcome: The API returned 2,981 daily closes from 2015-01-02 through 2026-11-09. The pipeline wrote `data/raw/gold_futures.csv` and 2,975 prepared rows to `data/processed/gold_futures.csv`; the prepared file includes Yahoo/instrument metadata and a forward five-observation realized-variance label. These data files remain local and Git-ignored.
- Artifacts: `src/gold_vol_pipeline/`, `data/raw/gold_futures.csv`, `data/processed/gold_futures.csv`
- Next: Inspect source quality/calendar and confirm the futures series remains the team's intended instrument before formal comparisons.

## 2026-11-10 — Use Yahoo gold futures as the default downloadable series
- Author: Keir
- Type: decision
- Context: The pipeline needed a runnable datasource API path and the team had not finalized spot versus futures.
- Outcome: Use Yahoo Finance `GC=F` (COMEX gold futures continuous series) as the current default because the spot ticker `XAUUSD=X` was unavailable. The CLI and notebook should download from Yahoo by default and save raw and prepared CSVs. This is a working default, not a claim that futures and spot are interchangeable; keep any future instrument change explicit.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`, `presentations/project-progress.qmd`
- Next: Team may revisit instrument choice; if changed, verify source, instrument and pipeline behavior before using or comparing series.

## 2026-11-10 — Added return-only baseline forecast functions
- Author: Enoch with Copilot
- Type: implementation
- Context: Model-development work requested baseline functions that can accept future gold-return inputs; the project checklist calls for return-only ARCH/GARCH models and simple historical-variance benchmarks.
- Outcome: Added expanding and rolling historical variance, EWMA, AR(1)/MA(1) squared-return models, and ARCH(1)/GARCH(1,1) forecasts. Functions forecast summed variance over a configurable future horizon and require returns ending at the forecast origin. No actual gold data or model-selection result is included.
- Artifacts: `src/gold_vol_pipeline/models.py`, `tests/test_models.py`, `README.md`, `pyproject.toml`
- Next: Confirm the instrument, target horizon, and available data with the team; then build chronological out-of-sample comparisons using the selected series.

## 2026-11-10 — Spot versus futures decision deferred to team meeting
- Author: Keir
- Type: open question
- Context: Keir raised that the team may want to discuss whether the project should forecast spot-gold or gold-futures volatility in a meeting.
- Outcome: Treat the instrument choice as pending team confirmation; the earlier spot selection is not final. No Bloomberg data has been requested or downloaded.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`
- Next: After the meeting, confirm spot or futures and the provider. Request a Bloomberg export only if the team selects spot and no suitable accessible source is available, or if Bloomberg is otherwise chosen.

## 2026-11-10 — Working instrument changed to spot gold
- Author: Keir
- Type: decision
- Context: After reviewing the distinction between spot and futures volatility, Keir selected spot-gold volatility as the working project instrument.
- Outcome: Use one consistent daily gold spot series. Yahoo `XAUUSD=X` was previously checked and returned no data; the existing `GC=F` futures default is not an acceptable substitute. Pipeline ingestion is CSV-based until a suitable spot provider is selected.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`, `llms.txt`, `.github/copilot-instructions.md`
- Next: Keir and the team should select/confirm a spot-price source and verify access, licensing, currency, units, calendar, and available history before data collection.

## 2026-11-06 — Default data series set to COMEX gold futures
- Author: Keir and Copilot
- Type: decision
- Context: Yahoo Finance returned no data for `XAUUSD=X`; Yahoo's `GC=F` returned daily prices.
- Outcome: The default series is Yahoo Finance `GC=F`, a COMEX gold futures continuous series. It is not spot gold or a single fixed-expiry contract. Keep its identity explicit and note potential roll effects; never silently combine it with spot observations.
- Artifacts: `src/gold_vol_pipeline/`, `notebooks/01_inspect_gold_data.ipynb`, `README.md`, `llms.txt`, `.github/copilot-instructions.md`
- Next: Revisit the series definition if the research question or data availability changes. Preserve source, instrument, currency, units, and dates for any alternate CSV input.

## 2026-11-06 — Initial preparation choices and pipeline behavior
- Author: Keir and Copilot
- Type: decision
- Context: Enoch's data checklist specifies daily actual/realized volatility, while leaving the horizon and exact realized-volatility measure for the team to choose.
- Outcome: The implemented default horizon is five observed price sessions. The default target is forward realized variance (sum of squared close-to-close log returns); forward realized volatility (its square root) is selectable. The future window is the label for each origin, not an available predictor.
- Artifacts: `src/gold_vol_pipeline/pipeline.py`, `tests/test_pipeline.py`, `notebooks/01_inspect_gold_data.ipynb`
- Next: Revisit the target definition if the model-development requirements or research question call for a different measure or horizon.

## 2026-11-06 — Data source and interactive workflow
- Author: Copilot
- Type: implementation
- Context: The Yahoo spot symbol was unavailable, and switching to a futures symbol changes the instrument.
- Outcome: Implemented Yahoo `GC=F` download, explicitly labeled local CSV ingestion, a command-line pipeline, and a notebook for inspecting prices, target alignment, and saved run metadata. Raw/processed data under `data/` are excluded from Git.
- Artifacts: `pyproject.toml`, `src/gold_vol_pipeline/`, `tests/test_pipeline.py`, `notebooks/01_inspect_gold_data.ipynb`, `.gitignore`
- Next: Review source availability and terms for the team's intended use. The current pipeline has not committed downloaded price data.
