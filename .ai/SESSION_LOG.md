# AI and team handoff log

This is an append-only record of project context that may help future sessions. Confirm current implementation and Git state before acting on entries.

## 2026-11-18 — Reorganized README around original workflow and merged analysis
- Author: User-directed documentation update by Copilot
- Type: implementation
- Context: The user requested README coverage of the original procedures and the work merged from `test`, using the original `main` README format.
- Outcome: Restored the original workflow/team/repository-status organization, documented active LSEG Yang-Zhang analysis procedures and findings, and linked the archived Yahoo/FRED README as historical reference to avoid presenting superseded commands as current.
- Artifacts: `README.md`; original README on `archive/main-before-test-merge`.
- Next: Team to keep procedures and reported results synchronized with future analysis changes.

## 2026-11-18 — Integrated the test branch into main
- Author: User-directed implementation by Copilot
- Type: decision
- Context: The user requested integrating `origin/test` into `main`, preferring the test branch where specifications conflict, and preserving the original main.
- Outcome: Prepared the merge from `origin/main`, backed up its original tip as `archive/main-before-test-merge`, and resolved conflicting report deletions in favor of `origin/test`. Preserved both branches' append-only project-log entries.
- Artifacts: `archive/main-before-test-merge`; merged project tree.
- Next: Validate the merged test suite before updating `origin/main`.

## 2026-11-16 — Added test-branch forecast-error diagnostics
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested a reproducible residual study focused on the `test` branch.
- Outcome: Added descriptive diagnostics for out-of-sample forecast errors on 761 common dates from the six-model joint-predictor holdout. Errors show positive mean bias, heavy tails, and raw-error autocorrelation flags; overlapping target windows make Ljung–Box p-values exploratory. Retrospective volatility-tercile results and extreme misses are reported, with HistGradientBoosting treated as provisional. These are forecast errors, not fitted GARCH innovations, and no paired forecast-loss test or untouched holdout was performed.
- Artifacts: `src/gold_vol_pipeline/forecast_diagnostics.py`, `scripts/analyze_yang_zhang_forecast_errors.py`, `tests/test_forecast_diagnostics.py`, `reports/yang-zhang-residual-diagnostics.qmd`, `reports/yang-zhang-residual-diagnostics.pdf`, `reports/figures/residual_diagnostics/`, `REPRODUCE.md`.
- Next: Validate on a later pre-specified holdout and use overlap-aware paired loss inference before an operational model decision.

## 2026-11-16 — Removed calibration subsection from report
- Author: User-directed update by Copilot
- Type: implementation
- Context: The user requested deleting the calibration discussion from the Yang–Zhang report.
- Outcome: Removed the Past-error calibration diagnostic subsection from the report; the separate calibration code and saved experiment outputs remain unchanged.
- Artifacts: `reports/yang-zhang-volatility.qmd`.
- Next: None.

## 2026-11-16 — Interpreted calibration in report evaluation
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: The user asked for the calibration result to be explained in Part 5 of the Yang–Zhang report.
- Outcome: Added the 509-date prequential calibration comparison, including mean error, MAE, and empirical interval coverage for each of the six models. Clarified the reduction in average bias, the MAE trade-off, below-nominal interval coverage, and holdout limitations.
- Artifacts: `reports/yang-zhang-volatility.qmd`; results sourced from `data/processed/stage2_yz_diagnostics/empirical_error_model/distribution_summary.csv` and `forecast_error_tests.csv`.
- Next: Treat calibration performance as exploratory until evaluated on a later independent period.

## 2026-11-16 — Added volatility-regime results
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: The user requested a report note clarifying that holdout volatility-tercile labels are retrospective diagnostics, not deployable forecast-origin regimes.
- Outcome: Added the six expanding-history predictor models' low-, middle-, and high-tercile MAE to the report; recorded the tercile boundaries and rank-based assignment rule in the Stage 2 manifest.
- Artifacts: `reports/yang-zhang-volatility.qmd`, `src/gold_vol_pipeline/volatility_forecasting.py`, `tests/test_yang_zhang.py`; full variant-level results remain in `data/processed/stage2_yz_diagnostics/stage2_yz_summary.csv`.
- Next: Team to interpret high-tercile errors descriptively, not as evidence from a deployable regime classifier.

## 2026-11-16 — Added paired predictor forecast plots
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: The user requested plotting the no-external-predictor and joint-predictor forecasts together for the same model.
- Outcome: Added test-period paired panels for the six joint models, with matching dates, origins and actuals checked before plotting. Retained existing full-history and baseline figures; comparisons remain descriptive.
- Artifacts: `scripts/plot_yang_zhang_report.py`, `tests/test_plot_yang_zhang_report.py`, `reports/yang-zhang-volatility.qmd`, `reports/figures/stage2_yz_selected/paired_forecasts_*.png`, `REPRODUCE.md`.
- Next: Team to inspect visual differences alongside the existing holdout and predictor-timing caveats.

## 2026-11-16 — Removed report section 5.1
- Author: User-directed update by Copilot
- Type: implementation
- Context: The user requested deleting the report's drift, retransformation, and variance-loss section.
- Outcome: Removed section 5.1 and its forecast-error testing and empirical calibration subsections from the report; the evaluation table, forecast figure, experiment code, results and reproduction instructions remain unchanged.
- Artifacts: `reports/yang-zhang-volatility.qmd`; the separate experiment details remain in `REPRODUCE.md`.
- Next: Team may consult the retained experiment artifacts if needed.

## 2026-11-16 — Included all experiment performance results
- Author: User-directed update by Copilot
- Type: implementation
- Context: The drift, retransformation, and variance-loss section described 32 candidates but displayed only HGB variant metrics.
- Outcome: Expanded its results table to list all 32 candidates, including rolling-window and 252-origin bias-corrected variants across six model families and HGB-only smearing/Gamma experiments. Clarified the experiment scope and retained the descriptive-comparison caveats.
- Artifacts: `reports/yang-zhang-volatility.qmd`; performance values sourced from the local `data/processed/stage2_yz_diagnostics/stage2_yz_summary.csv`.
- Next: Team to review trade-offs and limitations before any model-selection decision.

## 2026-11-16 — Expanded forecast performance comparison
- Author: User-directed update by Copilot
- Type: implementation
- Context: The Stage 2 report's evaluation table showed only the six model families rerun with DXY, gold IV 1M and gold IV 1Y, omitting the broader no-external-predictor comparison.
- Outcome: Expanded the table to all 20 no-external-predictor models and retained joint-predictor metrics for the six matched model families; joint cells are blank for models not evaluated with external predictors.
- Artifacts: `reports/yang-zhang-volatility.qmd`; values sourced from `data/processed/stage2_yz_summary.csv` and `data/processed/stage2_yz_selected/stage2_yz_summary.csv`.
- Next: Team to interpret the descriptive rankings with the report's holdout and model-comparison caveats.

## 2026-11-16 — Evaluated empirical forecast-error calibration
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user chose all six baseline models and empirical past standardized errors to address mean bias and changing uncertainty. Implemented separate 252-origin relative-error bias and scale calibration, with empirical 90% intervals; prior target availability is checked and the baseline forecasts remain unchanged.
- Outcome: On 509 common post-warm-up dates, all six calibrated raw means moved closer to zero and do not reject zero with Holm-adjusted HAC(10), but all six raw squared-error slopes still reject. No standardized squared-error slope versus the baseline forecast rejects after separate Holm adjustment; range GARCH-X standardized mean does reject (p=0.031), and HGB standardized errors remain right-skewed/heavy-tailed. Interval coverage ranges from 87.6% to 89.4%; MAE worsens for five models. These do not prove zero future bias, constant variance, or guaranteed interval coverage, and the holdout is not untouched.
- Artifacts: `src/gold_vol_pipeline/empirical_error_model.py`, `scripts/evaluate_empirical_error_model.py`, `tests/test_empirical_error_model.py`, `reports/yang-zhang-volatility.qmd`, `REPRODUCE.md`; ignored local output `data/processed/stage2_yz_diagnostics/empirical_error_model/`.
- Next: Samuel to review raw versus standardized errors, interval coverage and accuracy trade-offs; test on an independent later period before any model-selection decision.

## 2026-11-16 — Tested forecast-error mean and variance scaling
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested testing mean and constant-variance assumptions on the saved Stage 2 experiment errors. Tested all 32 candidates on 761 common holdout dates with HAC(10) inference and separate Holm correction across models.
- Outcome: All six baseline models reject zero mean actual-minus-forecast error at 5%; HGB smearing, Gamma and 252-origin bias correction do not, which does not establish zero bias. A targeted regression of globally centered squared errors on forecast levels has positive slopes for all 32 and 21 adjusted rejections, including four baselines. HGB and HARX baseline slopes do not reject after correction, but non-rejection is not proof of constant variance; changing conditional means can also affect the slopes. HAC lag 4 and 20 sensitivity yields 20/19 mean and 20/28 slope rejections. These are exploratory tests on a previously examined holdout with overlapping targets and extreme errors, not comparisons of model bias or a model decision.
- Artifacts: `src/gold_vol_pipeline/forecast_error_diagnostics.py`, `scripts/diagnose_yang_zhang_forecast_errors.py`, `tests/test_forecast_error_diagnostics.py`, `reports/yang-zhang-volatility.qmd`, `REPRODUCE.md`; local ignored outputs in `data/processed/stage2_yz_diagnostics/error_tests*/`.
- Next: Samuel to review lag sensitivity, residual-dependence limitations, and the forecast-loss trade-offs before a model recommendation.

## 2026-11-16 — Ran drift and loss-function diagnostics
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested rolling-history, leakage-safe bias-calibration, HGB log-smearing/Gamma-loss comparisons, and QLIKE diagnostics while retaining the existing Stage 2 models.
- Outcome: Added optional variants and evaluated them on the same 761 sequential holdout origins; all 32 candidates completed with zero fit failures. HGB 252-origin additive calibration reduced forecast-minus-actual bias magnitude from 0.821e-3 to 0.184e-3 but increased MAE from 0.002423 to 0.002489. Smearing and Gamma loss reduced RMSE and QLIKE but slightly increased MAE; no rolling window beat expanding HGB on overall MAE or RMSE. The high/low volatility-tercile results are retrospective diagnostics, and all comparisons reuse an already examined holdout, so they are descriptive and not model-selection evidence.
- Artifacts: `src/gold_vol_pipeline/volatility_forecasting.py`, `scripts/evaluate_yang_zhang_volatility.py`, `tests/test_yang_zhang.py`, `reports/yang-zhang-volatility.qmd`, `REPRODUCE.md`; local outputs under ignored `data/processed/stage2_yz_diagnostics/`.
- Next: Samuel to review the trade-offs and overlapping-target dependence before recommending any model change; preserve the expanding-history baseline.

## 2026-11-16 — Evaluated the three selected predictors jointly
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested replacing VIX in the six external-input forecast models with DXY, gold IV 1M and gold IV 1Y together, comparing against their no-external-predictor baselines.
- Outcome: All six models completed 761 matched test dates with zero recorded failures. Both MAE and RMSE improved for Ridge, Lasso and VAR, worsened for HARX, and changed in opposite directions for gradient boosting and GARCH-X. This comparison does not isolate individual effects or test forecast-loss significance. Optimizer warnings occurred, quote timing remains a weekday proxy, and the selection was made after earlier holdout results were observed. GARCH-X uses nonnegative predictor coefficients in the variance recursion and reports the fitted conditional mean.
- Artifacts: `src/gold_vol_pipeline/lseg.py`, `src/gold_vol_pipeline/volatility_forecasting.py`, `scripts/evaluate_yang_zhang_volatility.py`, `scripts/compare_selected_predictors.py`, `data/processed/stage2_yz_selected/`, `reports/yang-zhang-volatility.qmd`.
- Next: Team to review joint forecast performance, source quote units, timing and selection caveats before a model decision.

## 2026-11-16 — Tightened the VIF cutoff
- Author: User-directed implementation by Copilot
- Type: decision
- Context: The user changed the VIF cutoff from 10 to 5.
- Outcome: The same three IV tenors are removed and the eight remaining predictors and significance results are unchanged; the maximum final VIF including the control is 4.31.
- Artifacts: `src/gold_vol_pipeline/yang_zhang_significance.py`, `scripts/screen_yang_zhang_predictors.py`, `data/processed/stage1_predictor_significance/`, `reports/yang-zhang-volatility.qmd`.
- Next: Team to review the resulting exploratory screen and its limitations.

## 2026-11-16 — Applied VIF screening before predictor inference
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user selected a VIF threshold of 10 and requested the current-volatility control remain included in every VIF calculation and regression.
- Outcome: On the same 3,039 training pairs, iterative highest-VIF elimination removed gold IV 6M, 3M, then 2M. Eight predictors remain; maximum final VIF including the control is 4.31. HAC(4)/Holm inference on those eight yields adjusted significance for gold IV 1M, gold IV 1Y and DXY, but not VIX. Results are exploratory conditional associations after same-sample selection, not a causal or out-of-sample model decision. The Stage 2 VIX forecasts were not rerun and their earlier-selection holdout caveat remains.
- Artifacts: `src/gold_vol_pipeline/yang_zhang_significance.py`, `scripts/screen_yang_zhang_predictors.py`, `data/processed/stage1_predictor_significance/`, `reports/yang-zhang-volatility.qmd`, `README.md`, `REPRODUCE.md`.
- Next: Team to review collinearity, predictor availability and selection before any forecast-model changes.

## 2026-11-16 — Reorganized the active analysis into two stages
- Author: User-directed implementation by Copilot
- Type: decision
- Context: The user requested removal of the earlier Stage 1–3 analyses and confirmed retaining shared code and inputs, with predictor significance as Stage 1 and Yang-Zhang volatility forecasting as Stage 2.
- Outcome: Removed superseded analysis notebooks, code, tests, and reproduction instructions while preserving shared LSEG and ARIMA components, raw input exports, and both saved Yang-Zhang forecast runs. Renamed active outputs and figure paths from `stage4_yz` to `stage2_yz`; renamed screen outputs to `stage1_predictor_significance`. No models were rerun or selection decisions made. The earlier VIX-selection/holdout-overlap caveat still applies.
- Artifacts: `README.md`, `REPRODUCE.md`, `src/gold_vol_pipeline/`, `scripts/`, `data/processed/`, `reports/yang-zhang-volatility.qmd`.
- Next: Team to review the two-stage workflow and any downstream references to the retired paths.

## 2026-11-16 — Screened predictors for Yang-Zhang volatility
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested replacing both earlier predictor-significance screens with an 11-predictor LSEG screen of next-valid Yang-Zhang volatility, controlling for current volatility, using HAC(4) and Holm correction.
- Outcome: On 3,039 complete training pairs, with responses ending 2023-06-01 before the 2023-06-02 holdout, none of 11 predictors is individually significant at the 5% Holm-adjusted level. DXY adjusted p=0.0595; VIX raw p=0.2936 and adjusted p=1.000. Option IV tenors are highly collinear (max absolute correlation 0.9904); the weekday availability proxy and finite HAC lag limit interpretation. The earlier notebook screens were removed; forecast models and existing results were not reselected or rerun.
- Artifacts: `src/gold_vol_pipeline/yang_zhang_significance.py`, `scripts/screen_yang_zhang_predictors.py`, `data/processed/yang_zhang_predictor_significance/`, `notebooks/05_advanced_models.ipynb`, `README.md`, `REPRODUCE.md`, `reports/yang-zhang-volatility.qmd`.
- Next: Samuel and Enoch to review predictor selection and holdout reuse before changing forecasting models; Keir to review LSEG availability and quote conventions.

## 2026-11-15 — Completed VIX-augmented Yang-Zhang holdout
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The user requested the next-weekday as-of LSEG VIX feature for five external-input models and as a third VAR series, while keeping the Yang-Zhang volatility target and existing split.
- Outcome: All 20 models produced forecasts on all 761 common test dates with zero recorded failures. HistGradientBoosting with VIX had the lowest MAE (0.00242640) and RMSE (0.00478093). Against the no-VIX run on the same dates, HARX and GARCH-X improved on both metrics while Ridge, Lasso, gradient boosting, and VAR worsened on both; no significance tests were run. Optimizer/convergence warnings occurred. The earlier VIX significance screen used a different target and its training interval overlaps the start of this holdout.
- Artifacts: `data/processed/stage4_yz_vix/`, `reports/yang-zhang-volatility.qmd`, `reports/yang-zhang-volatility.pdf`, `reports/figures/stage4_yz_vix/`, `presentations/project-progress.qmd`.
- Next: Samuel to review the comparative evidence, VIX availability proxy, warning caveats, and earlier-screen overlap before any recommendation.

## 2026-11-15 — Added VIX to the Yang-Zhang comparison
- Author: User-directed implementation by Copilot
- Type: decision
- Context: The user requested model-specific report subsections, an explanation of the Yang-Zhang variance and HAR horizons, and VIX in models with external predictors. The user confirmed retaining volatility (the square root of Yang-Zhang variance) as the dependent variable, also adding VIX as a third VAR series, and using the latest VIX available under the next-weekday proxy.
- Outcome: The VIX-augmented experiment adds the LSEG VIX close to HARX, Ridge, Lasso, gradient boosting, range GARCH-X, and VAR. The original no-VIX artifacts remain separate. The 20-model sequential holdout rerun is in progress. VIX was previously significant for a different five-session realized-variance target, not shown to be significant for this target.
- Artifacts: `src/gold_vol_pipeline/lseg.py`, `src/gold_vol_pipeline/volatility_forecasting.py`, `scripts/evaluate_yang_zhang_volatility.py`, `data/processed/stage4_yz_vix/`, `reports/yang-zhang-volatility.qmd`.
- Next: Complete the holdout, inspect common-date metrics and fit warnings, and regenerate the report and figures.

## 2026-11-15 — Published Yang-Zhang formulas and forecast graphs
- Author: User-directed implementation by Copilot
- Type: implementation
- Context: The user requested a report containing all target/model formulas and forecast graphs with the test period shaded.
- Outcome: Added an editable Quarto report covering the two-observation Yang-Zhang estimator, all 20 model specifications, metrics, and limitations; generated comparison plots with the chronological test interval shaded and rendered the PDF. All charts use the saved prepared target and forecast outputs.
- Artifacts: `reports/yang-zhang-volatility.qmd`, `reports/yang-zhang-volatility.pdf`, `reports/figures/stage4_yz/`, `scripts/plot_yang_zhang_report.py`, `README.md`, `REPRODUCE.md`, `presentations/project-progress.qmd`.
- Next: Samuel to review the formulas and descriptive model comparison alongside the fit-warning limitations.

## 2026-11-15 — Set Yang-Zhang window to two observations
- Author: User-directed implementation by Copilot
- Type: decision
- Context: The user first requested n=1, then selected the minimum valid standard Yang-Zhang window after n=1 was explained to be undefined for the canonical estimator.
- Outcome: The default target is n=2 with a chronological 80/20 split and sequential one-step forecasts. The input yields 3,801 targets (3,040 train / 761 test). All 20 candidates completed on the same 761 test dates with zero recorded failures; HistGradientBoosting had the lowest MAE (0.00241765) and RMSE (0.00476724). These are descriptive rankings only because target windows overlap; no significance tests were run. Optimizer and convergence warnings were emitted. The earlier n=20 evaluation stopped at 325/471 dates and produced no final metrics.
- Artifacts: `data/processed/GCc1_yang_zhang_volatility.csv`, `data/processed/stage4_yz_forecasts.csv`, `data/processed/stage4_yz_summary.csv`, `data/processed/stage4_yz_manifest.json`, `src/gold_vol_pipeline/yang_zhang.py`, `scripts/evaluate_yang_zhang_volatility.py`, `README.md`, `REPRODUCE.md`, `presentations/project-progress.qmd`.
- Next: Samuel to review the comparative evidence and optimizer warnings before any recommendation.

## 2026-11-15 — Prepared Yang-Zhang target and started holdout
- Author: User-directed implementation by Copilot
- Type: finding
- Context: The approved 20-observation Yang-Zhang workflow was run on the supplied LSEG `GCc1.csv` OHLC data without filling incomplete quotes.
- Outcome: The 4,241 source rows contain 4,064 complete OHLC bars; strict rolling-window validity yields 2,354 daily-volatility labels, split into 1,883 training and 471 test observations. The full 20-model sequential one-step holdout evaluation is running. Missing-bar windows are excluded; test steps advance between valid target labels and may skip market sessions.
- Artifacts: `data/processed/GCc1_yang_zhang_volatility.csv`, `src/gold_vol_pipeline/yang_zhang.py`, `presentations/project-progress.qmd`.
- Next: Record the common-date MAE/RMSE ranking, failures, and target-window caveat after the evaluation finishes.

## 2026-11-15 — Switched current comparison to Yang-Zhang daily volatility
- Author: User-directed implementation by Copilot
- Type: decision
- Context: The user selected LSEG `GCc1.csv` OHLC, unannualized 20-observation Yang-Zhang volatility as the dependent variable, a chronological 80/20 split, sequential one-step test forecasts, and a Yang-Zhang persistence benchmark. This is a user-selected working specification, not yet a team-wide decision.
- Outcome: Added a shared target-preparation and model-comparison workflow covering the project’s univariate, ARMA/HK, ARCH/GARCH, regression, tree, and VAR families. The dedicated rolling-variance baseline is omitted. Incomplete OHLC is not imputed; windows containing incomplete rows have no target. The pre-existing deletion of `src/gold_vol_pipeline/pipeline.py` was restored with user approval; all other existing deletions remain untouched.
- Artifacts: `src/gold_vol_pipeline/yang_zhang.py`, `src/gold_vol_pipeline/volatility_forecasting.py`, `scripts/evaluate_yang_zhang_volatility.py`, `tests/test_yang_zhang.py`, `README.md`, `presentations/project-progress.qmd`.
- Next: Run the complete holdout evaluation, report common-date MAE/RMSE and fit failures, and ask Samuel/team to review the target and comparison before treating a ranking as a project decision.

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
