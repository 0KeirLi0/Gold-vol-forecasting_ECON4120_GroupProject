# Archived broad model exploration

This folder preserves the superseded, broad model-comparison report and its
rendered output. The active Group 2 deliverable is
[`report_group2/Group2_Report.qmd`](../../report_group2/Group2_Report.qmd),
which focuses on GARCH-X with lagged VIX and seven basic benchmarks and
generates both the A4 report and Beamer presentation.

## Contents

- `model-performance-stage2-stage3.qmd` and `.pdf`: full report covering
  baseline, predictor-screen, LSEG, and advanced-model explorations.
- `previous_group2/`: snapshot of the earlier, broader Group 2 report and
  slides, retained as historical context.

The forecast code, source notebooks, and locally stored evaluation inputs
remain in their existing project locations to preserve reproducibility. The
Stage 3 evaluation still uses the Yahoo/FRED panel; the separate LSEG screen
is not the source of the integrated model ranking. Rerunning archived reports
requires the original local inputs under the Git-ignored `data/` directory.
