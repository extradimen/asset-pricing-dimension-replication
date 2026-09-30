# Reconstructing the underlying experiments

This is a documented original-code path, not a claim that an independent licensed-data clean-room rebuild was executed during publication. Work in `reproduction/`; all original relative paths are preserved. Read the main methods and supplement before interpreting any historical `sealed` or `preregistered` label.

## 1. Acquire inputs under your own permissions

Obtain CRSP CIZ Daily Stock File, Compustat North America annual and quarterly fundamentals, and CCM link history. The exact downloaded filenames, coverage, sizes and SHA-256 values are in `archive/data_snapshots/wrds_us_equity_2025-12_v1.json`. Place these archives under `data/raw/licensed/` or explicitly pass their locations to the builders. Never commit these inputs. The original full CRSP archive is approximately 6.8 GB compressed; acquisition and processing require substantial disk and memory.

Public benchmark characteristics and test assets are listed with source URLs and hashes in the `gkx_*` and `ken_french_*` manifests. Acquire them from the named providers under their terms. Later revisions need not reproduce the frozen vintage byte-for-byte. No WRDS authentication details are required by, or included in, this package. This study used WRDS exports, not an archived SQL query; the variable names and filters used by the builders define the relevant input schema.

## 2. Construct monthly and characteristic panels

The ordered entry points are:

1. `scripts/build_crsp_monthly.py`, `build_compustat_ccm.py`, `build_research_master.py`.
2. `build_gkx_feature_benchmark.py`, `build_daily_rolling_characteristics.py`, `build_momentum_characteristics.py`, `build_security_weekly_returns.py`, `build_weekly_market_characteristics.py`, `build_industry_momentum.py`, `build_price_delay.py`.
3. `build_core20_monthly_panel.py`, `build_compustat_annual_extended.py`, `build_quarterly_characteristics.py`, `build_quarterly_core10_bridge.py`, `build_annual_core56_bridge.py`, `build_structured_core86.py`.
4. `build_presealed_core86_training_panel.py`, `build_core86_gpu_development_input.py`, `build_core86_evaluation_panel.py` and their validation scripts.
5. `build_public_pricing_targets.py` for the pricing test assets.

Each original builder declares required arguments through `--help`. `ENTRYPOINT_ARGUMENTS.json` lists those declarations without executing a model. The numbered P1-G0 configurations record each data stage, expected experiment IDs and construction decisions. `table_12_core86_feature_dictionary.csv` lists membership; formulas, dates and missing treatment are in the source and supplement. Some stages require predecessor audit artifacts: retain their original experiment/output IDs when reconstructing. Do not substitute a similarly named vendor field or silently change a lag.

## 3. Pricing networks and dimension diagnostics

`run_multi_factor_sdf_teacher.py` trains each of six dimensions and five seeds. Use the hyperparameters in `configs/paper1/P1-G1-V022.json`. The Core-86 run requires `--feature-set core86 --split-clock target_month`; legacy Core-92 uses the historical feature-month convention. For every run provide `--input`, `--factors`, `--pricing-targets`, `--output-dir`, `--source-git-revision`, `--experiment-id`, `--factor-count`, `--seed` and `--device`. Retain the 128,64,32 architecture, 80-epoch cap, patience 12, batch 120 and fixed objective coefficients. Do not retune to match the article.

The `audit_*geometry.py`, `audit_market_factor_completion.py` and `audit_sealed_core86_confirmation.py` entry points generate the diagnostic outputs. The configurations P1-G1-V015 through V022 and P1-G2-V001 determine test assets, weighting paths, timing, bootstrap and locked comparison rules. Existing aggregate outputs can be audited without regenerating checkpoints.

## 4. Known-truth simulation

`scripts/run_pricing_dimension_identification_simulation.py` with `configs/paper1/P1-G3-V001.json` defines the complete fixed simulation: six candidate dimensions, 144 base cells, four evaluation/geometry combinations, six loading families and 500 replications. Original output summaries and frequencies are included. Rerunning this is an evidence-producing simulation; on the authors' Mac/campus infrastructure it must be launched through the global experiment controller after preflight. `reproduce_public.py` deliberately does not start it.

## 5. Temporal return predictors

The temporal handoff retains a separate scientific source identity. `build_paper2_stock_day_panel.py`, `build_paper2_monthly_moments.py`, `build_paper2_exposure_input.py`, `run_paper2_exposure_baselines.py` and `build_paper2_joint_stock_outcomes.py` are included only for the shared panel/outcome/preprocessing dependencies named by P3-STOCK-V001, not to publish the second paper's results. Historical preprocessing must be reconstructed with its original selection rule; replacing it with new prospective preprocessing is a different experiment.

Then run the P3 environment-label, stock-anchor, frozen-neural, neural-update and tree-update entry points with their corresponding configurations. Follow with update decomposition, common-break, sequential-gate and portfolio analyses. Configurations record input hashes, dimensions, learning rules and output locations. Preserve the selected-sample caveat and the original monthly-versus-annual refit difference. Aggregate reports and monthly loss/portfolio outputs are supplied; security-level forecasts and model weights are not.

## 6. Geometry

Use `prepare_paper7_representation_sample.py` and `run_paper7_representation_geometry.py` with `configs/paper7/P7-G1-V001.json`, then the audit/summarize entry points. This needs the 30 Core-86 checkpoints and licensed sampled stocks. The complete aggregate geometry, CKA, drift and endpoint outputs are included, so figure replay needs neither input.

## Runtime and validation boundaries

`requirements-public.txt` pins the tested exhibit runtime. Historical CUDA environment information and hash-locked requirements accompany the geometry and temporal runs. Additional data-build dependencies include Polars/PyArrow and scikit-learn; use the recorded source environments rather than assuming the plotting environment is suitable for retraining. `SOURCE_MANIFEST.json` identifies original and released source hashes. Deployment-prefix sanitization is recorded separately. The package has no hidden credential, download-on-import or remote-execution step.

Use numerical/algorithmic tolerances for cross-platform retraining; use hashes for retained source files. A different vendor vintage, changed eligible population or changed random-number/library implementation is not an exact replay. Preserve differences rather than altering specifications until the headline numbers match.
