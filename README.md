# Replication package 1.3.0

Current working paper: **Latent Dimension Selection in Neural Asset Pricing: Sensitivity and Pricing-Loss Costs**, Jinpeng Wang and Yan Jiang. QF preparation does not mean acceptance or submission.

## Start here

1. Read `01_overview/manuscript/main.pdf` (current article) and `audit_supplement.pdf` (new four-page numeric audit).
2. Follow `01_overview` through `07_discussion_and_audit`, in article order. Each index points to current sources and the exact preserved earlier evidence.
3. Run `python3 reproduce_qf.py --no-pdf` for checksum validation, re-aggregation of the frozen 900 rows, all 180 means/30 tasks/12 endpoint transfers, and the new 15-row table. Standard library suffices. Without `--no-pdf`, TeX Live/latexmk compiles the two QF documents.
4. The old complete 1.2.1 directory is preserved, byte for byte, under `historical_v1.2.1/`. For the old IREF checks, change into that directory and run `python3 reproduce_iref.py --no-pdf`. Its scientific evidence and metadata are historical, not the current working-paper citation.

## What changed

The new audit recomputed 1,800 saved pricing distances from licensed prepared arrays on the original local project, with maximum absolute error 1.255e-12. Public replay uses only author-generated aggregate distances; it verifies arithmetic, not the restricted array-to-distance step. The original audit script and hash receipt document that distinction. No licensed security-level observations, stock predictions, checkpoints or vendor panels are redistributed. No new model fitting, resampling or significance claim.

All 15 CAE task minima are at the largest candidate K=8; tasks share fitted models/seeds/returns and are not independent repeats. Linear winners vary. Relative norm costs are not investor returns or welfare losses. Failed inference calibration remains fully disclosed. Original eight figure assets are unchanged. Bibliographic metadata and contextual citation use were audited; no full-text systematic review is claimed.

## Version and citation

Version-specific archive: https://doi.org/10.5281/zenodo.24398249 (1.3.0, 2026-11-20). See CITATION.cff for authors and title. This archive supplies the exact current manuscript, code and aggregate evidence. Earlier published release: https://doi.org/10.5281/zenodo.23177771 (1.2.1), also preserved unchanged under historical_v1.2.1.

GitHub project: https://github.com/extradimen/asset-pricing-dimension-replication. The GitHub branch may lag this archive; use this version-specific DOI and SHA256SUMS for the exact files supporting the QF working revision. This package does not imply journal acceptance or submission.

## Rights

Code MIT; author-generated original aggregate/synthetic outputs and documentation CC BY 4.0. Third-party data keep their terms. Existing public source links are not relicensed. The inherited historical receipt is historical evidence; full retraining needs licensed inputs and applicable resource-control approvals. The public replay performs only integrity and exhibit verification.
