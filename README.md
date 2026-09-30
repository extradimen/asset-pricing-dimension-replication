# Low-Dimensional Asset-Pricing Networks: replication package

Jinpeng Wang and Yan Jiang (corresponding author), International Business School, Guangzhou City University of Technology.

Version 1.0.1 accompanies *Low-Dimensional Asset-Pricing Networks: Identification, Stability, and Economic Use*. DOI: https://doi.org/10.5281/zenodo.23064062. Code: https://github.com/extradimen/asset-pricing-dimension-replication.

**Download:** the [GitHub v1.0.1 release](https://github.com/extradimen/asset-pricing-dimension-replication/releases/tag/v1.0.1) provides the complete main ZIP and the separate calibration supplement. The [published Zenodo record](https://zenodo.org/records/23064062) preserves the same main ZIP as 13 checksum-verified parts after a gateway timeout during single-file upload. Follow `README_ZENODO.txt` and run `python3 join_archive.py` to reconstruct the byte-identical archive. No scientific contents differ between these distribution formats.

Read the numbered 01–07 directories in the order of the paper. The original relative runtime paths are preserved under `reproduction/` so existing scripts and frozen manifests remain traceable. These are two views of one package, not different analyses.

## Reproduce the public exhibits

Use Python 3.12 or newer with the pinned packages in `requirements-public.txt`, and TeX Live with latexmk and elsarticle. From this repository:

```sh
python -m pip install -r requirements-public.txt
python reproduce_public.py
```

This verifies frozen numerical sources, extracts temporal tables from JSON, regenerates six main figures plus the optional graphical abstract, five tables and supplement long tables, and compiles the authored paper, anonymous paper, title page and supplement. It does not train models, reacquire data or rerun Monte Carlo estimation. The public check needs no WRDS account. Main PDF: `reproduction/main_paper_a_irfa/manuscript/main_full.pdf`.

## Reconstruct underlying experiments

Read `LICENSED_RECONSTRUCTION.md` before running original entry points. Source code, scientific configurations, raw-input hashes and aggregate results are included. CRSP, Compustat, CCM and third-party security-level inputs must be obtained by the replicator under their own access terms. This release does not claim a clean-room licensed-data rebuild, prospective evaluation, externally registered protocol, or bitwise-identical GPU retraining. The original project used a separate resource controller; users on that Mac or its campus nodes must retain its launch requirements.

## Interpretation and provenance

Read `AUDIT_QUALIFICATIONS.md`. In particular, legacy Core-92 development includes January 2020 returns. The 2020–2025 evaluation is a later locked-model test, not a completely untouched research-wide holdout. Core-86 is a public-benchmark/self-built bridge. Temporal predictors are separate selected-sample models. The theory concerns known linear spans. Archived files retain historical labels and outcomes, including failed criteria.

`SOURCE_MANIFEST.json` records original and released SHA-256 values. `SHA256SUMS` covers release files. Path sanitization affects deployment prefixes, not numerical results. Source code is MIT; original aggregate results and documentation are CC BY 4.0. Third-party data and TeX packages retain their original rights and are not relicensed. No raw licensed records, stock-level predictions, model checkpoints, credentials or institutional connection details are included.

The release includes a separate small geometry-calibration supplement, preserving both initial failed and corrected synthetic diagnostic runs. Browse `06_representation_geometry/calibration/` or download `geometry-calibration-supplement-1.0.1.zip` from the v1.0.1 release/Zenodo record. Its own manifest and checksums are supplied.
