# Replication package version 1.2.0

Jinpeng Wang and Yan Jiang. Low-Dimensional Asset-Pricing Networks: Identification, Stability, and Economic Use.

This update accompanies the IREF preparation manuscript, not a journal acceptance. The numbered directories follow Sections 1–7. The original 1.1.0 evidence and replay remain intact. New CAE evidence is indexed from `04_dimension_identification/CAE_CONTROL.txt`; calibration and qualifications are indexed from `07_discussion_and_audit/INFERENCE_CALIBRATION.txt`. Appendix A belongs to the new main article; Supplements A–F describe the earlier evidence.

## Reproduce the current article

Run `python3 reproduce_iref.py` from this directory. It checks released checksums, verifies Table 4 against its aggregate source, re-aggregates all 12,000 released synthetic outcomes, checks Table A.1, and compiles the current authored article using latexmk and elsarticle. It neither trains nor runs new simulations. Python standard library and TeX Live suffice for this current-update verification. The earlier `reproduce_public.py` retains the version 1.1.0 exhibit workflow and its pinned requirements.

## Scientific changes and limits

V003 adds 120 candidate fits, 60 selected models, matched linear versus nonlinear loadings, six factor counts and five seeds. K=8 wins 87.2% of conditional development resamples; it is the grid boundary, not an identified structural count. The new simultaneous intervals fail their V004 calibration: 83.5% and 87.0% coverage in the two primary 120-month equal-distance scenarios. No equivalence or structural nonidentification is inferred from non-rejection. The calibration only targets the 15 simultaneous within-CAE intervals. Full training, validation-coefficient and weight-estimation uncertainty are not covered. All later controls use familiar historical data.

## Data access and execution

Only author-generated aggregates and synthetic outputs are newly released. No licensed stock observations, stock-level arrays, predictions or model weights are included. `LICENSED_RECONSTRUCTION.md` describes the old pipeline; `04_dimension_identification/CAE_CONTROL.txt` describes the extension. Exact historical prepared arrays are not public, and newly downloaded input vintages need not have their frozen checksums. No clean-room licensed pipeline or bitwise cross-platform retraining claim is made.

Scientific runners must be launched through the applicable institutional resource controller where required; documentation commands are child commands, not authorization to bypass that controller. The package contains no server access credentials. Code is MIT; original aggregate/synthetic outputs and documentation are CC BY 4.0. Third-party sources retain their own terms.

Zenodo version 1.2.0 publication is pending; the old DOI covers version 1.1.0 only.
Earlier release: https://doi.org/10.5281/zenodo.23066253
GitHub: https://github.com/extradimen/asset-pricing-dimension-replication
