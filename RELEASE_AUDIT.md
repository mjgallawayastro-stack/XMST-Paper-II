# v1.0.0 release audit

Release audit completed 2026-09-18.

- Manuscript `paper/Paper_II.tex` is byte-identical to the final proofed source used for the submitted PDF.
- `paper/RASTI_Paper_II.pdf` is byte-identical to the final `RASTI_Paper_II (22).pdf`.
- Refinement-order outputs reproduce the manuscript values, including J=0.8079 vs 0.8354 and 2.5% vs 98.6% formal T01-T02 resolution.
- S2 uncertainty outputs reproduce the manuscript values to the quoted precision.
- S3 gate-sensitivity outputs reproduce the manuscript values to the quoted precision and use the same DeltaV=3 realisations as Table 3.
- S3 observational-uncertainty outputs are the final 1000-run exact-provenance experiment, not the superseded targeted run.
- The blind Q26 checkpoint reproduces median best-match Jaccard 0.6535088 -> 0.7973301 -> 0.8861284 for S1 -> S3 -> S2.
- Q25/Q26 source catalogues are not redistributed.
- Zenodo DOI is pending.

- All bundled Python sources compile successfully and their command-line parsers load in the audit environment.
- A deterministic 100-star synthetic smoke test completed successfully: Stage 1 recovered two planted spatial groups and S2 completed without error.
- `run_xmst_s3.py` is retained byte-for-byte as a frozen provenance dependency; its original CLI order is S1 -> S2 -> S3. The preferred S1 -> S3 -> S2 ordering is explicitly exercised by `run_paired_order_mc.py` and is the ordering represented by the blind-Q26 checkpoint.
- A fresh online `pip install -r requirements.txt` could not be tested in the network-isolated audit environment; the available runtime had NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0, under which the smoke checks passed.
