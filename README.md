# XMST Paper II

Reproducibility materials for:

**XMST II: The spatial and kinematic structure of Galactic OB associations in Gaia DR3**  
Mark J. Gallaway (2026)

This repository contains the Paper II staged-XMST code and compact validation
products available in the research archive at submission.

## Method

The hierarchy used in the paper is:

`XMST-S1 spatial clustering -> XMST-S3 transverse-kinematic refinement -> XMST-S2 reddening refinement`

Additional observables refine an existing spatial hierarchy; they are not
combined with position in a common multidimensional metric. Spatial
reconstruction during refinement uses the Stage-1 fracture scale.

## Contents

- `code/run_xmst_s3.py` — frozen implementation containing the Stage-1, S2 and S3 functions. Its original CLI executes S1 -> S2 -> S3 and is retained unchanged for provenance; it is **not** the preferred Paper II execution order.
- `code/run_paired_order_mc.py` — paired execution-order experiment. It runs both S1 -> S2 -> S3 and S1 -> S3 -> S2 on matched realisations; the latter is the preferred order adopted in Paper II.
- `code/run_paper2_s2_uncertainty_1000.py` — 1000-field S2 reddening-uncertainty experiment.
- `code/run_paper2_s3_uncertainty_exact_provenance_fast.py` — final 1000-field exact-provenance S3 velocity-uncertainty experiment.
- `code/run_xmst_s3_mc1000_same_fields.py`, `code/cdf_worker.py`, and `code/run_xmsts2.py` — frozen validation dependencies required by that exact-provenance rerun.
- `code/run_paper2_s3_gate_sensitivity_exact_table3_fast.py` — S3 gate-sensitivity rerun tied to the Table 3 ΔV=3 km/s realisations.
- `results/` — compact/all-run CSV outputs for the paired-order, S2 uncertainty,
  exact-provenance S3 uncertainty and S3 gate-sensitivity experiments, plus a compact blind-Q26 checkpoint.
- `paper/` — submitted manuscript source, bibliography and compiled PDF.

## Stage 1

XMST-S1 was introduced and validated in Paper I and is maintained separately
in the `XMST-Stage-1` repository. Paper II does not change the spatial
Percolation-Jenks method.

## External catalogue data

The Q25/Q26 catalogue data are not redistributed here. They should be obtained
from the original published catalogue sources cited in the paper. This archive
contains derived validation products rather than a replacement copy of the
published source catalogues. The blind-Q26 directory contains only compact
aggregate checkpoints; source-level Q25/Q26 catalogue rows are not included.

## Reproducibility checkpoints

For the full Q25 parent catalogue the Paper II Stage-1 run uses:
- 24,706 input stars
- fracture scale ≈ 17.048 pc
- minimum retained membership N = 10
- 103 Stage-1 groups containing 2,339 stars

The paper reports all numerical acceptance gates and random-state choices.

## Software

Python 3 with NumPy, pandas, SciPy and scikit-learn. The release audit includes Python syntax/CLI checks and a deterministic synthetic Stage-1/S2 smoke run. A network-isolated audit environment prevented a fresh package download, so installation from an empty environment was not used as a release criterion.

## Licence

Code: MIT Licence.  
Paper and derived tabular research products: CC BY 4.0 is recommended for the
Zenodo record.

## Citation

Please cite the Paper II manuscript and the archived software release.

- Version 1.0.0 DOI: [10.5281/zenodo.22830657](https://doi.org/10.5281/zenodo.22830657)
- Concept DOI (all versions): [10.5281/zenodo.22830656](https://doi.org/10.5281/zenodo.22830656)

Software citation metadata are provided in `CITATION.cff`.
