# Exact-provenance XMST-S3 uncertainty rerun

1000 realisations; DeltaV=5 km/s; intrinsic sigma_v=1 km/s per component.
The original Table-3 intrinsic velocity seed is MC seed + run + 30,000,000.
Frozen S3 SHA256: cd95b3a87a0942732f007f7ca82c9986080706c4b46f575039efe00f4710eb32.
S3 is applied only to the Stage-1 T01/T02 parent, as in the original targeted Table-3 experiment.
The zero-error rows are the original archived Table-3 DeltaV=5 per-realisation results and match exactly.
Non-zero observational errors use one independent paired Gaussian draw per realisation, seed 20260831 + run, rescaled across 0.25, 0.5, 1 and 2 km/s per component.

See PROVENANCE_CHECK.json and the aggregate/CI CSVs for results.
