# Protocol 023 round 2A — H2-v2 specialization probe

Streams: `single_compute_first_seed700_steps1200_calibrated` (reliability) and `dev_seed700_steps2880_calibrated` (gap) · h=1

| regime | within AP | mean cross AP | gap | gap ok | positives@0.50 | positives@0.30 | positives ok | passes |
|---|---:|---:|---:|---:|---:|---:|---|---|
| compute_first | 0.2808 | 0.0545 | 0.2262 | True | 185 | 249 | True | True |
| memory_first | 0.2262 | 0.0100 | 0.2162 | True | 124 | 165 | True | True |
| io_first | 0.1366 | 0.0138 | 0.1228 | True | 31 | 44 | True | True |

**3/3 regimes pass both conditions** (gap ok 3/3, positives ok 3/3); ideal 3/3. specialization opportunity present (H2-v2 PASS)

Rule: within AP - mean cross AP >= 0.05 AND within positives >= 30, for >= 2 of 3 regimes

Reliability is measured on the calibrated single-regime streams; the gap is measured on the development stream. See the JSON for the per-cell windows and prevalences.
