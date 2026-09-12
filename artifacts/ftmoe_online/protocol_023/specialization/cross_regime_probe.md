# Protocol 023 — S3 cross-regime specialization probe

Stream: `artifacts/ftmoe_online/protocol_023/development_streams/dev_seed700_steps2880` · steps 2880 · features 49 · horizon(s) [1, 4]

Onset resource per regime: `{'compute_first': 'cpu', 'memory_first': 'ram', 'io_first': 'disk'}`

## Cells (train -> test)

| train | test | h | target | AP | ROC-AUC | FPR | prevalence | positives |
|---|---|---:|---|---:|---:|---:|---:|---:|
| compute_first | compute_first | 1 | cpu_onset | 0.3383 | 0.9652 | 0.0159 | 0.0234 | 127 |
| compute_first | memory_first | 1 | cpu_onset | 0.0565 | 0.8882 | 0.0076 | 0.0088 | 50 |
| compute_first | io_first | 1 | cpu_onset | 0.0476 | 0.7998 | 0.0122 | 0.0132 | 74 |
| compute_first | compute_first | 4 | cpu_onset | 0.2586 | 0.6862 | 0.0656 | 0.0886 | 480 |
| compute_first | memory_first | 4 | cpu_onset | 0.1048 | 0.7233 | 0.0297 | 0.0340 | 193 |
| compute_first | io_first | 4 | cpu_onset | 0.2010 | 0.8120 | 0.0467 | 0.0530 | 296 |
| memory_first | compute_first | 1 | ram_onset | 0.0109 | 0.8354 | 0.0026 | 0.0025 | 14 |
| memory_first | memory_first | 1 | ram_onset | 0.0807 | 0.9300 | 0.0071 | 0.0074 | 42 |
| memory_first | io_first | 1 | ram_onset | 0.0054 | 0.7462 | 0.0021 | 0.0021 | 12 |
| memory_first | compute_first | 4 | ram_onset | 0.0634 | 0.8280 | 0.0088 | 0.0098 | 56 |
| memory_first | memory_first | 4 | ram_onset | 0.0643 | 0.6294 | 0.0270 | 0.0285 | 162 |
| memory_first | io_first | 4 | ram_onset | 0.0117 | 0.5292 | 0.0090 | 0.0089 | 51 |
| io_first | compute_first | 1 | disk_onset | 0.2407 | 0.9896 | 0.0010 | 0.0016 | 9 |
| io_first | memory_first | 1 | disk_onset | 0.0306 | 0.8716 | 0.0023 | 0.0023 | 13 |
| io_first | io_first | 1 | disk_onset | 0.0136 | 0.9599 | 0.0009 | 0.0009 | 5 |
| io_first | compute_first | 4 | disk_onset | 0.0613 | 0.8577 | 0.0058 | 0.0063 | 36 |
| io_first | memory_first | 4 | disk_onset | 0.1946 | 0.9212 | 0.0067 | 0.0091 | 52 |
| io_first | io_first | 4 | disk_onset | 0.0285 | 0.8249 | 0.0035 | 0.0035 | 20 |

## §13 specialization gate

Rule: within-regime AP - mean cross-regime AP >= 0.05 for at least 2 of 3 regimes

| regime | within AP | mean cross AP | gap | passes |
|---|---:|---:|---:|---|
| compute_first | 0.3383 | 0.0520 | 0.2863 | True |
| memory_first | 0.0807 | 0.0081 | 0.0726 | True |
| io_first | 0.0136 | 0.1356 | -0.1220 | False |

**2/3 regimes pass at h=1.** specialization opportunity present
