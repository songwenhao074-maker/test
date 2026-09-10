# Protocol 020 R1 development analysis

Status: **COMPLETE**. The grid is development data only; this report does not claim generalization, D effectiveness, or final confirmation.

Protective attribution eligibility: **True**.

| replay | method | status | full F1 | full PR-AUC | worst registered F1 | anchor F1 | conditional resource F1 | end-to-end resource F1 | deploy ratio | base-only ratio | accepted / rejected | elapsed s | RSS GiB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | A | complete | 0.5118 | 0.6779 | 0.0179 | 0.6120 | 0.9336 | 0.4598 | — | — | — / — | 22.09 | 0.319 |
| 500 | C-legacy | complete | 0.4701 | 0.6179 | 0.0000 | 0.5217 | 0.9663 | 0.4512 | — | — | — / — | 39.34 | 0.375 |
| 500 | C-residual-off | complete | 0.5026 | 0.6782 | 0.0183 | 0.5750 | 0.9346 | 0.4547 | 1.0000 | 0.0000 | 0 / 0 | 117.19 | 0.422 |
| 500 | C-residual-on | complete | 0.5091 | 0.6769 | 0.0180 | 0.6120 | 0.9336 | 0.4577 | 0.8585 | 0.1415 | 6 / 12 | 133.91 | 0.475 |
| 501 | A | complete | 0.1053 | 0.1270 | 0.1053 | 0.6120 | 0.3195 | 0.0381 | — | — | — / — | 22.04 | 0.323 |
| 501 | C-legacy | complete | 0.1301 | 0.0969 | 0.1301 | 0.6217 | 0.4524 | 0.0557 | — | — | — / — | 39.29 | 0.376 |
| 501 | C-residual-off | complete | 0.1057 | 0.1284 | 0.1057 | 0.6157 | 0.3195 | 0.0383 | 1.0000 | 0.0000 | 0 / 0 | 116.22 | 0.422 |
| 501 | C-residual-on | complete | 0.1053 | 0.1285 | 0.1053 | 0.6170 | 0.3195 | 0.0381 | 0.4955 | 0.5045 | 3 / 15 | 102.90 | 0.440 |

Registered phases use the stream manifest. A single-phase dev501 stream remains one phase; its 400-step slices are reported separately as time blocks.

Candidate counts refer to evaluated residual assessment snapshots, not new experts.

Input/completeness issues: none detected.
