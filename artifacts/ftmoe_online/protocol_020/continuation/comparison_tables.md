# Protocol 020 continuation comparisons

All numbers are development results from the common frozen S6 checkpoint. No independent confirmation has been performed. Labels use the registered one-step tolerance.

Reference drops from legacy Protocol 004 and current P20 anchors must not be compared directly.

## drift

| Run | F1 | PR-AUC | FP | FN | Births | Reference F1 drop | Reference source |
|---|---:|---:|---:|---:|---:|---:|---|
| A_model1_seed500_drift | 0.5118 | 0.6779 | 1159 | 142 | 0 | +0.0000 | legacy_protocol004_reference |
| C_model1_seed500_drift | 0.4969 | 0.6720 | 1260 | 135 | 0 | -0.0674 | legacy_protocol004_reference |
| B_lr1e4_dev500 | 0.4721 | 0.6104 | 1377 | 144 | 0 | -0.1305 | legacy_protocol004_reference |
| B_lr1e5_dev500 | 0.4989 | 0.6710 | 1240 | 138 | 0 | +0.0747 | same_domain_train_anchor |
| B_w1_lr1e4_dev500 | 0.5450 | 0.5783 | 762 | 230 | 0 | +0.0402 | same_domain_train_anchor |
| C_lr1e4_dev500 | 0.4701 | 0.6179 | 1412 | 137 | 0 | -0.1619 | legacy_protocol004_reference |
| C_lr1e5_dev500_verified | 0.4969 | 0.6720 | 1260 | 135 | 0 | +0.0681 | same_domain_train_anchor |
| C_w1_lr1e4_dev500 | 0.5419 | 0.5946 | 761 | 235 | 0 | +0.0475 | same_domain_train_anchor |
| D_v3_lr1e4_dev500 | 0.4719 | 0.6313 | 1391 | 140 | 1 | +0.0987 | same_domain_train_anchor |
| D_v3_lr1e5_dev500 | 0.4977 | 0.6717 | 1256 | 135 | 1 | +0.0351 | same_domain_train_anchor |
| D_v3_w1_lr1e4_dev500 | 0.5419 | 0.5946 | 761 | 235 | 0 | +0.0475 | same_domain_train_anchor |

## stationary

| Run | F1 | PR-AUC | FP | FN | Births | Reference F1 drop | Reference source |
|---|---:|---:|---:|---:|---:|---:|---|
| A_model1_seed501 | 0.1053 | 0.1270 | 1054 | 17 | 0 | +0.0000 | legacy_protocol004_reference |
| B_model1_seed501 | 0.1338 | 0.1320 | 701 | 24 | 0 | -0.0050 | legacy_protocol004_reference |
| C_model1_seed501 | 0.1272 | 0.1349 | 774 | 22 | 0 | -0.0101 | legacy_protocol004_reference |
| D_v3_lr1e4_dev501 | 0.1301 | 0.0969 | 696 | 26 | 0 | -0.0098 | same_domain_train_anchor |
| D_v3_lr1e5_dev501 | 0.1272 | 0.1349 | 774 | 22 | 0 | -0.0181 | same_domain_train_anchor |
| D_v3_w1_lr1e4_dev501 | 0.1345 | 0.1186 | 544 | 35 | 0 | +0.0104 | same_domain_train_anchor |

