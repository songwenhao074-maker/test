"""Tau sweep for the overload chain: per-variant best F1 over P1 logit
thresholds on the test segment (protocol-consistent normalization).

The online QoS path uses DETECT_TAU (P1 > tau); eval_holdout uses argmax
(tau=0).  Variants may have different P1 offsets — sweep tau to compare
fairly at each variant's best operating point.

Usage: python tau_sweep_chain.py <ckpt_dir> [variants...]
"""
import os
import sys
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

from recovery.PreGANSrc.src.constants import data_folder
from recovery.PreGANSrc.src.utils import (load_npyfile, convert_to_windows,
                                           convert_schedule_to_windows,
                                           load_model)
from recovery.PreGANSrc.src.models_v2 import (FTMoE_v0_16_v2, FTMoE_v1b_16_v2,
                                              FTMoE_v2c2_16_v2,
                                              FTMoE_v3b2_16_v2,
                                              FTMoE_v4c1_16_v2)

os.environ['V2_MOE_EXPERTS'] = '12'
os.environ['V2_DROPOUT'] = '0'
os.environ['V2_GRAPH_DROPOUT'] = '0'
os.environ['V2_MOE_HEAD'] = '1'
os.environ['V2_PROTO_DIM'] = os.environ.get('V2_PROTO_DIM', '2')
DATA_VERSION = os.environ.get('DATA_VERSION', 'qos_overload_final')

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    'recovery/PreGANSrc/checkpoints_qos_overload_final_q2'
VARIANTS = {'v0': FTMoE_v0_16_v2, 'v1b': FTMoE_v1b_16_v2,
            'v2c2': FTMoE_v2c2_16_v2, 'v3b2': FTMoE_v3b2_16_v2,
            'v4c1': FTMoE_v4c1_16_v2}
SEEDS = [1, 2, 6]
SPLIT = int(os.environ.get('TRAIN_ROWS', '140'))

X = np.load(os.path.join(data_folder, DATA_VERSION, 'time_series.npy'))
S = np.load(os.path.join(data_folder, DATA_VERSION, 'schedule_series.npy'))
Y = np.load(os.path.join(data_folder, DATA_VERSION, 'labels_overload_class.npy'))
T, D = X.shape
N = D // 7
mx = np.max(X[:SPLIT], axis=0) + 1e-8
time_n = X / mx
ts_full = torch.tensor(S).double()
yt = (Y[SPLIT:] > 0).ravel()

taus = np.arange(-2.0, 2.01, 0.1)

for v, cls in VARIANTS.items():
    p1_all = []
    for s in SEEDS:
        path = os.path.join(CKPT, f'simulator_{cls().name}_s{s}_snap85.ckpt')
        if not os.path.exists(path):
            continue
        model, _, _, _ = load_model(CKPT, os.path.basename(path), cls.__name__)
        tt = convert_to_windows(time_n, model)
        ts = convert_schedule_to_windows(ts_full, model)
        model.eval()
        p1 = np.zeros((T, N))
        with torch.no_grad():
            for i in range(T):
                sa, _sp = model(tt[i], ts[i])
                for j, s_ in enumerate(sa):
                    p1[i, j] = s_[0, 1].item()
        p1_all.append(p1[SPLIT:].ravel())
    if not p1_all:
        continue
    P = np.stack(p1_all)          # (seeds, test_hosts)
    best = []
    for si, p in enumerate(p1_all):
        row = []
        for tau in taus:
            pred = p > tau
            tp = int((pred & yt).sum()); fp = int((pred & ~yt).sum())
            fn = int((~pred & yt).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
            row.append((f1, tp, fp, fn))
        b = max(row, key=lambda r: r[0])
        tau_b = taus[row.index(b)]
        best.append((tau_b, b))
    mean_tau = np.mean([b[0] for b in best])
    mean_f1 = np.mean([b[1][0] for b in best])
    print(f'{v}: best-tau per seed {[round(b[0], 1) for b in best]} '
          f'F1 {[round(b[1][0], 3) for b in best]} '
          f'mean-tau {mean_tau:.2f} mean-F1 {mean_f1:.4f}')
    # common tau = mean of per-seed best taus
    for si, p in enumerate(p1_all):
        tau_c = best[si][0]
        pred = p > tau_c
        tp = int((pred & yt).sum()); fp = int((pred & ~yt).sum())
        fn = int((~pred & yt).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
        print(f'    s{SEEDS[si]} tau={tau_c:.1f}: f1={f1:.3f} '
              f'tp={tp} fp={fp} fn={fn}')
