"""Tolerance-window evaluation of the overload chain (TranAD-style).

A prediction at (t,h) is a TRUE POSITIVE if the label is positive at any
step in [t-W, t+W] on the same host.  This is the paper's temporal
matching convention; the training labels stay strict.

Usage: python tol_eval.py <ckpt_dir> [tol] [variants...]
Env: DATA_VERSION, TRAIN_ROWS, V2_GRAPH_EXPLICIT, V2_GRAPH_SCALE
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
DATA_VERSION = os.environ.get('DATA_VERSION', 'qos_overload_ms_cls_b')
SPLIT = int(os.environ.get('TRAIN_ROWS', '404'))

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    'recovery/PreGANSrc/checkpoints_qos_overload_ms_cls'
TOL = int(sys.argv[2]) if len(sys.argv) > 2 else 1
VARIANTS = {'v0': FTMoE_v0_16_v2, 'v1b': FTMoE_v1b_16_v2,
            'v2c2': FTMoE_v2c2_16_v2, 'v3b2': FTMoE_v3b2_16_v2,
            'v4c1': FTMoE_v4c1_16_v2}
if len(sys.argv) > 3:
    VARIANTS = {k: VARIANTS[k] for k in sys.argv[3:] if k in VARIANTS}
SEEDS = [1, 2, 6]

X = np.load(os.path.join(data_folder, DATA_VERSION, 'time_series.npy'))
S = np.load(os.path.join(data_folder, DATA_VERSION, 'schedule_series.npy'))
Y = np.load(os.path.join(data_folder, DATA_VERSION, 'labels_overload.npy'))
T, D = X.shape
N = D // 7
mx = np.max(X[:SPLIT], axis=0) + 1e-8
time_n = X / mx
ts_full = torch.tensor(S).double()
yt = (Y[SPLIT:] > 0)                                   # (T-SPLIT, N) bool

# dilated label for TP matching
Yd = np.zeros_like(yt)
for h in range(N):
    col = yt[:, h]
    for t in range(len(col)):
        lo, hi = max(0, t - TOL), min(len(col), t + TOL + 1)
        if col[lo:hi].any():
            Yd[t, h] = 1

print(f'=== tol-eval: {CKPT} tol={TOL} (pos {int(yt.sum())} -> '
      f'{int(Yd.sum())} dilated) ===')
print(f'{"variant":6s} {"meanF1":>7s} {"per-seed":>22s} '
      f'{"prec":>6s} {"rec":>6s}')
for v, cls in VARIANTS.items():
    f1s, precs, recs = [], [], []
    for s in SEEDS:
        path = os.path.join(CKPT, f'simulator_{cls().name}_s{s}_snap85.ckpt')
        if not os.path.exists(path):
            continue
        model, _, _, _ = load_model(CKPT, os.path.basename(path), cls.__name__)
        tt = convert_to_windows(time_n, model)
        ts = convert_schedule_to_windows(ts_full, model)
        model.eval()
        preds = np.zeros((T, N), dtype=int)
        with torch.no_grad():
            for i in range(T):
                sa, _sp = model(tt[i], ts[i])
                for j, s_ in enumerate(sa):
                    preds[i, j] = torch.argmax(s_).item()
        pt = preds[SPLIT:] > 0
        tp = int((pt & Yd).sum()); fp = int((pt & ~Yd).sum())
        fn = int((~pt & Yd).sum())
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        f1s.append(f1); precs.append(prec); recs.append(rec)
    if not f1s:
        continue
    print(f'{v:6s} {np.mean(f1s):7.4f} '
          f'{str([round(x, 3) for x in f1s]):>22s} '
          f'{np.mean(precs):6.3f} {np.mean(recs):6.3f}')
