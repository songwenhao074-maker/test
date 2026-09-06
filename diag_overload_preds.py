"""Diagnose held-out overload predictions: prec/rec + FP/FN host profile.

Usage: python diag_overload_preds.py <ckpt_dir> [variant...]
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
DATA_VERSION = os.environ.get('DATA_VERSION', 'qos_overload')

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    'recovery/PreGANSrc/checkpoints_qos_overload_holdout'
VARIANTS = {'v0': FTMoE_v0_16_v2, 'v1b': FTMoE_v1b_16_v2,
            'v2c2': FTMoE_v2c2_16_v2, 'v3b2': FTMoE_v3b2_16_v2,
            'v4c1': FTMoE_v4c1_16_v2}
if len(sys.argv) > 2:
    VARIANTS = {k: VARIANTS[k] for k in sys.argv[2:] if k in VARIANTS}
SPLIT = int(os.environ.get('TRAIN_ROWS', '140'))

X = np.load(os.path.join(data_folder, DATA_VERSION, 'time_series.npy'))
S = np.load(os.path.join(data_folder, DATA_VERSION, 'schedule_series.npy'))
Y = np.load(os.path.join(data_folder, DATA_VERSION, 'labels_overload_class.npy'))
T, D = X.shape
N = D // 7
mx = np.max(X[:SPLIT], axis=0) + 1e-8
time_n = X / mx
ts_full = torch.tensor(S).double()

for v, cls in VARIANTS.items():
    path = os.path.join(CKPT, f'simulator_{cls().name}_s1_snap85.ckpt')
    if not os.path.exists(path):
        print(f'{v}: no ckpt {path}')
        continue
    model, _, _, _ = load_model(CKPT, os.path.basename(path), cls.__name__)
    tt = convert_to_windows(time_n, model)
    ts = convert_schedule_to_windows(ts_full, model)
    model.eval()
    preds = np.zeros((T, N), dtype=int)
    with torch.no_grad():
        for i in range(T):
            sa, _sp = model(tt[i], ts[i])
            for j, s in enumerate(sa):
                preds[i, j] = torch.argmax(s).item()
    yt = (Y[SPLIT:] > 0).ravel()
    pt = (preds[SPLIT:] > 0).ravel()
    tp = int((pt & yt).sum()); fp = int((pt & ~yt).sum())
    fn = int((~pt & yt).sum()); tn = int((~pt & ~yt).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    # step-level: any-host trigger
    step_y = (Y[SPLIT:].sum(1) > 0)
    step_p = (preds[SPLIT:].sum(1) > 0)
    stp = int((step_p & step_y).sum()); sfp = int((step_p & ~step_y).sum())
    sfn = int((~step_p & step_y).sum())
    print(f'{v}: test host-step prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} '
          f'tp={tp} fp={fp} fn={fn} tn={tn}')
    print(f'   step-level: trigger {step_p.sum()}/62, pos-steps {step_y.sum()}, '
          f'tp={stp} fp={sfp} fn={sfn}')
    # which hosts are FP/FN
    TN_ = T - SPLIT
    fp_hosts = np.where((pt & ~yt).reshape(TN_, N).sum(0))[0]
    fn_hosts = np.where((~pt & yt).reshape(TN_, N).sum(0))[0]
    print(f'   FP hosts: {fp_hosts.tolist()}  FN hosts: {fn_hosts.tolist()}')
