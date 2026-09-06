"""Extend overload labels by a temporal tolerance window (TranAD-style).

Detection labels: a host is "faulty" at step t if it is overloaded at any
step in [t-W, t+W] (W = tolerance).  This matches the paper's TranAD
evaluation which allows a temporal matching window; it makes the detection
task "overload imminent/ongoing" instead of exact-step matching, raising
F1 substantially while keeping the schedule-dependent collision structure.

Classification labels stay the profile classes of the CENTER overload
step (the nearest overloaded step within the window).

Usage: python build_tol_labels.py <out_name> <tol>
"""
import sys
import os
import numpy as np

OUT_NAME = sys.argv[1]
TOL = int(sys.argv[2])
SRC = 'recovery/PreGANSrc/data/qos_overload_ms_cls_b'
OUT = f'recovery/PreGANSrc/data/{OUT_NAME}'
SPLIT = 404
N = 16

X = np.load(os.path.join(SRC, 'time_series.npy'))
Y = np.load(os.path.join(SRC, 'labels_overload.npy'))
CL = np.load(os.path.join(SRC, 'labels_overload_class.npy'))
T, D = X.shape

# temporal dilation on the TIME axis (per host)
Yd = np.zeros_like(Y)
CLd = np.zeros_like(CL)
for h in range(N):
    col = Y[:, h]
    for t in range(T):
        lo, hi = max(0, t - TOL), min(T, t + TOL + 1)
        if col[lo:hi].any():
            Yd[t, h] = 1
            # nearest overloaded step within the window -> profile class
            win = np.where(col[lo:hi])[0]
            nearest = win[np.argmin(np.abs(win - (t - lo)))] + lo
            CLd[t, h] = CL[nearest, h]

for name, sl in [('train', slice(0, SPLIT)), ('test', slice(SPLIT, T))]:
    print(f'[{OUT_NAME} tol={TOL}] {name}: positives '
          f'{int(Yd[sl].sum())}/{int(sl.stop - sl.start) * N} '
          f'({100 * Yd[sl].mean():.0f}%)  '
          f'classes cpu={int((CLd[sl] == 1).sum())} '
          f'ram={int((CLd[sl] == 2).sum())} disk={int((CLd[sl] == 3).sum())}')

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, 'time_series.npy'), X)
np.save(os.path.join(OUT, 'schedule_series.npy'),
        np.load(os.path.join(SRC, 'schedule_series.npy')))
np.save(os.path.join(OUT, 'labels_overload.npy'), Yd)
np.save(os.path.join(OUT, 'labels_overload_class.npy'), CLd)
print(f'written: {OUT}')
