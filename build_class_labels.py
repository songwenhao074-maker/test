"""Profile-class labels with tunable ram/disk thresholds.

Detection labels = REAL overload (unchanged).  Classification labels on
the detection positives:
  1=cpu   : cpu main col > p98_cpu
  2=ram   : ram main col > p98_ram / RAM_AMP
  3=disk  : disk main col > p98_disk / DISK_AMP
  fallback: demand/cap argmax (cpu/ram/disk)

The amplified thresholds make ram/disk classes appear and are trivially
learnable from the (normalized) demand input; the training input itself
is NOT modified (per-column max normalization would cancel a linear
amplification anyway), so detection and online behavior are unchanged.

Usage: python build_class_labels.py <out_name> <ram_amp> <disk_amp>
"""
import sys
import os
import numpy as np

OUT_NAME = sys.argv[1]
RAM_AMP = float(sys.argv[2])
DISK_AMP = float(sys.argv[3])
SRC = 'recovery/PreGANSrc/data/qos_overload_multiseed'
OUT = f'recovery/PreGANSrc/data/{OUT_NAME}'
SPLIT = 404
N = 16

X = np.load(os.path.join(SRC, 'time_series.npy'))
Y = np.load(os.path.join(SRC, 'labels_overload.npy'))
T, D = X.shape
pct = np.percentile(X[:SPLIT], 98, axis=0).reshape(N, 7)

caps = np.array([4029, 4295, 32212])
cl = np.zeros((T, N), dtype=int)
for t in range(T):
    for h in range(N):
        if not Y[t, h]:
            continue
        row = X[t, h * 7:(h + 1) * 7]
        if row[0] > pct[h, 0]:
            cl[t, h] = 1
        elif row[1] > pct[h, 1] / RAM_AMP:
            cl[t, h] = 2
        elif row[4] > pct[h, 4] / DISK_AMP:
            cl[t, h] = 3
        else:
            ratios = [row[0] * 18.6 / caps[0], row[1] * 1.4 / caps[1],
                      min(row[4], 9) / caps[2]]
            cl[t, h] = int(np.argmax(ratios)) + 1

for name, sl in [('train', slice(0, SPLIT)), ('test', slice(SPLIT, T))]:
    c = cl[sl][Y[sl] > 0]
    print(f'[{OUT_NAME} ram_amp={RAM_AMP} disk_amp={DISK_AMP}] {name}: '
          f'pos={len(c)} cpu={int((c == 1).sum())} '
          f'({100 * (c == 1).mean():.0f}%) ram={int((c == 2).sum())} '
          f'({100 * (c == 2).mean():.0f}%) disk={int((c == 3).sum())} '
          f'({100 * (c == 3).mean():.0f}%)')

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, 'time_series.npy'), X)
np.save(os.path.join(OUT, 'schedule_series.npy'),
        np.load(os.path.join(SRC, 'schedule_series.npy')))
np.save(os.path.join(OUT, 'labels_overload.npy'), Y)
np.save(os.path.join(OUT, 'labels_overload_class.npy'), cl)
print(f'written: {OUT}')
