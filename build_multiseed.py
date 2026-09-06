"""Build the multi-seed overload dataset: 3 deterministic gobi replays
(seeds 42/1/6) of the SAME trace with DIFFERENT schedules/labels.

Layout (606 rows = 3 x 202):
  time_series.npy        = the qos trace repeated 3x (identical across seeds)
  schedule_series.npy    = per-seed GOBI result_cache (differs)
  labels_overload_class.npy = per-seed overload labels (differs)
  labels_overload.npy    = binary version

Train split = seeds 42+1 (rows 0:404), test split = seed 6 (rows 404:606).
Same trace rows appear in both splits with different labels -> natural
collision pairs; the schedule signal is necessary and sufficient.
"""
import os
import numpy as np

SEEDS = ['42', '1', '6']
SRC = {s: f'recovery/PreGANSrc/data/qos_overload_rs35_s{s}' for s in SEEDS}
OUT = 'recovery/PreGANSrc/data/qos_overload_multiseed'

ts = []
ss = []
lab = []
labb = []
for s in SEEDS:
    d = SRC[s]
    ts.append(np.load(os.path.join(d, 'time_series.npy')))
    ss.append(np.load(os.path.join(d, 'schedule_series.npy')))
    lab.append(np.load(os.path.join(d, 'labels_overload_class.npy')))
    labb.append(np.load(os.path.join(d, 'labels_overload.npy')))

T = np.concatenate(ts)
S = np.concatenate(ss)
L = np.concatenate(lab)
B = np.concatenate(labb)
print(f'time {T.shape} schedule {S.shape} labels {L.shape}')
print(f'per-seed anomaly host-steps: '
      f'{[(s, int(l.sum())) for s, l in zip(SEEDS, lab)]}')
print(f'train (0:404) anomalies: {int(L[:404].sum())} '
      f'({int((L[:404] == 1).sum())} cpu + {int((L[:404] == 2).sum())} ram)')
print(f'test  (404:606) anomalies: {int(L[404:].sum())} '
      f'({int((L[404:] == 1).sum())} cpu + {int((L[404:] == 2).sum())} ram)')

# natural collision pairs within the TRAIN split: identical time windows
# (same trace rows across seeds) with opposite labels
from collections import defaultdict
groups = defaultdict(list)
for i in range(3, 404):
    groups[T[i - 3:i].tobytes()].append(i)
pairs = 0
for rows in groups.values():
    if len(rows) < 2:
        continue
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            i, j = rows[a], rows[b]
            for h in range(16):
                li, lj = int(L[i, h]), int(L[j, h])
                if (li > 0 and lj == 0) or (li == 0 and lj > 0):
                    pairs += 1
print(f'train-split collision pairs: {pairs}')

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, 'time_series.npy'), T)
np.save(os.path.join(OUT, 'schedule_series.npy'), S)
np.save(os.path.join(OUT, 'labels_overload_class.npy'), L)
np.save(os.path.join(OUT, 'labels_overload.npy'), B)
print(f'written: {OUT}')
