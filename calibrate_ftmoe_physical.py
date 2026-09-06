"""Model-free training-seed calibration of physical resource bottlenecks."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import psutil
from prepare_ftmoe_end_to_end import ART, ROOT, guard, sha


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    registration = json.loads((ART / 'protocol_004_physical.json').read_text())
    destination = ART / 'physical_calibration/candidate_001_seed42'
    if (destination / 'summary.json').exists():
        print((destination / 'summary.json').read_text()); return
    guard()
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, **registration['initial_environment'], QOS_OVERLOAD_OUT=str(destination))
    environment.pop('FIXED_SCHEDULE_PATH', None)
    with (destination / 'generation.log').open('w') as log:
        subprocess.run([sys.executable, '-u', 'dump_replay.py', '42'], cwd=ROOT, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    labels = np.load(destination / 'labels_overload_class.npy')
    counts = np.bincount(labels.ravel(), minlength=4)
    summary = {'environment': registration['initial_environment'], 'class_counts': counts.tolist(),
               'positive_fraction': float((labels > 0).mean()),
               'positive_class_fractions': (counts[1:] / max(counts[1:].sum(), 1)).tolist(),
               'replay_sha256': sha(destination / 'replay_log.npz'),
               'code_sha256': {p: sha(ROOT / p) for p in ['dump_replay.py', 'simulator/workload/OfflineTraceWorkloadV2.py', 'simulator/environment/RPiEdge.py']}}
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
