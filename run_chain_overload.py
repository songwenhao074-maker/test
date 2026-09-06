"""Full-chain training scheduler for the QoS-overload final dataset.

15 runs (5 variants x 3 seeds), held-out protocol (TRAIN_ROWS=140),
Q=8 (V2_PROTO_DIM=8), graph-path variants (v3b2/v4c1) get RULE_AUX +
V2_GRAPH_LR supervision.  Three parallel workers, serial queue, per-run
logs.  Polls logs only (never job_output waits on the train jobs).

Usage: python run_chain_overload.py
"""
import os
import sys
import time
import subprocess
from collections import deque

VARIANTS = [v for v in ['v0', 'v1b', 'v2c2', 'v3b2', 'v4c1']
            if not os.environ.get('CHAIN_VARIANTS')
            or v in os.environ['CHAIN_VARIANTS'].split(',')]
SEEDS = [1, 2, 6]
# v3b2/v4c1: FREEZE_AFTER forces the graph path to absorb the collision-
# pair errors (multi-seed data has 130 natural pairs).  EXPLICIT/SCALE
# boost the graph signal; GRAPH_AUX mounts the aux_fn; GRAPH_LR speeds the
# graph params.  v4c1 additionally keeps its host-MHA trainable.
GRAPH_AUX_ENV = {
    'GRAPH_AUX': '1.0',
    'GRAPH_AUX_MARGIN_POS': '1.0',
    'GRAPH_AUX_MARGIN_NEG': '2.0',
    'V2_GRAPH_LR': '0.01',
    'V2_GRAPH_EXPLICIT': '1',
    'V2_GRAPH_SCALE': '3',
    'FREEZE_AFTER': '30',
    'FREEZE_KEEP': 'graph,attn,mha',
}
CKPT = 'recovery/PreGANSrc/checkpoints_qos_overload_ms_tol1'
LOG = 'logs/train_overload_ms_tol1'


def build_env(variant):
    env = dict(os.environ)
    env.update({
        'OMP_NUM_THREADS': '3',
        'DATA_VERSION': 'qos_overload_ms_tol1',
        'V2_CKPT': CKPT + '/',
        'LABEL_MODE': 'overload',
        'TRAIN_ROWS': '404',
        'NUM_EPOCHS': '85',
        'EVAL_EVERY': '5',
        'FORCE': '1',
        'SAVE_EPOCH': '85',
        'V2_MOE_EXPERTS': '12',
        'V2_DROPOUT': '0',
        'V2_GRAPH_DROPOUT': '0',
        'V2_MOE_HEAD': '1',
        'V2_PROTO_DIM': '2',
    })
    if variant in ('v3b2', 'v4c1'):
        env.update(GRAPH_AUX_ENV)
    else:
        # force-clean graph envs so a stale process env never leaks in
        for k in GRAPH_AUX_ENV:
            env.pop(k, None)
    return env


def main():
    queue = deque((v, s) for v in VARIANTS for s in SEEDS)
    workers = {}          # slot -> (proc, tag)
    t0 = time.time()
    while queue or workers:
        # fill free slots
        for slot in list(workers):
            proc, tag = workers[slot]
            if proc.poll() is not None:
                print(f'[done] {tag} rc={proc.returncode} '
                      f'{time.time() - t0:.0f}s', flush=True)
                del workers[slot]
        while len(workers) < 3 and queue:
            v, s = queue.popleft()
            log_path = os.path.join(LOG, f'{v}_s{s}.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                proc = subprocess.Popen(
                    [sys.executable, 'train_progressive_v2.py', v, str(s)],
                    env=build_env(v), stdout=f, stderr=subprocess.STDOUT,
                    cwd=os.getcwd())
            workers[proc.pid] = (proc, f'{v}_s{s}')
            print(f'[start] {v}_s{s} (workers={len(workers)})', flush=True)
        time.sleep(15)
    print(f'ALL 15 RUNS DONE in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
