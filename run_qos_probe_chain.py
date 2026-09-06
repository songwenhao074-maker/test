"""Serial QoS probes for the final overload chain (15 runs).

Usage: python run_qos_probe_chain.py [--ckpt DIR] [--start v0] [--seed 1]
Env: CHAIN_CKPT_DIR, ONLINE_TUNE (default 0), RAM_SCALE (default 3.5)
"""
import os
import sys
import time
import argparse
import subprocess

MODES = ['v0', 'v1b', 'v2c2', 'v3b2', 'v4c1']
SEEDS = [1, 2, 6]


def run_one(mode, seed, log_path, timeout=1500):
    env = dict(os.environ)
    env.update({
        'PYTHONUTF8': '1',
        'PYTHONIOENCODING': 'utf-8',
        'CHAIN_CKPT_DIR': env.get(
            'CHAIN_CKPT_DIR',
            'recovery/PreGANSrc/checkpoints_qos_overload_final_q2'),
        'V2_MOE_EXPERTS': '12',
        'V2_DROPOUT': '0',
        'V2_GRAPH_DROPOUT': '0',
        'V2_MOE_HEAD': '1',
        'V2_PROTO_DIM': '2',
        'ONLINE_TUNE': env.get('ONLINE_TUNE', '0'),
        'QOS_PROBE_OUT': f'logs/qos_probe_final_q2/{mode}_s{seed}.csv',
    })
    os.makedirs('logs/qos_probe_final_q2', exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'===== {mode} s{seed} =====\n')
        log.flush()
        cmd = [sys.executable, 'probe_model.py', mode, '60', str(seed)]
        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, cwd=os.getcwd())
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 'TIMEOUT'
        print(f'[{time.strftime("%H:%M:%S")}] {mode} s{seed}: rc={rc} '
              f'{(time.time() - t0) / 60:.1f}min', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-mode', default=None)
    ap.add_argument('--start-seed', type=int, default=None)
    args = ap.parse_args()
    started = False
    for seed in SEEDS:
        for mode in MODES:
            if args.start_mode is not None and not started:
                if mode != args.start_mode:
                    continue
                if args.start_seed is not None and seed != args.start_seed:
                    continue
                started = True
            run_one(mode, seed, 'logs/qos_probe_chain_final_q2.log')
    print('ALL QOS PROBES DONE', flush=True)


if __name__ == '__main__':
    main()
