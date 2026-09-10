from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(r"F:\PreGANPlus-master")
PYTHON = Path(r"D:\Anaconda\envs\dynmoe\python.exe")
FORMAL = REPO / "artifacts/ftmoe_online/protocol_020/revision_20260909/r1/final_runs_20260909_1435"
CHECKPOINT = REPO / "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt"

RUNS = (
    ("C-residual-on", 500),
    ("A", 501),
    ("C-legacy", 501),
    ("C-residual-off", 501),
    ("C-residual-on", 501),
)


def main() -> int:
    ram_probe = subprocess.run(
        [str(PYTHON), "-c", "import psutil; print(psutil.virtual_memory().available / 1024**3)"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ram_probe.returncode != 0:
        print(ram_probe.stdout, end="")
        print(ram_probe.stderr, end="", file=sys.stderr)
        return ram_probe.returncode
    ram_gib = float(ram_probe.stdout.strip())
    print(f"RAM_AVAILABLE_GIB={ram_gib:.3f}", flush=True)
    if ram_gib < 3.0:
        print(f"Available RAM below 3 GiB: {ram_gib}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.update(
        FTMOE020_RAM_GUARD_GIB="3.0",
        OMP_NUM_THREADS="3",
        MKL_NUM_THREADS="3",
        OPENBLAS_NUM_THREADS="3",
    )

    for method, replay_seed in RUNS:
        stem = f"{method}_dev{replay_seed}_m1"
        output = FORMAL / stem
        log = FORMAL / f"{stem}.stdout_stderr.log"
        if output.exists():
            print(f"Refusing to overwrite existing output directory: {output}", file=sys.stderr)
            return 3
        if log.exists():
            print(f"Refusing to overwrite existing log: {log}", file=sys.stderr)
            return 3

        cmd = [
            str(PYTHON),
            "run_ftmoe_protocol020_r1.py",
            "--method",
            method,
            "--model-seed",
            "1",
            "--replay-seed",
            str(replay_seed),
            "--checkpoint-path",
            "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt",
            "--stream",
            f"artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed{replay_seed}_steps2000",
            "--output",
            str(output),
        ]
        print(f"LAUNCH method={method} replay_seed={replay_seed} output={output} log={log}", flush=True)
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                cmd,
                cwd=REPO,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\nEXITCODE={completed.returncode}\n")
        print(
            f"DONE method={method} replay_seed={replay_seed} exitcode={completed.returncode} output={output} log={log}",
            flush=True,
        )
        if completed.returncode != 0:
            print(f"Run failed: {stem}; see {log}", file=sys.stderr)
            return completed.returncode
        summary = output / "summary.json"
        if not summary.exists():
            print(f"Run completed without summary.json: {summary}", file=sys.stderr)
            return 4

    print("ALL_FIVE_REMAINING_RUNS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
