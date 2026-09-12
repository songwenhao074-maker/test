"""Protocol 022 — real-subprocess runner for data collection.

Plan §21 explicitly forbids the P20/P21 pattern where a PowerShell wrapper turned
a torch stderr warning into an apparent ``exitcode 1`` (and once into a missing
exit code).  Every formal P22 run therefore goes through this module, which
captures the child's **real** ``subprocess`` return code and writes it next to
the run:

    run_ftmoe_protocol022_pilot.py --probability 0.15
    run_ftmoe_protocol022_pilot.py --all

Per-run sidecar (same basename as ``--exitcode-file`` that P20 used, but written
by the parent process, not inferred):

    <output dir>/run_provenance.json   command, git sha, python/torch versions,
                                       seed, start/end, exit code, elapsed,
                                       peak RSS, stdout/stderr tails
"""
import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ftmoe_online/protocol_022/pilot_streams"
COLLECTOR = ROOT / "prepare_ftmoe_protocol022_unseen.py"
REGISTERED_PROBABILITIES = (0.15, 0.25, 0.35)
REGISTERED_SEED = 600
REGISTERED_STEPS = 1200
PYTHON = sys.executable


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git_head():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return out.stdout.decode("utf8", "replace").strip()


def versions():
    import numpy
    info = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except Exception as exc:                      # pragma: no cover
        info["torch"] = "unavailable: %s" % exc
    info["cwd"] = str(ROOT)
    return info


def run_one(probability, steps=REGISTERED_STEPS, smoke=False, output_root=None):
    output_root = Path(output_root or OUT)
    tag = "p%03d_seed%d_steps%d" % (round(probability * 100), REGISTERED_SEED,
                                    steps)
    directory = (output_root / "_smoke" if smoke else output_root) / tag
    if directory.exists():
        raise FileExistsError("refusing to overwrite an existing run: %s" % directory)
    command = [PYTHON, str(COLLECTOR), "--probability", str(probability),
               "--steps", str(steps)]
    if smoke:
        command.append("--smoke")
    started = _dt.datetime.now().isoformat(timespec="seconds")
    clock = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf8"
    completed = subprocess.run(command, cwd=str(ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - clock
    stdout = completed.stdout.decode("utf8", "replace")
    stderr = completed.stderr.decode("utf8", "replace")
    provenance = {
        "protocol": "022",
        "command": command,
        "exit_code": completed.returncode,
        "exit_code_source": "subprocess.CompletedProcess.returncode (real child code)",
        "probability": probability,
        "steps": steps,
        "smoke": bool(smoke),
        "seed": REGISTERED_SEED,
        "output_dir": str(directory.relative_to(ROOT)).replace("\\", "/"),
        "started_at": started,
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "git_head": git_head(),
        "collector_sha256": sha(COLLECTOR),
        "core_sha256": sha(ROOT / "ftmoe_protocol022_core.py"),
        "generator_sha256": sha(ROOT / "simulator/workload/BitbrainWorkloadProtocol022.py"),
        "environment": versions(),
        "stdout_tail": stdout.splitlines()[-5:],
        "stderr_tail": stderr.splitlines()[-10:],
        "stderr_note": ("torch/numpy warnings on stderr are recorded verbatim and "
                        "are NOT interpreted as failure; the exit code above is "
                        "the child's real return code"),
    }
    if directory.is_dir():
        directory.joinpath("run_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability", type=float, default=None)
    parser.add_argument("--all", action="store_true",
                        help="run the three registered candidates in order")
    parser.add_argument("--steps", type=int, default=REGISTERED_STEPS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.all:
        probabilities = list(REGISTERED_PROBABILITIES)
    elif args.probability is not None:
        probabilities = [args.probability]
    else:
        raise SystemExit("pass --probability or --all")
    results = []
    for probability in probabilities:
        provenance = run_one(probability, steps=args.steps, smoke=args.smoke,
                             output_root=args.output_root)
        results.append(provenance)
        print(json.dumps({"probability": probability,
                          "exit_code": provenance["exit_code"],
                          "elapsed_seconds": round(provenance["elapsed_seconds"], 1),
                          "output_dir": provenance["output_dir"]},
                         ensure_ascii=False), flush=True)
    failed = [r for r in results if r["exit_code"] != 0]
    print(json.dumps({"runs": len(results), "failed": len(failed),
                      "exit_codes": [r["exit_code"] for r in results]},
                     ensure_ascii=False))
    if failed:
        # the per-run failure.json / generation.log hold the diagnosis
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
