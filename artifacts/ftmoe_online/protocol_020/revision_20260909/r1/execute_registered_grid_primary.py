"""Execute the frozen R1 grid sequentially; never repair or retune a run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path("F:/PreGANPlus-master")
OUT = ROOT / "artifacts/ftmoe_online/protocol_020/revision_20260909/r1/final_runs_20260909_1435"
PYTHON = "D:/Anaconda/envs/dynmoe/python.exe"
CHECKPOINT = "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt"
EXPECTED = {
    "run_ftmoe_protocol020_r1.py": "bd3c0f72d8cef4a763a995830277758ebd6b83dfd9f55e797d60738e0453f8e8",
    "recovery/PreGANSrc/src/ftmoe_online_r1.py": "72ae878eebaf8a5abca76123c1127a9c1910b49c39dd60aa8d1d04b92acd80cb",
}
environment = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", FTMOE020_RAM_GUARD_GIB="3.0", PYTHONUNBUFFERED="1")


def execute(name, arguments):
    available = psutil.virtual_memory().available / 2**30
    if available < 3:
        raise RuntimeError("Available RAM below 3 GiB")
    stdout, stderr = OUT / (name + ".stdout.log"), OUT / (name + ".stderr.log")
    if stdout.exists() or stderr.exists():
        raise FileExistsError("Execution logs already exist: " + name)
    command = [PYTHON, "-u", *arguments]
    print(json.dumps({"started": name, "command": command, "available_gib": available}), flush=True)
    started = time.time()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=out, stderr=err)
    (OUT / (name + ".execution.json")).write_text(json.dumps({"command": command, "cwd": str(ROOT), "started_unix": started, "ended_unix": time.time(), "exitcode": result.returncode, "stdout": str(stdout), "stderr": str(stderr)}, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"finished": name, "exitcode": result.returncode}), flush=True)
    if result.returncode:
        print(stderr.read_text(encoding="utf8", errors="replace")[-5000:], flush=True)
        raise SystemExit(result.returncode)


for source, expected in EXPECTED.items():
    if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Frozen source changed: " + source)
for seed in (500, 501):
    for method in ("A", "C-legacy", "C-residual-off", "C-residual-on"):
        name = "%s_dev%d_m1" % (method, seed)
        folder = OUT / name
        if (folder / "summary.json").is_file():
            summary = json.loads((folder / "summary.json").read_text(encoding="utf8"))
            config = summary["configuration"]
            if summary["status"] != "complete" or config["method"] != method or config["replay_seed"] != seed or config["code_sha256"]["run_ftmoe_protocol020_r1.py"] != EXPECTED["run_ftmoe_protocol020_r1.py"]:
                raise RuntimeError("Existing run mismatch: " + name)
            print(json.dumps({"retained_completed": name}), flush=True)
            continue
        if folder.exists():
            raise FileExistsError("Incomplete run must be reviewed: " + str(folder))
        execute(name, ["run_ftmoe_protocol020_r1.py", "--method", method, "--model-seed", "1", "--replay-seed", str(seed), "--checkpoint-path", CHECKPOINT, "--stream", "artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed%d_steps2000" % seed, "--output", str(folder)])
execute("analysis_final", ["analyze_ftmoe_protocol020_r1.py", "--runs", str(OUT), "--output", str(OUT / "r1_analysis.json"), "--markdown", str(OUT / "r1_analysis.md")])
execute("verification_final", ["verify_ftmoe_protocol020_r1_results.py", "--runs", str(OUT), "--output", str(OUT / "verification_results.json")])
print("REGISTERED_GRID_AND_CHECKS_COMPLETED", flush=True)
