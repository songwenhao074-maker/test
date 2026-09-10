"""Record the primary review without inferring missing process exit codes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import psutil

root = Path("F:/PreGANPlus-master")
folder = root / "artifacts/ftmoe_online/protocol_020/revision_20260909/r1/final_runs_20260909_1435"
read = lambda p: json.loads(p.read_text(encoding="utf8"))
verification = read(folder / "verification_results.json")
analysis = read(folder / "r1_analysis.json")
assert verification["status"] == "PASS"
assert len(verification["checks"]) == 106
assert all(c["passed"] for c in verification["checks"])
assert analysis["status"] == "COMPLETE"
assert len(analysis["fairness"]["pairs"]) == 2
assert all(c["equal"] for pair in analysis["fairness"]["pairs"] for c in pair["checks"].values())
assert analysis["fairness"]["protective_attribution_eligible"]
files = ["r1_analysis.json", "r1_analysis.md", "verification_results.json", "inspection_primary.json", "parameter_counts.json"]
runs = []
for summary_path in sorted(folder.rglob("summary.json")):
    summary = read(summary_path)
    stem = summary_path.parent.name
    if stem == "C-residual-off_dev500_m1":
        exitcode = None
        exit_evidence = "not captured: PowerShell wrapper failed on stderr warning; appended EXITCODE=0 was inferred and is not an observed exit code"
    elif (folder / (stem + ".log.exitcode")).is_file():
        exitcode = int((folder / (stem + ".log.exitcode")).read_text().strip())
        exit_evidence = "captured process return code in .log.exitcode"
    else:
        log = (folder / (stem + ".stdout_stderr.log")).read_text(encoding="utf8")
        exitcode = int(re.findall(r"^EXITCODE=(-?\d+)$", log, flags=re.M)[-1])
        exit_evidence = "captured subprocess.run returncode appended by sequential Python coordinator"
    runs.append({"run": stem, "completed_summary": summary["status"] == "complete", "actual_process_exitcode": exitcode, "exitcode_evidence": exit_evidence, "completion_evidence": "summary status complete; final resume status/cursor checked by analyzer; independent 2000-step prediction checks passed"})
assert len(runs) == 8
processes = []
for process in psutil.process_iter(["name", "cmdline"]):
    if process.pid == os.getpid():
        continue
    name = (process.info["name"] or "").lower()
    if name.startswith("python") and any("run_ftmoe_protocol020_r1.py" in arg for arg in (process.info["cmdline"] or [])):
        processes.append(process.pid)
result = {
    "reviewed_at": datetime.datetime.now().astimezone().isoformat(),
    "python_version": sys.version,
    "experiment_complete": True,
    "performance_goal_achieved": False,
    "dynamic_D_evaluated_in_this_experiment": False,
    "data_scope": "two previously used development streams; no independent confirmation",
    "test_count": 16,
    "test_evidence": "../tests/20260909_142841",
    "independent_result_checks_passed": len(verification["checks"]),
    "paired_training_checks": analysis["fairness"],
    "runs": runs,
    "remaining_R1_runner_pids": processes,
    "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("recovery/PreGANSrc/src/ftmoe_online_r1.py", "run_ftmoe_protocol020_r1.py", "analyze_ftmoe_protocol020_r1.py", "verify_ftmoe_protocol020_r1_results.py", "test_ftmoe_protocol020_r1.py")},
    "report_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files},
    "analysis_corrections": ["Python 3.8-compatible dictionary merge", "Markdown column count and ratio labels", "append both evaluated fairness pairs before computing eligibility"],
    "execution_note": "First formal off/dev500 numeric exit status is unknown; original inferred marker retained for traceability. All final artifacts passed independent acceptance. Later five runs used direct subprocess returncode capture.",
}
(folder / "acceptance_primary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
print(json.dumps({"experiment_complete": True, "performance_goal_achieved": False, "checks": 106, "paired_checks": 12, "remaining_runner_pids": processes}))
