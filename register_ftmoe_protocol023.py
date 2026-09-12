"""Register the Protocol-023 bootstrap state (S0).

Writes the machine-readable registration for the round:
  artifacts/ftmoe_online/protocol_023/protocol.json
  artifacts/ftmoe_online/protocol_023/gate_status.json
  artifacts/ftmoe_online/protocol_023/source_sha256_initial.json

The registered values are frozen BEFORE the first formal stream is collected:
the plan's own hashes, the branch/parent identity, the three regime physics
tables, the thresholds each gate is evaluated with, and the round scope.  A gate
status is never written as PASS here — it starts as ``pending`` and is only
changed by the analyzer that actually measured it.

Usage:
    python register_ftmoe_protocol023.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/ftmoe_online/protocol_023"
PLAN = ROOT / "指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md"
PARENT_SHA = "6c5db86fde8afd10e3997d4ff0bf8016e2e87130"
BRANCH = "protocol-023"
PARENT_BRANCH = "protocol-022"

SOURCE_FILES = (
    "simulator/workload/BitbrainWorkloadProtocol023.py",
    "simulator/workload/BitbrainWorkloadProtocol022.py",
    "simulator/workload/BitbrainWorkloadProtocol021.py",
    "simulator/workload/BitbrainWorkloadProtocol020.py",
    "ftmoe_protocol023_core.py",
    "ftmoe_protocol022_core.py",
    "prepare_ftmoe_protocol023_stream.py",
    "verify_ftmoe_protocol023_stream.py",
    "run_ftmoe_protocol023_s2.py",
    "analyze_ftmoe_protocol023_s2.py",
    "probe_ftmoe_protocol023_specialization.py",
    "probe_ftmoe_protocol023_gradient.py",
    "verify_ftmoe_protocol023_generator.py",
    "test_ftmoe_protocol023_core.py",
    "test_ftmoe_protocol023_regimes.py",
    "test_ftmoe_protocol023_s2.py",
    "test_ftmoe_protocol023_gradient.py",
    "stats/Stats.py",
)

DATA_GATE = {
    "anomaly_prevalence_min": 0.03,
    "anomaly_prevalence_max": 0.12,
    "deployment_rejection_max": 0.25,
    "migration_rejection_max": 0.40,
    "independent_fault_events_min_per_regime": 80,
    "valid_onset_followup_min_per_regime": 50,
    "worst_event_share_max": 0.10,
    "regime_prevalence_difference_max_pp": 4.0,
}

SPECIALIZATION_GATE = {
    "rule": "within-regime AP - mean cross-regime AP >= 0.05 for at least 2 "
            "of 3 regimes (plan §13)",
    "min_within_minus_cross_ap": 0.05,
    "min_regimes_passing": 2,
    "primary_horizon": 1,
    "horizons": [1, 4],
}

GRADIENT_GATE = {
    "rule": "at least one regime pair with mean residual-gradient cosine "
            "<= -0.05, or >= 30% of sampled cross-regime event pairs with "
            "cosine < 0 (plan §14)",
    "pair_mean_cosine_max": -0.05,
    "negative_pair_fraction_min": 0.30,
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(*args):
    try:
        out = subprocess.run(["git"] + list(args), cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return out.stdout.decode("utf-8", "replace").strip()
    except OSError:
        return ""


def regime_registration():
    from simulator.workload.BitbrainWorkloadProtocol023 import (
        INHERITED_CHAIN_REGIMES, MECHANISM_SEED_CONFIRM, MECHANISM_SEED_DEV,
        REGIME_IDS, REGIME_MECHANISM_SEEDS, REGISTERED_ONSET_TAU,
        PROVISIONAL_ONSET_TAU, REGIMES_V3, registered_family)
    family = registered_family()
    return {
        "family": "cascade_v3",
        "regime_ids": list(REGIME_IDS),
        "mechanism_seed_dev": MECHANISM_SEED_DEV,
        "mechanism_seed_confirm": MECHANISM_SEED_CONFIRM,
        "mechanism_seed_by_regime": dict(REGIME_MECHANISM_SEEDS),
        "onset_thresholds": dict(REGISTERED_ONSET_TAU),
        "onset_thresholds_provisional": dict(PROVISIONAL_ONSET_TAU),
        "onset_threshold_rule": ("a threshold must be strictly above the "
                                 "measured familiar per-task maximum of its "
                                 "resource and strictly below the regime's own "
                                 "floor; the RAM and disk thresholds are "
                                 "provisional and are reported as such"),
        "inherited_chain_regimes": list(INHERITED_CHAIN_REGIMES),
        "observability_rule": ("every registered lag < observable_history (12) "
                               "for every regime; the stricter longest-chain "
                               "rule applies only to the newly registered "
                               "orders (memory_first, io_first)"),
        "regimes": family,
        "physics_notes": {
            regime: REGIMES_V3[regime]["physical_story"] for regime in REGIME_IDS},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES
               if (ROOT / name).is_file()}
    missing = [name for name in SOURCE_FILES if name not in sources]
    registration = {
        "protocol": "023",
        "title": ("Multi-regime recurring drift: dynamic add/delete experts vs "
                  "fixed online fine-tuning"),
        "plan": "指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md",
        "plan_sha256": sha256(PLAN),
        "branch": BRANCH,
        "parent_branch": PARENT_BRANCH,
        "parent_sha": PARENT_SHA,
        "bootstrap_head": git("rev-parse", "HEAD"),
        "registered_before_collection": True,
        "round1_scope": {
            "allowed": ["S0 freeze P22", "S1 three regime generators + tests",
                        "S2 data-only streams + Data Gate / marginal matching",
                        "S3 cross-regime specialization probe",
                        "S3 gradient interference probe",
                        "S3 sequential forgetting probe"],
            "forbidden": ["implement or run D", "modify the FT-MoE backbone",
                          "retrain offline v4 / P19 / P20 / P22 checkpoints",
                          "redefine a gate after seeing its result",
                          "use phase/regime/mechanism ids as model inputs"],
        },
        "gates": [
            {"id": "H0", "name": "generator_registration_correctness",
             "stage": "S1",
             "rule": ("compute-first is byte-identical to cascade_v2; the "
                      "familiar transform is exact for every regime; every "
                      "registered lag is observable"),
             "status": "pending"},
            {"id": "H1", "name": "data_gate_and_marginal_matching",
             "stage": "S2", "rule": "plan §9 thresholds per regime",
             "status": "pending"},
            {"id": "H2", "name": "specialization_opportunity", "stage": "S3",
             "rule": SPECIALIZATION_GATE["rule"], "status": "pending"},
            {"id": "H3", "name": "gradient_interference", "stage": "S3",
             "rule": GRADIENT_GATE["rule"], "status": "pending"},
            {"id": "H4", "name": "sequential_forgetting", "stage": "S3",
             "rule": ("fixed C trained on regime R loses performance on the "
                      "earlier regime; recurrence requires re-adaptation"),
             "status": "pending"},
        ],
        "stop_table": {
            "STOP-MR": ("no specialization opportunity across the three "
                        "regimes -> do not implement D (plan §13)"),
            "STOP-GI": ("regime gradients are co-directional -> no continual "
                        "conflict -> do not implement D (plan §14)"),
            "STOP-DATA": ("Data Gate or marginal matching FAIL -> the scenes "
                          "are not matched, so no method comparison is "
                          "admissible"),
            "STOP-NOADAPT": ("fixed C adapts to every regime within its dwell "
                             "time -> the stability-plasticity conflict this "
                             "protocol is built on does not exist"),
        },
        "registered_timeline": {
            "development": {
                "tag": "dev_seed700_steps2880",
                "phases": [["F0_baseline", 300, "familiar"],
                           ["A1_compute", 420, "compute_first"],
                           ["B1_memory", 420, "memory_first"],
                           ["C1_io", 420, "io_first"],
                           ["F1_baseline", 240, "familiar"],
                           ["A2_recur", 360, "compute_first"],
                           ["C2_recur", 360, "io_first"],
                           ["B2_recur", 360, "memory_first"]],
                "scored_intervals": 2880,
                "cascade_task_probability": 0.25,
            },
            "single_regime": {
                "tag_template": "single_<regime>_seed700_steps1200",
                "phases": [["F0_baseline", 150, "familiar"],
                           ["<regime>_only", 1050, "<regime>"]],
                "scored_intervals": 1200,
                "purpose": ("clean per-regime cohorts for the marginal "
                            "matching gate and the cross-regime probe"),
            },
        },
        "registered_seeds": {
            "replay_seed_development": 700,
            "replay_seed_confirmation": [701, 702, 703],
            "mechanism_seed_dev": 22022,
            "mechanism_seed_confirm": 22023,
            "note": ("701+ are confirmation seeds and must not be used for "
                     "tuning; they are not used in round 1"),
        },
        "regime_registration": regime_registration(),
        "data_gate": DATA_GATE,
        "specialization_gate": SPECIALIZATION_GATE,
        "gradient_gate": GRADIENT_GATE,
        "frozen_checkpoint": {
            "path": "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt",
            "sha256_expected":
                "10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b",
            "policy": "read-only; never overwritten or retrained",
        },
        "resource_discipline": {
            "python": "3.8 (D:\\Anaconda\\envs\\dynmoe\\python.exe)",
            "single_process": True,
            "ram_guard_gib": 2.5,
            "disk_guard_gib": 20.0,
            "torch_threads": 3,
            "interop_threads": 1,
            "priority": "BelowNormal",
            "exit_code_source": "subprocess returncode, never a wrapper",
        },
        "honesty_rules": [
            "A gate is written pending until the analyzer that measured it "
            "writes its verdict.",
            "STOP outcomes are results and are reported with evidence.",
            "No threshold is redefined after seeing a result.",
            "The compute-first regime inherits cascade_v2 verbatim; any "
            "divergence is a defect, not a new regime.",
            "Phase, regime, mechanism and event identifiers are audit-only "
            "and may never reach a model input.",
        ],
        "source_sha256": sources,
        "source_files_missing_at_registration": missing,
    }
    (OUT / "protocol.json").write_text(
        json.dumps(registration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8")
    (OUT / "source_sha256_initial.json").write_text(
        json.dumps({"protocol": "023",
                    "captured_before_collection": True,
                    "head": registration["bootstrap_head"],
                    "n_sources": len(sources),
                    "sources": sources}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8")
    status = {
        "protocol": "023",
        "stage": "S0-registered",
        "registered_seeds": registration["registered_seeds"],
        "gates": [{"id": g["id"], "name": g["name"], "stage": g["stage"],
                   "status": g["status"], "evidence": None}
                  for g in registration["gates"]],
        "streams": {
            "development": {"tag": "dev_seed700_steps2880", "status": "pending"},
            "single_compute_first": {"tag": "single_compute_first_seed700_steps1200",
                                     "status": "pending"},
            "single_memory_first": {"tag": "single_memory_first_seed700_steps1200",
                                    "status": "pending"},
            "single_io_first": {"tag": "single_io_first_seed700_steps1200",
                                "status": "pending"},
        },
        "d_implemented": False,
        "note": ("round 1 covers S0-S3 only; D is not implemented and must not "
                 "be started before H2 and H3 are evaluated (plan §37/§38)"),
    }
    (OUT / "gate_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"written": [str((OUT / n).relative_to(ROOT)).replace("\\", "/")
                                  for n in ("protocol.json", "gate_status.json",
                                            "source_sha256_initial.json")],
                      "plan_sha256": registration["plan_sha256"],
                      "n_sources": len(sources),
                      "missing_sources": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
