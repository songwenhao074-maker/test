"""Protocol 022 P22-S1 — register the ``cascade_v2`` regime before any model work.

The registry is DERIVED from the generator module, so the registered mechanism
can never drift from the implemented one: ``CASCADE_V2`` in
``simulator/workload/BitbrainWorkloadProtocol022.py`` is the single source of
truth and this script records its hash next to the mechanism.

Claim levels (plan §2, §7.2):

    U1 source-disjoint from P20 S6   : PASS   (evidence: exclusion_registry.json;
                                                re-verified against vm_split.json)
    U2 parameter-combo unseen        : PASS   (the cascade envelope combination
                                                was never generated offline)
    U3 generator mechanism unseen    : PASS   (evidence: exposure_ledger.json
                                                from the P21 audit of 1003 files)
    U4 task-level distributional     : null   (filled by
                                                analyze_ftmoe_protocol022_unseen.py)

U1 limitation, reported rather than hidden: source-unseen relative to the FULL
historical chain (P014's upstream VM selection is not registered anywhere)
remains UNVERIFIED.  The primary claim of protocol 022 is therefore
mechanism-level U3 plus task-level distributional U4-v2, never absolute
source-unseen.

Usage:
    python register_ftmoe_protocol022_unseen.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts/ftmoe_online/protocol_022"
REGISTRY = OUT_DIR / "unseen_registry/cascade_v2.json"
P21_AUDIT = ROOT / "artifacts/ftmoe_online/protocol_021/offline_coverage_audit"
GENERATOR = ROOT / "simulator/workload/BitbrainWorkloadProtocol022.py"
COLLECTOR = ROOT / "prepare_ftmoe_protocol022_unseen.py"
CORE = ROOT / "ftmoe_protocol022_core.py"
SPLIT_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_sha(payload):
    body = {k: v for k, v in payload.items() if k != "registry_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf8")).hexdigest()


def main():
    from simulator.workload.BitbrainWorkloadProtocol022 import (
        CASCADE_PROBABILITY_CANDIDATES, CASCADE_V2, MATCHED_CONTROLS,
        MECHANISM_SEED_CONFIRM, MECHANISM_SEED_DEV, Protocol022CascadeBWGD2)

    ledger_path = P21_AUDIT / "exposure_ledger.json"
    exclusion_path = P21_AUDIT / "exclusion_registry.json"
    split = json.loads(SPLIT_PATH.read_text(encoding="utf8"))
    cohort_sizes = {name: len(ids) for name, ids in split["cohorts"].items()}
    online = set(split["cohorts"]["online"])
    disjoint = {
        "p20_s6_train": not (online & set(split["cohorts"]["train"])),
        "p20_dev": not (online & set(split["cohorts"]["dev"])),
    }
    mechanism = {k: v for k, v in CASCADE_V2.items() if k != "forbidden_inputs"}
    ledger = json.loads(ledger_path.read_text(encoding="utf8")) \
        if ledger_path.is_file() else {}
    sources_audited = sum(len(stage.get("sources", []))
                          for stage in ledger.get("learning_chain", []))
    registry = {
        "protocol": "022",
        "regime_id": CASCADE_V2["regime_id"],
        "supersedes": "cascade_v1 (P21 STOP-A: host-aggregate U4 failed as an instrument)",
        "registered_at": "2026-09-10",
        "registered_before_model_runs": True,
        "round1_scope": "S0-S4 only (audit, data-only pilot, U4-v2, learnability-v2)",
        "claim_level": {
            "U0_new_seed_or_replay": True,
            "U1_source_disjoint_p20_s6": all(disjoint.values()),
            "U2_parameter_combo_unseen": True,
            "U3_generator_mechanism_unseen": True,
            "U4_distributional_novelty_task_level": None,
            "U4_status": "pending analyze_ftmoe_protocol022_unseen.py",
        },
        "claim_level_notes": {
            "U0_U1_not_sufficient": "A new seed/replay and a disjoint VM cohort are not, by themselves, evidence of a new regime (P21 §4).",
            "U3_evidence": "artifacts/ftmoe_online/protocol_021/offline_coverage_audit/exposure_ledger.json -> mechanism_exclusion.temporal_cpu_ram_disk_cascade: searched 1003 files of the 014/019/020 chain; cascade/cpu_to_ram/delayed/mechanism_seed/hysteresis/feedback = 0 matches.",
            "U1_evidence": "The pilot cohort p20_online (94 VMs) is disjoint from the P20 S6 train (277) and dev (86) cohorts; re-verified here from vm_split.json, not quoted.",
            "U1_limitation": "source-unseen relative to the full historical chain (P014 upstream VM selection is not registered anywhere) remains UNVERIFIED.",
            "U2_note": "The envelope (burst 3-5 intervals at >= host CPU capacity from age 1; RAM target ramp at +4; capped self-cleaning retained disk term at +8) does not exist anywhere in the offline chain, so the parameter combination is unseen by construction.",
            "what_changed_from_v1": [
                "new mechanism seed 22022 (confirm 22023 unused in round 1)",
                "audit scale host aggregate -> task (creation_id) / event, because the P21 U4 gate measured co-residency rather than the mechanism (P21-09)",
                "matched negative controls M0/M1/M2 registered in the audit layer",
                "the physical envelope is UNCHANGED so the two rounds stay comparable (plan §5.1)",
            ],
            "harking_guard": "probability 0.35 retains its high-load stress status: if it again exceeds the 40% migration-rejection ceiling it stays FAIL and is not selected (plan §5.2).",
        },
        "mechanism": mechanism,
        "onset_definition": Protocol022CascadeBWGD2.onset_definition(),
        "matched_controls": dict(MATCHED_CONTROLS),
        "observable_history": CASCADE_V2["observable_history"],
        "forbidden_inputs": list(CASCADE_V2["forbidden_inputs"]),
        "development_only": True,
        "cascade_probability_candidates": list(CASCADE_PROBABILITY_CANDIDATES),
        "mechanism_seeds": {
            "dev": MECHANISM_SEED_DEV,
            "confirm": MECHANISM_SEED_CONFIRM,
            "confirm_used_in_round1": False,
        },
        "pilot_registration": {
            "cohort": "online",
            "cohort_source": "artifacts/ftmoe_online/protocol_020/vm_split.json cohorts.online",
            "cohort_size": cohort_sizes.get("online"),
            "cohort_sizes_all": cohort_sizes,
            "disjointness_reverified": disjoint,
            "replay_seed": 600,
            "scored_intervals": 1200,
            "guard_intervals": 1,
            "familiar_environment": "Protocol-020 baseline phase (capacity scales 1.0/1.0/0.9, adapter ram_upper=1400)",
            "data_only": True,
        },
        "generator": {"path": str(GENERATOR.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": sha(GENERATOR)},
        "collector": {"path": str(COLLECTOR.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": sha(COLLECTOR)},
        "audit_instrument": {
            "path": str(CORE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha(CORE),
            "unit": "creation_id + CPU onset event",
        },
        "audit_evidence": {
            "exposure_ledger": {
                "path": str(ledger_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha(ledger_path),
                "sources_audited": sources_audited,
                "provenance": "produced by protocol 021; reused without modification",
            },
            "exclusion_registry": {
                "path": str(exclusion_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha(exclusion_path),
                "provenance": "produced by protocol 021; reused without modification",
            },
        },
        "construct_validity": {
            "statement": "controlled unseen temporal resource-demand regime",
            "not_claimed": "that real industrial faults must follow this law",
            "simulator_control_s0": "same admission-safe age0 with CPU burst only, no RAM/Disk cascade (plan §19)",
            "simulator_control_s1": "same CPU/RAM/Disk marginal envelope with random lag/order (plan §19)",
            "reason_admission_shaping": "Simulator.getPlacementPossible() evaluates feasibility with the demand at the admission interval, so a flat over-capacity burst at age 0 is rejected before it can overload anything (P21-01: 93.7% deployment rejection).",
        },
        "not_a_claim": [
            "The registry itself proves nothing about performance; it records what the data layer was allowed to be.",
            "U4-v2 is null until the task-level audit runs; a generator difference alone is not distributional novelty.",
            "A new mechanism seed is not evidence of novelty.",
        ],
    }
    registry["registry_sha256"] = canonical_sha(registry)

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf8")
    print(json.dumps({"registered": str(REGISTRY.relative_to(ROOT)).replace("\\", "/"),
                      "regime_id": registry["regime_id"],
                      "generator_sha256": registry["generator"]["sha256"],
                      "registry_sha256": registry["registry_sha256"],
                      "u1_disjoint": registry["claim_level"]["U1_source_disjoint_p20_s6"],
                      "u3": registry["claim_level"]["U3_generator_mechanism_unseen"],
                      "u4": registry["claim_level"]["U4_distributional_novelty_task_level"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
