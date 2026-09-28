"""Validate the Protocol-033 plan only; does not generate or qualify a dataset."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/ftmoe_online/protocol_033/scenario_registration.json"


def validate(reg):
    expected = [("F0", "baseline", 300),
                ("U_first", "U", 1000), ("V_first", "V", 1000),
                ("W_long", "W", 1000)]
    for index, service in enumerate(("U", "V", "U", "V", "U", "V")):
        expected.append(("%s_rec%d" % (service, index // 2 + 1), service, 128))
        if index < 5:
            expected.append(("W_gap%d" % (index + 1), "W", 380))
    assert reg["protocol"] == "033" and reg["plan_revision"] == 1
    assert reg["scenario_id"] == "protocol033_rare_recurrence_5969_v1"
    assert reg["data_revision"] == "protocol033_data_revision_001"
    assert reg["replay_seed"] == 700 and reg["model_seed"] == 1
    assert len(reg["timeline"]) == len(expected)
    cursor = 0
    recurrence = []
    for row, (name, service, length) in zip(reg["timeline"], expected):
        assert (row["name"], row["service"], row["length"]) == (name, service, length)
        assert (row["start"], row["end"]) == (cursor, cursor + length)
        cursor += length
        if "_rec" in name:
            recurrence.append(name)
    assert cursor == reg["scored_intervals"] == 5968
    assert reg["guard_intervals"] == 1 and reg["total_intervals"] == 5969
    assert reg["evaluation"]["primary_windows"] == recurrence
    assert reg["evaluation"]["primary_window_length"] == 128
    assert reg["evaluation"]["W_block_names"] == [
        row["name"] for row in reg["timeline"] if row["service"] == "W"]
    gen = reg["generation"]
    assert gen["event_probability"] == 0.30
    assert gen["chunk_intervals"] == gen["segment_max_intervals"] == 200
    quotient, remainder = divmod(reg["total_intervals"], gen["chunk_intervals"])
    assert (quotient, remainder) == (29, 169)
    assert gen["expected_chunk_count"] == quotient + 1
    assert gen["final_chunk_intervals"] == remainder
    birth = reg["D_capacity_and_birth"]
    assert birth["birth_start_matured"] == 600
    assert birth["birth_every_matured"] == 1000
    assert birth["candidate_train_horizon_intervals"] == 128
    assert birth["candidate_validation_intervals"] == 32
    first_due = [600, 1600, 2600]
    # The margin includes t+2 maturity and eight crossfade predictions.
    # This checks opportunity, never asserts candidate acceptance.
    for phase, due in zip(reg["timeline"][1:4], first_due):
        assert phase["start"] <= due
        assert due + 128 + 32 + 2 + 8 < phase["end"]
    train = reg["training"]
    assert train["online_update_every_scored_intervals"] == 16
    assert train["replay_matured_intervals"] == 64
    assert all(row["length"] > 64 for row in reg["timeline"] if row["name"].startswith("W_gap"))
    assert reg["causality"]["target"] == "raw_next_fault"
    assert reg["causality"]["target_maturity"] == "t_plus_2"
    assert reg["evaluation"]["development_reference"]["required_valid_windows"] == 6
    return {
        "protocol": "033", "plan_validation_passed": True,
        "scored_intervals": cursor, "guard_rows": 1, "total_rows": cursor + 1,
        "recurrence_windows": recurrence,
        "chunk_count": quotient + 1, "last_chunk_rows": remainder,
        "first_three_birth_due_matured": first_due,
        "actual_dataset_generated": False, "actual_dataset_qualified": False,
        "RAM_fit_verified": False, "models_run": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text(encoding="utf-8"))
    print(json.dumps(validate(reg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
