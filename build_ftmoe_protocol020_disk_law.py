"""Protocol 020 P17-b2 — build protocol-020 disk law variants (age-profile surgery).

Derives protocol-020 disk laws from the 016 training law
(artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json, 10 states, values
0..9000).  Two data-only modifications (registered, problem log P18-b2):

1. initial_probability: mass on states with value > 4000 is zeroed and the
   remainder renormalized -> containers are born small, so mid/low capacity
   scales no longer produce a deployment-rejection wall.
2. transition rows: upward-biased growth so that containers reach large
   occupancy within their lifetime (self-stay of the 016 law is 0.78-0.90,
   which keeps almost every container small).

Variants (growth = probability mass moved up one/two/three states):
    v1 (L1, moderate): stay .30, up1 .40, up2 .20, up3 .10
    v2 (L2, strong):   stay .15, up1 .45, up2 .30, up3 .10
Top state 9 stays absorbing-ish with the leftover mass on itself.

Output:
    artifacts/ftmoe_online/protocol_020/disk_law_p20_<variant>.json
(same schema keys as the 016 law plus provenance fields)

Usage:
    python build_ftmoe_protocol020_disk_law.py --variant v1 [--variant v2]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LAW_016 = ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json"
OUT_DIR = ROOT / "artifacts/ftmoe_online/protocol_020"

GROWTH = {
    "v1": {"stay": 0.30, "up1": 0.40, "up2": 0.20, "up3": 0.10},
    "v2": {"stay": 0.15, "up1": 0.45, "up2": 0.30, "up3": 0.10},
}
BIRTH_CAP_VALUE = 4000.0  # states above this are removed from the birth law


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build(variant: str, out: Path):
    growth = GROWTH[variant]
    law = json.loads(LAW_016.read_text(encoding="utf8"))
    values = np.asarray(law["values"], dtype=np.float64)
    n = len(values)
    initial = np.asarray(law["initial_probability"], dtype=np.float64)
    mask = values <= BIRTH_CAP_VALUE
    initial_new = np.where(mask, initial, 0.0)
    total = initial_new.sum()
    if total <= 0:
        raise ValueError("birth truncation removed all probability mass")
    initial_new = initial_new / total
    transition_new = np.zeros((n, n))
    for i in range(n):
        stay = growth["stay"]
        transition_new[i, i] = stay
        for offset, mass in ((1, growth["up1"]), (2, growth["up2"]),
                             (3, growth["up3"])):
            j = min(i + offset, n - 1)
            transition_new[i, j] += mass
        row_sum = transition_new[i].sum()
        transition_new[i] /= max(row_sum, 1e-12)
        # normalization drift (top state receives overflow already) is fine
    # keep the same keys as the 016 law for schema compatibility
    derived = dict(law)
    derived.update({
        "values": law["values"],
        "transition": transition_new.tolist(),
        "initial_probability": initial_new.tolist(),
        "p20_law_variant": variant,
        "p20_growth": growth,
        "p20_birth_cap_value": BIRTH_CAP_VALUE,
        "derived_from": "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json",
        "derived_from_sha256": sha(LAW_016),
        "registration": "protocol-020 P17-b2 disk age-profile surgery (problem log P18); "
                        "data-only, no model involvement",
    })
    out.write_text(json.dumps(derived, indent=1) + "\n", encoding="utf8")
    print(json.dumps({"variant": variant, "out": str(out),
                      "transition_diag": [round(transition_new[i, i], 3)
                                          for i in range(n)],
                      "init_low_mass": round(float(initial_new[:3].sum()), 3),
                      "sha256": sha(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(GROWTH), action="append",
                        required=True)
    args = parser.parse_args()
    for variant in args.variant:
        build(variant, OUT_DIR / f"disk_law_p20_{variant}.json")


if __name__ == "__main__":
    main()
