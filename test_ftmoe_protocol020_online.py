"""Protocol 020 S7 — online rare-event sampling tests (plan Test 10 etc.).

- Test 10 (plan): with CPU/RAM/Disk events present in the recent buffer, one
  update draw must obtain all three event classes at the registered quotas
  (4 each) plus 12 uniform intervals (24 total).
- Exposure caps: uniform intervals stop being drawn at 3; rare-event
  intervals stop at 5 (independent caps, plan §18.4).
- Determinism: the same rng state reproduces the same draw.

Run from the repository root:
    python test_ftmoe_protocol020_online.py
"""
import random
import unittest

from recovery.PreGANSrc.src.ftmoe_online_s7 import (
    stratified_draw, RECENT_UNIFORM_PER_UPDATE, EVENT_PER_UPDATE,
    MAX_EXPOSURE, MAX_RARE_EXPOSURE,
)

NORMAL = 80
CPU = 20
RAM = 20
DISK = 20
TOTAL = NORMAL + CPU + RAM + DISK


def tags_of(index):
    if index < NORMAL:
        return set()
    if index < NORMAL + CPU:
        return {"cpu"}
    if index < NORMAL + CPU + RAM:
        return {"ram"}
    return {"disk"}


def fresh_state():
    return (list(range(TOTAL)), tags_of, random.Random(7), {}, {})


class StratifiedDrawTests(unittest.TestCase):
    def test_10_all_three_event_classes_drawn(self):
        pool, tags_of, rng, exposure, rare = fresh_state()
        drawn = stratified_draw(pool, tags_of, rng, exposure, rare,
                                total_target=24)
        event_tags = [tags_of(i) for i in drawn]
        cpu = sum(1 for tags in event_tags if "cpu" in tags)
        ram = sum(1 for tags in event_tags if "ram" in tags)
        disk = sum(1 for tags in event_tags if "disk" in tags)
        self.assertEqual(len(drawn), 24)
        # event quotas are guaranteed slots; uniform draws may add more
        self.assertGreaterEqual(cpu, EVENT_PER_UPDATE, "CPU-event quota unmet")
        self.assertGreaterEqual(ram, EVENT_PER_UPDATE, "RAM-event quota unmet")
        self.assertGreaterEqual(disk, EVENT_PER_UPDATE, "Disk-event quota unmet")
        self.assertGreaterEqual(len(drawn) - (cpu + ram + disk), 6)

    def test_exposure_caps_uniform_3_rare_5(self):
        pool, tags_of, rng, exposure, rare = fresh_state()
        for _ in range(40):
            drawn = stratified_draw(pool, tags_of, rng, exposure, rare,
                                    total_target=24)
            self.assertLessEqual(len(drawn), 24)
        self.assertLessEqual(max(exposure.values()), MAX_EXPOSURE)
        self.assertLessEqual(max(rare.values()), MAX_RARE_EXPOSURE)
        # every event interval reaches its independent rare cap (5), uniform
        # caps never exceeded
        rare_used = [i for i in range(NORMAL, TOTAL)
                     if rare.get(i, 0) == MAX_RARE_EXPOSURE]
        self.assertEqual(len(rare_used), CPU + RAM + DISK)

    def test_deterministic(self):
        pool, tags_of, rng, exposure, rare = fresh_state()
        first = stratified_draw(pool, tags_of, rng, exposure, rare,
                                total_target=24)
        pool2, _, rng2, exp2, rare2 = fresh_state()
        second = stratified_draw(pool2, tags_of, rng2, exp2, rare2,
                                 total_target=24)
        self.assertEqual(first, second)

    def test_shortfall_backfilled_without_exceeding_target(self):
        # Only 10 normal intervals -> draw falls back to those only.
        pool = list(range(10))
        rng = random.Random(3)
        drawn = stratified_draw(pool, lambda i: set(), rng, {}, {},
                                total_target=24)
        self.assertEqual(len(drawn), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
