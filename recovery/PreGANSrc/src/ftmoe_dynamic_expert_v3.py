"""Protocol 020 S8 — dynamic expert v3 trigger logic (pure functions).

Plan §21-§24: expert-birth triggering is DECOUPLED from routing eligibility
(unmatched == no eligible expert).  Two signals are combined:

1. routing novelty  — per-window routing entropy / top-2 margin vs a
   stationary baseline (registered from dev-stationary data, never from
   D-C results); novelty_ema >= novelty_threshold (e.g. stationary p95).
2. matured error    — detection/classification loss EMA fed by the session
   after labels mature (loss_ema >= 1.25 * stationary baseline).

Birth gate (plan §23, values registered before S8):
    candidate samples >= 128
    routing novelty EMA  >= novelty threshold (for >= 3 consecutive windows)
    loss ratio           >= 1.25
    expert_count         < 8

Everything here is pure (dict in / dict out) so it can be unit-tested without
model weights; the container integration (OnlineEAGateV3, candidate buffer,
shadow experts) lives in the same module once S7 gates pass.
"""
from collections import deque

V3_CANDIDATE_SAMPLES = 128
V3_CONSECUTIVE_WINDOWS = 3
V3_LOSS_RATIO_MIN = 1.25
V3_EXPERT_MAX = 8
V3_NOVELTY_EMA_ALPHA = 0.1
V3_LOSS_EMA_ALPHA = 0.1


def update_ema(ema, value, alpha=V3_NOVELTY_EMA_ALPHA):
    return alpha * value + (1.0 - alpha) * ema


class V3TriggerState:
    """Stateless-ish accumulator for the v3 birth gate."""

    def __init__(self):
        self.novelty_ema = 0.0
        self.loss_ema = 0.0
        self.loss_baseline = None          # stationary baseline, registered
        self.novelty_threshold = None      # stationary p95, registered
        self.novelty_windows_above = 0
        self.candidate_samples = 0
        self.history = deque(maxlen=8)

    def register_baselines(self, novelty_threshold, loss_baseline):
        self.novelty_threshold = float(novelty_threshold)
        self.loss_baseline = float(loss_baseline)

    def observe_window(self, novelty_value, samples, matured_loss=None,
                       loss_alpha=V3_LOSS_EMA_ALPHA):
        """novelty_value: per-window mean routing novelty (entropy or margin
        deficit).  matured_loss: mean per-sample matured loss of the window
        (None until labels mature)."""
        if self.novelty_threshold is None:
            raise ValueError("v3 trigger baselines must be registered first")
        self.novelty_ema = update_ema(self.novelty_ema, novelty_value)
        above = self.novelty_ema >= self.novelty_threshold
        self.novelty_windows_above = (self.novelty_windows_above + 1) if above else 0
        self.candidate_samples += int(samples)
        record = {"novelty_ema": self.novelty_ema,
                  "novelty_above": above,
                  "candidate_samples": self.candidate_samples}
        if matured_loss is not None:
            self.loss_ema = update_ema(self.loss_ema, matured_loss,
                                       alpha=loss_alpha)
            record["loss_ema"] = self.loss_ema
            record["loss_ratio"] = self.loss_ema / max(self.loss_baseline, 1e-9)
        self.history.append(record)
        return dict(record)

    def birth_gate(self, expert_count):
        """Plan §23 gate (all arms AND-ed)."""
        if self.novelty_threshold is None or self.loss_baseline is None:
            raise ValueError("v3 trigger baselines must be registered first")
        return {
            "samples_ok": self.candidate_samples >= V3_CANDIDATE_SAMPLES,
            "novelty_ok": self.novelty_windows_above >= V3_CONSECUTIVE_WINDOWS,
            "loss_ok": (self.loss_ema / max(self.loss_baseline, 1e-9)
                        >= V3_LOSS_RATIO_MIN),
            "expert_capacity_ok": expert_count < V3_EXPERT_MAX,
        }

    def ready(self, expert_count):
        checks = self.birth_gate(expert_count)
        return all(checks.values()), checks

    def state(self):
        return {"novelty_ema": self.novelty_ema, "loss_ema": self.loss_ema,
                "loss_baseline": self.loss_baseline,
                "novelty_threshold": self.novelty_threshold,
                "novelty_windows_above": self.novelty_windows_above,
                "candidate_samples": self.candidate_samples,
                "history": list(self.history)}

    def restore(self, state):
        self.novelty_ema = float(state["novelty_ema"])
        self.loss_ema = float(state["loss_ema"])
        self.loss_baseline = state.get("loss_baseline")
        self.novelty_threshold = state.get("novelty_threshold")
        self.novelty_windows_above = int(state["novelty_windows_above"])
        self.candidate_samples = int(state["candidate_samples"])
        self.history = deque(state.get("history", []), maxlen=8)
