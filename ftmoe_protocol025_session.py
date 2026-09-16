"""Protocol-025 prequential sessions for the preregistered comparators."""
from __future__ import annotations

import torch

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol024_session import Protocol024Session, NEXT_TARGET_MODE
from ftmoe_protocol024_v2c import V2CProtocol024Session, V2C_DEFAULT
from ftmoe_protocol025_model import Protocol025ResidualFTMoE


FIXED_COMPARATORS = {
    "C_fixed4": (4, None),
    "C_fixed5": (5, None),
    "C_fixed8_dense": (8, None),
    "C_fixed8_top5": (8, 5),
}


def _new_registered_model(replay_bundle, seed, expert_count, topk):
    replay = replay_bundle["replay"]
    return Protocol025ResidualFTMoE(
        replay_bundle["checkpoint"], int(seed), int(expert_count), topk,
        replay.time_scale, replay_bundle["graph_scale"])


class Protocol025FixedSession(Protocol024Session):
    """One fixed-topology Protocol-025 comparator, strict prequential raw-next."""
    def __init__(self, comparator, seed, replay_bundle, budget, out_dir,
                 *, run_id, stream_dir, phase_defs, stream_sha,
                 registration, learning_rate=1e-4):
        if comparator not in FIXED_COMPARATORS:
            raise ValueError("unknown Protocol025 fixed comparator %r" % comparator)
        count, topk = FIXED_COMPARATORS[comparator]
        super().__init__(
            "C", seed, replay_bundle, budget, out_dir, anchor=None,
            learning_rate=learning_rate, run_id=run_id, stream_dir=stream_dir,
            phase_defs=phase_defs, target_mode=NEXT_TARGET_MODE,
            stream_sha=stream_sha, registration=registration)
        self.comparator = comparator
        self.model = _new_registered_model(replay_bundle, seed, count, topk)
        self.model.eval(); self.model.set_deployment("learner", 1.0)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate, weight_decay=1e-4)
        self.frozen_hash = self.model.frozen_hash()
        self.learner_hash = self.learner_state_hash()

    def comparator_manifest(self):
        count, topk = FIXED_COMPARATORS[self.comparator]
        return {
            "name": self.comparator, "expert_count": int(count),
            "topk": topk, "active_experts": int(count if topk is None else topk),
            "residual_input_dim": 73,
        }


class Protocol025DynamicSession(V2CProtocol024Session):
    """Protocol-024 v2c lifecycle on the registered Protocol-025 73-D bank."""
    def __init__(self, seed, replay_bundle, budget, out_dir, *, guard_anchor,
                 run_id, stream_dir, phase_defs, stream_sha, registration,
                 learning_rate=1e-4, v2c_config=None):
        cfg = dict(V2C_DEFAULT); cfg.update(v2c_config or {})
        # Construct the validated v2c lifecycle/session state first, then replace
        # only the residual model before any scored prediction is made.
        super().__init__(
            "D", seed, replay_bundle, budget, out_dir,
            guard_anchor=guard_anchor, v2c_config=cfg,
            learning_rate=learning_rate, max_experts=8,
            run_id=run_id, stream_dir=stream_dir, phase_defs=phase_defs,
            target_mode=NEXT_TARGET_MODE, stream_sha=stream_sha,
            registration=registration)

        model = _new_registered_model(replay_bundle, seed, 4, None)
        source = model.learner
        model.learner = DynamicResidualBank(
            source, max_experts=8,
            ramp_updates=int(cfg.get("crossfade_prediction_intervals", 8)))
        model.requested_method = "D"
        model.set_deployment("learner", 1.0)
        model.learner.set_role_trainability(); model.eval()
        self.model = model

        self.optimizer_archive = {}
        self._live_optimizer_name_map = {}
        self.shadow_optimizer = None
        self.shadow_optimizer_state = None
        self.optimizer = torch.optim.AdamW(
            [p for _, p in self._active_named_parameters()],
            lr=self.learning_rate, weight_decay=1e-4)
        self._refresh_optimizer_name_map()
        self.frozen_hash = self.model.frozen_hash()
        self.learner_hash = self.learner_state_hash()
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator = "D_dynamic"

    def comparator_manifest(self):
        return {
            "name": "D_dynamic", "generalists": 4,
            "max_live_specialists": 1, "resident_limit": 8,
            "residual_input_dim": 73,
            "lifecycle": "Protocol024-v2c replacement-consistent",
        }
