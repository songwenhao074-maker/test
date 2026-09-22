"""Protocol-028 memory-protected lifecycle.

The only scientific change from Protocol-027 is full-resident-capacity birth
admission: an accepted dormant specialist is never purged merely to make room
for an unvalidated shadow. When the resident bank is full, that birth is
skipped and the accepted memory is preserved.
"""
from __future__ import annotations

from copy import deepcopy

from ftmoe_protocol024_v2c import V2C_DEFAULT
from ftmoe_protocol027_normal_guard import (
    Protocol027DynamicSession,
    Protocol027NormalGuardLifecycle,
)


class Protocol028MemoryProtectedLifecycle(Protocol027NormalGuardLifecycle):
    """Protocol-027 lifecycle with memory-protected full-capacity admission."""

    def __init__(self, guard_anchor, config=None):
        super().__init__(guard_anchor=guard_anchor, config=config)
        self.capacity_preserve_skips = 0

    def state_dict(self):
        state = super().state_dict()
        state["protocol028_memory_protected"] = {
            "capacity_preserve_skips": int(self.capacity_preserve_skips),
            "policy": "skip_birth_at_full_resident_capacity_preserve_accepted_memory",
        }
        return state

    def load_state_dict(self, state):
        payload = dict(state)
        p028 = payload.pop("protocol028_memory_protected", None)
        super().load_state_dict(payload)
        if p028 is None:
            raise ValueError("Protocol028 lifecycle checkpoint lacks memory-protected state")
        self.capacity_preserve_skips = int(p028["capacity_preserve_skips"])

    def _free_capacity_for_shadow(self, session):
        bank = session.model.learner
        if bank.resident_count() < bank.max_experts:
            return True

        dormant = [
            str(k) for k in bank.dormant_experts.keys()
            if str(k) in self.specialist_memory
        ]
        preserved = sorted(dormant, key=int)
        would_have_purged = None
        if dormant:
            would_have_purged = min(
                dormant,
                key=lambda k: (
                    int(self.specialist_memory[k]["last_causal_accept_matured"]),
                    int(k),
                ),
            )

        self.capacity_blocked += 1
        self.capacity_preserve_skips += 1
        self._event(
            session,
            "birth_skipped_capacity_preserve_memory",
            expert_id=would_have_purged,
            preserved_expert_ids=preserved,
            reason="resident_capacity_full_preserve_accepted_memory",
            resident_count=int(bank.resident_count()),
            max_experts=int(bank.max_experts),
            shadow_created=False,
            purge_performed=False,
        )
        return False


class Protocol028DynamicSession(Protocol027DynamicSession):
    """Protocol-027 dynamic session with only the Protocol-028 capacity policy."""

    def __init__(self, *args, guard_anchor, v2c_config=None, **kwargs):
        cfg = dict(V2C_DEFAULT)
        cfg.update(v2c_config or {})
        super().__init__(*args, guard_anchor=guard_anchor, v2c_config=cfg, **kwargs)
        self.lifecycle_controller = Protocol028MemoryProtectedLifecycle(
            guard_anchor=guard_anchor, config=cfg
        )
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator = "D_memory_protected"

    def comparator_manifest(self):
        base = deepcopy(super().comparator_manifest())
        base.update({
            "name": "D_memory_protected",
            "lifecycle": "Protocol028 memory-protected full-capacity birth admission",
            "capacity_policy": "skip birth when resident_count==8; preserve accepted dormant memory",
            "source_lifecycle": "Protocol027NormalGuardLifecycle",
        })
        return base
