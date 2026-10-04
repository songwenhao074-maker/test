"""Protocol-033 lifecycle: Protocol-032 guard budget plus the registered 1000-step birth clock.

Decision semantics remain Protocol-032.  This wrapper also fixes reuse logging so
each accepted reactivation gets its own event id and first-influence index.
"""
from __future__ import annotations

from copy import deepcopy

from ftmoe_protocol031_nonblocking_reuse import P031_CONFIG
from ftmoe_protocol032_guard_budget import (
    Protocol032GuardBudgetLifecycle,
    Protocol032DynamicSession,
)

P033_CONFIG = dict(P031_CONFIG)
P033_CONFIG.update({
    "proposal_start_matured": 600,
    "proposal_every_matured": 1000,
})


class Protocol033GuardBudgetLifecycle(Protocol032GuardBudgetLifecycle):
    def __init__(self, guard_anchor, config=None):
        cfg = dict(P033_CONFIG)
        cfg.update(config or {})
        super().__init__(guard_anchor=guard_anchor, config=cfg)
        self.p033_reuse_event_seq = 0
        self.p033_reuse_event_records = []

    def state_dict(self):
        state = super().state_dict()
        state["protocol033_logging"] = {
            "reuse_event_seq": int(self.p033_reuse_event_seq),
            "reuse_event_records": deepcopy(self.p033_reuse_event_records),
        }
        return state

    def load_state_dict(self, state):
        payload = dict(state)
        p033 = payload.pop("protocol033_logging", None)
        if p033 is None:
            raise ValueError("Protocol033 lifecycle checkpoint lacks per-event reuse logging state")
        super().load_state_dict(payload)
        self.p033_reuse_event_seq = int(p033["reuse_event_seq"])
        self.p033_reuse_event_records = deepcopy(p033["reuse_event_records"])

    def _start_candidate(self, session, reason):
        # Protocol031's event label contains the old 1600 cadence as prose only;
        # normalize that label without changing the inherited due calculation.
        if reason == "periodic_background_budget_1600":
            reason = "periodic_background_budget_1000_protocol033"
        return super()._start_candidate(session, reason)

    def _begin_transition(self, session, old_id, new_id, kind):
        result = super()._begin_transition(session, old_id, new_id, kind)
        if str(kind) == "reactivation":
            self.p033_reuse_event_seq += 1
            event_id = "reuse_%03d" % self.p033_reuse_event_seq
            rec = {
                "reuse_event_id": event_id,
                "expert_id": str(new_id),
                "replaced_active_specialist_id": None if old_id is None else str(old_id),
                "accepted_cursor": int(session.cursor),
                "accepted_matured": int(self.matured_count),
                "first_influence_prediction_index": None,
            }
            self.p033_reuse_event_records.append(rec)
            if self.transition is not None:
                self.transition["protocol033_reuse_event_id"] = event_id
            self._event(session, "protocol033_reuse_event_accepted",
                        reuse_event_id=event_id, expert_id=str(new_id),
                        accepted_cursor=int(session.cursor))
        return result

    def on_pre_label_prediction(self, session, index, output):
        super().on_pre_label_prediction(session, index, output)
        transition = self.transition
        if transition is None or transition.get("kind") != "reactivation":
            return
        event_id = transition.get("protocol033_reuse_event_id")
        if not event_id:
            return
        for rec in reversed(self.p033_reuse_event_records):
            if rec["reuse_event_id"] == event_id:
                if rec["first_influence_prediction_index"] is None:
                    rec["first_influence_prediction_index"] = int(index)
                    if int(index) < int(rec["accepted_cursor"]):
                        raise AssertionError("Protocol033 reuse influence predates acceptance")
                    self._event(session, "protocol033_reuse_event_first_influence",
                                reuse_event_id=event_id, expert_id=rec["expert_id"],
                                accepted_cursor=rec["accepted_cursor"],
                                prediction_index=int(index))
                break


class Protocol033DynamicSession(Protocol032DynamicSession):
    def __init__(self, *args, guard_anchor, v2c_config=None, **kwargs):
        cfg = dict(P033_CONFIG)
        cfg.update(v2c_config or {})
        super().__init__(*args, guard_anchor=guard_anchor, v2c_config=cfg, **kwargs)
        self.lifecycle_controller = Protocol033GuardBudgetLifecycle(
            guard_anchor=guard_anchor, config=cfg)
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator = "D_guard_budget"

    def comparator_manifest(self):
        base = deepcopy(super().comparator_manifest())
        base.update({
            "name": "D_guard_budget",
            "protocol": "033",
            "birth_start_matured": 600,
            "birth_every_matured": 1000,
            "protocol033_changes": [
                "registered 5969-row timeline",
                "birth cadence 1600 -> 1000 matured intervals",
                "per-reactivation reuse event logging only",
            ],
            "decision_policy_source": "Protocol032GuardBudgetLifecycle",
        })
        return base
