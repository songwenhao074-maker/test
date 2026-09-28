"""Protocol-034 D_live and read-only complete/fragment prediction memory.

Scientific changes relative to Protocol-033 are deliberately narrow:
  * a stable active specialist and its router row participate in the common
    online optimizer together with the four generalists;
  * dormant specialists remain frozen with archived AdamW state;
  * both specialist participants are frozen during the registered 8-prediction
    crossfade while generalists continue to learn;
  * accepted shadow AdamW moments transfer to the newly activated specialist;
  * a read-only sidecar snapshots complete residual predictors and records
    full/fragment counterfactual probabilities before labels are revealed.

The sidecar never feeds back into training, lifecycle decisions, or routing.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
from ftmoe_protocol024_session import _copy_cpu, _state_to_param
from ftmoe_protocol033_guard_budget import (
    P033_CONFIG,
    Protocol033DynamicSession,
    Protocol033GuardBudgetLifecycle,
)

GENERALISTS = ("0", "1", "2", "3")


def _tensor_hash_update(h, label, tensor):
    x = torch.as_tensor(tensor).detach().cpu().contiguous()
    h.update(str(label).encode("utf8"))
    h.update(str(tuple(x.shape)).encode("utf8"))
    h.update(str(x.dtype).encode("utf8"))
    h.update(x.numpy().tobytes())


def _state_bytes(value):
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_state_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_state_bytes(v) for v in value)
    return 0


def _phase(index, phase_defs):
    return p31.phase_name(max(0, int(index)), phase_defs)


def dormant_state_hash(session):
    """Hash only dormant specialist tensors plus their archived optimizer state."""
    bank = session.model.learner
    ctrl = session.lifecycle_controller
    h = hashlib.sha256()
    for key in sorted(bank.dormant_experts.keys(), key=int):
        key = str(key)
        h.update(("dormant|" + key).encode("utf8"))
        for name, tensor in sorted(bank.dormant_experts[key].state_dict().items()):
            _tensor_hash_update(h, key + "|expert|" + name, tensor)
        _tensor_hash_update(h, key + "|router_weight", bank.dormant_router_weights[key])
        _tensor_hash_update(h, key + "|router_bias", bank.dormant_router_biases[key])
        memory = ctrl.specialist_memory.get(key, {})
        h.update(json.dumps(p31.json_ready(memory), sort_keys=True,
                            separators=(",", ":")).encode("utf8"))
        prefix = key + "|"
        for name in sorted(session.optimizer_archive):
            if str(name).startswith(prefix):
                h.update(str(name).encode("utf8"))
                state = session.optimizer_archive[name]
                for sk in sorted(state):
                    h.update(str(sk).encode("utf8"))
                    value = state[sk]
                    if torch.is_tensor(value):
                        _tensor_hash_update(h, sk, value)
                    else:
                        h.update(repr(value).encode("utf8"))
    return h.hexdigest()


def active_specialist_hash(session):
    ctrl = session.lifecycle_controller
    key = None if ctrl.active_specialist_id is None else str(ctrl.active_specialist_id)
    if key is None or key not in session.model.learner.experts:
        return None
    bank = session.model.learner
    h = hashlib.sha256()
    for name, tensor in sorted(bank.experts[key].state_dict().items()):
        _tensor_hash_update(h, key + "|expert|" + name, tensor)
    _tensor_hash_update(h, key + "|router_weight", bank.router_weights[key])
    _tensor_hash_update(h, key + "|router_bias", bank.router_biases[key])
    return h.hexdigest()


def optimizer_names(session):
    return [name for name, _ in session._active_named_parameters()]


class PredictionMemoryObserver:
    """Read-only sidecar retaining every registered complete predictor snapshot."""

    def __init__(self, steps, hosts, phase_defs, base_hash):
        self.steps = int(steps)
        self.hosts = int(hosts)
        self.phase_defs = list(phase_defs)
        self.base_hash = str(base_hash)
        self.snapshots = []
        self._by_id = {}
        self._dedup = set()
        self.capture_log = []
        self.isolation_checks = []
        self._seq = 0

    @staticmethod
    def _clone_expert(expert):
        clone = deepcopy(expert).cpu().eval()
        for p in clone.parameters():
            p.requires_grad_(False)
            p.grad = None
        return clone

    @staticmethod
    def _snapshot_hash(experts, weights, biases, ids, base_hash):
        h = hashlib.sha256()
        h.update(str(base_hash).encode("utf8"))
        for key in ids:
            h.update(str(key).encode("utf8"))
            for name, tensor in sorted(experts[key].state_dict().items()):
                _tensor_hash_update(h, key + "|expert|" + name, tensor)
            _tensor_hash_update(h, key + "|router_weight", weights[key])
            _tensor_hash_update(h, key + "|router_bias", biases[key])
        return h.hexdigest()

    def capture(self, session, trigger, creation_prediction_index,
                specialist_id=None):
        bank = session.model.learner
        ctrl = session.lifecycle_controller
        key = str(specialist_id if specialist_id is not None
                  else ctrl.active_specialist_id)
        if key in ("None", "") or key not in bank.experts:
            self.capture_log.append({
                "trigger": str(trigger),
                "creation_prediction_index": int(creation_prediction_index),
                "captured": False,
                "reason": "no_stable_resident_active_specialist",
            })
            return None
        ids = list(GENERALISTS) + [key]
        if any(g not in bank.experts for g in GENERALISTS):
            raise AssertionError("Protocol034 snapshot missing a generalist")
        experts = {k: self._clone_expert(bank.experts[k]) for k in ids}
        weights = {k: bank.router_weights[k].detach().cpu().clone() for k in ids}
        biases = {k: bank.router_biases[k].detach().cpu().clone() for k in ids}
        digest = self._snapshot_hash(experts, weights, biases, ids, self.base_hash)
        dedup_key = (key, digest)
        if dedup_key in self._dedup:
            self.capture_log.append({
                "trigger": str(trigger),
                "creation_prediction_index": int(creation_prediction_index),
                "captured": False,
                "reason": "deduplicated_same_lineage_and_state",
                "lineage_id": key,
                "state_sha256": digest,
            })
            return None
        self._dedup.add(dedup_key)
        self._seq += 1
        sid = "snapshot_%03d" % self._seq
        parameter_bytes = 0
        for k in ids:
            parameter_bytes += sum(int(p.numel() * p.element_size())
                                   for p in experts[k].parameters())
            parameter_bytes += int(weights[k].numel() * weights[k].element_size())
            parameter_bytes += int(biases[k].numel() * biases[k].element_size())
        record = {
            "snapshot_id": sid,
            "lineage_id": key,
            "trigger": str(trigger),
            "creation_prediction_index": int(creation_prediction_index),
            "first_usable_prediction_index": int(creation_prediction_index) + 1,
            "source_phase_id_audit_only": _phase(creation_prediction_index, self.phase_defs),
            "source_phase_used_for_selection": False,
            "state_sha256": digest,
            "base_reference_sha256": self.base_hash,
            "parameter_bytes": int(parameter_bytes),
            "full_definition": "frozen four-generalist plus one-specialist residual predictor; immutable base referenced by hash",
            "fragment_definition": "frozen specialist/router row plus current preprediction D_live generalists/router rows; unit ramps",
            "experts": experts,
            "weights": weights,
            "biases": biases,
            "full_probability": np.full((self.steps, self.hosts), np.nan, np.float32),
            "full_class_probability": np.full((self.steps, self.hosts, 3), np.nan, np.float32),
            "fragment_probability": np.full((self.steps, self.hosts), np.nan, np.float32),
            "fragment_class_probability": np.full((self.steps, self.hosts, 3), np.nan, np.float32),
        }
        self.snapshots.append(record)
        self._by_id[sid] = record
        self.capture_log.append({k: p31.json_ready(v) for k, v in record.items()
                                 if k not in ("experts", "weights", "biases",
                                              "full_probability", "full_class_probability",
                                              "fragment_probability", "fragment_class_probability")}
                                | {"captured": True})
        return sid

    @staticmethod
    def _mixture(z, ids, experts, weights, biases):
        weight = torch.stack([torch.as_tensor(weights[k], device=z.device,
                                              dtype=z.dtype) for k in ids], dim=0)
        bias = torch.stack([torch.as_tensor(biases[k], device=z.device,
                                            dtype=z.dtype) for k in ids], dim=0)
        logits = z @ weight.t() + bias
        routed = torch.softmax(logits, dim=-1)
        outputs = torch.stack([experts[k](z) for k in ids], dim=-2)
        return (routed.unsqueeze(-1) * outputs).sum(dim=-2)

    def predict_all(self, session, index, output, *, isolation_check=False):
        index = int(index)
        if not self.snapshots:
            return
        bank = session.model.learner
        z = session.model._last_z.detach()
        base_det = output["base_final_detection_logits"].detach()
        base_cls = output["base_final_class_logits"].detach()
        behavior_before = bank.behavior_state_hash() if isolation_check else None
        rng_before = torch.get_rng_state().clone() if isolation_check else None
        py_before = random.getstate() if isolation_check else None
        with torch.no_grad():
            for snap in self.snapshots:
                if index < int(snap["first_usable_prediction_index"]):
                    continue
                sid = snap["lineage_id"]
                full_ids = list(GENERALISTS) + [sid]
                full_corr = self._mixture(z, full_ids, snap["experts"],
                                          snap["weights"], snap["biases"])
                fd = torch.softmax(base_det + full_corr[..., :2], -1)[0, :, 1]
                fc = torch.softmax(base_cls + full_corr[..., 2:], -1)[0]

                current_experts = {g: bank.experts[g] for g in GENERALISTS}
                current_weights = {g: bank.router_weights[g] for g in GENERALISTS}
                current_biases = {g: bank.router_biases[g] for g in GENERALISTS}
                current_experts[sid] = snap["experts"][sid]
                current_weights[sid] = snap["weights"][sid]
                current_biases[sid] = snap["biases"][sid]
                frag_corr = self._mixture(z, full_ids, current_experts,
                                          current_weights, current_biases)
                rd = torch.softmax(base_det + frag_corr[..., :2], -1)[0, :, 1]
                rc = torch.softmax(base_cls + frag_corr[..., 2:], -1)[0]
                snap["full_probability"][index] = fd.cpu().numpy().astype(np.float32)
                snap["full_class_probability"][index] = fc.cpu().numpy().astype(np.float32)
                snap["fragment_probability"][index] = rd.cpu().numpy().astype(np.float32)
                snap["fragment_class_probability"][index] = rc.cpu().numpy().astype(np.float32)
        if isolation_check:
            behavior_after = bank.behavior_state_hash()
            if behavior_after != behavior_before:
                raise AssertionError("Protocol034 snapshot observer changed live behavior")
            if not torch.equal(torch.get_rng_state(), rng_before):
                raise AssertionError("Protocol034 snapshot observer changed torch RNG")
            if random.getstate() != py_before:
                raise AssertionError("Protocol034 snapshot observer changed python RNG")
            self.isolation_checks.append({
                "prediction_index": index,
                "behavior_sha256": behavior_before,
                "torch_rng_unchanged": True,
                "python_rng_unchanged": True,
            })

    def save(self, root):
        root = Path(root)
        cache_dir = root / "snapshot_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata, files = [], []
        for snap in self.snapshots:
            sid = snap["snapshot_id"]
            path = cache_dir / (sid + ".npz")
            np.savez_compressed(
                path,
                full_probability=snap["full_probability"],
                full_class_probability=snap["full_class_probability"],
                fragment_probability=snap["fragment_probability"],
                fragment_class_probability=snap["fragment_class_probability"],
            )
            digest = p31.sha(path)
            files.append({"snapshot_id": sid, "file": path.name,
                          "sha256": digest, "bytes": path.stat().st_size})
            metadata.append({k: p31.json_ready(v) for k, v in snap.items()
                             if k not in ("experts", "weights", "biases",
                                          "full_probability", "full_class_probability",
                                          "fragment_probability", "fragment_class_probability")})
        p31.write_json(root / "snapshot_metadata.json", metadata)
        p31.write_json(root / "snapshot_capture_log.json", self.capture_log)
        p31.write_json(root / "observer_isolation_checks.json", self.isolation_checks)
        p31.write_json(root / "snapshot_cache_manifest.json", {
            "protocol": "034",
            "snapshot_count": len(metadata),
            "base_reference_sha256": self.base_hash,
            "files": files,
            "all_versions_retained": True,
            "read_only_sidecar": True,
            "learning_feedback": False,
        })
        return metadata


class Protocol034LiveLifecycle(Protocol033GuardBudgetLifecycle):
    """Protocol033 lifecycle plus trainability/snapshot hooks, no decision changes."""

    def _begin_transition(self, session, old_id, new_id, kind):
        observer = getattr(session, "snapshot_observer", None)
        if observer is not None and old_id is not None and self.transition is None:
            observer.capture(session, "immediately_before_stable_active_specialist_is_replaced",
                             session.cursor, specialist_id=old_id)
        result = super()._begin_transition(session, old_id, new_id, kind)
        session._apply_specialist_freeze()
        session._p034_transition_specialist_hash = session._specialist_pair_hash()
        return result

    def before_prediction(self, session, index):
        if self.transition is not None:
            expected = getattr(session, "_p034_transition_specialist_hash", None)
            current = session._specialist_pair_hash()
            if expected is not None and current != expected:
                raise AssertionError("Protocol034 crossfade specialist changed before prediction")
        return super().before_prediction(session, index)

    def on_pre_label_prediction(self, session, index, output):
        was_transition = self.transition is not None
        super().on_pre_label_prediction(session, index, output)
        if was_transition and self.transition is None:
            session._p034_transition_specialist_hash = None
            session._apply_specialist_freeze()
            observer = getattr(session, "snapshot_observer", None)
            if observer is not None and self.active_specialist_id is not None:
                observer.capture(session, "first_full_ramp_after_birth_or_reactivation",
                                 int(index), specialist_id=self.active_specialist_id)

    def on_matured(self, session, index, target):
        super().on_matured(session, index, target)
        observer = getattr(session, "snapshot_observer", None)
        if (observer is not None and self.matured_count > 0
                and self.matured_count % 128 == 0
                and self.transition is None
                and self.active_specialist_id is not None
                and self.phase == "monitoring"):
            observer.capture(session, "every128_matured_intervals_when_stable",
                             session.cursor, specialist_id=self.active_specialist_id)


class Protocol034LiveSession(Protocol033DynamicSession):
    """D_live: stable active specialist learns; dormant specialists never do."""

    def __init__(self, *args, guard_anchor, v2c_config=None, **kwargs):
        cfg = dict(P033_CONFIG)
        cfg.update(v2c_config or {})
        self.snapshot_observer = None
        self.optimizer_transfer_events = []
        self._p034_transition_specialist_hash = None
        super().__init__(*args, guard_anchor=guard_anchor, v2c_config=cfg, **kwargs)
        self.lifecycle_controller = Protocol034LiveLifecycle(
            guard_anchor=guard_anchor, config=cfg)
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator = "D_live"
        self.optimizer_moment_rule = (
            "stable_active_specialist_keeps_live_adamw; dormant_archive_restore; "
            "accepted_shadow_adamw_state_transfers; transition_specialists_frozen"
        )

    def attach_snapshot_observer(self, observer):
        self.snapshot_observer = observer
        return observer

    def _active_named_parameters(self):
        named = []
        if not hasattr(self, "model") or not hasattr(self.model, "learner"):
            return named
        bank = self.model.learner
        generalists = set(getattr(self, "generalist_ids", GENERALISTS))
        allowed = set(str(x) for x in generalists)
        ctrl = getattr(self, "lifecycle_controller", None)
        if ctrl is not None and getattr(ctrl, "transition", None) is None:
            active = getattr(ctrl, "active_specialist_id", None)
            if active is not None:
                allowed.add(str(active))
        for key, parameter in bank.active_named_parameters():
            if str(key[0]) in allowed and parameter.requires_grad:
                named.append((str(key[0]) + "|" + str(key[1]), parameter))
        return named

    def _apply_specialist_freeze(self):
        if getattr(self, "arm", None) != "D" or not hasattr(self, "model"):
            return
        bank = self.model.learner
        bank.set_role_trainability()
        ctrl = getattr(self, "lifecycle_controller", None)
        stable_active = None
        if ctrl is not None and getattr(ctrl, "transition", None) is None:
            value = getattr(ctrl, "active_specialist_id", None)
            stable_active = None if value is None else str(value)
        for key in list(bank.experts.keys()):
            if str(key) in GENERALISTS:
                continue
            train = (stable_active is not None and str(key) == stable_active)
            for p in bank.experts[key].parameters():
                p.requires_grad_(train)
                if not train:
                    p.grad = None
            bank.router_weights[key].requires_grad_(train)
            bank.router_biases[key].requires_grad_(train)
            if not train:
                bank.router_weights[key].grad = None
                bank.router_biases[key].grad = None
        for key in list(bank.dormant_experts.keys()):
            for p in bank.dormant_experts[key].parameters():
                p.requires_grad_(False); p.grad = None
            bank.dormant_router_weights[key].requires_grad_(False)
            bank.dormant_router_biases[key].requires_grad_(False)
            bank.dormant_router_weights[key].grad = None
            bank.dormant_router_biases[key].grad = None
        if hasattr(self, "optimizer"):
            self._rebuild_live_optimizer()

    def _specialist_pair_hash(self):
        bank = self.model.learner
        ctrl = self.lifecycle_controller
        ids = set()
        if getattr(ctrl, "transition", None) is not None:
            ids.update(str(x) for x in (ctrl.transition.get("old_id"),
                                        ctrl.transition.get("new_id")) if x is not None)
        elif ctrl.active_specialist_id is not None:
            ids.add(str(ctrl.active_specialist_id))
        h = hashlib.sha256()
        for key in sorted(ids, key=int):
            if key in bank.experts:
                expert, weight, bias = bank.experts[key], bank.router_weights[key], bank.router_biases[key]
            elif key in bank.dormant_experts:
                expert, weight, bias = bank.dormant_experts[key], bank.dormant_router_weights[key], bank.dormant_router_biases[key]
            else:
                continue
            for name, tensor in sorted(expert.state_dict().items()):
                _tensor_hash_update(h, key + "|" + name, tensor)
            _tensor_hash_update(h, key + "|w", weight)
            _tensor_hash_update(h, key + "|b", bias)
        return h.hexdigest()

    def activate_shadow(self):
        """Activate an accepted shadow while preserving its own AdamW moments."""
        if self.model.learner.shadow_id is None:
            raise RuntimeError("Protocol034 activate_shadow requires shadow")
        self._capture_live_optimizer_state()
        key = str(self.model.learner.shadow_id)
        transferred, missing = [], []
        if self.shadow_optimizer is not None:
            state = self.shadow_optimizer.state
            for name, parameter in self._shadow_named_parameters():
                if parameter in state and state[parameter]:
                    self.optimizer_archive[name] = _copy_cpu(state[parameter])
                    transferred.append(name)
                else:
                    missing.append(name)
        activated = self.model.learner.activate_shadow()
        if str(activated) != key:
            raise AssertionError("Protocol034 shadow ID changed on activation")
        self.shadow_optimizer = None
        self.shadow_optimizer_state = None
        self._rebuild_live_optimizer()
        self.learner_hash = self.learner_state_hash()
        event = {
            "expert_id": key,
            "cursor": int(self.cursor),
            "transferred_state_names": transferred,
            "missing_state_names": missing,
            "transferred_state_count": len(transferred),
            "unrelated_optimizer_archive_preserved": True,
        }
        self.optimizer_transfer_events.append(event)
        return activated

    def update(self, t):
        """Common online update with dormant-only immutability invariant."""
        before = dormant_state_hash(self)
        result = s4.PrequentialS4.update(self, t)
        after = dormant_state_hash(self)
        if before != after:
            raise AssertionError("Protocol034 dormant specialist changed during common live update")
        return result

    def step(self):
        """Protocol033 causal step with a pre-label read-only snapshot observer."""
        t = self.cursor
        if t >= self.steps:
            raise StopIteration
        self.lifecycle_controller.before_prediction(self, t)
        started = time.perf_counter()
        x, sched, graph, context = s4.window_batch(self.replay, [t])
        out = self.model.predict_deployment(x, sched, graph, graph_context=context)
        probability = torch.softmax(out["detection_logits"], -1)[0, :, 1].detach().cpu().numpy().astype(np.float32)
        classes = torch.softmax(out["class_logits"], -1)[0].detach().cpu().numpy().astype(np.float32)
        if not (np.isfinite(probability).all() and np.isfinite(classes).all()):
            raise RuntimeError("non-finite Protocol034 prediction at %d" % t)
        self.predictions["probability"][t] = probability
        self.predictions["class_probability"][t] = classes
        self.predictions["detection_logits"][t] = out["detection_logits"][0].detach().cpu().numpy()
        self.predictions["class_logits"][t] = out["class_logits"][0].detach().cpu().numpy()
        self.predictions["model_version"][t] = self.model_version
        self.predictions["learner_hash"][t] = self.learner_hash
        self.predictions["prediction_seconds"][t] = time.perf_counter() - started
        if not np.all(self.raw_seen[t] == -1):
            raise AssertionError("Protocol034 label read before prediction")

        self.lifecycle_controller.on_pre_label_prediction(self, t, out)
        if self.snapshot_observer is not None:
            isolation = (len(self.snapshot_observer.isolation_checks) == 0
                         and len(self.snapshot_observer.snapshots) > 0)
            self.snapshot_observer.predict_all(self, t, out, isolation_check=isolation)

        self.raw_seen[t] = np.asarray(self.bundle["arrays"]["raw_labels"][t], dtype=np.int64).copy()
        self.predictions["raw_labels"][t] = self.raw_seen[t]
        settled_now = t - 2
        if settled_now >= 0:
            target = self._mature_label(settled_now, t)
            self.predictions["labels"][settled_now] = target
            self.predictions["settled_at"][settled_now] = t
            self.settled[settled_now] = True
            self.buffer.append(settled_now)
            if len(self.buffer) > self.buffer_limit:
                self.buffer = self.buffer[-self.buffer_limit:]
            self.lifecycle_controller.on_matured(self, settled_now, target)
        if (t + 1) % self.update_every == 0:
            self.update(t)
            self.lifecycle_controller.on_after_live_update(self, t)
        self.model_version += 1
        self.cursor = t + 1
        self.lifecycle_controller._sync_session(self)
        req = self.lifecycle_controller.pop_checkpoint_request()
        if req is not None:
            record = self.save_checkpoint(req["label"], t)
            self.lifecycle_controller.reactivation_checkpoint_records.append({**req, **record})
            self.lifecycle_controller._sync_session(self)
        return probability, classes

    def comparator_manifest(self):
        base = deepcopy(super().comparator_manifest())
        base.update({
            "name": "D_live",
            "protocol": "034",
            "active_specialist_trainable_when_stable": True,
            "active_specialist_router_trainable_when_stable": True,
            "dormant_specialist_trainable": False,
            "transition_specialists_trainable": False,
            "generalists_trainable": True,
            "accepted_shadow_optimizer_moments_transfer": True,
            "snapshot_observer_learning_feedback": False,
        })
        return base
