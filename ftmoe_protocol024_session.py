"""Protocol-024 strict-prequential session for the next-round causal pilot.

This file intentionally keeps the frozen Protocol-023 base/model family but
owns every Protocol-024 output boundary: target maturation, phase timeline,
evaluation, checkpoint namespace, dynamic topology serialization and optimizer
state.  No Protocol-023 global checkpoint directory or phase table is used by
these methods.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch import nn

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol024_eval import (
    assert_complete_settlement,
    binary_detection_metrics,
    positive_resource_macro_f1,
    raw_next_target,
    temporal_onset_metrics,
    tolerance_label,
)
from recovery.PreGANSrc.src.ftmoe_online_r1 import (
    FixedResidualBank,
    FrozenResidualFTMoE,
)


PROTOCOL = "024"
DEFAULT_TARGET_MODE = "tol1_regression"
NEXT_TARGET_MODE = "raw_next_fault"


def _copy_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_cpu(v) for v in value)
    return deepcopy(value)


def _safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not value:
        raise ValueError("empty checkpoint label")
    return value


def _stable_key(key_tuple):
    expert_id, tensor_name = key_tuple
    return "%s|%s" % (str(expert_id), str(tensor_name))


def _optimizer_hparams(optimizer, fallback_lr=1e-4):
    if optimizer is None or not optimizer.param_groups:
        return {"lr": float(fallback_lr), "betas": (0.9, 0.999),
                "eps": 1e-8, "weight_decay": 1e-4, "amsgrad": False}
    group = optimizer.param_groups[0]
    return {"lr": float(group.get("lr", fallback_lr)),
            "betas": tuple(group.get("betas", (0.9, 0.999))),
            "eps": float(group.get("eps", 1e-8)),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "amsgrad": bool(group.get("amsgrad", False))}


def _state_to_param(state, parameter):
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().clone().to(parameter.device)
        else:
            out[key] = deepcopy(value)
    return out


def _normalize_phases(phases):
    """Return [{name,start,end,regime}] with end exclusive."""
    result = []
    for item in phases or []:
        if isinstance(item, dict):
            name = item.get("name", item.get("phase"))
            start = item.get("start", item.get("start_interval"))
            if "end_exclusive" in item:
                end = item["end_exclusive"]
            else:
                end = item.get("end", item.get("end_interval"))
            regime = item.get("regime", item.get("mode", item.get("response_law")))
        else:
            if len(item) < 3:
                raise ValueError("phase tuple requires name,start,end")
            name, start, end = item[:3]
            regime = item[3] if len(item) > 3 else None
        if name is None or start is None or end is None:
            raise ValueError("phase requires name/start/end")
        start, end = int(start), int(end)
        if start < 0 or end <= start:
            raise ValueError("invalid phase interval %r" % (item,))
        result.append({"name": str(name), "start": start, "end": end,
                       "regime": regime})
    previous = 0
    for phase in result:
        if phase["start"] != previous:
            raise ValueError("phase timeline must be contiguous from zero")
        previous = phase["end"]
    return result


def _manifest_phases(manifest):
    if not isinstance(manifest, dict):
        return []
    for key in ("phases", "phase_table", "timeline"):
        value = manifest.get(key)
        if isinstance(value, list) and value:
            return _normalize_phases(value)
    return []


class DynamicResidualFTMoE(FrozenResidualFTMoE):
    """Standalone fixed-C model with only its learner residual bank dynamic."""

    def __init__(self, checkpoint, method, seed, max_experts=8, ramp_updates=10):
        if method != "D":
            raise ValueError("Protocol-024 dynamic residual model requires method D")
        super().__init__(checkpoint, "C", seed)
        source = self.learner
        self.learner = DynamicResidualBank(
            source, max_experts=max_experts, ramp_updates=ramp_updates)
        self.requested_method = "D"
        # This call is safe only before any dormant/shadow role exists.
        self.set_trainability()
        self.learner.set_role_trainability()
        self.set_deployment("learner", 1.0)

    def auxiliary_losses(self, output):
        prototype, _ = super().auxiliary_losses(output)
        probabilities = output.get("correction_router_probabilities")
        if probabilities is None:
            balance = prototype.new_zeros(())
        else:
            count = probabilities.shape[-1]
            balance = ((probabilities.mean(dim=(0, 1)) - 1.0 / count) ** 2).mean()
        return prototype, balance


class Protocol024Session(s4.PrequentialS4):
    """P24 session with manifest timeline, run-local checkpoints and D state."""

    def __init__(self, arm, seed, replay_bundle, budget, out_dir,
                 probe_paths=None, anchor=None, learning_rate=1e-4,
                 max_experts=8, ramp_updates=10, *, run_id=None,
                 stream_dir=None, phase_defs=None, target_mode=DEFAULT_TARGET_MODE,
                 stream_sha=None, checkpoint_source_sha=None,
                 registration=None):
        if arm not in ("A", "C", "D"):
            raise ValueError("arm must be A, C, or D")
        if target_mode not in (DEFAULT_TARGET_MODE, NEXT_TARGET_MODE):
            raise ValueError("unsupported target_mode %r" % target_mode)
        super().__init__("A" if arm == "A" else "C", seed, replay_bundle,
                         budget, out_dir, probe_paths=probe_paths, anchor=anchor,
                         learning_rate=learning_rate)
        self.arm = arm
        self.run_id = str(run_id or "engineering_compat")
        self.stream_dir = None if stream_dir is None else Path(stream_dir)
        self.target_mode = str(target_mode)
        self.registration = deepcopy(registration or {})
        self.stream_sha = str(stream_sha or
                              replay_bundle.get("manifest", {}).get("stream_sha256", "unknown"))
        self.checkpoint_source_sha = checkpoint_source_sha
        inherited = _manifest_phases(replay_bundle.get("manifest", {}))
        self.phase_defs = _normalize_phases(phase_defs) if phase_defs is not None else inherited
        self.checkpoint_dir = Path(self.out_dir) / "checkpoints"
        self.lifecycle_events = []
        self.lifecycle_state = {"enabled": False, "hard_delete_enabled": False}
        self.shadow_optimizer = None
        self.shadow_optimizer_state = None
        self.optimizer_archive = {}
        self._live_optimizer_name_map = {}
        self.optimizer_moment_rule = "shadow_reset_on_activation;dormant_restore_on_reactivation"

        if arm == "D":
            source = self.model.learner
            self.model.learner = DynamicResidualBank(
                source, max_experts=max_experts, ramp_updates=ramp_updates)
            self.model.requested_method = "D"
            self.model.set_deployment("learner", 1.0)
            self.model.eval()
            self.model.learner.set_role_trainability()
            self.optimizer = torch.optim.AdamW(
                [p for _, p in self._active_named_parameters()],
                lr=self.learning_rate, weight_decay=1e-4)
            self.frozen_hash = self.model.frozen_hash()
            self.protocol024_dynamic_container = True
            self._refresh_optimizer_name_map()
        self.learner_hash = self.learner_state_hash()

    # ------------------------------------------------------------------
    # Dynamic optimizer identity/state
    # ------------------------------------------------------------------
    def _active_named_parameters(self):
        if self.arm != "D":
            return [(name, p) for name, p in self.model.named_parameters()
                    if p.requires_grad]
        return [(_stable_key(key), p)
                for key, p in self.model.learner.active_named_parameters()]

    def _shadow_named_parameters(self):
        if self.arm != "D":
            return []
        return [(_stable_key(key), p)
                for key, p in self.model.learner.shadow_named_parameters()]

    def _refresh_optimizer_name_map(self):
        self._live_optimizer_name_map = {
            id(p): name for name, p in self._active_named_parameters()}

    def _capture_live_optimizer_state(self):
        if self.optimizer is None:
            return
        for parameter, state in self.optimizer.state.items():
            name = self._live_optimizer_name_map.get(id(parameter))
            if name is not None:
                self.optimizer_archive[name] = _copy_cpu(state)

    def _rebuild_live_optimizer(self):
        if self.arm != "D":
            return
        self._capture_live_optimizer_state()
        hparams = _optimizer_hparams(self.optimizer, self.learning_rate)
        named = self._active_named_parameters()
        self.optimizer = torch.optim.AdamW([p for _, p in named], **hparams)
        for name, parameter in named:
            if name in self.optimizer_archive:
                self.optimizer.state[parameter] = _state_to_param(
                    self.optimizer_archive[name], parameter)
        self._refresh_optimizer_name_map()

    def create_shadow_optimizer(self):
        if self.arm != "D" or self.model.learner.shadow_id is None:
            raise RuntimeError("D shadow must exist before creating shadow optimizer")
        named = self._shadow_named_parameters()
        self.shadow_optimizer = torch.optim.AdamW(
            [p for _, p in named], lr=self.learning_rate, weight_decay=1e-4)
        self.shadow_optimizer_state = {"names": [name for name, _ in named],
                                       "moment_rule": "reset_on_activation"}
        return self.shadow_optimizer

    def activate_shadow(self):
        if self.arm != "D":
            raise RuntimeError("only D has lifecycle topology")
        self._capture_live_optimizer_state()
        key = self.model.learner.activate_shadow()
        # Registered rule: candidate Adam moments are NOT migrated into live;
        # existing active-expert moments are preserved by stable names.
        self.shadow_optimizer = None
        self.shadow_optimizer_state = None
        self._rebuild_live_optimizer()
        self.learner_hash = self.learner_state_hash()
        return key

    def discard_shadow(self):
        if self.arm != "D":
            raise RuntimeError("only D has lifecycle topology")
        key = self.model.learner.discard_shadow()
        self.shadow_optimizer = None
        self.shadow_optimizer_state = None
        self.learner_hash = self.learner_state_hash()
        return key

    def retire_expert(self, key):
        self._capture_live_optimizer_state()
        key = self.model.learner.retire(key)
        self._rebuild_live_optimizer()
        self.learner_hash = self.learner_state_hash()
        return key

    def reactivate_expert(self, key):
        key = self.model.learner.reactivate(key)
        self._rebuild_live_optimizer()  # archived moments return by stable name
        self.learner_hash = self.learner_state_hash()
        return key

    def purge_expert(self, key):
        key = str(key)
        result = self.model.learner.purge(key)
        prefix = key + "|"
        self.optimizer_archive = {
            name: state for name, state in self.optimizer_archive.items()
            if not name.startswith(prefix)}
        self.learner_hash = self.learner_state_hash()
        return result

    # ------------------------------------------------------------------
    # Strict prequential target maturation
    # ------------------------------------------------------------------
    def learner_state_hash(self):
        if self.arm == "D" and isinstance(self.model.learner, DynamicResidualBank):
            return self.model.learner.behavior_state_hash()
        return self.model.learner_state_hash()

    def _mature_label(self, index, observed_until):
        if self.target_mode == NEXT_TARGET_MODE:
            return raw_next_target(self.raw_seen, index, observed_until)
        return tolerance_label(self.raw_seen, index, observed_until - 1)

    def step(self):
        """Prediction -> physical observation -> t-2 settlement -> update."""
        t = self.cursor
        if t >= self.steps:
            raise StopIteration
        started = time.perf_counter()
        x, s, g, context = s4.window_batch(self.replay, [t])
        out = self.model.predict_deployment(x, s, g, graph_context=context)
        probability = torch.softmax(out["detection_logits"], -1)[0, :, 1] \
            .detach().numpy().astype(np.float32)
        classes = torch.softmax(out["class_logits"], -1)[0] \
            .detach().numpy().astype(np.float32)
        if not (np.isfinite(probability).all() and np.isfinite(classes).all()):
            raise RuntimeError("non-finite prediction at %d" % t)
        self.predictions["probability"][t] = probability
        self.predictions["class_probability"][t] = classes
        self.predictions["detection_logits"][t] = out["detection_logits"][0].detach().numpy()
        self.predictions["class_logits"][t] = out["class_logits"][0].detach().numpy()
        self.predictions["model_version"][t] = self.model_version
        self.predictions["learner_hash"][t] = self.learner_hash
        self.predictions["prediction_seconds"][t] = time.perf_counter() - started
        if not np.all(self.raw_seen[t] == -1):
            raise AssertionError("label of %d was read before prediction" % t)

        # Physical row t becomes observable only after prediction t is fixed.
        self.raw_seen[t] = np.asarray(
            self.bundle["arrays"]["raw_labels"][t], dtype=np.int64).copy()
        self.predictions["raw_labels"][t] = self.raw_seen[t]

        settled_now = t - 2
        if settled_now >= 0:
            label = self._mature_label(settled_now, t)
            self.predictions["labels"][settled_now] = label
            self.predictions["settled_at"][settled_now] = t
            self.settled[settled_now] = True
            self.buffer.append(settled_now)
            if len(self.buffer) > self.buffer_limit:
                self.buffer = self.buffer[-self.buffer_limit:]

        if self.arm in ("C", "D") and (t + 1) % self.update_every == 0:
            self.update(t)
        self.model_version += 1
        self.cursor = t + 1
        return probability, classes

    def finish(self):
        if self.cursor != self.steps:
            raise ValueError("cannot finalize a partial stream (%d/%d)"
                             % (self.cursor, self.steps))
        self.raw_seen[self.steps] = np.asarray(
            self.bundle["arrays"]["raw_labels"][self.steps], dtype=np.int64).copy()

        for index in range(max(0, self.steps - 2), self.steps):
            if self.settled[index]:
                continue
            if self.target_mode == NEXT_TARGET_MODE:
                # Final scoring may use the guard row; no post-stream training
                # is performed. The recorded settlement time still reflects
                # the one-interval publication delay for the last prediction.
                label = self.raw_seen[index + 1].copy()
                settled_at = max(self.steps, index + 2)
            else:
                label = tolerance_label(self.raw_seen, index, self.steps)
                settled_at = self.steps
            self.predictions["labels"][index] = label
            self.predictions["settled_at"][index] = settled_at
            self.settled[index] = True
            self.buffer.append(index)
        if len(self.buffer) > self.buffer_limit:
            self.buffer = self.buffer[-self.buffer_limit:]
        integrity = assert_complete_settlement(
            self.predictions["labels"], self.predictions["raw_labels"],
            self.settled, self.steps)
        self.lifecycle_state["final_integrity"] = integrity
        self.assert_frozen()
        return integrity

    # ------------------------------------------------------------------
    # P24 evaluator wiring -- no P23 PHASES/legacy onset implementation
    # ------------------------------------------------------------------
    def _metric_block(self, start, end):
        probability = self.predictions["probability"][start:end]
        classes = self.predictions["class_probability"][start:end]
        target = self.predictions["labels"][start:end]
        raw = self.predictions["raw_labels"][start:end]
        if (target < 0).any() or (raw < 0).any():
            raise AssertionError("phase metrics require fully settled rows")
        return {
            "intervals": [int(start), int(end)],
            "rows": int((end - start) * target.shape[1]),
            "detection": binary_detection_metrics(probability, target),
            "detection_current_raw": binary_detection_metrics(probability, raw),
            "diagnosis": positive_resource_macro_f1(classes, target),
            "onset": temporal_onset_metrics(probability, raw, horizon=1),
        }

    def phase_metrics(self):
        if not self.phase_defs:
            raise RuntimeError("Protocol-024 phase_metrics requires manifest phase_defs")
        if self.phase_defs[-1]["end"] > self.steps:
            raise ValueError("phase timeline exceeds scored stream")
        out = []
        for phase in self.phase_defs:
            start, end = phase["start"], phase["end"]
            entry = {"phase": phase["name"], "regime": phase.get("regime"),
                     "arm": self.arm, "target_mode": self.target_mode,
                     "intervals": [start, end]}
            entry["whole"] = self._metric_block(start, end)
            entry["first_100"] = self._metric_block(start, min(end, start + 100))
            entry["late"] = self._metric_block(max(start, end - 100), end)
            entry["rows"] = entry["whole"]["rows"]
            entry["detection"] = entry["whole"]["detection"]
            entry["diagnosis"] = entry["whole"]["diagnosis"]
            entry["onset"] = entry["whole"]["onset"]
            out.append(entry)
        return out

    def _probe_replays(self):
        if self._probe_cache is not None:
            return self._probe_cache
        if self.stream_dir is None:
            raise RuntimeError("Protocol-024 probes require explicit stream_dir")
        from run_ftmoe_protocol020 import ReplayV3
        scale_bundle = s4.build_replay(self.stream_dir)
        cache = {}
        for regime, path in sorted(self.probe_paths.items()):
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
            steps = int(arrays["raw_labels"].shape[0]) - 1
            caps = np.asarray(arrays["capacities"], np.float64)
            probe_arrays = dict(arrays)
            probe_arrays["capacities_per_interval"] = caps
            probe_arrays["capacities"] = caps[0]
            cache[regime] = {
                "replay": ReplayV3(probe_arrays,
                                   scale_bundle["replay"].time_scale,
                                   scale_bundle["replay"].graph_scale, steps),
                "steps": steps,
                "raw": np.asarray(arrays["raw_labels"][:steps + 1])}
        self._probe_cache = cache
        return cache

    def probe_scores(self, checkpoint_label, index):
        if not self.probe_paths:
            return []
        cache = self._probe_replays()
        added = []
        for regime, entry in sorted(cache.items()):
            replay, steps, raw = entry["replay"], entry["steps"], entry["raw"]
            probabilities, classes = [], []
            with torch.no_grad():
                for offset in range(0, steps, 32):
                    chunk = list(range(offset, min(steps, offset + 32)))
                    x, s, g, context = s4.window_batch(replay, chunk)
                    out = self.model.predict_deployment(x, s, g, graph_context=context)
                    probabilities.append(torch.softmax(
                        out["detection_logits"], -1)[..., 1].cpu().numpy())
                    classes.append(torch.softmax(
                        out["class_logits"], -1).cpu().numpy())
            probability = np.concatenate(probabilities, axis=0)
            class_probability = np.concatenate(classes, axis=0)
            current = raw[:steps]
            target = raw[1:steps + 1] if self.target_mode == NEXT_TARGET_MODE else current
            row = {"checkpoint": checkpoint_label, "time_index": int(index),
                   "probe": regime, "model_version": int(self.model_version),
                   "learner_hash": self.learner_hash,
                   "target_mode": self.target_mode,
                   "detection": binary_detection_metrics(probability, target),
                   "diagnosis": positive_resource_macro_f1(class_probability, target),
                   "onset": temporal_onset_metrics(probability, current, horizon=1)}
            self.probe_matrix.append(row)
            added.append(row)
        return added

    # ------------------------------------------------------------------
    # Gap-safe dynamic bank snapshot
    # ------------------------------------------------------------------
    def save_checkpoint(self, phase_name, index):
        """Save a P24 resumable checkpoint under this arm's run directory."""
        label = _safe_name(phase_name)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / (label + ".pt")
        if path.exists():
            raise FileExistsError("refusing to overwrite P24 checkpoint %s" % path)
        payload = self._checkpoint_payload(label, index)
        torch.save(payload, path)
        digest = s4.sha(path)
        record = {"file": path.name, "path": str(path), "sha256": digest,
                  "protocol": PROTOCOL, "run_id": self.run_id,
                  "arm": self.arm, "phase": label, "time_index": int(index),
                  "stream_sha256": self.stream_sha,
                  "learner_hash": self.learner_state_hash(),
                  "update_count": int(self.updates)}
        self.checkpoints[label] = record
        return record

    def _checkpoint_payload(self, label, index):
        if self.arm == "D":
            self._capture_live_optimizer_state()
            optimizer_payload = {"kind": "named_adamw",
                                 "hparams": _optimizer_hparams(
                                     self.optimizer, self.learning_rate),
                                 "archive": _copy_cpu(self.optimizer_archive)}
            dynamic = export_dynamic_bank(self.model.learner)
        else:
            optimizer_payload = (None if self.optimizer is None else
                                 {"kind": "state_dict",
                                  "state": _copy_cpu(self.optimizer.state_dict())})
            dynamic = None
        shadow_payload = None
        if self.shadow_optimizer is not None:
            shadow_payload = {"state": _copy_cpu(self.shadow_optimizer.state_dict()),
                              "meta": _copy_cpu(self.shadow_optimizer_state)}
        return {
            "protocol": PROTOCOL, "run_id": self.run_id, "arm": self.arm,
            "target_mode": self.target_mode, "phase": label,
            "time_index": int(index), "stream_sha256": self.stream_sha,
            "stream_dir": None if self.stream_dir is None else str(self.stream_dir),
            "checkpoint_source_sha256": self.checkpoint_source_sha,
            "registration": _copy_cpu(self.registration),
            "model": _copy_cpu(self.model.state_dict()),
            "dynamic_bank": dynamic,
            "optimizer": optimizer_payload,
            "shadow_optimizer": shadow_payload,
            "base_hash": self.model.frozen_hash(),
            "learner_hash": self.learner_state_hash(),
            "session": {
                "cursor": int(self.cursor), "model_version": int(self.model_version),
                "updates": int(self.updates), "raw_seen": self.raw_seen.copy(),
                "predictions": _copy_cpu(self.predictions),
                "buffer": list(self.buffer), "settled": self.settled.copy(),
                "update_log": _copy_cpu(self.update_log),
                "settlement_log": _copy_cpu(self.settlement_log),
                "probe_matrix": _copy_cpu(self.probe_matrix),
                "lifecycle_events": _copy_cpu(self.lifecycle_events),
                "lifecycle_state": _copy_cpu(self.lifecycle_state),
                "optimizer_archive": _copy_cpu(self.optimizer_archive),
                "rng": {"torch": torch.get_rng_state(),
                         "numpy": np.random.get_state(),
                         "python": random.getstate()},
            },
        }

    def restore_checkpoint(self, path):
        """Restore full P24 session into an identically configured instance."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        for key, expected in (("protocol", PROTOCOL), ("run_id", self.run_id),
                              ("arm", self.arm), ("target_mode", self.target_mode),
                              ("stream_sha256", self.stream_sha)):
            if payload.get(key) != expected:
                raise ValueError("checkpoint %s mismatch: %r != %r"
                                 % (key, payload.get(key), expected))
        if self.arm == "D":
            source = FixedResidualBank()
            self.model.learner = restore_dynamic_bank(source, payload["dynamic_bank"])
            self.model.set_deployment("learner", 1.0)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        if self.arm == "D":
            self.model.learner.set_role_trainability()
            self.optimizer_archive = _copy_cpu(payload["optimizer"]["archive"])
            named = self._active_named_parameters()
            self.optimizer = torch.optim.AdamW(
                [p for _, p in named], **payload["optimizer"]["hparams"])
            for name, parameter in named:
                if name in self.optimizer_archive:
                    self.optimizer.state[parameter] = _state_to_param(
                        self.optimizer_archive[name], parameter)
            self._refresh_optimizer_name_map()
        elif self.optimizer is not None and payload["optimizer"] is not None:
            self.optimizer.load_state_dict(payload["optimizer"]["state"])

        state = payload["session"]
        self.cursor = int(state["cursor"])
        self.model_version = int(state["model_version"])
        self.updates = int(state["updates"])
        self.raw_seen = state["raw_seen"].copy()
        self.predictions = _copy_cpu(state["predictions"])
        self.buffer = list(state["buffer"])
        self.settled = state["settled"].copy()
        self.update_log = _copy_cpu(state["update_log"])
        self.settlement_log = _copy_cpu(state["settlement_log"])
        self.probe_matrix = _copy_cpu(state["probe_matrix"])
        self.lifecycle_events = _copy_cpu(state["lifecycle_events"])
        self.lifecycle_state = _copy_cpu(state["lifecycle_state"])
        self.optimizer_archive.update(_copy_cpu(state.get("optimizer_archive", {})))
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        self.frozen_hash = payload["base_hash"]
        self.learner_hash = self.learner_state_hash()
        if self.model.frozen_hash() != payload["base_hash"]:
            raise AssertionError("restored frozen base hash mismatch")
        if self.learner_hash != payload["learner_hash"]:
            raise AssertionError("restored learner behavior hash mismatch")
        return payload

    def save(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(self.out_dir) / "predictions.npz"
        if path.exists():
            raise FileExistsError("refusing to overwrite predictions %s" % path)
        np.savez_compressed(
            path,
            probability=self.predictions["probability"],
            class_probability=self.predictions["class_probability"],
            detection_logits=self.predictions["detection_logits"],
            class_logits=self.predictions["class_logits"],
            labels=self.predictions["labels"],
            raw_labels=self.predictions["raw_labels"],
            model_version=self.predictions["model_version"],
            settled_at=self.predictions["settled_at"],
            prediction_seconds=self.predictions["prediction_seconds"])
        return path


def export_dynamic_bank(bank):
    """Torch-save-safe active/dormant/shadow tensor + topology snapshot."""
    if not isinstance(bank, DynamicResidualBank):
        raise TypeError("bank must be DynamicResidualBank")
    topology = deepcopy(bank.topology_manifest())
    groups = {}
    for group_name, experts, weights, biases in (
        ("active", bank.experts, bank.router_weights, bank.router_biases),
        ("dormant", bank.dormant_experts, bank.dormant_router_weights,
         bank.dormant_router_biases),
        ("shadow", bank.shadow_experts, bank.shadow_router_weights,
         bank.shadow_router_biases),
    ):
        payload = {}
        for key in experts.keys():
            payload[str(key)] = {
                "expert": {name: tensor.detach().cpu().clone()
                           for name, tensor in experts[key].state_dict().items()},
                "router_weight": weights[key].detach().cpu().clone(),
                "router_bias": biases[key].detach().cpu().clone(),
            }
        groups[group_name] = payload
    return {"topology": topology, "groups": groups,
            "behavior_hash": bank.behavior_state_hash()}


def restore_dynamic_bank(source_bank, snapshot):
    """Restore actual IDs directly; rejected/purged ID gaps are valid."""
    topology = deepcopy(snapshot["topology"])
    target = DynamicResidualBank(
        source_bank, max_experts=int(topology["max_experts"]),
        ramp_updates=int(topology["ramp_updates"]))
    template = deepcopy(source_bank.experts[0])
    target.experts = nn.ModuleDict()
    target.router_weights = nn.ParameterDict()
    target.router_biases = nn.ParameterDict()
    target.dormant_experts = nn.ModuleDict()
    target.dormant_router_weights = nn.ParameterDict()
    target.dormant_router_biases = nn.ParameterDict()
    target.shadow_experts = nn.ModuleDict()
    target.shadow_router_weights = nn.ParameterDict()
    target.shadow_router_biases = nn.ParameterDict()

    destinations = {
        "active": (target.experts, target.router_weights, target.router_biases),
        "dormant": (target.dormant_experts, target.dormant_router_weights,
                    target.dormant_router_biases),
        "shadow": (target.shadow_experts, target.shadow_router_weights,
                   target.shadow_router_biases),
    }
    for group_name, payloads in snapshot["groups"].items():
        experts, weights, biases = destinations[group_name]
        for key, payload in payloads.items():
            expert = deepcopy(template)
            expert.load_state_dict(payload["expert"], strict=True)
            experts[str(key)] = expert
            weights[str(key)] = nn.Parameter(payload["router_weight"].clone())
            biases[str(key)] = nn.Parameter(payload["router_bias"].clone())

    target.ids = [str(x) for x in topology["active_ids"]]
    target.ramp = {str(k): float(v) for k, v in topology["ramp"].items()}
    target.shadow_id = (None if topology.get("shadow_id") is None
                        else str(topology["shadow_id"]))
    target.next_id = int(topology["next_id"])
    target.topology_version = int(topology.get("topology_version", 0))
    target.behavior_version = int(topology.get("behavior_version", 0))
    target.set_role_trainability()

    if set(target.ids) != set(target.experts.keys()):
        raise ValueError("active ID snapshot mismatch")
    if set(map(str, topology.get("dormant_ids", []))) != set(target.dormant_experts.keys()):
        raise ValueError("dormant ID snapshot mismatch")
    expected_shadow = set([] if target.shadow_id is None else [target.shadow_id])
    if expected_shadow != set(target.shadow_experts.keys()):
        raise ValueError("shadow ID snapshot mismatch")
    if target.resident_count() > target.max_experts:
        raise ValueError("restored resident capacity exceeds max")
    expected_hash = snapshot.get("behavior_hash")
    if expected_hash is not None and target.behavior_state_hash() != expected_hash:
        raise AssertionError("dynamic behavior hash changed during restore")
    return target
