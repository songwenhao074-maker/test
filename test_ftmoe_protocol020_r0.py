"""Causal regression tests for Protocol 020 ``prequential_v1``.

These tests deliberately use tiny stand-ins for the model and replay.  They
exercise the R0 ledger and serialization contracts without loading a
checkpoint in the unit cases; the final class adds one short real
checkpoint/stream integration slice using the registered S8 fixture.
"""

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import random
import unittest

import numpy as np
import torch
from torch import nn

from run_ftmoe_protocol020 import ReplayV3
from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, EAGateMoE
from recovery.PreGANSrc.src.ftmoe_dynamic_expert_v3 import V3TriggerState
from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from recovery.PreGANSrc.src.ftmoe_online_s7 import S7Session, s7_loss
from recovery.PreGANSrc.src.ftmoe_online_s8 import OnlineEAGateV3, S8Session
import test_ftmoe_protocol020_s8 as s8_fixture_module


def _assert_nested_equal(testcase, left, right):
    """Compare checkpoint-shaped values, including tensors and arrays."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        testcase.assertIsInstance(left, torch.Tensor)
        testcase.assertIsInstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        testcase.assertIsInstance(left, np.ndarray)
        testcase.assertIsInstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        testcase.assertIsInstance(left, dict)
        testcase.assertIsInstance(right, dict)
        testcase.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(testcase, left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        testcase.assertIsInstance(left, type(right))
        testcase.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(testcase, left_item, right_item)
    else:
        testcase.assertEqual(left, right)


def _make_window(hidden=4):
    """Return the six-tensor ReplayV3 window contract."""
    x = torch.arange(16 * 12 * hidden, dtype=torch.float32).reshape(
        16, 12, hidden) / 1000.0
    schedule = torch.zeros(12, 16, 16, dtype=torch.float32)
    graph = torch.zeros(16, 12, hidden, dtype=torch.float32)
    ids = torch.arange(16, dtype=torch.int64).repeat(12, 1)
    before = (ids - 1).clamp_min(0)
    capacities = torch.ones(12, 16, 3, dtype=torch.float32)
    return x, schedule, graph, ids, before, capacities


def _make_observation(hidden=4, entropy=.9, margin=.05):
    return {
        "entropy": torch.full((1, 16, 12), entropy, dtype=torch.float32),
        "margin": torch.full((1, 16, 12), margin, dtype=torch.float32),
        "vectors": torch.ones((1, 16, 12, hidden), dtype=torch.float32),
    }


def _make_row():
    probability = np.linspace(.08, .92, 16, dtype=np.float64)
    classes = np.tile(np.asarray([.70, .20, .10], dtype=np.float64), (16, 1))
    return {
        "step": 1,
        "model_version": 5,
        "probability": probability.tolist(),
        "class_probability": classes.tolist(),
        "expert_count": 4,
        "mean_active": 2.0,
        "unmatched_ratio": .125,
    }


class _ReplayStub:
    def __init__(self, steps=4):
        self.steps = steps
        self.arrays = {
            "raw_labels": np.zeros((steps + 1, 16), dtype=np.int64),
            "capacities": np.ones((steps + 1, 16, 3), dtype=np.float32),
        }


class _TinyGate(nn.Module):
    """State-bearing gate sufficient for S8's causal helpers."""

    def __init__(self, hidden=4):
        super().__init__()
        self.cfg = AblationConfig(hidden=hidden, experts=4)
        self.shadow_parameter = nn.Parameter(torch.tensor(1.0))
        self.ids = ["0", "1", "2", "3"]
        self.next_id = 4
        self.shadow_id = None
        self.preview_id = None
        self.ramp_steps_done = {key: 10 for key in self.ids}
        self.activation_counts = {key: 0 for key in self.ids}
        self.weight_mass = {key: 0.0 for key in self.ids}
        self.routing_samples = 0
        self.unmatched_count = 0
        self.record_enabled = True
        self.last_routing = {"sentinel": "deployment"}
        self.last_observation = None
        self.context_capacity = None
        self.capacity_aware = False

    def shadow_parameters(self):
        return [self.shadow_parameter] if self.shadow_id is not None else []

    @contextmanager
    def preview(self):
        previous = self.preview_id
        self.preview_id = self.shadow_id
        try:
            yield
        finally:
            self.preview_id = previous

    def topology_state(self):
        return {
            "ids": list(self.ids),
            "next_id": self.next_id,
            "shadow_id": self.shadow_id,
            "ramp_steps_done": dict(self.ramp_steps_done),
            "activation_counts": dict(self.activation_counts),
            "weight_mass": dict(self.weight_mass),
            "routing_samples": self.routing_samples,
            "unmatched_count": self.unmatched_count,
        }

    def restore_topology(self, state):
        self.ids = list(state["ids"])
        self.next_id = int(state["next_id"])
        self.shadow_id = state.get("shadow_id")
        self.ramp_steps_done = dict(state.get("ramp_steps_done", {}))
        self.activation_counts = dict(state.get("activation_counts", {}))
        self.weight_mass = dict(state.get("weight_mass", {}))
        self.routing_samples = int(state.get("routing_samples", 0))
        self.unmatched_count = int(state.get("unmatched_count", 0))

    def reset_statistics(self):
        self.activation_counts = {key: 0 for key in self.ids}
        self.weight_mass = {key: 0.0 for key in self.ids}
        self.routing_samples = 0
        self.unmatched_count = 0


class _TinyDiagnosticModel(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.cfg = AblationConfig(hidden=hidden, experts=4)
        self.eagate = _TinyGate(hidden)
        self.routing_outputs = {"eagate": torch.tensor([7.0])}
        self.scale = nn.Parameter(torch.tensor(.15))
        self.forward_calls = 0

    def forward(self, x, schedule=None, graph=None, graph_context=None):
        self.forward_calls += 1
        batch, hosts, width, _ = x.shape
        value = self.scale + x[..., -1].mean(dim=2)
        detection = torch.stack((-value, value), dim=-1)
        class_value = value / 2
        classes = torch.stack((class_value, -class_value, value * 0), dim=-1)
        self.routing_outputs["eagate"] = torch.full(
            (batch, hosts, width, 2), float(self.scale.detach()))
        self.eagate.last_routing = {
            "expert_count": len(self.eagate.ids),
            "mean_active": 2.0,
            "unmatched_ratio": .0,
        }
        if self.eagate.record_enabled:
            self.eagate.last_observation = _make_observation(
                self.cfg.hidden, entropy=.9, margin=.05)
        return {"detection_logits": detection,
                "class_logits": classes}

    def auxiliary_losses(self, output):
        zero = output["detection_logits"].new_zeros(())
        return zero, zero

    @staticmethod
    def _hash_parameters(parameters):
        digest = hashlib.sha256()
        for name, parameter in sorted(parameters):
            digest.update(name.encode("utf8"))
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def state_hash(self):
        return self._hash_parameters(self.named_parameters())

    def frozen_hash(self):
        return self._hash_parameters(
            (name, parameter) for name, parameter in self.named_parameters()
            if not name.startswith("eagate."))

    def set_trainability(self):
        return None


class _RoutingHookModel(nn.Module):
    """Tiny end-to-end hook model whose forward overwrites routing_outputs."""

    def __init__(self):
        super().__init__()
        cfg = AblationConfig(hidden=64, experts=4)
        self.cfg = cfg
        self.eagate = OnlineEAGateV3(EAGateMoE(cfg), seed=3)
        self.routing_outputs = {"eagate": torch.tensor([99.0])}
        self.eagate.register_forward_hook(self._routing_hook)

    def _routing_hook(self, module, arguments, result):
        self.routing_outputs["eagate"] = result[1]

    def forward(self, x, schedule=None, graph=None, graph_context=None):
        resources = x.new_zeros((*x.shape[:-1], 3))
        result = self.eagate(x, resources)
        return {
            "detection_logits": result[3].mean(dim=2),
            "class_logits": result[4].mean(dim=2),
        }

    def auxiliary_losses(self, output):
        zero = output["detection_logits"].new_zeros(())
        return zero, zero


class _SessionHarness(S8Session):
    """Construct just the state used by the R0 methods under test."""

    def __init__(self, steps=4, hidden=4):
        self.dynamic_config = {
            "validation_protocol": "prequential_v1",
            "shadow_lr": .01,
            "calibration": {
                "novelty_threshold": .5,
                "loss_baseline": .8,
                "novelty_mean": .2,
                "entropy_p95": .7,
                "margin_p05": .3,
            },
            "shadow": {
                "validation_after_updates": 3,
                "training_updates": 10,
                "minimum_validation_records": 1,
            },
        }
        self.validation_protocol = "prequential_v1"
        self.learning_rate = .01
        self.method = "D"
        self.seed = 1
        self.replay_seed = 2
        self.model = _TinyDiagnosticModel(hidden)
        self.anchor_teacher = _TinyDiagnosticModel(hidden)
        self.replay = _ReplayStub(steps)
        self.detection_weights = [1.0, 2.0]
        self.resource_weights = [1.0, 1.5, 2.0]
        self.trigger = V3TriggerState()
        calibration = self.dynamic_config["calibration"]
        self.trigger.register_baselines(calibration["novelty_threshold"],
                                        calibration["loss_baseline"])
        self.trigger.novelty_ema = calibration["novelty_mean"]
        self.trigger.loss_ema = calibration["loss_baseline"]
        self.pending_observations = {}
        self.matured_observations = deque(maxlen=10)
        self.candidate_vectors = deque(maxlen=256)
        self.dynamic_events = []
        self.shadow_started = None
        self.shadow_training_steps = 0
        self.shadow_validation = []
        self.shadow_validation_indices = set()
        self.cooldown_until = 0
        self.topology_version = 0
        self.ramp_version = 0
        self.pending_predictions = {}
        self.qualification_records = []
        self.qualification_by_index = {}
        self.matured_indices = set()
        self.causal_audit = []
        self.audit_violation_count = 0
        self.diagnostic_cost = {
            "teacher_forward_count": 0,
            "teacher_forward_seconds": 0.0,
            "candidate_forward_count": 0,
            "candidate_forward_seconds": 0.0,
        }
        self.shadow_optimizer = None

        n = steps
        self.predictions = {
            "probability": np.zeros((n, 16), dtype=np.float32),
            "class_probability": np.zeros((n, 16, 3), dtype=np.float32),
            "labels": np.full((n, 16), -1, dtype=np.int64),
            "raw_labels": np.full((n, 16), -1, dtype=np.int64),
            "model_version": np.zeros(n, dtype=np.int64),
            "expert_count": np.zeros(n, dtype=np.int64),
            "mean_active": np.zeros(n, dtype=np.float32),
            "unmatched_ratio": np.zeros(n, dtype=np.float32),
            "prediction_seconds": np.zeros(n, dtype=np.float64),
        }
        self.raw_seen = np.full((n + 1, 16), -1, dtype=np.int64)
        self.buffer = deque(maxlen=128)
        self.exposure = {}
        self.rare_exposure = {}
        self.sample_rng = random.Random(17)
        self.cursor = 0
        self.update_number = 0
        self.updates = []
        self.reference = []
        self.finished = False
        self.initial_state_hash = self.model.state_hash()
        self.initial_frozen_hash = self.model.frozen_hash()
        # S8Session.make_optimizer excludes a shadow parameter and keeps a
        # small main parameter, which makes save/restore test the real parent
        # serialization path without constructing a production model.
        self.optimizer = S8Session.make_optimizer(self)

    def reset_observation(self):
        self.model.eagate.last_observation = _make_observation(
            self.model.cfg.hidden)

    def auxiliary_losses(self, output):
        return self.model.auxiliary_losses(output)


def _capture(harness, index, shadow=False):
    harness.model.eagate.shadow_id = "7" if shadow else None
    harness.model.eagate.last_observation = _make_observation(
        harness.model.cfg.hidden)
    if shadow:
        harness.shadow_started = 1
        harness.shadow_training_steps = 4
    return harness._capture_prediction_record(index, _make_row(),
                                              _make_window(harness.model.cfg.hidden))


class R0LossTests(unittest.TestCase):
    def _loss_case(self, labels):
        model = _SessionHarness()
        detection_logits = torch.tensor(
            [[[1.2, -0.2], [-.4, .8], [1.0, -.5], [.2, .4]]],
            dtype=torch.float64)
        class_logits = torch.tensor(
            [[[.3, -.2, .1], [.2, 1.0, -.7], [-.3, .4, .8],
              [.8, -.1, .2]]], dtype=torch.float64)
        output = {"detection_logits": detection_logits,
                  "class_logits": class_logits}
        score = {
            "probability": detection_logits.softmax(-1)[..., 1].numpy(),
            "class_probability": class_logits.softmax(-1).numpy(),
        }
        cached = model._cached_score_loss(score, labels.numpy())
        expected = s7_loss(
            model, output, labels, model.detection_weights,
            model.resource_weights, .7, .3, 0., 0., .5)
        self.assertAlmostEqual(cached["total"], float(expected),
                               delta=1e-10)
        self.assertEqual(cached["loss_formula"],
                         ".7*detectionCE+.3*positiveclassCE+.5*joint_ranking")
        self.assertFalse(cached["auxiliary_included"])

    def test_cached_score_loss_matches_s7_loss_all_normal(self):
        self._loss_case(torch.zeros((1, 4), dtype=torch.int64))

    def test_cached_score_loss_matches_s7_loss_mixed_anomalies(self):
        self._loss_case(torch.tensor([[0, 1, 2, 0]], dtype=torch.int64))

    def test_cached_score_loss_clips_probability_endpoints(self):
        harness = _SessionHarness()
        score = {
            "probability": np.asarray([0., 1., 0.5, 1.]),
            "class_probability": np.asarray([
                [1., 0., 0.], [0., 1., 0.],
                [0., 0., 1.], [0., 0., 1.],
            ]),
        }
        result = harness._cached_score_loss(
            score, np.asarray([0, 1, 2, 0], dtype=np.int64))
        self.assertTrue(np.isfinite(result["total"]))
        self.assertTrue(np.isfinite(result["detection_ce"]))
        self.assertTrue(np.isfinite(result["positiveclass_ce"]))
        self.assertTrue(np.isfinite(result["joint_ranking"]))


class R0CausalLedgerTests(unittest.TestCase):
    def test_candidate_preview_restores_gate_statistics_and_end_to_end_hook(self):
        harness = _SessionHarness(hidden=64)
        model = _RoutingHookModel()
        model.eagate.create_shadow(torch.ones(64))
        model.eagate.record_enabled = True
        model.eagate.routing_samples = 11
        model.eagate.activation_counts = {key: 3 for key in model.eagate.ids}
        model.eagate.weight_mass = {key: 1.25 for key in model.eagate.ids}
        model.eagate.last_observation = _make_observation(64)
        model.eagate.last_routing = {"sentinel": "deployment"}
        model.eagate.context_capacity = torch.full((1, 12, 16, 3), 2.0)
        model.eagate.preview_id = None
        model.routing_outputs = {"eagate": torch.tensor([99.0])}
        before_stats = {
            "routing_samples": model.eagate.routing_samples,
            "activation_counts": deepcopy(model.eagate.activation_counts),
            "weight_mass": deepcopy(model.eagate.weight_mass),
            "last_observation": deepcopy(model.eagate.last_observation),
            "last_routing": deepcopy(model.eagate.last_routing),
            "context_capacity": deepcopy(model.eagate.context_capacity),
            "preview_id": model.eagate.preview_id,
            "record_enabled": model.eagate.record_enabled,
            "routing_outputs": deepcopy(model.routing_outputs),
        }

        result = harness._diagnostic_output(
            model, _make_window(64), "candidate", candidate=True)
        self.assertEqual(len(result["probability"]), 16)
        self.assertEqual(len(result["class_probability"]), 16)
        self.assertEqual(model.eagate.routing_samples,
                         before_stats["routing_samples"])
        self.assertEqual(model.eagate.activation_counts,
                         before_stats["activation_counts"])
        self.assertEqual(model.eagate.weight_mass,
                         before_stats["weight_mass"])
        _assert_nested_equal(self, model.eagate.last_observation,
                             before_stats["last_observation"])
        _assert_nested_equal(self, model.eagate.last_routing,
                             before_stats["last_routing"])
        _assert_nested_equal(self, model.eagate.context_capacity,
                             before_stats["context_capacity"])
        self.assertEqual(model.eagate.preview_id, before_stats["preview_id"])
        self.assertEqual(model.eagate.record_enabled,
                         before_stats["record_enabled"])
        # FTMoEEndToEnd's hook writes this during forward; the diagnostic
        # context must restore the deployment hook value afterwards.
        _assert_nested_equal(self, model.routing_outputs,
                             before_stats["routing_outputs"])

    def test_cached_prediction_is_immutable_and_maturity_does_not_forward(self):
        harness = _SessionHarness()
        record = _capture(harness, 0, shadow=True)
        cached_candidate = deepcopy(record["candidate"])
        cached_deployment = deepcopy(record["deployment"])
        forward_calls = harness.model.forward_calls
        diagnostic_cost = deepcopy(harness.diagnostic_cost)
        labels = np.asarray([0, 1, 2, 0] * 4, dtype=np.int64)
        harness.predictions["labels"][0] = labels

        # A later model/weight mutation must not rewrite prediction-time
        # scores; maturity consumes the stored score dictionaries only.
        with torch.no_grad():
            harness.model.scale.add_(4.)
            harness.model.eagate.shadow_parameter.add_(2.)
        matured = harness.observe_matured(0)
        self.assertEqual(matured["candidate"], cached_candidate)
        self.assertEqual(matured["deployment"], cached_deployment)
        self.assertEqual(harness.model.forward_calls, forward_calls)
        self.assertEqual(harness.diagnostic_cost, diagnostic_cost)
        self.assertEqual(matured["matured_step"], 2)
        self.assertEqual(matured["maturity_raw_label_index"], 1)

        # The trigger remains the legacy unweighted binary BCE.  The
        # composite cached qualification loss is a separate decision metric.
        trigger_loss = harness._legacy_trigger_loss(cached_deployment, labels)
        self.assertAlmostEqual(harness.matured_observations[-1][1],
                               trigger_loss, delta=1e-12)
        self.assertNotAlmostEqual(
            trigger_loss, matured["losses"]["deployment"]["total"],
            delta=1e-6)

        # Repeated maturity is idempotent and cannot append a second ledger
        # row or perform a diagnostic forward.
        again = harness.observe_matured(0)
        self.assertIs(again, matured)
        self.assertEqual(len(harness.qualification_records), 1)
        self.assertEqual(harness.model.forward_calls, forward_calls)

    def test_candidate_has_no_prebirth_history_and_settles_before_reuse(self):
        harness = _SessionHarness()
        before_birth = _capture(harness, 0, shadow=False)
        after_birth = _capture(harness, 1, shadow=True)
        harness.predictions["labels"][0] = np.asarray(
            [0, 1, 2, 0] * 4, dtype=np.int64)
        harness.predictions["labels"][1] = np.asarray(
            [0, 2, 1, 0] * 4, dtype=np.int64)
        harness.observe_matured(0)
        harness.observe_matured(1)

        self.assertIsNone(before_birth["candidate"])
        self.assertIsNotNone(after_birth["candidate"])
        self.assertEqual(after_birth["candidate"]["id"], "7")
        self.assertTrue(after_birth["candidate"]["prediction_version"].startswith("7@"))
        self.assertEqual(after_birth["candidate"]["birth_step"], 1)

        blocks = harness._settle_shadow_validation()
        self.assertEqual(len(harness.qualification_records), 2)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["window_indices"], [1])
        self.assertEqual(blocks[0]["candidate_id"], "7")
        self.assertEqual(blocks[0]["candidate_prediction_versions"],
                         [after_birth["candidate"]["prediction_version"]])
        self.assertTrue(blocks[0]["validated_before_training"])
        self.assertTrue(harness.qualification_records[1]["qualification_blocks"])

        # The same settled window is allowed to be used by both target and
        # candidate training after qualification.  The test intentionally
        # permits validation/training overlap after this causal boundary.
        main_audit = harness._record_training_audit(
            "main", [1], 8, settled_before={1})
        candidate_audit = harness._record_training_audit(
            "candidate", [1], 4, settled_before={1}, candidate_id="7")
        self.assertEqual(main_audit["early_training_window_indices"], [])
        self.assertEqual(candidate_audit["early_training_window_indices"], [])
        self.assertEqual(harness.audit_violation_count, 0)
        self.assertEqual(
            [use["kind"] for use in harness.qualification_records[1]["training_uses"]],
            ["main", "candidate"])
        self.assertTrue(all(not use["before_qualification"]
                            for use in harness.qualification_records[1]["training_uses"]))

        # Future labels and capacities are not consulted again, and cannot
        # rewrite a non-empty candidate ledger or its qualification decision.
        ledger = deepcopy(harness.qualification_records)
        decision = deepcopy(harness.shadow_validation)
        harness.predictions["labels"][2] = 3
        harness.replay.arrays["raw_labels"][2:] = 3
        harness.replay.arrays["capacities"][2:] *= .01
        self.assertEqual(harness._settle_shadow_validation(), [])
        _assert_nested_equal(self, harness.qualification_records, ledger)
        _assert_nested_equal(self, harness.shadow_validation, decision)
        self.assertGreater(len(harness.qualification_records), 0)
        self.assertGreater(len(harness.shadow_validation), 0)

    def test_qualification_block_uses_one_cached_batch_loss(self):
        harness = _SessionHarness()
        _capture(harness, 0, shadow=True)
        _capture(harness, 1, shadow=True)
        harness.predictions["labels"][0] = np.asarray(
            [0, 1, 2, 0] * 4, dtype=np.int64)
        harness.predictions["labels"][1] = np.asarray(
            [2, 0, 0, 1] * 4, dtype=np.int64)
        harness.observe_matured(0)
        harness.observe_matured(1)
        block = harness._settle_shadow_validation()[0]
        records = harness.qualification_records
        expected = harness._cached_score_loss_block(records, "deployment")
        self.assertAlmostEqual(block["base_loss"], expected["total"],
                               delta=1e-12)
        expected_candidate = harness._cached_score_loss_block(records, "candidate")
        self.assertAlmostEqual(block["candidate_loss"],
                               expected_candidate["total"], delta=1e-12)
        self.assertEqual(block["window_indices"], [0, 1])


class R0LifecycleTests(unittest.TestCase):
    def test_finish_matures_last_prediction_once_without_training(self):
        harness = _SessionHarness(steps=2)
        _capture(harness, 1, shadow=False)
        # The guard interval and the two already-seen raw labels are known.
        # ``finish`` only supplies the final tolerance label; it must not
        # trigger an optimizer update or recompute a prediction.
        harness.raw_seen[:2] = 0
        harness.replay.arrays["raw_labels"][:] = 0
        harness.cursor = 2
        forward_calls = harness.model.forward_calls
        diagnostic_cost = deepcopy(harness.diagnostic_cost)
        update_count = len(harness.updates)

        harness.finish()
        self.assertTrue(harness.finished)
        self.assertEqual(len(harness.pending_predictions), 0)
        self.assertEqual(len(harness.qualification_records), 1)
        record = harness.qualification_records[0]
        self.assertEqual(record["window_index"], 1)
        self.assertEqual(record["matured_step"], 3)
        self.assertEqual(record["maturity_raw_label_index"], 2)
        self.assertTrue(record["label_matured"])
        self.assertEqual(len(harness.updates), update_count)

        # Calling finish again must be a no-op at both the raw-label and
        # qualification-ledger levels.
        record_identity = id(record)
        harness.finish()
        self.assertEqual(len(harness.qualification_records), 1)
        self.assertEqual(id(harness.qualification_records[0]), record_identity)
        self.assertEqual(harness.model.forward_calls, forward_calls)
        self.assertEqual(harness.diagnostic_cost, diagnostic_cost)
        self.assertEqual(len(harness.updates), update_count)

    @staticmethod
    def _seed_adam_state(optimizer, parameter):
        optimizer.state[parameter]["step"] = torch.tensor(2.0)
        optimizer.state[parameter]["exp_avg"] = torch.full_like(parameter, .25)
        optimizer.state[parameter]["exp_avg_sq"] = torch.full_like(parameter, .125)

    def _populate_dynamic_state(self, harness):
        harness.model.eagate.shadow_id = "7"
        harness.model.eagate.ramp_steps_done["7"] = 4
        # Birth removes the shadow parameters from the main optimizer's
        # membership; mirror the production sync boundary before serializing.
        harness.optimizer = S8Session.make_optimizer(harness)
        harness.shadow_started = 3
        harness.shadow_training_steps = 4
        harness.shadow_optimizer = torch.optim.AdamW(
            harness.model.eagate.shadow_parameters(), lr=.01,
            weight_decay=1e-4)
        self._seed_adam_state(harness.shadow_optimizer,
                              harness.model.eagate.shadow_parameter)
        record = {
            "window_index": 1,
            "prediction_step": 2,
            "candidate": {
                "id": "7", "prediction_version": "7@main2@shadow4",
                "training_steps_at_prediction": 4,
            },
            "labels": [0] * 16,
            "label_matured": True,
            "matured_step": 3,
            "losses": {"deployment": {"total": .8},
                        "candidate": {"total": .7}},
            "training_uses": [],
        }
        harness.pending_predictions = {
            2: {"window_index": 2, "prediction_step": 3,
                "validation_protocol": "prequential_v1"}
        }
        harness.qualification_records = [record]
        harness.qualification_by_index = {1: record}
        harness.matured_indices = {1}
        harness.matured_observations.extend([(0.8, .5)])
        harness.candidate_vectors.extend([torch.ones(4)])
        harness.shadow_validation = [{
            "window_indices": [1], "candidate_id": "7",
            "candidate_prediction_versions": ["7@main2@shadow4"],
            "base_loss": .8, "candidate_loss": .7,
            "validated_before_training": True,
        }]
        harness.shadow_validation_indices = {1}
        harness.dynamic_events = [{"step": 4, "decisions": ["shadow_created"]}]
        harness.causal_audit = [{"kind": "candidate", "window_indices": [1],
                                 "early_training_window_indices": []}]
        harness.audit_violation_count = 0
        harness.diagnostic_cost["teacher_forward_count"] = 2
        harness.exposure = {1: 2}
        harness.rare_exposure = {1: 3}
        harness.buffer.append(1)
        harness.trigger.novelty_windows_above = 3
        harness.trigger.candidate_samples = 144

    def test_active_shadow_save_restore_preserves_causal_state_and_optimizer(self):
        original = _SessionHarness()
        self._populate_dynamic_state(original)
        saved = deepcopy(original.save())

        resumed = _SessionHarness()
        resumed.restore(deepcopy(saved))
        self.assertEqual(original.model.state_hash(), resumed.model.state_hash())
        self.assertEqual(original.model.frozen_hash(), resumed.model.frozen_hash())
        self.assertEqual(resumed.validation_protocol, "prequential_v1")
        self.assertEqual(resumed.shadow_started, original.shadow_started)
        self.assertEqual(resumed.shadow_training_steps,
                         original.shadow_training_steps)
        self.assertEqual(resumed.matured_indices, original.matured_indices)
        self.assertEqual(resumed.shadow_validation_indices,
                         original.shadow_validation_indices)
        self.assertEqual(resumed.exposure, original.exposure)
        self.assertEqual(resumed.rare_exposure, original.rare_exposure)
        _assert_nested_equal(self, resumed.pending_predictions,
                             original.pending_predictions)
        _assert_nested_equal(self, resumed.qualification_records,
                             original.qualification_records)
        _assert_nested_equal(self, resumed.shadow_validation,
                             original.shadow_validation)
        _assert_nested_equal(self, resumed.causal_audit, original.causal_audit)
        _assert_nested_equal(self, resumed.shadow_optimizer.state_dict(),
                             original.shadow_optimizer.state_dict())

    def test_restore_without_shadow_optimizer_clears_stale_optimizer(self):
        source = _SessionHarness()
        source.model.eagate.shadow_id = None
        source.shadow_optimizer = None
        saved = deepcopy(source.save())
        self.assertIsNone(saved["dynamic"]["shadow_optimizer"])

        resumed = _SessionHarness()
        resumed.shadow_optimizer = torch.optim.AdamW(
            [resumed.model.scale], lr=.01)
        self.assertIsNotNone(resumed.shadow_optimizer)
        resumed.restore(saved)
        self.assertIsNone(resumed.shadow_optimizer)

    def test_prequential_restore_rejects_legacy_or_missing_dynamic_checkpoint(self):
        source = _SessionHarness()
        saved = deepcopy(source.save())
        saved["dynamic"]["validation_protocol"] = "legacy_v3"
        with self.assertRaisesRegex(ValueError, "validation_protocol"):
            _SessionHarness().restore(saved)
        with self.assertRaisesRegex(ValueError, "dynamic"):
            _SessionHarness().restore({"model": {}})


def _without_timing(value):
    """Remove wall-clock fields before comparing deterministic resume state."""
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items()
                if key not in ("seconds", "prediction_seconds")}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_timing(item) for item in value)
    return value


class R0RealFixtureTests(unittest.TestCase):
    """A short real stream slice using the registered S8 fixture only."""

    @classmethod
    def setUpClass(cls):
        # Reuse the original test's registered checkpoint/stream fixture, but
        # do not inherit its tests (several of those intentionally exercise
        # legacy behavior and would multiply the model load).
        s8_fixture_module.IntegrationTests.setUpClass()
        cls.fixture = s8_fixture_module.IntegrationTests

    def session(self):
        fixture = self.fixture
        replay = ReplayV3(
            deepcopy(fixture.arrays), fixture.scale,
            fixture.checkpoint["normalization"]["graph_scale"], 120)
        teacher = OnlineFTMoE(fixture.checkpoint, "A", 1)
        config = deepcopy(fixture.config)
        config["validation_protocol"] = "prequential_v1"
        session = S8Session(
            fixture.checkpoint, "D", 1, replay, 1e-4, 500,
            fixture.scale, fixture.pool, teacher, fixture.balance,
            dynamic_config=config)
        # Deployment prediction must record its routing observation so the
        # prequential prediction is paired with the same input window.
        session.model.eagate.record_enabled = True
        return session

    @staticmethod
    def start_shadow(session):
        gate = session.model.eagate
        key, parent = gate.create_shadow(gate.key_rows["0"].detach().clone())
        session.shadow_started = session.cursor
        session.shadow_training_steps = 0
        session.shadow_validation = []
        session.shadow_validation_indices = set()
        session.shadow_optimizer = torch.optim.AdamW(
            gate.shadow_parameters(), lr=session.dynamic_config.get(
                "shadow_lr", session.learning_rate), weight_decay=1e-4)
        return key, parent

    @staticmethod
    def ledger_projection(records):
        return _without_timing(deepcopy(records))

    def test_real_shadow_slice_has_qualification_and_exact_resume(self):
        original = self.session()
        key, _ = self.start_shadow(original)
        for _ in range(40):
            original.step()

        self.assertEqual(original.validation_protocol, "prequential_v1")
        self.assertEqual(original.model.eagate.shadow_id, key)
        self.assertGreater(len(original.qualification_records), 0)
        self.assertGreater(len(original.shadow_validation), 0)
        self.assertTrue(all(
            int(record["matured_step"]) == int(record["window_index"]) + 2
            for record in original.qualification_records))
        self.assertTrue(any(audit["kind"] == "candidate"
                            for audit in original.causal_audit))
        self.assertGreaterEqual(len(original.updates), 3)
        self.assertEqual(original.audit_violation_count, 0)
        self.assertTrue(all(block["validated_before_training"]
                            for block in original.shadow_validation))

        # Save with an active shadow and continue both copies for ten steps.
        saved = deepcopy(original.save())
        for _ in range(10):
            original.step()
        resumed = self.session()
        resumed.restore(deepcopy(saved))
        for _ in range(10):
            resumed.step()

        # A future-only label/capacity change must not affect this same
        # resumed slice: indices 50 onward are not in any window below 50.
        future = self.session()
        future.restore(deepcopy(saved))
        future.replay.arrays["raw_labels"][50:] = 3
        future.replay.arrays["capacities_per_interval"][50:] *= .01
        for _ in range(10):
            future.step()

        self.assertEqual(original.model.state_hash(), resumed.model.state_hash())
        self.assertEqual(original.model.state_hash(), future.model.state_hash())
        self.assertEqual(original.model.eagate.shadow_id,
                         resumed.model.eagate.shadow_id)
        self.assertEqual(original.exposure, resumed.exposure)
        self.assertEqual(original.exposure, future.exposure)
        self.assertEqual(original.rare_exposure, resumed.rare_exposure)
        self.assertEqual(original.rare_exposure, future.rare_exposure)
        _assert_nested_equal(
            self, self.ledger_projection(original.qualification_records),
            self.ledger_projection(resumed.qualification_records))
        _assert_nested_equal(self, self.ledger_projection(original.qualification_records),
                             self.ledger_projection(future.qualification_records))
        _assert_nested_equal(
            self, _without_timing(original.pending_predictions),
            _without_timing(resumed.pending_predictions))
        _assert_nested_equal(self,
                             _without_timing(original.pending_predictions),
                             _without_timing(future.pending_predictions))
        _assert_nested_equal(self, original.shadow_validation,
                             resumed.shadow_validation)
        _assert_nested_equal(self, original.shadow_validation,
                             future.shadow_validation)
        _assert_nested_equal(self, original.causal_audit, resumed.causal_audit)
        _assert_nested_equal(self, original.causal_audit, future.causal_audit)
        self.assertEqual(original.trigger.state(), resumed.trigger.state())
        self.assertEqual(original.trigger.state(), future.trigger.state())
        self.assertEqual(
            {key: original.diagnostic_cost[key] for key in original.diagnostic_cost
             if key.endswith("_count")},
            {key: resumed.diagnostic_cost[key] for key in resumed.diagnostic_cost
             if key.endswith("_count")})
        self.assertEqual(
            {key: original.diagnostic_cost[key] for key in original.diagnostic_cost
             if key.endswith("_count")},
            {key: future.diagnostic_cost[key] for key in future.diagnostic_cost
             if key.endswith("_count")})
        _assert_nested_equal(self, original.optimizer.state_dict(),
                             resumed.optimizer.state_dict())
        _assert_nested_equal(self, original.shadow_optimizer.state_dict(),
                             resumed.shadow_optimizer.state_dict())
