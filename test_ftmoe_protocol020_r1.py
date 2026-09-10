"""Targeted Protocol 020 R1 checks.

The integration fixture is intentionally imported as a module.  Importing the
fixture's ``TestCase`` directly would make unittest discover the S8 tests a
second time.  These tests are deliberately small in stream length; the one
assessment/resume check uses the registered 120-step fixture because the first
assessment is created after the tenth online update.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch

from run_ftmoe_protocol020 import ReplayV3
from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
import recovery.PreGANSrc.src.ftmoe_online_r1 as r1_module
from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE, R1Session
from recovery.PreGANSrc.src.ftmoe_online_s7 import S7Session, s7_loss
import test_ftmoe_protocol020_s8 as s8_fixture_module


def _state_hash(module):
    """Hash parameters and buffers, retaining names, shapes and dtypes."""
    digest = hashlib.sha256()
    values = list(module.named_parameters()) + list(module.named_buffers())
    for name, value in sorted(values):
        digest.update(name.encode("utf8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _without_timing(value):
    """Drop wall-clock fields while retaining all causal/session state."""
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items()
                if key not in {"seconds", "prediction_seconds",
                               "elapsed_seconds"}}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_timing(item) for item in value)
    return value


def _assert_nested_equal(testcase, left, right):
    """Exact comparison for checkpoint-shaped state."""
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


class R1IntegrationTests(unittest.TestCase):
    """R1's public session contract over the registered S8 fixture."""

    @classmethod
    def setUpClass(cls):
        # Reuse exactly the S8 fixture data and loader; no duplicate TestCase
        # import is intentional here.
        cls.fixture = s8_fixture_module.IntegrationTests
        cls.fixture.setUpClass()
        cls.checkpoint = cls.fixture.checkpoint
        # The S8 fixture normally keeps a 121-row slice.  R1's real
        # assessment regression needs the complete dev500 stream so that two
        # 100-window assessment blocks can mature causally.
        stream_path = (Path("artifacts") / "ftmoe_online" / "protocol_020" /
                       "drift_streams" / "dev_seed500_steps2000" /
                       "stream.npz")
        with np.load(stream_path) as data:
            cls.arrays = {
                key: data[key].copy()
                for key in data.files
                if data[key].ndim and data[key].shape[0] == 2001
            }
        cls.arrays["capacities_per_interval"] = cls.arrays.pop("capacities")
        cls.arrays["capacities"] = cls.arrays["capacities_per_interval"][0]
        if cls.arrays["capacities_per_interval"].shape[0] < 321:
            raise AssertionError("dev500 fixture is shorter than the R1 regression")
        cls.scale = cls.fixture.scale
        cls.pool = cls.fixture.pool
        cls.balance = cls.fixture.balance

    @classmethod
    def _arrays_for(cls, steps):
        """Return a guarded ``steps + 1`` replay slice."""
        arrays = {}
        for key, value in cls.arrays.items():
            if isinstance(value, np.ndarray) and value.ndim:
                if value.shape[0] == cls.arrays["capacities_per_interval"].shape[0]:
                    arrays[key] = value[:steps + 1].copy()
                else:
                    arrays[key] = value.copy()
        return arrays

    @classmethod
    def session(cls, protection_enabled=True, steps=40):
        replay = ReplayV3(
            cls._arrays_for(steps), cls.scale,
            cls.checkpoint["normalization"]["graph_scale"], steps)
        teacher = OnlineFTMoE(cls.checkpoint, "A", 1)
        return R1Session(
            cls.checkpoint, "C", 1, replay, 1e-4, 500,
            cls.scale, cls.pool, teacher, cls.balance,
            protection_enabled=protection_enabled)

    @classmethod
    def s7_a_session(cls, steps=40):
        replay = ReplayV3(
            cls._arrays_for(steps), cls.scale,
            cls.checkpoint["normalization"]["graph_scale"], steps)
        teacher = OnlineFTMoE(cls.checkpoint, "A", 1)
        return S7Session(
            cls.checkpoint, "A", 1, replay, 0., 500,
            cls.scale, cls.pool, teacher, cls.balance)

    @staticmethod
    def _window(session, index=0):
        return session._window(index)

    @staticmethod
    def _context(session, window):
        _, _, _, ids, before, caps = window
        return session._context(ids[None], before[None], caps[None])

    @classmethod
    def _forward(cls, session, index=0, which=None):
        """Run a selected R1 path on one registered window.

        R1 exposes its complete frozen-base path through ``_base_forward`` and
        its fixed assessment output through ``_prediction_bundle``.  Keeping
        these calls in one place makes snapshot checks explicit.
        """
        window = cls._window(session, index)
        x, schedule, graph, ids, before, caps = window
        context = cls._context(session, window)
        if which is None:
            return session.model(
                x[None], schedule[None], graph[None],
                graph_context=context)
        if which == "base":
            output, _ = session.model._base_forward(
                x[None], schedule[None], graph[None],
                graph_context=context, record=False)
            return output
        if which == "assessment":
            output = session.model._prediction_bundle(
                x[None], schedule[None], graph[None],
                graph_context=context, include_assessment=True, record=False)
            return {
                "detection_logits": output["r1_assessment_detection_logits"],
                "class_logits": output["r1_assessment_class_logits"],
            }
        raise ValueError("unknown R1 path: %s" % which)

    @staticmethod
    def _snapshot(session, name):
        snapshot = getattr(session.model, name)
        if snapshot is None:
            return None
        return snapshot

    @staticmethod
    def _learner_hash(session):
        return _state_hash(session.model.learner)

    @staticmethod
    def _base_hash(session):
        # ``frozen_hash`` covers every inherited parameter and buffer and
        # excludes only the four correction banks.
        return session.model.frozen_hash()

    @staticmethod
    def _prediction_records(session):
        for name in ("prediction_ledger", "ledger", "prediction_records",
                     "qualification_records"):
            records = getattr(session, name, None)
            if records is not None:
                return records
        raise AttributeError("R1 session has no prediction ledger")

    @staticmethod
    def _pending_records(session):
        for name in ("pending_predictions", "pending_records"):
            records = getattr(session, name, None)
            if records is not None:
                return records
        raise AttributeError("R1 session has no pending prediction records")

    def test_initial_on_and_off_predictions_equal_frozen_a(self):
        on = self.session(protection_enabled=True)
        off = self.session(protection_enabled=False)
        baseline = self.s7_a_session()
        self.assertEqual((on.active_parameter, on.alpha), ("live", 0.0))
        self.assertEqual((off.active_parameter, off.alpha), ("learner", 1.0))
        on_probability, on_classes = on.step()
        off_probability, off_classes = off.step()
        base_probability, base_classes = baseline.step()
        np.testing.assert_allclose(on_probability, base_probability,
                                   atol=1e-6, rtol=0)
        np.testing.assert_allclose(on_classes, base_classes,
                                   atol=1e-6, rtol=0)
        np.testing.assert_array_equal(on_probability, off_probability)
        np.testing.assert_array_equal(on_classes, off_classes)

    def test_base_parameters_and_buffers_and_output_stay_frozen(self):
        session = self.session(protection_enabled=False)
        baseline = self.s7_a_session()
        window = self._window(session)
        before_hash = self._base_hash(session)
        before_base = self._forward(session, which="base")
        before_a = self._forward(baseline)
        for _ in range(40):
            session.step()
        self.assertEqual(before_hash, self._base_hash(session))
        after_base = self._forward(session, which="base")
        for key in ("detection_logits", "class_logits"):
            torch.testing.assert_close(before_base[key], after_base[key],
                                       rtol=0, atol=0)
            torch.testing.assert_close(before_a[key], after_base[key],
                                       rtol=1e-6, atol=1e-6)
        # Guard against a nominally frozen model whose module buffers still
        # mutate through BatchNorm or running statistics.
        self.assertEqual(session.model.frozen_hash(), before_hash)
        self.assertEqual(window[0].shape[-1], 7)

    def test_zero_initialized_correction_has_first_update_gradient_and_moves(self):
        session = self.session(protection_enabled=False)
        session.step()
        session.step()  # index 0 is now mature; the correction heads are zero.
        indices = [0]
        x, schedule, graph, ids, before, caps = session._stack(indices)
        labels = torch.from_numpy(session.predictions["labels"][indices])
        output = session.model(
            x, schedule, graph,
            graph_context=session._context(ids, before, caps))
        loss = s7_loss(
            session.model, output, labels,
            session.detection_weights, session.resource_weights,
            .7, .3, 0., .01, .5)
        session.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in session.model.learner.parameters()]
        self.assertTrue(any(gradient is not None and
                            torch.isfinite(gradient).all() and
                            float(gradient.abs().sum()) > 0.
                            for gradient in gradients))

        session.optimizer.zero_grad(set_to_none=True)
        learner_before = self._learner_hash(session)
        for _ in range(8):
            session.step()
        self.assertTrue(session.updates)
        self.assertTrue(any(item.get("batches", 0) for item in session.updates))
        self.assertNotEqual(learner_before, self._learner_hash(session))

    def test_optimizer_contains_learner_only(self):
        session = self.session(protection_enabled=True)
        self.assertEqual(session.model.expert_count, 4)
        for bank_name in ("learner", "live", "assessment", "previous"):
            self.assertEqual(len(getattr(session.model, bank_name).experts), 4)
        learner_ids = {id(parameter)
                       for parameter in session.model.learner.parameters()}
        optimizer_ids = {id(parameter)
                         for group in session.optimizer.param_groups
                         for parameter in group["params"]}
        self.assertEqual(optimizer_ids, learner_ids)
        self.assertTrue(optimizer_ids)
        for name in ("live", "assessment", "previous"):
            snapshot = self._snapshot(session, name)
            self.assertIsNotNone(snapshot)
            self.assertFalse(any(id(parameter) in optimizer_ids
                                 for parameter in snapshot.parameters()))

    def test_s7_update_budget_and_maturation_are_causal_and_recorded(self):
        rows = []
        session = self.session(protection_enabled=False)
        for _ in range(40):
            session.step(rows.append)
        self.assertTrue(rows)
        self.assertTrue(session.updates)
        records = self._prediction_records(session)
        self.assertTrue(records)
        # Every S7 update uses the existing 24 recent + 8 anchor budget, with
        # shortfalls allowed only when the matured buffer is small.
        for update in session.updates:
            self.assertLessEqual(len(update.get("buffer_indices", [])), 24)
            self.assertLessEqual(len(update.get("anchor_indices", [])), 8)
            self.assertEqual(update.get("components", {}).get("recent_windows"),
                             len(update.get("buffer_indices", [])))
            for index in update.get("buffer_indices", []):
                self.assertLess(int(index), int(update["step"]) - 1)
        # A prediction exists before its label and mature records are appended
        # only after the one-step tolerance delay.
        self.assertTrue(all(int(row["step"]) >= 1 for row in rows))
        self.assertTrue(all(int(row["window_index"]) + 2 ==
                            int(row["matured_step"])
                            for row in records if row.get("label_matured")))
        self.assertTrue(any(
            isinstance(row.get("base"), dict)
            and "detection_logits" in row["base"]
            and "class_logits" in row["base"]
            and isinstance(row.get("deployment"), dict)
            and "detection_logits" in row["deployment"]
            for row in records))

    def test_update_loss_matches_s7_on_the_captured_raw_logit_batch(self):
        session = self.session(protection_enabled=False)
        captured = []
        original_forward = session.model.forward

        def capture_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            captured.append({key: value.detach().clone()
                             for key, value in output.items()
                             if isinstance(value, torch.Tensor)})
            return output

        # S7.step uses predict_online for deployment; the wrapper therefore
        # captures only the recent and anchor training batches in update().
        session.model.forward = capture_forward
        try:
            for _ in range(10):
                session.step()
        finally:
            session.model.forward = original_forward
        self.assertGreaterEqual(len(captured), 2)
        update = session.updates[-1]
        recent_indices = list(update["buffer_indices"])
        labels = torch.from_numpy(session.predictions["labels"][recent_indices])
        expected = s7_loss(
            session.model, captured[0], labels,
            session.detection_weights, session.resource_weights,
            .7, .3, 0., .01, .5)
        self.assertAlmostEqual(
            float(update["components"]["loss_online"]), float(expected), places=5)

    def test_assessment_prediction_is_fixed_while_learner_trains(self):
        session = self.session(protection_enabled=True, steps=120)
        for _ in range(110):
            session.step()
        assessment = self._snapshot(session, "assessment")
        self.assertIsNotNone(assessment)
        records = self._prediction_records(session)
        self.assertTrue(records)
        self.assertTrue(any(record.get("assessment") is not None
                            or record.get("assessment_logits") is not None
                            for record in records))
        before_state = deepcopy(assessment.state_dict())
        before_output = self._forward(session, index=100, which="assessment")
        learner_before = self._learner_hash(session)
        for _ in range(10):
            session.step()
        after_output = self._forward(session, index=100, which="assessment")
        _assert_nested_equal(self, before_state, assessment.state_dict())
        for key in ("detection_logits", "class_logits"):
            torch.testing.assert_close(before_output[key], after_output[key],
                                       rtol=0, atol=0)
        self.assertNotEqual(learner_before, self._learner_hash(session))

    def test_real_dev500_assessment_blocks_flatten_raw_logits_for_s7_loss(self):
        """Run the registered dev500 stream through two real assessment blocks."""
        session = self.session(protection_enabled=True, steps=320)
        for _ in range(320):
            session.step()
        session.finish()

        events = [event for event in session.r1_events
                  if event.get("event") == "assessment_evaluated"]
        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(any(
            int(event["candidate"]["n_positive"]) > 0 for event in events))
        records = {
            int(record["window_index"]): record
            for record in self._prediction_records(session)
        }

        for event in events:
            block = [records[int(index)] for index in event["window_indices"]]
            detection_parts = []
            class_parts = []
            label_parts = []
            for record in block:
                score = record["assessment"]
                detection = torch.as_tensor(
                    score["detection_logits"], dtype=torch.float32
                ).reshape(-1, 2)
                classification = torch.as_tensor(
                    score["class_logits"], dtype=torch.float32
                ).reshape(-1, 3)
                labels = torch.as_tensor(
                    record["labels"], dtype=torch.long
                ).reshape(-1)
                # Assert the per-window host axis before flattening the block;
                # this catches [windows, hosts, classes] versus [hosts]
                # concatenation errors directly.
                self.assertEqual(detection.shape[0], classification.shape[0])
                self.assertEqual(detection.shape[0], labels.numel())
                detection_parts.append(detection)
                class_parts.append(classification)
                label_parts.append(labels)

            detection = torch.cat(detection_parts, dim=0)
            classification = torch.cat(class_parts, dim=0)
            labels = torch.cat(label_parts, dim=0)
            output = {
                "detection_logits": detection,
                "class_logits": classification,
                "router_probabilities": None,
                "correction_router_probabilities": None,
            }
            score = s7_loss(
                session.model, output, labels,
                session.detection_weights, session.resource_weights,
                .7, .3, 0., 0., .5)
            np.testing.assert_allclose(
                float(score), float(event["candidate"]["loss"]),
                rtol=1e-5, atol=1e-6)

    def test_assessment_boundary_keeps_current_prediction_on_old_snapshot(self):
        session = self.session(protection_enabled=True)
        session.step()
        session.step()
        old_support = r1_module.ASSESSMENT_SUPPORT
        try:
            # This is only a test-local shortening of the registered 100-window
            # support.  It exercises the same step boundary with real logits.
            r1_module.ASSESSMENT_SUPPORT = 2
            created = session._start_assessment()
            old_id = created["snapshot_id"]
            for _ in range(5):
                session.step()
        finally:
            r1_module.ASSESSMENT_SUPPORT = old_support
        records = {int(record["window_index"]): record
                   for record in self._prediction_records(session)}
        self.assertEqual(records[2]["assessment_snapshot_id"], old_id)
        self.assertEqual(records[3]["assessment_snapshot_id"], old_id)
        # The boundary prediction is made before the current label is settled,
        # so it still belongs to the old snapshot.  A following prediction gets
        # the replacement snapshot.
        self.assertEqual(records[4]["assessment_snapshot_id"], old_id)
        self.assertEqual(records[4]["assessment_excluded_reason"],
                         "snapshot_closed_before_maturity")
        self.assertNotEqual(records[5]["assessment_snapshot_id"], old_id)
        self.assertTrue(all(record["assessment_snapshot_id"] != old_id
                            for record in session.assessment_block_records))
        self.assertTrue(any(event.get("event") == "assessment_evaluated"
                            for event in session.r1_events))

    def test_live_protection_rechecks_each_support_block(self):
        session = self.session(protection_enabled=True)
        session._set_deployment("live", 1.0, reason="test_live")
        session.live_version = 1
        labels = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        base_det = torch.tensor([[[4., -4.], [-4., 4.],
                                 [-4., 4.], [-4., 4.]]])
        base_cls = torch.tensor([[[4., 0., 0.], [4., 0., 0.],
                                  [0., 4., 0.], [0., 0., 4.]]])
        bad_det = torch.tensor([[[-4., 4.], [4., -4.],
                                [4., -4.], [4., -4.]]])

        def score(det, cls):
            return {
                "detection_logits": det.numpy().copy(),
                "class_logits": cls.numpy().copy(),
                # The recorded R1 schema stores the anomaly probability,
                # rather than the two-class softmax tensor.
                "probability": det.softmax(-1)[..., 1].numpy().copy(),
                "class_probability": cls.softmax(-1).numpy().copy(),
            }

        def record(index, live_det):
            return {
                "window_index": index,
                "prediction_step": index + 1,
                "matured_step": index + 2,
                "live_version": 1,
                "deployment_parameter": "live",
                "deployment_alpha": 1.0,
                "deployment_version": int(session.deployment_version),
                "labels": labels.tolist(),
                "base": score(base_det, base_cls),
                "live": score(live_det, base_cls),
            }

        old_support = r1_module.LIVE_SUPPORT
        try:
            r1_module.LIVE_SUPPORT = 2
            first = [session._maybe_rollback(record(i, base_det))
                     for i in range(2)]
            self.assertIsNotNone(first[-1])
            self.assertEqual(first[-1]["decision"], "live_retained")
            second = [session._maybe_rollback(record(i, bad_det))
                      for i in range(2, 4)]
        finally:
            r1_module.LIVE_SUPPORT = old_support
        self.assertIsNotNone(second[-1])
        self.assertEqual(second[-1]["decision"], "rollback_to_frozen_A")
        self.assertEqual(session.alpha, 0.0)

    def test_live_policy_version_excludes_old_evidence(self):
        session = self.session(protection_enabled=True)
        session._set_deployment("live", 1.0, reason="test_live")
        session.live_version = 1
        labels = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        base_det = torch.tensor([[[4., -4.], [-4., 4.],
                                 [-4., 4.], [-4., 4.]]])
        base_cls = torch.tensor([[[4., 0., 0.], [4., 0., 0.],
                                  [0., 4., 0.], [0., 0., 4.]]])
        bad_det = torch.tensor([[[-4., 4.], [4., -4.],
                                [4., -4.], [4., -4.]]])

        def score(det, cls):
            return {
                "detection_logits": det.numpy().copy(),
                "class_logits": cls.numpy().copy(),
                "probability": det.softmax(-1)[..., 1].numpy().copy(),
                "class_probability": cls.softmax(-1).numpy().copy(),
            }

        def record(index, deployment_version):
            return {
                "window_index": index,
                "prediction_step": index + 1,
                "matured_step": index + 2,
                "live_version": 1,
                "deployment_parameter": "live",
                "deployment_alpha": 1.0,
                "deployment_version": deployment_version,
                "labels": labels.tolist(),
                "base": score(base_det, base_cls),
                "live": score(bad_det, base_cls),
            }

        old_support = r1_module.LIVE_SUPPORT
        try:
            r1_module.LIVE_SUPPORT = 2
            old_policy = int(session.deployment_version)
            # One old-policy record must not combine with one new-policy
            # record to reach the support threshold.
            session._maybe_rollback(record(0, old_policy))
            session._set_deployment(
                "live", 1.0, reason="test_new_live_policy",
                force_new_policy=True)
            new_policy = int(session.deployment_version)
            self.assertGreater(new_policy, old_policy)
            self.assertIsNone(session._maybe_rollback(record(1, new_policy)))
            self.assertFalse(session.deployment_evidence_history)
            event = session._maybe_rollback(record(2, new_policy))
        finally:
            r1_module.LIVE_SUPPORT = old_support
        self.assertIsNotNone(event)
        self.assertEqual(event["decision"], "rollback_to_frozen_A")
        self.assertEqual(event["deployment_version"], new_policy)
        self.assertEqual(event["support"]["first_window_index"], 1)
        self.assertEqual(event["support"]["last_window_index"], 2)
        self.assertEqual(len(session.deployment_evidence_history[0]["records"]), 2)

    def test_future_labels_and_capacities_do_not_change_history_or_decisions(self):
        original = self.session(protection_enabled=True)
        future = self.session(protection_enabled=True)
        # Perturb the unread suffix before either session starts.  Any
        # difference before index 20 would prove that the replay path looked
        # ahead through labels, capacities, or the guard interval.
        future.replay.arrays["raw_labels"][20:] = 3
        future.replay.arrays["capacities_per_interval"][20:] *= .01
        for _ in range(20):
            original.step()
            future.step()
        self.assertEqual(original.exposure, future.exposure)
        self.assertEqual(original.sample_rng.getstate(),
                         future.sample_rng.getstate())
        self.assertEqual(_without_timing(original.updates),
                         _without_timing(future.updates))
        self.assertEqual(_without_timing(original.r1_events),
                         _without_timing(future.r1_events))
        np.testing.assert_array_equal(
            original.predictions["probability"][:20],
            future.predictions["probability"][:20])
        np.testing.assert_array_equal(
            original.predictions["class_probability"][:20],
            future.predictions["class_probability"][:20])
        self.assertEqual(_without_timing(self._prediction_records(original)),
                         _without_timing(self._prediction_records(future)))

        # Once the changed suffix is actually read, later adaptation may
        # diverge.  Historical arrays and records must remain immutable.
        for _ in range(20):
            original.step()
            future.step()
        np.testing.assert_array_equal(
            original.predictions["probability"][:20],
            future.predictions["probability"][:20])
        np.testing.assert_array_equal(
            original.predictions["class_probability"][:20],
            future.predictions["class_probability"][:20])
        self.assertEqual(_without_timing(original.updates[:1]),
                         _without_timing(future.updates[:1]))
        self.assertEqual(_without_timing(self._prediction_records(original)[:10]),
                         _without_timing(self._prediction_records(future)[:10]))

    def test_sampling_and_update_budget_on_off_match_with_same_seed(self):
        on = self.session(protection_enabled=True)
        off = self.session(protection_enabled=False)
        for _ in range(40):
            on.step()
            off.step()
        self.assertEqual(on.exposure, off.exposure)
        self.assertEqual(on.sample_rng.getstate(), off.sample_rng.getstate())
        self.assertEqual(
            _without_timing(on.updates), _without_timing(off.updates))
        self.assertEqual(on.model.learner_state_hash(),
                         off.model.learner_state_hash())
        _assert_nested_equal(self, on.optimizer.state_dict(),
                             off.optimizer.state_dict())

    def test_save_restore_after_assessment_evidence_is_exact(self):
        original = self.session(protection_enabled=True, steps=120)
        for _ in range(110):
            original.step()
        self.assertIsNotNone(self._snapshot(original, "assessment"))
        self.assertTrue(self._prediction_records(original))
        saved = deepcopy(original.save())
        self.assertEqual(saved["r1"]["events"], original.r1_events)
        self.assertEqual(saved["r1"]["update_events"], original.r1_update_events)
        for _ in range(10):
            original.step()
        original.finish()

        resumed = self.session(protection_enabled=True, steps=120)
        resumed.restore(deepcopy(saved))
        for _ in range(10):
            resumed.step()
        resumed.finish()
        resumed.finish()

        self.assertEqual(original.model.learner_state_hash(),
                         resumed.model.learner_state_hash())
        self.assertEqual(original.model.frozen_hash(), resumed.model.frozen_hash())
        _assert_nested_equal(self, original.optimizer.state_dict(),
                             resumed.optimizer.state_dict())
        self.assertEqual(original.exposure, resumed.exposure)
        self.assertEqual(original.sample_rng.getstate(), resumed.sample_rng.getstate())
        np.testing.assert_array_equal(original.predictions["probability"],
                                      resumed.predictions["probability"])
        np.testing.assert_array_equal(original.predictions["class_probability"],
                                      resumed.predictions["class_probability"])
        self.assertEqual(_without_timing(original.updates),
                         _without_timing(resumed.updates))
        self.assertEqual(_without_timing(self._prediction_records(original)),
                         _without_timing(self._prediction_records(resumed)))
        self.assertEqual(_without_timing(original.r1_events),
                         _without_timing(resumed.r1_events))
        self.assertEqual(_without_timing(original.r1_update_events),
                         _without_timing(resumed.r1_update_events))

    def test_finish_matures_last_record_once_and_is_idempotent(self):
        session = self.session(protection_enabled=True)
        for _ in range(40):
            session.step()
        before = deepcopy(session.save())
        session.finish()
        self.assertTrue(session.finished)
        self.assertTrue((session.predictions["labels"][-1] >= 0).all())
        after_first = deepcopy(session.save())
        session.finish()
        after_second = deepcopy(session.save())
        _assert_nested_equal(self, _without_timing(after_first),
                             _without_timing(after_second))
        self.assertEqual(before["cursor"], after_first["cursor"])

    def test_protection_reject_accept_and_rollback_use_recorded_logits(self):
        session = self.session(protection_enabled=True)
        labels = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        base_det = torch.tensor([[[4., -4.], [-4., 4.],
                                 [-4., 4.], [-4., 4.]]])
        base_cls = torch.tensor([[[4., 0., 0.], [4., 0., 0.],
                                  [0., 4., 0.], [0., 0., 4.]]])
        good_det = torch.tensor([[[4., -4.], [-4., 4.],
                                 [-4., 4.], [-4., 4.]]])
        good_cls = base_cls.clone()
        bad_det = torch.tensor([[[-4., 4.], [4., -4.],
                                [4., -4.], [4., -4.]]])
        bad_cls = base_cls.clone()
        # A less confident but still correctly classified candidate exercises
        # the explicit candidate_loss <= base_loss * 1.02 guard.
        weak_det = base_det / 2.
        weak_cls = base_cls / 2.

        def score(det, cls):
            return {
                "detection_logits": det.detach().numpy().copy(),
                "class_logits": cls.detach().numpy().copy(),
                "probability": det.softmax(-1)[..., 1].numpy().copy(),
                "class_probability": cls.softmax(-1).numpy().copy(),
            }

        def row(assessment, live, version, index=0):
            return {
                "window_index": int(index),
                "prediction_step": int(index + 1),
                "matured_step": int(index + 2),
                "live_version": int(version),
                "labels": labels.tolist(),
                "base": score(base_det, base_cls),
                "live": score(live[0], live[1]),
                "assessment": score(assessment[0], assessment[1]),
            }

        # Decisions must consume recorded logits.  There is no model forward
        # in _block_event; its result is a pure score over the supplied rows.
        accepted_event, _, _ = session._block_event(
            [row((good_det, good_cls), (bad_det, bad_cls), 1)],
            "assessment-v1", "recorded")
        self.assertTrue(accepted_event["accepted_by_rules"])
        self.assertLess(accepted_event["candidate"]["loss"],
                        accepted_event["live"]["loss"] * .995)
        self.assertLessEqual(accepted_event["candidate"]["loss"],
                             accepted_event["base"]["loss"] * 1.02)
        session._apply_assessment_decision(accepted_event)
        self.assertEqual(session.active_parameter, "live")
        self.assertEqual(session.alpha, 1.0)

        rejected_event, _, _ = session._block_event(
            [row((bad_det, bad_cls), (bad_det, bad_cls), 1)],
            "assessment-v2", "recorded")
        self.assertFalse(rejected_event["accepted_by_rules"])
        session._apply_assessment_decision(rejected_event)
        self.assertEqual(rejected_event["decision"], "rejected_live_unchanged")
        self.assertEqual(session.live_version, 1)

        boundary_event, _, _ = session._block_event(
            [row((weak_det, weak_cls), (bad_det, bad_cls), 1)],
            "assessment-v3", "recorded")
        self.assertFalse(boundary_event["accepted_by_rules"])
        self.assertTrue(any("base" in reason
                            for reason in boundary_event["reasons"]))

        # Mutating the replay after recording cannot change a decision or
        # rewrite the cached logits.
        recorded = deepcopy(row((good_det, good_cls), (bad_det, bad_cls), 1))
        accepted_before, _, _ = session._block_event(
            [recorded], "assessment-v4", "recorded")
        session.replay.arrays["raw_labels"][:] = 3
        session.replay.arrays["capacities_per_interval"][:] *= .01
        accepted_after, _, _ = session._block_event(
            [recorded], "assessment-v4", "recorded")
        self.assertEqual(_without_timing(accepted_before),
                         _without_timing(accepted_after))
        self.assertTrue(np.array_equal(
            recorded["base"]["detection_logits"],
            base_det.numpy()))

        # A bad live version over its recorded 100-window support must roll
        # back to frozen A while preserving the failed evidence.
        session.deployment_evidence = {1: []}
        session.evidence_checked_versions = set()
        rollback_event = None
        for index in range(100):
            rollback_event = session._maybe_rollback(
                row((bad_det, bad_cls), (bad_det, bad_cls), 1, index=index))
        self.assertIsNotNone(rollback_event)
        self.assertEqual(rollback_event["decision"], "rollback_to_frozen_A")
        self.assertEqual(session.alpha, 0.0)
        self.assertTrue(session.deployment_evidence_history)
        self.assertEqual(len(session.deployment_evidence_history[0]["records"]),
                         100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
