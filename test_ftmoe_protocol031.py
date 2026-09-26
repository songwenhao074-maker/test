import json, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

import prepare_ftmoe_protocol031_stream as p31d
from ftmoe_protocol031_nonblocking_reuse import (
    Protocol031NonblockingReuseLifecycle, P031_CONFIG,
)
from run_ftmoe_protocol031_pilot import json_ready

class DummyBank:
    def __init__(self):
        self.dormant_experts={"9":object(),"15":object()}
        self.max_experts=8
        self.shadow_id=None

class DummyModel:
    def __init__(self):
        self.learner=DummyBank()

class DummySession:
    def __init__(self):
        self.model=DummyModel()
        self.cursor=1000
        self.lifecycle_state={}
        self.lifecycle_events=[]

class Protocol031Tests(unittest.TestCase):
    def test_revision003_registration_and_timeline(self):
        reg=p31d.registration()
        self.assertEqual(reg["protocol"],"031")
        self.assertEqual(reg["plan_revision"],3)
        self.assertEqual(reg["scored_intervals"],9868)
        self.assertEqual(reg["guard_intervals"],1)
        phases=p31d.phase_table(reg)
        self.assertEqual(phases[0]["name"],"F0")
        self.assertEqual(phases[-1]["name"],"V_rec3")
        self.assertEqual(phases[-1]["end"],9868)
        self.assertEqual([p["name"] for p in phases if "_rec" in p["name"]],
                         ["U_rec1","V_rec1","U_rec2","V_rec2","U_rec3","V_rec3"])
        self.assertEqual(p31d.SOURCE_SERVICE,{"U":"S1","V":"S3","W":"S4"})
        self.assertEqual(reg["generation"]["chunk_intervals"],200)
        self.assertEqual(reg["generation"]["full_stream_budget"],1)

    def test_low_similarity_is_logged_not_admission_gate(self):
        ctrl=Protocol031NonblockingReuseLifecycle(guard_anchor={},config=P031_CONFIG)
        ctrl.matured_count=640
        ctrl.phase="candidate_training"
        ctrl.active_specialist_id="5"
        ctrl.recent_z=[np.asarray([1.0,0.0],dtype=np.float64) for _ in range(32)]
        ctrl.specialist_memory={
            "9":{"centroid":np.asarray([0.0,1.0]),"similarity_threshold":0.99,
                 "last_causal_accept_matured":100},
            "15":{"centroid":np.asarray([-1.0,0.0]),"similarity_threshold":0.99,
                  "last_causal_accept_matured":200},
        }
        session=DummySession()
        started=ctrl._start_pending_reuse(session)
        self.assertTrue(started)
        self.assertEqual(ctrl.pending_reuse["expert_id"],"9")
        self.assertLess(ctrl.pending_reuse["selection_similarity"],
                        ctrl.pending_reuse["logged_similarity_threshold"])
        self.assertFalse(ctrl.pending_reuse["similarity_threshold_used_as_gate"])
        self.assertEqual(ctrl.phase,"candidate_training")

    def test_pending_cancel_does_not_change_birth_phase(self):
        ctrl=Protocol031NonblockingReuseLifecycle(guard_anchor={},config=P031_CONFIG)
        ctrl.matured_count=700
        ctrl.phase="candidate_validation"
        ctrl.pending_reuse={
            "expert_id":"9","started_cursor":100,"started_matured":640,
            "outcome":"pending"
        }
        session=DummySession()
        ctrl._cancel_pending(session,"unit_test_topology_change")
        self.assertIsNone(ctrl.pending_reuse)
        self.assertEqual(ctrl.phase,"candidate_validation")
        self.assertEqual(ctrl.reuse_outcomes["cancelled"],1)

    def test_reuse_due_and_started_conservation(self):
        ctrl=Protocol031NonblockingReuseLifecycle(guard_anchor={},config=P031_CONFIG)
        ctrl.reuse_due_accounting.update({
            "due":6,"started":3,"skip_pending":1,"skip_transition":1,
            "skip_no_memory":1,"skip_insufficient_z":0,
        })
        ctrl.reuse_outcomes.update({
            "accepted":1,"rejected":1,"cancelled":1,"censored":0,"pending":0,
        })
        report=ctrl.due_conservation()["reuse_nonblocking"]
        self.assertTrue(report["due_conserved"])
        self.assertTrue(report["started_conserved"])

    def test_json_ready_handles_numpy_without_default_str(self):
        value={
            "i":np.int64(7),"f":np.float32(0.25),
            "a":np.asarray([1,2],dtype=np.int64),
            "nan":float("nan"),
        }
        ready=json_ready(value)
        self.assertEqual(ready["i"],7)
        self.assertAlmostEqual(ready["f"],0.25)
        self.assertEqual(ready["a"],[1,2])
        self.assertIsNone(ready["nan"])
        json.dumps(ready,allow_nan=False)

    def test_method_registration_freezes_two_model_replays(self):
        reg=json.loads(Path("artifacts/ftmoe_online/protocol_031/method_registration.json").read_text())
        self.assertEqual(reg["full_replay_budget"],2)
        self.assertEqual(reg["arms"],["C_fixed5","D_nonblocking_reuse"])
        self.assertEqual(reg["C"]["online_update_every_scored_intervals"],16)
        self.assertEqual(reg["D"]["online_update_every_scored_intervals"],16)
        self.assertEqual(reg["D"]["birth_every_matured"],1600)
        self.assertEqual(reg["D"]["reuse_every_matured"],32)
        self.assertEqual(reg["D"]["reuse_validation_future_intervals"],16)
        self.assertEqual(reg["D"]["similarity_threshold_role"],
                         "logged_only_not_admission_gate")

if __name__=="__main__":
    unittest.main(verbosity=2)
