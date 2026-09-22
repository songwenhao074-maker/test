import json, random, unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn

from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol030_paired import (
    Protocol030PassiveAuditLifecycle, algorithm_state_summary
)

class Expert(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(16),nn.Linear(16,12),nn.GELU(),nn.Linear(12,5))

class Fixed(nn.Module):
    def __init__(self):
        super().__init__(); self.router=nn.Linear(16,4); self.experts=nn.ModuleList([Expert() for _ in range(4)])

class Model(nn.Module):
    def __init__(self,bank,z):
        super().__init__(); self.learner=bank; self._last_z=z

class Session:
    def __init__(self,bank,z):
        self.model=Model(bank,z)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        self.optimizer_archive={}; self.shadow_optimizer=None; self.shadow_optimizer_state=None
        self.raw_seen=np.full((1,16),-1,dtype=np.int64); self.cursor=0; self.updates=0

class Protocol030Tests(unittest.TestCase):
    def test_passive_prediction_preserves_content_state(self):
        torch.manual_seed(30030); np.random.seed(30030); random.seed(30030)
        bank=DynamicResidualBank(Fixed(),max_experts=8,ramp_updates=8)
        key=bank.create_shadow("0"); bank.activate_shadow(); bank.retire(key)
        z=torch.randn(1,16,16); s=Session(bank,z)
        ctrl=Protocol030PassiveAuditLifecycle(guard_anchor={})
        ctrl.specialist_memory={key:{
            "centroid":np.ones(16,dtype=np.float64)/4.0,
            "similarity_threshold":-1.0,"last_causal_accept_matured":10,
        }}
        s.lifecycle_controller=ctrl
        before=algorithm_state_summary(s)
        out={"base_final_detection_logits":torch.zeros(1,16,2),
             "base_final_class_logits":torch.zeros(1,16,3)}
        ctrl._passive_prediction(s,0,out)
        after=algorithm_state_summary(s)
        self.assertEqual(before,after)
        self.assertEqual(ctrl.passive_state_check_failures,0)

    def test_registration_freezes_two_replays_and_original_tolerance(self):
        reg=json.loads(Path("artifacts/ftmoe_online/protocol_030/registration.json").read_text())
        self.assertEqual(reg["protocol"],"030")
        self.assertEqual(reg["full_replay_budget"],2)
        self.assertFalse(reg["rerun_C"])
        self.assertEqual(reg["new_scientific_methods"],0)
        self.assertEqual(reg["paired_requirements"]["max_detection_probability_error"],1e-6)
        self.assertEqual(reg["paired_requirements"]["max_class_probability_error"],1e-6)
        self.assertEqual(reg["runtime_profile"]["OMP_NUM_THREADS"],1)
        self.assertTrue(reg["runtime_profile"]["torch_deterministic_algorithms"])

if __name__=="__main__":
    unittest.main(verbosity=2)
