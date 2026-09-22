"""Protocol-030 same-job paired passive-diagnostic validation helpers."""
from __future__ import annotations
from copy import deepcopy
import hashlib, json, os, random
import numpy as np
import torch

from ftmoe_protocol024_v2c import V2C_DEFAULT
from ftmoe_protocol029_memory_utility import (
    Protocol029DiagnosticSession,
    Protocol029PassiveAuditLifecycle,
)

def set_deterministic_runtime():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }

def _update_hash(h, value):
    if torch.is_tensor(value):
        x=value.detach().cpu().contiguous()
        h.update(b"T"); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes()); return
    if isinstance(value, np.ndarray):
        x=np.ascontiguousarray(value)
        h.update(b"N"); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
        h.update(x.tobytes()); return
    if isinstance(value, dict):
        h.update(b"D")
        for k in sorted(value, key=lambda z: str(z)):
            h.update(str(k).encode()); _update_hash(h, value[k])
        return
    if isinstance(value, (list, tuple)):
        h.update(b"L")
        for x in value: _update_hash(h, x)
        return
    if isinstance(value, set):
        h.update(b"S")
        for x in sorted(value, key=lambda z: str(z)): _update_hash(h, x)
        return
    h.update(repr(value).encode())

def stable_digest(value):
    h=hashlib.sha256(); _update_hash(h, value); return h.hexdigest()

def rng_snapshot():
    ns=np.random.get_state()
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": (ns[0], ns[1].copy(), ns[2], ns[3], ns[4]),
        "python": random.getstate(),
    }

def module_modes(model):
    return {name: bool(module.training) for name,module in model.named_modules()}

def requires_grad_map(model):
    return {name: bool(p.requires_grad) for name,p in model.named_parameters()}

def optimizer_payload(session):
    payload={
        "live": None if session.optimizer is None else session.optimizer.state_dict(),
        "archive": getattr(session, "optimizer_archive", {}),
        "shadow_state": getattr(session, "shadow_optimizer_state", None),
    }
    shadow=getattr(session, "shadow_optimizer", None)
    payload["shadow"] = None if shadow is None else shadow.state_dict()
    return payload

def algorithm_state_summary(session):
    ctrl=session.lifecycle_controller
    last_z=getattr(session.model, "_last_z", None)
    return {
        "model_state_sha256": stable_digest(session.model.state_dict()),
        "optimizer_state_sha256": stable_digest(optimizer_payload(session)),
        "controller_state_sha256": stable_digest(ctrl.state_dict()),
        "rng_state_sha256": stable_digest(rng_snapshot()),
        "topology": deepcopy(session.model.learner.topology_manifest()),
        "module_modes": module_modes(session.model),
        "requires_grad": requires_grad_map(session.model),
        "last_z_sha256": None if last_z is None else stable_digest(last_z),
        "cursor": int(getattr(session, "cursor", 0)),
        "updates": int(getattr(session, "updates", 0)),
    }

class Protocol030PassiveAuditLifecycle(Protocol029PassiveAuditLifecycle):
    """P029 observer with stronger content-based algorithm-state isolation checks."""

    def _integrity_snapshot(self, session):
        return algorithm_state_summary(session)

    def _assert_integrity(self, session, before):
        after=algorithm_state_summary(session)
        ok = before == after
        self.passive_state_checks += 1
        if not ok:
            self.passive_state_check_failures += 1
            fields=[k for k in before if before.get(k) != after.get(k)]
            raise AssertionError("Protocol030 passive observation mutated online state: "+",".join(fields))

class Protocol030DiagnosticSession(Protocol029DiagnosticSession):
    """Protocol-028 D policy plus the strengthened Protocol-030 passive observer."""

    def __init__(self, *args, guard_anchor, v2c_config=None, **kwargs):
        cfg=dict(V2C_DEFAULT); cfg.update(v2c_config or {})
        super().__init__(*args, guard_anchor=guard_anchor, v2c_config=cfg, **kwargs)
        self.lifecycle_controller=Protocol030PassiveAuditLifecycle(
            guard_anchor=guard_anchor, config=cfg
        )
        self.lifecycle_state=self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"]=True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator="D_memory_protected_with_protocol030_audit"

    def comparator_manifest(self):
        base=deepcopy(super().comparator_manifest())
        base.update({
            "name":"D_memory_protected_with_protocol030_audit",
            "protocol030_paired_audit":True,
            "online_method_changed":False,
            "integrity_snapshot":"content hashes for model/buffers, optimizer, controller, RNG, modes, requires_grad, topology, last_z",
        })
        return base
