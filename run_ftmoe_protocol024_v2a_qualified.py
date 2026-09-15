"""Entrypoint that binds the pre-decision v2a qualification implementation."""
import numpy as np

from ftmoe_protocol024_v2a_qualification import QualifiedV2AProtocol024Session
import run_ftmoe_protocol024_v2a as _runner

# Bind before runner constructs any session.  The split keeps the causal pair
# assembly auditable and prevents the base decision hook from running before
# fixed-threshold FPR counts are attached to each validation pair.
_runner.V2AProtocol024Session = QualifiedV2AProtocol024Session

# Engineering fix after run 34980970874: deployable/reference pressure baselines
# are detection baselines, so the 0/1/2/3 raw-next resource target must be mapped
# to binary fault/no-fault exactly as binary_detection_metrics does.  The failed
# run had already completed A/C/D replay and crashed only while summarizing this
# baseline.  This patch changes no stream/model/lifecycle/budget parameter.
def _binary_detection_ap(y, score):
    truth = (np.asarray(y, dtype=np.int64).reshape(-1) > 0).astype(np.int64)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    return _runner.average_precision_report(truth, values)


_runner._ap = _binary_detection_ap


if __name__ == "__main__":
    _runner.main()
