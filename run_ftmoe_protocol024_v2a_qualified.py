"""Entrypoint that binds the pre-decision v2a qualification implementation."""
from ftmoe_protocol024_v2a_qualification import QualifiedV2AProtocol024Session
import run_ftmoe_protocol024_v2a as _runner

# Bind before runner constructs any session.  The split keeps the causal pair
# assembly auditable and prevents the base decision hook from running before
# fixed-threshold FPR counts are attached to each validation pair.
_runner.V2AProtocol024Session = QualifiedV2AProtocol024Session


if __name__ == "__main__":
    _runner.main()
