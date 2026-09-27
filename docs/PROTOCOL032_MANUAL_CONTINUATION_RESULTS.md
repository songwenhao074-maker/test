# Protocol-032 Manual Post-Blocker Diagnostic Continuation Results

Continuation run: `36311650762`. Diagnostic continuation completed: **yes**. Registered Protocol-032 completed: **no**. The registered run `36310338191` remains blocked because C-vs-031 probability max abs diff was `1.0132789611816406e-6` against the `1e-6` reference limit. This continuation was started only after explicit user authorization.

No new simulation was generated. C was not rerun; the completed C from run `36310338191` was preserved. D_guard_budget was replayed exactly once on the same frozen 6800-row input. No model gate, threshold, seed, window, lifecycle budget, or target timing was changed.

## Diagnostic recurrence result

- U_rec1 AP delta D-C: -0.006167211
- V_rec1 AP delta D-C: -0.002919309
- Equal-weight mean AP delta D-C: -0.004543260
- Pooled recurrence normal-FPR delta D-C: -0.003962207
- Complete W_long/W_gap1 mean AP delta D-C: -0.007698366
- All fixed development references pass: False

## Mechanism

- Real dormant expert before a recurrence: True
- Accepted reuse with first influence inside U_rec1/V_rec1: False
- Reuse mechanism established for a registered recurrence: False
- Retirements: 5; reactivations: 2

## Cost/provenance

- Cross-run D/C wall-time ratio: 53.092
- C and D were not measured in the same job for this continuation, so same-job cost equivalence/advantage claims are not allowed.
- This output is development-only diagnostic evidence after a formal blocker, not a completed registered Protocol-032 pair and not independent confirmation.

No automatic follow-up experiment is authorized by this continuation.
