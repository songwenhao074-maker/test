# Protocol-032 manual post-blocker diagnostic continuation

Date: 2026-09-27. Status: user-authorized after the registered Protocol-032 run stopped at its baseline-C consistency blocker.

The registered Protocol-032 result from run `36310338191` remains **blocked** and must not be rewritten as a completed preregistered pair. Its only baseline-consistency failure was `C_probability_max_abs_diff = 1.0132789611816406e-6` versus the registered reference upper limit `1e-6`; labels and raw labels were exactly equal and the shared-first4 initialization hash matched exactly. D did not start in that run.

The user subsequently explicitly instructed ChatGPT to continue the experiment. This authorizes one **diagnostic continuation** only. It does not retroactively waive or alter the registered Protocol-032 blocker.

Rules for this continuation:

- Do not generate or extend simulator data. Use only the already repaired/frozen 6800-row Protocol-032 input bundle from run `36310338191` (`6799 scored + 1 guard`, stream SHA256 `1e8b6bde1fa3f906586030547777428b28fca23c43e06868077dc0e1e46e4d1a`).
- Do not rerun C. Reuse the completed `C_fixed5` outputs already saved by run `36310338191`.
- Run `D_guard_budget` exactly once with the existing Protocol-032 implementation, model seed 1, replay seed 700, and unchanged method rule `L_candidate <= L_live + max(0.02*L_live, 0.01) + 1e-6`.
- Do not change any other gate, lifecycle budget, feature, target timing, threshold, window, seed, or model parameter.
- Preserve the failed registered baseline-consistency report alongside the continuation result. Never mark it as passed.
- Compare D against the reused completed C on U_rec1 and V_rec1 and report the same fixed development references, mechanism evidence, and measured costs. Because C and D are now from different GitHub jobs, cost comparison must be labelled cross-run diagnostic rather than same-job measurement.
- Clearly label all continuation outputs as `manual_post_blocker_diagnostic_continuation`, `registered_protocol032_completed=false`, and `statistical_confirmation=false`.
- No automatic seed sweep, threshold search, A/B, new scenario, or further model replay after this single D run.

This continuation exists to answer the user's explicit request to continue after the registered blocker while preserving the scientific provenance of the original Protocol-032 stop decision.
