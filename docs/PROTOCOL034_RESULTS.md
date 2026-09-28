# Protocol-034 Results

Run: `36436945806`. Development-only diagnostics selected after Protocol-033; not independent confirmation.

## H1 — active specialist continuous learning

- Six-window mean AP delta D_live − D_frozen_ref: **0.001090**.
- Late-four mean AP delta: **-0.002271**.
- Positive recurrence windows: **2/6**.
- Registered development signal pass: **False**.

## H2 — complete prediction memory

- Retained snapshot versions: **45**.
- Six-window equal-weight paired Full − Fragment AP delta: **-0.002799**.
- Hindsight best-snapshot results are diagnostic only and are not online routing results.

## H3 — fixed causal routing

- Six-window mean AP delta D_route_full − C_ref: **-0.003463**.
- Six-window mean AP delta D_route_full − D_live: **0.002375**.
- Six-window mean AP delta D_route_full − D_route_fragment: **-0.000626**.
- Registered development signal vs C_ref pass: **False**.
- Both cache passes used only labels satisfying `i + 2 < t`; no gradients or learning feedback were used.

## Budget / validity

- Frozen Protocol-033 reference reproduction passed: **True**.
- Exactly **3** full training replays and **2** fixed no-training cache routing passes were executed.
- No new simulator stream, extra seed, threshold sweep, or automatic scientific follow-up was started.
- Resource-equivalence and statistical-confirmation claims are not made.

