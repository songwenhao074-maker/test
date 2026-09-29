# Protocol-035 Results

Run: `36585058415`. Development-only structural-component experiment; not independent statistical confirmation.

## Registered result

- Result label: **recurrence_only**.
- Full-stream AP delta D_corr − C: **+0.004534**.
- Six-recurrence equal-weight AP delta: **+0.005737**; positive windows: **6/6**.
- Late-four mean AP delta: **+0.006900**.
- Global online signal: **False**; recurrence signal: **True**; common guardrails: **True**; joint goal: **False**.

## Capability coverage diagnosis

- Existing snapshot count: **45**.
- Supported predefined cells with hindsight local positive AP contrast: **76**.
- These cells are hindsight diagnostics, not online routing results or independent seeds.

## Isolation and causality

- C reference reproduction passed: **True**.
- D_off exactly equals C: **True**; classification exact copy: **True**.
- Tape read-only/no gradient to C: **True**.
- Branch actually updated: **True**; extra trainable parameters: **2401**.

## Guardrails

- Full FPR delta: **-0.000919**; full recall delta: **+0.003515**.
- Pooled recurrence FPR delta: **-0.001816**; recall delta: **-0.001262**.
- W-block equal-weight AP delta: **+0.008091**.

## Scope limits

- No claim is made for dynamic birth, protected memory, learned routing, hard deletion, equal-capacity superiority, or cross-seed generalization.
- The branch uses a distinct binary correction objective, so any gain cannot be attributed to topology alone.
- Exactly two new scientific training sequences were used; no new stream, extra seed, threshold sweep, old-D retraining, or automatic follow-up was run.

