# Protocol-030 Results

GitHub Actions run: 35716480519.
Task: same-job `audit_off`/`audit_on` paired noninterference validation for the unchanged Protocol-028 D policy.
Completed: True. **Paired audit valid: True.**
C rerun: False. New scientific methods: 0. Full replay budget used: 2/2.

## Paired validity gate

- Detection probability max absolute error: `0.0` (required ≤1e-6); exact-equal fraction `1.0`.
- Class probability max absolute error: `0.0` (required ≤1e-6); exact-equal fraction `1.0`.
- Labels/raw labels exact: `True/True`.
- Nine recurrence first100 AP differences: all `0.0`; equal-weight mean AP difference `0.0`.
- Full-stream AP difference: `0.0`.
- Discrete lifecycle events: `981/981`, exact=`True`.
- Final topology exact: `True`.
- Passive state violations: `0`.
- Stable non-timing state-trace fields exact: `True`.

Raw `controller_state_sha256` differed from checkpoint 767 onward because that digest included `extra_compute` timing fields. Protocol-030 explicitly excludes timing and pure diagnostic counters from cross-process algorithm-state comparison, so the timing-contaminated digest is not part of the registered paired gate. Model, optimizer, RNG, topology, module mode, `requires_grad`, `_last_z`, predictions and lifecycle were exact under the registered comparison.

## Main finding

Under the preregistered single-thread deterministic CPU profile, enabling the passive memory observer did **not** alter the online Protocol-028 D trajectory within the registered gate; the saved detection and class probabilities were byte-for-byte equal between `audit_off` and `audit_on`.

The valid audit found **5** candidate-opportunity records satisfying all original reuse acceptance gates. All five use expert 9 at cursors 3937, 3969, 4001, 4033 and 4065 in `S6_first`; each was simultaneously blocked by the controller being busy and by similarity being below threshold. These are `post_hoc_oracle_diagnostic` opportunities, not online reactivations and not D>C evidence.

## Historical Protocol-028 comparison

- `audit_off` historical_028_match: `False`.
- `audit_on` historical_028_match: `False`.
- Historical detection-probability max error: `8.642673492431641e-07`.
- Historical class-probability max error: `1.430511474609375e-06` (exceeds 1e-6).
- Historical full AP absolute difference: `9.551872326429844e-08`.
- All nine recurrence first100 AP values match historical Protocol-028, and lifecycle/final topology match. Protocol-030 therefore validates same-job observer noninterference but does **not** retroactively make Protocol-029 valid or claim exact historical-028 reproduction.

## Runtime

- `audit_off` wall time: `716.866` s.
- `audit_on` wall time: `1722.988` s.
- Passive prediction overhead: `956.481` s.
- Passive guard overhead: `50.656` s.

These are Protocol-030 diagnostic timings under a single-thread profile and are not a new D-vs-C efficiency comparison.

## Post-processing recovery

Both registered full replays completed successfully. The Actions parent process then failed only while serializing a `numpy.int64` index into `paired_consistency.json`. The immutable uploaded artifact contained both complete replay outputs, so the paired/historical reports were recovered from artifact `10690558198` without starting any additional replay. The formal full-replay budget remains 2/2.

## Evidence

- Run: https://github.com/songwenhao074-maker/test/actions/runs/35716480519
- Artifact ID: `10690558198`.
- Artifact digest: `sha256:16c0731df2712ac5cfc5554da0d2cd3f5cf83ce74bb2ae1230c94f19fc2502af`.

## Stop

Protocol-030 has reached its registered stop point. Do not automatically add replays, methods, seeds, threshold changes, scenario redesign or parameter search.
