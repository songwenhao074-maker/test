"""Read-only diagnosis of actual residual predictions and recent sampling."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / "final_runs_20260909_1435"
report = {}
for seed in (500, 501):
    with np.load(root / ("A_dev%d_m1/predictions.npz" % seed)) as data:
        base_p = data["probability"].copy()
        base_c = data["class_probability"].copy()
        labels = data["labels"].copy()
    for method in ("C-legacy", "C-residual-off", "C-residual-on"):
        folder = root / ("%s_dev%d_m1" % (method, seed))
        with np.load(folder / "predictions.npz") as data:
            probability = data["probability"].copy()
            classification = data["class_probability"].copy()
        updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text(encoding="utf8").splitlines() if line]
        indices = np.asarray([i for update in updates for i in update["buffer_indices"]], dtype=int)
        delta = np.abs(probability - base_p)
        original_error = (base_p >= .5) != (labels > 0)
        error = (probability >= .5) != (labels > 0)
        report["%s/dev%d" % (method, seed)] = {
            "absolute_probability_change_mean": float(delta.mean()),
            "absolute_probability_change_median_p95_p99_max": np.quantile(delta, [.5, .95, .99, 1.]).tolist(),
            "changed_detection_decisions": int(((probability >= .5) != (base_p >= .5)).sum()),
            "corrected_A_errors": int((original_error & ~error).sum()),
            "new_errors_vs_A": int((~original_error & error).sum()),
            "changed_resource_decisions_on_positive_hosts": int(((classification.argmax(-1) != base_c.argmax(-1)) & (labels > 0)).sum()),
            "stream_host_positive_fraction": float((labels > 0).mean()),
            "recent_sample_host_positive_fraction": float((labels[indices] > 0).mean()),
            "recent_window_draw_count": len(indices),
            "recent_distinct_window_count": len(set(indices.tolist())),
            "all_recent_draws_mature_before_update": all(i <= update["step"] - 2 for update in updates for i in update["buffer_indices"]),
            "sampling_scope": "recent windows only; anchors and class weighting excluded",
        }
(root / "inspection_primary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
print(json.dumps(report))
