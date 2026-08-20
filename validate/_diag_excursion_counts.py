"""Scratch diagnostic (NOT part of the validated pipeline): counts how many qualifying
forced-rise excursions (peak-valley diff >= 1.5) contribute to each subject's
earlyDelay/relaxationTime median, to test the hypothesis that per-subject event count is a
confound (redesign motivation for candidate_tensor_staging_matrix.md 候选 #5).
Duplicates (does not modify) the peak/valley detection logic from
_legacy_metrics_v4.compute_excursion_kinetics for read-only inspection.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt


def count_events(times, vals):
    peaks, valleys = [], []
    direction = 0
    for i in range(1, len(vals)):
        if vals[i] is None or vals[i - 1] is None:
            continue
        diff = vals[i] - vals[i - 1]
        if abs(diff) < 0.01:
            continue
        new_dir = 1 if diff > 0 else -1
        if direction != 0 and new_dir != direction:
            if new_dir == 1:
                valleys.append({"idx": i - 1, "val": vals[i - 1]})
            else:
                peaks.append({"idx": i - 1, "val": vals[i - 1]})
        direction = new_dir

    n_qualifying = 0
    for p in peaks:
        prev_valley = None
        for v in reversed(valleys):
            if v["idx"] < p["idx"]:
                prev_valley = v
                break
        if prev_valley is None and p["idx"] > 0:
            prev_valley = {"idx": 0, "val": vals[0]}
        if prev_valley is None or p["val"] - prev_valley["val"] < 1.5:
            continue
        n_qualifying += 1
    return n_qualifying


def analyze(cohort_label, subjects_path, results_path):
    with open(subjects_path, "r", encoding="utf-8") as f:
        subj_data = json.load(f)
    subjects = {s["id"]: s for s in subj_data["subjects"]}

    with open(results_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    counts = []
    relax_vals = []
    for r in res_data["results"]:
        if "error" in r or r.get("relaxationTime") is None:
            continue
        sid = r["id"]
        subj = subjects[sid]
        ts, raw_vs = wt.resample_raw(subj["timestamps"], subj["values"])
        filt_res = json.loads(wt.eng.engine.filter_chunks(json.dumps(raw_vs), 2, 0.08))
        smooth_vs = filt_res.get("result")
        if smooth_vs is None:
            continue
        n = count_events(ts, smooth_vs)
        counts.append(n)
        relax_vals.append(r["relaxationTime"])

    counts = np.array(counts)
    relax_vals = np.array(relax_vals)
    print(f"\n=== {cohort_label} ===")
    print(f"n_subjects={len(counts)}, event count: min={counts.min()}, median={np.median(counts):.1f}, "
          f"max={counts.max()}, subjects_with_1_event={np.sum(counts==1)}, subjects_with_le2={np.sum(counts<=2)}")
    if len(counts) > 3:
        corr = np.corrcoef(counts, relax_vals)[0, 1]
        print(f"Pearson corr(event_count, relaxationTime) = {corr:.4f}")
        low_n = relax_vals[counts <= 2]
        high_n = relax_vals[counts > 2]
        if len(low_n) > 0 and len(high_n) > 0:
            print(f"variance when n<=2 events (n_subj={len(low_n)}): {np.var(low_n):.2f}  vs  "
                  f"variance when n>2 events (n_subj={len(high_n)}): {np.var(high_n):.2f}")


analyze("Stanford SSPG", "output/stanford_sspg_subjects.json",
        "reports/wind_tunnel_stanford_sspg_legacymetrics_night_taumax120_20260819_1730.json")
analyze("Shanghai T2DM", "output/shanghai_t2dm_subjects.json",
        "reports/wind_tunnel_shanghai_t2dm_legacymetrics_night_taumax120_20260819_1947.json")
