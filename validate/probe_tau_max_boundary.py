#!/usr/bin/env python3
"""EXPERIMENTAL BOUNDARY PROBE -- NOT part of the production wind-tunnel harness.

Purpose: the ShanghaiT1DM report (2026-08-19 11:41) found 75% of recordings
pinned at the ACF tau-extraction ceiling (max_lag=60 steps = 180 min at the
engine's native 3-min grid), vs only 23% for the same-protocol ShanghaiT2DM
cohort. This script answers: is max_lag=60 a genuinely too-narrow probe
window for T1DM (real tau lives higher, we just can't see it), or does the
autocorrelation simply never decay within any reasonable window (a different,
non-periodicity problem)?

STRICT SCOPE GUARDRAILS (why this is a separate file, not an edit to
_extracted_tensor_engine_v4.py):
  - Section 9.1.1 Calculation Firewall / Section 9.4 Bit-for-Bit Truth Across
    Tracks: the production `extract_tau()` in _extracted_tensor_engine_v4.py
    is a byte-slice of index_v4.html's PYTHON_ENGINE_CODE and MUST NOT be
    edited for an exploratory probe -- doing so would desynchronize the
    "same math on both tracks" guarantee for every OTHER cohort already
    tested against max_lag=60.
  - `_extract_tau_variable()` below is a manually-verified line-for-line copy
    of the ACF/decay-lock logic inside extract_tau() (lines ~47-77 of
    _extracted_tensor_engine_v4.py), with ONLY `max_lag` turned into a
    parameter instead of the hardcoded literal `60`. No other logic is
    changed. This is a read-only diagnostic; nothing here writes back to
    subjects.json, the wind-tunnel result JSONs, or any production file.
  - Per Section 9.4, if this probe concludes tau_max=60 IS too narrow for a
    specific cohort, the fix is NOT to silently raise it here -- it would
    need a separate, explicitly-approved atomic update to BOTH
    _extracted_tensor_engine_v4.py's constant AND index_v4.html's
    PYTHON_ENGINE_CODE (Section B.5 Atomic Blueprint Evolution), which is
    outside this probe's authority. This script only produces the evidence
    needed to decide whether that conversation is warranted.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt  # noqa: E402  (resample_raw reuse only)

CANDIDATE_MAX_LAGS = [60, 90, 120, 180, 240]


def _extract_tau_variable(values_list, max_lag):
    """Line-for-line copy of extract_tau()'s ACF/decay-lock math, max_lag parameterized."""
    chunk_indices = []
    cur_indices = []
    for i, v in enumerate(values_list):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            cur_indices.append(i)
        else:
            if len(cur_indices) > 0:
                chunk_indices.append(cur_indices)
                cur_indices = []
    if len(cur_indices) > 0:
        chunk_indices.append(cur_indices)

    chunks = [np.array([values_list[i] for i in idxs], dtype=np.float64) for idxs in chunk_indices if len(idxs) >= 30]
    if not chunks:
        return None

    total_len = sum(len(c) for c in chunks)
    avg_acf = np.zeros(max_lag + 1)

    for c in chunks:
        n = len(c)
        mean = np.mean(c)
        var = np.var(c)
        if var < 1e-9:
            continue
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        for lag in range(1, max_lag + 1):
            if n <= lag:
                break
            acf[lag] = np.sum((c[:-lag] - mean) * (c[lag:] - mean)) / ((n - lag) * var)
        avg_acf += acf * (n / total_len)

    tau = max_lag
    decayed = False
    for i in range(1, max_lag):
        if avg_acf[i] < 0.7:
            decayed = True
        if avg_acf[i] < 0.3678:
            tau = i
            break
        if decayed and avg_acf[i] < avg_acf[i - 1] and avg_acf[i] < avg_acf[i + 1]:
            tau = i
            break

    return tau


def probe_cohort(cohort_name, subjects):
    print(f"\n=== {cohort_name} (n={len(subjects)}) ===")
    per_lag_taus = {ml: [] for ml in CANDIDATE_MAX_LAGS}
    per_subject_rows = []

    for s in subjects:
        try:
            ts, raw_vs = wt.resample_raw(s["timestamps"], s["values"])
        except Exception as e:
            print(f"  SKIP {s['id']}: resample_raw crashed: {e}")
            continue
        valid_n = sum(1 for v in raw_vs if v is not None)
        if valid_n < 60:
            continue

        row = {"id": s["id"]}
        for ml in CANDIDATE_MAX_LAGS:
            tau = _extract_tau_variable(raw_vs, ml)
            per_lag_taus[ml].append(tau)
            row[f"tau_maxlag{ml}"] = tau
        per_subject_rows.append(row)

    print(f"{'max_lag':>8}{'n':>6}{'median_tau':>12}{'mean_tau':>10}{'pct_at_ceiling':>16}")
    summary = {}
    for ml in CANDIDATE_MAX_LAGS:
        taus = [t for t in per_lag_taus[ml] if t is not None]
        if not taus:
            continue
        arr = np.array(taus)
        pct_ceiling = float(np.mean(arr == ml))
        print(f"{ml:>8}{len(arr):>6}{np.median(arr):>12.1f}{np.mean(arr):>10.2f}{pct_ceiling:>15.1%}")
        summary[ml] = {
            "n": len(arr),
            "median_tau": float(np.median(arr)),
            "mean_tau": float(np.mean(arr)),
            "pct_at_ceiling": pct_ceiling,
        }

    return summary, per_subject_rows


def main():
    results = {}

    t1dm_path = Path("output/shanghai_t1dm_subjects.json")
    if not t1dm_path.exists():
        import export_shanghai_t1dm_subjects
        export_shanghai_t1dm_subjects.export_shanghai_t1dm_subjects()
    with open(t1dm_path, "r", encoding="utf-8") as f:
        t1dm_subjects = json.load(f)["subjects"]
    summary_t1dm, rows_t1dm = probe_cohort("ShanghaiT1DM", t1dm_subjects)
    results["shanghai_t1dm"] = {"summary": summary_t1dm, "per_subject": rows_t1dm}

    t2dm_path = Path("output/shanghai_t2dm_subjects.json")
    if not t2dm_path.exists():
        import export_shanghai_subjects
        export_shanghai_subjects.export_shanghai_t2dm_subjects()
    with open(t2dm_path, "r", encoding="utf-8") as f:
        t2dm_subjects = json.load(f)["subjects"]
    # First-visit only, for a fair n-comparable reference (mirrors T1DM's mostly-single-visit shape).
    t2dm_first_visit = [s for s in t2dm_subjects if s.get("visit_index") == 0]
    summary_t2dm, rows_t2dm = probe_cohort("ShanghaiT2DM (first visit only, reference)", t2dm_first_visit)
    results["shanghai_t2dm_first_visit"] = {"summary": summary_t2dm, "per_subject": rows_t2dm}

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"probe_tau_max_boundary_{ts_tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote full probe results to {out_file}")


if __name__ == "__main__":
    main()
