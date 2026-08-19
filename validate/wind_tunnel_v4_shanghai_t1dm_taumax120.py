"""
Supplementary (non-production) re-run: ShanghaiT1DM full tensor pipeline with
a corrected tau-extraction window (max_lag=120), per Option 3 chosen in
wind_tunnel_shanghai_t1dm_20260819_1200_tau_max_boundary_calibration.md.

WHY THIS IS A SEPARATE FILE, NOT A PRODUCTION CHANGE:
  - Section 9.4 Bit-for-Bit Truth Across Tracks forbids letting one cohort
    silently run on different core-operator parameters than the other 9
    already-tested cohorts. Raising the PRODUCTION `max_lag` constant in
    `_extracted_tensor_engine_v4.py` (and the mirrored PYTHON_ENGINE_CODE in
    index_v4.html) would require re-running and re-validating all 9 prior
    cohorts (Option 2 in the calibration report) -- an explicit-approval,
    large-blast-radius decision this script does NOT make.
  - Instead, this script produces a clearly-labeled SUPPLEMENTARY result set
    for ShanghaiT1DM only, reported ALONGSIDE (never silently replacing) the
    production max_lag=60 result already on file
    (reports/wind_tunnel_shanghai_t1dm_night_taumax60_20260819_1141.json).
    Both are kept, both are dated, neither overwrites the other (Section 8.2
    Honest Fail-Closed / v1.4 No History Truncation).

BIT-FOR-BIT SCOPE: every downstream operator call below
(estimate_dimension / filter_chunks / compute_rqa / compute_work_integral)
is the UNMODIFIED production method from _extracted_tensor_engine_v4.py,
called exactly as _wind_tunnel_common.run_subject() calls them. The ONLY
substitution is the tau value fed into them: instead of
`eng.engine.extract_tau(raw_json)` (hardcoded max_lag=60), this script uses
`_extract_tau_variable(raw_vs, max_lag=120)` -- a manually-verified
line-for-line copy of that same function's ACF/decay-lock math with only the
window size parameterized (identical to probe_tau_max_boundary.py's copy,
duplicated here rather than imported to keep this file self-contained and
auditable in isolation).
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt  # noqa: E402
import _extracted_tensor_engine_v4 as eng  # noqa: E402

MAX_LAG_CORRECTED = 120
PERIOD = "night"


def _extract_tau_variable(values_list, max_lag):
    """Verbatim copy of extract_tau()'s ACF/decay-lock math; see probe_tau_max_boundary.py."""
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


def run_subject_corrected(subject, period="night"):
    """Mirrors _wind_tunnel_common.run_subject() exactly, substituting only the
    tau-extraction call. All eng.engine.* calls below are the unmodified
    production methods."""
    sid = subject["id"]
    log = []
    try:
        ts, raw_vs = wt.resample_raw(subject["timestamps"], subject["values"])
    except Exception as e:
        return {"id": sid, "error": f"[HARNESS ERROR] resample_raw crashed: {e}"}

    valid_n = sum(1 for v in raw_vs if v is not None)
    if valid_n < 60:
        return {"id": sid, "error": f"[L0 ERROR] Insufficient valid points after resample ({valid_n})."}

    raw_json = json.dumps(raw_vs)

    tau = _extract_tau_variable(raw_vs, MAX_LAG_CORRECTED)
    if tau is None:
        return {"id": sid, "error": "[L1 ERROR] extract_tau_variable failed.", "events": log}

    probe_vs = wt.slice_by_period(ts, raw_vs, period)
    probe_valid_n = sum(1 for v in probe_vs if v is not None)
    if probe_valid_n < 30:
        return {"id": sid, "error": f"[L1 ERROR] Insufficient {period}-sliced points for dimension estimation ({probe_valid_n}).", "events": log}

    dim_res = json.loads(eng.engine.estimate_dimension(json.dumps(probe_vs), tau))
    log += dim_res.get("events", [])
    dim = dim_res.get("result")
    if dim is None or dim < 2:
        return {"id": sid, "error": f"[L1 ERROR] estimate_dimension failed or <2: {dim_res.get('error')}", "events": log}

    filt_res = json.loads(eng.engine.filter_chunks(raw_json, 2, 0.08))
    log += filt_res.get("events", [])
    smooth_vs = filt_res.get("result")
    if smooth_vs is None:
        return {"id": sid, "error": f"[L0 ERROR] filter_chunks failed: {filt_res.get('error')}", "events": log}

    sliced_smooth_vs = wt.slice_by_period(ts, smooth_vs, period)
    if len(sliced_smooth_vs) - (dim - 1) * tau < 10:
        return {"id": sid, "error": "[L2 ERROR] Series too short for tau/dim after embedding.", "events": log}

    points_smooth = wt.takens_embedding(sliced_smooth_vs, tau, dim)
    points_json = json.dumps(points_smooth)

    rqa_res = json.loads(eng.engine.compute_rqa(points_json, tau))
    log += rqa_res.get("events", [])
    rqa = rqa_res.get("result")

    wi_res = json.loads(eng.engine.compute_work_integral(points_json))
    log += wi_res.get("events", [])
    work_integral = wi_res.get("result")

    record = {
        "id": sid,
        "period": period,
        "tau": tau,
        "max_lag_used": MAX_LAG_CORRECTED,
        "dim": dim,
        "det": rqa.get("det") if rqa else None,
        "entr": rqa.get("entr") if rqa else None,
        "rr": rqa.get("rr") if rqa else None,
        "workIntegral": work_integral,
        "events": log,
    }
    for k, v in subject.items():
        if k not in ("timestamps", "values", "id"):
            record[k] = v
    return record


def main():
    data_path = Path("output/shanghai_t1dm_subjects.json")
    if not data_path.exists():
        import export_shanghai_t1dm_subjects
        export_shanghai_t1dm_subjects.export_shanghai_t1dm_subjects()
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} Shanghai_T1DM recordings from {data_path}.")
    print(f"Re-running full pipeline with CORRECTED tau extraction (max_lag={MAX_LAG_CORRECTED}), "
          f"period='{PERIOD}'. All downstream operators (estimate_dimension/filter_chunks/"
          f"compute_rqa/compute_work_integral) are the UNMODIFIED production methods.")

    results = [run_subject_corrected(s, period=PERIOD) for s in subjects]
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"Processed {len(subjects)} subjects: {len(ok)} succeeded, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"wind_tunnel_shanghai_t1dm_{PERIOD}_taumax{MAX_LAG_CORRECTED}_SUPPLEMENTARY_{ts_tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "shanghai_t1dm",
            "note": "SUPPLEMENTARY, non-production result. Production default remains max_lag=60 "
                     "(see wind_tunnel_shanghai_t1dm_night_taumax60_20260819_1141.json). This file "
                     "exists alongside it, per Option 3 of the tau_max boundary calibration report.",
            "period": PERIOD, "max_lag_used": MAX_LAG_CORRECTED,
            "n_total": len(subjects), "n_success": len(ok), "n_failed": len(failed),
            "results": results,
        }, f, indent=2)
    print(f"Wrote SUPPLEMENTARY (non-production) results to {out_file}")


if __name__ == "__main__":
    main()
