"""
Wind-Tunnel shared harness (AGENTS.md Section 3.3 / Section 9).

Single-ownership module (DSH doctrine): the JS->Python boundary ports and the
run_subject() call sequence live HERE ONCE. Per-cohort driver scripts
(wind_tunnel_v4_hall.py, wind_tunnel_v4_colas.py, ...) must import from this
module rather than re-implementing it, to prevent behavioral drift between
cohort runs (the exact failure mode Section 9.4 Bit-for-Bit Truth Across
Tracks exists to prevent).

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: the heavy math operators
    (extract_tau / estimate_dimension / filter_chunks / compute_rqa /
    compute_work_integral) are NOT reimplemented here. They are the literal
    PYTHON_ENGINE_CODE block extracted verbatim from index_v4.html (see
    _extracted_tensor_engine_v4.py, produced by a byte-slice of
    index_v4.html, not retyped). This satisfies Section 9.4.
  - Section 9.1.2 Labels as Prisms, Not Targets: any label fields on a
    subject record (diagnosis / y / SSPG / insulin / ...) are carried
    through to the output record UNTOUCHED and NEVER passed into any of
    the operators above and NEVER used in a fit/regression.
  - This module only PRODUCES data (Section 9.5 Product Isolation). It does
    not draw conclusions; that happens in the Homomorphic-Anchor-Forge
    report the LLM Navigator writes afterward from the JSON output.

The only code below that is NOT a verbatim extraction is the JS->Python port
of three purely structural (non-mathematical) boundary functions that
Contract v1.3 Law Two does not require to be Pyodide/SciPy-resident:
  - resample_raw():   port of resampleDataImpl()'s non-smooth branch,
                       index_v4.html lines ~1956-1974.
  - slice_by_period(): port of sliceByPeriod(), index_v4.html ~2053-2067.
  - takens_embedding(): port of takensEmbedding(), index_v4.html ~2074-2089.
All three are simple deterministic index/gap/hour arithmetic (no filtering,
no distance matrices), ported 1:1 to avoid behavioral drift from production.
"""
import sys
import json
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _extracted_tensor_engine_v4 as eng


def resample_raw(timestamps_iso, values):
    """Port of resampleDataImpl(data, smooth=false), index_v4.html:~1960-1984.

    [v1 residual, discovered via mcPHASES 2026-08-16, FIXED 2026-08-16] The
    production JS at index_v4.html previously relied on JS's implicit null->0
    coercion when a neighboring sample was missing during 4-15min-gap linear
    interpolation: it did NOT throw, it silently interpolated as if the
    missing reading were glucose=0, corrupting every downstream operator with
    no log trace. This was fixed under Section 10.4's UI/Code Constitution as
    a double atomic transaction (index_v4.html `bothValid` guard +
    Implementation Contract v1.3 Section 1.3 + Blueprint v3.3 Section 2.1
    [v3.5]) -- JS now skips the interpolation branch and lets the null
    propagate honestly, matching this Python port's behavior below. This
    guard is kept here (rather than removed) because it makes the harness
    robust even if a *future* JS edit reintroduces the coercion bug: this
    Python port never had the coercion in the first place (None - float
    raises TypeError in Python), so it always skip the small-gap
    interpolation whenever either endpoint is None, per AGENTS.md Section
    8.1/8.3 (No Fabrication / Zero Magic-Constant).
    """
    ts = [dt.datetime.fromisoformat(t) for t in timestamps_iso]
    vs = list(values)
    new_ts = [ts[0]]
    new_vs = [vs[0]]
    for i in range(1, len(ts)):
        gap = (ts[i] - ts[i - 1]).total_seconds() / 60.0
        if 4 < gap <= 15 and vs[i] is not None and vs[i - 1] is not None:
            steps = round(gap / 3)
            if steps < 1:
                steps = 1
            t_step = (ts[i] - ts[i - 1]) / steps
            v_step = (vs[i] - vs[i - 1]) / steps
            for j in range(1, steps):
                new_ts.append(ts[i - 1] + t_step * j)
                new_vs.append(vs[i - 1] + v_step * j)
        elif gap > 15:
            new_ts.append(ts[i - 1] + dt.timedelta(minutes=15))
            new_vs.append(None)
        new_ts.append(ts[i])
        new_vs.append(vs[i])
    return new_ts, new_vs


def slice_by_period(timestamps, values, period):
    """Port of sliceByPeriod(ts, vs, period), index_v4.html:~2053-2067.

    Preserves the FULL array length (Contract v1.3 4.4 Spatiotemporal Alignment) --
    out-of-window samples become None in-place, never compacted, so physical-time
    correspondence with `timestamps` is untouched for every downstream consumer.
    """
    if period == "all":
        return list(values)
    start_h = 0 if period == "night" else 6
    end_h = 6 if period == "night" else 18
    out = [None] * len(values)
    for i, (t, v) in enumerate(zip(timestamps, values)):
        if v is None:
            continue
        h = t.hour + t.minute / 60.0
        in_range = (h >= start_h and h < end_h) if start_h <= end_h else (h >= start_h or h < end_h)
        out[i] = v if in_range else None
    return out


def takens_embedding(values, tau, dim):
    """Port of takensEmbedding(values, tau, dim), index_v4.html:~2074-2089."""
    n = len(values)
    points = []
    for i in range(n - (dim - 1) * tau):
        pt = []
        has_null = False
        for d in range(dim):
            v = values[i + d * tau]
            if v is None:
                has_null = True
                break
            pt.append(v)
        points.append(None if has_null else pt)
    return points


# Label fields are cohort-agnostic prism candidates. A cohort's subject dict
# may be missing some of these (e.g. Colas has no diagnosis/insulin/SSPG) --
# subject.get() below simply yields None for those, which is honest (Section
# 8.1 No Inference & No Fabrication: absent label stays absent, never guessed).
LABEL_FIELDS = ("cohort", "diagnosis", "y", "insulin", "SSPG")


def run_subject(subject, period="night"):
    """period='night' (Contract-faithful): estimate_dimension and the pointsSmooth fed to
    RQA/Work-Integral are computed on the night-sliced (0:00-6:00) series, exactly mirroring
    processEpochPyodide's period-gated path (index_v4.html ~3534, 3577, 3603). Tau extraction
    stays on the FULL raw series unconditionally, matching onDataReady (index_v4.html
    ~3318-3326) which never period-slices before calling extract_tau.
    """
    sid = subject["id"]
    log = []
    try:
        ts, raw_vs = resample_raw(subject["timestamps"], subject["values"])
    except Exception as e:
        return {"id": sid, "error": f"[HARNESS ERROR] resample_raw crashed: {e}"}

    valid_n = sum(1 for v in raw_vs if v is not None)
    if valid_n < 60:
        return {"id": sid, "error": f"[L0 ERROR] Insufficient valid points after resample ({valid_n})."}

    raw_json = json.dumps(raw_vs)

    tau_res = json.loads(eng.engine.extract_tau(raw_json))
    log += tau_res.get("events", [])
    tau = tau_res.get("result")
    if tau is None:
        return {"id": sid, "error": f"[L1 ERROR] extract_tau failed: {tau_res.get('error')}", "events": log}

    probe_vs = slice_by_period(ts, raw_vs, period)
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

    sliced_smooth_vs = slice_by_period(ts, smooth_vs, period)
    if len(sliced_smooth_vs) - (dim - 1) * tau < 10:
        return {"id": sid, "error": "[L2 ERROR] Series too short for tau/dim after embedding.", "events": log}

    points_smooth = takens_embedding(sliced_smooth_vs, tau, dim)
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
        "dim": dim,
        "det": rqa.get("det") if rqa else None,
        "entr": rqa.get("entr") if rqa else None,
        "rr": rqa.get("rr") if rqa else None,
        "workIntegral": work_integral,
        "events": log,
    }
    # Labels: prism-only, NEVER fed back into any computation above (Section 9.1.2).
    # Pass through any cohort-specific metadata attached to subject.
    for k, v in subject.items():
        if k not in ("timestamps", "values", "id"):
            record[k] = v
    return record


def run_cohort(cohort_name, subjects, period="night"):
    results = [run_subject(s, period=period) for s in subjects]
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"Processed {len(subjects)} {cohort_name} subjects: {len(ok)} succeeded, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")
    return results, ok, failed


def write_results(cohort_name, subjects, results, ok, failed, period, tau_max, out_dir="reports"):
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_path / f"wind_tunnel_{cohort_name}_{period}_taumax{tau_max}_{ts_tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": cohort_name, "period": period, "tau_max": tau_max,
            "n_total": len(subjects), "n_success": len(ok), "n_failed": len(failed),
            "results": results,
        }, f, indent=2)
    print(f"Wrote raw per-subject results to {out_file}")
    return out_file
