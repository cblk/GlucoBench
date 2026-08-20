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
import _legacy_metrics_v4 as legacy
from _legacy_metrics_v4 import _median as _legacy_median
import _legacy_metrics_group1_v4 as legacy_g1


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


def run_subject(subject, period="night", tau_max=120):
    """period='night' (Contract-faithful): estimate_dimension and the pointsSmooth fed to
    RQA/Work-Integral are computed on the night-sliced (0:00-6:00) series, exactly mirroring
    processEpochPyodide's period-gated path (index_v4.html ~3534, 3577, 3603). Tau extraction
    stays on the FULL raw series unconditionally, matching onDataReady (index_v4.html
    ~3318-3326) which never period-slices before calling extract_tau.

    tau_max (default 120, matching PRODUCTION index_v4.html's max_lag as of 2026-08-19
    /Blueprint v3.6 -- see reports/wind_tunnel_fleet_taumax60_vs_120_option2_evaluation_
    20260819_1520.md for the fleet-wide evidence behind that atomic transaction): threaded
    straight into eng.engine.extract_tau's max_lag parameter. Callers reproducing the PRE-2026
    -08-19 production behavior (or diffing against historical taumax60 archives) must pass
    tau_max=60 explicitly -- the default here always tracks current production, per Section
    9.4 Bit-for-Bit Truth Across Tracks.
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

    tau_res = json.loads(eng.engine.extract_tau(raw_json, max_lag=tau_max))
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


def run_subject_legacy(subject, period="night", tau_max=120):
    """Companion to run_subject() for the 2026-08-19 Group-2 test plan (dataset_fleet_registry.md
    changelog "开始进行第二组6张判色卡的风洞测试基础设施搭建"): computes the 6 legacy-JS-ported
    candidate metrics (Early Phase Delay / Relaxation Time / AR1 / Angular Velocity / Ascend
    Friction / Night Friction) via _legacy_metrics_v4.py.

    Deliberately NOT merged into run_subject(): that function's byte-for-byte output has been
    regression-verified against historical taumax60 archives (Hall/Colas) multiple times this
    session, and editing its body -- even only appending new fields -- would require re-earning
    that guarantee from scratch. This costs ~15 duplicated lines of shared preprocessing
    (resample/tau/dim/smooth) in exchange for run_subject()'s existing guarantee staying inert.

    Track conventions (mirrors index_v4.html's default chk-smooth=checked UI state, Blueprint
    v3.3 L0 2.2 Dual-Track Law):
      - Ascend Friction: period-sliced RAW points, gravity core from period-sliced SMOOTH points
        (index_v4.html:3667's shapePoints argument under the default smooth=true toggle).
      - Night Friction: RAW night-sliced (00:00-06:00) points and core, UNCONDITIONALLY,
        regardless of the `period` argument (index_v4.html:3640-3642).
      - AR1/variance/skewness: full RAW series, unconditional (not period-sliced).
      - Early Phase Delay / Relaxation Time: full SMOOTH series, unconditional.
      - Angular Velocity / Sweep Rate: full SMOOTH series vs scalar nightMean (mean of RAW
        night-sliced values, requires >=6 valid night points or stays None -- index_v4.html
        :3659-3660).
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
    tau_res = json.loads(eng.engine.extract_tau(raw_json, max_lag=tau_max))
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

    def median_col(points, d):
        return _legacy_median([p[d] for p in points])

    # --- Ascend Friction ---
    period_raw_vs = slice_by_period(ts, raw_vs, period)
    period_smooth_vs = slice_by_period(ts, smooth_vs, period)
    points_raw = takens_embedding(period_raw_vs, tau, dim)
    points_shape = takens_embedding(period_smooth_vs, tau, dim)
    valid_shape = [p for p in points_shape if p is not None]
    ascend_friction = None
    if len(valid_shape) >= 4:
        gravity_core = [median_col(valid_shape, d) for d in range(dim)]
        ascend_res = legacy.compute_asymmetric_friction(points_raw, gravity_core)
        ascend_friction = ascend_res.get("ascendFriction")
        log.append(f"[INFO] [Legacy] Ascend Friction computed from {len(valid_shape)} shape points.")
    else:
        log.append(f"[ERROR] [Legacy] Insufficient valid shape points for gravity core ({len(valid_shape)}). ascendFriction=None.")

    # --- Night Friction (unconditional RAW night track, per L0 2.2) ---
    night_raw_vs = slice_by_period(ts, raw_vs, "night")
    night_points_all = takens_embedding(night_raw_vs, tau, dim)
    night_points_valid = [p for p in night_points_all if p is not None]
    night_friction = None
    if len(night_points_valid) > 0:
        night_core = [median_col(night_points_valid, d) for d in range(dim)]
        night_res = legacy.compute_asymmetric_friction(night_points_all, night_core)
        night_friction = night_res.get("asymFriction")
        log.append(f"[INFO] [Legacy] Night Friction computed from {len(night_points_valid)} night points.")
    else:
        log.append("[ERROR] [Legacy] No valid night points. nightFriction=None.")

    # --- AR1 (full RAW series, unconditional) ---
    csd_res = legacy.compute_critical_slowing_down(ts, raw_vs)
    if csd_res.get("ar1") is None:
        log.append("[ERROR] [Legacy] No night with >=30 valid points found. ar1=None.")

    # --- Early Phase Delay / Relaxation Time (full SMOOTH series, unconditional) ---
    kinetics_res = legacy.compute_excursion_kinetics(ts, smooth_vs)
    if kinetics_res.get("earlyDelay") is None:
        log.append("[ERROR] [Legacy] No valid forced-rise excursion (>1.5) found. earlyDelay/relaxationTime=None.")

    # --- Angular Velocity / Sweep Rate (full SMOOTH series vs scalar RAW nightMean) ---
    night_raw_valid = [v for v in night_raw_vs if v is not None]
    night_mean = sum(night_raw_valid) / len(night_raw_valid) if len(night_raw_valid) >= 6 else None
    kepler_res = legacy.compute_kepler_kinematics(ts, smooth_vs, night_mean)
    if kepler_res.get("angularVelocity") is None:
        log.append("[ERROR] [Legacy] Kepler kinematics unavailable (insufficient night mean or valid r^2>=0.05 points). angularVelocity=None.")

    record = {
        "id": sid,
        "period": period,
        "tau": tau,
        "dim": dim,
        "earlyDelay": kinetics_res.get("earlyDelay"),
        "relaxationTime": kinetics_res.get("relaxationTime"),
        "ar1": csd_res.get("ar1"),
        "nightVariance": csd_res.get("variance"),
        "nightSkewness": csd_res.get("skewness"),
        "angularVelocity": kepler_res.get("angularVelocity"),
        "sweepRate": kepler_res.get("sweepRate"),
        "ascendFriction": ascend_friction,
        "nightFriction": night_friction,
        "events": log,
    }
    for k, v in subject.items():
        if k not in ("timestamps", "values", "id"):
            record[k] = v
    return record


def run_subject_legacy_group1(subject, period="night", tau_max=120):
    """Companion to run_subject_legacy() for the 2026-08-19 Group-1 test plan (dataset_fleet_
    registry.md changelog "第一组6张中性卡的冗余审计基础设施搭建"): computes the 6
    purely-neutral (no warn/bad color coding) legacy-JS-ported candidate metrics (Volume,
    Recovery, Shape Ratio λ1/λ2, Box-Counting Dimension, Lyapunov, Core Dist) via
    _legacy_metrics_group1_v4.py.

    Track conventions (mirrors index_v4.html's default chk-smooth=checked UI state, Blueprint
    v3.3 L0 2.2 Dual-Track Law, and computeAttractorMetrics()'s three-argument call signature
    at index_v4.html:3669):
      - Volume / Shape Ratio / Box-Counting Dimension: period-sliced SMOOTH points
        (`shapePoints` == `pointsSmooth` under the production default smooth=true toggle,
        since `currentValues` then equals `smoothValues` -- index_v4.html:3639 vs 3659).
      - Recovery: period-sliced RAW points (`rawPoints`, always unsmoothed per the Dual-Track
        Law -- index_v4.html:2902/3658).
      - Lyapunov: period-sliced SMOOTH points (`smoothPoints` -- index_v4.html:2914/3659).
      - Core Dist: index_v4.html:3754-3757 special-cases period==='night' to a HARDCODED 0
        (same-night-as-itself is definitionally zero displacement), which would make Core
        Dist untestable under this harness's period="night" default. This function therefore
        ALWAYS additionally computes coreDistAll using the SAME "all"-period gravity core the
        production UI actually displays Core Dist against (nightCore is unconditionally the
        RAW night-sliced track regardless of `period`, matching Night Friction's night_core in
        run_subject_legacy -- index_v4.html:3645-3650), independent of whatever `period` was
        requested for the other five metrics.
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
    tau_res = json.loads(eng.engine.extract_tau(raw_json, max_lag=tau_max))
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

    # --- Group-1 metrics on the requested period (default "night") ---
    period_raw_vs = slice_by_period(ts, raw_vs, period)
    period_smooth_vs = slice_by_period(ts, smooth_vs, period)
    points_shape = takens_embedding(period_smooth_vs, tau, dim)
    points_raw = takens_embedding(period_raw_vs, tau, dim)
    points_smooth = points_shape  # identical under default smooth=true (currentValues==smoothValues)

    g1_metrics = legacy_g1.compute_group1_attractor_metrics(points_shape, points_raw, points_smooth)
    if g1_metrics.get("volume") is None:
        log.append(f"[ERROR] [Group1] compute_volume_shape returned None (<4 valid {period}-sliced shape points). volume/shapeRatio/dimension/lyapunov unaffected fields stay independently computed.")
    if g1_metrics.get("dimension") is None:
        log.append(f"[ERROR] [Group1] box_counting_dimension returned None (<20 valid {period}-sliced shape points).")
    if g1_metrics.get("lyapunov") is None:
        log.append(f"[ERROR] [Group1] lyapunov_proxy returned None (<30 valid {period}-sliced smooth points or <10 divergence samples).")

    # --- Core Dist: ALWAYS computed on the "all"-period gravity core vs the RAW night core,
    # independent of `period` above (see docstring: period="night" would trivialize this to 0
    # by construction, matching index_v4.html:3754-3757's own special-case). ---
    all_smooth_vs = slice_by_period(ts, smooth_vs, "all")
    points_shape_all = takens_embedding(all_smooth_vs, tau, dim)
    vs_all = legacy_g1.compute_volume_shape(points_shape_all)

    night_raw_vs = slice_by_period(ts, raw_vs, "night")
    night_points_all = takens_embedding(night_raw_vs, tau, dim)
    night_points_valid = [p for p in night_points_all if p is not None]
    core_dist_all = None
    if vs_all is not None and len(night_points_valid) > 0:
        night_core = [_legacy_median([p[d] for p in night_points_valid]) for d in range(dim)]
        core_dist_all = legacy_g1.calc_distance(vs_all["gravityCore"], night_core)
        log.append(f"[INFO] [Group1] Core Dist (all-period gravity core vs RAW night core) computed from {len(night_points_valid)} night points.")
    else:
        log.append("[ERROR] [Group1] coreDistAll unavailable (insufficient all-period shape points or no valid night points).")

    record = {
        "id": sid,
        "period": period,
        "tau": tau,
        "dim": dim,
        "volume": g1_metrics.get("volume"),
        "shapeRatio": g1_metrics.get("shapeRatio"),
        "avgRecovery": g1_metrics.get("avgRecovery"),
        "boxCountingDim": g1_metrics.get("dimension"),
        "lyapunov": g1_metrics.get("lyapunov"),
        "coreDistAll": core_dist_all,
        "events": log,
    }
    for k, v in subject.items():
        if k not in ("timestamps", "values", "id"):
            record[k] = v
    return record


def run_cohort_legacy_group1(cohort_name, subjects, period="night", tau_max=120):
    results = [run_subject_legacy_group1(s, period=period, tau_max=tau_max) for s in subjects]
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"[Group1] Processed {len(subjects)} {cohort_name} subjects: {len(ok)} succeeded, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")
    return results, ok, failed


def run_cohort_legacy(cohort_name, subjects, period="night", tau_max=120):
    results = [run_subject_legacy(s, period=period, tau_max=tau_max) for s in subjects]
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"[Legacy] Processed {len(subjects)} {cohort_name} subjects: {len(ok)} succeeded, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")
    return results, ok, failed


def run_cohort(cohort_name, subjects, period="night", tau_max=120):
    results = [run_subject(s, period=period, tau_max=tau_max) for s in subjects]
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
