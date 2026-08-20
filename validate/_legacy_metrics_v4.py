"""
Faithful Python ports of six pre-doctrinal (v8.0-v8.4 era) legacy JS metrics from
index_v4.html, for Wind-Tunnel testing purposes ONLY (AGENTS.md Section 9). Ported for the
2026-08-19 dataset_fleet_registry.md changelog entry "开始进行第二组6张判色卡的风洞测试基础
设施搭建" (Early Phase Delay / Relaxation Time / AR1 / Angular Velocity / Ascend Friction /
Night Friction).

Why this file exists instead of just importing index_v4.html's PYTHON_ENGINE_CODE like
_extracted_tensor_engine_v4.py does: these six cards currently have NO Python/Pyodide
implementation in production at all -- they are pure JavaScript (computeCriticalSlowingDown,
computeAsymmetricFriction, computeExcursionKinetics, computeKeplerKinematics). There is no
existing "production Python" for Section 9.4's Bit-for-Bit Truth doctrine to mirror; this
module IS the translation, hand-ported line-for-line from the JS source (exact line ranges
cited per function below, as of commit bf8113a / 2026-08-19 16:10) and cross-checked against
the actual JS via Node.js (see _js_legacy_metrics_crosscheck.mjs and
reports/legacy_metrics_js_vs_python_crosscheck_*.json) rather than trusted on manual
transcription alone.

Section 9.5 Product Isolation applies in full: NOTHING here is wired into index_v4.html.
The JS remains the sole production source of truth for what users actually see; this module
is a read-only research proxy of that JS. If a metric here ever clears Section 9.3
Topological Victory and the full B.5 Atomic Blueprint Evolution sign-off, the correct next
step is to build a *production* Pyodide-resident Python engine method from scratch (and
verify IT against the still-live JS), not to promote this file wholesale into
_extracted_tensor_engine_v4.py.
"""
import datetime as dt
import math
from typing import List, Optional, Sequence


def _median(arr: Sequence[float]) -> float:
    """Port of the global getMedian(arr), index_v4.html:1692-1697. Returns 0.0 for an
    empty input (matches JS's `if (!arr || arr.length === 0) return 0`), NOT None --
    callers that need to distinguish "no data" must check emptiness before calling this."""
    if not arr:
        return 0.0
    s = sorted(arr)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0


def compute_critical_slowing_down(times: List[dt.datetime], vals: List[Optional[float]]) -> dict:
    """Port of computeCriticalSlowingDown(times, vals), index_v4.html:2791-2841. Feeds the
    "夜间临界慢化 (AR1)" card.

    Per Contract v1.3 4.3 "No-Cross-Night Pooling Law": AR1/variance/skewness are computed
    independently per natural night (00:00-06:00, grouped by calendar day) on the RAW series,
    then aggregated via median across nights -- never pooled into one flat array (pooling
    would fabricate a spurious lag-1 jump at every night-to-night seam). Call site passes the
    FULL (not period-sliced) raw series: computeCriticalSlowingDown(timestamps, rawValues),
    index_v4.html:3663.
    """
    nights_by_day: dict = {}
    for t, v in zip(times, vals):
        if t is None or v is None:
            continue
        if 0 <= t.hour < 6:
            day_key = (t.year, t.month, t.day)
            nights_by_day.setdefault(day_key, []).append(v)

    per_night_ar1: List[float] = []
    per_night_var: List[float] = []
    per_night_skew: List[float] = []

    for night_vals in nights_by_day.values():
        n = len(night_vals)
        if n < 30:
            continue
        mean = sum(night_vals) / n
        variance = sum((b - mean) ** 2 for b in night_vals) / (n - 1)
        m3 = sum((b - mean) ** 3 for b in night_vals) / n
        skewness = m3 / (variance ** 1.5) if variance > 0 else 0.0

        num = sum((night_vals[i] - mean) * (night_vals[i - 1] - mean) for i in range(1, n))
        den = sum((v - mean) ** 2 for v in night_vals)
        ar1 = num / den if den > 1e-8 else 0.0

        per_night_ar1.append(ar1)
        per_night_var.append(variance)
        per_night_skew.append(skewness)

    if not per_night_ar1:
        return {"ar1": None, "variance": None, "skewness": None}
    return {
        "ar1": _median(per_night_ar1),
        "variance": _median(per_night_var),
        "skewness": _median(per_night_skew),
    }


def compute_asymmetric_friction(points: List[Optional[List[float]]], core: Optional[List[float]]) -> dict:
    """Port of computeAsymmetricFriction(points, core), index_v4.html:2485-2592. Feeds BOTH
    "夜间相变阻力 (Night Friction)" (via asymFriction, on night-sliced RAW points/nightCore)
    and "上升相阻力 (Ascend Friction)" (via ascendFriction, on period-sliced RAW points against
    the period-sliced gravity core) -- see index_v4.html:3640-3656 and 2881-2884 for the two
    call sites and which track feeds which card.

    `points`: Takens-embedded phase-space series (each element a dim-D coordinate list, or
    None for a gap). `core`: the dim-D reference point (gravity core / night core) friction is
    measured relative to.

    Returns asymFriction (avg friction during the DESCENDING phase -> Night Friction),
    ascendFriction (avg friction during the ASCENDING phase -> Ascend Friction), workIntegral
    (a legacy 2D successive-distance calc that neither target card actually uses -- production
    Work Integral comes exclusively from the Python engine's compute_work_integral per the
    Dual-Track Law; kept here only so this is a complete 1:1 port of the JS function) and
    frictionGradient (inner/outer descending-friction ratio, also unused by either target
    card).
    """
    if not points or len(points) < 3 or core is None:
        return {"asymFriction": None, "workIntegral": None, "ascendFriction": None, "frictionGradient": None}

    dim = 3
    for p in points:
        if p is not None:
            dim = len(p)
            break

    friction_sum = 0.0
    count = 0
    work_integral = 0.0
    asc_friction_sum = 0.0
    asc_count = 0
    desc_distances: List[float] = []
    desc_frictions: List[float] = []

    for i in range(1, len(points)):
        p_i, p_prev = points[i], points[i - 1]
        if p_i is None or p_prev is None:
            continue
        v_g = p_i[0] - p_prev[0]

        v_sq = 0.0
        dist_sq = 0.0
        for d in range(dim):
            v = p_i[d] - p_prev[d]
            v_sq += v * v
            dist = p_i[d] - core[d]
            dist_sq += dist * dist
        len_v = math.sqrt(v_sq)
        len_dist = math.sqrt(dist_sq)

        if dim >= 2:
            dx = p_i[0] - p_prev[0]
            dy = p_i[1] - p_prev[1]
            work_integral += math.sqrt(dx * dx + dy * dy)

        if len_v > 1e-6:
            current_fric = len_dist / len_v
            if v_g < -0.01:
                friction_sum += current_fric
                count += 1
                desc_distances.append(len_dist)
                desc_frictions.append(current_fric)
            elif v_g > 0.01:
                asc_friction_sum += current_fric
                asc_count += 1

    asym_friction = friction_sum / count if count > 0 else None
    ascend_friction = asc_friction_sum / asc_count if asc_count > 0 else None

    valid_phase_points = sum(1 for p in points if p is not None)
    if valid_phase_points > 0:
        work_integral = work_integral / (valid_phase_points / 480.0)

    friction_gradient = None
    if count >= 4:
        sorted_dists = sorted(desc_distances)
        mid = len(sorted_dists) // 2
        median_dist = (
            sorted_dists[mid]
            if len(sorted_dists) % 2 != 0
            else (sorted_dists[mid - 1] + sorted_dists[mid]) / 2.0
        )

        inner_sum, inner_count = 0.0, 0
        outer_sum, outer_count = 0.0, 0
        for i in range(count):
            if desc_distances[i] < median_dist:
                inner_sum += desc_frictions[i]
                inner_count += 1
            else:
                outer_sum += desc_frictions[i]
                outer_count += 1
        if inner_count > 0 and outer_count > 0:
            inner_fric = inner_sum / inner_count
            outer_fric = outer_sum / outer_count
            if inner_fric > 1e-6:
                friction_gradient = outer_fric / inner_fric

    return {
        "asymFriction": asym_friction,
        "workIntegral": work_integral,
        "ascendFriction": ascend_friction,
        "frictionGradient": friction_gradient,
    }


def compute_excursion_kinetics(times: List[dt.datetime], vals: List[Optional[float]]) -> dict:
    """Port of computeExcursionKinetics(times, vals), index_v4.html:2698-2785. Feeds
    "早相加速度迟滞 (Time-to-Decel)" (earlyDelay) and "弛豫衰减疲劳度 (Relaxation Time)"
    (relaxationTime).

    Operates on the FULL (not period-sliced) SMOOTH-track series -- matches the JS call site
    computeExcursionKinetics(timestamps, smoothValues), index_v4.html:3662.
    """
    if not vals or not times or len(vals) < 10:
        return {"earlyDelay": None, "relaxationTime": None}

    peaks: List[dict] = []
    valleys: List[dict] = []
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

    early_delays: List[float] = []
    relax_times: List[float] = []

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

        max_dv = -math.inf
        max_dv_idx = prev_valley["idx"]
        for i in range(prev_valley["idx"], p["idx"]):
            if vals[i] is not None and vals[i + 1] is not None:
                dv = vals[i + 1] - vals[i]
                if dv > max_dv:
                    max_dv = dv
                    max_dv_idx = i
        early_delays.append((times[max_dv_idx] - times[prev_valley["idx"]]).total_seconds() / 60.0)

        target_val = p["val"] - 0.5 * (p["val"] - prev_valley["val"])
        relax_end_idx = p["idx"]
        found = False

        next_valley = None
        for v in valleys:
            if v["idx"] > p["idx"]:
                next_valley = v
                break
        search_end = next_valley["idx"] if next_valley is not None else len(vals) - 1

        for i in range(p["idx"], search_end + 1):
            if vals[i] is not None and vals[i] <= target_val:
                relax_end_idx = i
                found = True
                break

        if found:
            relax_times.append((times[relax_end_idx] - times[p["idx"]]).total_seconds() / 60.0)
        else:
            duration = (times[search_end] - times[p["idx"]]).total_seconds() / 60.0
            relax_times.append(duration * 1.5)

    def _local_median(arr: List[float]) -> Optional[float]:
        # Port of the LOCAL getMedian defined inside computeExcursionKinetics
        # (index_v4.html:2773-2779) -- returns None (JS null) for empty input, unlike the
        # global _median() above which returns 0.0. This distinction matters here because
        # an empty early_delays/relax_times means "no valid excursion found", which the JS
        # (and this port) must report as N/A, not a fabricated zero.
        if not arr:
            return None
        s = sorted(arr)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0

    return {
        "earlyDelay": _local_median(early_delays),
        "relaxationTime": _local_median(relax_times),
    }


def compute_kepler_kinematics(
    times: List[dt.datetime], vals: List[Optional[float]], core: Optional[float]
) -> dict:
    """Port of computeKeplerKinematics(times, vals, core), index_v4.html:2630-2695. Feeds
    "代谢引力场角速度 (Angular Vel.)".

    `core` here is a SCALAR (nightMean), not a phase-space vector -- distinct from the `core`
    parameter of compute_asymmetric_friction despite the shared name in both the JS source and
    this port. Operates on the FULL (not period-sliced) SMOOTH-track series, matching the JS
    call site computeKeplerKinematics(timestamps, smoothValues, nightMean), index_v4.html:3664.

    T_hours differences are computed here via direct timedelta subtraction rather than via an
    absolute epoch-hours array (as the JS does with Date.getTime()/(3600*1000)): since every
    use of T_hours in the JS is a DIFFERENCE between two entries, this is mathematically
    identical to the JS's own arithmetic and avoids Python naive-datetime .timestamp()
    ambiguity, which is not a translation shortcut but the numerically exact equivalent.
    """
    if not vals or not times or len(vals) < 3 or core is None:
        return {"angularVelocity": None, "sweepRate": None}

    n = len(vals)
    X: List[Optional[float]] = [None] * n
    Y: List[Optional[float]] = [None] * n

    for i in range(n):
        X[i] = (vals[i] - core) if vals[i] is not None else None

    def hours_between(a: dt.datetime, b: dt.datetime) -> float:
        return (a - b).total_seconds() / 3600.0

    for i in range(n):
        if X[i] is None:
            Y[i] = None
            continue
        if 0 < i < n - 1 and X[i - 1] is not None and X[i + 1] is not None:
            Y[i] = (X[i + 1] - X[i - 1]) / hours_between(times[i + 1], times[i - 1])
        elif i == 0 and X[i + 1] is not None:
            Y[i] = (X[i + 1] - X[i]) / hours_between(times[i + 1], times[i])
        elif i == n - 1 and X[i - 1] is not None:
            Y[i] = (X[i] - X[i - 1]) / hours_between(times[i], times[i - 1])
        else:
            Y[i] = None

    dY: List[Optional[float]] = [None] * n
    for i in range(n):
        if Y[i] is None:
            dY[i] = None
            continue
        if 0 < i < n - 1 and Y[i - 1] is not None and Y[i + 1] is not None:
            dY[i] = (Y[i + 1] - Y[i - 1]) / hours_between(times[i + 1], times[i - 1])
        else:
            dY[i] = None

    sum_sweep = 0.0
    sum_r2 = 0.0
    valid_count = 0
    for i in range(1, n - 1):
        if X[i] is None or Y[i] is None or dY[i] is None:
            continue
        r2 = X[i] * X[i] + Y[i] * Y[i]
        if r2 < 0.05:
            continue
        sweep = 0.5 * abs(X[i] * dY[i] - Y[i] * Y[i])
        sum_sweep += sweep
        sum_r2 += r2
        valid_count += 1

    if valid_count < 5 or sum_r2 == 0:
        return {"angularVelocity": None, "sweepRate": None}

    avg_sweep_rate = sum_sweep / valid_count
    weighted_angular_vel = 2.0 * sum_sweep / sum_r2
    return {"angularVelocity": weighted_angular_vel, "sweepRate": avg_sweep_rate}
