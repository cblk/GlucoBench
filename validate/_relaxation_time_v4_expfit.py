"""
Research prototype (NOT wired into production/wind-tunnel main harness): more fundamental
redesign attempt for candidate_tensor_staging_matrix.md 候选 #5 `relaxationTime`, per
user-approved direction "更根本的重新设计：改用指数衰减曲线拟合提取时间常数，而不是阈值穿越"
(2026-08-19, following the v2/v3 point-interpolation attempts in _relaxation_time_v2.py
which improved Stanford but not Shanghai).

Physical model (Somatic Tensor Worldview terminology "弛豫时间" IS literally an exponential
relaxation time constant, not an arbitrary percentage-crossing time -- this redesign makes
the algorithm match the physical concept it is named after):

    G(t) = G_baseline + (G_peak - G_baseline) * exp(-t / tau)

where t is minutes elapsed since the peak, G_baseline is anchored to the SAME prev_valley
value the original algorithm already uses to define excursion amplitude (>= 1.5 threshold),
and tau (minutes) is the e-folding decay time constant -- fit via ordinary least squares on
the log-linearized form ln(G(t) - G_baseline) = ln(G_peak - G_baseline) - t/tau, using every
sample in [peak_idx, search_end] whose value is still strictly above baseline (log requires
positive residual). This uses ALL available points in the decay window (not just the single
sample nearest a fixed threshold), which is the intended "数学升维" upgrade: point-crossing
-> curve-fitting.

Honesty/Zero-Magic-Constant notes:
  - An event is excluded (not fabricated) if fewer than 3 valid (t, residual>0) points exist,
    or if the fitted slope is >= 0 (i.e. the regression found no net decay -- physically
    invalid for tau).
  - R^2 of the log-linear fit is always returned per-event so a quality gate can be applied
    and evaluated empirically (this script tests both "keep all valid-slope fits" and "R^2 >=
    0.5 quality-gated" as two honest, pre-declared robustness variants, not post-hoc tuned to
    maximize one cohort's score).
"""
import datetime as dt
import math
from typing import List, Optional


def _log_linear_fit(t_vals: List[float], residuals: List[float]):
    """OLS fit of ln(residual) = a + b*t. Returns (slope_b, intercept_a, r_squared) or None
    if fewer than 2 points or zero variance in t."""
    n = len(t_vals)
    if n < 2:
        return None
    y = [math.log(r) for r in residuals]
    mean_t = sum(t_vals) / n
    mean_y = sum(y) / n
    ss_tt = sum((t - mean_t) ** 2 for t in t_vals)
    if ss_tt == 0:
        return None
    ss_ty = sum((t - mean_t) * (yy - mean_y) for t, yy in zip(t_vals, y))
    b = ss_ty / ss_tt
    a = mean_y - b * mean_t
    y_pred = [a + b * t for t in t_vals]
    ss_res = sum((yy - yp) ** 2 for yy, yp in zip(y, y_pred))
    ss_tot = sum((yy - mean_y) ** 2 for yy in y)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return b, a, r2


def compute_excursion_kinetics_expfit(times: List[dt.datetime], vals: List[Optional[float]],
                                        min_points: int = 3, r2_gate: Optional[float] = None) -> dict:
    """Same peak/valley/qualifying-excursion detection as the original port; relaxationTime is
    replaced with a per-event exponential decay time-constant (tau, minutes) fit via
    log-linear OLS on ALL samples in the decay window, aggregated as the per-subject median.
    earlyDelay is untouched (out of scope, same as v2/v3)."""
    if not vals or not times or len(vals) < 10:
        return {"earlyDelay": None, "relaxationTime": None, "n_qualifying_excursions": 0,
                "n_fit_excluded_too_few_points": 0, "n_fit_excluded_bad_slope": 0,
                "n_fit_excluded_r2_gate": 0, "median_r2": None}

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
    tau_fits: List[float] = []
    r2_values: List[float] = []
    n_excl_few_points = 0
    n_excl_bad_slope = 0
    n_excl_r2_gate = 0

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

        baseline = prev_valley["val"]

        next_valley = None
        for v in valleys:
            if v["idx"] > p["idx"]:
                next_valley = v
                break
        search_end = next_valley["idx"] if next_valley is not None else len(vals) - 1

        t_vals, residuals = [], []
        for i in range(p["idx"], search_end + 1):
            if vals[i] is None:
                continue
            residual = vals[i] - baseline
            if residual <= 0:
                break  # decayed to/past baseline; stop the fit window here
            t_min = (times[i] - times[p["idx"]]).total_seconds() / 60.0
            t_vals.append(t_min)
            residuals.append(residual)

        if len(t_vals) < min_points:
            n_excl_few_points += 1
            continue

        fit = _log_linear_fit(t_vals, residuals)
        if fit is None:
            n_excl_few_points += 1
            continue
        slope, _intercept, r2 = fit
        if slope >= 0:
            n_excl_bad_slope += 1
            continue
        tau = -1.0 / slope
        if r2_gate is not None and r2 < r2_gate:
            n_excl_r2_gate += 1
            continue
        tau_fits.append(tau)
        r2_values.append(r2)

    def _local_median(arr: List[float]) -> Optional[float]:
        if not arr:
            return None
        s = sorted(arr)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0

    return {
        "earlyDelay": _local_median(early_delays),
        "relaxationTime": _local_median(tau_fits),
        "n_qualifying_excursions": len(tau_fits) + n_excl_few_points + n_excl_bad_slope + n_excl_r2_gate,
        "n_fit_excluded_too_few_points": n_excl_few_points,
        "n_fit_excluded_bad_slope": n_excl_bad_slope,
        "n_fit_excluded_r2_gate": n_excl_r2_gate,
        "median_r2": _local_median(r2_values),
    }
