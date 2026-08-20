"""
Research prototype (NOT a verbatim JS port, NOT wired into the production
run_subject_legacy() path): redesign attempt for candidate_tensor_staging_matrix.md
候选 #5 `relaxationTime`, per user-approved direction "重新设计 relaxationTime 的事件检测
算法" (2026-08-19).

Anomaly Targeting (Section 4.B step 1): the original computeExcursionKinetics port
(_legacy_metrics_v4.py) finds the 50%-decay crossing by scanning for the first RAW SAMPLE
INDEX where value <= target, i.e. snaps the crossing time to whatever sample happens to be
available on the native sampling grid. Stanford (Dexcom, ~5min) and Shanghai (~15min,
inferred from Shanghai's relaxationTime absolute range being roughly 2x Stanford's: 42-108
vs 27.5-62.5 min) have different native grids, so the SAME underlying decay dynamics would
be quantized to different discrete bins depending on cohort -- a point-crossing measurement
artifact, not a physiological difference (AGENTS.md Axiom 3: 强制延迟嵌入/数学升维 --
prefer continuous curve-based measures over single-point threshold crossings).

High-Dimensional Hypothesis (Section 4.B step 2): linearly interpolating the exact crossing
TIME between the two bracketing samples (rather than snapping to the later sample's
timestamp) should reduce this sampling-grid quantization noise without introducing any new
free parameters (Zero Magic-Constant axiom) or touching the peak/valley detection logic
that already works. This is a continuous-measurement upgrade of the SAME algorithm, not a
new algorithm.
"""
import datetime as dt
import math
from typing import List, Optional


def compute_excursion_kinetics_interp(times: List[dt.datetime], vals: List[Optional[float]]) -> dict:
    """Identical peak/valley/qualifying-excursion detection to
    _legacy_metrics_v4.compute_excursion_kinetics; the ONLY change is that the 50%-decay
    relaxation-time crossing is linearly interpolated between the bracketing samples instead
    of snapped to the later sample's timestamp. earlyDelay is left byte-identical (it is a
    max-rate-of-rise point between adjacent samples, not a threshold crossing, so
    interpolation does not apply the same way and is out of scope for this redesign).
    """
    if not vals or not times or len(vals) < 10:
        return {"earlyDelay": None, "relaxationTime": None, "relaxationTime_raw": None,
                "n_qualifying_excursions": 0, "n_undecayed_fallback": 0}

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
    relax_times_interp: List[float] = []
    relax_times_raw: List[float] = []
    relax_times_decayed_only: List[float] = []
    n_undecayed = 0

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

        next_valley = None
        for v in valleys:
            if v["idx"] > p["idx"]:
                next_valley = v
                break
        search_end = next_valley["idx"] if next_valley is not None else len(vals) - 1

        relax_end_idx = None
        for i in range(p["idx"], search_end + 1):
            if vals[i] is not None and vals[i] <= target_val:
                relax_end_idx = i
                break

        if relax_end_idx is not None:
            raw_minutes = (times[relax_end_idx] - times[p["idx"]]).total_seconds() / 60.0
            relax_times_raw.append(raw_minutes)
            if relax_end_idx == p["idx"]:
                relax_times_interp.append(0.0)
            else:
                v0, v1 = vals[relax_end_idx - 1], vals[relax_end_idx]
                t0, t1 = times[relax_end_idx - 1], times[relax_end_idx]
                if v0 is None or v1 is None or v0 == v1:
                    frac = 1.0
                else:
                    frac = (v0 - target_val) / (v0 - v1)
                    frac = min(1.0, max(0.0, frac))
                crossing_time = t0 + (t1 - t0) * frac
                crossing_minutes = (crossing_time - times[p["idx"]]).total_seconds() / 60.0
                relax_times_interp.append(crossing_minutes)
                relax_times_decayed_only.append(crossing_minutes)
        else:
            # v3 ablation path (Zero Magic-Constant axiom, AGENTS.md Section 8.3): the
            # original JS's "duration * 1.5" fallback fabricates a number for an event whose
            # true decay time was never observed (censored by the next valley/end of
            # record). relaxationTime_decayed_only excludes these censored events entirely
            # (honest Fail-Closed at the event level) instead of feeding a fabricated
            # constant into the per-subject median; relaxationTime keeps the original
            # fallback behavior for direct comparison.
            duration = (times[search_end] - times[p["idx"]]).total_seconds() / 60.0
            relax_times_raw.append(duration * 1.5)
            relax_times_interp.append(duration * 1.5)
            n_undecayed += 1

    def _local_median(arr: List[float]) -> Optional[float]:
        if not arr:
            return None
        s = sorted(arr)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0

    return {
        "earlyDelay": _local_median(early_delays),
        "relaxationTime": _local_median(relax_times_interp),
        "relaxationTime_raw": _local_median(relax_times_raw),
        "relaxationTime_decayed_only": _local_median(relax_times_decayed_only),
        "n_qualifying_excursions": len(relax_times_interp),
        "n_undecayed_fallback": n_undecayed,
    }
