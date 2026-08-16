"""
Wind-Tunnel Cross-Validation: Vector 1/2 Standardized 75g OGTT Perturbation
Cohort: Stanford OGTT-CGM (21 subjects with SSPG gold-standard label)

Purpose: Independent cross-source replication of the CGMacros meal-dynamics
candidate operators (w_carb, strain_carb, tau_relax) registered in
reports/candidate_tensor_staging_matrix.md. The OGTT here is a FIXED,
protocol-controlled 75g glucose load (not free-living meals), making this a
genuinely independent perturbation source per the Staging Matrix's
"Cross-Source Replication" criterion (Section 9.3 multi-cohort standard).

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: purely thermodynamic differential/
    integral operators; no ML regression, no label weights.
  - Section 9.1.2 Labels as Prisms, Not Targets: SSPG / sspg_class / di are
    used strictly for post-hoc grouping and rank-separation evaluation.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory only.
  - Staging Matrix discipline: this script and its output remain in
    validate/ and reports/ -- no production code or Blueprint is touched
    regardless of outcome.
"""
import json
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_ogtt_subject(subject):
    sid = subject["id"]
    tp = np.array(subject["timepoints_min"], dtype=float)
    gl = np.array(subject["values_mmol"], dtype=float)
    carbs = subject["carbs_g"]

    order = np.argsort(tp)
    tp, gl = tp[order], gl[order]

    # Interpolate onto a 1-minute grid from -10 to 180 for consistent
    # integration/threshold-crossing resolution (same spirit as the
    # CGMacros meal script's 1-min resample).
    grid = np.arange(tp.min(), tp.max() + 1, 1.0)
    gl_grid = np.interp(grid, tp, gl)

    # Baseline: min glucose in [-10, 10] minutes relative to load (t=0)
    base_mask = (grid >= max(tp.min(), -10)) & (grid <= 10)
    if not base_mask.any():
        return {"id": sid, "error": "No baseline window available."}
    g_base = float(gl_grid[base_mask].min())

    # Peak search window: [0, 150] minutes post-load
    peak_mask = (grid >= 0) & (grid <= 150)
    if not peak_mask.any():
        return {"id": sid, "error": "No peak search window available."}
    peak_idx_local = np.argmax(gl_grid[peak_mask])
    g_peak = float(gl_grid[peak_mask][peak_idx_local])
    t_peak = float(grid[peak_mask][peak_idx_local])

    delta_g = g_peak - g_base
    if delta_g <= 0.2:
        return {"id": sid, "error": f"Insignificant glucose rise (delta_g={delta_g:.3f})."}

    t_ascend_min = max(1.0, t_peak - 0.0)
    ascend_slope = delta_g / t_ascend_min
    strain_per_carb = delta_g / carbs

    # Relaxation: from t_peak to when glucose falls back to g_base + max(0.5, 0.2*delta_g)
    relax_thresh = g_base + max(0.5, 0.2 * delta_g)
    post_peak_mask = grid >= t_peak
    post_peak_grid = grid[post_peak_mask]
    post_peak_gl = gl_grid[post_peak_mask]
    below = post_peak_gl <= relax_thresh
    if below.any():
        t_recovered = float(post_peak_grid[np.argmax(below)])
        tau_relax = t_recovered - t_peak
    else:
        tau_relax = float(grid.max() - t_peak)  # Capped at window edge, not fabricated beyond it

    # Work integral over [0, t_max] of (G(t) - G_base)+
    full_mask = grid >= 0
    elevations = np.maximum(0.0, gl_grid[full_mask] - g_base)
    work_meal = float(np.trapezoid(elevations, dx=1.0)) if hasattr(np, "trapezoid") else float(np.trapz(elevations, dx=1.0))
    specific_work = work_meal / carbs

    return {
        "id": sid,
        "source_location": subject.get("source_location"),
        "g_base": g_base,
        "g_peak": g_peak,
        "delta_g": delta_g,
        "t_ascend_min": t_ascend_min,
        "ascend_slope": ascend_slope,
        "strain_per_carb": strain_per_carb,
        "tau_relax_min": tau_relax,
        "work_meal": work_meal,
        "specific_work": specific_work,
        # Labels: prism-only (Section 9.1.2).
        "sspg": subject.get("sspg"),
        "sspg_class": subject.get("sspg_class"),
        "di": subject.get("di"),
    }


def main():
    data_path = Path("output/stanford_ogtt_subjects.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    results = [analyze_ogtt_subject(s) for s in subjects]
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"Processed {len(subjects)} Stanford OGTT subjects: {len(ok)} succeeded, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")

    out_path = Path("reports")
    out_path.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_path / f"wind_tunnel_stanford_ogtt_dynamics_{ts_tag}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "stanford_ogtt",
            "protocol": "standardized_75g_ogtt_perturbation_dynamics",
            "carbs_g": data["carbs_g"],
            "n_total": len(subjects),
            "n_success": len(ok),
            "n_failed": len(failed),
            "results": results,
        }, f, indent=2)

    print(f"Stanford OGTT perturbation dynamics results saved to {out_file}")


if __name__ == "__main__":
    main()
