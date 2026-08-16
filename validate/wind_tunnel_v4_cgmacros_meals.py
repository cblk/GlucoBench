"""
Wind-Tunnel Experiment: Vector 1/2 Meal Perturbation Dynamics on CGMacros
Source: PhysioNet 2026 / CGMacros (45 subjects, 1706 meals)

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: purely thermodynamic/mechanical
    differential and integral operators, no label weights, no ML regression.
  - Section 9.1.2 Labels as Prisms, Not Targets: group_a1c / homa_ir are used
    strictly for post-hoc grouping and rank separation evaluation.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory.
"""
import json
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_subject_meals(subject, min_carbs=25.0, window_min=240):
    sid = subject["id"]
    ts_list = [dt.datetime.fromisoformat(t) for t in subject["timestamps"]]
    vs_list = subject["values"]

    # Create continuous minute-level series using pandas
    ts_series = pd.Series(vs_list, index=pd.to_datetime(ts_list))
    # Resample to 1-min grid and interpolate small gaps
    ts_1min = ts_series.resample("1min").interpolate(method="time", limit=15)

    meals = subject.get("meals", [])
    valid_meals = []

    for idx, m in enumerate(meals):
        carbs = m.get("carbs", 0.0)
        if carbs < min_carbs:
            continue

        t_meal = dt.datetime.fromisoformat(m["timestamp"])
        t_pre = t_meal - dt.timedelta(minutes=30)
        t_post = t_meal + dt.timedelta(minutes=window_min)

        # Slice 1-min window
        win = ts_1min.loc[t_pre:t_post]
        if win.empty or win.isna().sum() > 30:
            continue

        # Pre-meal baseline: min glucose in [-30m, +10m]
        pre_win = win.loc[t_pre:t_meal + dt.timedelta(minutes=10)]
        if pre_win.empty:
            continue
        g_base = float(pre_win.min())

        # Post-meal search window for peak: [t_meal, t_meal + 180m]
        post_win = win.loc[t_meal:t_meal + dt.timedelta(minutes=180)]
        if post_win.empty:
            continue
        g_peak = float(post_win.max())
        t_peak = post_win.idxmax()

        delta_g = g_peak - g_base
        if delta_g <= 0.2:  # Insignificant rise
            continue

        t_ascend_min = max(1.0, (t_peak - t_meal).total_seconds() / 60.0)
        ascend_slope = delta_g / t_ascend_min  # mmol/L per min
        strain_per_carb = delta_g / carbs      # mmol/L per g carb

        # Relaxation time: from t_peak until glucose drops to g_base + 0.2*delta_g or g_base + 0.5
        relax_thresh = g_base + max(0.5, 0.2 * delta_g)
        post_peak = win.loc[t_peak:t_post]
        below_thresh = post_peak[post_peak <= relax_thresh]

        if not below_thresh.empty:
            t_recovered = below_thresh.index[0]
            tau_relax = (t_recovered - t_peak).total_seconds() / 60.0
        else:
            tau_relax = (t_post - t_peak).total_seconds() / 60.0  # Capped at window limit

        # Meal Work Integral: Integral of (G(t) - G_base)+ over [t_meal, t_post]
        post_full = win.loc[t_meal:t_post]
        elevations = np.maximum(0.0, post_full.values - g_base)
        # Trapezoidal integration across 1-minute steps
        work_meal = float(np.trapezoid(elevations, dx=1.0)) if hasattr(np, "trapezoid") else float(np.trapz(elevations, dx=1.0))
        specific_work = work_meal / carbs  # (mmol*min/L) per g carb

        valid_meals.append({
            "meal_type": m.get("meal_type"),
            "carbs": carbs,
            "calories": m.get("calories"),
            "g_base": g_base,
            "g_peak": g_peak,
            "delta_g": delta_g,
            "t_ascend_min": t_ascend_min,
            "ascend_slope": ascend_slope,
            "strain_per_carb": strain_per_carb,
            "tau_relax_min": tau_relax,
            "work_meal": work_meal,
            "specific_work": specific_work,
        })

    if not valid_meals:
        return {
            "id": sid,
            "n_meals": 0,
            "error": "No valid meal challenges found."
        }

    # Aggregate subject-level mean & median dynamic metrics
    df_m = pd.DataFrame(valid_meals)
    return {
        "id": sid,
        "n_meals": len(valid_meals),
        "mean_carbs": float(df_m["carbs"].mean()),
        "mean_delta_g": float(df_m["delta_g"].mean()),
        "median_delta_g": float(df_m["delta_g"].median()),
        "mean_strain_per_carb": float(df_m["strain_per_carb"].mean()),
        "median_strain_per_carb": float(df_m["strain_per_carb"].median()),
        "mean_ascend_slope": float(df_m["ascend_slope"].mean()),
        "median_ascend_slope": float(df_m["ascend_slope"].median()),
        "mean_tau_relax": float(df_m["tau_relax_min"].mean()),
        "median_tau_relax": float(df_m["tau_relax_min"].median()),
        "mean_work_meal": float(df_m["work_meal"].mean()),
        "median_work_meal": float(df_m["work_meal"].median()),
        "mean_specific_work": float(df_m["specific_work"].mean()),
        "median_specific_work": float(df_m["specific_work"].median()),
        # Metadata labels (prism only)
        "group_a1c": subject.get("group_a1c"),
        "a1c": subject.get("a1c"),
        "fpg": subject.get("fpg"),
        "insulin": subject.get("insulin"),
        "homa_ir": subject.get("homa_ir"),
        "bmi": subject.get("bmi"),
        "age": subject.get("age"),
    }


def main():
    data_path = Path("output/cgmacros_subjects.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subjects = data["subjects"]
    results = [analyze_subject_meals(s, min_carbs=25.0, window_min=240) for s in subjects]

    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"Processed {len(subjects)} subjects: {len(ok)} succeeded, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")

    out_path = Path("reports")
    out_path.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_path / f"wind_tunnel_cgmacros_meal_dynamics_{ts_tag}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "cgmacros",
            "protocol": "meal_perturbation_dynamics",
            "min_carbs": 25.0,
            "window_min": 240,
            "n_total": len(subjects),
            "n_success": len(ok),
            "n_failed": len(failed),
            "results": results,
        }, f, indent=2)

    print(f"Meal perturbation dynamics results saved to {out_file}")


if __name__ == "__main__":
    main()
