import json
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats

# 1. Load Night Results
night_files = sorted(Path("reports").glob("wind_tunnel_cgmacros_night_*.json"))
latest_night = night_files[-1]
print("=== CGMacros Night Wind Tunnel Results ===")
print("File:", latest_night)

with open(latest_night, "r", encoding="utf-8") as f:
    night_data = json.load(f)

df_night = pd.DataFrame(night_data["results"])

def print_group_stats(df, metric_col, group_col="group_a1c"):
    print(f"\n--- Metric: {metric_col} by {group_col} ---")
    for g in ["Normal", "Pre-diabetes", "T2D"]:
        sub = df[df[group_col] == g][metric_col].dropna()
        if not sub.empty:
            s_sorted = sorted(sub)
            n = len(s_sorted)
            print(f"  {g:14s} (n={n}): mean={sub.mean():.4f}, median={sub.median():.4f}, IQR=[{s_sorted[int(0.25*(n-1))] :.4f}, {s_sorted[int(0.75*(n-1))] :.4f}], min={sub.min():.4f}, max={sub.max():.4f}")

    # Rank separation
    normal_v = df[df[group_col] == "Normal"][metric_col].dropna().tolist()
    predia_v = df[df[group_col] == "Pre-diabetes"][metric_col].dropna().tolist()
    t2d_v = df[df[group_col] == "T2D"][metric_col].dropna().tolist()

    def rank_sep(a_list, b_list):
        if not a_list or not b_list: return None
        wins = 0.0
        for a in a_list:
            for b in b_list:
                if a > b: wins += 1.0
                elif a == b: wins += 0.5
        return wins / (len(a_list) * len(b_list))

    print(f"  Rank Separation P(Pre-diabetes > Normal): {rank_sep(predia_v, normal_v):.4f}")
    print(f"  Rank Separation P(T2D > Normal)         : {rank_sep(t2d_v, normal_v):.4f}")

for m in ["workIntegral", "tau", "dim", "det", "entr"]:
    print_group_stats(df_night, m)

# 2. Load Meal Perturbation Dynamics Results
meal_files = sorted(Path("reports").glob("wind_tunnel_cgmacros_meal_dynamics_*.json"))
latest_meal = meal_files[-1]
print("\n" + "="*50)
print("=== CGMacros Meal Perturbation Dynamics Results ===")
print("File:", latest_meal)

with open(latest_meal, "r", encoding="utf-8") as f:
    meal_data = json.load(f)

df_meal = pd.DataFrame(meal_data["results"])

meal_metrics = [
    "mean_delta_g",
    "mean_strain_per_carb",
    "mean_ascend_slope",
    "mean_tau_relax",
    "mean_work_meal",
    "mean_specific_work",
]

for m in meal_metrics:
    print_group_stats(df_meal, m)

print("\n" + "="*50)
print("=== Correlations with A1c and HOMA-IR ===")
for m in meal_metrics:
    valid_a1c = df_meal.dropna(subset=[m, "a1c"])
    r_a1c, p_a1c = stats.spearmanr(valid_a1c[m], valid_a1c["a1c"])
    valid_homa = df_meal.dropna(subset=[m, "homa_ir"])
    r_homa, p_homa = stats.spearmanr(valid_homa[m], valid_homa["homa_ir"])
    print(f"{m:22s}: Spearman with A1c = {r_a1c:+.3f} (p={p_a1c:.3e}) | with HOMA-IR = {r_homa:+.3f} (p={p_homa:.3e})")
