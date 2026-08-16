import json
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats

files = sorted(Path("reports").glob("wind_tunnel_stanford_ogtt_dynamics_*.json"))
latest = files[-1]
print("File:", latest)

with open(latest, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data["results"])
print("N subjects:", len(df))
print("SSPG class counts:", df["sspg_class"].value_counts().to_dict())

def rank_sep(a_list, b_list):
    if not a_list or not b_list:
        return None
    wins = 0.0
    for a in a_list:
        for b in b_list:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(a_list) * len(b_list))

metrics = ["delta_g", "strain_per_carb", "ascend_slope", "tau_relax_min", "work_meal", "specific_work"]

print("\n=== Group Stats: IS vs IR ===")
for m in metrics:
    is_v = df[df["sspg_class"] == "IS"][m].dropna().tolist()
    ir_v = df[df["sspg_class"] == "IR"][m].dropna().tolist()
    sep = rank_sep(ir_v, is_v)
    is_med = np.median(is_v) if is_v else None
    ir_med = np.median(ir_v) if ir_v else None
    print(f"{m:18s}: IS median={is_med:.4f} (n={len(is_v)}) | IR median={ir_med:.4f} (n={len(ir_v)}) | P(IR>IS)={sep:.4f}")

print("\n=== Spearman correlation with continuous SSPG and DI ===")
for m in metrics:
    valid = df.dropna(subset=[m, "sspg"])
    r_sspg, p_sspg = stats.spearmanr(valid[m], valid["sspg"])
    valid_di = df.dropna(subset=[m, "di"])
    r_di, p_di = stats.spearmanr(valid_di[m], valid_di["di"]) if len(valid_di) > 2 else (None, None)
    print(f"{m:18s}: rho(SSPG) = {r_sspg:+.3f} (p={p_sspg:.3f}) | rho(DI) = {r_di:+.3f} (p={p_di:.3f})" if r_di is not None else f"{m:18s}: rho(SSPG) = {r_sspg:+.3f} (p={p_sspg:.3f})")

print("\n=== Per-Subject Detail (sorted by SSPG) ===")
for _, r in df.sort_values("sspg").iterrows():
    print(f"{r['id']:6s} SSPG={r['sspg']:6.1f} ({r['sspg_class']:2s}) strain_carb={r['strain_per_carb']:.4f} specific_work={r['specific_work']:.4f} tau_relax={r['tau_relax_min']:.1f}")
