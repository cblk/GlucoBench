"""
Fleet-wide comparison: tau_max=60 (production default) vs tau_max=120
(corrected/candidate default) across the 8 cohorts swept by
wind_tunnel_v4_taumax120_sweep.py, plus the already-existing Shanghai T1DM
taumax120 supplementary run.

Purpose (Option 2 evaluation, AGENTS.md Section 3.3/9): determine whether
raising the production max_lag from 60 to 120 would REVERSE any existing
directional conclusion, not just shift magnitudes. Per Section 9.3
Topological Victory discipline, this script does NOT compute p-values or
fit anything -- it reports (a) tau ceiling-saturation rate at each window,
(b) rank-correlation of each operator's per-subject value between the two
windows (does relative ordering survive?), and (c) group-mean / paired-delta
direction at 60 vs 120 for whatever grouping variable each cohort's own
already-published report used as its headline comparison.

Honest Fail-Closed: any cohort/metric pair where correlation cannot be
computed (e.g. <3 matched subjects, zero variance) is reported as None, not
silently skipped or estimated.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

METRICS = ("tau", "dim", "det", "entr", "workIntegral")


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["results"]


def index_by_id(results):
    return {r["id"]: r for r in results if "error" not in r}


def tau_ceiling_rate(results, max_lag):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return None
    n_ceiling = sum(1 for r in ok if r.get("tau") == max_lag)
    return n_ceiling / len(ok)


def rank_correlations(idx60, idx120):
    common_ids = sorted(set(idx60) & set(idx120))
    out = {"n_matched": len(common_ids)}
    for m in METRICS:
        xs, ys = [], []
        for sid in common_ids:
            a, b = idx60[sid].get(m), idx120[sid].get(m)
            if a is not None and b is not None:
                xs.append(a)
                ys.append(b)
        if len(xs) < 3 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            out[m] = None
            continue
        rho, p = spearmanr(xs, ys)
        out[m] = {"rho": float(rho), "p": float(p), "n": len(xs)}
    return out


def group_means(results, group_fn, metric):
    groups = {}
    for r in results:
        if "error" in r or r.get(metric) is None:
            continue
        g = group_fn(r)
        if g is None:
            continue
        groups.setdefault(g, []).append(r[metric])
    return {g: (float(np.mean(v)), len(v)) for g, v in groups.items()}


def paired_within_subject(results, subject_key, group_fn, metric):
    """For each subject with >=2 groups present, compute (group_high_mean -
    group_low_mean) using group_fn's own ordering (group_fn must return a
    value ordered such that 'high' > 'low', e.g. weekly step count)."""
    by_subject = {}
    for r in results:
        if "error" in r or r.get(metric) is None:
            continue
        sid = r.get(subject_key)
        by_subject.setdefault(sid, []).append(r)
    deltas = []
    for sid, recs in by_subject.items():
        if len(recs) < 2:
            continue
        recs_sorted = sorted(recs, key=group_fn)
        lo_val = recs_sorted[0][metric]
        hi_val = recs_sorted[-1][metric]
        if group_fn(recs_sorted[0]) == group_fn(recs_sorted[-1]):
            continue  # no real spread for this subject
        deltas.append(hi_val - lo_val)
    if not deltas:
        return None
    return {"mean_delta_hi_minus_lo": float(np.mean(deltas)), "n_subjects": len(deltas)}


def median_split_group(results, field):
    vals = [r.get(field) for r in results if "error" not in r and r.get(field) is not None]
    if not vals:
        return lambda r: None
    med = float(np.median(vals))

    def fn(r):
        v = r.get(field)
        if v is None:
            return None
        return "high" if v >= med else "low"
    return fn


COHORTS = [
    {
        "name": "hall",
        "f60": "reports/wind_tunnel_hall_night_taumax60_20260815_2216.json",
        "f120": "reports/wind_tunnel_hall_night_taumax120_20260819_1509.json",
        "group": ("categorical", "y"),
    },
    {
        "name": "colas",
        "f60": "reports/wind_tunnel_colas_night_taumax60_20260815_2217.json",
        "f120": "reports/wind_tunnel_colas_night_taumax120_20260819_1509.json",
        "group": ("categorical", "y"),
    },
    {
        "name": "stanford_sspg",
        "f60": "reports/wind_tunnel_stanford_sspg_night_taumax60_20260816_2049.json",
        "f120": "reports/wind_tunnel_stanford_sspg_night_taumax120_20260819_1510.json",
        "group": ("categorical", "sspg_class"),
    },
    {
        "name": "cgmacros_night",
        "f60": "reports/wind_tunnel_cgmacros_night_night_taumax60_20260816_2100.json",
        "f120": "reports/wind_tunnel_cgmacros_night_night_taumax120_20260819_1510.json",
        "group": ("categorical", "group_a1c"),
    },
    {
        "name": "shanghai_t2dm",
        "f60": "reports/wind_tunnel_shanghai_t2dm_night_taumax60_20260819_1105.json",
        "f120": "reports/wind_tunnel_shanghai_t2dm_night_taumax120_20260819_1510.json",
        "group": ("median_split", "duration_days"),
    },
    {
        "name": "big_ideas",
        "f60": "reports/wind_tunnel_big_ideas_night_taumax60_20260819_1128.json",
        "f120": "reports/wind_tunnel_big_ideas_night_taumax120_20260819_1510.json",
        "group": ("median_split", "hba1c_pct"),
    },
    {
        "name": "t1d_uom_activity",
        "f60": "reports/wind_tunnel_t1d_uom_activity_night_taumax60_20260819_1119.json",
        "f120": "reports/wind_tunnel_t1d_uom_activity_night_taumax120_20260819_1510.json",
        "group": ("paired", "original_id", "weekly_step_count_total"),
    },
    {
        "name": "mcphases_phase",
        "f60": "reports/wind_tunnel_mcphases_phase_night_taumax60_20260816_2237.json",
        "f120": "reports/wind_tunnel_mcphases_phase_night_taumax120_20260819_1511.json",
        "group": ("categorical", "phase"),
    },
    {
        "name": "shanghai_t1dm",
        "f60": "reports/wind_tunnel_shanghai_t1dm_night_taumax60_20260819_1141.json",
        "f120": "reports/wind_tunnel_shanghai_t1dm_night_taumax120_SUPPLEMENTARY_20260819_1209.json",
        "group": ("median_split", "duration_days"),
        "note": "already reported separately in wind_tunnel_shanghai_t1dm_20260819_1200_tau_max_boundary_calibration.md",
    },
]


def main():
    report = {}
    for cfg in COHORTS:
        name = cfg["name"]
        p60, p120 = Path(cfg["f60"]), Path(cfg["f120"])
        if not p60.exists() or not p120.exists():
            report[name] = {"error": f"missing file(s): {p60.exists()=} {p120.exists()=}"}
            continue
        r60 = load_results(p60)
        r120 = load_results(p120)
        idx60, idx120 = index_by_id(r60), index_by_id(r120)

        entry = {
            "n60_ok": len(idx60), "n120_ok": len(idx120),
            "tau_ceiling_rate_60": tau_ceiling_rate(r60, 60),
            "tau_ceiling_rate_120": tau_ceiling_rate(r120, 120),
            "rank_corr": rank_correlations(idx60, idx120),
        }

        gkind = cfg["group"][0]
        if gkind == "categorical":
            field = cfg["group"][1]
            gfn = lambda r, f=field: r.get(f)
            entry["group_field"] = field
            entry["group_means_workIntegral_60"] = group_means(r60, gfn, "workIntegral")
            entry["group_means_workIntegral_120"] = group_means(r120, gfn, "workIntegral")
            entry["group_means_dim_60"] = group_means(r60, gfn, "dim")
            entry["group_means_dim_120"] = group_means(r120, gfn, "dim")
        elif gkind == "median_split":
            field = cfg["group"][1]
            gfn60 = median_split_group(r60, field)
            gfn120 = median_split_group(r120, field)
            entry["group_field"] = f"{field} (median split, computed independently per tau setting's ok-subset)"
            entry["group_means_workIntegral_60"] = group_means(r60, gfn60, "workIntegral")
            entry["group_means_workIntegral_120"] = group_means(r120, gfn120, "workIntegral")
            entry["group_means_dim_60"] = group_means(r60, gfn60, "dim")
            entry["group_means_dim_120"] = group_means(r120, gfn120, "dim")
        elif gkind == "paired":
            subject_key, order_field = cfg["group"][1], cfg["group"][2]
            order_fn = lambda r, f=order_field: (r.get(f) if r.get(f) is not None else float("-inf"))
            entry["group_field"] = f"paired within {subject_key} by {order_field} (hi-lo)"
            entry["paired_delta_workIntegral_60"] = paired_within_subject(r60, subject_key, order_fn, "workIntegral")
            entry["paired_delta_workIntegral_120"] = paired_within_subject(r120, subject_key, order_fn, "workIntegral")
            entry["paired_delta_dim_60"] = paired_within_subject(r60, subject_key, order_fn, "dim")
            entry["paired_delta_dim_120"] = paired_within_subject(r120, subject_key, order_fn, "dim")

        if "note" in cfg:
            entry["note"] = cfg["note"]
        report[name] = entry

    out_path = Path("reports/fleet_taumax60_vs_120_comparison_20260819_1520.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote fleet comparison to {out_path}")

    # Human-readable console summary
    for name, entry in report.items():
        print(f"\n=== {name} ===")
        if "error" in entry:
            print("  ERROR:", entry["error"])
            continue
        print(f"  n_ok: 60={entry['n60_ok']} 120={entry['n120_ok']}")
        print(f"  tau ceiling-hit rate: 60={entry['tau_ceiling_rate_60']:.3f} -> 120={entry['tau_ceiling_rate_120']:.3f}")
        rc = entry["rank_corr"]
        print(f"  rank corr (n_matched={rc['n_matched']}):")
        for m in METRICS:
            v = rc.get(m)
            if v is None:
                print(f"    {m}: N/A")
            else:
                print(f"    {m}: rho={v['rho']:.3f} (p={v['p']:.4f}, n={v['n']})")
        if "group_means_workIntegral_60" in entry:
            print(f"  group field: {entry['group_field']}")
            print(f"  WI group means @60: {entry['group_means_workIntegral_60']}")
            print(f"  WI group means @120: {entry['group_means_workIntegral_120']}")
            print(f"  Dim group means @60: {entry['group_means_dim_60']}")
            print(f"  Dim group means @120: {entry['group_means_dim_120']}")
        if "paired_delta_workIntegral_60" in entry:
            print(f"  group field: {entry['group_field']}")
            print(f"  WI paired delta @60: {entry['paired_delta_workIntegral_60']}")
            print(f"  WI paired delta @120: {entry['paired_delta_workIntegral_120']}")
            print(f"  Dim paired delta @60: {entry['paired_delta_dim_60']}")
            print(f"  Dim paired delta @120: {entry['paired_delta_dim_120']}")


if __name__ == "__main__":
    main()
