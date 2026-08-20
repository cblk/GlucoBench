"""
Group-1 Redundancy Audit (AGENTS.md Section 9.1.3 anti-Frankenstein safeguard,
dataset_fleet_registry.md changelog "第一组6张中性卡的冗余审计").

Joins the 6 newly-ported Group-1 neutral metrics (Volume / Recovery / Shape Ratio λ1/λ2 /
Box-Counting Dimension / Lyapunov / Core Dist) against the already-GRADUATED metrics
(workIntegral / det / entr / dim, from the tau_max=120 production-parameter fleet sweep) on
the SAME subjects in Stanford SSPG and Shanghai T2DM, and computes within-cohort Spearman rank
correlations.

Purpose: before spending a Stanford/Shanghai/(potentially T1D-UOM) rank-separation campaign on
each of the 6 Group-1 metrics individually (as was done for Group 2), first check whether any
of them is simply a monotonic restatement of an existing graduated metric -- in which case a
"significant" rank-separation result would not be new evidence, just the graduated metric's
signal counted twice (Section 9.1.3 No Frankenstein Scores, mirror-image failure mode: one
signal masquerading as several independent ones).

This is diagnostic-only (Section 9.5 Product Isolation): produces a correlation table for
human/LLM-navigator judgment, draws no pass/fail verdict on its own, and does not touch
production code.
"""
import json
from pathlib import Path

GRADUATED_FIELDS = ["workIntegral", "det", "entr", "dim"]
GROUP1_FIELDS = ["volume", "shapeRatio", "avgRecovery", "boxCountingDim", "lyapunov", "coreDistAll"]

COHORTS = [
    {
        "name": "stanford_sspg",
        "graduated_path": "reports/wind_tunnel_stanford_sspg_night_taumax120_20260819_1510.json",
        "group1_path": "reports/wind_tunnel_stanford_sspg_legacymetrics_group1_night_taumax120_20260819_2100.json",
    },
    {
        "name": "shanghai_t2dm",
        "graduated_path": "reports/wind_tunnel_shanghai_t2dm_night_taumax120_20260819_1510.json",
        "group1_path": "reports/wind_tunnel_shanghai_t2dm_legacymetrics_group1_night_taumax120_20260819_2100.json",
    },
]


def rankdata(values):
    """Average-rank ties, 1-indexed (matches scipy.stats.rankdata default)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    n = len(xs)
    if n < 4:
        return None, n
    rx = rankdata(xs)
    ry = rankdata(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((r - mean_rx) ** 2 for r in rx)
    var_y = sum((r - mean_ry) ** 2 for r in ry)
    if var_x < 1e-12 or var_y < 1e-12:
        return None, n
    rho = cov / ((var_x ** 0.5) * (var_y ** 0.5))
    return rho, n


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["id"]: r for r in data["results"] if "error" not in r}


def main():
    for cohort in COHORTS:
        name = cohort["name"]
        grad_path = Path(cohort["graduated_path"])
        g1_path = Path(cohort["group1_path"])
        if not grad_path.exists() or not g1_path.exists():
            print(f"[SKIP] {name}: missing {grad_path if not grad_path.exists() else g1_path}")
            continue

        grad_records = load_records(grad_path)
        g1_records = load_records(g1_path)
        common_ids = sorted(set(grad_records) & set(g1_records))

        print(f"\n{'=' * 70}")
        print(f"Cohort: {name}  (n_graduated={len(grad_records)}, n_group1={len(g1_records)}, n_common={len(common_ids)})")
        print(f"{'=' * 70}")

        header = f"{'':<16}" + "".join(f"{gf:>14}" for gf in GRADUATED_FIELDS)
        print(header)
        for g1f in GROUP1_FIELDS:
            row = f"{g1f:<16}"
            for gf in GRADUATED_FIELDS:
                xs, ys = [], []
                for sid in common_ids:
                    xv = g1_records[sid].get(g1f)
                    yv = grad_records[sid].get(gf)
                    if xv is not None and yv is not None:
                        xs.append(xv)
                        ys.append(yv)
                rho, n = spearman(xs, ys)
                if rho is None:
                    cell = f"n={n}"
                else:
                    flag = "**" if abs(rho) >= 0.70 else ("*" if abs(rho) >= 0.50 else "")
                    cell = f"{rho:+.3f}{flag}(n={n})"
                row += f"{cell:>14}"
            print(row)

        print("\n  ** |rho|>=0.70 : flagged as REDUNDANT with that graduated metric")
        print("  *  |rho|>=0.50 : flagged as MODERATE overlap, needs judgment")

        # Also cross-correlate the 6 Group-1 metrics against EACH OTHER (shared covariance-
        # matrix ancestry risk: Volume/Shape/BoxCountingDim/CoreDist all derive from the same
        # gravity-core/eigen-decomposition of the same phase points).
        print(f"\n  --- Group-1 internal cross-correlation ({name}) ---")
        header2 = f"{'':<16}" + "".join(f"{f:>14}" for f in GROUP1_FIELDS)
        print(header2)
        for f1 in GROUP1_FIELDS:
            row = f"{f1:<16}"
            for f2 in GROUP1_FIELDS:
                if f1 == f2:
                    row += f"{'--':>14}"
                    continue
                xs, ys = [], []
                for sid in common_ids:
                    xv = g1_records[sid].get(f1)
                    yv = g1_records[sid].get(f2)
                    if xv is not None and yv is not None:
                        xs.append(xv)
                        ys.append(yv)
                rho, n = spearman(xs, ys)
                if rho is None:
                    cell = "n/a"
                else:
                    flag = "**" if abs(rho) >= 0.70 else ("*" if abs(rho) >= 0.50 else "")
                    cell = f"{rho:+.2f}{flag}"
                row += f"{cell:>14}"
            print(row)


if __name__ == "__main__":
    main()
