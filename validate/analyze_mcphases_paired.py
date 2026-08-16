"""
Paired (same-body, cross-phase) analysis for the mcPHASES cohort.

Why this script exists (and why it is NOT the same as analyze_*_results.py for
other cohorts): every other wind-tunnel analysis script in this directory
compares an operator's distribution ACROSS subjects, grouped by a STATIC
clinical label (SSPG class, T2DM diagnosis, ...). mcPHASES's `phase` (menstrual
cycle: Follicular/Fertility/Luteal/Menstrual) is TIME-VARYING WITHIN a single
subject's ~88-day recording window (AGENTS.md Section 7 "同一肉身跨周期对比").
Naively pooling all segments into 4 cross-sectional buckets and comparing
POOLED means (what the wind_tunnel_v4_mcphases_phase.py driver's own summary
printout already does) throws away the far more powerful and far less
confounded comparison this dataset affords: does the SAME body's tensor
operators systematically shift between phases, controlling for inter-subject
baseline variance entirely by construction? This script answers that specific,
narrower question via a per-subject, per-phase aggregation followed by a
PAIRED contrast (delta = phase_B - phase_A per subject), never a pooled/
unpaired group comparison.

Metric scope: `workIntegral` is the user's explicit target. `tau`/`dim`/`det`/
`entr` are swept in the SAME way, unconditionally (not added post-hoc after
seeing which one "worked"), following the precedent set by the Stanford SSPG
report where Work Integral failed cross-sectionally but Dim independently
leaked a real signal. Section 9.1.3 No Frankenstein Scores forbids combining
these into one verdict, so each metric's pairwise table is reported standalone.

Doctrine compliance:
  - Section 9.1.2 Labels as Prisms, Not Targets: `phase` only determines which
    two per-subject cells get subtracted from each other. It is never fed into
    a fit/regression and no composite score is produced.
  - Section 9.1.3 No Frankenstein Scores: every metric x every phase pair is
    reported independently; none are weighted/combined into a single index.
  - Section 8.1 No Inference & No Fabrication: this script computes neutral
    statistics only (deltas, sign counts, Wilcoxon/Friedman tests). It does
    NOT label any direction as "healthy"/"abnormal" or attribute a clinical
    mechanism -- that interpretive step belongs to the Homomorphic Anchor
    Forge report written afterward by the LLM Navigator, not to this script.
  - Section 9.4 Bit-for-Bit Truth Across Tracks: consumes the wind-tunnel JSON
    produced by wind_tunnel_v4_mcphases_phase.py verbatim; performs no
    re-computation of any operator itself.
"""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import scipy.stats as stats

PHASES = ["Follicular", "Fertility", "Luteal", "Menstrual"]
METRICS = ["workIntegral", "tau", "dim", "det", "entr"]

files = sorted(Path("reports").glob("wind_tunnel_mcphases_phase_night_taumax60_*.json"))
if not files:
    raise FileNotFoundError("No wind_tunnel_mcphases_phase_night_taumax60_*.json found under reports/.")
latest = files[-1]
print("File:", latest)

with open(latest, "r", encoding="utf-8") as f:
    data = json.load(f)

raw = data["results"]
n_total = len(raw)
ok = [r for r in raw if r.get("workIntegral") is not None]
failed = [r for r in raw if r.get("workIntegral") is None]
print(f"Total segments: {n_total} | successful: {len(ok)} | failed: {len(failed)}")

df = pd.DataFrame(ok)

structured_out = {
    "source_file": str(latest),
    "n_total_segments": n_total,
    "n_success_segments": len(ok),
    "metrics": {},
}

for metric in METRICS:
    print("\n" + "#" * 78)
    print(f"# METRIC: {metric}")
    print("#" * 78)

    # Step 1: per-subject, per-phase aggregation (collapse multiple cycles of
    # the same phase within one subject's window into ONE cell per subject x
    # phase, so each subject contributes at most one value per phase -- the
    # explicit guard against pseudo-replication the user asked for).
    cell = (
        df.groupby(["original_id", "phase"])[metric]
        .agg(mean_val="mean", n_segments="count")
        .reset_index()
    )
    pivot_mean = cell.pivot(index="original_id", columns="phase", values="mean_val")
    pivot_n = cell.pivot(index="original_id", columns="phase", values="n_segments")
    for p in PHASES:
        if p not in pivot_mean.columns:
            pivot_mean[p] = float("nan")

    n_all4 = pivot_mean[PHASES].notna().all(axis=1).sum()
    print(f"Subjects with valid data in ALL 4 phases: {n_all4} / {pivot_mean.shape[0]}")

    # Step 2: pairwise paired contrasts (delta = phase_B - phase_A, same subject)
    pair_records = []
    for a, b in combinations(PHASES, 2):
        sub = pivot_mean[[a, b]].dropna()
        n_pairs = len(sub)
        if n_pairs < 3:
            print(f"{a} vs {b}: n_pairs={n_pairs} (too few, skipped)")
            continue
        delta = sub[b] - sub[a]
        n_pos = int((delta > 0).sum())
        n_neg = int((delta < 0).sum())
        n_tie = n_pairs - n_pos - n_neg
        frac_pos = n_pos / n_pairs
        # Wilcoxon requires non-degenerate (non-all-zero) differences.
        if (delta != 0).any():
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(sub[b], sub[a])
        else:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
        sign_p = stats.binomtest(n_pos, n_pos + n_neg, p=0.5).pvalue if (n_pos + n_neg) > 0 else float("nan")
        mean_delta = float(delta.mean())
        median_delta = float(delta.median())
        sd_delta = float(delta.std(ddof=1))
        cohens_d = mean_delta / sd_delta if sd_delta > 0 else float("nan")

        pair_records.append({
            "phase_a": a, "phase_b": b, "n_pairs": n_pairs,
            "n_pos_b_gt_a": n_pos, "n_neg_b_lt_a": n_neg, "n_tie": n_tie,
            "frac_pos_b_gt_a": frac_pos,
            "mean_delta_b_minus_a": mean_delta, "median_delta_b_minus_a": median_delta,
            "sd_delta": sd_delta, "cohens_d_paired": cohens_d,
            "wilcoxon_stat": float(wilcoxon_stat), "wilcoxon_p": float(wilcoxon_p),
            "sign_test_p": float(sign_p),
        })

        print(f"{b:12s} vs {a:12s} (n={n_pairs}): mean_delta={mean_delta:+.4f} "
              f"cohen_d={cohens_d:+.3f} frac(b>a)={frac_pos:.3f} "
              f"Wilcoxon_p={wilcoxon_p:.4f} sign_p={sign_p:.4f}")

    # Step 3: omnibus Friedman test (subjects with complete data in all 4 phases)
    complete = pivot_mean[PHASES].dropna()
    if len(complete) >= 3:
        friedman_stat, friedman_p = stats.friedmanchisquare(*[complete[p] for p in PHASES])
    else:
        friedman_stat, friedman_p = float("nan"), float("nan")
    print(f"\nFriedman omnibus (n={len(complete)}): chi2={friedman_stat:.3f}, p={friedman_p:.4f}")
    for p in PHASES:
        v = complete[p]
        print(f"  {p:12s}: mean={v.mean():.4f} median={v.median():.4f} sd={v.std(ddof=1):.4f}")

    structured_out["metrics"][metric] = {
        "n_subjects_all4_phases": int(n_all4),
        "n_subjects_total": int(pivot_mean.shape[0]),
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "pairwise_paired_contrasts": pair_records,
        "per_phase_complete_case_subject_level": {
            p: {
                "mean": float(complete[p].mean()),
                "median": float(complete[p].median()),
                "sd": float(complete[p].std(ddof=1)),
                "n": int(len(complete)),
            }
            for p in PHASES
        },
        "per_phase_pooled_segment_level": {
            p: {
                "mean": float(df[df["phase"] == p][metric].mean()),
                "median": float(df[df["phase"] == p][metric].median()),
                "sd": float(df[df["phase"] == p][metric].std(ddof=1)),
                "n_segments": int((df["phase"] == p).sum()),
            }
            for p in PHASES
        },
    }

out_path = Path("reports") / f"mcphases_paired_analysis_{latest.stem.split('_')[-2]}_{latest.stem.split('_')[-1]}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(structured_out, f, indent=2, ensure_ascii=False)
print(f"\nStructured output written to: {out_path}")
