"""
Wind-Tunnel Experiment Driver -- Group 2 legacy-metrics THIRD independent-cohort test
(AGENTS.md Section 3.3/9), candidate #5 `relaxationTime` third replication attempt using the
v4a exponential-decay-fit redesign (see validate/_relaxation_time_v4_expfit.py and
reports/relaxation_time_redesign_20260819_evaluation.md).
Cohort: T1D-UOM (17 subjects, weekly-segmented, same-body high/low activity paired design)

Design note (dataset_fleet_registry.md Section 4 strong-warning): T1D-UOM subjects run
exogenous basal insulin around the clock, so cross-SUBJECT pooling against T2D/non-diabetic
cohorts (as done for Stanford SSPG/Shanghai T2DM) is forbidden. This driver instead reuses
wind_tunnel_v4_t1d_uom_activity.py's weekly segmentation (each subject's own weeks become
independent pseudo-subjects) so the downstream analysis can do a SAME-BODY paired contrast
(own-median activity split), exactly mirroring analyze_t1d_uom_paired.py's precedent for
candidate #4 (`dim`).

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall / Section 9.4 Bit-for-Bit Truth: uses
    run_cohort_legacy() in _wind_tunnel_common.py for the 6 original (v1) legacy metrics.
  - Section 9.1.2 Labels as Prisms, Not Targets: weekly_step_count_total/
    weekly_basal_dose_total are metadata only, used downstream purely for the paired-split
    and confound-disclosure analysis, never as fit targets.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 120
MIN_WEEK_POINTS = 30


def weeks_to_pseudo_subjects(subject):
    pseudo_subjects = []
    for w in subject["weeks"]:
        if len(w["values"]) < MIN_WEEK_POINTS:
            continue
        pseudo_subjects.append({
            "cohort": "t1d_uom",
            "id": f"{subject['id']}_wk_{w['week_start'][:10]}",
            "original_id": subject["id"],
            "subject_num": subject["subject_num"],
            "week_start": w["week_start"],
            "week_end": w["week_end"],
            "weekly_step_count_total": w["weekly_step_count_total"],
            "weekly_active_kcal_total": w["weekly_active_kcal_total"],
            "weekly_basal_dose_total": w["weekly_basal_dose_total"],
            "n_activity_records": w["n_activity_records"],
            "timestamps": w["timestamps"],
            "values": w["values"],
        })
    return pseudo_subjects


def main():
    data_path = Path("output/t1d_uom_subjects.json")
    if not data_path.exists():
        import export_t1d_uom_subjects
        export_t1d_uom_subjects.export_t1d_uom_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]
    print(f"Loaded {len(subjects)} T1D-UOM subjects from {data_path}.")

    all_weeks = []
    for s in subjects:
        all_weeks.extend(weeks_to_pseudo_subjects(s))
    print(f"Segmented into {len(all_weeks)} weekly runs (>= {MIN_WEEK_POINTS} raw points each).")

    results, ok, failed = wt.run_cohort_legacy("t1d_uom_activity", all_weeks, period=PERIOD, tau_max=TAU_MAX)
    out_file = wt.write_results("t1d_uom_activity_legacymetrics", all_weeks, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
