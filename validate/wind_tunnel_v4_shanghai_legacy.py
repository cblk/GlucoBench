"""
Wind-Tunnel Experiment Driver -- Group 2 legacy-metrics SECOND independent-cohort
replication attempt for candidate #5 `relaxationTime` (AGENTS.md Section 3.3/9,
candidate_tensor_staging_matrix.md 候选 #5 registered 2026-08-19 17:38).
Cohort: Shanghai_T2DM (100 unique patients, 109 recordings, 2.6-13.9 day CGM windows)

Why this cohort: largest available n, HbA1c label available, NEVER part of the
Hall/Colas v8.0-v8.4 threshold-fitting history (same non-circularity argument as
Stanford SSPG). Also lets us check whether the record-duration variance in this
cohort (2.6-13.9 days) is a confound for excursion-count-dependent metrics
(earlyDelay/relaxationTime need enough forced-rise events to compute a median).

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall / Section 9.4 Bit-for-Bit Truth: uses
    run_cohort_legacy() in _wind_tunnel_common.py (same code path as the Stanford
    SSPG run, itself cross-checked against real JS via crosscheck_legacy_metrics.py).
  - Section 9.1.2 Labels as Prisms, Not Targets: hba1c / duration_days /
    patient_base_id / admission_date are metadata only, used downstream by
    analyze_shanghai_legacy_results.py purely for group-wise distribution
    comparison, never as fit/regression targets.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 120


def main():
    data_path = Path("output/shanghai_t2dm_subjects.json")
    if not data_path.exists():
        import export_shanghai_subjects
        export_shanghai_subjects.export_shanghai_t2dm_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} Shanghai_T2DM recordings "
          f"({data['n_unique_patients']} unique patients) from {data_path}.")
    results, ok, failed = wt.run_cohort_legacy("shanghai_t2dm", subjects, period=PERIOD, tau_max=TAU_MAX)
    out_file = wt.write_results("shanghai_t2dm_legacymetrics", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
