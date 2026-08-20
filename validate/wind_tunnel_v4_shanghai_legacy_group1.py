"""
Wind-Tunnel Experiment Driver -- Group 1 legacy-metrics redundancy-audit SECOND cohort
(AGENTS.md Section 3.3/9, dataset_fleet_registry.md changelog "第一组6张中性卡的冗余审计基础
设施搭建"). Cohort: Shanghai_T2DM (100 unique patients, 109 recordings).

Runs the 6 purely-neutral legacy-JS-ported candidate metrics so they can be joined (by subject
id) against the already-graduated metrics (workIntegral / det / entr / dim) from
reports/wind_tunnel_shanghai_t2dm_night_taumax120_20260819_1510.json for the redundancy audit
-- see validate/analyze_group1_redundancy.py.

Doctrine compliance: same as wind_tunnel_v4_stanford_sspg_legacy_group1.py.
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
    results, ok, failed = wt.run_cohort_legacy_group1("shanghai_t2dm", subjects, period=PERIOD, tau_max=TAU_MAX)
    out_file = wt.write_results("shanghai_t2dm_legacymetrics_group1", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
