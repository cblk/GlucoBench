"""
Wind-Tunnel Experiment Driver -- Group 1 legacy-metrics redundancy-audit test (AGENTS.md
Section 3.3/9, dataset_fleet_registry.md changelog "第一组6张中性卡的冗余审计基础设施搭建").
Cohort: Stanford Metabolic Subphenotype (29 subjects with Home CGM + SSPG gold standard)

Runs the 6 purely-neutral legacy-JS-ported candidate metrics (Volume / Recovery / Shape Ratio
λ1/λ2 / Box-Counting Dimension / Lyapunov / Core Dist) so they can be joined (by subject id)
against the already-graduated metrics (workIntegral / det / entr / dim) from
reports/wind_tunnel_stanford_sspg_night_taumax120_20260819_1510.json for a correlation-based
redundancy audit -- see validate/analyze_group1_redundancy.py.

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall / Section 9.4 Bit-for-Bit Truth: uses
    run_cohort_legacy_group1() in _wind_tunnel_common.py, itself cross-checked against the
    real JS via _js_legacy_metrics_group1_crosscheck.mjs / crosscheck_legacy_metrics_group1.py.
  - Section 9.1.2 Labels as Prisms, Not Targets: sspg / sspg_class / di / hba1c / fpg / bmi are
    attached purely as metadata, never used in fits or regressions.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory only, index_v4.html
    untouched.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 120


def main():
    data_path = Path("output/stanford_sspg_subjects.json")
    if not data_path.exists():
        import export_stanford_sspg_subjects
        export_stanford_sspg_subjects.export_stanford_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} Stanford subjects from {data_path}.")
    results, ok, failed = wt.run_cohort_legacy_group1("stanford_sspg", subjects, period=PERIOD)
    out_file = wt.write_results("stanford_sspg_legacymetrics_group1", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
