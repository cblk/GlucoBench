"""
Wind-Tunnel Experiment Driver -- Group 2 legacy-metrics OUT-OF-SAMPLE test (AGENTS.md
Section 3.3/9, dataset_fleet_registry.md changelog 2026-08-19 17:21).
Cohort: Stanford Metabolic Subphenotype (29 subjects with Home CGM + SSPG gold standard)

This is the first genuine (non-circular) test of the 6 legacy-JS-ported candidate metrics
(Early Phase Delay / Relaxation Time / AR1 / Angular Velocity / Ascend Friction / Night
Friction): Stanford SSPG was NEVER part of the Hall/Colas v8.0-v8.4 threshold-fitting history,
so a positive finding here is not circular the way a "positive" result on Hall would be.

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: math operators are verbatim/ported from index_v4.html
    (see validate/_legacy_metrics_v4.py for the port provenance and cross-check evidence).
  - Section 9.1.2 Labels as Prisms, Not Targets: sspg / sspg_class / di / hba1c / fpg / bmi are
    attached purely as metadata and NEVER used in fits or regressions -- only for post-hoc
    group-wise distribution comparison via validate/analyze_stanford_sspg_legacy_results.py.
  - Section 9.4 Bit-for-Bit Truth Across Tracks: uses run_subject_legacy() in
    _wind_tunnel_common.py, itself cross-checked against the real JS via
    _js_legacy_metrics_crosscheck.mjs / crosscheck_legacy_metrics.py.
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
    results, ok, failed = wt.run_cohort_legacy("stanford_sspg", subjects, period=PERIOD)
    out_file = wt.write_results("stanford_sspg_legacymetrics", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
