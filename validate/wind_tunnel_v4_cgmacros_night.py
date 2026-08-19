"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: CGMacros (45 subjects with ~10-day CGM + detailed metabolic panel)
Protocol: Baseline Night-time Phase Space Dynamics (period='night', tau_max=60)
Source: PhysioNet 2026

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: math operators are verbatim from
    index_v4.html via _extracted_tensor_engine_v4.py.
  - Section 9.1.2 Labels as Prisms, Not Targets: A1c / FPG / Insulin / HOMA-IR /
    BMI / group_a1c are attached purely as metadata and NEVER used
    in fits or regressions.
  - Section 9.4 Bit-for-Bit Truth Across Tracks: uses shared _wind_tunnel_common.py.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 120  # 2026-08-19/Blueprint v3.6: tracks production index_v4.html's max_lag default


def main():
    data_path = Path("output/cgmacros_subjects.json")
    if not data_path.exists():
        import export_cgmacros_subjects
        export_cgmacros_subjects.export_cgmacros_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} CGMacros subjects from {data_path}.")
    results, ok, failed = wt.run_cohort("cgmacros_night", subjects, period=PERIOD)
    out_file = wt.write_results("cgmacros_night", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Night wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
