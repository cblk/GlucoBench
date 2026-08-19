"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: Stanford Metabolic Subphenotype (29 subjects with Home CGM + SSPG gold standard)
Source: Snyder Lab, Nature Biomedical Engineering 2025

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: math operators are verbatim from
    index_v4.html via _extracted_tensor_engine_v4.py.
  - Section 9.1.2 Labels as Prisms, Not Targets: SSPG / sspg_class / di /
    hba1c / fpg / bmi are attached purely as metadata and NEVER used
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
    data_path = Path("output/stanford_sspg_subjects.json")
    if not data_path.exists():
        import export_stanford_sspg_subjects
        export_stanford_sspg_subjects.export_stanford_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} Stanford subjects from {data_path}.")
    results, ok, failed = wt.run_cohort("stanford_sspg", subjects, period=PERIOD)
    out_file = wt.write_results("stanford_sspg", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
