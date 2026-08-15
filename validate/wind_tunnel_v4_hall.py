"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: Hall (57 subjects, output/phase_screening_subjects.json)

Thin per-cohort driver. All engine calls, boundary ports, and doctrine
compliance notes live in _wind_tunnel_common.py (single-ownership, Section
9.4 Bit-for-Bit Truth Across Tracks) -- this file only loads the Hall
cohort's subject list and calls into the shared harness.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 60


def main():
    with open("output/phase_screening_subjects.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    hall = data["hall"]

    results, ok, failed = wt.run_cohort("hall", hall, period=PERIOD)
    wt.write_results("hall", hall, results, ok, failed, PERIOD, TAU_MAX)


if __name__ == "__main__":
    main()
