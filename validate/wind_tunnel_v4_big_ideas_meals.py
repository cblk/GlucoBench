"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: BIG IDEAs Lab Glycemic Variability Data (PhysioNet, n=16, ~704 free-living meals)
Purpose: Vector 1/2 meal-perturbation dynamics -- deep-dive requested by the user
  after the 2026-08-19 11:28 baseline scan report flagged Food_Log as this
  cohort's unique deep-dive opportunity (structured carb grams, absent from
  Shanghai/T1D-UOM).

Wind-Tunnel Use-Case Forge (The_Cybernetic_Wind_Tunnel_Doctrine_v1.1.md Section 4,
六段式), filled BEFORE writing the computation below:

  1. [物理目标] 这不是在追求候选算子暂存区 #1（`w_carb`/specific_work）的"跨异构
     扰动源多重印证"毕业标准——那条标准已经在 CGMacros(P=0.8333)与 Stanford
     OGTT-CGM(P=0.8333) 两个完全独立协议上满足，无需第三个来源来"凑数"。本次
     测试的物理目标是一个不同的问题："在一个 HbA1c 全员落在 5.3-6.4（非糖尿病/
     糖尿病前期边界）这种极窄、极健康区间的队列里，`w_carb`/`strain_per_carb`
     这两个算子是否还能测出与 HbA1c 方向一致的连续位移？"——这是对算子敏感度
     下限（floor effect）的探测，而不是重复已满足的毕业条件。
  2. [动力学限制] 复用 `analyze_subject_meals()`（`wind_tunnel_v4_cgmacros_meals.py`,
     逐字节导入，不重新实现，满足第 9.4 节同源纪律），默认参数
     `min_carbs=25.0g, window_min=240min` 与 CGMacros/Stanford OGTT 完全一致，
     确保跨队列可比。
  3. [因果手性限制] BIG IDEAs 是自由生活自报饮食日志（非临床监督下的标准化餐食），
     与 CGMacros 同为自由餐协议（因此这是"自由餐 vs 自由餐"的重复扰动类型，
     不是像 Stanford OGTT 那样的异构扰动源）。这意味着即便本次复现成功，也不能
     算作对候选算子的"第 3 个独立扰动源"（已有的 CGMacros 自由餐 + Stanford OGTT
     标准糖负荷已经是两种不同扰动类型），只能算作"同扰动类型、不同人群"的
     补充验证。
  4. [预判的失效点] 若本队列 HbA1c 的组内变异纯粹是测量噪声（在如此窄的区间内
     生理意义有限），预期 `w_carb`/`strain_per_carb` 与 HbA1c 无方向性关联；
     若观察到方向一致（哪怕未达显著），则是对算子在健康区间连续谱上的敏感度
     的鼓励性证据。
  5. [第一性原理裁决] 见下方残差锁定。
  6. [热力学与残差账单] Food_Log 是自报日志（非实验室秤重），存在真实的自我
     报告误差（份量估算误差、遗漏零食）；受试者 003 因 Food_Log 结构性缺陷
     （缺失表头且缺 3 列）被诚实排除，不参与本次分析。

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall / Section 9.4 Bit-for-Bit Truth Across
    Tracks: analyze_subject_meals() imported verbatim from
    wind_tunnel_v4_cgmacros_meals.py, NOT reimplemented here.
  - Section 9.1.2 Labels as Prisms, Not Targets: HbA1c used strictly for
    post-hoc grouping/correlation, never fed into the meal operators.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory.
"""
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wind_tunnel_v4_cgmacros_meals import analyze_subject_meals  # noqa: E402  (bit-for-bit reuse)


def main():
    data_path = Path("output/big_ideas_subjects.json")
    if not data_path.exists():
        import export_big_ideas_subjects
        export_big_ideas_subjects.export_big_ideas_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    total_meals_available = sum(len(s.get("meals", [])) for s in subjects)
    print(f"Loaded {len(subjects)} BIG IDEAs subjects, {total_meals_available} raw meal events, from {data_path}.")

    # analyze_subject_meals()'s label passthrough whitelist is CGMacros-shaped
    # (subject.get("a1c") etc.) -- Section 9.4 forbids touching that shared
    # function's math/structure, so the metadata key is aliased here at the
    # call site instead (structural relabeling only, zero effect on any
    # computed metric).
    aliased_subjects = [{**s, "a1c": s.get("hba1c_pct")} for s in subjects]
    results = [analyze_subject_meals(s, min_carbs=25.0, window_min=240) for s in aliased_subjects]
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"Processed {len(subjects)} subjects: {len(ok)} succeeded, {len(failed)} failed.")
    for r in failed:
        print(f"  FAILED {r['id']}: {r['error']}")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"wind_tunnel_big_ideas_meal_dynamics_{ts_tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "cohort": "big_ideas",
            "protocol": "meal_perturbation_dynamics",
            "min_carbs": 25.0,
            "window_min": 240,
            "n_total": len(subjects),
            "n_success": len(ok),
            "n_failed": len(failed),
            "results": results,
        }, f, indent=2)

    print(f"Meal perturbation dynamics results saved to {out_file}")


if __name__ == "__main__":
    main()
