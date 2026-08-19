"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: BIG IDEAs Lab Glycemic Variability + Wearable Device Data (PhysioNet, n=16)
Source: Dexcom G6 CGM (7-9 day windows) + static per-subject HbA1c (Demographics.csv)

Wind-Tunnel Use-Case Forge (The_Cybernetic_Wind_Tunnel_Doctrine_v1.1.md Section 4,
六段式), filled BEFORE writing the computation below:

  1. [物理目标] 这是"先跑一遍数据集"阶段的基线测试，不是深挖某个算子——目标是把
     BIG IDEAs 这支已确认原始数据落地但从未跑过风洞的队列，用与既往所有队列完全
     同源的标准管线（夜间相，tau_max=60）跑一遍，产出 Work Integral / Dim / RQA
     的组间分离度基线读数，供后续排队深挖。
  2. [动力学限制] 全部取夜间静息相（period='night'），不引入 Food_Log 的进食冲击
     分析（那是本队列下一轮"深挖"阶段的候选题目，Food_Log 有结构化碳水克数,
     具备做 CGMacros 式急性冲击分析的条件，但本轮先只做基线扫描）。
  3. [因果手性限制] 无跨访次/纵向复诊问题（每位受试者仅一次连续佩戴记录）。
     但本队列的 HbA1c 分布带极窄（5.3-6.4%，全员在非糖尿病/糖尿病前期区间，
     无一人越过 ADA 糖尿病诊断线 6.5%），这意味着即便算出组间分离，其生理意义
     也只是"血糖调节能力在正常范围内的连续谱两端"，而非既往队列常见的
     "糖尿病 vs 非糖尿病"这种更粗的对比——必须在报告中显性声明，不得暗示等价。
  4. [预判的失效点] 若 Work Integral / Dim 在如此窄的 HbA1c 带内仍能产生方向一致
     的组间位移，则是比既往任何队列都更强的证据（说明该算子对代谢弹性的敏感度
     高于临床诊断阈值的粗粒度）；若無法分离，则是意料之中的阴性结果（样本量 n=16
     且标签带窄，统计功效本就低），不构成对算子的证伪，只构成对该假设排队顺序的
     降级。
  5. [第一性原理裁决] 本轮为基线扫描，不做终局判定；只记录方向性与效应量，供
     后续决定是否值得为此队列设计更深的同体/多模态对比（如可穿戴设备数据的
     跨域代偿分析）。
  6. [热力学与残差账单] Dexcom Clarity 原始 CSV 混杂多种 Event Type（FirstName/
     LastName/PatientIdentifier/Device/Alert/EGV），提取脚本已机械性剔除非 EGV
     行（诚实丢弃，非插补）；本次不涉及胰岛素/碳水时间戳对齐，因此任何"血糖
     异常是否由饮食引起"的归因在本报告中都是盲区，必须显性声明。

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: math operators are verbatim from
    index_v4.html via _extracted_tensor_engine_v4.py.
  - Section 9.1.2 Labels as Prisms, Not Targets: HbA1c / gender are attached
    purely as metadata and NEVER used in fits or regressions.
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
    data_path = Path("output/big_ideas_subjects.json")
    if not data_path.exists():
        import export_big_ideas_subjects
        export_big_ideas_subjects.export_big_ideas_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} BIG IDEAs subjects from {data_path}.")
    results, ok, failed = wt.run_cohort("big_ideas", subjects, period=PERIOD)
    out_file = wt.write_results("big_ideas", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
