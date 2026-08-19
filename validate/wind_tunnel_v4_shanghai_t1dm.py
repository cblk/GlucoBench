"""
Wind-Tunnel Experiment Driver (AGENTS.md Section 3.3 / Section 9)
Cohort: Shanghai_T1DM (12 unique patients, 16 recordings, same archive/protocol as Shanghai_T2DM)
Source: Wang et al. 2023 (Shanghai_T2DM/T1DM public archive)

Wind-Tunnel Use-Case Forge (The_Cybernetic_Wind_Tunnel_Doctrine_v1.1.md Section 4,
六段式), filled BEFORE writing the computation below:

  1. [物理目标] 这是"先跑一遍数据集"阶段的基线测试：把此前因 T1D 强警示而被刻意
     搁置的 Shanghai_T1DM 队列跑通。目标不是与 T2DM 对比（那是被 AGENTS.md 明令
     禁止的跨池比较），而是纯粹在 T1DM 内部，检验 Work Integral/Dim 等指标是否
     随 HbA1c 严重度出现方向一致的位移——这是对"该算子能否在外源胰岛素扰动的
     血糖动力学里仍然测出代偿信号"这一问题的一次小样本探测。
  2. [动力学限制] 全部取夜间静息相（period='night', tau_max=60），与既往所有
     队列保持同源比较基准。T1D 患者夜间仍有持续基础胰岛素/胰岛素泵输注，"夜间相"
     不等同于 T2D/非糖尿病队列的"无外源扰动静息"——这个边界必须显性声明，不能
     暗示等价（与 T1D-UOM 报告的立场一致）。
  3. [因果手性限制] 12 名患者中 2 人（1002、1006）各有 3 次独立住院复诊记录，
     相隔 4-19 个月，治疗方案在复诊之间可能已调整（如 1002 从 Humulin R+insulin
     detemir 换成 Novolin R+insulin glargine 再换成单用 Novolin R）。主要横截面
     对比（HbA1c 分组）严格只取每位患者的首次访次（n=12，避免非独立重复样本污染
     秩分离度统计），多访次记录仅作为**描述性观察**（n=2，远不足以做正式统计
     检验）附录报告，不计入主要结论。
  4. [预判的失效点] 若外源胰岛素给药完全抹平了血糖动力学层面的代偿信号，预期
     Work Integral/Dim 与 HbA1c 无任何方向性关联；若仍有方向一致的位移，则说明
     即使在外源胰岛素干扰下，代偿摩擦力的某些成分仍可被测度——但 n=12 的样本量
     天然不足以把这类发现坐实为统计结论，只能是方向性观察。
  5. [第一性原理裁决] 本轮为基线扫描（非深挖），不做终局判定；结果只记录方向性
     与效应量，供后续决定是否值得为此队列设计更深的分析（如结合 Fasting C-peptide
     区分残余 β 细胞功能对代偿摩擦力的调节作用）。
  6. [热力学与残差账单] 无结构化碳水/胰岛素给药时间戳（Summary 表仅有静态的
     Fasting/2h-postprandial 快照及用药清单文本，无法重建精确的药代动力学窗口），
     因此任何"血糖异常是否由外源胰岛素给药时序引起"的因果归因在本报告中都是
     盲区，必须显性声明，不得暗示因果。

Doctrine compliance:
  - Section 9.1.1 Calculation Firewall: math operators are verbatim from
    index_v4.html via _extracted_tensor_engine_v4.py.
  - Section 9.1.2 Labels as Prisms, Not Targets: HbA1c / diabetes duration /
    complications / insulin regimen are attached as pure metadata and NEVER
    used in fits or regressions.
  - Section 9.4 Bit-for-Bit Truth Across Tracks: uses shared _wind_tunnel_common.py.
  - Section 9.5 Product Isolation: outputs JSON to reports/ directory.
  - T1D isolation: this driver NEVER merges with shanghai_t2dm_subjects.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _wind_tunnel_common as wt

PERIOD = "night"
TAU_MAX = 120  # 2026-08-19/Blueprint v3.6: tracks production index_v4.html's max_lag default


def main():
    data_path = Path("output/shanghai_t1dm_subjects.json")
    if not data_path.exists():
        import export_shanghai_t1dm_subjects
        export_shanghai_t1dm_subjects.export_shanghai_t1dm_subjects()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    subjects = data["subjects"]

    print(f"Loaded {len(subjects)} Shanghai_T1DM recordings "
          f"({data['n_unique_patients']} unique patients) from {data_path}.")
    results, ok, failed = wt.run_cohort("shanghai_t1dm", subjects, period=PERIOD)
    out_file = wt.write_results("shanghai_t1dm", subjects, results, ok, failed, PERIOD, TAU_MAX)
    print(f"Wind tunnel run complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
