# 数据集舰队登记本 (Dataset Fleet Registry)

- 定位：本文件是数据集舰队状态的**唯一真源（Single Source of Truth）**，Git 追踪、跨协作者可见。
- 本地可视化：[`dataset-fleet-audit.canvas.tsx`](C:\Users\Oliver\.cursor\projects\d-AI-Work-GlucoBench\canvases\dataset-fleet-audit.canvas.tsx) 是本文件内容的本地 Canvas 渲染，仅存在于单机 Cursor IDE 环境，不随仓库同步；Canvas 与本文件冲突时，以本文件为准。
- 关联文档：[`candidate_tensor_staging_matrix.md`](./candidate_tensor_staging_matrix.md)（候选算子暂存区，与本登记本是正交的两本账——一本记数据集，一本记算子）。
- 更新纪律：任何数据集完成风洞测试、退役、或从"待获取"变为"已落地"，必须先改本文件，再同步 Canvas（如需要）。禁止只改 Canvas 不改本文件。

最后更新：2026-08-16 22:21

---

## 一、判定分类总览

| 判定 | 数量 | 含义 |
|---|---|---|
| 核心研究梯队 | 5 | 有强机制标签或已验证拓扑胜利，是新算子挖矿的主力 |
| 基线/压力测试 | 2 | 框架自带，已跑过风洞，继续作参考但非挖矿优先 |
| 多模态互补 | 1 | 承接跨域代偿研究角色 |
| 保留·强警示 | 2 | T1D 外源胰岛素混杂，不可与 T2D/非糖尿病队列同池比较 |
| 建议退役 | 2 | 样本量或周期被其他数据集全面超越，不删数据只降优先级 |
| 待人工授权 | 1 | mcPHASES，需签署 PhysioNet DUA（**2026-08-16 22:21 更新：原始文件已到位，见下方状态变更**） |

---

## 二、全量清单

| 数据集 | n | 周期 | 标签/模态 | 风洞状态 | 判定 |
|---|---|---|---|---|---|
| Stanford SSPG | 29 | 居家长程 CGM | SSPG(钳夹金标准)+DI+HbA1c | 已测：夜间与全周期 Work Integral 均失效（秩分离度 0.48/0.45，等同抛硬币）；Dim 与 DI/HbA1c 显著相关 | 核心 |
| Stanford OGTT | 21(与 SSPG 重叠 11 人) | 标准化 75g 糖负荷窗口 | SSPG+DI | 已测：specific_work 跨源复现通过(0.8333) | 核心 |
| CGMacros | 45 | 10 天+1706 次真实进餐 | A1C/FPG/insulin/HOMA-IR | 已测：拓扑胜利(specific_work/strain_per_carb) | 核心 |
| Kobe CGM_AC | 64 | ~3 天(原始仓库实测，短于论文声称的 5.5 天均值) | 双钳夹 Clamp DI(未接入，需从论文补充材料提取) | 原始数据已落地，未跑风洞 | 核心 |
| ShanghaiT1DM/T2DM | 112(12+100) | 3–14 天(完整周期居多) | HbA1c/C-peptide/并发症/用药(33 字段) | 原始数据已落地，未跑风洞 | 核心 |
| BIG IDEAs | 16 | 8–10 天 | HR/EDA/皮温+进餐日志；HbA1c 窄带 5.3–6.4 | 原始数据已落地(Dexcom+Food_Log 已核实)，未跑风洞 | 多模态互补 |
| Hall | 57 | ~10 天 | diagnosis/glucotype/SSPG | 已测：Work Integral 软位移，未达拓扑胜利 | 基线 |
| Colas | 208 | ~2 天(短周期) | T2DM 二值 | 已测：方向复现但判别力崩塌(AUC 0.699→0.563) | 基线 |
| Weinstock | 200 | 15.4 天 | T1D，40 列合并症/用药史 | 未跑风洞 | 保留·强警示(T1D) |
| T1D-UOM | 17 | 90 天纵向 | T1D，胰岛素/活动/睡眠/营养 | 原始数据已落地，未跑风洞 | 保留·强警示(T1D) |
| Dubosson | 9 | 3.2 天 | 胰岛素+HR+HRV+呼吸+体位+核心体温 | 未跑风洞 | **建议退役**——被 BIG IDEAs 全面超越 |
| IGLU | 5 | 11.7 天 | 无标签 | 未跑风洞 | **建议退役**——样本量过小，仅作管道回归测试 |
| mcPHASES | 42 | study_interval=2022（仅此一段有 CGM，中位跨度 88 天） | Dexcom(仅 Interval 1)+Fitbit(HR/睡眠/体温/呼吸率/VO2max 等 20 表，含 2 个巨型文件 heart_rate.csv 2GB / calories.csv 646MB，本次未提取)+**时变**月经周期阶段`phase`(Follicular/Fertility/Luteal/Menstrual，按天对齐，非静态标量标签) | **2026-08-16 22:33：提取管道已完成并核实**（`validate/export_mcphases_subjects.py` → `output/mcphases_subjects.json`，32.26MB，42/42 成功）。id=6/11 的 mg/dL→mmol/L 单位修正已验证生效（修正后均值 6.39/6.56，与其余 40 人的 5.5–7 区间吻合）。 | 核心（待写风洞驱动） |

---

## 三、退役清单详情

### Dubosson（n=9，3.2 天）
被 BIG IDEAs（n=16，8–10 天）在样本量与周期长度上全面超越，多模态角色（HR/HRV/皮温）已被取代。退役 ≠ 删除数据，原始文件保留，仅从新拓扑算子挖矿优先队列移除。

### IGLU（n=5，11.7 天）
样本量过小，任何分组都无法产生统计意义上的拓扑撕裂，降级为管道健全性回归测试专用数据（验证提取脚本/风洞驱动本身是否跑得通，不用于科学结论）。

---

## 四、强警示清单详情

### Weinstock（n=200） / T1D-UOM（n=17，90 天）
两者均为 T1D 队列，血糖动力学被外源胰岛素给药行为主导，而非纯粹内源代谢反馈。**绝不可**与 Hall/Colas/Stanford/CGMacros/Kobe/Shanghai 等 T2D 或非糖尿病队列在同一张拓扑撕裂对比图里混池比较，否则会重演 2026-08-11 历史报告里 Kobe/Shanghai 跨队列相关系数崩塌的教训（详见 `AGENTS.md` 第 9 节）。T1D-UOM 90 天的价值应优先用于**同一肉身 Epoch0/Epoch1 天然对照**（如同一受试者高活动周 vs 低活动周），而非跨受试者分组比较。

---

## 五、已知的短周期坍缩风险（Kobe / Shanghai 共性提示）

Kobe（~3 天）与部分 Shanghai 记录（低至 3 天）都可能重演 Colas 上观察到的"样本量越大、判别力反而崩塌"现象。提取管道落地后，建议先看 Shanghai 里恰好满 14 天的子集，再决定是否需要对 Kobe 做周期截断敏感性分析。

---

## 七、mcPHASES 的特殊性：时变 prism，需要不同的风洞驱动设计

mcPHASES 的 `phase`（月经周期阶段）与此前所有队列的标签（SSPG、A1C、diagnosis）本质不同——那些是**每个受试者整段记录期内固定不变的静态标量**，`run_subject()` 跑一次就能拿到一个 Work Integral 去分组对比；`phase` 是**同一受试者 88 天内随天数周期性变化**的标签。这意味着 mcPHASES 的风洞驱动不能照抄 `wind_tunnel_v4_stanford.py` 的写法（一个受试者一次 `run_subject` 调用），而需要先按 `phase` 把每个受试者的 88 天切成多段（如卵泡期 vs 黄体期各自的连续区间），对每一段分别调用 `run_subject`，再看**同一肉身**在不同周期阶段下的摩擦力指标是否有天然的 Epoch0/Epoch1 式位移——这正是 AGENTS.md 第 7 节导航员协议里"同一肉身跨周期对比"的最佳应用场景之一，而不是传统的跨受试者分组对比。

## 六、变更日志

- 2026-08-16 22:33：完成 mcPHASES 提取管道（`export_mcphases_subjects.py`），产出 `output/mcphases_subjects.json`（42 人，32.26MB），单位修正已核实生效。
- 2026-08-16 22:21：新增本登记本作为唯一真源；标注 mcPHASES 原始文件已由用户放入 `output/external_datasets/raw/mcphases/`，待核实。
- 2026-08-16 21:xx：完成 Kobe/Shanghai/BIG IDEAs/T1D-UOM 四个数据集的重新获取与结构核实；Dubosson、IGLU 建议退役。
- 2026-08-16 20:49：完成 Stanford SSPG 队列风洞测试（夜间+全周期），Work Integral 判定 Fail-Closed。
