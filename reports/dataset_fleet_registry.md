# 数据集舰队登记本 (Dataset Fleet Registry)

- 定位：本文件是数据集舰队状态的**唯一真源（Single Source of Truth）**，Git 追踪、跨协作者可见。
- 本地可视化：[`dataset-fleet-audit.canvas.tsx`](C:\Users\Oliver\.cursor\projects\d-AI-Work-GlucoBench\canvases\dataset-fleet-audit.canvas.tsx) 是本文件内容的本地 Canvas 渲染，仅存在于单机 Cursor IDE 环境，不随仓库同步；Canvas 与本文件冲突时，以本文件为准。
- 关联文档：[`candidate_tensor_staging_matrix.md`](./candidate_tensor_staging_matrix.md)（候选算子暂存区，与本登记本是正交的两本账——一本记数据集，一本记算子）。
- 更新纪律：任何数据集完成风洞测试、退役、或从"待获取"变为"已落地"，必须先改本文件，再同步 Canvas（如需要）。禁止只改 Canvas 不改本文件。

最后更新：2026-08-19 16:50

---

## 一、判定分类总览

| 判定 | 数量 | 含义 |
|---|---|---|
| 核心研究梯队 | 5 | 有强机制标签或已验证拓扑胜利，是新算子挖矿的主力（Stanford SSPG/OGTT、CGMacros、mcPHASES、ShanghaiT2DM） |
| 基线/压力测试 | 2 | 框架自带，已跑过风洞，继续作参考但非挖矿优先（Hall、Colas） |
| 多模态互补 | 1 | 承接跨域代偿研究角色（BIG IDEAs——基线扫描+Food_Log 生化冲击深挖均已完成） |
| 保留·强警示 | 3 | T1D 外源胰岛素混杂，不可与 T2D/非糖尿病队列同池比较（Weinstock——原始数据未落地阻塞、T1D-UOM——已测、ShanghaiT1DM——已测队列内基线） |
| 建议退役 | 2 | 样本量或周期被其他数据集全面超越，不删数据只降优先级（Dubosson、IGLU） |
| **结构性受限** | **1** | **Kobe CGM_AC——无绝对时钟时间+无可关联标签，2026-08-19 核实暂缓（详见第八节）** |

---

## 二、全量清单

| 数据集 | n | 周期 | 标签/模态 | 风洞状态 | 判定 |
|---|---|---|---|---|---|
| Stanford SSPG | 29 | 居家长程 CGM | SSPG(钳夹金标准)+DI+HbA1c | 已测：夜间与全周期 Work Integral 均失效（秩分离度 0.48/0.45，等同抛硬币）；Dim 与 DI/HbA1c 显著相关 | 核心 |
| Stanford OGTT | 21(与 SSPG 重叠 11 人) | 标准化 75g 糖负荷窗口 | SSPG+DI | 已测：specific_work 跨源复现通过(0.8333) | 核心 |
| CGMacros | 45 | 10 天+1706 次真实进餐 | A1C/FPG/insulin/HOMA-IR | 已测：拓扑胜利(specific_work/strain_per_carb) | 核心 |
| Kobe CGM_AC | 64 | ~3 天(原始仓库实测，短于论文声称的 5.5 天均值) | 双钳夹 Clamp DI(**2026-08-19 核实：不可获取**) | **原始数据已落地，但结构性受限，暂不可跑风洞**（见下方第八节详情） | **核心→暂缓**（数据结构性受限，非优先级问题） |
| ShanghaiT2DM | 100(109 段记录，8 人 2-3 次复诊) | 2.6–13.9 天 | HbA1c/C-peptide/并发症/用药(33 字段) | **2026-08-19：已测（夜间，周期分层）**。"短周期坍缩"预登记假说未被证实（短周期子集 P=0.7899 反而略高于长周期子集 P=0.6716），但发现更严重的混杂残差：周期长度与 HbA1c 严重度显著负相关（ρ=-0.40, p<0.0001，短周期子集中位 HbA1c 89.6 vs 长周期 59.6 mmol/mol），结论强度受限，判定 Fail-Closed（反方向，非最终证伪）。 | 核心 |
| ShanghaiT1DM | 12(16 段记录，2 人各 3 次复诊) | 3.7–13.9 天 | HbA1c/C-peptide/并发症/用药，同 T2DM 协议但严格隔离不同池 | **2026-08-19 11:41：已测（夜间，队列内 HbA1c 分组）**+**2026-08-19 12:00：tau_max 边界标定完成**。意外发现：`Dim` 在 16/16 段记录中 100% 恒为 2（对照 T2DM 首访次 54%）；`Tau` 触顶率**更正为 68.8%**(11/16，原报告"75%"为算术错误，已追加更正)，对照 T2DM 首访次 24.0%。边界标定证实这是**真实测量天花板而非非平稳噪声**：`max_lag` 扫至 120 后 T1DM 触顶率降为 0%，tau 中位数/均值在 120→240 完全收敛冻结于 72.0/69.12（对照 T2DM 均值持续缓慢爬升未完全收敛，性质不同——T2DM 是尾部离群点问题，T1DM 是全队列分布中心问题）。**已选定并执行选项 3**（队列专属补充分析，不改动生产引擎）：用校正 tau(max_lag=120) 全管线复算，证实 `Dim`=2 恒定性与 Work Integral 反转方向**均非 tau 截断伪影**（两个 tau 窗口下结论一致），核心结论稳健，不触发选项 2（全队列重跑）。副产物发现：DET/ENTR 对 tau/Theiler 窗口高度敏感（均值分别位移 -0.27/-0.63），已记录为跨队列比较该二指标时的通用方法学注意事项。HbA1c 组间 Work Integral 方向与既往队列相反(AUC=0.233→校正后0.267，仍反转)，未达显著，判定 Fail-Closed 维持不变。 | 保留·强警示(T1D) |
| BIG IDEAs | 16 | 6.9–9.9 天 | HR/EDA/皮温+进餐日志；HbA1c 窄带 5.3–6.4 | **2026-08-19 11:28：已测（夜间基线）**+**2026-08-19 11:52：已测（Food_Log 生化冲击深挖，14/16 有效）**。夜间相六指标无一项达统计显著（非证伪）；Work Integral 方向与既往队列一致但未达显著。生化冲击相：`w_carb`/`strain_per_carb` 方向与 CGMacros/Stanford OGTT 一致但未达显著(P=0.61-0.63, n=14 电力不足)，登记为候选算子暂存区补充验证（不计入独立扰动源毕业计数）；意外发现未归一化 `delta_g` 反超归一化指标(P=0.796)，**2026-08-19 12:20 已排查**（复用 CGMacros/Stanford OGTT 既有结果重算，未见普适复现，判定为该队列小样本噪声，非归一化算子缺陷，详见候选算子暂存区）。`Dim` 夜间相呈完全零信号。 | 多模态互补 |
| Hall | 57 | ~10 天 | diagnosis/glucotype/SSPG | 已测：Work Integral 软位移，未达拓扑胜利 | 基线 |
| Colas | 208 | ~2 天(短周期) | T2DM 二值 | 已测：方向复现但判别力崩塌(AUC 0.699→0.563) | 基线 |
| Weinstock | 200 | 15.4 天 | T1D，40 列合并症/用药史 | **2026-08-19 核实：原始数据未落地**——`config/weinstock.yaml` 声明的 `raw_data/weinstock.csv` 在仓库中不存在，`output/external_datasets/raw/` 下也无对应目录。需用户先获取并放置原始文件才能推进，非分析方法可解决的缺口。 | 保留·强警示(T1D)——**阻塞：待获取原始数据** |
| T1D-UOM | 17 | 29–177 天纵向(中位数~99天) | T1D，胰岛素/活动/睡眠/营养 | **2026-08-19：已测（夜间，同体高/低活动周配对，246 个周分段，成功率 100%）**。Work Integral 零信号（第三次，与 Stanford SSPG/mcPHASES 一致）；`Dim` 出现方向一致信号（13/17 人，符号检验 p=0.0018，经 Holm-Bonferroni 5 指标家族校正后仍显著），但 Wilcoxon 幅度检验未通过校正——判定"候选线索，未达拓扑胜利"，已登记入候选算子暂存区 #4（观察性，0/3 毕业进度）。基础胰岛素剂量混杂检验未证实也未排除（n=12，p=0.176）。 | 保留·强警示(T1D) |
| Dubosson | 9 | 3.2 天 | 胰岛素+HR+HRV+呼吸+体位+核心体温 | 未跑风洞 | **建议退役**——被 BIG IDEAs 全面超越 |
| IGLU | 5 | 11.7 天 | 无标签 | 未跑风洞 | **建议退役**——样本量过小，仅作管道回归测试 |
| mcPHASES | 42 | study_interval=2022（仅此一段有 CGM，中位跨度 88 天） | Dexcom(仅 Interval 1)+Fitbit(HR/睡眠/体温/呼吸率/VO2max 等 20 表，含 2 个巨型文件 heart_rate.csv 2GB / calories.csv 646MB，本次未提取)+**时变**月经周期阶段`phase`(Follicular/Fertility/Luteal/Menstrual，按天对齐，非静态标量标签) | **2026-08-16 22:55：已测（同体跨周期配对，42/42 成功）**，见 [`wind_tunnel_mcphases_phase_20260816_2255_paired_bodywise.md`](./wind_tunnel_mcphases_phase_20260816_2255_paired_bodywise.md)。本行状态描述此前因登记本创建时序滞后于该测试而误标为"待写风洞驱动"，2026-08-19 11:41 核实修正，非新增测试。 | 核心 |

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

## 五、已知的短周期坍缩风险（Kobe / Shanghai 共性提示，2026-08-19 更新）

**2026-08-19 更新：** 已在 Shanghai_T2DM 内部完成周期分层复现测试（详见 [`wind_tunnel_shanghai_t2dm_20260819_1105_duration_stratified.md`](./wind_tunnel_shanghai_t2dm_20260819_1105_duration_stratified.md)）。结果**未支持**"短周期→判别力系统性坍缩"这一普适假说（短周期子集 $P=0.7899$ 不低于长周期子集 $P=0.6716$），这为 Colas 的坍缩提供了新证据：更可能是 Colas 队列自身的特有噪声，而非 Work Integral 对短序列的通用缺陷。**但该测试同时暴露了一个更严重的、此前未知的混杂残差**：Shanghai 队列内记录周期长度与 HbA1c 严重度显著负相关（$\rho=-0.40$, $p<0.0001$）——病情越重的患者实际获得的监测周期反而越短，这是一种真实世界的临床选择性偏倚，使本次周期分层测试的结论强度受限（无法把"周期"与"严重度构成比"两个变量正交拆开）。

Kobe（~3 天）目前仍未跑风洞。鉴于 Shanghai 的经验教训，对 Kobe 做周期截断敏感性分析之前，应先核查其样本是否存在类似的周期-严重度选择性偏倚，避免重蹈覆辙。

---

## 七、mcPHASES 的特殊性：时变 prism，需要不同的风洞驱动设计

mcPHASES 的 `phase`（月经周期阶段）与此前所有队列的标签（SSPG、A1C、diagnosis）本质不同——那些是**每个受试者整段记录期内固定不变的静态标量**，`run_subject()` 跑一次就能拿到一个 Work Integral 去分组对比；`phase` 是**同一受试者 88 天内随天数周期性变化**的标签。这意味着 mcPHASES 的风洞驱动不能照抄 `wind_tunnel_v4_stanford.py` 的写法（一个受试者一次 `run_subject` 调用），而需要先按 `phase` 把每个受试者的 88 天切成多段（如卵泡期 vs 黄体期各自的连续区间），对每一段分别调用 `run_subject`，再看**同一肉身**在不同周期阶段下的摩擦力指标是否有天然的 Epoch0/Epoch1 式位移——这正是 AGENTS.md 第 7 节导航员协议里"同一肉身跨周期对比"的最佳应用场景之一，而不是传统的跨受试者分组对比。

## 八、Kobe CGM_AC 结构性受限详情（2026-08-19 核实）

`output/external_datasets/raw/kobe_cgm_ac/CGM_data.csv` 是 [`HikaruSugimoto/CGM_AC`](https://github.com/HikaruSugimoto/CGM_AC) 仓库的**原始且唯一**内容（已用 GitHub API 核实仓库文件清单：仅 `CGM_data.csv` + `CGM_data.png` + `main.py` + `requirements.txt`，不含任何标签或时间戳元数据文件）。经核实存在两个此前未被记录的结构性障碍，两者任一都足以阻断本队列进入风洞：

1. **无绝对时钟时间——夜间切片会构成虚构数据。** `CGM_data.csv` 是宽表，列名是"距 CGM 佩戴起点的分钟数"（0, 5, 10, ..., 4315），不含任何日期或时钟时间。论文方法学（Sugimoto et al., medRxiv 2023 / Nature Communications Medicine 2025-2026）只说"受试者在晨间空腹口服糖耐量试验（OGTT）后佩戴 CGM"——这只是一个协议层面的粗略惯例，并非逐个受试者的精确戴表时刻。若强行假设一个统一的"起点=某个钟点"来切出夜间 0-6 点窗口，等同于**编造未被实际测量的时间戳**，违反第 8.1 节《禁止推断与虚构》。
2. **无可关联的临床标签。** 论文摘要与补充材料核实结果：Clamp DI（钳夹法处置指数）等指标只以**群体层面的相关系数（如 AC_Var 与 DI 的 $r=-0.31$）**形式出现在正文与 Supplementary Data 1 中，`Supplementary Table 1` 只是人群特征的汇总统计（均值±SD），**并不包含逐个受试者的原始数值表**。这意味着即便重新核对论文补充材料，也没有一份可以按 `ID`（0-63）与 `CGM_data.csv` 直接对齐的标签文件——任何"假设 ID 顺序与论文表格顺序一致"的拼接都是无法验证的猜测，同样违反第 8.1 节。

**判定：** 本队列在原始数据现状下**不满足风洞实验的最低前提**（既无法做时段切片，也无法做分光镜分组）。既不删除数据，也不强行凑合跑一次伪造时间基准的分析；保留原始文件，标记为"结构性受限"，与 Dubosson/IGLU 的"样本量不足退役"性质不同——如果未来能联系论文作者获取逐人 Clamp DI 对照表，或找到可靠的绝对时间基准，可重新评估。

## 六、变更日志

- 2026-08-19 16:50：**修复银色星标视觉不可辨识问题**（用户反馈"看不到银色星星"）。原因：`star-silver` 颜色 `#c9d2da` 与标签本身的 `--text-secondary`(`#94a3b8`) 亮度/色调过于接近，在深色背景下几乎融入普通灰色文字，无法被肉眼识别为"有颜色的星"（金/铜因色相跨度大不受此影响）。修复为更亮的 `#eef3f8` 并叠加浅蓝色 `text-shadow` 发光效果，经浏览器截图+DOM 计算颜色值双重验证确认现在清晰可辨，金/铜星标未受影响。纯 CSS 颜色调整，不涉及星标分级逻辑或依据文字改动。
- 2026-08-19 16:10：**`index_v4.html` HUD 新增风洞验证星标(★)，纯 UI 标注，未改动任何计算逻辑/阈值/参数**。用户要求"给验证过的指标名称加金星以标识可靠指标"，先盘点全部 16 个指标卡片+2 个顶部维度显示在 `_wind_tunnel_common.py`/全部 `wind_tunnel_v4_*.py` 中的实际引用情况，发现两个需要提前拦截的陷阱：(1) **命名撞车**——"盒计数几何维度 (Dimension)"卡片用纯 JS `boxCountingDimension()`，与风洞项目反复测试的"嵌入维度"(`estimate_dimension`,FNN+Jacobi Razor,显示于顶部)是完全不同的算法，已改名为"盒计数几何维度 (Box-Counting Dimension，非风洞测试的嵌入维度Dim)"并在卡片描述/tooltip 中显式声明二者不共享验证状态；(2) **无一张卡片真正"毕业"**——按第9.3节严格标准，唯一完整走完生产采纳流程的是 Tau 边界标定（16:10 前一小时刚完成的 `max_lag` 60→120）。经用户明确裁决三色分级语义（测过给星,毕业=金,正向未毕业=银,负向/脆弱=铜,未测=无星），最终标注：🥇金——Tau 延迟（唯一完整走完生产采纳流程）；🥈银——嵌入维度/Dim（tau_max标定秩相关≥0.95零反转+代偿指纹假说两次独立复现方向一致,但0/3毕业）；🥉铜——Work Integral 卡片(夜间基线口径,Stanford SSPG/T1D-UOM×3/Shanghai T1DM 均 Fail-Closed 或反转,仅 Hall 软位移)、DET、ENTR(均值分别位移-0.27/-0.63,对tau高度敏感)；其余11张卡片(早相加速度迟滞/弛豫衰减疲劳度/AR1/角速度/上升相阻力/夜间相变阻力/Volume/Recovery/λ₁λ₂/CoreDist/Lyapunov)在全部风洞脚本中零引用,保持无星(未测≠不可靠)。已停用的胰岛素预判卡片不参与分级。星标语义在页面顶部指标读卡规则区新增一行说明，并在每个星标的 title 悬浮提示中写明具体依据，明确"不是对本次读数的临床可靠性背书"以避免与第9.1.3节 No Frankenstein Score 精神冲突。
- 2026-08-19 15:55：**执行第 B.5 节《原子化蓝图收编》：生产 `max_lag` 正式从 60 提升到 120**（用户在看完 15:20/15:45 两份实证报告后明确签字批准）。双重原子事务：(1) 代码——`index_v4.html` 的 Python 引擎 `max_lag`、UI tau 滑块静态上限与动态扩展上限公式、JS 侧 dead-code `TAU_MAX` 常量共 4 处同步改为 120，diff 已核对无关无关改动；(2) 文档——Pipeline Blueprint v3.3 第 3.1 节追加 `[v3.6 修订]`、Implementation Contract v1.3 追加 `[v1.5 修订]`，均引用 `wind_tunnel_fleet_taumax60_vs_120_option2_evaluation_20260819_1520.md` 作为证据源，遵循已有的"追加式版本标注、不改文件名"惯例。同步同源纪律：`validate/_extracted_tensor_engine_v4.py`/`_wind_tunnel_common.py` 的默认值也从 60 改为 120（与新生产值保持逐字节同源），9 个生产镜像驱动脚本(`wind_tunnel_v4_hall/colas/stanford/cgmacros_night/shanghai/shanghai_t1dm/big_ideas/t1d_uom_activity/mcphases_phase.py`)的 `TAU_MAX` 文件名标签常量同步更新为 120。用 Hall/Colas 做最终回归验证：默认值驱动的重跑结果与此前 sweep 脚本显式传参的结果逐字段零差异。历史 `taumax60` 归档文件保留不动，作为旧生产值的永久证据留存。
- 2026-08-19 15:45：**纠正上一条(15:20)变更日志的一处错误声明**。用户要求先查明"Shanghai T2DM 病程分组 WI 组间差距收缩 ~86%"这一发现的机制,核查后确认这是 `compare_taumax60_vs_120_fleet.py` 自身的方法学误差(简化对比用的分组变量与原始 `wind_tunnel_shanghai_t2dm_20260819_1105_duration_stratified.md` 实际使用的方法不一致,且 @60/@120 两次中位数分组因样本流失而成员不同,并非同一批人的同口径对比)。用原始报告的真实方法学(固定 HbA1c 中位数阈值,短<7天/长≥10天病程分层内分别算秩分离度 P(高>低))在 `analyze_shanghai_duration_stratified_taumax_comparison.py` 中同源复现后,**该发现实际完全稳健,方向未变且略微加强**（headline delta 从 -0.1183 变为 -0.1603）。第一版报告的"86% 收缩"结论已撤销,详见 [`wind_tunnel_fleet_taumax60_vs_120_option2_evaluation_20260819_1520.md`](./wind_tunnel_fleet_taumax60_vs_120_option2_evaluation_20260819_1520.md) 第 3.1 节的完整纠错记录(未删除原错误声明,仅显性标注已修正,第 8.2 节 Honest Fail-Closed / v1.4 禁止历史截断)。修正后,tau_max=60→120 的唯一真实代价仅剩 Shanghai T2DM 新增 5 例(4.6%)诚实失败,不再有任何效应量弱化项。
- 2026-08-19 15:20：执行 tau_max 边界标定报告的**选项 2**（用户明确选定，需求超出单队列范围）：新增 `validate/wind_tunnel_v4_taumax120_sweep.py`，把 `_extracted_tensor_engine_v4.py`/`_wind_tunnel_common.py` 的 `max_lag`/`tau_max` 参数化（默认值仍为 60，`index_v4.html` 生产代码未被触及，已用 Hall/Colas 逐字节回归验证零漂移），对剩余 8 个依赖 tau 的已测队列（Hall/Colas/Stanford SSPG/CGMacros/Shanghai T2DM/BIG IDEAs/T1D-UOM/mcPHASES；ShanghaiT1DM 已单独测过，Stanford OGTT/CGMacros 与 BIG IDEAs 的进餐动力学不依赖 tau，均排除在外）用 `max_lag=120` 重新跑一遍。全舰队对比结果（详见 [`wind_tunnel_fleet_taumax60_vs_120_option2_evaluation_20260819_1520.md`](./wind_tunnel_fleet_taumax60_vs_120_option2_evaluation_20260819_1520.md)）：tau 触顶率全面下降（2.1%-68.8% → 0%-1.0%），`Dim`/`Work Integral` 秩相关全线 ≥0.95，**全部 9 个队列既有方向性结论零翻转**；唯一的真实代价是 Shanghai T2DM 的病程分组 WI 组间差距收缩 ~86%（未翻转，但证据强度需重新措辞）与新增 5 例（4.6%）诚实失败（此前的 `tau=60,dim=2` 本就是截断伪影，现在如实拒绝继续输出）。**是否据此把生产 `max_lag` 默认值从 60 提升到 120（并同步修订 Blueprint/Contract）仍未获最终签字**，本次只完成实证评估，未执行第 B.5 节原子化收编。
- 2026-08-19 12:24：完成候选算子暂存区遗留的两个开放项（非新数据集测试，详见 `candidate_tensor_staging_matrix.md` 同时刻变更日志）：修复 `analyze_subject_meals()` 对缺失碳水元数据的防御性降级路径（`w_carb` 候选推进至 2/3 毕业条件），并排查 BIG IDEAs 的 `delta_g` 反超假说（未见普适复现，判定队列特有噪声）。核查后确认：数据集舰队层面**已无可自主推进的开放项**——唯一剩余的 Weinstock 阻塞项需用户先提供原始数据文件，Kobe CGM_AC 的结构性受限需用户联系论文作者获取逐人标签或绝对时间基准，均非分析方法可解决。
- 2026-08-19 12:09：执行 tau_max 边界标定报告的**选项 3**（性价比最高，用户指定标准后由 Agent 选定）：新增 `wind_tunnel_v4_shanghai_t1dm_taumax120.py`，用校正 tau(max_lag=120) 对 ShanghaiT1DM 做全管线补充复算（不改动生产引擎，产物显式标注 SUPPLEMENTARY 且与生产版本并列存档，不覆盖）。核心发现：`Dim`=2 恒定性与 Work Integral 反转方向在两个 tau 窗口下结论完全一致，证实此前发现不是截断伪影；DET/ENTR 对 tau 高度敏感（分别位移 -0.27/-0.63），记录为新的跨队列比较注意事项。不触发选项 2（全队列重跑），未改动任何生产代码/蓝图。详见 [`wind_tunnel_shanghai_t1dm_20260819_1209_taumax120_supplementary_rerun.md`](./wind_tunnel_shanghai_t1dm_20260819_1209_taumax120_supplementary_rerun.md)。
- 2026-08-19 12:00：完成 ShanghaiT1DM 的 `tau_max` 方法学边界标定（`validate/probe_tau_max_boundary.py`，独立诊断脚本，不修改生产引擎）。**首先追加更正**：前置报告"12/16(75%)触顶"存在算术错误，实际为 11/16(68.8%)。标定结果：扫描 `max_lag∈{60,90,120,180,240}`，T1DM 的 tau 分布在 `max_lag=120` 处完全收敛（触顶率降为 0%，中位/均值 tau 冻结于 72.0/69.12），证实 `tau_max=60` 对该队列是真实的测量天花板（假设 A 胜出），而非非平稳/不衰减噪声（假设 B 已排除）；对照 T2DM 首访次子集(n=100)呈现完全不同模式（仅少数尾部离群点未收敛，中位数全程不受影响）。是否将生产 `max_lag` 从 60 提升至 120 涉及重跑全部 9 个已测队列（第 9.4 节同源纪律），已整理 3 个选项供人类架构师决策，本报告不擅自修改生产代码/蓝图。详见 [`wind_tunnel_shanghai_t1dm_20260819_1200_tau_max_boundary_calibration.md`](./wind_tunnel_shanghai_t1dm_20260819_1200_tau_max_boundary_calibration.md)。
- 2026-08-19 11:52：完成 BIG IDEAs Food_Log 生化冲击深挖（复用 CGMacros 的 `analyze_subject_meals()` 算子逐字节同源，14/16 有效受试者，704 次自报餐食）。`w_carb`/`strain_per_carb` 方向与既往两个独立协议来源(CGMacros/Stanford OGTT)一致但未达显著(n=14 电力不足)，登记为候选算子暂存区补充验证，不改变毕业进度。意外发现：未按碳水归一化的 `delta_g` 反而秩分离度更高(P=0.796)，提出"自报碳水估算误差污染归一化分母"假设，标注待验证。为此新增 `export_big_ideas_subjects.py` 的 `_extract_meals()`，诚实排除了受试者 003（Food_Log 缺表头且缺 3 列，拒绝猜测列映射）。
- 2026-08-19 11:41：完成 ShanghaiT1DM 队列风洞测试（队列内基线，严格与 T2DM 隔离不同池，12 名患者/16 段记录，成功率 100%）。意外发现：`Dim` 在 16/16 段记录中 100% 恒为 2（对照同协议 T2DM 仅 54%），`Tau` 75% 触顶 tau_max=60 硬边界（对照 T2DM 仅 23%）——两个各自独立分布的清晰分离，列为 tau_max 边界标定的下一步优先行动。HbA1c 组间 Work Integral 方向与既往队列相反但未达显著（n=11），判定 Fail-Closed。核实并修正 mcPHASES 行的过时状态描述（该队列早在 2026-08-16 22:55 已跑测，登记本创建时序滞后导致误标）；核实 Weinstock 原始数据从未落地，标记为阻塞待用户获取。
- 2026-08-19 11:28：完成 BIG IDEAs 队列风洞基线扫描（"先跑一遍数据集"阶段，夜间相，16/16 成功率 100%）。六指标全量扫描（HbA1c 中位数分组）无一项达统计显著（n=16、标签带仅 5.3–6.4% 电力天然不足，非证伪）；Work Integral 方向与既往队列一致但未达显著（AUC=0.625, ρ=0.320, p=0.227）；`Dim` 呈完全零信号（两组中位数皆为 4.0），与 T1D-UOM 的同体配对设计不同源，不构成对其假说的证伪。已具备条件进入 Food_Log 结构化碳水急性冲击深挖阶段（下一步候选）。
- 2026-08-19 11:19：完成 T1D-UOM 队列风洞测试（夜间，同体高/低活动周配对，17 名受试者/246 个周分段，成功率 100%）。执行了 mcPHASES 报告遗留的 `dim` 复现行动项：`dim` 方向一致性经 Holm-Bonferroni 校正后仍显著（符号检验 p=0.0018），Work Integral 第三次测出零信号。已登记入候选算子暂存区 #4（观察性，不触发任何代码/蓝图修改）。发现并修复了提取脚本中 README 文档与实际数据不符的残差（时间戳格式声称 MM/DD/YYYY，实测为 DD/MM/YYYY）。
- 2026-08-19 11:20：核实 Kobe CGM_AC 存在两个结构性障碍（无绝对时钟时间、无可关联的逐人临床标签），判定暂缓，不强行推进，详见第八节。
- 2026-08-19 11:05：完成 Shanghai_T2DM 队列风洞测试（夜间，周期分层，100 名独立患者/109 段记录，成功率 100%）。"短周期坍缩"预登记假说未被证实，但发现周期长度与 HbA1c 严重度存在显著混杂（ρ=-0.40），判定 Fail-Closed（反方向，非最终证伪）。提取脚本 `export_shanghai_subjects.py` 中修复了 `"/"` 缺失值哨兵未被识别为 None 的残差。ShanghaiT1DM(n=12) 仍未提取。
- 2026-08-16 22:33：完成 mcPHASES 提取管道（`export_mcphases_subjects.py`），产出 `output/mcphases_subjects.json`（42 人，32.26MB），单位修正已核实生效。
- 2026-08-16 22:21：新增本登记本作为唯一真源；标注 mcPHASES 原始文件已由用户放入 `output/external_datasets/raw/mcphases/`，待核实。
- 2026-08-16 21:xx：完成 Kobe/Shanghai/BIG IDEAs/T1D-UOM 四个数据集的重新获取与结构核实；Dubosson、IGLU 建议退役。
- 2026-08-16 20:49：完成 Stanford SSPG 队列风洞测试（夜间+全周期），Work Integral 判定 Fail-Closed。
