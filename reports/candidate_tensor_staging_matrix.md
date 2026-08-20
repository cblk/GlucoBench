# 候选张量算子暂存区 (Candidate Tensor Operator Staging Matrix)

> **本文件性质：** 这是《AGENTS.md》第 9 节《风洞隔离法则》与 B.5 节《原子化蓝图收编》之间的**缓冲隔离层**。任何在单一风洞队列中取得初步拓扑分离信号的候选算子，必须先在此登记并接受多队列交叉印证，**在满足下方"毕业标准"之前，严禁修改 `index_v4.html` 生产代码或《Pipeline Blueprint》/《Implementation Contract》正式文档**。
>
> 本文件只做登记与状态追踪，不产出诊断结论，不作为部署证据（第 8.2 节 Honest Fail-Closed 适用：候选状态本身就是诚实的"尚未确定"标注，不得升格）。

---

## 毕业标准 (Graduation Criteria)

候选算子若要从暂存区"毕业"并进入 B.5 节《原子化蓝图收编》，必须同时满足：

1. **跨异构扰动源多重印证 (Cross-Source Replication)：** 必须在至少 **2 个完全独立、且扰动来源/采集协议不同** 的队列中复现单指标秩分离度 $P > 0.80$（第 9.3 节标准）。仅在同一数据集的子集上重复验证不算独立印证。
2. **缺失鲁棒性与自适应降级 (Missing-Data Robustness)：** 若算子依赖生产环境不一定具备的元数据（如碳水克数），必须设计好该元数据缺失时的诚实降级路径（返回 `null` 或改用无需该元数据的替代表达式），不得在缺失时静默回退到虚构常量。
3. **人类架构师显式批准 (Explicit Human Sign-off)：** 完成 1-2 后，需人类架构师明确批准，才能触发 B.5 节的双重原子事务（代码 + 蓝图同步修改）。

---

## 候选算子登记表

### 候选 #1：`w_carb`（特异性代谢耗散做功 / Specific Postprandial Work Integral）

- **定义：** $w_{\text{carb}} = W_{\text{meal}} / \text{Carbs}$，其中 $W_{\text{meal}} = \int_{t_{\text{meal}}}^{t_{\text{post}}} \max(0, G(t) - G_{\text{base}})\, dt$（单位：$\text{mmol}\cdot\text{min}/(\text{L}\cdot\text{g})$）。
- **物理领地：** 生化冲击相（Vector 1/2 进食扰动），**不适用于**静息夜间相（已在 Stanford SSPG 队列证实静息相会被代偿掩盖，见下方印证记录 #2）。
- **实现位置（暂存，非生产）：** `validate/wind_tunnel_v4_cgmacros_meals.py`

| # | 印证队列 | 扰动来源 | 分光镜标签 | 秩分离度 $P$ | 状态 | 报告 |
|---|---|---|---|---|---|---|
| 1 | CGMacros ($n=45$) | 自由进食（1706 次真实餐食，碳水已知） | HbA1c 3 分组 | Pre-dia vs Normal: **0.8333**<br>T2D vs Normal: **0.9143** | ✅ 首次胜利 | [`wind_tunnel_cgmacros_20260816_2100_meal_dynamics.md`](./wind_tunnel_cgmacros_20260816_2100_meal_dynamics.md) |
| 2 | Stanford SSPG ($n=29$) | 静息夜间（无扰动，对照组） | SSPG 2 分类 | Work Integral: **0.4808**（等同随机） | ⚠️ 证实了"夜间相不适用"，非本算子的失败，而是划定了其合法物理领地边界 | [`wind_tunnel_stanford_sspg_20260816_2049_night_taumax60.md`](./wind_tunnel_stanford_sspg_20260816_2049_night_taumax60.md) |
| 3 | Stanford OGTT-CGM ($n=21$) | 标准化 75g 口服葡萄糖负荷（临床严格监督，与 CGMacros 完全独立协议；21 人中 11 人与 SSPG 夜间队列重叠，10 人全新） | SSPG 连续值 + 2 分类 | $P(\text{IR}>\text{IS})=$ **0.8333**<br>$\rho(\text{SSPG})=+0.581$ ($p=0.006$)<br>$\rho(\text{DI})=-0.674$ ($p=0.002$) | ✅ **第二次独立扰动源印证通过** | [`wind_tunnel_stanford_ogtt_20260816_2111_crossvalidation.md`](./wind_tunnel_stanford_ogtt_20260816_2111_crossvalidation.md) |

**毕业进度：** ✅ **毕业标准 1（跨异构扰动源多重印证）已满足**——两个协议独立的扰动源（自由餐 / 标准 75g OGTT）均给出 $P \ge 0.83$。✅ **标准 2（缺失碳水元数据时的降级路径设计）已于 2026-08-19 12:24 完成并通过回归验证**（见下方《标准 2 完成记录》）。标准 3（人类架构师显式批准）**仍未处理，因此仍不得触发 B.5 节收编**。**当前状态：暂存中，2/3 毕业条件达成，唯一缺口是人类签核——2026-08-19 12:31 已就此正式征询用户，裁决为"暂不批准，继续观察更多队列后再决定"（终局性，非临时搁置），不触发 B.5 节收编。**

**补充验证（不计入毕业标准 1 的独立扰动源计数，因扰动协议类型与 CGMacros 相同——均为自由生活自报饮食，非异构来源）：**

| # | 印证队列 | 扰动来源 | 分光镜标签 | 秩分离度 $P$ | 状态 | 报告 |
|---|---|---|---|---|---|---|
| 补充 | BIG IDEAs ($n=14$) | 自由生活自报饮食日志（HbA1c 窄带 5.3–6.4%，全员非糖尿病） | HbA1c 中位数分组 | $P=0.633$（均值聚合），未达显著($p=0.456$) | 🟡 方向一致（第三次），但窄标签带+小样本致电力不足；未归一化的 `delta_g` 反而 $P=0.796$ 反超——**2026-08-19 12:20 已排查，见下方《`delta_g` 反超现象排查记录》，判定为非普适性质** | [`wind_tunnel_big_ideas_20260819_1152_meal_dynamics.md`](./wind_tunnel_big_ideas_20260819_1152_meal_dynamics.md) |

#### 标准 2 完成记录（候选 #1/#2 共用，2026-08-19 12:24）

- **审计发现：** `analyze_subject_meals()`（`validate/wind_tunnel_v4_cgmacros_meals.py`）此前写法为 `carbs = m.get("carbs", 0.0)` + `if carbs < min_carbs: continue`——`.get(..., 0.0)` 的默认值只在字典**完全缺少** `"carbs"` 键时生效；若某条进餐记录携带显式 `carbs=None`（元数据缺失但键存在），`None < 25.0` 会在 Python 3 中直接抛出 `TypeError`，导致整个受试者的分析崩溃，而不是诚实跳过该条记录。当前生产链路能"幸免于难"，纯粹是因为 `export_cgmacros_subjects.py`/`export_big_ideas_subjects.py` 恰好都在导出层用 `pd.notna(...) else 0.0` 把缺失值提前填成了 `0.0`——这是一个**隐性契约**，一旦未来新增第三个数据集的导出脚本忘记做同样的填充，或生产环境（`index_v4.html`）开放手动记餐输入且用户留空碳水字段，就会复现崩溃。
- **修复（防御性加固，非生产代码，位于暂存区脚本内）：** 改为在使用点显式校验 `carbs is None or not isinstance(carbs, (int, float))`，与低于 `min_carbs` 门槛同等处理——**直接跳过该条进餐事件，不纳入 `w_carb`/`strain_per_carb` 的计算，且绝不用队列均值/固定常量顶替缺失的碳水克数**。这是毕业标准 2 明确列出的两条合法路径之一（"返回 null 或改用无需该元数据的替代表达式"中的前者，在单条记录粒度上体现为"排除该记录"；若某受试者的全部记录都被排除，函数原有的 `{"error": "No valid meal challenges found."}` 路径会诚实返回，不会伪造一个受试者级别的默认值）。
- **回归验证：** 用补丁后的代码重跑 CGMacros（45/45 成功）与 BIG IDEAs（14/16 成功，排除数与之前一致）meal 管线，产出 `reports/wind_tunnel_cgmacros_meal_dynamics_20260819_1224.json` 与 `reports/wind_tunnel_big_ideas_meal_dynamics_20260819_1224.json`，与补丁前的历史结果逐字段比对后**完全一致（byte-for-byte identical）**——证明该加固对现有真实数据零影响，纯粹是面向未来数据集/生产环境的防御性收紧，不构成任何结论回溯修改。
- **结论：** 该修复对共用此函数的两个候选算子同等生效，但对毕业进度的实际影响不同——候选 #1（`w_carb`）已满足标准 1，标准 2 的完成使其**推进至 2/3**（见上方毕业进度）；候选 #2（`strain_per_carb`）标准 1 本身尚未达标（见下方候选 #2 记录，$P=0.7731<0.80$），本次修复只是消除了它作为候选算子的一项潜在稳健性缺口，**暂存状态本身不变**（仍卡在标准 1，标准 2 的达成对已经未毕业的候选没有实质推进意义）。

#### `delta_g` 反超现象排查记录（2026-08-19 12:20）

- **背景假说：** BIG IDEAs 报告观察到未归一化的原始峰值升幅 `delta_g`（$P=0.796$）反而比按碳水归一化的 `w_carb`/`strain_per_carb`（$P=0.61$–$0.63$）秩分离度更高，提出"自报碳水克数误差污染了归一化分母，反而拖累了本应更精确的指标"这一待验证假设。
- **排查方法：** 直接复用 CGMacros（实验室称重进食，碳水记录几乎无自报误差）与 Stanford OGTT-CGM（标准化 75g 纯葡萄糖负荷，全员碳水剂量完全恒定）这两个已跑过风洞且拥有精确碳水数据的既有队列结果文件，用同样的秩分离度口径重新计算 `delta_g` 对比 `specific_work`/`strain_per_carb` 的表现，若该假说具有普适性，理应在"碳水记录精确"的队列上看到 `delta_g` 不再反超（甚至落后）归一化指标。
- **CGMacros 结果（$n=45$，HbA1c 3 分组）：** `mean_delta_g` 在 Pre-diabetes vs Normal 上 $P=0.7292$（`mean_specific_work`=**0.8333**，`mean_strain_per_carb`=**0.8292**），归一化指标明显更优；但在 T2D vs Normal 上 `mean_delta_g`=**0.9286** 略高于 `specific_work`(0.9143)/`strain_per_carb`(0.9143)。**结论：混合结果，delta_g 并未在这个碳水记录精确的队列上系统性反超。**
- **Stanford OGTT-CGM 结果（$n=21$）：** 该协议下全员碳水剂量固定为 75g，`strain_per_carb = delta_g / 75` 是 `delta_g` 的纯比例缩放，秩排序**数学上必然完全相同**（两者 $P=0.7731$，$\rho(\text{SSPG})=0.486$ 逐位一致）——这是一次**退化对照**（碳水恒定时归一化不可能改变秩序），不能作为该假说的证据，仅作诚实记录排除。
- **裁决：** CGMacros 的混合结果（一升一降，且从未观察到 BIG IDEAs 那种全面反超）**不支持**"归一化在碳水记录精确时应稳定跑赢原始值"这一假说的镜像预测，也不支持"delta_g 天生更优"的反向假说。现有证据更倾向于：BIG IDEAs 那次的 $P=0.796$ 反超**是小样本（$n=14$）+ 窄 HbA1c 带下的抽样噪声，或该队列自报饮食日志误差的队列特有效应，而非任何归一化算子的普遍缺陷**。按第 8.2 节《诚实失败》，本假说不升格为结论，也不据此调整候选 #1/#2 的算子定义；`specific_work`/`strain_per_carb` 归一化设计维持不变，$w_{\text{carb}}$ 仍是暂存区里唯一同时满足标准 1+2 的候选。

---

### 候选 #2：`strain_carb`（碳水应变敏感度 / Thermodynamic Strain per Carb）

- **定义：** $\text{strain}_{\text{carb}} = \Delta G / \text{Carbs}$，其中 $\Delta G = G_{\text{peak}} - G_{\text{base}}$（单位：$\text{mmol}/(\text{L}\cdot\text{g})$）。
- **物理领地：** 同候选 #1，生化冲击相专属。
- **实现位置（暂存，非生产）：** `validate/wind_tunnel_v4_cgmacros_meals.py`

| # | 印证队列 | 扰动来源 | 分光镜标签 | 秩分离度 $P$ | 状态 | 报告 |
|---|---|---|---|---|---|---|
| 1 | CGMacros ($n=45$) | 自由进食 | HbA1c 3 分组 | Pre-dia vs Normal: **0.8292**<br>T2D vs Normal: **0.9143** | ✅ 首次胜利 | [`wind_tunnel_cgmacros_20260816_2100_meal_dynamics.md`](./wind_tunnel_cgmacros_20260816_2100_meal_dynamics.md) |
| 2 | Stanford OGTT-CGM ($n=21$) | 标准化 75g 口服葡萄糖负荷 | SSPG 连续值 + 2 分类 | $P(\text{IR}>\text{IS})=$ **0.7731**（未达 0.80）<br>$\rho(\text{SSPG})=+0.486$ ($p=0.025$) | 🟡 方向一致、连续相关显著，但秩分离度差 0.03 未达门槛 | [`wind_tunnel_stanford_ogtt_20260816_2111_crossvalidation.md`](./wind_tunnel_stanford_ogtt_20260816_2111_crossvalidation.md) |

**毕业进度：** 第二次独立扰动源验证**未达标**（0.7731 < 0.80 门槛，差距 0.03）。按第 9.3 节严格纪律，不因"接近"而放宽标准。**当前状态：继续暂存观察，不进入毕业候选，需更大样本队列复测才能定论是否为小样本噪声。**

**补充验证：** BIG IDEAs ($n=14$，同类自由餐扰动，不计入独立源计数)：中位数聚合 $P=0.612$，方向一致但更弱，与窄 HbA1c 带下电力不足的预期一致。详见候选 #1 表下方同一份补充记录。

---

### 候选 #3：`tau_relax`（弛豫恢复时间 / Post-Perturbation Relaxation Time）

- **定义：** 从血糖峰值回落至 $G_{\text{base}} + \max(0.5, 0.2\Delta G)$ 阈值所需的时间（分钟）。
- **物理领地：** 生化冲击相专属。
- **实现位置（暂存，非生产）：** `validate/wind_tunnel_v4_cgmacros_meals.py`

| # | 印证队列 | 扰动来源 | 分光镜标签 | 秩分离度 $P$ | 状态 | 报告 |
|---|---|---|---|---|---|---|
| 1 | CGMacros ($n=45$) | 自由进食 | HbA1c 3 分组 | Pre-dia vs Normal: **0.7583**<br>T2D vs Normal: **0.7952** | 🟡 方向一致但未达 0.80 门槛 | [`wind_tunnel_cgmacros_20260816_2100_meal_dynamics.md`](./wind_tunnel_cgmacros_20260816_2100_meal_dynamics.md) |
| 2 | Stanford OGTT-CGM ($n=21$) | 标准化 75g 口服葡萄糖负荷 | SSPG 2 分类 | **0.2593（方向反转！）**<br>IS 组中位弛豫时间 109 min 反而长于 IR 组 77 min | ❌ **判定失败，方向不可复现** | [`wind_tunnel_stanford_ogtt_20260816_2111_crossvalidation.md`](./wind_tunnel_stanford_ogtt_20260816_2111_crossvalidation.md) |

**毕业进度：** ❌ **候选资格撤销。** 第二次独立扰动源不仅未达标，方向直接反转（很可能是 75g 纯葡萄糖负荷 vs 混合餐胃排空动力学差异导致的协议混杂，而非真实生理信号）。**当前状态：降级处理，归类为与历史上的 `DET`/`ENTR` 同类的"方向不可复现"指标，除非改用完全不同的算法（如基于曲线形状而非单点阈值穿越）重新设计，否则不再作为候选。**

---

### 候选 #4：`dim`（嵌入维度，作为"跨负荷类型的代偿指纹"假说，观察性登记）

- **性质说明：** `dim`（False Nearest Neighbors 估计的相空间嵌入维度）**已经是生产算子**（`index_v4.html` 第一层硬计算测度仪的既有输出），不是一个新发明的公式。本条登记的不是"要不要把 `dim` 加入生产代码"（它已经在），而是**观察性地追踪一个新兴假说**："`dim` 在 Work Integral 失效的场景下，可能独立携带与代偿负荷相关的方向性信号"——这个假说本身尚未满足任何收编标准，登记于此仅为防止两次独立观察被遗忘，且明确其**不触发**任何蓝图/代码修改。
- **物理领地（假说，未锁定）：** 或许是某种跨负荷类型（代谢代偿负荷、机械活动负荷均已观察到信号）的通用指纹，机制完全未知，不得升格为任何因果断言。

| # | 印证队列 | 负荷/扰动来源 | 分光镜/设计 | 观察到的信号 | 状态 | 报告 |
|---|---|---|---|---|---|---|
| 1 | Stanford SSPG ($n=29$) | 静息夜间（Work Integral 已确认失效场景） | SSPG 连续值 + 2 分类，跨受试者相关 | $\rho(\text{DI})$ 显著（$p<0.02$），方向：DI 越低（代偿越差）dim 越高 | 🟡 初次观察，未做多重比较校正 | [`wind_tunnel_stanford_sspg_20260816_2049_night_taumax60.md`](./wind_tunnel_stanford_sspg_20260816_2049_night_taumax60.md) |
| 2 | T1D-UOM ($n=17$) | 同一受试者高活动周 vs 低活动周（机械负荷，同体配对） | 周步数中位数分割，同体配对 | 符号检验 $p=0.0018$（**Holm-Bonferroni 5 指标家族校正后仍显著**），13/17 人方向一致（高活动周 dim 更高）；但 Wilcoxon（幅度加权）$p=0.0154$ **未通过**同一校正 | 🟡 方向一致性经校正确认，但效应量显著性未通过校正——"弱而稳，非强而清" | [`wind_tunnel_t1d_uom_20260819_1119_activity_paired.md`](./wind_tunnel_t1d_uom_20260819_1119_activity_paired.md) |

**毕业进度：** 0/3。本假说**不是**候选 #1-3 意义上的"新公式候选"，第 9.3 节《拓扑胜利判定》与本文件《毕业标准》并不直接适用（没有新代码要收编）。此登记的唯一作用是**防止遗忘**：若未来在第三个独立数据集、第三种负荷定义下再次观察到方向一致的信号，才值得认真讨论"是否要把 `dim` 的这个方向性质写入 Blueprint 的解读附注"（依然不涉及修改计算逻辑本身，`dim` 早已在生产代码里）。**当前状态：观察中，不触发任何收编或代码修改。**

---

### 候选 #5：`relaxationTime`（弛豫衰减疲劳度 / Excursion-Recovery Time-to-Half-Amplitude）

- **定义：** 每次自然发生的强迫性血糖偏移（峰-谷幅度 $>1.5$ mmol/L）从峰值回落到 50% 幅度所需时间（分钟），取全记录期内所有偏移事件的中位数。与候选 #3 `tau_relax`（进食冲击相专属、基于绝对阈值 $G_{\text{base}}+\max(0.5,0.2\Delta G)$ 穿越、已撤销）是**两个不同算法**：本候选来自 `index_v4.html` 的遗留 JS 函数 `computeExcursionKinetics`（v8.0-v8.4 时代，非风洞项目产物，第一次被系统性检验），不区分进食/自然诱因，作用于全记录期而非单次进餐窗口。为避免与已撤销的候选 #3 混淆而特别注明。
- **物理领地：** 全时段通用（不区分干预相/自然相），衡量"代偿弹簧回弹时间常数"，是 AGENTS.md 第 7.2 节《动力学映射接口》里 `Relaxation Time` 概念的直接产物。
- **性质说明：** 该指标当前**已经是 HUD 生产卡片**（"弛豫衰减疲劳度"，纯 JS，无 Pyodide/Python 对应实现），但其 warn/bad 判色阈值源自 v8.0-v8.4 时代仅在 Hall/Colas（N=265）上做标签回归拟合的结果，从未接受过第 9 节风洞方法论的样本外检验。本条登记是"是否要把现有卡片的阈值判色升级为经过风洞验证"的候选，不是"是否要新增一个尚不存在的算子"。
- **实现位置（暂存，非生产）：** `validate/_legacy_metrics_v4.py::compute_excursion_kinetics`（JS 端口，经 `validate/crosscheck_legacy_metrics.py` 对真实 JS 交叉验证 0 处不匹配）。

| # | 印证队列 | 扰动来源 | 分光镜标签 | 秩分离度 $P(\text{高危}>\text{低危})$ | 置换检验 $p$（6 指标 Holm-Bonferroni） | 状态 | 报告 |
|---|---|---|---|---|---|---|---|
| 1 | Stanford SSPG ($n=29$) | 自由生活自然发生偏移（非标准化负荷） | SSPG 2 分类 (IS $n=16$ / IR $n=13$) | **0.7909** | **0.0060**（6 指标族校正后存活，排名 1/6） | 🟡 统计显著、方向与生理预期一致，但效应量**未达** 0.80 门槛（差 0.0091） | [`wind_tunnel_stanford_sspg_legacymetrics_20260819_1730_analysis.md`](./wind_tunnel_stanford_sspg_legacymetrics_20260819_1730_analysis.md) |
| 2 | Shanghai_T2DM ($n=87$ 有效HbA1c，去重后) | 住院病历自由生活（异构协议，与 Stanford 完全不同的采集方式与标签语义） | HbA1c 全体固定中位数二分 (high $n=43$ / low $n=44$) | **0.7090** | **0.0008**（6 指标族校正后存活，排名 4/6） | 🟡 **第二次方向一致 + 统计显著复现**，效应量仍**未达** 0.80 门槛（差 0.0910） | [`wind_tunnel_shanghai_t2dm_legacymetrics_20260819_1947_analysis.md`](./wind_tunnel_shanghai_t2dm_legacymetrics_20260819_1947_analysis.md) |

**毕业进度：** 标准 1 要求"至少 2 个异构队列复现秩分离度 $P>0.80$"——两次独立复现方向一致、统计显著（双重穿透各自的 6 指标 Holm-Bonferroni 校正），是本轮 6 个候选中唯一实现"双队列一致复现"的指标，但**两次效应量均未突破 0.80**（0.7909、0.7090）。按第 9.3 节严格纪律，"方向一致+统计显著"不等价于"拓扑胜利"，不因两次都"接近但未达"而放宽标准或宣布毕业。**当前状态：候选级观察中，0/3 毕业条件严格意义上仍未满足。**

**算法重新设计尝试第一轮（渐进修正，2026-08-19，详见 [`relaxation_time_redesign_20260819_evaluation.md`](./relaxation_time_redesign_20260819_evaluation.md)）：** 诊断否定了"个体内事件计数过少导致噪声"的原假说（两队列事件数中位数均 $\ge 30$）；转而发现原算法用离散采样点索引做阈值穿越判定，对采样网格粒度敏感（Zero Magic-Constant 公理排查还发现原算法对未衰减事件用 `duration×1.5` 人工外推，属未文档化魔法常数）。测试两个零自由参数改造：**v2（线性插值穿越时间）** 在 Stanford 上首次突破 0.80（$P=0.8077$），Shanghai 几乎不变（$P=0.7101$）；**v3（插值+剔除未衰减事件）** 在 Stanford 进一步改善（$P=0.8413$）但在 Shanghai 反而变差（$P=0.6839$，跌破原版）——跨队列方向不一致，判定为对 Stanford 噪声结构的过拟合，**不予采纳**。

**算法重新设计尝试第二轮（更根本重设计，用户批准）：** 认识到"弛豫时间"本身在世界观协议里就是指数衰减时间常数概念，而非任意百分比阈值穿越，遂放弃阈值穿越设计，改为对整段衰减轨迹做对数线性最小二乘拟合提取时间常数 $\tau$（$G(t)=G_{\text{baseline}}+(G_{\text{peak}}-G_{\text{baseline}})e^{-t/\tau}$，锚点复用既有的"前一谷值"，不新增自由参数）。**关键验证**：两队列拟合质量中位数 $R^2$ 均 $>0.92$，独立证实该物理模型是衰减轨迹的良好描述。结果（`validate/_relaxation_time_v4_expfit.py`）：**v4a（无质量门槛）** 是三轮重新设计中**唯一让两队列同时改善（无此消彼长）**的版本——Stanford $P=0.7933$（基本持平原版），Shanghai $P=0.7500$（较原版 0.7090 提升 0.041，$n=95$ 队列上有意义的改善）；加 $R^2\geq0.5$ 质量门槛（v4b）几乎不排除任何事件（说明拟合本身已经很干净），反而略微降低效应量，判定不采纳门槛。**即便如此 Shanghai 仍未突破 0.80**，两队列差距缩小但未消除。

**第三队列同体配对测试（T1D-UOM，用户批准方向，详见 [`wind_tunnel_t1d_uom_legacymetrics_20260819_2027_paired_analysis.md`](./wind_tunnel_t1d_uom_legacymetrics_20260819_2027_paired_analysis.md)）：** ⚠️ **方法学性质不同，不计入毕业标准 1 的异构队列计数**——本次测的是同一受试者自己的高/低活动周同体配对（与候选 #4 `dim` 相同设计），而非跨受试者慢性代偿状态分组，是不同时间尺度的生理学问题。结果：v1（原版）留下一个未经 7 指标 Holm-Bonferroni 校正就消失的弱趋势（Wilcoxon $p=0.0305$，符号检验 $p=0.0490$，均未存活）；**v4a（指数拟合重设计版）在此配对设计上几乎无信号**（效应量 Cohen's d $\approx -0.05$）——与其在两个横截面队列上"同时改善"的表现形成反差,提示 v4a 的改进可能是横截面比较特有的,不构成"v4a 全面优于 v1"的证据。意外发现：代谢引力场角速度以极小差距（$p=0.0079$ vs Holm 阈值 $0.0071$）未能过校正,记录在案但不登记为候选。

**用户裁决（2026-08-19）：** 三队列测试完毕后，用户明确决定不再继续投入 `relaxationTime` 的进一步验证/重新设计。**终局状态：维持观察级，0/3 毕业条件未满足，暂停研究，不升格、不撤销。** 若未来出现新的独立队列或更根本的算法思路，可重新评估；本条目本身作为"已充分探索但未能突破效应量门槛"的诚实记录保留。

**同批次 Fail-Closed 记录（不登记为候选，仅存档避免重复测试）：**
- 早相加速度迟滞、AR1：**两个独立队列均 Fail-Closed**（Stanford $p=0.18/0.53$；Shanghai $p=0.65/0.39$），一致性反而确认了这两张卡的现有判色阈值在样本外数据上完全没有区分度实证支持。
- 上升相阻力、夜间相变阻力：**队列间不一致**——Stanford 上 Fail-Closed（$P=0.625,p=0.26$），但 Shanghai 上强显著（$P=0.76/0.78,p<0.0001$）。经《同态锚定熔炉》损耗预判排查：Shanghai 用 HbA1c（血糖暴露 3 个月汇总）做分光镜，而这两个摩擦力指标本身直接由血糖轨迹偏离幅度计算，二者存在标签语义重叠（同一暴露过程的两种时间尺度读数），Shanghai 的强信号判定为**标签同义反复伪影，不可采信为独立生理证据**，不登记为候选。
- 代谢引力场角速度：方向两次一致（危险组更低），但显著性不稳定（Stanford $p=0.12$ 未过校正；Shanghai $p=0.0006$ 过校正），弱于 relaxationTime 的双重显著复现，暂不登记为独立候选。

---

### 第一组6张中性卡（Volume/Recovery/λ1λ2/Box-Counting Dim/Lyapunov/Core Dist）——审计+检验存档

不登记为候选（无一项进入候选生命周期），仅存档避免未来重复投入：

- **Phase B 冗余审计（Stanford SSPG $n=29$ + Shanghai T2DM $n=104$，与已毕业指标 workIntegral/DET/ENTR/Dim 做队列内 Spearman 秩相关）：**
  - **Volume**：与 Work Integral 强相关（Stanford $\rho=+0.545$，Shanghai $\rho=+0.725$）——判定**实质性冗余**，未投入 Phase C 检验。
  - **Lyapunov**：与嵌入维度 Dim 强相关（Stanford $\rho=-0.753$，Shanghai $\rho=-0.557$）——判定**实质性冗余**（机制上 `dim` 直接决定 Lyapunov 计算所用相空间坐标数，本就不独立），未投入检验。
  - **Recovery**：与 Volume 内部强相关（Shanghai $\rho=-0.74$）+ 概念上与候选 #5 `relaxationTime` 高度重叠（同为"扰动后向核心恢复速率"，一个用几何步长一个用时间轴）——判定**双重冗余**，未投入检验。
  - **Shape Ratio (λ1/λ2)**、**Box-Counting Dim**：与 Dim 中等耦合（$\rho\in[0.27,0.57]$，队列间不完全一致）——用户裁决暂不投入检验。
  - **Core Dist**：与全部4个已毕业指标相关性最弱（$|\rho|\le0.29$），**审计通过，判定为6者中最独立**，是唯一进入 Phase C 的指标。
- **Phase C 拓扑对撞检验（仅 Core Dist，用户批准范围）：** Stanford SSPG (`sspg_class`) $P(\text{IR}>\text{IS})=0.5192$，置换检验 $p=0.8598$；Shanghai T2DM (HbA1c 中位数分组) $P(\text{high}>\text{low})=0.4704$，置换检验 $p=0.6454$（方向与预期相反）。**两队列均与抛硬币无法区分，判定 Fail-Closed**——"与已毕业指标无冗余"不等于"携带有效信号"。

详见 [`group1_neutral_metrics_redundancy_audit_20260819_2100.md`](./group1_neutral_metrics_redundancy_audit_20260819_2100.md)（含 Phase A 工程移植/交叉验证记录）。

---

## 变更日志 (Append-Only, 不可回填删除)

- **2026-08-19 21:50** — 完成 Work Integral / relaxationTime / Angular Velocity / Ascend Friction / Night Friction 五张卡的判色中性化（用户批准，`index_v4.html` 原子化 UI 修改，延续 earlyDelay/AR1 的处理模式）。这五张卡此前虽已在本文件登记为候选 #5 同批 Fail-Closed/观察级记录，但其生产 UI 的 warn/bad 判色逻辑此前未被同步修改——本次修复了这一"证据已记录但 UI 未跟进"的缺口。详见 `dataset_fleet_registry.md` 同时刻变更日志。不涉及任何计算逻辑或候选算子毕业进度变化，纯 UI 层面的判色关闭。
- **2026-08-19 21:30** — 完成"第一组6张中性卡"（Volume/Recovery/λ1λ2/Box-Counting Dim/Lyapunov/Core Dist）的完整审计+检验闭环（用户批准顺序：先 Phase A 工程移植+交叉验证 → Phase B 冗余审计 → 用户裁决 Phase C 范围"仅测最独立的 Core Dist"）。Phase A：JS→Python 移植 0 处不匹配，Hall 烟雾测试零异常。Phase B：与已毕业指标（workIntegral/DET/ENTR/Dim）做队列内 Spearman 秩相关，Volume（与 workIntegral $\rho$ 高至 0.725）、Lyapunov（与 Dim $\rho$ 达 -0.753，机制上本就非独立）、Recovery（与 Volume 内部 $\rho=-0.74$ + 概念上与候选 #5 重叠）判定冗余未测；Shape Ratio/Box-Counting Dim 中等耦合用户裁决不测；Core Dist 审计通过（$|\rho|\le0.29$）。Phase C：Core Dist 在 Stanford SSPG（$P=0.5192,p=0.8598$）与 Shanghai T2DM（$P=0.4704,p=0.6454$，方向相反）两队列均判定 Fail-Closed，与抛硬币无法区分。**结论：第一组6张卡无一项进入候选生命周期**，全部诚实存档为 Fail-Closed，不登记为候选，不改动 `index_v4.html`/Blueprint/Contract。详见 [`group1_neutral_metrics_redundancy_audit_20260819_2100.md`](./group1_neutral_metrics_redundancy_audit_20260819_2100.md)。
- **2026-08-19 20:38** — 用户明确决定：候选 #5（`relaxationTime`）三队列测试（Stanford SSPG / Shanghai T2DM / T1D-UOM 配对）+ 两轮算法重新设计（v2/v3 插值修正、v4 指数拟合）后，不再继续投入。**终局状态：维持观察级，暂停研究，不升格为已毕业候选，不撤销候选资格**。这是本轮"11 张遗留 HUD 卡片风洞检验"计划里第二组 6 张判色卡的第一个也是投入最深的候选，为该系列研究画上一个诚实的（非胜利、非失败）阶段性结论。
- **2026-08-19 20:32** — 完成候选 #5（含 v4a 重设计版）在 T1D-UOM 队列的同体配对测试（用户批准方向，与候选 #4 `dim` 同设计）。**明确声明该测试方法学性质与 Stanford/Shanghai 不同（同体纵向 vs 跨受试者横截面），不计入毕业标准 1 的异构队列复现计数**。结果：v1 留下未经 7 指标 Holm-Bonferroni 校正就消失的弱趋势；**v4a 在此配对设计上几乎无信号**，与其在横截面队列上的改善形成反差，提示改进效果可能场景特定，不应假设 v4a 全面优于 v1。意外发现代谢引力场角速度以极小差距未过校正，记录在案不登记。详见 [`wind_tunnel_t1d_uom_legacymetrics_20260819_2027_paired_analysis.md`](./wind_tunnel_t1d_uom_legacymetrics_20260819_2027_paired_analysis.md)。生产代码/风洞主管线均未改动。
- **2026-08-19 20:45** — 完成候选 #5 更根本的重新设计（用户批准方向）：放弃阈值穿越，改用指数衰减曲线拟合提取时间常数 $\tau$（`validate/_relaxation_time_v4_expfit.py`）。两队列拟合质量中位数 $R^2>0.92$，独立证实"弛豫时间常数"物理模型合理。v4a（无质量门槛）是三轮重设计中唯一让两队列同时改善的版本，Shanghai 提升幅度最大（$P$: 0.7090→0.7500），但仍未突破 0.80；$R^2$ 质量门槛测试后判定不采纳（几乎不排除数据，反而略微降低效应量）。候选 #5 维持观察级。详见 [`relaxation_time_redesign_20260819_evaluation.md`](./relaxation_time_redesign_20260819_evaluation.md) 更新章节。生产代码/风洞主管线均未改动。
- **2026-08-19 20:20** — 完成候选 #5（`relaxationTime`）的算法重新设计尝试（用户批准方向）。诊断否定"事件计数过少"原假说；发现原算法存在采样网格敏感性与未文档化的 `duration×1.5` 魔法常数（未衰减事件回退）。测试两个零参数改造：插值穿越版本（v2）单调不劣于原版并在 Stanford 首次突破 0.80，但 Shanghai 几乎不变；插值+剔除未衰减版本（v3）在 Stanford 进一步改善但在 Shanghai 反而变差——跨队列方向不一致，判定为过拟合，不予采纳。候选 #5 维持观察级，未满足毕业标准 1。详见 [`relaxation_time_redesign_20260819_evaluation.md`](./relaxation_time_redesign_20260819_evaluation.md)。生产代码/风洞主管线均未改动。
- **2026-08-19 19:52** — 候选 #5（`relaxationTime`）完成第二个异构队列（Shanghai_T2DM，HbA1c 分光镜）复现：方向一致、统计显著（$p=0.0008$，穿透 6 指标 Holm-Bonferroni），是本轮唯一双队列一致复现的指标，但效应量仍未达 0.80 门槛（0.7090），毕业进度维持未满足标准 1，不因两次"接近"而放宽。同批测试意外发现上升相阻力/夜间相变阻力在 Shanghai 上强显著（$p<0.0001$）而在 Stanford 上沉默——经《同态锚定熔炉》损耗预判排查，判定为 HbA1c 分光镜与摩擦力算子本身的标签语义重叠伪影（同一血糖暴露过程的两种时间尺度读数，非独立生理证据），不采信、不登记为候选。早相加速度迟滞/AR1 两队列均 Fail-Closed，确认现有 HUD 判色阈值在样本外数据无区分度支持。详见 [`wind_tunnel_shanghai_t2dm_legacymetrics_20260819_1947_analysis.md`](./wind_tunnel_shanghai_t2dm_legacymetrics_20260819_1947_analysis.md)。
- **2026-08-19 17:38** — 新增候选 #5（`relaxationTime`/弛豫衰减疲劳度）观察性登记。这是"11 张未受风洞检验的遗留 HUD 卡片"清理工作的首个真正样本外结果：6 个被移植验证的 v8.0-v8.4 时代判色指标（早相加速度迟滞/弛豫衰减疲劳度/AR1/角速度/上升阻力/夜间阻力），在从未参与过旧阈值拟合历史的 Stanford SSPG 队列（$n=29$）上首次测试，**仅 `relaxationTime` 一项**穿透 6 指标 Holm-Bonferroni 校正（置换检验 $p=0.0060$），但效应量 $P=0.7909$ 未达 0.80 门槛，不因"接近"放宽为已满足标准；其余 5 项全部 Fail-Closed。详见 [`wind_tunnel_stanford_sspg_legacymetrics_20260819_1730_analysis.md`](./wind_tunnel_stanford_sspg_legacymetrics_20260819_1730_analysis.md)。
- **2026-08-19 12:31** — 就候选 #1（`w_carb`）是否触发标准 3（人类架构师显式批准）向用户明确提问。**用户裁决：暂不批准，继续观察更多队列后再决定。** 遵照第 9.5 节《产物隔离》与《毕业标准》第 3 条，本决定为终局性——**不触发任何 B.5 节原子化蓝图收编**，`index_v4.html`/`Pipeline Blueprint` 均未改动。`w_carb` 维持"暂存中，2/3 毕业条件达成"状态，标准 3 的批准时点留给未来更多队列证据积累后由人类重新发起。
- **2026-08-19 12:24** — 完成候选 #1/#2 遗留的两个开放项：(a) **毕业标准 2** 审计发现 `analyze_subject_meals()` 对显式 `carbs=None` 会抛出 `TypeError`（当前生产链路靠导出脚本的隐性填充 0.0 幸免），已修复为在使用点显式判空并跳过该条记录（不纳入计算，不用常量顶替），回归验证 CGMacros(45/45)与 BIG IDEAs(14/16)结果逐字段 byte-for-byte 一致，零回溯影响；`w_carb` 候选 #1 由此推进至 2/3 毕业条件（仅缺标准 3 人类签核）。(b) 排查 BIG IDEAs 报告遗留的 `delta_g` 反超未归一化假说：复用 CGMacros（称重进食）与 Stanford OGTT（碳水恒定，退化对照）已有结果重新计算，未观察到该反超模式的普适复现（CGMacros 结果一升一降），判定为该队列小样本噪声，不升格为结论，不改动候选算子定义。
- **2026-08-19 12:09** — 【方法学注意事项，非候选算子登记】ShanghaiT1DM 的 `tau_max` 边界标定补充复算（详见 `wind_tunnel_shanghai_t1dm_20260819_1209_taumax120_supplementary_rerun.md`）发现：`DET`/`ENTR` 对 tau 窗口大小高度敏感（同一队列同一批受试者，tau 从 60 校正到 120 后，均值分别位移 -0.27/-0.63），机制是 RQA 的 Theiler 窗口 $T=\max(5,\tau)$ 随 tau 增大而排除更多邻近点对。**任何未来跨队列比较 DET/ENTR 绝对数值时，必须先核实各队列的 tau 是否处于同一截断状态**，否则差异可能只是窗口大小的副产品。同一次复算证实 `Dim`（恒为 2）与 Work Integral（反转方向）在两个 tau 窗口下结论一致，非截断伪影。
- **2026-08-19 11:52** — 完成 BIG IDEAs Food_Log 生化冲击深挖（$n=14$ 有效餐食受试者，704 次自报餐食事件）。`w_carb`/`strain_per_carb` 在这个 HbA1c 窄带(5.3-6.4%)全非糖尿病队列上方向仍与 CGMacros/Stanford OGTT 一致但未达显著（$P=0.61$-$0.63$），登记为补充验证，不计入毕业标准 1 的独立扰动源计数（同为自由餐协议，非异构来源）。意外发现：未按碳水归一化的原始峰值升幅 `delta_g` 反而秩分离度更高（$P=0.796$），提出"自报碳水克数误差污染归一化分母"假设，标注为待验证观察，不触发任何代码/蓝图修改。
- **2026-08-16 21:03** — 候选 #1/#2/#3 登记建立，来源 CGMacros 首轮胜利与 Stanford SSPG 夜间对照。决定暂缓直接收编，理由：(a) 单队列胜利存在幸存偏差风险；(b) 生产环境 `index_v4.html` 目前无碳水摄入输入通道，算子在缺失该元数据时会退化为 `null`；(c) 需要至少一个完全独立的第二扰动源交叉印证才满足第 9.3 节多队列标准。
- **2026-08-16 21:11** — 完成 Stanford 标准化 75g OGTT-CGM 队列（$n=21$）的第二次独立扰动源交叉验证：
  - `specific_work`/`w_carb`：**✅ 通过**（$P=0.8333$，$\rho(\text{SSPG})=+0.581$，$\rho(\text{DI})=-0.674$），满足毕业标准 1，但标准 2/3 仍未处理，**不触发收编**。
  - `strain_per_carb`：🟡 未达标（$P=0.7731 < 0.80$），继续暂存观察，不因"接近"放宽标准。
  - `tau_relax`：❌ 方向反转（$P=0.2593$），候选资格撤销，降级为"方向不可复现"指标。
  - 诚实记录：本轮样本量小（$n=21$，IR 仅 9 人），且 21 人中 11 人与此前夜间队列受试者重叠（非完全独立个体，但扰动协议独立）。
- **2026-08-19 11:19** — 新增候选 #4 观察性登记（`dim` 作为跨负荷类型代偿指纹假说），来源 T1D-UOM 同体高/低活动周配对分析对 Stanford SSPG 相关性观察的第二次独立复现。方向一致性通过 Holm-Bonferroni 校正，效应量未通过，明确标注 0/3 毕业进度、不触发任何代码/蓝图修改。
