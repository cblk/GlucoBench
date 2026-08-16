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

**毕业进度：** ✅ **毕业标准 1（跨异构扰动源多重印证）已满足**——两个协议独立的扰动源（自由餐 / 标准 75g OGTT）均给出 $P \ge 0.83$。标准 2（缺失碳水元数据时的降级路径设计）与标准 3（人类架构师显式批准）**仍未处理，因此仍不得触发 B.5 节收编**。**当前状态：暂存中，1/3 毕业条件达成。**

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

## 变更日志 (Append-Only, 不可回填删除)

- **2026-08-16 21:03** — 候选 #1/#2/#3 登记建立，来源 CGMacros 首轮胜利与 Stanford SSPG 夜间对照。决定暂缓直接收编，理由：(a) 单队列胜利存在幸存偏差风险；(b) 生产环境 `index_v4.html` 目前无碳水摄入输入通道，算子在缺失该元数据时会退化为 `null`；(c) 需要至少一个完全独立的第二扰动源交叉印证才满足第 9.3 节多队列标准。
- **2026-08-16 21:11** — 完成 Stanford 标准化 75g OGTT-CGM 队列（$n=21$）的第二次独立扰动源交叉验证：
  - `specific_work`/`w_carb`：**✅ 通过**（$P=0.8333$，$\rho(\text{SSPG})=+0.581$，$\rho(\text{DI})=-0.674$），满足毕业标准 1，但标准 2/3 仍未处理，**不触发收编**。
  - `strain_per_carb`：🟡 未达标（$P=0.7731 < 0.80$），继续暂存观察，不因"接近"放宽标准。
  - `tau_relax`：❌ 方向反转（$P=0.2593$），候选资格撤销，降级为"方向不可复现"指标。
  - 诚实记录：本轮样本量小（$n=21$，IR 仅 9 人），且 21 人中 11 人与此前夜间队列受试者重叠（非完全独立个体，但扰动协议独立）。
