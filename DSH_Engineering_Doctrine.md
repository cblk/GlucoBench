# DSH Engineering Doctrine (DSH 工程第一性原理统御契约)
*The Ultimate Map-Territory Law for GlucoBench*

> **跨平台声明:** 本文件是纯 Markdown 权威全文,任何具备文件读取能力的 Agent 运行时（Cursor / Codex CLI / DeepSeek Harness / Claude Code 等）均可直接打开本文件获取完整规则。Cursor 环境下,`.cursor/rules/dsh_engineering_doctrine.mdc` 会自动将本文件内容注入上下文；非 Cursor 环境下，请依据 `AGENTS.md` 顶部的"跨平台兼容性声明"主动读取本文件。
>
> 备注：本契约的命名与 E1-E16/T1-T12 编号体系源自真实存在的开源项目 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（`dsh`，deepseek-ai 于 2026 年发布的插件化 Agent 运行时），这里将其工程哲学移植为 GlucoBench 的执行纪律，并非该项目本身的文档。

本契约是 GlucoBench 系统架构的最高统御法则。它将 DeepSeek Harness (DSH) 的工程第一性原理（E1-E16）与地图-疆域信念（T1-T12）完整编译为适用于“肉身张量引擎（Somatic Tensor Engine）”的执行纪律。在任何代码编写、系统架构设计与数据高维解析中，**必须绝对服从**本契约。

## 第一卷：无限接近实在 (The Map-Territory Epistemology)
> 核心信念：传感器数据与患者肉身是“疆域（Territory）”，前端 UI、拓扑指标、大模型推理都是“地图（Map）”。地图必须无限逼近疆域，对地图保持怀疑，对疆域保持敬畏。

*   **[T2/T7] 严禁推断与虚构 (No Inference & No Fabrication):**
    *   疆域里没有的，地图绝不得创造。当数据缺失、长度不足或底层拓扑算子（如 ACF/FNN/RQA）崩溃时，**必须返回 `null`，绝不允许使用预设常量（如 `tau=20` 或 `return 0`）来粉饰太平。**
    *   未知就是未知，绝不升格为断言。大模型（Agent）在看到缺失或 `null` 数据时，严禁动用医学预训练知识去“平滑”或“脑补”患者的代谢状态。
*   **[T10/E9] 地图永远是派生物 (Disposable Maps):**
    *   UI（Plotly 图表、特征卡片）与缓存仅仅是“读模型（Read Models）”。它们可以随时被丢弃和重建（Plotly.purge）。
    *   **唯一真源（Single Source of Truth）是只追加的 `Event Log Sequence`**。只要事件日志记录了准确的计算轨迹与残差，UI 白屏随时可以修复；但若事件日志被篡改，系统即宣告死亡。
*   **[T11/T6] 诚实报告失败 (Honest Fail-Closed Analysis):**
    *   拒绝伪装成确定。如果在分析时系统陷入崩溃，或者由于数据维度不够导致算子抛出异常，报告必须诚实地记录 `[ERROR]` 与 Traceback。**把系统的无能如实暴露给人类，是一种最高级别的工程美德。**
*   **[T12] 尊重真实的物理成本 (Thermodynamic Bill for Inference):**
    *   大模型的推理消耗 Token，患者执行物理干预（冰浴、断食）消耗代谢冗余。干预策略必须极度保守并锁定补偿事务，绝不进行毫无代价约束的“微操（Micro-management）”。

## 第二卷：工程第一性原理 (Engineering First Principles)
> 核心信条：用只追加的日志约束所有地图，用 fail-closed 的姿态对待每次不确定，用显式标注对待每个猜测。

*   **[E5] 失败即关闭，沙箱不协商 (Fail-Closed Default):**
    *   **绝对律：** 在 `index_v4.html` 的双层架构中，一旦 Python (Pyodide/SciPy) 引擎在执行数学计算时抛出异常，**必须立即中断当前 Epoch (ABORTING Epoch)**。
    *   **绝对禁止“静默降级（Silent Fallback）”**：严禁在 Python 失败时退回使用不精准的 JavaScript 替补函数。失败的机床必须停机，不能换塑料锤子继续敲。
*   **[E4] 语言域的单一所有权 (Single Ownership):**
    *   Python/WASM 拥有“复杂拓扑数学运算（filtfilt, FNN, RQA）”的**唯一所有权**。
    *   JavaScript 拥有“UI 渲染与事件路由”的**唯一所有权**。
    *   严禁越权：JS 绝不能写数学算法，Python 绝不能直接操控 DOM。
*   **[E6] 历史仅追加与禁止截断 (Append-Only History):**
    *   代码崩溃时的 Traceback、异常数据的报错、物理干预导致的恶化，都是“残差”。**严禁**为了追求对话的美观而“截断”或删除这些耻辱的记录。它们必须作为不可变的事件永久封存在历史中，供后续演化（Darwinian Evolution）使用。
*   **[E14] 结构化异常作为系统契约 (Explicit Error Vocabulary):**
    *   错误必须是结构化的。Python 层传递给 JS 的错误不能只是含糊的字符串，必须在 `Event Log` 中明确打上 `[L0 ERROR]`, `[L1 WARNING]` 等阶段标签。大模型的决策强依赖于这些硬编码的故障锚点。
*   **[E7/E16] 内存与资源的原子性 (Atomic LIFO Destructor):**
    *   资源的分配与回收必须是原子的。任何产生大矩阵的操作（如 RP Matrix）必须以稀疏格式传输，任何 Plotly 图表的重绘必须先经历彻底的 `purge` 销毁。资源的所有权随生命周期结束必须立刻回收，防范 OOM。

## 第三卷：Agent 行为约束 (Agent Operational Mandate)
在阅读到本契约后，作为系统领航员的你（Agent），必须在所有代码输出、数据分析与诊断行为中：
1.  **猎杀 Fallback：** 审查一切新旧代码，消灭所有为了让程序“跑通”而设置的 Magic Constants（魔法常数回退）。
2.  **拦截幻觉：** 在输出诊断时，如果数据字段为 `null`，你必须在回答中明确写出：“由于该维度数据由于 [具体错误/数据不足] 缺失，我拒绝推断此部分的代谢状态。”
3.  **遵守沙箱：** 明白自己只是一个在高维推断节点上的“客体”，绝不试图越权修改系统底层的网络配置或隐瞒前端沙箱的安全阻断。
