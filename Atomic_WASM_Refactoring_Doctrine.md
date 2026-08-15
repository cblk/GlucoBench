# ☢️ The Atomic WASM Refactoring Doctrine (原子化重构法则)

> **跨平台声明:** 本文件是纯 Markdown 权威全文,任何具备文件读取能力的 Agent 运行时（Cursor / Codex CLI / DeepSeek Harness / Claude Code 等）均可直接打开本文件获取完整规则。Cursor 环境下,`.cursor/rules/atomic_wasm_workflow.mdc` 会在编辑 `*.html`/`*.js`/`*.py` 时自动将本文件内容注入上下文；非 Cursor 环境下，请在编辑 `index_v4.html` 等巨石文件前主动读取本文件。

当 AI 面对本项目中极度复杂的单文件（如 `index_v4.html` 这种超 3000 行，且融合了 JS UI、WASM/Pyodide 和内联 Python 代码的巨石应用）时，**绝对禁止使用常规的“通读全改”战术**。必须严格执行以下原子化工作流：

## 1. 绝对切片法则 (The Slicing Mandate)
- **禁止巨石写入：** 永远不要试图在一次输出中重写超过 100 行的核心逻辑代码。
- **定位即锁定：** 修改前，必须先使用 `Grep` 和 `Read` 工具精准锁定目标函数的作用域。
- **语义切片：** 架构修改必须按阶段推进（如：切片1处理I/O，切片2处理滤波，切片3处理降维）。前一个切片未获验证，绝不推进下一个。

## 2. 语言域隔离 (The Boundary of Languages)
- **前端 JS 的克制：** JavaScript / DOM 仅作为“展示面板”与“数据搬运工”。**绝对禁止**使用 JS/Vanilla Math 进行任何信号处理、矩阵分解、相空间重构或复杂的非线性运算。
- **Python 科学栈的统治：** 所有动力学算子（`filtfilt`, `cKDTree`, `SVD/PCA`, 稀疏矩阵）必须封装在内联的 `PYTHON_ENGINE_CODE` 中，交由 Pyodide / SciPy 运行。
- **跨界降维传输 (Sparse Transfer)：** Python 向 JS 传递矩阵（如 RQA 递归图）时，绝对禁止传输庞大的密集矩阵（会导致浏览器 OOM）。必须在 Python 端转化为稀疏坐标 (Sparse x, y) 再通过 JSON 序列化传出。

## 3. 沙盒验证防线 (Test-Driven Injection)
- **盲写禁止：** 在修改 `PYTHON_ENGINE_CODE` 中的 Python 代码时，**禁止直接使用 `StrReplace` 盲目替换**。
- **后台验证步骤：**
  1. 先将生成的 Python 代码使用 `Shell` 工具写入到后台临时文件（如 `_temp_test.py`）。
  2. 运行 `python -m py_compile _temp_test.py` 进行静态语法检查。
  3. 确认无 `SyntaxError` 和缩进错误后，才能将其合并进 HTML 字符串。
- **JS 语法校验：** 修改大段 JS 前，同样建议导出至临时文件，运行 `node -c _temp.js` 防止丢失括号导致白屏。

## 4. 资源绝对回收 (The LIFO Destructor)
- 任何 UI 重绘（特别是 Plotly 的 WebGL 图表，如 `scatter3d`, `scattergl`），在重新实例化（`Plotly.newPlot`）之前，必须**强制清空**过去的上下文（`Plotly.purge(id)`），或者复用上下文（`Plotly.react`）。
- 绝不允许由于高频滑动（如 `Tau` 滑块）导致的内存泄漏。

> **AI 系统指令 (System Override):** “你是一个进行纳米级手术的神经外科医生。不要试图举起整个病人，每次只切割你需要修复的那 1 毫米病灶。使用隔离环境进行排雷，确保不引起任何未捕获的级联崩溃。”
