# GlucoBench 外部数据集增补调研（2026-08-11）

## 1. 本轮结论

本轮找到 **2 个高价值、带直接代谢生理参照且已有公开数据入口的候选队列**，以及 **4 个适合申请后用于独立外部验证的大型队列**。最值得优先核验的是：

1. **Kobe CGM + OGTT + 双 clamp 队列**：64 人，CGM 与 OGTT、连续高血糖 clamp/高胰岛素-正常血糖 clamp 同期采集；CGM 和论文源数据已有公开入口。它最接近“CGM 动力学对应独立胰岛素敏感性/分泌能力”的目标。
2. **Stanford metabolic subphenotype 队列**：29 人具有居家 CGM 和深度代谢表型，参照包括 SSPG、16 点 OGTT、胰岛素/C-peptide、处置指数、肝胰岛素抵抗与肠促胰素效应；数据包公开。样本虽小，但标签强度最高。
3. **AI-READI v3**：2,280 人、Dexcom G6 约 10 天，临床表包含胰岛素、C-peptide、葡萄糖和 HbA1c；适合作为较大规模的 HOMA/临床分层外部验证候选，但必须先核实采血是否严格空腹以及用药字段的访问等级。
4. **Human Phenotype Project / 10K、PREDICT 1、Framingham Exam 4**：规模和 CGM 时长更适合作为最终盲测或跨设备验证，但均需要申请，且必须在数据字典中确认“同一参与者、同一观察窗”的胰岛素或直接胰岛素抵抗标签。

本轮没有下载参与者级数据、没有拟合模型、没有制定新阈值，也没有改动 `index.html`。因此，以下分级是**数据集适配度评估**，不是算法效能或临床有效性结论。

## 2. 冻结的研究问题与准入门槛

### 2.1 当前真正缺少的证据

现有 GlucoBench 队列可以支持多队列一致性、夜间样本数稳定性和部分血糖学终点，但当前候选分数尚未达到冻结/部署条件，且 Hall 已参与探索，不能再充当最终未见临床确认集。新增队列应优先补足：

- 参与者级原始 CGM，而非只给汇总指标；
- 同期空腹胰岛素、OGTT 胰岛素/C-peptide、SSPG 或 clamp 等独立代谢参照；
- 至少约 5–7 个可评估夜晚，以支持现有 `Nq`/前缀稳定性方案；
- CGM 设备、时区、采样频率、缺失、用药和糖尿病治疗状态；
- 与 Hall、CGMacros 等现有队列没有参与者重叠；
- 清楚的许可、访问流程和可审计的数据字典。

### 2.2 角色隔离

候选数据不能全部用于选特征或改公式。建议预先冻结三类角色：

| 角色 | 用途 | 允许的操作 | 禁止的操作 |
|---|---|---|---|
| 机制核验集 | 检查 CGM 特征是否确实对应 SSPG/clamp/胰岛素分泌 | 预定义相关、方向、一致性分析 | 反复选特征后再宣称独立验证 |
| 鲁棒性/迁移集 | 检查设备、族群、时长和治疗状态的迁移 | 设备分层、`Nq`、缺失与时间窗分析 | 用临床结局反向调阈值 |
| 最终盲测集 | 冻结公式、阈值和排除规则后的确认 | 一次性执行预注册协议 | 在揭盲后继续调参并沿用同一结果 |

## 3. The Homomorphic Anchor Forge：本轮元数据清算

> ⚓ **The Homomorphic Anchor Forge v1.0（残差与熵增清算表）**
>
> **I. 拓扑骨架提取（Topological Skeleton）**：本研究所需的不变量不是单点血糖高低，而是同一参与者的密集 CGM 自然态动力学，与独立的胰岛素敏感性、胰岛素分泌/处置能力及治疗背景共同观测。
>
> **II. 同态损耗预判（Homomorphic Loss）**：若数据只有 CGM、HbA1c 或人群类别，而没有同期胰岛素/OGTT/clamp、设备和治疗信息，CGM 只是多种生理过程的混合投影；扩大样本量不能恢复这些已经丢失的机制信息。
>
> **III. 物理残差锁定（Residual Identification）**：本轮只检索元数据，尚未读取参与者级分析数据，因而不能诚实声称锁定了生物物理残差。当前唯一可锁定的是“数据资格残差”：原始 CGM 与同期空腹胰岛素、OGTT 胰岛素/C-peptide、SSPG 或 clamp 结果能否由稳定 ID 连接，并保留治疗/设备上下文。
>
> **IV. 非对称咬合（Asymmetric Deadlock）**：**本轮未达成刚性咬合。**候选只通过了文献与数据门户层面的字段/访问初筛；在参与者级连接、缺失、时间对齐和独立性审计完成之前，不输出诊断性结论，也不把任何队列结果视为部署证据。

这项结论是刻意保守的：强制协议不能把“找到了可能合适的数据集”升级成“已经发现了生理残差”。

## 4. 候选数据集矩阵

分级只表示对当前研究的预期信息增益：A 为直接机制标签，B 为临床/外部验证价值高，C 为辅助或边界验证，D 为当前不适配。

| 优先级 | 数据集 | 可用规模与 CGM | 可用参照 | 获取状态 | 最适合角色 | 主要阻断点 |
|---|---|---|---|---|---|---|
| A1 | Kobe CGM_AC | 64 人；iPro CGM 平均约 5.5 天，论文主要分析 72 小时 | 75 g OGTT 的葡萄糖/胰岛素多时间点；连续高血糖 clamp 与高胰岛素-正常血糖 clamp；分泌、敏感性和处置指数 | CGM GitHub + Zenodo 源数据公开；其余数据可能需作者申请 | 机制核验 | 公开 CGM 与 clamp/OGTT 个体 ID 能否直接连接；夜晚数可能不足 5–7 |
| A2 | Stanford Metabolic Subphenotype | 居家 CGM 子队列 29 人；Dexcom G6 Pro，约 10 天，2 次标准化居家 OGTT | SSPG、16 点 OGTT、胰岛素/C-peptide、处置指数、肝 IR、肠促胰素效应 | 论文数据包与代码公开 | 机制核验 | 样本小；须审计与 Hall 的参与者/方案重叠；论文模型不可直接移植为白盒结论 |
| B1 | AI-READI v3 | 2,280 人；Dexcom G6，约 10 天；另有 100 人 mini 版 | 胰岛素、C-peptide、葡萄糖、HbA1c、血脂等临床表；可能构造 HOMA 类参照 | FAIRhub 申请/许可，字段可能分公开、受限和受控 | 大样本外部验证；可保留作最终盲测 | 必须确认采血空腹状态、同访视时间、用药字段和可导出粒度；无 clamp/SSPG |
| B2 | Human Phenotype Project / 10K | 相关公开研究纳入约 8,025 名无糖尿病者；FreeStyle Libre Pro 超过 7 天 | 深度临床/多组学；项目内存在空腹血糖、胰岛素/HOMA 研究线索 | 面向大学/研究机构申请，安全环境使用 | 大规模、跨设备、跨族群最终确认 | 需在正式数据字典中确认同一访视胰岛素/HOMA 与 CGM 的连接及可访问性 |
| B3 | PREDICT 1 / TwinsUK | 原始研究约 1,002 人；后续汇总分析约 863 人；居家 CGM 与标准餐；377 人双监测器子集 | 空腹及餐后胰岛素、C-peptide、甘油三酯等 | TwinsUK/DTR 研究申请 | 膳食扰动动力学、设备重复性与迁移 | 不是直接 clamp/SSPG；须确认原始时间序列和个体级生化字段获批范围 |
| B4 | Framingham Heart Study Exam 4 | 1,175 人有至少 7 个完整 CGM 日并有 HbA1c/FPG；Dexcom G6 Pro | HbA1c、空腹血糖；Exam 4 有混合餐试验及抽血设计 | FHS/BioLINCC/dbGaP 或辅助研究申请 | 长期结局、北美独立外部确认 | 同期胰岛素/C-peptide 是否可申请尚未证实；历史胰岛素不能替代同期标签 |
| B5 | Singapore Asian non-diabetes cohort | 151 人可分析；Libre Pro，CGM 中位约 13.9 天 | 75 g OGTT、FPG、2 h glucose、HbA1c；124 正常、27 糖尿病前期 | 论文公开；参与者数据未见公共下载入口，需联系作者 | 亚洲无糖尿病人群、长记录迁移 | 没有明确空腹胰岛素或直接 IR 标签；论文阈值不能替代 GlucoBench 终点 |
| C1 | ShanghaiT2DM | 100 名 T2D；3–14 天、15 分钟 Libre；109 个记录期 | 空腹/餐后葡萄糖、胰岛素、C-peptide、HbA1c、用药和饮食 | Figshare 公开 | 治疗分层、中文字段解析、真实世界缺失/多访视 | 已确诊且接受多种治疗，不代表“隐性、未治疗基线异常”；化验可在 CGM 前后 6 个月 |
| C2 | T2Help | 实际入组 306；计划多次 10 天 Dexcom G6、最长约 16 周 | 糖尿病前期/T2D、用药、饮食、活动；小型 5 h OGTT/C-peptide 子集 | 临床试验登记可见，未找到公共数据发布 | 长记录 `Nq` 与重复测量 | 数据状态/发布渠道未知；不能作为当前可用资源计入 |
| C3 | Healthy Dexcom G6 reference cohort | 153 名无糖尿病者；最长约 10 天 | HbA1c、抗体与健康筛选 | 论文开放；Dryad 主要为补充材料，原始 CGM 获取未确认 | 健康参照与设备背景 | 无胰岛素抵抗标签；原始数据入口不稳定/未证实 |
| C4 | BIG IDEAs Glycemic Variability | 16 人；Dexcom G6 约 8–10 天，含食物/可穿戴 | HbA1c 和人口学 | PhysioNet 开放 | 导入管线与多模态时间对齐 | 样本极小；OGTT 文件存在质量问题；无胰岛素标签 |

## 5. 第一优先候选的证据细节

### 5.1 Kobe：最接近直接生理验证，但先做连接审计

2025 年 Communications Medicine 研究纳入 64 名既往无糖尿病诊断的参与者。受试者完成 OGTT、CGM，并在连续两天接受高血糖 clamp 和高胰岛素-正常血糖 clamp；OGTT 包含 0、30、60、90、120 分钟的葡萄糖和胰岛素测量。论文给出胰岛素分泌、敏感性及处置指数等参照。CGM 文件公开在 [CGM_AC GitHub](https://github.com/HikaruSugimoto/CGM_AC)，论文源数据位于 [Zenodo record 15067145](https://zenodo.org/records/15067145)，原始论文见 [Communications Medicine](https://www.nature.com/articles/s43856-025-00819-5)。

它的价值在于同时覆盖“CGM 动力学”和“独立生理参照”，但不能直接假定 Zenodo 源表已包含可与每条 CGM 连接的个体 clamp 标签。下载前应先审计文件清单、ID、时间基准和许可；若只有汇总源数据，就需要向作者申请参与者级去标识化生理终点。

### 5.2 Stanford：标签最强，但样本和独立性是限制

[Nature Biomedical Engineering 原始论文](https://www.nature.com/articles/s41551-024-01311-6)描述 32 人初始队列、24 人独立验证队列和 29 人居家 CGM 子队列；29 人由初始队列 5 人和验证队列 24 人组成。参照包括 SSPG、16 点 OGTT、7 个时间点 C-peptide、胰岛素分泌率/处置指数、肝 IR 及肠促胰素效应。论文公开了[数据包](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/metabolic_subphenotype_db/metabolic_subphenotypes_db.zip)和[代码仓库](https://github.com/aametwally/Metabolic_Subphenotype_Predictor)。

该队列适合回答“某个冻结 CGM 特征是否对应特定代谢表型”，不适合单独证明普适筛查能力。由于研究地点同为 Stanford，还必须用受试者 ID、IRB/招募期和基线特征审计其与 Hall 队列是否重叠；在审计完成前只能称为“可能独立”。

## 6. 最值得申请的外部确认集

### 6.1 AI-READI v3

[FAIRhub v3 门户](https://fairhub.io/datasets/3)显示 2025-11-17 发布的 v3 含 2,280 名参与者，并提供 100 人 mini 版本。[CGM 文档](https://docs.aireadi.org/docs/3/dataset/wearable-glucose-monitor/)说明 Dexcom G6 以约 5 分钟间隔记录约 10 天；[临床实验室文档](https://docs.aireadi.org/docs/3/dataset/clinical-data/OMOP-Clinical-Data-Structure/)列出胰岛素、C-peptide、葡萄糖、HbA1c、hs-CRP 和血脂等字段。

这里最大的风险不是样本量，而是标签语义：HOMA 类指标只有在采血空腹、单位可靠、同访视且未受治疗严重混杂时才有意义。应先用 mini 版验证文件结构，不应先浏览全队列结局后再决定公式。若计划把 AI-READI 留作最终盲测，应在申请前冻结分析协议。

### 6.2 Human Phenotype Project / 10K

[项目官网](https://humanphenotypeproject.org/home)说明数据面向大学与研究机构的人类健康研究开放申请，并提供联系入口；[数据目录](https://data-browser.humanphenotypeproject.org/)明确包含 CGM 模态。相关 2026 年研究报告 10K/HPP 中约 8,025 名无糖尿病参与者拥有超过 7 天的 Libre Pro 数据，见 [Communications Medicine 论文](https://www.nature.com/articles/s43856-026-01523-8)。

其优势是规模、记录长度和不同设备，适合作为最终跨族群/跨设备确认。但“项目中存在胰岛素或 HOMA 研究”不等于申请一定可获得与 CGM 同访视的个体标签；该字段必须写进正式询价清单。

### 6.3 PREDICT 1 / TwinsUK

[PREDICT 1 原始论文](https://www.nature.com/articles/s41591-020-0934-0)描述大规模标准餐、居家连续监测与多组学表型，并包含空腹/餐后胰岛素、C-peptide 等测量。TwinsUK 的[数据访问政策](https://twinsuk.ac.uk/wp-content/uploads/2025/08/DTR_DataAccessPolicy_August2025-1.pdf)列出 PREDICT 研究的申请路径。

该队列尤其适合剥离进食干预、检查餐后弛豫，以及利用双监测器子集估计设备/重复性误差。它不是直接 clamp 队列，因此更适合作为迁移与扰动响应验证，而不是单一“胰岛素抵抗真值”。

### 6.4 Framingham Exam 4

[JCEM 队列论文](https://academic.oup.com/jcem/article/110/4/1128/7754867)报告 1,175 人具有至少 7 个完整 Dexcom G6 Pro 日，同时有 HbA1c 和空腹血糖；[FHS 数据总览](https://www.framinghamheartstudy.org/fhs-for-researchers/data-available-overview/)给出研究申请和数据获取路径。Exam 4 还包含混合餐挑战设计。

其最大价值是完全不同的北美社区队列和长期结局历史。除非数据字典确认混合餐/空腹胰岛素或 C-peptide 与 CGM 同期存在，否则它只能承担血糖学/长期风险确认，不能替代直接胰岛素抵抗验证。

## 7. 辅助数据与明确排除项

### 7.1 有用但不能作为主确认集

- **Singapore 亚洲无糖尿病队列**：151 人、Libre Pro 中位约 13.9 天，75 g OGTT + HbA1c，适合亚洲族群和长记录迁移；但论文未显示胰岛素标签或公共原始数据入口。来源：[PMC 原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12540373/)。
- **ShanghaiT2DM**：公开、字段丰富并保留治疗信息，适合验证“治疗状态必须显式分层”和真实世界数据解析；但实验室可在 CGM 前后 6 个月，且样本是已确诊/治疗 T2D。来源：[Scientific Data](https://www.nature.com/articles/s41597-023-01940-7)、[Figshare 数据集](https://figshare.com/articles/dataset/diabetes_datasets_zip/21600933)。
- **T2Help**：重复 CGM 设计很适合 `Nq` 研究，但尚无可确认的公共发布。来源：[ClinicalTrials.gov NCT04503239](https://clinicaltrials.gov/study/NCT04503239)。
- **BIG IDEAs Glycemic Variability**：开放且带多模态日志，可用于导入和时间对齐回归测试；生理标签不足。来源：[PhysioNet v1.1.3](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/)。

### 7.2 当前排除

- **WEAR-ME**：有空腹血糖、空腹胰岛素/HOMA 与可穿戴设备，但没有 CGM，不能直接验证 CGM 公式。来源：[Nature 论文](https://www.nature.com/articles/s41586-026-10179-2)。
- **RISE**：具备高质量 OGTT/clamp 代谢标签，但研究未使用 CGM；可参考标签设计，不能作为当前 CGM 验证集。来源：[NIDDK Central Repository](https://repository.niddk.nih.gov/study/268)。
- **T1D-only forecasting/control datasets**：治疗、外源胰岛素和任务定义与“隐性基线异常早筛”不一致，除非仅用于解析器或治疗分层的压力测试。
- **Hall**：已参与当前探索并揭盲，不能重新包装成新的最终独立临床确认集。
- **CGMacros**：已在仓库现有研究中使用，不属于新增外部证据。

## 8. 推荐的访问顺序与数据治理

### 第 0 阶段：只做结构核验，不看结局分布

1. 读取 Kobe GitHub/Zenodo 与 Stanford 数据包的文件清单、许可和数据字典。
2. 只检查：参与者 ID 是否稳定、CGM 原始时间戳/单位/设备/时区、标签表是否可连接、记录天数和缺失字段。
3. 对 Stanford 与现有 Hall 做参与者/IRB/招募期重叠审计；不先运行分数与标签相关。

### 第 1 阶段：冻结机制核验协议

在任何结局揭盲前写清：

- 主终点优先级：clamp/SSPG > OGTT 胰岛素/C-peptide/处置指数 > 经确认的空腹 HOMA > HbA1c/FPG；
- 预定义 CGM 时间窗、`Nq` 资格、缺失规则、设备分层和治疗分层；
- 一名参与者一票，重复记录不得伪装成独立样本；
- 报告效应方向、置信区间/置换检验、灵敏度分析和失败结果；
- 机制核验集不得再充当最终盲测集。

### 第 2 阶段：同步提交三类申请

1. **AI-READI**：先申请/读取 mini 数据字典，重点询问空腹状态、单位、访视日期、药物和 CGM-person ID 映射。
2. **HPP/10K**：询问同期空腹胰岛素、FPG、HbA1c、HOMA、用药、Libre 原始时间序列和安全环境导出限制。
3. **PREDICT 1 或 FHS Exam 4**：二选一作为跨设备/族群盲测候选；根据个体级胰岛素字段是否可得再决定优先级。

### 第 3 阶段：保留真正未见的确认集

如果 Kobe/Stanford 被用于筛选特征、选择公式或确定方向，它们只能算开发/机制核验。至少保留 AI-READI、HPP/10K、PREDICT 1 或 FHS 中的一个，在公式、阈值、排除条件和失败标准全部冻结后再一次性揭盲。

## 9. 访问前的最小核对清单

每个候选必须回答“是/否/未知”：

- 原始 CGM 是否参与者级、带时间戳而非只有 AGP/汇总？
- CGM 与临床表是否共享稳定去标识化 ID？
- 空腹胰岛素是否真的空腹，单位和检测平台是否明确？
- OGTT/clamp/SSPG 是否与 CGM 同一访视或足够接近？
- 是否有糖尿病诊断、降糖药、胰岛素、GLP-1/SGLT2 等治疗上下文？
- 是否能计算当前夜间资格标准及 `Nq`？
- 设备型号、采样间隔、时区/夏令时、校准和缺失原因是否可得？
- 是否可能与 Hall、CGMacros 或其他 Stanford 队列重叠？
- 许可是否允许派生指标、结果发表与可复现实验报告？
- 该队列被指定为开发、迁移还是最终盲测，是否在揭盲前冻结？

## 10. 最终建议

**最优组合不是单一更大的数据集，而是“强标签小队列 + 长记录大队列 + 真正未见确认集”的证据链：**

1. 先用 **Kobe + Stanford** 验证预定义 CGM 动力学特征是否对应直接胰岛素敏感性/分泌表型；
2. 用 **ShanghaiT2DM / PREDICT 1** 检查治疗、进食扰动和设备迁移边界；
3. 将 **AI-READI 或 HPP/10K** 中至少一个保留为冻结后的大规模确认集；
4. 用 **Framingham** 补充北美社区样本和潜在长期结局，但不在缺少同期胰岛素时夸大为直接 IR 验证。

下一步在获得单独批准后，应只做 Kobe 与 Stanford 的**文件清单/字段连接审计**并形成第二份报告；通过该门槛后，才提出参与者级分析方案。任何候选表现良好也不自动触发 `index.html`、阈值或产品语义更新。
