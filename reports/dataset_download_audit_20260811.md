# GlucoBench 外部数据集下载审计（2026-08-11）

## 1. 执行结论

本轮已下载所有**无需研究者身份、机构审批或数据使用协议即可直接取得**的候选数据，并将完成文件放在 `output/external_datasets/raw/` 下。完成数据均设为只读，未解压、未读取参与者数值、未运行特征或模型。

“所有候选数据集”无法在本轮全部落盘，原因包括：

- AI-READI 完整版约 3.82 TB，超过本机约 395 GB 的可用空间；179.68 GB mini 版也要求机构认证、研究伦理信息、许可确认和 Azure 存储容器。
- HPP/10K、PREDICT 1、Framingham 属于受控研究数据，需要以真实研究者/机构身份申请。
- JAEB 下载表要求姓名、邮箱、机构、用途及条款确认；本轮没有虚构或代填这些信息。
- Singapore 和 T2Help 未找到公开参与者级下载入口。
- PhysioNet BIG IDEAs 完整 ZIP 约 4.7 GiB，但官方动态 ZIP 单连接速度对应十余小时；已下载当前研究所需的全部 v1.1.3 Dexcom、食物日志、人口学、许可与校验文件，并将不完整全模态 ZIP 明确隔离。

因此，本轮状态是：**5 个公开来源已取得其可用数据/文件，其中 4 个取得完整公共包，PhysioNet 取得完整 CGM 研究子集；其余为受控或尚未发布，不能声称已下载。**

## 2. 保存位置与不可变性

- 根目录：`output/external_datasets/`
- 完成数据：`output/external_datasets/raw/`
- 未完成 PhysioNet 全模态包：`output/external_datasets/raw/big_ideas_physionet/incomplete_full_archive/`
- 完成文件只读数：47
- Git 状态：`output/` 已由仓库 `.gitignore` 排除；没有修改 `raw_data.zip`、`index.html`、模型、阈值或现有原始数据。

## 3. 已下载与校验结果

### 3.1 Kobe CGM_AC：完整公共资源

目录：`output/external_datasets/raw/kobe_cgm_ac/`

已下载：

- Zenodo `SourceData.xlsx`：41,496 bytes
- Zenodo `CGM_data.csv`：201,148 bytes
- Zenodo `requirements.txt`：2,881 bytes
- Zenodo `CGM_data.png`：492,389 bytes
- Zenodo `main.py`：2,227 bytes
- GitHub `CGM_AC-main.zip`：527,183 bytes，ZIP 中央目录可读，6 个条目

Zenodo 五个文件的 MD5 均与官方 API 一致：

| 文件 | MD5 |
|---|---|
| `SourceData.xlsx` | `CF70B3AEB111CECA39940F15E9810C28` |
| `CGM_data.csv` | `7153CACD34A826E7FEE086DB1F771E46` |
| `requirements.txt` | `F869C4061C99E50CDE2DBB9BF24E1ECC` |
| `CGM_data.png` | `629432D8CDD9047118E94ADE6441A267` |
| `main.py` | `3EA2E2E1056D7C338347AB4ABCA889A1` |

来源：[Zenodo record 15067145](https://zenodo.org/records/15067145)、[GitHub CGM_AC](https://github.com/HikaruSugimoto/CGM_AC)。

### 3.2 Stanford metabolic subphenotype：完整公共数据包

文件：`output/external_datasets/raw/stanford_metabolic_subphenotype/metabolic_subphenotypes_db.zip`

- 大小：27,295 bytes
- SHA-256：`CAD0570F918271851EF96004538F27FF3DA3CA77BAE04882A2F1B43B5FD42432`
- ZIP 中央目录：可读，10 个条目
- 可见内容包括 initial cohort metabolic phenotypes 等 CSV；本轮只检查归档结构，没有打开参与者值。

来源：[Nature Biomedical Engineering 论文](https://www.nature.com/articles/s41551-024-01311-6)、[论文公开数据包](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/metabolic_subphenotype_db/metabolic_subphenotypes_db.zip)。

### 3.3 ShanghaiT1DM/ShanghaiT2DM：完整公共数据包

文件：`output/external_datasets/raw/shanghai_t2dm/diabetes_datasets.zip`

- 大小：3,738,354 bytes
- SHA-256：`59B5F5C4053A32BB6B7827844A0191597DC82228FE88CE332A189FDFC659C4CB`
- ZIP 中央目录：可读，264 个条目
- 包含 Shanghai_T1DM、Shanghai_T2DM、汇总表及分析 notebook。

来源：[Scientific Data 数据说明](https://www.nature.com/articles/s41597-023-01940-7)、[Figshare 数据页](https://figshare.com/articles/dataset/diabetes_datasets_zip/21600933)。

### 3.4 Dryad healthy-reference：完整下载其公开文件，但不是原始 CGM

目录：`output/external_datasets/raw/healthy_reference_dryad/`

Dryad 数据页对两个版本各暴露一个 `Online Supplemental Material.docx`：

- 每个文件 303,074 bytes；
- 两者 SHA-256 相同：`4729DB7A9AD6BF9C678C2BBACDF7509B3321A5F98EFBB64FB094BAEC1AB26DAA`；
- DOCX 归档结构可读，各 17 个条目。

这意味着 Dryad 上的“全部公开文件”已下载，但它们是重复的论文补充文档，**不是 153 人原始 CGM**。原始数据若通过 JAEB 表单获取，需要用户提供真实身份、机构、用途并接受条款。

来源：[Dryad DOI 页面](https://datadryad.org/dataset/doi%3A10.5061%2Fdryad.h7d11cd)、[JAEB 公共数据门户](https://public.jaeb.org/datasets/diabetes)。

### 3.5 PhysioNet BIG IDEAs v1.1.3：完整 CGM 研究子集

完成目录：`output/external_datasets/raw/big_ideas_physionet/research_relevant_v1.1.3/`

已下载并验证：

- 16 个参与者的 `Dexcom_*.csv`；
- 16 个参与者的 `Food_Log_*.csv`；
- `Demographics.csv`；
- `LICENSE.txt`；
- 官方 `SHA256SUMS.txt`。

32 个参与者文件合计 2,406,534 bytes，全部逐文件通过官方 SHA-256。OGTT 文件在该数据集本身即因损坏而未提供，并非本轮遗漏。

完整 v1.1.3 压缩包官方大小为 5,015,250,233 bytes，解压约 34.1 GB，绝大部分是高频 Empatica E4 的 ACC/BVP/EDA/HR/IBI/TEMP。慢速全包下载在约 159.8 MB 时停止；ZIP 和范围片段均被放入 `incomplete_full_archive/`，不得作为有效数据使用。

来源：[PhysioNet v1.1.3](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/)。

## 4. 未下载候选与阻断条件

| 数据集 | 当前状态 | 无法直接下载的原因 | 继续所需动作 |
|---|---|---|---|
| AI-READI v3 | 未下载 | 3.82 TB、356,343 文件；需九步访问流程 | 机构 CILogon、伦理/培训信息、研究用途、许可、Azure 容器；建议只申请 CGM+临床表或 mini |
| AI-READI mini | 未下载 | 179.68 GB；仍需同类认证与 Azure | 同上，且需确认是否值得占用近半可用磁盘 |
| HPP / 10K | 未下载 | 受控研究数据 | 以大学/研究机构身份联系 HPP，申请 CGM + 同访视胰岛素/HOMA/用药字段 |
| PREDICT 1 / TwinsUK | 未下载 | DTR 数据申请与使用政策 | 提交研究方案、研究者/机构信息及所需变量清单 |
| Framingham Exam 4 | 未下载 | FHS/BioLINCC/dbGaP 受控访问 | 确认 Exam 4 CGM、FPG/HbA1c、MMTT/胰岛素字段后申请 |
| Singapore Asian cohort | 未下载 | 论文未给公共参与者级下载 | 联系通讯作者申请去标识化 CGM + OGTT/HbA1c |
| T2Help | 未下载 | ClinicalTrials.gov 无结果数据发布 | 联系 Dexcom/研究方，确认数据共享状态 |
| JAEB healthy raw CGM | 未下载 | 表单要求个人信息和条款确认 | 用户本人提供/填写姓名、邮箱、机构、用途并确认条款 |

访问入口：[AI-READI](https://fairhub.io/datasets/3/access)、[HPP](https://humanphenotypeproject.org/home)、[PREDICT/TwinsUK 政策](https://twinsuk.ac.uk/wp-content/uploads/2025/08/DTR_DataAccessPolicy_August2025-1.pdf)、[FHS 数据总览](https://www.framinghamheartstudy.org/fhs-for-researchers/data-available-overview/)、[T2Help](https://clinicaltrials.gov/study/NCT04503239)。

## 5. 数据边界与下一步

本轮只完成传输、文件级结构核验和哈希校验，没有读取参与者数据，因此没有进入 Aether 参与者级残差分析，也没有产生健康/诊断结论。

继续获取受控数据前，需要用户明确授权使用真实研究者/机构身份提交申请；不应在聊天中提供密码或验证码。建议下一步优先：

1. 为 AI-READI 只申请 CGM、临床实验室、用药和参与者映射字段，避免拉取 3.82 TB 全模态资源；
2. 准备一份可复用的 HPP/PREDICT/FHS 数据申请摘要和字段清单；
3. 由用户本人完成 JAEB 表单的身份与条款部分；
4. 获得新数据后先做 ID、时间、单位、空腹状态、治疗和队列重叠审计，仍不立即拟合模型。
