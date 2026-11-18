# WattBot 2025 RAG 实验记录

实验于 2026 年 11 月 18 日运行。训练集 41 题用于本地诊断，测试集 282 题用于生成提交文件。最终提交配置为 **BM25 多查询检索 + DeepSeek Flash 1017**。训练集本地加权分为 **0.7604**，答案值准确率为 **73.17%**。

## 方法与配置

| 环节 | 实验配置 |
| --- | --- |
| 语料 | 仅使用 `metadata.csv` 列出的 32 篇 PDF，共 641 页。用 PyMuPDF 抽取正文块和检测到的表格行，保留文档 ID 与页码。共索引 16,372 个文本块和 835 个表格行。 |
| 切分与上下文 | 超过 1,300 字符的文本块按词边界切分。命中后用父页中邻近的正文扩展上下文。表格行保留表头与行内容。 |
| 查询规划 | 原问题加最多 3 条 DeepSeek 改写查询，去除重复项。 |
| 检索与融合 | BM25 使用 `k1=1.2`、`b=0.75`。每条查询在每种检索方式中最多取当轮目标数量的 3 倍，以倒数排名融合 `1/(60 + rank)`，再去除重复的页内片段。独立对照的混合检索额外使用本地 `BAAI/bge-small-en-v1.5`，与 BM25 一起参与融合。 |
| 检索预算 | 首轮最多取 10 条，每条父页上下文上限 1,700 字符。证据不足、输出无效或复核未通过时重试，最多共 3 轮。第二、三轮最多取 20 条，每条父页上下文上限 2,400 字符。 |
| 生成模型 | HKUST(GZ) AIGC 的 `DeepSeek-V4.1-Flash`，按组内确认对应 Flash 1017。查询规划、结构化答题与证据复核均用该模型。请求参数为 `temperature=0`、`max_tokens=2400`、JSON 对象响应，并请求关闭 thinking。校园服务回传模型别名与指纹，版本映射来自组内确认。 |
| 答案验证 | 校验引用片段 ID 与引文，将引用 ID 映射到元数据 URL。用同一模型单独复核原文是否支持答案、单位与计算。用独立脚本回放 trace，核对原文引用、CSV 行及产物哈希。 |
| 导出规范化 | 判断题用 `1` / `0`。当题目未要求缩放单位时，按单位提示将如 `billion parameters` 转为基础单位。明确缺少数值的结果转为标准拒答。拒答的 `answer_value`、`answer_unit`、`ref_id`、`ref_url` 和 `supporting_materials` 均为 `is_blank`。 |
| 运行环境 | Python 3.13.13，PyMuPDF 2.1，FastEmbed 0.9.1。端到端运行的 worker 数为训练集 8、测试集 16。 |

## 结果

### 仅检索对照

对训练集中的 **39 道有参考文档的题**，直接用原问题检索。对每题计算前 10 个片段所覆盖的参考文档比例，再取题均值。该对照没有使用 LLM 改写或生成答案。

| 检索 | 文档 Recall@10 |
| --- | ---: |
| BM25 | 94.87% |
| BM25 + BGE | 92.31% |

最终全量运行使用 BM25，并启用多查询规划。混合检索路径另通过 3 题端到端试跑和 trace 回放。

### 训练集端到端诊断

| 指标 | 结果 |
| --- | ---: |
| 本地加权分 | **0.7604 / 1** |
| 答案值准确率 | **73.17%（30/41）** |
| 参考文档 Jaccard（题均） | **0.8252** |
| 可回答性判断准确率 | **87.80%（36/41）** |
| 不可回答题正确拒答 | **2/2** |
| 可回答题误拒答 | **5/39** |

本地分按答案值 75%、引用集合 Jaccard 15%、可回答性判断 10% 加权。数值题采用 ±0.1% 相对误差，类别值规范化后比较。使用原始导出时，判断题的 `True` / `False` 与训练标签的 `1` / `0` 不一致。修正 12 行后，答案值准确率从 43.90%（18/41）变为 73.17%（30/41），加权分从 0.5409 变为 0.7604。上述指标来自有标签的训练集，应作为本地诊断解读。

### 测试集提交产物

最终 CSV 通过 282 行覆盖、列顺序、文档 ID 与 URL、拒答字段校验。最终 **236 题作答，46 题拒答**。原始 trace 包含 282 次查询规划、413 次候选答题和 266 次证据复核。导出记录包含 42 行变更，其中判断题编码 36 行、单位换算 5 行、标准拒答转换 1 行。测试集无可用于本地评分的参考答案，本次实验没有 Kaggle 分数。

| 本地产物 | 用途 |
| --- | --- |
| `outputs/rag-train-metrics.json` | 训练集汇总与逐题评分 |
| `outputs/rag-train-bm25.jsonl`、`outputs/rag-train-hybrid.jsonl` | 两种检索方式的排序记录 |
| `outputs/rag-train-final.jsonl`、`outputs/rag-test-final.jsonl` | 原始端到端 trace，包含查询、检索、生成、复核和导出行 |
| `outputs/submission_rag_verified.csv` | 最终规范化后的提交文件 |
| `outputs/submission_rag_verified.normalization.json` | 逐字段变更和输入、输出哈希 |

最终 CSV SHA-256 为 `05e9ce8376a6f4dcd600b8e6c5c2bf9efa82770cc708026e8ac446abc3f76f47`。语料索引 SHA-256 为 `819c9d90668b99421718d49c8dc0d81a906451177f68c1e703d6ae49591a83bd`。表中的产物路径为本地路径，CSV、数据和含原文片段的 trace 按项目约定保存在 Git 忽略的目录。

## 回放与复跑

使用现有产物复核最终文件和训练分数：

```bash
.venv/bin/python -m scripts.verify_rag_trace \
  --trace outputs/rag-test-final.jsonl --output outputs/submission_rag.csv \
  --canonical-output outputs/submission_rag_verified.csv
.venv/bin/python -m wattbot verify --output outputs/submission_rag_verified.csv
.venv/bin/python -m wattbot score --output outputs/rag-train-normalized.csv
```

使用当前代码从 PDF 复跑：

```bash
.venv/bin/python -m wattbot index
.venv/bin/python -m wattbot run --questions data/raw/train_QA.csv \
  --backend hkustgz --top-k 10 --workers 8 \
  --output outputs/rag-train-repro.csv --trace outputs/rag-train-repro.jsonl
.venv/bin/python -m wattbot score --output outputs/rag-train-repro.csv
.venv/bin/python -m wattbot run --backend hkustgz --top-k 10 --workers 16 \
  --output outputs/rag-test-repro.csv --trace outputs/rag-test-repro.jsonl
.venv/bin/python -m scripts.verify_rag_trace \
  --trace outputs/rag-test-repro.jsonl --output outputs/rag-test-repro.csv
```

当前代码在候选复核前执行规范化，本次全量实验的原始 trace 则保留当时的候选与复核记录，最终 CSV 由确定性导出生成。复跑命令用于复现实验流程，历史 CSV 的逐行复现使用原始 trace 和规范化记录。
