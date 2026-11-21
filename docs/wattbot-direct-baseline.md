# WattBot Direct LLM Baseline

这个实验衡量 DeepSeek Flash 在只看到题目和单位提示时的表现。与 [RAG 实验](wattbot-rag-experiment-2026-11-18.md)使用相同的校园模型、题目文件、单位提示、输出规范化和本地评分函数。

## 实验设置

| 项目 | 配置 |
| --- | --- |
| 模型 | HKUST(GZ) AIGC 的 `DeepSeek-V4.1-Flash`，按组内确认对应 Flash 1017。 |
| 模型输入 | 固定 prompt、题目原文、题目文件自带的单位提示。 |
| 答题 | 一次直接生成答案、数值、单位和简短解释，或明确拒答。流程没有检索、工具调用和模型复核。 |
| 请求参数 | `temperature=0`、`max_tokens=2400`、JSON 对象响应，请求关闭 thinking。 |
| 异常处理 | 格式无效时最多重试一次。数值仅出现在 `answer` 且是 JSON 数值或布尔值时，可直接恢复 `answer_value`。两次仍无法解析的结果记为拒答，trace 保留失败原因。 |
| 导出 | 判断题及单位按与 RAG 相同的规则规范化。所有引用字段均为 `is_blank`。 |
| 并发 | 训练集 8 个 worker，测试集 16 个 worker。 |

## 运行与校验

```bash
.venv/bin/python -m wattbot.baselines.direct --split train --workers 8
.venv/bin/python -m wattbot.baselines.direct --split test --workers 16
.venv/bin/python -m scripts.verify_direct_baseline outputs/baseline-direct/train
.venv/bin/python -m scripts.verify_direct_baseline outputs/baseline-direct/test
```

每次运行在 `outputs/baseline-direct/<split>/` 保存：

| 文件 | 内容 |
| --- | --- |
| `config.json` | Prompt、模型、参数、输入文件和代码哈希 |
| `trace.jsonl` | 逐题请求、模型响应、格式恢复、作答或拒答决定和最终行 |
| `predictions.csv` | 按输入题号顺序导出的预测 |
| `summary.json` | 题数、作答和拒答数、产物哈希 |
| `metrics.json` | 训练集汇总和逐题诊断分 |

校验脚本检查每次模型请求是否只发了题目和单位提示，核对预测、trace、哈希，并重算训练分数。本地运行产物位于 Git 忽略目录。

## 训练集结果

同一批 41 道有标签问题，使用相同的评分函数。

| 指标 | Direct LLM | RAG |
| --- | ---: | ---: |
| 答案值准确率，含不可回答题 | 36.59%（15/41） | 73.17%（30/41） |
| 答案值准确率，仅可回答题 | 35.90%（14/39） | 71.79%（28/39） |
| 本地加权诊断分 | 0.3354 | 0.7604 |
| 参考文档 Jaccard（题均） | 0.0488 | 0.8252 |
| 可回答性判断准确率 | 53.66%（22/41） | 87.80%（36/41） |
| 不可回答题正确拒答 | 1/2 | 2/2 |
| 可回答题误拒答 | 18/39 | 5/39 |

逐题对比中，RAG 独自答对 16 题，Direct LLM 独自答对 1 题，两者都答对 14 题。本地评分给双方都无引用的题计 1 分，因此 Direct LLM 的引用得分来自两道参考引用为空的题。加权分同时反映检索、证据复核和引用等流程差异，比较直接答题能力可优先看答案值准确率。

## 测试集结果

282 道测试题均已生成预测，其中 134 题作答、148 题拒答。本次 Direct LLM 与 RAG 两组的测试产物分别作答 134 题和 236 题。2026 年 11 月 20 日提交至 Kaggle 赛后通道（编号 `58605665`），Direct LLM 返回公开榜 **0.286**、私有榜 **0.419**（对照：原始 v1 RAG 为 `0.814 / 0.750`，全 `is_blank` 冒烟基线为 `0.138 / 0.175`，完整汇总见 [Kaggle 测试集结果汇总](kaggle-test-results.md)）。

两组 baseline trace 均已通过校验。最终训练集发出 41 次请求，测试集发出 282 次请求，每题各一次。测试集预测 SHA-256 为 `8adebc524b9a483dba5a8de873f00ee28a9612ac3f0bd50623d5cb7ec44b0a07`。

## Token 用量对比

以下使用 HKUST(GZ) API 在响应中返回的 `usage`，覆盖四次最终运行的全部响应。RAG 统计查询规划、候选答题和证据复核，包含重试。每题均值以最终导出的题数为分母。

| 数据集 | 方法 | 请求数 | 输入 token | 输出 token | 总 token | 每题平均总 token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 训练（41 题） | Direct LLM | 41 | 6,889 | 3,394 | 10,283 | 250.80 |
| 训练（41 题） | RAG | 153 | 483,014 | 18,278 | 501,292 | 12,226.63 |
| 测试（282 题） | Direct LLM | 282 | 47,466 | 22,012 | 69,478 | 246.38 |
| 测试（282 题） | RAG | 961 | 2,993,919 | 108,443 | 3,102,362 | 11,001.28 |

RAG 在训练集消耗了 Direct LLM 约 48.7 倍的 token，在测试集约 44.7 倍。测试集中，RAG 的候选答题阶段消耗 2,939,087 token，占 RAG 总用量约 94.7%。该对比的 Direct LLM 为 closed-book，全文对照的 token 用量需从其独立运行结果统计。

下列命令输出四次运行的总量和各阶段明细。本次结果保存在 `outputs/token-usage-comparison.json`。

```bash
.venv/bin/python -m scripts.summarize_token_usage \
  outputs/baseline-direct/train/trace.jsonl outputs/rag-train-final.jsonl \
  outputs/baseline-direct/test/trace.jsonl outputs/rag-test-final.jsonl
```

[全文直读对照](wattbot-full-context-baseline.md)记录了三种方法的全量训练集分数和测试集覆盖，以及两组数据的实测 token 用量。
