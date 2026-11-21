# WattBot Pi 工具型 RAG 实验

实验继续使用 HKUST(GZ) 的 `DeepSeek-V4.1-Flash`（组内确认对应 Flash 1017）、竞赛的 32 篇文档和既有本地评分函数。Node.js 通过 Pi Agent Core 1.5.2 运行工具循环，Python 复用现有索引、引文检查、证据复核及 CSV 输出。

| 实验组 | 初始检索 | 后续动作 |
| --- | --- | --- |
| 现有 RAG | 查询改写、多查询 BM25 | 固定的扩大检索范围与证据复核 |
| Calculator RAG | 原题 BM25，最多 8 个片段 | 调用计算器，提交候选并复核 |
| Pi RAG | 原题 BM25，最多 8 个片段 | 可额外搜索、读候选原页、调用计算器、提交候选并复核 |

新增两组共享同一提示词、同一初始上下文、同一计算器和复核。Pi 补检索每次最多返回 4 个片段，读页只限此前已发现的文档。每题最多 5 次 Agent 模型调用，证据复核的独立 API 调用另计。计算器接受原文引句及其中的数值，通过 Python `Decimal` 执行四则运算、比例、变化百分比和支持的单位换算。工具结果、引用、最终数值与复核决策都写入 trace。

## 复跑

```bash
npm ci --prefix pi_rag
.venv/bin/python -m wattbot.pi_agent --mode calculator --split train --workers 6
.venv/bin/python -m wattbot.pi_agent --mode pi --split train --workers 6
.venv/bin/python -m scripts.verify_rag_trace --questions data/raw/train_QA.csv \
  --trace outputs/pi-rag/pi/train/trace.jsonl --output outputs/pi-rag/pi/train/predictions.csv
.venv/bin/python -m unittest discover -s tests
```

每组的运行参数、预测、逐题评分和模型及工具 trace 存于 `outputs/pi-rag/<mode>/train/`。改用 `--split test` 可生成测试集预测，测试集没有用于本地评分的标签。

## 41 道训练题实测

| 方法 | 答案值准确率 | 本地加权分 | 引用 Jaccard | 误拒答 | API 调用 | 总 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 现有 RAG | 73.17%（30/41） | 0.7604 | 0.8252 | 5/39 | 153 | 501,292 |
| Calculator RAG | 65.85%（27/41） | 0.6939 | 0.7805 | 7/39 | 91 | 216,086 |
| Pi RAG | 65.85%（27/41） | 0.6939 | 0.7805 | 7/39 | 114 | 385,305 |

两组新变体各成功计算 3 次。Pi 另成功补检索 18 次、读页 7 次。逐题比较时，两组都答对 25 题，分别独自答对 2 题。Pi 多用 78.3% 的 token，汇总分相同。两组的预测、模型标识及引用都通过了 trace 回放。

与现有 RAG 的比较同时改变了起始检索、上下文预算和答题控制流程。要单独测量计算器的增益，需要补一组使用同样初始证据及答题方式的无计算器对照。

## 输出规范化修正

Pi 的区间输出统一为 `[low,high]`。数据集命名题在通过引文确认后去除小写的通用 `dataset` 描述。计算工具支持 `percent_decrease`，对明确问下降幅度的百分比题输出正值，带符号的变化率仍使用 `percent_change`。这些规则在语义复核之前执行。

| Pi 运行 | 答案值准确率 | 本地加权分 | 引用 Jaccard | 误拒答 |
| --- | ---: | ---: | ---: | ---: |
| 修正前 | 65.85%（27/41） | 0.6939 | 0.7805 | 7/39 |
| 原预测离线应用规范化 | 78.05%（32/41） | 0.7854 | 0.7805 | 7/39 |
| 修正后重新运行 | 78.05%（32/41） | 0.7933 | 0.8171 | 6/39 |

离线回放只改了五道已识别格式或语义的题，保持原证据和其余答案。修正后的完整运行包含 113 次 API 调用，共 378,897 token，41 行均通过 trace 回放。这 41 题也用于定位错误，分数是修复回归结果。

```bash
.venv/bin/python -m wattbot.pi_agent --mode pi --split train --workers 6 \
  --output-dir outputs/pi-rag/pi/train-normalized
.venv/bin/python -m scripts.verify_rag_trace --questions data/raw/train_QA.csv \
  --trace outputs/pi-rag/pi/train-normalized/trace.jsonl \
  --output outputs/pi-rag/pi/train-normalized/predictions.csv
```

## 提交字段与计算单位校验

`submit_answer` 明确要求 `answer_value` 和 `answer_unit`。作答还需解释和可校验的引文。模型省略展示用的答案文字时由数值和单位补齐，若只提供了理由，使用该理由作为解释。补齐动作写入 trace。拒答的数值和单位必须为 `is_blank`。计算 ID 只有在当前题工具已返回该 ID 时才有效。

计算结果包含 `calculation_id`、数值和单位。后续计算可以引用前一步结果，并继承其引文。提交时用相同单位比较模型填写的数值与可执行计算结果，兼容单位先换算。不兼容单位或不一致数值会被拒绝，trace 中会保留具体反馈。通过检查的绑定在 trace 中记录为 `calculation_binding`。

提交与单位校验后的 41 题重跑保存在 `outputs/pi-rag/pi/train-validated/`，通过完整 trace 回放。答案值准确率为 70.73%（29/41），本地加权分为 0.7183。相比上一轮，`q262` 从拒答变为正确的 `1438 lbs`。`q280` 从错误的 `312.5 days` 问题中保住了小时到天的换算，导出的 `13.020833… days` 仍与标签 `13 days` 相差约 0.16%，超出 0.1% 评分容差。另有四题从答对变成拒答，包括三道证据不足和一道引文及解释校验未通过。

## 判断题字段语义

提交工具的提示词和字段描述均区分命题真假与证据是否充分：

| 证据状态 | `answerable` | `answer_value` | `answer_unit` |
| --- | --- | --- | --- |
| 证据支持命题 | `true` | `1` | `is_blank` |
| 证据反驳命题 | `true` | `0` | `is_blank` |
| 证据不足以判定 | `false` | `is_blank` | `is_blank` |

证据支持或反驳命题时都需提交引文。证据不足时用 `reason` 说明缺少什么。本轮复测沿用上轮的检索、工具调用和证据复核配置。

字段语义回归测试已通过。两次 41 题 API 复测均因校园网关返回 HTTP 429 未完成。一次只跑一道题的定向重试覆盖了 `q102`、`q230`、`q246`，三题均在得到有效模型响应前失败，本轮没有可用的新准确率。全量失败 trace 保存在 `outputs/pi-rag/pi/train-field-semantics/` 和 `outputs/pi-rag/pi/train-field-semantics-retry/`，定向重试保存在 `outputs/pi-rag/pi/field-semantics-targeted.jsonl`。

校园 API 恢复后，以 2 个 worker 重跑完整训练集并通过 trace 回放。三道目标判断题 `q102`、`q230`、`q246` 均作答为 `0` 并得分。相比上一轮 29/41，本轮为 31/41（75.61%），本地加权分从 0.7183 升至 0.7628，误拒答从 9/39 降为 8/39。`q262` 从正确变为拒答。共 114 次 API 调用、413,014 token。新产物位于 `outputs/pi-rag/pi/train-field-semantics-retest/`。

## 282 题测试集与 Kaggle 评测

2026 年 11 月 20 日使用 4 个 worker 在 `data/raw/test_Q.csv` 上运行完整 282 题（产物位于 `outputs/pi-rag/pi/test/predictions.csv`，通过 `wattbot verify` 校验）：

- **作答覆盖**：226 题作答 / 56 题拒答。
- **Token 开销**：总计 3,041,665 tokens（均耗 **10,786.0 tok/题**）。
- **Kaggle 赛后评测（Ref `58607853`）**：Public **0.785** / Private **0.735**（对照：Pi v2 Thinking off 升至 `0.873 / 0.818`，完整汇总见 [Kaggle 测试集结果汇总](kaggle-test-results.md)）。
