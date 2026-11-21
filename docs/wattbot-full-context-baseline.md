# WattBot 全文直读 Baseline

全文直读组用与 RAG 相同的 HKUST(GZ) `DeepSeek-V4.1-Flash`。每道题的输入由固定提示词、同一份完整语料、原题和单位提示组成。语料使用竞赛索引中的 32 篇文档、641 页正文和 835 条表格行，按文档 ID、页码顺序拼接。图片本身没有传入；正文中已抽取的图注仍在语料中。各题使用同一份完整语料，流程无查询改写、检索、片段排序或第二次 LLM 复核。程序根据文档 ID 和页码核对返回的引文。

## 单题实测

2026 年 11 月 19 日，校园 API 成功处理训练集第一题 `q003` 的单次全文请求。答案值和文档引用均通过该题的本地校验。这个样本用于验证请求能够装下全文。

| 实测项 | 结果 |
| --- | ---: |
| 输入 token | 687,124 |
| 输出 token | 150 |
| 总 token | 687,274 |
| API 耗时 | 80.0 秒 |
| 服务端报告的缓存 token | 0 |

如果每题仍需传入约 68.7 万输入 token，41 题约需 2,817 万输入 token，282 题约需 1.94 亿输入 token，均为按单题外推。实际全量运行另计格式重试和服务端返回的用量。

## 全量训练集结果

以下三组按同一套 41 题和本地评分函数对比。全文直读组使用 4 个并发 worker，全部 41 道题已通过配置、语料和逐题 trace 回放。

| 指标 | Closed-book | 全文直读 | RAG |
| --- | ---: | ---: | ---: |
| 答案值准确率 | 36.59%（15/41） | 80.49%（33/41） | 73.17%（30/41） |
| 本地加权分 | 0.3354 | 0.7988 | 0.7604 |
| 参考文档 Jaccard（题均） | 0.0488 | 0.6829 | 0.8252 |
| 可回答性判断准确率 | 53.66%（22/41） | 92.68%（38/41） | 87.80%（36/41） |
| 正确拒答 | 1/2 | 2/2 | 2/2 |
| 可回答题误拒答 | 18/39 | 3/39 | 5/39 |
| 总 token | 10,283 | 28,870,040 | 501,292 |
| 平均每题总 token | 250.80 | 704,147.32 | 12,226.63 |

全文组共 42 次调用，API 报告 28,859,500 输入 token、10,540 输出 token、0 缓存 token。API 响应时间中位数为 15.15 秒。全文直读和 RAG 都答对 28 题，全文组独自答对 5 题，RAG 独自答对 2 题。两组的端到端流程不同，结果反映整体方案。

## 全量测试集结果

| 指标（282 题） | Closed-book | 全文直读 | RAG |
| --- | ---: | ---: | ---: |
| 作答 | 134 | 259 | 236 |
| 拒答 | 148 | 23 | 46 |
| 输入 token | 47,466 | 194,458,171 | 2,993,919 |
| 输出 token | 22,012 | 66,633 | 108,443 |
| 总 token | 69,478 | 194,524,804 | 3,102,362 |
| 平均每题总 token | 246.38 | 689,804.27 | 11,001.28 |

全文组共 283 次调用并通过全量 trace 回放。API 响应时间中位数为 13.45 秒，4 个 worker 的整批用时为 17 分 24 秒。279 个响应提供缓存统计，均为 0。259 道作答题中有 206 道至少保留一条通过页码与引文校验的引用。2026 年 11 月 20 日将该原始产物提交至 Kaggle 赛后通道（编号 `58605663`），返回公开榜 **0.858**、私有榜 **0.815**；经 `parser-fix` 离线引用恢复后（编号 `58605673`）进一步提升至公开榜 **0.883**、私有榜 **0.837**（完整汇总见 [Kaggle 测试集结果汇总](kaggle-test-results.md)）。

相较于 RAG，全文组在训练集多答对 3 题，引用 Jaccard 低 0.1423，训练集与测试集总 token 分别为 RAG 的 57.6 倍和 62.7 倍。训练集整批用时为 3 分 54 秒，与测试任务并行运行。

## 复跑与本地产物

```bash
.venv/bin/python -m wattbot.baselines.full_context --split train --workers 4
.venv/bin/python -m wattbot.baselines.full_context --split test --workers 4
.venv/bin/python -m scripts.verify_direct_baseline outputs/baseline-full-context/train
.venv/bin/python -m scripts.verify_direct_baseline outputs/baseline-full-context/test
.venv/bin/python -m scripts.summarize_token_usage \
  outputs/baseline-full-context/train/trace.jsonl outputs/baseline-full-context/test/trace.jsonl
```

默认输出目录为 `outputs/baseline-full-context/`。整份语料只在每次运行的 `corpus.txt` 保存一份。请求 trace 记录它的 SHA-256、问题和单位提示，响应记录模型返回、token 用量和耗时。配置和结果保存在 `config.json`、`predictions.csv`、`summary.json` 及训练集的 `metrics.json`。各组另存 `token-usage.json`。重新运行时可通过 `--output-dir` 指定新目录。可用 `--resume` 恢复同一配置下未完成的运行。
