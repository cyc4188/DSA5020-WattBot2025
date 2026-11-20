# WattBot 分级检索问答系统与复现实验

WattBot 根据 32 份能源、碳排放和可持续性相关文档回答问题。系统先用保留版面结构的索引做文本检索与证据核验，再把文本无法可靠作答的题交给多模态模型。它从问题 CSV 生成一份包含答案、单位、引用和解释的 CSV，供本地评估或课程演示。系统的质量和开销在同一次运行中按题目和阶段记录。

## 文档索引与推理流程

1. **构建索引。** 用 PDF 中的字词坐标恢复表格单元格，用 Markdown 提取补抓无边框表格，并把列标题和单位保留在表格行及多行块中。对正文做 Unicode 归一化和断行修复。索引记录文档 ID、页码和源文件哈希。第一阶段读取清洗版 v2，多模态分支沿用其原先评估用的 v1。 [PDF 索引实验](exp-parser-fix.md)
2. **执行文本 RAG。** 使用问题和单位提示生成备选查询，默认用 BM25 召回文本和表格片段。模型从检索内容提取答案。程序定位引用片段、归一化答案单位，再要求模型核验引用是否支持答案。证据不足时扩大检索范围，最多尝试三轮。
3. **路由拒答题。** 文本阶段的有效答案直接保留。对规范化拒答，多模态分支先尝试其文本路径，必要时渲染检索到的文档页。视觉模型从页面和页面文本中回答，再对所引用页面执行一次视觉核验。若仍无法作答，输出规范拒答。
4. **生成 CSV。** 保持输入题目顺序，核查题目覆盖、引用 ID 与 URL 对应关系，以及拒答的五个 `is_blank` 字段。JSONL trace 保存每个阶段的请求、路由结果、模型用量和最终行。最终输出的哈希在运行结束时写入 trace。

## 推理和评分复现

从仓库根目录使用项目虚拟环境。将竞赛 CSV 放到 `data/raw/` 并下载源文档，再构建两份索引。校园模型配置沿用仓库现有的 OpenCode provider，模型为 `DeepSeek-V4.1-Flash`。

```bash
.venv/bin/python -m pip install -r requirements-rag.txt
.venv/bin/python scripts/download_docs.py
.venv/bin/python -m wattbot index
.venv/bin/python -m wattbot.clean_index
.venv/bin/python -m wattbot cascade --questions data/raw/train_QA.csv --workers 2 \
  --output outputs/reproduction/train/predictions.csv \
  --trace outputs/reproduction/train/trace.jsonl
.venv/bin/python -m wattbot score --output outputs/reproduction/train/predictions.csv \
  --metrics outputs/reproduction/train/metrics.json
```

生成测试提交时使用 `data/raw/test_Q.csv`，并选择独立的输出和 trace 路径。中断后用同一配置和 trace 加 `--resume`，已完成的题会从 trace 恢复。预测、文档、索引和 trace 都留在 Git 忽略目录。

## 实验口径

41 道有标签训练题用于开发和方案对比。加权分由答案值（75%）、引用集合 Jaccard（15%）和拒答判断（10%）组成。数值按本地评分器的 ±0.1% 相对误差判定，其余规则见项目评分实现。282 道无标签测试题用于验证全量运行、输出格式和调度开销。

同一次 API 实跑的分阶段对照从最终 trace 还原文本基线。进入多模态分支的题在文本基线中保留为拒答，其他题保留同一次运行的文本答案。历史预测离线回放用于比较组合，并记录已保存的每题用量。历史回放与重新执行模型的结果分别报告。

| 41 题有标签比较 | 答案值正确 | 引用 Jaccard | 加权分 | 平均 tokens/题 |
| --- | ---: | ---: | ---: | ---: |
| 实跑中的文本阶段 | 33 | 0.8659 | 0.8262 | 9,301 |
| 实跑中的完整级联 | 34 | 0.8902 | 0.8530 | 13,801 |
| 历史预测离线回放 | 35 | 0.9431 | 0.8817 | 14,077 |

实跑中 5 道题升级，多模态阶段额外返回 2 个可用答案，其中 1 个值正确。误拒答从 3 个降至 1 个。这个比较测量完整后备阶段的影响，既包含多模态分支的文本检索，也包含看图调用。 [级联实跑与历史回放](exp-ensemble.md#6-live-clean-index-rag-with-multimodal-fallback)

## 历史候选与取舍

这些结果来自同一组 41 道开发题上的历史运行或回放。各方案的索引、提示和调用策略可能不同，表中的分差反映各配置的整体表现。平均 tokens 包含答题阶段调用，文档卡片等一次性离线构建成本另见各实验记录。

| 方案 | 关键改动 | 加权分 | 平均 tokens/题 | 实验来源 |
| --- | --- | ---: | ---: | --- |
| 原 Chunk RAG | BM25 召回正文块与表格行 | 0.7604 | 12,227 | [索引对照](exp-parser-fix.md#4-end-to-end-41-question-train-evaluation) |
| 清洗版 RAG | 版面表格索引、单位换算和证据核验 | 0.8287 | 9,199 | [索引对照](exp-parser-fix.md#4-end-to-end-41-question-train-evaluation) |
| 多跳检索 | 上下文增强索引、混合召回、重排与迭代补检 | 0.7988 | 9,271 | `2bccef0` 中的 `docs/exp-multihop-rag.md` |
| 多模态单跑 | 文本和按需看图 | 0.8354 | 12,101 | `dba59cf` 与 AGY 的 `multimodal/train-mode2` 产物 |
| Pi v2 Thinking off | 工具调用、受限计算和补检 | 0.8628 | 23,218 | `4251ab5` 中的 `docs/exp-pi-v2.md` |
| Pi v2 Thinking on | 保留工具流程并启用推理 | 0.8866 | 40,414 | `4251ab5` 中的 `docs/exp-pi-v2.md` |
| 两阶段文档路由 | 文档卡片做路由，再精读选中的文档 | 0.8951 | 84,975 | `a0e05c9` 中的 `docs/exp-doc-routing.md` |
| 历史 Full-Context 级联 | Flash RAG 和 Pi 首轮，困难题升级全上下文 | 0.8698 | 223,486 | [历史级联实验](exp-ensemble.md#2-41-question-train-accuracytoken-pareto-frontier) |

清洗索引提供了成本较低的文本主路径。文档路由与 Pi v2 Thinking 在开发题上的分数更高，适合作为精度与开销的参照。只按拒答升级时，后备模型只能改动第一阶段拒答的题，文本阶段已作答但答错的题需要不同的检测信号。

## 课程论文可用材料

方法部分可给出从版面索引到证据核验再到按拒答升级的流程图。结果部分同时报告答案正确数、引用 Jaccard、误拒答数和每题 token，保留索引修复、候选方法和多模态消融的实验条件。误差分析可用表格小数点、无边框表、跨文档证据和视觉读图作为类型，按同一口径记录成功与失败。

- RAG 基本框架。 [Lewis et al., NeurIPS 2020](https://proceedings.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
- PDF 版面与表格结构识别。 [Auer et al., Docling Technical Report](https://arxiv.org/abs/2408.09869)
- 表格与正文混合问答和数值推理。 [Zhu et al., TAT-QA, ACL 2021](https://aclanthology.org/2021.acl-long.254/)
