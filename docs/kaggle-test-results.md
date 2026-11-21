# WattBot 2025 Kaggle 测试集结果汇总

本文件汇总已提交至 Kaggle [WattBot 2025](https://www.kaggle.com/competitions/WattBot2025/submissions)（282 道测试题，赛后评估通道同时返回 Public / Private 分数）的全部 **28 份**测试集评测结果与消融对比。所有测试产物与 trace 均位于 Git 忽略的 `outputs/` 目录。

> 赛题官方冠军参考（KohakuRAG，*arXiv:2603.07612*）：Public **0.902** / Private **0.861**。

## 1. 全量测试集结果总表（按 Private Score 降序）

最后更新：2026-11-20（28 份提交均已 `COMPLETE`）。

| # | 方案 | Public | Private | 41 题训练集加权分 | 测试集作答 / 拒答 | 测试集平均 tok/题 | Kaggle Ref | 本地产物 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | **5 范式加权共识 (`q0.5`) → 多模态兜底级联** | 0.896 | **0.891** | — | 266 / 16 | 840,795 | `58614553` | `outputs/multi-paradigm-ensemble/ens-5paradigm-q05-mmcascade.csv` |
| 2 | **5 范式加权共识 (`pi_on+doc+fc+mhop+mm`, blank-filter `q0/s0`)** | **0.900** | 0.888 | 0.9024 | 268 / 14 | 840,795 | `58614531` | `outputs/multi-paradigm-ensemble/ens-5paradigm-wt-bf.csv` |
| 3 | **5 范式加权共识 (`pi_on+doc+fc+mhop+mm`, quorum `q0.5/s0.5`)** | 0.894 | 0.878 | 0.8780 | 261 / 21 | 840,795 | `58614539` | `outputs/multi-paradigm-ensemble/ens-5paradigm-wt-q05.csv` |
| 4 | **纯 RAG 级联：`pi_on+mm → doc_routing` (23.4% 升级，无 FC)** | 0.895 | **0.872** | 0.8970 | 267 / 15 | **79,711** | `58615756` | `outputs/multi-paradigm-ensemble/rag-cascade-pion-mm-doc.csv` |
| 5 | **4 范式无 Full-Context 共识 (`pi_on+doc+mhop+mm`, blank-filter)** | 0.896 | 0.872 | **0.9171** | 267 / 15 | 150,991 | `58614523` | `outputs/multi-paradigm-ensemble/ens-4paradigm-nofc-bf.csv` |
| 6 | **纯 RAG 3 路共识：`mhop+mm+pi_off` (无 FC/无路由/无 Thinking)** | 0.888 | **0.865** | 0.8982 | 267 / 15 | **44,485** | `58615761` | `outputs/multi-paradigm-ensemble/rag-consensus-mhop-mm-pioff.csv` |
| 7 | **纯 RAG 3 路共识：`mhop+v2cas+pi_on` (无 FC/无文档路由)** | 0.880 | 0.865 | **0.9171** | 265 / 17 | 67,182 | `58615754` | `outputs/multi-paradigm-ensemble/rag-consensus-mhop-v2cas-pion.csv` |
| 8 | **4 范式无 Full-Context (`q0.5`) → 多模态兜底级联** | 0.896 | 0.865 | — | 265 / 17 | 150,991 | `58614580` | `outputs/multi-paradigm-ensemble/ens-4paradigm-nofc-q05-mmcascade.csv` |
| 9 | **7 路混合加权共识 (`+pi_off+v2cas`, blank-filter `q0/s0`)** | 0.896 | 0.865 | 0.8988 | 268 / 14 | 879,016 | `58614568` | `outputs/multi-paradigm-ensemble/ens-7diverse-wt-bf.csv` |
| 10 | **纯 RAG 级联：`mhop+v2cas → pi_on` (20.6% 升级，无 FC/无路由)** | 0.879 | 0.864 | **0.9171** | 265 / 17 | 54,273 | `58615752` | `outputs/multi-paradigm-ensemble/rag-cascade-mhop-v2cas-pion.csv` |
| 11 | **7 路混合加权共识 (`+pi_off+v2cas`, quorum `q0.3/s0.4`)** | 0.896 | 0.864 | 0.8988 | 266 / 16 | 879,016 | `58614546` | `outputs/multi-paradigm-ensemble/ens-7diverse-wt-q03.csv` |
| 12 | **Pi Agent v2（Thinking on）** *(单路径最高 Private)* | 0.870 | 0.854 | 0.8866 | 253 / 29 | 43,182 | `58608983` | `outputs/pi-v2-sandbox/pi-v2-thinking-on/test/predictions.csv` |
| 13 | **Cascade: v1 RAG → Full-Context (on abstain)** | 0.868 | 0.845 | 0.8134 | 262 / 20 | 125,557 | `58605666` | `outputs/cascade-ensemble/test-cascade-rag-fc.csv` |
| 14 | **两阶段文档路由 (`doc_routing`, top-3)** *(单路径最高 Public)* | 0.885 | 0.841 | 0.8951 | 258 / 24 | 86,253 | `58606928` | `outputs/doc-routing/test/predictions.csv` |
| 15 | **Full-Context + 引用恢复 (`parser-fix`)** | 0.883 | 0.837 | 0.8290 | 259 / 23 *(255 有效引用)* | 689,804 *(+离线回放)* | `58605673` | `outputs/parser-fix/full-context-recovered/test/predictions.csv` |
| 16 | **多跳上下文混合检索 (`multihop_rag`)** | 0.852 | 0.825 | 0.7988 | 249 / 33 | **8,708** | `58607220` | `outputs/multihop-retrieval/test/predictions.csv` |
| 17 | **Pi Agent v2（Thinking off）** | 0.873 | 0.818 | 0.8628 | 253 / 29 | 22,929 | `58607225` | `outputs/pi-v2-sandbox/pi-v2-thinking-off/test/predictions.csv` |
| 18 | **Consensus: v1 RAG + Full-Context** | 0.855 | 0.816 | — | 252 / 30 | — | `58605676` | `outputs/cascade-ensemble/test-consensus-rag-fc.csv` |
| 19 | **Clean-index v2 RAG + 多模态级联 (`c17c822`)** | 0.858 | 0.815 | 0.8530 | 259 / 23 | 15,292 | `58587706` | `outputs/wattbot-final/candidate/submission.csv` |
| 20 | **Full-Context Baseline (raw v1)** | 0.858 | 0.815 | 0.7988 | 259 / 23 *(206 有效引用)* | 689,804 | `58605663` | `outputs/baseline-full-context/test/predictions.csv` |
| 21 | **独立多模态单跑 (`mode2-only`, v1 索引)** | 0.854 | 0.807 | 0.8354 | 258 / 24 *(39 视觉升级)* | 12,848 | `58607180` | `outputs/multimodal/test-mode2/predictions.csv` |
| 22 | **Clean-index v2 纯文本 RAG** *(级联 Stage 1)* | 0.848 | 0.768 | 0.8262 | 241 / 41 | — | `58605736` | `outputs/wattbot-final/candidate/stage1_text_only.csv` |
| 23 | **原始 v1 Chunk RAG (DeepSeek-V4.1-Flash)** | 0.814 | 0.750 | 0.7604 | 236 / 46 | 11,001 | `58605661` | `outputs/submission_rag_verified.csv` |
| 24 | **Pi Agent v1 (工具增强 RAG)** | 0.785 | 0.735 | 0.7628 | 226 / 56 | 10,786 | `58607853` | `outputs/pi-rag/pi/test/predictions.csv` |
| 25 | **跨模型 v1 Chunk RAG (`DeepSeek-V4-Pro`)** | 0.781 | 0.720 | 0.7713 | 225 / 57 | 12,523 | `58606992` | `outputs/cross-model/rag-pro-test.csv` |
| 26 | **跨模型 v1 Chunk RAG (`Qwen3.8-27B`)** | 0.830 | 0.694 | 0.7543 | 230 / 52 | 12,764 | `58606569` | `outputs/cross-model/rag-qwen-test.csv` |
| 27 | **Direct LLM Baseline (Closed-book 无检索)** | 0.286 | 0.419 | 0.3354 | 134 / 148 | 246 | `58605665` | `outputs/baseline-direct/test/predictions.csv` |
| 28 | **Minimal 全 `is_blank` 冒烟基线** | 0.138 | 0.175 | — | 0 / 282 | 0 | `58605680` | `outputs/submission_minimal.csv` |

## 2. 测试集精度–开销帕累托前沿（Private Score vs. Mean Tokens/Q）

在全部 28 份测试集评测中，按 Private Score 严格位于帕累托前沿的核心工作点（全部仅使用单一开源模型 `DeepSeek-V4.1-Flash`）：

1. **Minimal 全 `is_blank`**：Private `0.175` @ `0 tok/题`
2. **Direct LLM (Closed-book)**：Private `0.419` @ `246 tok/题`
3. **多跳上下文混合检索 (`multihop_rag`)**：Private **0.825** @ **8,708 tok/题**（纯片段 RAG，1/79 Full-Context 开销超越 raw Full-Context `0.815`）
4. **Pi Agent v2（Thinking on）**：Private **0.854** @ **43,182 tok/题**（单路径 Private 第一）
5. **纯 RAG 3 路共识 (`mhop+mm+pi_off`，无 FC、无路由、无 Thinking)**：Private **0.865** / Public **0.888** @ **44,485 tok/题**（**仅 44.5k tok/题即超越 KohakuRAG 冠军 Private `0.861`**，比 Full-Context 便宜 **15.5×**）
6. **纯 RAG 级联 (`pi_on+mm → doc_routing`，仅 23.4% 分歧题升级路由，无 FC)**：Private **0.872** / Public **0.895** @ **79,711 tok/题**（比单独跑 `doc_routing` 还省 token，超越 KohakuRAG 冠军 **+0.011**）
7. **5 范式加权共识 (`q0.5`) → 多模态兜底级联** *(含 Full-Context 参照上界)*：Private **0.891** / Public **0.896** @ **840,795 tok/题**（全局 Private 最高，超冠军 **+0.030**）

## 3. 测试集关键消融结论

1. **异构多范式集成 + 多模态兜底超越冠军**：
   - 将 5 个正交范式（工具沙盒推理 `Pi v2 Thinking on` + 整篇精读 `doc_routing` + 全局直读 `fc_recov` + 多跳稠密重排 `multihop` + 视觉读图 `mode2-only`）组合后：
     - 纯 quorum（`q0.5/s0.5`）：`0.894 / 0.878`（相对最强单路径 `0.854` 暴涨 **+0.024 Private**）
     - 纯 Kohaku 式 blank-filtering（`q0/s0`）：**`0.900 / 0.888`**（Public 追平冠军级别，Private 超冠军 **+0.027**）
     - **Quorum (`q0.5`) + 仅对剩余拒答题多模态兜底**：**`0.896 / 0.891`**（Private 较纯 quorum 再涨 **+0.013**、超 KohakuRAG 冠军 **+0.030**——既拦截了单路文本在不可回答题上的幻觉，又救回了纯文本全员拒答的图表题）
   - **正交多样性 > 相关堆叠**：往正交 5 范式里塞入高度相关的 `pi_off` 和 `v2cas` 组成 7 路投票，Private 反而从 `0.888` 跌至 `0.865`（−0.023）。
2. **索引清洗与多模态 2×2 消融**：
   - 原始 v1 纯文本：`0.814 / 0.750` → Clean-index v2 纯文本：`0.848 / 0.768`（Public +0.034 / Private +0.018）
   - 独立多模态 `mode2-only`（v1 索引）：`0.854 / 0.807` → Clean-index v2 + 多模态级联：`0.858 / 0.815`（相对 v2 纯文本 Private +0.047）
3. **Pi Agent v1 → v2 与 Thinking 消融**：
   - Pi v1：`0.785 / 0.735` → Pi v2 Thinking off：`0.873 / 0.818`（+0.088 / +0.083）→ Pi v2 Thinking on：`0.870 / 0.854`（Private 再涨 **+0.036**）
4. **两阶段文档路由 vs. 全文直读**：
   - 两阶段文档路由（`0.885 / 0.841`，86.3k tok/题）以 **1/8** 的 token 全面超越 raw Full-Context（`0.858 / 0.815`）与引用恢复版 Full-Context（`0.883 / 0.837`）。
5. **离线引用恢复零 token 收益**：
   - Full-Context raw（`0.858 / 0.815`）→ + Citation Recovery（`0.883 / 0.837`）：Public **+0.025**，Private **+0.022**。
6. **同索引跨模型对比**（v1 Chunk RAG，BM25 top-10）：
   - `DeepSeek-V4.1-Flash`：`0.814 / 0.750`；`DeepSeek-V4-Pro`：`0.781 / 0.720`；`Qwen3.8-27B`：`0.830 / 0.694`。
