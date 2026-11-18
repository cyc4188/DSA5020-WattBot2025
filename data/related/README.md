# WattBot2025 retrieval / agentic-RAG papers

KohakuRAG (WattBot 2025 winner) publishes no formal bibliography. These are curated canonical references for the components its pipeline implements (hierarchical indexing, dense + BM25 hybrid retrieval, multi-query planning, reranking, multimodal embeddings, ensembling).

Collected 95 papers; shortlist 95; 95 seed references.

## KohakuRAG component -> reference area

| KohakuRAG component | Paper topic |
|---|---|
| Document -> section -> paragraph -> sentence tree | hierarchical-long-context (RAPTOR) |
| Sentence embeddings + vector store | dense-sparse-retrieval, embeddings-chunking |
| BM25 hybrid search | dense-sparse-retrieval (BM25/SPLADE/BEIR) |
| LLM planner / multi-query | query-planning, agentic-retrieval |
| Dedup + frequency/score rerank | reranking |
| Jina v3/v4 text + image embeddings | embeddings-chunking, multimodal-document-rag |
| Answer generation over context | rag-foundations, self-reflective-rag |
| Ensemble voting / validation | rag-evaluation |

## agentic-retrieval (22)

- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136) (2025) (seed)
- [DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models](https://arxiv.org/abs/2403.10081) (2024) (seed)
- [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509) (2022) (seed)
- [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983) (2023) (seed)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) (2025) (seed)
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) (2025) (seed)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (2023) (seed)
- [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350) (2022) (seed)
- [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) (2023) (seed)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) (2025) (seed)
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789) (2023) (seed)
- [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864) (2023) (seed)
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) (2023) (seed)
- [An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511) (2023) (seed)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854) (2023) (seed)

## context-compression (2)

- [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408) (2023) (seed)
- [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736) (2023) (seed)

## dense-sparse-retrieval (14)

- [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648) (2023) (seed)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) (2020) (seed)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (2020) (seed)
- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) (2021) (seed)
- [Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118) (2021) (seed)
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) (2022) (seed)
- [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281) (2023) (seed)
- [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) (2021) (seed)
- [C-Pack: Packed Resources For General Chinese Embeddings](https://arxiv.org/abs/2309.07597) (2023) (seed)
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) (2022) (seed)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) (2021) (seed)
- [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) (2019) (seed)
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) (2017) (seed)
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (2016) (seed)

## embeddings-chunking (4)

- [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/abs/2409.04701) (2024) (seed)
- [jina-embeddings-v3: Multilingual Embeddings With Task LoRA](https://arxiv.org/abs/2409.10173) (2024) (seed)
- [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663) (2024) (seed)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) (2022) (seed)

## graph-rag (4)

- [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921) (2024) (seed)
- [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779) (2024) (seed)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) (2024) (seed)
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) (2024) (seed)

## hierarchical-long-context (5)

- [Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach](https://arxiv.org/abs/2407.16833) (2024) (seed)
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) (2024) (seed)
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) (2024) (seed)
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508) (2023) (seed)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (2023) (seed)

## multimodal-document-rag (11)

- [VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents](https://arxiv.org/abs/2410.10594) (2024) (seed)
- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) (2024) (seed)
- [jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval](https://arxiv.org/abs/2506.18902) (2025) (seed)
- [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](https://arxiv.org/abs/2411.04952) (2024) (seed)
- [MinerU: An Open-Source Solution for Precise Document Content Extraction](https://arxiv.org/abs/2409.18839) (2024) (seed)
- [olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models](https://arxiv.org/abs/2502.18443) (2025) (seed)
- [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418) (2023) (seed)
- [DocLLM: A layout-aware generative language model for multimodal document understanding](https://arxiv.org/abs/2401.00908) (2023) (seed)
- [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387) (2022) (seed)
- [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) (2021) (seed)
- [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398) (2020) (seed)

## query-planning (8)

- [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (2022) (seed)
- [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283) (2023) (seed)
- [Query Expansion by Prompting Large Language Models](https://arxiv.org/abs/2305.03653) (2023) (seed)
- [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406) (2022) (seed)
- [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117) (2023) (seed)
- [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620) (2024) (seed)
- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091) (2023) (seed)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) (2023) (seed)

## rag-evaluation (8)

- [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391) (2024) (seed)
- [CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2401.17043) (2024) (seed)
- [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2407.11005) (2024) (seed)
- [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://arxiv.org/abs/2408.08067) (2024) (seed)
- [CRAG -- Comprehensive RAG Benchmark](https://arxiv.org/abs/2406.04744) (2024) (seed)
- [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) (2023) (seed)
- [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431) (2023) (seed)
- [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476) (2023) (seed)

## rag-foundations (6)

- [Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473) (2024) (seed)
- [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652) (2023) (seed)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020) (seed)
- [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083) (2023) (seed)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) (2020) (seed)
- [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299) (2022) (seed)

## reranking (3)

- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) (2019) (seed)
- [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/abs/2304.09542) (2023) (seed)
- [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713) (2020) (seed)

## routing (1)

- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665) (2024) (seed)

## self-reflective-rag (3)

- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) (2024) (seed)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) (2023) (seed)
- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403) (2024) (seed)

## surveys-multihop (4)

- [Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/abs/2407.08223) (2024) (seed)
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (2023) (seed)
- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600) (2018) (seed)
- [MuSiQue: Multihop Questions via Single-hop Question Composition](https://arxiv.org/abs/2108.00573) (2021) (seed)
