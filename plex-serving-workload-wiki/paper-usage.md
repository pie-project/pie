# Paper-to-workload usage matrix

This matrix records confirmed use from the 87-paper audit or the workload's
representative role in that literature. It does not claim that every paper
using a derived ShareGPT/Azure trace is exhaustively listed.

| Workload | Surveyed-paper usage / role |
|---|---|
| [Azure LLM Inference Trace 2023](entries/azure-llm-2023.md) | Splitwise; Vidur examples |
| [Azure LLM Inference Trace 2024](entries/azure-llm-2024.md) | DynamoLLM; LMetric comparison; multiple serving simulators |
| [BurstGPT v2](entries/burstgpt-v2.md) | BurstGPT workload study; suitable replacement for Poisson arrivals in serving papers |
| [Qwen-Bailian Anonymous Usage Traces](entries/qwen-bailian.md) | KVCache Cache in the Wild; LMetric; SMetric |
| [Mooncake FAST'25 Traces](entries/mooncake-fast25.md) | Mooncake; LMetric ToolAgent workload |
| [ServeGen](entries/servegen.md) | ServeGen; temporal holdout beyond most surveyed policies |
| [LMSYS-Chat-1M](entries/lmsys-chat-1m.md) | Common content source for ShareGPT-like serving workloads and model routing |
| [WildChat-1M](entries/wildchat-1m.md) | Not a common legacy serving trace; recommended content/session overlay |
| [ShareGPT / Vicuna Conversation Corpus](entries/sharegpt.md) | InferCept; VTC-derived traces; QLM; DLPM; Llumnix; CachedAttention; many serving baselines |
| [Azure Functions Trace 2019](entries/azure-functions-2019.md) | PARD uses an Azure Functions trace; useful legacy baseline for burst and cold-start policies |
| [SWE-bench Verified](entries/swe-bench-verified.md) | Continuum; ThunderAgent; ConServe; KAIROS; Dyserve; SAGA |
| [mini-SWE-agent Trajectories](entries/mini-swe-agent.md) | Continuum and KAIROS use mini-SWE-agent-style workloads |
| [OpenHands Evaluation Trajectories](entries/openhands.md) | ThunderAgent; ScienceAgentBench integrations |
| [Berkeley Function Calling Leaderboard V4](entries/bfcl-v4.md) | Continuum |
| [tau3-bench](entries/tau3-bench.md) | Recommended new workload; not broadly used in first-wave serving papers |
| [ToolBench / StableToolBench](entries/toolbench.md) | ToolLLM-style workloads in SLOs-Serve and agent papers |
| [Terminal-Bench 2.0](entries/terminal-bench-2.md) | KAIROS |
| [AgentBench](entries/agentbench.md) | Recommended breadth workload |
| [OSWorld-Verified](entries/osworld-verified.md) | Recommended multimodal agent workload |
| [WebArena / BrowserGym](entries/webarena.md) | Recommended browser-agent workload |
| [GAIA](entries/gaia.md) | Dyserve |
| [ScienceAgentBench](entries/scienceagentbench.md) | ThunderAgent; OpenHands integration |
| [TheAgentCompany](entries/theagentcompany.md) | Recommended new enterprise workflow workload |
| [LongMemEval](entries/longmemeval.md) | Recommended logical-request and long-session workload |
| [Dynasor / Math500 Reasoning Workload](entries/dynasor-math500.md) | Certaindex / Dynasor |
| [LiveCodeBench](entries/livecodebench.md) | Dyserve |
| [GSM8K / MATH / AIME / Math500 Suite](entries/math-reasoning-suite.md) | InferCept GSM8K-XL; Certaindex; many reasoning schedulers |
| [HumanEval / MBPP / APPS Code Suite](entries/code-reasoning-suite.md) | RouterBench includes MBPP; common serving and routing benchmarks |
| [LongBench v2](entries/longbench-v2.md) | Recommended new long-context workload |
| [InfiniteBench](entries/infinitebench.md) | Recommended extreme long-context workload |
| [BEIR](entries/beir.md) | RAG and prefix-cache papers; HotpotQA used by InferCept |
| [RAGBench](entries/ragbench.md) | Recommended RAG policy workload |
| [HotpotQA](entries/hotpotqa.md) | InferCept; BEIR/RAG workloads |
| [MMMU / MMMU-Pro](entries/mmmu.md) | Recommended multimodal serving extension |
| [Video-MME](entries/video-mme.md) | Recommended encoder-residency and long-prefill workload |
| [RouterBench](entries/routerbench.md) | Model-routing related work |
| [RouteLLM / Chatbot Arena Preference Data](entries/routellm-arena.md) | Router-tier control papers |
| [S-LoRA Synthetic Adapter Workload](entries/slora-synthetic.md) | S-LoRA; basis for Punica/dLoRA/CaraServe/Chameleon comparisons |
| [Vidur](entries/vidur.md) | Used broadly for scheduling and capacity studies |
| [LLMServingSim 2.0](entries/llmservingsim.md) | Recommended infrastructure stress backend |
| [vLLM Bench Serve](entries/vllm-bench-serve.md) | Common evaluation harness in serving artifacts |
| [SGLang Serving Benchmarks](entries/sglang-bench-serve.md) | SGLang/RadixAttention; KVFlow; PEEK; many prefix-cache papers |