# PLEX serving-workload wiki

Search cutoff: **2026-07-23**

This wiki catalogs **42 existing public
workload sources** and **21 new PLEX
workload families**.

## Navigation

- [Root survey report](../plex_serving_workload_report.md)
- [Methodology and scoring](methodology.md)
- [Minimum PLEX trace schema](trace-schema.md)
- [Paper-to-workload usage matrix](paper-usage.md)
- [Machine-readable catalog](catalog.json)

## Existing public sources

| Workload | Kind | Tier | PLEX operations |
|---|---|---|---|
| [Azure LLM Inference Trace 2023](entries/azure-llm-2023.md) | Production request trace | core | route, admit, schedule |
| [Azure LLM Inference Trace 2024](entries/azure-llm-2024.md) | Production request trace | core | route, admit, schedule, rebalance |
| [BurstGPT v2](entries/burstgpt-v2.md) | Production request/session trace | core | admit, route, schedule, feedback, rebalance |
| [Qwen-Bailian Anonymous Usage Traces](entries/qwen-bailian.md) | Production session and KV-prefix trace | core | route, schedule, evict, feedback, rebalance |
| [Mooncake FAST'25 Traces](entries/mooncake-fast25.md) | Production and synthetic KV-prefix traces | core | route, schedule, evict, prefetch |
| [ServeGen](entries/servegen.md) | Production-fitted workload generator | core | admit, route, schedule, feedback, rebalance |
| [LMSYS-Chat-1M](entries/lmsys-chat-1m.md) | Real conversation content corpus | supporting | route, schedule, evict |
| [WildChat-1M](entries/wildchat-1m.md) | Timestamped real conversation corpus | extension | route, admit, schedule, feedback |
| [ShareGPT / Vicuna Conversation Corpus](entries/sharegpt.md) | Community conversation content corpus | supporting | schedule, evict, feedback |
| [Azure Functions Trace 2019](entries/azure-functions-2019.md) | Legacy production serverless trace | supporting | admit, route, rebalance, feedback |
| [SWE-bench Verified](entries/swe-bench-verified.md) | Agent task benchmark | core | schedule, evict, feedback, prefetch |
| [mini-SWE-agent Trajectories](entries/mini-swe-agent.md) | Agent trajectory generator | core | schedule, evict, feedback, prefetch |
| [OpenHands Evaluation Trajectories](entries/openhands.md) | Agent execution framework / trajectory generator | extension | schedule, evict, feedback, prefetch |
| [Berkeley Function Calling Leaderboard V4](entries/bfcl-v4.md) | Tool/function-calling benchmark | core | schedule, evict, feedback |
| [tau3-bench](entries/tau3-bench.md) | Stateful tool-agent-user benchmark | core | admit, schedule, evict, feedback, prefetch |
| [ToolBench / StableToolBench](entries/toolbench.md) | Large-scale tool-use trajectory corpus | extension | schedule, feedback, prefetch |
| [Terminal-Bench 2.0](entries/terminal-bench-2.md) | Executable agent task benchmark | core | schedule, evict, feedback |
| [AgentBench](entries/agentbench.md) | Multi-environment agent benchmark | extension | schedule, feedback |
| [OSWorld-Verified](entries/osworld-verified.md) | Multimodal computer-use benchmark | extension | schedule, feedback, evict |
| [WebArena / BrowserGym](entries/webarena.md) | Web-agent benchmark and trajectory corpus | extension | schedule, feedback, prefetch |
| [GAIA](entries/gaia.md) | General assistant benchmark | extension | route, schedule, feedback |
| [ScienceAgentBench](entries/scienceagentbench.md) | Scientific coding-agent benchmark | extension | schedule, feedback, prefetch |
| [TheAgentCompany](entries/theagentcompany.md) | Enterprise workflow-agent benchmark | extension | route, schedule, feedback |
| [LongMemEval](entries/longmemeval.md) | Timestamped long-term conversation benchmark | extension | route, schedule, evict, feedback |
| [Dynasor / Math500 Reasoning Workload](entries/dynasor-math500.md) | Reasoning trajectory benchmark | core | admit, schedule, feedback |
| [LiveCodeBench](entries/livecodebench.md) | Temporal code-reasoning benchmark | extension | route, schedule, feedback |
| [GSM8K / MATH / AIME / Math500 Suite](entries/math-reasoning-suite.md) | Reasoning content suite | supporting | admit, schedule, feedback |
| [HumanEval / MBPP / APPS Code Suite](entries/code-reasoning-suite.md) | Code-generation content suite | supporting | route, schedule, feedback |
| [LongBench v2](entries/longbench-v2.md) | Long-context content benchmark | extension | admit, schedule, evict, prefetch |
| [InfiniteBench](entries/infinitebench.md) | 100K+ long-context benchmark | extension | admit, schedule, evict, prefetch |
| [BEIR](entries/beir.md) | Retrieval benchmark suite | supporting | route, schedule, evict, prefetch |
| [RAGBench](entries/ragbench.md) | RAG evaluation corpus | extension | route, schedule, evict, prefetch, feedback |
| [HotpotQA](entries/hotpotqa.md) | Multi-hop QA content benchmark | supporting | route, schedule, evict, prefetch |
| [MMMU / MMMU-Pro](entries/mmmu.md) | Multimodal reasoning benchmark | extension | admit, route, schedule, evict, feedback |
| [Video-MME](entries/video-mme.md) | Long multimodal/video benchmark | extension | admit, route, schedule, evict, prefetch |
| [RouterBench](entries/routerbench.md) | Multi-model routing outcome dataset | extension | route, feedback |
| [RouteLLM / Chatbot Arena Preference Data](entries/routellm-arena.md) | Preference-based model-routing workload | supporting | route, feedback |
| [S-LoRA Synthetic Adapter Workload](entries/slora-synthetic.md) | Synthetic multi-adapter workload | extension | admit, route, schedule, evict, prefetch |
| [Vidur](entries/vidur.md) | High-fidelity serving simulator and workload generator | core | route, admit, schedule, feedback |
| [LLMServingSim 2.0](entries/llmservingsim.md) | Cycle-level heterogeneous/disaggregated simulator | extension | route, schedule, evict, prefetch, rebalance |
| [vLLM Bench Serve](entries/vllm-bench-serve.md) | Serving benchmark harness | supporting | admit, schedule, feedback |
| [SGLang Serving Benchmarks](entries/sglang-bench-serve.md) | Serving benchmark harness | supporting | schedule, evict, feedback |
## Proposed PLEX workloads

| Workload | Kind | Tier | PLEX operations |
|---|---|---|---|
| [Logical Continuation Mix](entries/logical-continuation-mix.md) | PLEX synthetic/reconstructed workload | core | admit, route, schedule, evict, feedback |
| [Continuation Storm](entries/continuation-storm.md) | PLEX synthetic/reconstructed workload | core | route, schedule, evict, feedback |
| [Heavy-Tailed Tool Latency and Failure](entries/tool-heavy-tail-failure.md) | PLEX synthetic/reconstructed workload | core | schedule, evict, feedback, prefetch |
| [Online-Revealed Agent DAG](entries/dynamic-agent-dag.md) | PLEX synthetic/reconstructed workload | core | route, admit, schedule, feedback |
| [Speculative Fanout with Cancellation](entries/speculative-fanout-cancel.md) | PLEX synthetic/reconstructed workload | core | admit, schedule, evict, feedback |
| [Prefix Hotspot and Popularity Shift](entries/prefix-hotspot-shift.md) | PLEX synthetic/reconstructed workload | core | route, evict, prefetch, rebalance |
| [Fairness versus Locality Conflict](entries/fair-locality-conflict.md) | PLEX synthetic/reconstructed workload | core | route, admit, schedule, feedback |
| [Dishonest Metadata and Learned Credibility](entries/dishonest-policy-hints.md) | PLEX synthetic/reconstructed workload | core | admit, route, schedule, evict, feedback |
| [Mixed Human, Agent, Batch, and Reasoning Service Classes](entries/mixed-service-classes.md) | PLEX synthetic/reconstructed workload | core | admit, route, schedule, feedback |
| [Session Abandonment, Retry, and Duplicate Delivery](entries/session-abandonment-retry.md) | PLEX synthetic/reconstructed workload | core | admit, schedule, evict, feedback |
| [Prefill/Decode/Append-Prefill Imbalance](entries/pd-append-prefill-imbalance.md) | PLEX synthetic/reconstructed workload | core | route, schedule, feedback, rebalance |
| [Memory-Tier Bandwidth Degradation](entries/memory-tier-degradation.md) | PLEX synthetic/reconstructed workload | extension | schedule, evict, prefetch, feedback |
| [Adapter Popularity Churn](entries/adapter-churn.md) | PLEX synthetic/reconstructed workload | extension | admit, route, schedule, evict, prefetch, rebalance |
| [Multimodal Encoder-State Residency](entries/multimodal-encoder-residency.md) | PLEX synthetic/reconstructed workload | extension | admit, route, schedule, evict, prefetch, feedback |
| [Multi-Region Failover, Cost, and Carbon](entries/multi-region-failover-carbon.md) | PLEX synthetic/reconstructed workload | extension | route, admit, rebalance, feedback |
| [Autoscaling with Stateful Cold Start](entries/autoscaling-coldstart.md) | PLEX synthetic/reconstructed workload | extension | admit, route, rebalance, feedback |
| [RAG Document Churn and Shared Retrieval](entries/rag-document-churn.md) | PLEX synthetic/reconstructed workload | extension | route, schedule, evict, prefetch, feedback |
| [Reasoning Progress and Marginal Utility](entries/reasoning-progress-utility.md) | PLEX synthetic/reconstructed workload | extension | admit, schedule, feedback |
| [Full-Duplex Voice and Interruption](entries/full-duplex-voice.md) | PLEX synthetic/reconstructed workload | extension | admit, route, schedule, feedback |
| [Correlated Session and Workflow Bursts](entries/correlated-session-bursts.md) | PLEX synthetic/reconstructed workload | core | admit, route, schedule, feedback, rebalance |
| [Tool Resource Lifecycle and Cross-Resource Scheduling](entries/tool-resource-lifecycle.md) | PLEX synthetic/reconstructed workload | extension | admit, schedule, feedback, prefetch |
