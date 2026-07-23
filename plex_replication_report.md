# PLEX v0.6 Replication Report

This report is generated from committed replication metadata by
`scripts/generate-plex-replication-report.py`.

- Candidates: 31
- Passing: 31
- Contract: `0.6`

## Evidence

- `policy-kernel-reproduction`: 31

## Candidates

| ID | Title | Operations | Evidence | Deferred mechanics |
|---|---|---|---|---:|
| `agentix` | Agentix: An Efficient Serving Engine for LLM Agents as General Programs | `schedule`, `feedback` | `policy-kernel-reproduction` | 3 |
| `continuum` | Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live | `schedule`, `cache`, `feedback` | `policy-kernel-reproduction` | 3 |
| `kvflow` | KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows | `schedule`, `cache` | `policy-kernel-reproduction` | 1 |
| `preble` | Preble: Efficient Distributed Prompt Scheduling for LLM Serving | `route` | `policy-kernel-reproduction` | 2 |
| `helium` | Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective | `schedule` | `policy-kernel-reproduction` | 2 |
| `vtc` | Fairness in Serving Large Language Models | `schedule`, `feedback` | `policy-kernel-reproduction` | 0 |
| `lmetric` | Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling | `route` | `policy-kernel-reproduction` | 0 |
| `fairserve` | Ensuring Fair LLM Serving Amid Diverse Applications | `admit`, `schedule`, `feedback` | `policy-kernel-reproduction` | 1 |
| `marconi` | Marconi: Prefix Caching for the Era of Hybrid LLMs | `cache`, `feedback` | `policy-kernel-reproduction` | 1 |
| `ragcache` | RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation | `cache` | `policy-kernel-reproduction` | 0 |
| `dlpm` | Locality-aware Fair Scheduling in LLM Serving | `route`, `schedule`, `feedback` | `policy-kernel-reproduction` | 0 |
| `infercept` | InferCept: Efficient Intercept Support for Augmented Large Language Model Inference | `schedule`, `cache` | `policy-kernel-reproduction` | 1 |
| `peek` | PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving | `schedule`, `cache` | `policy-kernel-reproduction` | 0 |
| `qlm` | Queue Management for SLO-Oriented Large Language Model Serving | `admit`, `route`, `schedule`, `feedback` | `policy-kernel-reproduction` | 0 |
| `slos-serve` | SLOs-Serve: Optimized Serving of Multi-SLO LLMs | `admit`, `route`, `schedule` | `policy-kernel-reproduction` | 1 |
| `dynasor` | Efficiently Scaling LLM Reasoning with Certaindex | `schedule`, `feedback` | `policy-kernel-reproduction` | 1 |
| `justitia` | Justitia: Fair and Efficient Scheduling of Task-parallel LLM Agents with Selective Pampering | `schedule`, `feedback` | `policy-kernel-reproduction` | 1 |
| `chameleon` | Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments | `admit`, `schedule`, `cache` | `policy-kernel-reproduction` | 0 |
| `hotprefix` | HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems | `cache`, `feedback` | `policy-kernel-reproduction` | 1 |
| `pard` | PARD: Enhancing Goodput for Inference Pipeline via Proactive Request Dropping | `schedule`, `feedback` | `policy-kernel-reproduction` | 1 |
| `branch-regulation` | Regulating Branch Parallelism in LLM Serving | `admit`, `schedule` | `policy-kernel-reproduction` | 1 |
| `dualmap` | DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving | `route` | `policy-kernel-reproduction` | 1 |
| `llumnix` | Llumnix: Dynamic Scheduling for Large Language Model Serving | `route`, `feedback` | `policy-kernel-reproduction` | 1 |
| `smetric` | SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling | `route` | `policy-kernel-reproduction` | 1 |
| `thunderagent` | ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System | `schedule`, `cache`, `feedback` | `policy-kernel-reproduction` | 1 |
| `pythia` | Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving | `route`, `schedule`, `cache`, `feedback` | `policy-kernel-reproduction` | 1 |
| `goodserve` | GoodServe: Towards High-Goodput Serving of Agentic LLM Inferences over Heterogeneous Resources | `route`, `feedback` | `policy-kernel-reproduction` | 2 |
| `conserve` | Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving | `route` | `policy-kernel-reproduction` | 0 |
| `parrot` | Parrot: Efficient Serving of LLM-based Applications with Semantic Variable | `route`, `schedule` | `policy-kernel-reproduction` | 1 |
| `saga` | SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters | `route`, `schedule`, `cache` | `policy-kernel-reproduction` | 1 |
| `routebalance` | RouteBalance: Fused Model Routing and Load Balancing for Heterogeneous LLM Serving | `route` | `policy-kernel-reproduction` | 1 |

Evidence levels follow `plex_0.6.md`. Physical movement, provisioning,
predictor training, and other deferred mechanics are not counted as
replicated behavior.
