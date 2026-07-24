# PLEX v0.6 Replication and Fidelity Report

This report is generated from committed replication metadata by
`scripts/generate-plex-replication-report.py`.

- Candidates: 31
- Runtime smoke passing: 31
- Contract: `0.6`

## Evidence

- `inspired-adaptation`: 31

## Independent fidelity

- `incorrect`: 14
- `material-semantic-gap`: 17

## Candidates

| ID | Title | Operations | Evidence | Fidelity | Deferred mechanics |
|---|---|---|---|---|---:|
| `agentix` | Agentix: An Efficient Serving Engine for LLM Agents as General Programs | `schedule`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 2 |
| `continuum` | Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live | `schedule`, `cache`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 2 |
| `kvflow` | KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows | `schedule`, `cache`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 1 |
| `preble` | Preble: Efficient Distributed Prompt Scheduling for LLM Serving | `route`, `schedule`, `cache`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 1 |
| `helium` | Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective | `schedule` | `inspired-adaptation` | `material-semantic-gap` | 2 |
| `vtc` | Fairness in Serving Large Language Models | `schedule`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 0 |
| `lmetric` | Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling | `route` | `inspired-adaptation` | `material-semantic-gap` | 0 |
| `fairserve` | Ensuring Fair LLM Serving Amid Diverse Applications | `admit`, `schedule`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `marconi` | Marconi: Prefix Caching for the Era of Hybrid LLMs | `cache`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `ragcache` | RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation | `schedule`, `cache`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 0 |
| `dlpm` | Locality-aware Fair Scheduling in LLM Serving | `route`, `schedule`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 0 |
| `infercept` | InferCept: Efficient Intercept Support for Augmented Large Language Model Inference | `schedule`, `cache` | `inspired-adaptation` | `incorrect` | 1 |
| `peek` | PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving | `schedule`, `cache` | `inspired-adaptation` | `material-semantic-gap` | 0 |
| `qlm` | Queue Management for SLO-Oriented Large Language Model Serving | `admit`, `route`, `schedule`, `feedback` | `inspired-adaptation` | `incorrect` | 0 |
| `slos-serve` | SLOs-Serve: Optimized Serving of Multi-SLO LLMs | `admit`, `route`, `schedule` | `inspired-adaptation` | `incorrect` | 1 |
| `dynasor` | Efficiently Scaling LLM Reasoning with Certaindex | `schedule`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 1 |
| `justitia` | Justitia: Fair and Efficient Scheduling of Task-parallel LLM Agents with Selective Pampering | `schedule`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `chameleon` | Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments | `admit`, `schedule`, `cache` | `inspired-adaptation` | `material-semantic-gap` | 0 |
| `hotprefix` | HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems | `cache`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `pard` | PARD: Enhancing Goodput for Inference Pipeline via Proactive Request Dropping | `schedule`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `branch-regulation` | Regulating Branch Parallelism in LLM Serving | `admit`, `schedule` | `inspired-adaptation` | `incorrect` | 1 |
| `dualmap` | DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving | `route` | `inspired-adaptation` | `material-semantic-gap` | 1 |
| `llumnix` | Llumnix: Dynamic Scheduling for Large Language Model Serving | `route`, `feedback` | `inspired-adaptation` | `material-semantic-gap` | 1 |
| `smetric` | SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling | `route` | `inspired-adaptation` | `material-semantic-gap` | 1 |
| `thunderagent` | ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System | `schedule`, `cache`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `pythia` | Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving | `route`, `schedule`, `cache`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `goodserve` | GoodServe: Towards High-Goodput Serving of Agentic LLM Inferences over Heterogeneous Resources | `route`, `feedback` | `inspired-adaptation` | `incorrect` | 2 |
| `conserve` | Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving | `route` | `inspired-adaptation` | `material-semantic-gap` | 0 |
| `parrot` | Parrot: Efficient Serving of LLM-based Applications with Semantic Variable | `route`, `schedule` | `inspired-adaptation` | `material-semantic-gap` | 1 |
| `saga` | SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters | `route`, `schedule`, `cache`, `feedback` | `inspired-adaptation` | `incorrect` | 1 |
| `routebalance` | RouteBalance: Fused Model Routing and Load Balancing for Heterogeneous LLM Serving | `route` | `inspired-adaptation` | `incorrect` | 1 |

Runtime smoke means the package loads and its committed fixture path is
covered by the release suite. It is not a claim of paper fidelity.
Independent reviewers found no faithful or faithful-with-deferred-mechanics
implementation; all entries are therefore classified as inspired
adaptations until paper/artifact differential traces pass.
