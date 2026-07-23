# PLEX serving-workload and dataset survey

Search cutoff: **2026-07-23**

This report surveys workloads used by existing LLM-serving papers and proposes
new workloads for validating policies that have not yet been studied.

- **Workload wiki:** [`plex-serving-workload-wiki/README.md`](plex-serving-workload-wiki/README.md)
- **Machine-readable catalog:** [`plex-serving-workload-wiki/catalog.json`](plex-serving-workload-wiki/catalog.json)
- **Methodology:** [`plex-serving-workload-wiki/methodology.md`](plex-serving-workload-wiki/methodology.md)
- **Minimum trace schema:** [`plex-serving-workload-wiki/trace-schema.md`](plex-serving-workload-wiki/trace-schema.md)
- **Paper-usage matrix:** [`plex-serving-workload-wiki/paper-usage.md`](plex-serving-workload-wiki/paper-usage.md)

The catalog contains **42 existing public sources** and **21 proposed PLEX
workload families**.

## Executive findings

1. There is no public dataset that combines realistic arrivals, logical-request
   continuity, workflow dependencies, tools, SLOs, tenants, cache lineage, and
   enacted outcomes.
2. Production traces and application benchmarks are complementary, not
   substitutes:
   - Azure, BurstGPT, Bailian, and Mooncake expose traffic shape.
   - SWE-bench, tau3-bench, OSWorld, and WebArena expose application semantics.
3. ShareGPT plus Poisson arrivals remains common but is now a weak default:
   it removes production burstiness, model mix, tenant behavior, tool pauses,
   prefix popularity shifts, cancellation, and workflow correlation.
4. Qwen-Bailian is the strongest current public trace for cache-aware policy
   evaluation because it combines timestamps, session lineage, turns, request
   type, token lengths, and block hashes.
5. ServeGen is the best public generator for production-like arrival and length
   distributions, but it still lacks explicit agent DAGs, tool lifecycle, SLOs,
   cancellations, and declared-versus-observed metadata.
6. Agent benchmarks become serving workloads only after instrumentation. Their
   task files alone do not contain model-call timing, cache lineage, or request
   concurrency.
7. Novel PLEX policies should be validated with **composed workloads**:
   production traffic envelope + application trajectory + policy annotations +
   measured engine outcomes.

## Coverage of current public sources

Among the 42 surveyed public sources:

| Dimension | Sources with any coverage |
|---|---:|
| Arrival timing or generated arrivals | 13 |
| Input/output token lengths | 11 |
| Prompt/response/task content | 32 |
| Session or logical-request identity | 21 |
| Tenant/principal identity | 3 |
| Prefix/cache lineage | 10 |
| Tool interaction | 14 |
| Workflow graph or branch structure | 12 |
| SLO or priority | 1 |
| Failure, cancellation, retry, or quality failure | 21 |
| Multimodal payload | 7 |
| Model or adapter identity | 20 |
| Hardware/topology state | 5 |

Important intersections are even rarer:

| Required combination | Public sources |
|---|---:|
| Arrival + session | 5 |
| Arrival + session + prefix | 3 |
| Arrival + workflow DAG | **0** |
| Tenant + SLO | **0** |
| Session + tool + DAG | 8, but none include production arrivals |
| Arrival + failure + session | 2 |

This explains why existing papers can reproduce individual point policies but
cannot fairly compare coordinated lifecycle policies.

## What existing papers actually use

The initial 87-paper metadata audit recovered a named dataset or workload for
only 34 papers. The remainder commonly rely on a paper-specific synthetic
generator, a private production trace, or an incompletely documented
transformation of ShareGPT/Azure-style length distributions.

### Production request traces

| Source | Useful fields | Main limitation |
|---|---|---|
| [Azure LLM 2023](plex-serving-workload-wiki/entries/azure-llm-2023.md) | Timestamp, input tokens, output tokens | No session, content, prefix, tenant, SLO, or outcome |
| [Azure LLM 2024](plex-serving-workload-wiki/entries/azure-llm-2024.md) | One-week production arrivals and lengths | Same sparse three-column schema |
| [BurstGPT v2](plex-serving-workload-wiki/entries/burstgpt-v2.md) | Long duration, sessions, model, lengths, elapsed time, failures | No cache, tools, TTFT/TPOT, tenant, or SLO |
| [Qwen-Bailian](plex-serving-workload-wiki/entries/qwen-bailian.md) | Timestamp, parent/session, turn, type, lengths, block hashes | No tool duration/outcome, tenant, SLO, cancel |
| [Mooncake FAST'25](plex-serving-workload-wiki/entries/mooncake-fast25.md) | Real conversation/tool-agent arrivals, lengths, block hashes | Session and tool lifecycle are implicit |
| [ServeGen](plex-serving-workload-wiki/entries/servegen.md) | Production-fitted burst, drift, reasoning, multimodal generation | Generated; no explicit tool/DAG/SLO/failure |

### Conversation corpora

[LMSYS-Chat-1M](plex-serving-workload-wiki/entries/lmsys-chat-1m.md),
[WildChat-1M](plex-serving-workload-wiki/entries/wildchat-1m.md), and
[ShareGPT](plex-serving-workload-wiki/entries/sharegpt.md) provide real content
and multi-turn structure. They are useful for reconstructing token blocks,
prefix overlap, languages, safety classes, and model-routing queries. They are
not production serving traces.

WildChat is the most useful of the three for reconstruction because it includes
per-turn timestamps, model identity, a hashed user identifier, and conversation
metadata. LMSYS-Chat-1M has stronger scale and model diversity but restrictive
redistribution terms. ShareGPT remains common for historical comparability but
has weaker provenance and workload metadata.

### Agent and workflow benchmarks

| Source | Serving value |
|---|---|
| [SWE-bench Verified](plex-serving-workload-wiki/entries/swe-bench-verified.md) + [mini-SWE-agent](plex-serving-workload-wiki/entries/mini-swe-agent.md) | Long tool-interleaved sessions, context growth, objective success, easy model-call instrumentation |
| [BFCL V4](plex-serving-workload-wiki/entries/bfcl-v4.md) | Multi-turn function calls, tool schemas, released responses, cost/latency measurements |
| [tau3-bench](plex-serving-workload-wiki/entries/tau3-bench.md) | Stateful users/tools, policy constraints, RAG, banking, and full-duplex voice |
| [ToolBench](plex-serving-workload-wiki/entries/toolbench.md) | 16K APIs, multi-tool decision trees, execution results |
| [Terminal-Bench 2.0](plex-serving-workload-wiki/entries/terminal-bench-2.md) | Long executable terminal tasks and objective tests |
| [AgentBench](plex-serving-workload-wiki/entries/agentbench.md) | Eight diverse interactive environments |
| [OSWorld-Verified](plex-serving-workload-wiki/entries/osworld-verified.md) | Multimodal GUI actions, screenshots, videos, and long trajectories |
| [WebArena](plex-serving-workload-wiki/entries/webarena.md) | Reproducible web state and released human/agent trajectories |
| [ScienceAgentBench](plex-serving-workload-wiki/entries/scienceagentbench.md) | Authentic scientific code/data workloads and execution costs |
| [TheAgentCompany](plex-serving-workload-wiki/entries/theagentcompany.md) | Multi-application enterprise tasks and coworker communication |
| [LongMemEval](plex-serving-workload-wiki/entries/longmemeval.md) | Timestamped long-lived sessions, knowledge updates, temporal reasoning |

These sources need an instrumentation adapter that records each LLM call,
tool boundary, token count, cache lineage, and outcome in the PLEX trace schema.

### Private and paper-specific workloads

Several influential results cannot be independently replayed from a public
trace:

| Paper/workload | Evidence available | Reproducibility limitation |
|---|---|---|
| FairServe | Millions of Microsoft Copilot requests across 34 applications | Production trace is not public |
| Continuum | TensorMesh internal agent-serving testbed plus public agent tasks | Internal arrival/tool-duration trace is not public |
| Pythia | Production multi-agent coding-assistant trace | Workflow and burst trace is private |
| SMetric | Two large production agent-serving traces, including Bailian | Full production data is not public; only related anonymized traces are released |
| GoodServe | Heterogeneous-GPU testbed and large simulation | Agent demand/resource predictions are paper-specific |
| Helium / KVFlow | Workflow DAGs and cache-aware schedules | Workload graph construction is tied to each artifact |
| Justitia / Autellix | Task-parallel and program-level synthetic workloads | Program mixtures and arrivals are generated rather than production traces |

These papers remain useful policy sources, but source-policy parity should be
reported separately from workload parity.

### Reasoning, RAG, long context, multimodal, and routing

- [Dynasor/Math500](plex-serving-workload-wiki/entries/dynasor-math500.md)
  provides reasoning progress and early-stop experiments.
- [LiveCodeBench](plex-serving-workload-wiki/entries/livecodebench.md) provides
  temporal holdouts and executable correctness.
- [LongBench v2](plex-serving-workload-wiki/entries/longbench-v2.md) and
  [InfiniteBench](plex-serving-workload-wiki/entries/infinitebench.md) stress
  admission, prefill, residency, and long-context scheduling.
- [BEIR](plex-serving-workload-wiki/entries/beir.md),
  [RAGBench](plex-serving-workload-wiki/entries/ragbench.md), and
  [HotpotQA](plex-serving-workload-wiki/entries/hotpotqa.md) support document
  overlap, retrieval fanout, and quality-aware cache experiments.
- [MMMU](plex-serving-workload-wiki/entries/mmmu.md) and
  [Video-MME](plex-serving-workload-wiki/entries/video-mme.md) add heterogeneous
  image/video/audio encoder work and reusable encoder state.
- [RouterBench](plex-serving-workload-wiki/entries/routerbench.md) and
  [RouteLLM](plex-serving-workload-wiki/entries/routellm-arena.md) provide
  counterfactual model quality/cost, but not replica load or cache state.
- [S-LoRA's synthetic workload](plex-serving-workload-wiki/entries/slora-synthetic.md)
  remains necessary because no public production multi-adapter request trace
  was found.

## Recommended PLEXBench design

### Tier 1: Real traffic replay

Use at least four complementary traffic sources:

1. Azure LLM 2024 for week-scale arrivals and token lengths.
2. BurstGPT for periodicity, sessions, model mix, elapsed time, and failures.
3. Qwen-Bailian for parent/turn structure and block-level cache reuse.
4. Mooncake ToolAgent for agent-like prefix reuse and long prompts.

Do not replace their timestamps with Poisson arrivals except in a controlled
ablation.

### Tier 2: Instrumented application replay

Run application benchmarks through an instrumented client:

1. mini-SWE-agent on SWE-bench Verified.
2. tau3-bench text, banking/RAG, and voice.
3. Terminal-Bench 2.0.
4. WebArena or OSWorld.
5. Dynasor on Math500.
6. LongBench v2 and BEIR/RAGBench.
7. MMMU or Video-MME.

Replay the resulting model-call/tool-event trace independently from task
execution so policy experiments are repeatable.

### Tier 3: Factorial synthetic stress

Generate a balanced design over:

- arrival process: Poisson, empirical replay, self-similar, periodic, flash;
- input/output joint distribution;
- session depth and context growth;
- prefix overlap graph and popularity drift;
- tool duration, failure, and retry;
- DAG shape and online revelation;
- tenant weights and SLO classes;
- model/adapter/modality mix;
- memory pressure and P/D topology;
- network, tier bandwidth, and region state.

Every factor should include a default production-fitted setting and explicit
extreme settings.

### Tier 4: Adversarial and failure workloads

The following are required to validate new policies:

| Workload | Main PLEX operations |
|---|---|
| [Dishonest metadata](plex-serving-workload-wiki/entries/dishonest-policy-hints.md) | admit, route, schedule, evict, feedback |
| [Continuation storm](plex-serving-workload-wiki/entries/continuation-storm.md) | route, schedule, evict, feedback |
| [Speculative fanout/cancel](plex-serving-workload-wiki/entries/speculative-fanout-cancel.md) | admit, schedule, evict, feedback |
| [Prefix popularity shift](plex-serving-workload-wiki/entries/prefix-hotspot-shift.md) | route, evict, prefetch, rebalance |
| [Session abandonment/retry](plex-serving-workload-wiki/entries/session-abandonment-retry.md) | admit, schedule, evict, feedback |
| [Memory-tier degradation](plex-serving-workload-wiki/entries/memory-tier-degradation.md) | schedule, evict, prefetch, feedback |
| [Multi-region failover](plex-serving-workload-wiki/entries/multi-region-failover-carbon.md) | route, admit, rebalance, feedback |

### Tier 5: Temporal holdout

Freeze workload generators and policies on pre-2026 sources. Use ServeGen,
tau3-bench voice/knowledge, LongMemEval-V2, and later production/agent traces as
holdouts. This tests generalization instead of retuning every policy to every
trace.

## New workload families

The catalog defines 21 workloads not jointly represented in prior public data.
The highest-priority set is:

1. [Logical continuation mix](plex-serving-workload-wiki/entries/logical-continuation-mix.md)
2. [Continuation storm](plex-serving-workload-wiki/entries/continuation-storm.md)
3. [Heavy-tailed tool latency/failure](plex-serving-workload-wiki/entries/tool-heavy-tail-failure.md)
4. [Online-revealed agent DAG](plex-serving-workload-wiki/entries/dynamic-agent-dag.md)
5. [Speculative fanout with cancellation](plex-serving-workload-wiki/entries/speculative-fanout-cancel.md)
6. [Fairness versus locality](plex-serving-workload-wiki/entries/fair-locality-conflict.md)
7. [Dishonest policy hints](plex-serving-workload-wiki/entries/dishonest-policy-hints.md)
8. [Mixed service classes](plex-serving-workload-wiki/entries/mixed-service-classes.md)
9. [P/D append-prefill imbalance](plex-serving-workload-wiki/entries/pd-append-prefill-imbalance.md)
10. [Correlated session bursts](plex-serving-workload-wiki/entries/correlated-session-bursts.md)

## Minimum trace contract

A useful workload must preserve:

- tenant/principal, logical request, generation, workflow, and node identity;
- lifecycle events: create, continue, boundary, progress, preempt, cancel,
  finish;
- model, adapter, modality, stage, token lengths, and prefix block hashes;
- caller-declared versus host-observed values;
- SLO/priority and final task quality or success;
- tool/retrieval events, durations, outcomes, and retries;
- route, schedule, cache, migration, and feedback outcomes.

Raw text is optional. Salted block hashes and stable anonymized identities can
preserve policy-relevant behavior without releasing user content.

## Metrics

Every policy/workload pair should report:

1. workflow/task completion latency, not only request latency;
2. SLO goodput by class;
3. TTFT, TPOT/TBT, and queueing;
4. attained-service fairness and victim-tenant p99;
5. cache hit rate, recomputed tokens, and resident byte-time;
6. preemptions, migrations, retries, cancellations, and wasted tokens;
7. tool wait overlap and continuation delay;
8. task quality or correctness when policy changes compute/model choice;
9. cost, energy, and transfer bytes where applicable;
10. policy overhead, fallback rate, and state conflicts.

## Experimental protocol

1. Replay identical event streams for every policy.
2. Separate workload randomness from policy randomness.
3. Run low, knee, saturation, and overload intensity levels.
4. Preserve empirical timestamp correlation and input/output dependence.
5. Report workload-factor sensitivity, not one tuned operating point.
6. Compare source policy, PLEX replica, engine default, and oracle where
   possible.
7. Publish the transformed trace and seed so every result is reproducible.

The objective is not to identify one universal policy. It is to demonstrate
that different policy hypotheses win on different, explicitly characterized
workloads while the PLEX contract remains unchanged.
