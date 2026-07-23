# PLEX serving-policy literature report

Search cutoff: **2026-07-23**

This report identifies serving-policy papers that are useful as PLEX examples,
source-policy replications, comparison baselines, or API-boundary tests.

- **Per-paper wiki:** [`plex-serving-policy-wiki/README.md`](plex-serving-policy-wiki/README.md)
- **Machine-readable catalog:** [`plex-serving-policy-wiki/catalog.json`](plex-serving-policy-wiki/catalog.json)
- **Methodology and limitations:** [`plex-serving-policy-wiki/methodology.md`](plex-serving-policy-wiki/methodology.md)

The corpus contains **87 unique papers**:

| Area | Papers |
|---|---:|
| Agent, workflow, and multi-turn serving | 17 |
| Scheduling, fairness, SLOs, and admission | 24 |
| Routing, placement, and rebalancing | 21 |
| Residency, cache admission, eviction, and prefetch | 13 |
| Foundational pre-LLM serving policies | 12 |

## Executive findings

1. The current five PLEX replicas are useful but do not cover `admit` or
   `rebalance`.
2. The canonical serving-fairness baseline, VTC, is missing.
3. PEEK is the strongest compact composition example because one shared
   pending-demand structure controls both `schedule` and `evict`.
4. LMetric is the cleanest modern routing example: its core policy is one
   multiplication with an explicit hotspot guard.
5. FairServe and SLOs-Serve provide concrete admission policies instead of
   treating admission as a structural hook without a source replication.
6. Llumnix, DualMap, and SMetric justify keeping `rebalance` explicitly
   auxiliary rather than pretending migration is universally available.
7. KVFlow, HotPrefix, PRESERVE, and ECHO similarly justify an explicit optional
   `prefetch` operation.
8. The current draft's “bundle fairness” claim does not match the Helium
   fixture. Helium is cache-aware critical-path scheduling. Certaindex/Dynasor
   or Justitia is a more defensible task/bundle-level policy.

## Audit of the current five replicas

| Fixture | Keep? | What it demonstrates | Required correction |
|---|---|---|---|
| [Agentix / Autellix](plex-serving-policy-wiki/papers/autellix-agentix.md) | Yes | Program-level attained-service scheduling (`schedule+feedback`) | Treat Autellix and final Agentix as one paper lineage. |
| [Continuum](plex-serving-policy-wiki/papers/continuum.md) | Yes | Tool-aware KV TTL with program FCFS (`schedule+evict+feedback`) | Compare explicitly against InferCept's preserve/swap/discard source policy. |
| [KVFlow](plex-serving-policy-wiki/papers/kvflow.md) | Yes | Workflow-aware retention and prefetch intent (`evict+prefetch`) | State that fully overlapped prefetch remains engine mechanics unless reproduced. |
| [Preble](plex-serving-policy-wiki/papers/preble.md) | Yes | Prefix-locality versus load routing (`route`) | Describe the fixture as an E2 policy-kernel replica, not full autoscaling/replication. |
| [Helium](plex-serving-policy-wiki/papers/helium.md) | Yes, relabeled | Cache-aware critical-path workflow scheduling (`schedule`, optional `prefetch`) | Remove the “bundle fairness” label and associated fairness-number placeholder. |

The unchecked `fu2024efficient` citation in `plex_paper.md` must be resolved
before claiming source parity. If it denotes
[Certaindex/Dynasor](plex-serving-policy-wiki/papers/certaindex-dynasor.md),
that paper supplies a plausible progress-aware reasoning-program example.

## Recommended replication set

### Core additions

| Priority | Paper | PLEX surface | Compact policy to reproduce | Artifact |
|---:|---|---|---|---|
| 1 | [VTC](plex-serving-policy-wiki/papers/vtc.md) | `schedule+feedback` | Select the client with minimum virtual-token counter and charge observed input/output-token service. | Public |
| 2 | [FairServe](plex-serving-policy-wiki/papers/fairserve.md) | `admit+schedule+feedback` | Overload/interaction-aware throttling plus weighted service counters. | Not confirmed |
| 3 | [DLPM / D²LPM](plex-serving-policy-wiki/papers/dlpm-d2lpm.md) | `route+schedule+feedback` | Longest-prefix order gated by per-client deficit credits. | Not confirmed |
| 4 | [InferCept](plex-serving-policy-wiki/papers/infercept.md) | `schedule+evict+feedback` | At an API/tool boundary, choose preserve, swap, or discard by expected waste. | Public |
| 5 | [PEEK](plex-serving-policy-wiki/papers/peek.md) | `schedule+evict` | Cluster-aware LPM, fairness lane, and demand-depth retention score over one pending tree. | Public |
| 6 | [LMetric](plex-serving-policy-wiki/papers/lmetric.md) | `route` | Minimize `new_prefill_tokens × current_batch_size`, with hotspot detection. | Public |
| 7 | [DualMap](plex-serving-policy-wiki/papers/dualmap.md) | `route+rebalance` | Two prefix hashes, SLO-aware target choice, and hotspot migration. | Public |
| 8 | [QLM](plex-serving-policy-wiki/papers/qlm.md) | `admit+route+schedule+feedback` | Request groups, waiting-time estimation, virtual queues, and queue operations. | Not confirmed |

This eight-paper set covers every core operation except that `feedback` is used
as observation/state maintenance rather than an independent optimization goal.

### Extended additions

| Paper | Why add it |
|---|---|
| [Llumnix](plex-serving-policy-wiki/papers/llumnix.md) | Canonical live-migration and rebalancing policy; separates virtual-usage policy from migration mechanics. |
| [SLOs-Serve](plex-serving-policy-wiki/papers/slos-serve.md) | Strong source for SLO-based soft admission, token allocation, and multi-replica routing. |
| [Certaindex/Dynasor](plex-serving-policy-wiki/papers/certaindex-dynasor.md) | Progress-aware allocation and early termination for reasoning programs; best replacement for ambiguous bundle fairness. |
| [Justitia](plex-serving-policy-wiki/papers/justitia.md) | Fair-completion-order scheduling for task-parallel agents. |
| [RAGCache](plex-serving-policy-wiki/papers/ragcache.md) | Exceptionally compact eviction formula using recency, frequency, recompute cost, and size. |
| [Chameleon](plex-serving-policy-wiki/papers/chameleon.md) | Exercises admission, scheduling, adapter residency, heterogeneous request size, and starvation control. |

## Suggested evaluation matrix

| Evaluation question | Policies |
|---|---|
| Can PLEX reproduce established point policies? | VTC, Preble, InferCept, RAGCache, LMetric |
| Can one binary coordinate multiple lifecycle operations? | Continuum, PEEK, Pythia |
| Does the surface cover admission? | FairServe, SLOs-Serve, QLM |
| Does the optional surface cover active movement? | KVFlow (`prefetch`), DualMap/Llumnix (`rebalance`) |
| Does logical-request state matter across turns? | Agentix, Continuum, InferCept, ConServe, SMetric |
| Can PLEX preserve fairness while exploiting locality? | VTC, DLPM/D²LPM, PEEK |
| Is the surface stable on later work? | 2026 temporal holdout below |

## Temporal holdout

Freeze the core PLEX operation set using papers available before 2026, then map
the following later papers without adding a core hook:

- [PEEK](plex-serving-policy-wiki/papers/peek.md)
- [LMetric](plex-serving-policy-wiki/papers/lmetric.md)
- [DualMap](plex-serving-policy-wiki/papers/dualmap.md)
- [SMetric](plex-serving-policy-wiki/papers/smetric.md)
- [ThunderAgent](plex-serving-policy-wiki/papers/thunderagent.md)
- [Pythia](plex-serving-policy-wiki/papers/pythia.md)
- [GoodServe](plex-serving-policy-wiki/papers/goodserve.md)
- [Conversation-level ConServe](plex-serving-policy-wiki/papers/observation-not-prediction-conserve.md)

This is stronger evidence for a stable waist than retrospectively fitting only
the five hand-selected source policies.

## Surface stress tests

### Cache admission

[Marconi](plex-serving-policy-wiki/papers/marconi.md) and
[HotPrefix](plex-serving-policy-wiki/papers/hotprefix.md) decide whether a newly
created cache entry should become resident. PLEX `evict` currently ranks
already-resident units. A semantics-preserving implementation must either:

1. expose insertion as a residency arbitration cause with the prospective entry
   included among candidates; or
2. express cache admission through a bounded retention-intent map.

Coverage should remain marked partial until one path is implemented.

### Mid-lifecycle dropping

[PARD](plex-serving-policy-wiki/papers/pard.md) and
[Regulating Branch Parallelism](plex-serving-policy-wiki/papers/regulating-branch-parallelism-in-llm-serving.md)
can cancel work after initial admission. This is not naturally `admit`-once.
The narrow options are a host cancellation action emitted from `schedule`, or
classifying these policies as residual. A new core hook is not justified by
trigger timing alone.

### Non-portable control-plane mechanics

Autoscaling, DVFS, model/verifier selection, and live state movement require
deployment capabilities beyond the five core operations. Papers such as KAIROS,
PolyServe, Pythia, Aladdin, and Dyserve belong in the corpus but must not be
used to imply that every backend can enact those choices.

## Replication and parity rules

1. Pin the source paper, artifact commit, and engine version.
2. Separate the policy kernel from source-only mechanics.
3. Replay identical candidate snapshots through the source and PLEX versions.
4. Compare dense scores, chosen candidates, token budgets, actions, and state
   transitions.
5. Report missing facts/capabilities rather than coercing semantics.
6. Claim end-to-end parity only when the source mechanics are also reproduced.
7. Otherwise report **decision-trace parity with deferred mechanics**.

## Metadata caveats

Each wiki page contains the paper link, venue/status, authors and group context,
citation count with source/date, artifact, datasets/workloads, editorial
abstract synopsis, policy summary, and PLEX mapping. Citation counts are
discovery signals: arXiv and proceedings records can split totals, and recent
2026 papers naturally have low counts. “Reputation evidence” reports venue,
affiliations, and artifact availability; it is not an endorsement.
