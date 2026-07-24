# PLEX 31-policy reproducibility roadmap

This roadmap separates three causes of missing fidelity:

1. **Current-model improvement**: rewrite the policy, facts, state loop,
   adapter, or benchmark without changing the PLEX contract.
2. **New primitive required**: add authority, atomicity, lifecycle, or an
   enacted mechanism that the current contract cannot safely express.
3. **Implementation gap**: a bug, simplification, missing state transition,
   wrong equation, or non-diagnostic benchmark.

- Policies: 31
- Policies needing no new primitive for their best current-model evidence: 17
- Deduplicated proposed primitives: 11

## Evidence ceiling with the current programming model

- `decision-trace-parity`: 17
- `faithful`: 13
- `inspired-only`: 1

## Primitive roadmap

| Primitive | Kind | Policies | Minimal contract |
|---|---|---:|---|
| `request.pause-resume@1` | action-and-lifecycle | 4 | request pause, preserve, resume or discard with enacted status and preserved-state identity |
| `target.provision@1` | control-plane-action | 4 | load or unload a model, create or remove a replica, resize capacity and report readiness/failure |
| `schedule.co-execute@1` | guarantee | 3 | all requests in a selection begin the same execution step or the selection is not enacted |
| `cache.move@1` | multi-phase-action | 2 | reserve destination tier, transfer object or range, commit placement, release source, and report actual bytes/latency |
| `request.migrate@1` | multi-phase-action | 2 | prepare destination, reserve resources, copy incremental state, commit or abort, and report downtime/bytes |
| `branch.defer-retain@1` | schedule-lifecycle-guarantee | 1 | defer a branch without terminating its parent, retain its execution and KV state, and reconsider it at a later step |
| `decode.speculate@1` | guarantee-and-action | 1 | select speculative token budget or mode, verify engine support, and report accepted/rejected tokens and latency |
| `request.complete-with-output@1` | action | 1 | terminate a reasoning request while committing a validated stable answer or aggregate result |
| `request.phase-handoff@1` | multi-phase-action | 1 | atomically transfer request ownership and state between prefill and decode roles, then persist the selected destination binding |
| `tool.resource@1` | action-and-lifecycle | 1 | prepare, acquire, release and fail a named tool/runtime resource with request or group ownership |
| `workflow.graph@1` | typed-context-guarantee | 1 | validated nodes, edges, AND/OR semantics, readiness, task groups and lineage revisions |

## Per-policy summary

| Policy | Current-model improvements | New primitives | P0 gaps | Current-model ceiling |
|---|---:|---|---:|---|
| `agentix` | 2 | - | 2 | `faithful` |
| `continuum` | 2 | - | 2 | `faithful` |
| `kvflow` | 2 | - | 0 | `faithful` |
| `preble` | 2 | - | 0 | `decision-trace-parity` |
| `helium` | 1 | - | 1 | `faithful` |
| `vtc` | 2 | - | 1 | `faithful` |
| `lmetric` | 2 | - | 1 | `faithful` |
| `fairserve` | 2 | - | 2 | `faithful` |
| `marconi` | 3 | - | 2 | `decision-trace-parity` |
| `ragcache` | 3 | - | 2 | `decision-trace-parity` |
| `dlpm` | 2 | - | 3 | `faithful` |
| `infercept` | 3 | `cache.move@1` | 3 | `decision-trace-parity` |
| `peek` | 3 | - | 3 | `faithful` |
| `qlm` | 3 | `request.pause-resume@1`, `target.provision@1` | 2 | `decision-trace-parity` |
| `slos-serve` | 2 | `schedule.co-execute@1`, `decode.speculate@1`, `request.pause-resume@1` | 2 | `decision-trace-parity` |
| `dynasor` | 2 | `request.complete-with-output@1`, `schedule.co-execute@1` | 3 | `decision-trace-parity` |
| `justitia` | 2 | - | 2 | `faithful` |
| `chameleon` | 3 | `request.pause-resume@1` | 3 | `decision-trace-parity` |
| `hotprefix` | 3 | `cache.move@1` | 2 | `decision-trace-parity` |
| `pard` | 3 | - | 3 | `faithful` |
| `branch-regulation` | 3 | `schedule.co-execute@1`, `branch.defer-retain@1` | 3 | `decision-trace-parity` |
| `dualmap` | 2 | `target.provision@1` | 1 | `decision-trace-parity` |
| `llumnix` | 2 | `request.migrate@1`, `target.provision@1` | 2 | `decision-trace-parity` |
| `smetric` | 2 | - | 0 | `decision-trace-parity` |
| `thunderagent` | 2 | `request.pause-resume@1`, `tool.resource@1` | 2 | `inspired-only` |
| `pythia` | 2 | `target.provision@1` | 4 | `decision-trace-parity` |
| `goodserve` | 2 | `request.migrate@1` | 1 | `decision-trace-parity` |
| `conserve` | 1 | `request.phase-handoff@1` | 2 | `decision-trace-parity` |
| `parrot` | 1 | `workflow.graph@1` | 3 | `decision-trace-parity` |
| `saga` | 3 | - | 3 | `faithful` |
| `routebalance` | 2 | - | 2 | `faithful` |

## agentix

### Improve with the current model

- **Waiting and service used different units.** — waiting_ms is converted to microseconds before starvation comparison; the benchmark oracle matches. (Typed schedule facts and WorkGroup scratch already carry wait and aggregate service, so no new authority was needed.)
- **Replication evidence and headline claim were overstated.** — Metadata now labels the kernel inspired-adaptation and targets only the verified 4-15x equal-latency claim. (Evidence boundaries now distinguish an implementable reduced kernel from full PLAS/ATLAS parity.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — Continuous least-service sorting still omits PLAS queue bands, quanta, demotion, and ATLAS critical-path state.**: Implement explicit PLAS and ATLAS modes using group/request scratch, trusted parent/thread facts, timer opportunities, and feedback-driven service accounting.
- **P0 — Request-scoped service feedback is written only to request scratch while scheduling reads only group scratch.**: Resolve the request's trusted group and charge enacted service to the group exactly once.
- **P1 — Anti-starvation lacks aggregate program-plus-call wait, promotion state, and reset semantics.**: Maintain queue index, remaining quantum, cumulative wait, current-call service, promotion, and reset state.
- **P2 — The benchmark remains a one-shot tuple oracle.**: Replay sequential and fork/join programs across multiple scheduling quanta and compare program JCT, throughput, and tail starvation with FCFS and MLFQ.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Define versioned PLAS/ATLAS fact schemas and an independent reference model.
1. Fix request-to-group feedback charging.
1. Implement queue, quantum, demotion, starvation, and ATLAS transitions.
1. Add stateful differential and metamorphic traces.
1. Replace the synthetic benchmark before promoting evidence.

## continuum

### Improve with the current model

- **Cache plans reclaimed every unpinned resident rather than the required capacity.** — A capacity-aware reclaim_prefix now stops after enough bytes are freed. (CacheContext already exposes resident/prospective sizes and capacity in one coherent snapshot.)
- **Artifact provenance and evidence were unpinned and overstated.** — Metadata pins artifact commit 316a5879 and labels the implementation inspired-adaptation. (The estimator-free artifact can now be separated from the paper's TTL optimizer claim.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — The paper's empirical-CDF utility optimizer is still replaced by host-supplied ttl_ms and a Boolean.**: Maintain tool-duration histories, tool identity, memory pressure, reload time, ordering correlation, and evaluate candidate TTLs including zero.
- **P0 — Active TTL is not timestamped and expiry does not check whether another request from the same program is queued.**: Store expires_at_ms in group/request scratch, consume trusted now_ms and queue facts, and clear the pin only under the paper's expiry condition.
- **P1 — Pinned residents are sorted last but can still be reclaimed without an explicit expiry or pin-break rule; reload_cost ordering is not the paper policy.**: Exclude active pins unless an explicitly modeled paper fallback applies; remove unsupported reload-cost ordering.
- **P2 — The benchmark conflates preempted and pinned priority and uses a strawman cache victim.**: Use multi-turn tool traces under pressure and compare LRU, fixed-2s, optimized-TTL, and oracle-TTL baselines.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Define trusted time, tool-history, pressure, and same-program queue facts.
1. Implement the TTL reference optimizer and timestamped state transitions.
1. Correct pin-preservation and pressure fallback semantics.
1. Differentially replay the pinned artifact for its supported subset and a paper oracle for estimation.
1. Replace the benchmark with response-time, throughput, and miss-cost traces.

## kvflow

### Improve with the current model

- **Non-ready work could run, max_selections was ignored, transfer-active nodes were evictable, and all residents were reclaimed.** — Scheduling now hard-filters cache_ready, honors all capacities, excludes loading/offloading objects, and reclaims only the required byte prefix. (Runnable facts, cache-object facts, bounded schedule capacity, and the existing cache operation already express these decisions.)
- **Transfer mechanics were conflated with the policy kernel.** — Metadata now declares loading/offloading facts, pins the artifact SHA, marks prefetch optional, and downgrades evidence. (cache.prefetch@1 and action feedback separate policy intent from adapter transfer overlap.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P1 — No implementation or validated producer constructs the Agent Step Graph, AND/OR STE recurrence, or minimum aggregation for shared radix nodes.**: Add a bounded trusted fact producer or policy-side reference module and validate its outputs against the artifact.
- **P1 — Prefetch action IDs and completion/failure do not update loading, CPU-backup, or GPU-resident state.**: Implement feedback, correlate action IDs, and transition explicit cache-tier states only after enacted outcomes.
- **P1 — Prospective objects are unconditionally admitted and shared-node beneficiary semantics are not tested.**: Apply the paper's node-level admission/state rules and add shared-beneficiary cases.
- **P2 — The benchmark still uses independent objects and never exercises prefetch or transfer overlap.**: Use AND/OR DAGs, shared radix nodes, four cache states, constrained capacity, and workflow makespan/cache-stall metrics.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Implement and validate the Agent Step Graph fact producer.
1. Add explicit tier/action state transitions through feedback.
1. Add artifact-derived shared-node and transfer-state traces.
1. Test prefetch success, failure, expiry, and unsupported mechanics.
1. Run end-to-end workflow-stall and makespan benchmarks.

## preble

### Improve with the current model

- **Exploit ties ignored load and exploration reused one target-invariant miss cost.** — Exploit now breaks ties by load; exploration consumes per-edge miss_prefill_cost; assignments track max_assignments. (Sparse route edges carry target-specific cache/load/eviction facts, and set-oriented targets expose assignment capacity.)
- **Artifact provenance and evidence were overstated.** — The public release commit is pinned and evidence is labeled inspired-adaptation. (Replication metadata can separate decision fidelity from deferred mechanics without changing the ABI.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P1 — Post-assignment load redirection, prefix replication, prefill/decode mixing, and local priority scheduling remain absent.**: Use policy state for rolling load and redirect decisions; add schedule/cache behavior and stage cache.prefetch@1 for explicit prefix replication.
- **P1 — The producer and benchmark do not guarantee that load_cost, eviction_cost, and miss_prefill_cost are consistently profiled GPU-time quantities.**: Define units in fact schemas and replay paper-derived per-target L_i, M_i, and P_i values.
- **P1 — Only max_assignments is tracked; aggregate ResourceLimit consumption is not decremented during multi-request routing.**: Add a shared capacity-accounting helper over edge demands and target limits.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Specify fact units and correct the benchmark miss-cost producer.
1. Add aggregate capacity accounting.
1. Implement rolling-load redirection and local scheduling decisions.
1. Add artifact differential traces, then test optional prefix-prefetch outcomes.

## helium

### Improve with the current model

- **Whole-DAG planning was previously conflated with a local scheduler hook.** — Metadata now explicitly classifies query-plan rewrite and proactive warming as deferred and labels the local kernel inspired-adaptation. (A bounded host planner can provide trusted DAG/TRT facts; WorkGroup scratch, route, repeated schedule opportunities, multi-request selections, and cache.prefetch@1 can enact the resulting plan.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — The implementation remains a one-request lexicographic heuristic rather than Algorithm 1's partitioning, scheduling tree, critical-path recursion, capacity timeline, and nested sequences.**: Port the whole-DAG reference planner or consume a validated host-produced plan with worker, sequence-path, start-step, and dependency facts.
- **P1 — The package implements neither worker assignment nor proactive warming.**: Add route and cache operations using worker-capacity facts and cache.prefetch@1, with action feedback.
- **P1 — No scoped state tracks emitted nested-sequence segments or forced-progress transitions.**: Persist the bounded scheduling-tree cursor and emit exact plan segments across opportunities.
- **P2 — The benchmark remains a tautological random tuple test.**: Execute complete batched DAGs on multiple workers and compare query-wise, op-wise, random, and LSPF using makespan, token steps, and cache hit rate.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Build an artifact-backed whole-DAG planner and plan schema.
1. Add route/cache surfaces and scoped execution-plan state.
1. Replay Algorithm 1 decisions, including forced progress and nested sequences.
1. Validate prefetch outcomes separately from schedule parity.
1. Replace the benchmark with the paper's multi-worker workflows.

## vtc

### Improve with the current model

- **Counter lift and batch-local accounting were absent.** — Added active-client tracking, returning-client lift, iterative minimum-counter selection, weighted dispatch-input charging, and zero-budget filtering. (Set-oriented schedule inputs, shared scoped state, deterministic request order, and per-request token budgets already provide the required authority.)
- **Feedback previously charged every record indiscriminately.** — Feedback now accepts only request progress records and applies weighted output-token deltas. (Typed feedback subjects/outcomes and idempotent feedback commits distinguish observed progress from terminal or unrelated records.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — Dispatch-input charge is persisted during schedule before engine enactment; a rejected or failed plan can overcharge service.**: Use tentative local counters only while constructing the plan and persist input charges from enacted schedule-selection or request-progress feedback.
- **P1 — Counter lift infers queue entry/exit from successive runnable sets and does not preserve the paper's last-client-to-leave behavior when the queue becomes empty.**: Consume explicit queue-membership/arrival facts or feedback, track the last exiting client's counter, and lift only on the first queued request for a previously inactive client.
- **P1 — The fixture and benchmark still test only homogeneous, permanently backlogged unit-token clients.**: Add executable ON/OFF, heterogeneous prompt/output, Poisson, multi-selection, and distribution-shift traces with theorem-bound disparity checks.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Move enacted input charging to feedback while retaining tentative batch-local counters.
1. Add exact queue-entry, queue-exit, and idle-system counter-lift state.
1. Create artifact-differential multi-step traces.
1. Replace the Jain-only benchmark with service-bound, TTFT, and work-conservation metrics.

## lmetric

### Improve with the current model

- **Idle replicas collapsed to score zero and hotspot status was ambiguous.** — The score now uses batch_size+1 and filters only hotspot_confirmed targets. (Per-edge P-token/batch facts and target facts directly encode the artifact score and detector output.)
- **Artifact revision and deferred detector were unrecorded.** — The exact artifact commit is pinned and the two-phase detector is explicitly deferred. (State can retain request-class history across route calls, so the detector is an implementation task rather than an ABI gap.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — Multi-request routing still ignores max_assignments and aggregate resource capacity.**: Use shared target-count and ResourceLimit accounting before selecting each edge.
- **P1 — The paper's popularity/cache-coverage alarm, 2|M| confirmation sequence, and load-only fallback are not implemented.**: Track per-class arrivals and cache coverage in shared state, confirm over consecutive requests, and derive hotspot_confirmed internally.
- **P2 — The benchmark excludes batch size zero and supplies independent random hotspot booleans.**: Add idle-replica boundaries and multi-invocation detector traces covering alarm, confirmation, clearing, and fallback.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Fix set-capacity accounting.
1. Implement the detector in policy state.
1. Add exhaustive product boundaries and stateful hotspot traces.
1. Run differential parity against a005ceb3.

## fairserve

### Improve with the current model

- **The prior implementation confused interaction handling with interference scoring.** — Admission now gates user/application RPM only under KV overload, preserves in-progress interactions, and respects max_accepted. (Admit receives a candidate set with host-provided overload, interaction, and quota facts; no new authority is needed.)
- **Interaction priority and token-category accounting were absent.** — Admission and scheduling prioritize in-progress interactions; feedback separately accounts input, system, and output tokens and filters for request progress. (WorkGroup/request facts, shared state, schedule ordering, and typed feedback can represent interaction lifecycle and observed service.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — Service counters are keyed by application rather than user, collapsing all users of an application into one counter.**: Key WSC state by user, retain application and stage as normalization dimensions, and use WorkGroup for each interaction.
- **P0 — The implementation does not calculate paper equations 2-3: it stores raw weighted tokens and divides by a generic request weight, omitting stage-specific expected token counts and user priority E_i.**: Add application_id, stage_id, expected input/system/output counts, and user priority facts; compute fixed-point normalized service at feedback time.
- **P1 — RPM and interaction state are entirely trusted as instantaneous facts, and the benchmark does not verify abusive windows, interaction completion, or token waste.**: Define fact provenance and delta semantics or maintain bounded per-user/application windows in scoped state; add overload and multi-call traces.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Correct state identity from application to user plus application-stage dimensions.
1. Implement equations 2-3 with fixed-point normalized service.
1. Model interactions as WorkGroups and validate overload/RPM transitions.
1. Benchmark token waste, served users, queue delay, TTFT, and stage throughput against VTC and RPM.

## marconi

### Improve with the current model

- **Mutable provenance and overstated evidence** — The artifact commit is pinned, evidence is downgraded to inspired-adaptation, and independent fidelity is reported separately from runtime smoke success. (Metadata and report schemas already support source commits and separate evidence/fidelity fields; no runtime primitive was needed.)
- **Paper ratios were conflated with proxy results** — The performance report now labels synthetic trends as adaptation proxies and records paper_end_to_end_ratio_reproduced=false. (The reporting model now explicitly separates proxy, live-mechanism, and paper end-to-end claims.)
- **Prospective admission must be compared with resident state under byte capacity** — Current Marconi evaluates resident and prospective objects in one cache opportunity and returns a capacity-valid retained set. (CacheContext already exposes resident and prospective objects, sizes, reclaimability, fixed bytes, and maximum bytes.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — The kernel still uses reuse_probability times recompute_flops per byte instead of separate speculative admission and normalized recency-plus-alpha-FLOP-efficiency eviction.**: Implement radix-tree speculative admission and the exact normalized eviction score over eligible nodes.
- **P0 — One-child eligibility, hit-node timestamp updates, alpha bootstrap/tuning, SSM-only intermediate reclaim, and logical KV merging are absent.**: Represent radix nodes and SSM/KV portions as separate cache objects/facts, persist timestamps and alpha in policy state, and use episodes for dependent reclaim.
- **P1 — Physical hybrid-state checkpointing and KV metadata merging are not implemented by the engine adapter.**: Implement them as adapter enactment of admitted/reclaimed logical objects; this is an engine mechanic, not a new policy operation.
- **P2 — Fixtures and benchmarks remain synthetic and do not compare decisions with the pinned artifact.**: Add differential traces for speculative branches, one-child eviction, alpha=0 LRU, timestamp updates, and hybrid LMSys/ShareGPT/SWE-bench traces.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement a pure stateful Marconi reference kernel using existing cache facts and state.
1. Add artifact-derived differential traces before adding physical hybrid kernels.
1. Model SSM and KV portions as separate logical objects and validate dependent episodes.
1. Implement adapter checkpoint/merge mechanics and feedback.
1. Replace the synthetic benchmark with hybrid trace replay.

## ragcache

### Improve with the current model

- **A single victim could leave the plan over capacity** — Victims are now ordered by the static PGDSF score and reclaim_prefix selects enough victims to satisfy byte capacity. (Resident sizes, prospective admissions, fixed bytes, maximum bytes, and reclaimability are all present in CacheContext.)
- **The replication report overstated fidelity** — RAGCache is now labeled inspired-adaptation with an independent material-semantic-gap finding. (Evidence and fidelity are represented independently from smoke validation.)
- **Proxy performance was presented without a claim boundary** — The generated performance report explicitly says the proxy is not the paper ratio and records zero exact end-to-end reproductions. (The report schema has separate proxy, live, fidelity, and paper-ratio fields.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — GPU and host logical clocks, retrieval-frequency updates, profiled average recomputation cost, and priority updates are absent.**: Persist both clocks and node statistics in State; update them through cache opportunities and feedback using fixed-point arithmetic.
- **P0 — The ordered list is computed once, so newly exposed parents cannot join the leaf frontier as eviction progresses.**: Use CacheEpisode iteration and parent/child facts to select one legal frontier step, update the clock, then re-invoke with newly eligible leaves.
- **P1 — GPU/host/free placement and swap-out-only-once state are not modeled.**: Encode tier and host-copy-exists in object facts/state and sequence copy-before-reclaim through cache.swap@1, feedback, and episodes.
- **P1 — The declared surface omits RAGCache cache-aware request reordering.**: Add schedule with cached-length/computation-length priority and starvation window, or narrow the claim explicitly to PGDSF cache replacement.
- **P2 — The fixture tests only an abstract iteration-zero leaf choice and the benchmark uses array index zero as LRU.**: Add multi-iteration parent exposure, separate-tier clocks, swap-once, and timestamped-LRU traces.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement stateful PGDSF updates and fixed-point score vectors.
1. Convert reclaim to bounded one-frontier-step episodes.
1. Add tier and host-copy state using existing swap actions.
1. Add cache-aware scheduling or narrow the replicated claim.
1. Validate against an independent paper oracle and then run vLLM+Faiss/SGLang workloads.

## dlpm

### Improve with the current model

- **Local scheduling previously prioritized deficit before locality and lacked refill and extend accounting.** — Scheduling now sorts by cached-prefix length, gates on positive deficit, performs a refill when no active client is positive, and charges extend and output tokens. (The schedule set, shared state, token budgets, host prefix facts, and progress feedback are sufficient for local DLPM decisions.)
- **Routing ignored target capacity and per-worker credit observations.** — Routing now enforces target max_assignments and filters by host-provided worker_deficit before selecting locality/load. (Route already exposes the full feasible edge graph, target capacities, and extensible per-edge facts.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — D²LPM does not own or update q_i,w; it trusts worker_deficit facts, never deducts selected-worker service, and skips the paper's worker-quantum refill loop when no worker is positive.**: Maintain a shared client-by-target deficit map, refill all worker credits until one is positive, debit input/extend service at assignment feedback, and debit output service at completion/progress feedback.
- **P0 — Local refill iterates requests rather than distinct active clients, so a client with multiple runnable requests receives quantum multiple times.**: Deduplicate client IDs before refill and use one configured Q_u per client per round.
- **P0 — Extend service is persisted optimistically during schedule and input/output weights are omitted.**: Use tentative deficit locally, commit enacted extend/output deltas through feedback, and apply w_e and w_q or require preweighted delta facts.
- **P1 — The fixture and benchmark remain mostly single-client and cannot establish bounded fairness or radix-tree behavior.**: Add multi-client local and multi-worker traces with longest-match sets, cache insertion/eviction, refill rounds, and adversarial locality.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Fix distinct-client local refill and feedback-only service commits.
1. Implement persistent per-client/per-worker D²LPM deficits and refill.
1. Require exact longest-prefix-set and weighted-service facts.
1. Add reference traces and locality/fairness/latency benchmarks.

## infercept

### Improve with the current model

- **The old plan reclaimed every reclaimable resident** — InferCept now truncates its ordered reclaim list once admitted and retained bytes fit capacity. (The cache snapshot already contains all object sizes, admissions, reclaimability, and byte limits.)
- **Artifact provenance and evidence were overstated** — The InferCept source commit is pinned and evidence is downgraded to inspired-adaptation with an independent incorrect finding. (Metadata and reports distinguish source provenance, runtime smoke, evidence strength, and fidelity.)
- **Synthetic ratios were conflated with paper performance** — The performance report now labels the result an inspired-adaptation proxy and records no paper end-to-end reproduction. (Proxy and paper claims are separate report fields.)

### New primitives

- **`cache.move@1`** — Whole-object cache.swap@1 cannot guarantee a budgeted ordered set of layer/page-range swaps with shared per-iteration bandwidth, capacity reservation, and partial-completion reporting. This is needed only for physical swap-pipeline fidelity, not preserve/discard decision parity. Minimal contract: Input ordered transfers with object_id, byte-or-token range, source tier, destination tier, per-iteration byte/time budget, and idempotency key; reserve capacity before transfer and return per-range completion/failure feedback.

### Implementation gaps

- **P0 — Preserve and chunk-discard waste equations are replaced by expected_reuse_ms < recompute_ms.**: Compute paper waste from context length, KV bytes per token, interception duration, running-context memory, chunk count, and profiled forward cost.
- **P0 — Ascending expected-waste resume ordering conflates interception-time swap allocation with resumed-request scheduling.**: Allocate swap-out budget to intercepted requests in descending minimum-waste order, then maintain separate original-arrival FCFS swap and waiting queues.
- **P0 — Explicit preserve, swap, discard, recompute-remaining, and swap-remaining state transitions are absent.**: Persist per-request mode/progress in State and expose trusted profile and original-arrival facts; schedule token chunks using existing token budgets.
- **P1 — The existing swap action is whole-object and does not enact the paper's layer/page pipeline.**: First reach exact decision traces with swap mechanics deferred, then implement cache.transfer-batch@1 in the adapter.
- **P2 — Fixtures and benchmarks never exercise all three modes or swap budgets.**: Add equation-boundary vectors, descending-budget traces, FCFS resume traces, chunk progress, and the six-augmentation mixed workload.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement exact waste calculations and explicit request modes.
1. Implement descending swap allocation and separate FCFS resume queues.
1. Add stateful differential traces against the pinned artifact.
1. Validate chunked recomputation using existing schedule token budgets.
1. Add cache.transfer-batch@1 only for physical swap-pipeline fidelity.
1. Run the paper's mixed workload and baselines.

## peek

### Improve with the current model

- **The old plan treated the full ordered list as the victim set** — PEEK now reclaims only the ordered prefix required to satisfy byte capacity. (Resident/prospective sizes and byte capacity are already first-class cache inputs.)
- **Mutable source and replication overclaim** — The PEEK artifact commit is pinned and the package is explicitly classified as inspired-adaptation/material-semantic-gap. (The evidence/report model separates smoke success from fidelity.)
- **Proxy and paper performance were conflated** — Current reporting labels the synthetic ratio as a proxy, includes live-mechanism counters, and states that no paper end-to-end ratio was reproduced. (The report schema separately records offline proxy, live mechanism, fidelity, and paper reproduction.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — No incremental pending trie, main-cache hit, warm/pioneer/sibling section, request score, or cluster size is maintained.**: Rebuild or incrementally synchronize the pending trie from runnable token facts and State, and compute the exact cLPM signals.
- **P0 — A hard overdue-first gate replaces the two-lane stride scheduler and dynamic EMA fairness share.**: Implement exact Lane A and Lane B keys, stride interleaving, group-major mode, singleton pressure, age pressure, and EMA state.
- **P0 — Eviction adds one global pending count to every object instead of computing maximum ancestor pending_count times depth.**: Store path/ancestor facts per cache object and rank by the exact ancestor-demand score, with the selected cluster/recency mode pinned to the artifact.
- **P1 — No-sharing fallback and engine-specific LPM probing are absent.**: Use candidate facts or host query results for cache hits and bypass cLPM when no root child has pending_count >= 2; no new WIT primitive is required.
- **P2 — The fixture checks only shared scalar state and the benchmark inspects only a synthetic first-victim utility.**: Add exact cold-cluster, warm-section, stride, EMA, no-sharing, group-major, and ancestor-demand differential traces plus W1-W5 workloads.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Implement a deterministic pending-trie reference model in State.
1. Add exact cLPM section and score vectors.
1. Implement stride lanes, dynamic fairness, group-major, and no-sharing fallback.
1. Implement exact ancestor-demand reclaim ordering.
1. Differentially replay against the pinned artifact on both SGLang-style radix and vLLM-style hash inputs.
1. Run W1-W5 with the paper baselines.

## qlm

### Improve with the current model

- **Admission and routing could emit contract-invalid over-capacity plans.** — Admission now enforces max_accepted and shared routing enforces target max_assignments. (Set-oriented admit/route inputs already carry bounded capacities and allow joint decisions.)
- **The declared feedback operation was missing.** — Feedback now writes virtual-wait observations into WorkGroup or Request scratch and records deliveries. (Scoped group/request state and typed feedback subjects provide the needed queue-accounting scope.)
- **Evidence and baseline metadata overstated the old kernel.** — The policy is now labeled inspired-adaptation and lists EDF, vLLM-FCFS, and SHEPHERD baselines. (The report can carry an independent fidelity classification separately from compilation status.)

### New primitives

- **`request.pause-resume@1`** — QLM request eviction must suspend an active request, preserve or swap its KV state, and later resume it; Paused exists as lifecycle state but no policy action requests this transition. Minimal contract: Idempotent pause/resume action over request generation with KV policy or cache-object references; enacted feedback reports status transition, transferred bytes, latency, and failure.
- **`target.provision@1`** — QLM's multi-model virtual queues actively warm and swap model weights among storage, CPU, and GPU; no current mechanic controls model residency. Minimal contract: Idempotent target-scoped model activation or tier-move action carrying model ID and desired residency tier; feedback reports residency revision, duration, capacity use, and failure.

### Implementation gaps

- **P0 — There is still no request-group construction, RWT equations 1-5, virtual-queue representation, or global equations 6-13 optimizer.**: Represent each request group in WorkGroup state, ingest profiled token/model/device facts, compute RWT distributions, and persist an optimized bounded virtual-queue order.
- **P0 — Route and schedule remain independent scalar greedy choices rather than one global model-aware assignment/order.**: Solve the complete set-oriented assignment with swap and completion costs, then emit current route and schedule decisions from the persisted plan.
- **P1 — Request-scoped virtual_wait written by feedback is never read by schedule; only candidate and group values are consulted.**: Read request scratch as a fallback or remove request-scoped writes and make WorkGroup the canonical queue state.
- **P1 — The case and performance score still validate a scalar inequality rather than queue evolution or LSOs.**: Create multi-step virtual-queue traces and separate decision parity from unavailable pause/model mechanics.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement grouped RWT and global optimizer with deterministic reference traces.
1. Make WorkGroup scratch the canonical virtual-queue state.
1. Validate optimizer decisions without claiming physical LSOs.
1. Add request pause/resume and model activation mechanics.
1. Run adapter-backed eviction, swap, throughput, and SLO experiments.

## slos-serve

### Improve with the current model

- **Admission and route could violate count capacities.** — Admission now bounds accepts by max_accepted and shared routing respects per-target max_assignments. (Joint set inputs and capacity-bearing targets are existing core semantics.)
- **Baseline and evidence metadata were inaccurate.** — Targets now name vLLM, Sarathi, and DistServe, and metadata labels the implementation inspired-adaptation with provisioning explicitly deferred. (Evidence classification is independent of the five-operation ABI and can honestly separate decision logic from deployment mechanics.)

### New primitives

- **`schedule.co-execute@1`** — SLOs-Serve predicts and optimizes one concrete mixed-token forward batch; current multi-request selections are all-or-none but explicitly do not guarantee simultaneous execution or exact next-step composition. Minimal contract: A negotiated guarantee that one selection's exact members and per-request token budgets form the next forward pass with no substitution; atomic failure if any member cannot fit; feedback reports realized composition and latency.
- **`decode.speculate@1`** — SchedulePlan cannot request a drafter model, per-tier speculation lengths, or a base-model verification pass. Minimal contract: Idempotent decode action or schedule extension specifying request IDs, drafter/base model, speculation length, and verification budget; feedback reports accepted tokens, verification latency, and failures.
- **`request.pause-resume@1`** — Burst-resilient best-effort service preempts declined work while preserving generated tokens and applying a specified KV discard/retain policy; no current action controls paused lifecycle. Minimal contract: Pause/resume action over a request generation with generated-output preservation and explicit KV policy; enacted feedback supplies lifecycle and recomputation outcome.

### Implementation gaps

- **P0 — No joint DP state, request value objective, memory dimension, prefill deadlines, TPOT tiers, or performance model is implemented.**: Persist the paper DP over bounded host-provided stage, memory, deadline, and profiled batch facts; reconstruct admitted requests and planned batches.
- **P0 — Schedule still emits least-slack singleton ordering instead of the current planned mixed prefill/decode token allocation.**: Store the planned batch sequence in scoped state and emit its current per-request budgets at each scheduling epoch.
- **P1 — Routing minimizes scalar stage latency rather than retrying declined requests according to replica-level SLO attainability.**: Route only after evaluating each replica's joint plan and carry bounded retry count and fallback state in request scratch.
- **P1 — The fixture and benchmark remain independent-feasibility tests with no stage or burst execution.**: Add multi-stage DP traces now, then gate end-to-end claims on the new exact-batch/speculation mechanics.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement the pure DP and performance-model reference logic using typed facts and scoped state.
1. Generate admission, routing, and planned-batch decision traces.
1. Add exact-forward-batch and speculative-decode mechanics.
1. Add pause/resume best-effort handling.
1. Run burst and multi-replica capacity experiments.

## dynasor

### Improve with the current model

- **The declared feedback operation was absent.** — A feedback hook now exists and commits bounded feedback accounting. (Typed feedback and scoped state support program-progress histories and future Certaindex updates without a new operation.)
- **Evidence and baseline metadata overstated the old threshold kernel.** — Metadata now labels it inspired-adaptation and names uniform, length, SGLang, and Parrot baselines. (Evidence classification and benchmark provenance are report concerns, not programming-model limitations.)

### New primitives

- **`request.complete-with-output@1`** — The paper stops reasoning and returns the stable probed or aggregated answer; request.cancel@1 yields cancellation and cannot enact successful completion with a final payload. Minimal contract: Idempotent action carrying request generation and a final-output reference or validated fields revision; terminal Completed occurs only after enacted feedback, which reports delivered output and failure.
- **`schedule.co-execute@1`** — Dynasor's online gang scheduler derives throughput from co-batching requests of one reasoning program; an all-or-none selection does not guarantee one physical forward batch. Minimal contract: Guarantee that selected program members and token budgets execute in the same next forward pass; feedback reports realized batch and latency.

### Implementation gaps

- **P0 — Confidence and progress remain opaque scalar inputs; no answer-consistency, semantic-entropy, reward aggregation, uncertainty-word filtering, or calibrated policy state is implemented.**: Define typed Certaindex provenance facts or compute them from bounded feedback histories; retain answer/reward windows in request or WorkGroup scratch.
- **P0 — High-confidence work is cancelled rather than successfully completed with the stable answer.**: Until early-complete exists, classify cancellation as intent-only decision parity and do not claim equal-quality completion.
- **P0 — request.cancel@1 is declared optional but invoked unconditionally when the threshold fires.**: Either require the mechanic or branch on negotiated mechanics and return a non-action fallback plan.
- **P1 — No program-level resource knob, WorkGroup accounting, gang priority, SJF estimate, or starvation escalation exists.**: Represent each reasoning program as a WorkGroup and schedule branches/iterations from group Certaindex, estimated remaining work, and wait-age state.
- **P1 — The benchmark still defines usefulness from the same random threshold that triggers cancellation.**: Replay real probe/reward histories and measure tokens-to-accuracy, final-answer equivalence, program SLOs, and probe cost.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Define Certaindex fact and feedback schemas and program WorkGroup state.
1. Implement adaptive branch/iteration allocation, gang priority, SJF, and starvation prevention.
1. Produce artifact-differential stop/continue traces while labeling cancellation as intent.
1. Add successful early-complete authority.
1. Add exact gang-batch guarantee and run equal-quality online/offline evaluations.

## justitia

### Improve with the current model

- **Task-parallel agent identity and all-or-none scheduling were previously treated as absent.** — The current model has trusted WorkGroups, group-private state, multi-request ScheduleSelection units, and schedule.atomic-enqueue@1. (GPS accounting can be group-scoped and all fitting branches of the selected agent can be emitted as one normalized selection without a new operation.)
- **Evidence no longer claims reproduction of the paper scheduler.** — Metadata labels the completed-branch heuristic inspired-adaptation. (The remaining discrepancy is correctly classified as missing implementation rather than a WorkGroup-model limitation.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — Fewest-completed-branches ordering is unrelated to GPS virtual time and predicted virtual finish tags.**: Replace it with memory-centric predicted cost, active-set virtual-time updates, immutable finish tags, and minimum-tag ordering.
- **P0 — The singleton helper cannot pamper all fitting ready branches of the chosen agent.**: Construct one multi-request selection from the selected WorkGroup and negotiate schedule.atomic-enqueue@1 where atomic adapter enqueue is required.
- **P1 — Feedback tracks branch count rather than arrivals, completions, active-agent count, virtual time, and consumed cost.**: Update GPS state from trusted timestamps and enacted completion/progress feedback.
- **P2 — The benchmark defines remaining branches as the objective.**: Replay heterogeneous agents with finite KV capacity and measure average/P90 JCT and finish-time fairness against VTC, Parrot, FCFS, and SRJF.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Define predicted-cost and trusted-time schemas plus a GPS reference oracle.
1. Implement group-scoped virtual time and finish-tag transitions.
1. Emit grouped pampering selections and negotiate atomic enqueue.
1. Add starvation, fairness, and conservation traces.
1. Replace the completed-branch benchmark.

## chameleon

### Improve with the current model

- **Cache policy over-reclaimed every resident** — Chameleon now truncates reclaim to the bytes required after prospective admissions. (CacheContext exposes object sizes, admissions, reclaimability, fixed bytes, and maximum bytes.)
- **The benchmark target used incorrect paper numbers and baseline** — The target now names S-LoRA and the accepted v2 results: 1.5x throughput, 80.7% lower P99 TTFT, and 48.1% lower P50 TTFT. (Benchmark metadata can accurately name external baselines and north-star metrics independently of the synthetic proxy.)
- **Replication status overstated fidelity** — Metadata and generated reports now classify Chameleon as inspired-adaptation with a material semantic gap. (Smoke validation and fidelity classification are separate report dimensions.)

### New primitives

- **`request.pause-resume@1`** — Chameleon's misprediction recovery squashes a younger bypassed request and later re-executes it. Existing request.cancel@1 is terminal and schedule plans cannot request a nonterminal atomic preempt-and-requeue transition. Minimal contract: Input request_id, preserve-or-discard execution state, target queue/class and ordering key, reason, and idempotency key; atomically transition active to paused/pending without terminal cancellation and report completion/failure.

### Implementation gaps

- **P0 — Admission decrements a global request count rather than per-queue token quotas and does not perform two-phase spare redistribution.**: Persist per-queue quotas, debit each request's token-equivalent resource demand, scan every queue in phase one, then redistribute leftovers small-to-large.
- **P0 — The adapter cache uses adapter_hot admission and lacks dynamic sizing, the 0.45/0.10/0.45 score, active reference counts, and queued-adapter protection.**: Use resident facts/beneficiaries for frequency, recency, size, active references and queued demand; rank only eligible adapters and reclaim the required byte prefix.
- **P0 — Queue-class sorting is not the paper's quota-driven batch construction.**: Construct schedule selections from all queues according to quota phases rather than globally sorting by class and wait.
- **P1 — The adapter-bypass fixture tests cache bypass, not head-of-line request bypass or squash.**: Add loaded/fits/predicted-duration/head-release facts and decision traces; defer only misprediction squash enactment until request.preempt-requeue@1 exists.
- **P2 — The paper version is not pinned to v2/DOI and benchmarks ignore quota/cache-pressure outcomes.**: Pin the accepted version and add Figure 10 quota vectors, cache-score ties, bypass traces, and S-LoRA P50/P99/SLO-throughput workloads.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Pin Chameleon v2 and define exact fact units.
1. Implement per-queue quota state and the two scheduling phases.
1. Implement the adapter-cache score, eligibility, and dynamic capacity behavior.
1. Implement correct bypass decisions and differential traces.
1. Add request.preempt-requeue@1 only for misprediction squash enactment.
1. Run S-LoRA comparative workloads.

## hotprefix

### Improve with the current model

- **The old plan reclaimed every cold resident** — HotPrefix now reclaims only enough threshold-eligible residents to restore byte capacity. (The existing cache snapshot provides all sizes, admissions, reclaimability, and capacity needed to truncate the victim order.)
- **The target conflated latency and throughput maxima and omitted paper baselines** — Performance metadata now records vLLM/SGLang latency maxima, the lower throughput maxima, and LRU/LFU/FIFO/2Q baselines. (North-star claims and baselines are metadata, independent of whether the current proxy reproduces them.)
- **Replication evidence overstated fidelity** — HotPrefix is now explicitly inspired-adaptation/incorrect, while smoke and live-mechanism results are reported separately. (The reporting schema separates evidence, fidelity, proxy trends, and live mechanics.)

### New primitives

- **`cache.move@1`** — Hotness promotion replaces one or more GPU victims with one or more host-resident prefixes under a fixed GPU budget. Independent prefetch actions plus reclaim indices cannot guarantee all-or-none tier exchange, source retention, or capacity reservation while transfers overlap decoding. Minimal contract: Input incoming objects/ranges and source tiers, victim objects/ranges and destination/discard disposition, target-tier byte budget, dependency constraints, and idempotency key; atomically reserve capacity and report per-object transfer/enactment results.

### Implementation gaps

- **P0 — Cumulative reuse_count lacks max-age clock initialization, access reset, periodic aging, node length, depth, and leaf eligibility.**: Persist per-prefix frequency, clock, length, depth, parent, and tier state; implement periodic clock decay and GPU leaf priority (frequency + clock) / length.
- **P0 — The threshold is applied to prospective primary-cache admission rather than to GPU-evicted objects being considered for host admission.**: Separate GPU insertion/eviction from host admission; after the frequency threshold, compare frequency * clock with the coldest host entry under host capacity.
- **P1 — Prefetch is issued for every hot prospective object rather than from an ordered host-root/GPU-leaf promotion plan.**: Generate the exact promotion plan using host roots descending by hotness, GPU leaves ascending by hotness, token-fit packing, and parent constraints; emit action intents and defer atomic enactment.
- **P2 — Fixtures cover only threshold admission and benchmarks have no resident tiers, eviction, promotion, or equal-capacity baseline.**: Add Figure 4 aging, Eq. 5 eviction, Eq. 6 host admission, Algorithm 2 promotion, and MMLU/Hellaswag/BBH/CEVAL traces.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement exact hotness state and aging.
1. Implement GPU leaf eviction and host admission as separate cache phases.
1. Implement the promotion-plan reference algorithm using tier and parent facts.
1. Add differential vectors for equations 5-6 and Algorithm 2.
1. Add cache.atomic-tier-exchange@1 only for physical all-or-none promotion.
1. Run the paper workloads and baselines.

## pard

### Improve with the current model

- **The previous two-term downstream-P95 gate did not implement PARD equation 3.** — Projection now sums upstream elapsed, current queue and execution, downstream queue and execution, and downstream batch-wait P10. (Host-provided latency-distribution facts and set-oriented scheduling can express the Request Broker decision without a new operation.)
- **The declared feedback operation was absent.** — Feedback now exists as a state-update point for future estimator and outcome accounting. (Typed progress/action feedback and idempotent scoped state are already available.)
- **Baseline metadata was too vague.** — Targets now identify Nexus, Clipper, and naive baselines. (Baseline fidelity is report metadata rather than a policy-surface capability.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — Adaptive request selection is absent: survivors retain input order and there is no remaining-budget DEPQ, load factor, HBF/LBF switch, or hysteresis.**: Compute remaining budgets and load factor from facts/state, sort HBF above the high threshold and LBF below the low threshold, and persist hysteresis mode.
- **P0 — request.cancel@1 is optional in metadata but is unconditionally required by every projected miss.**: Declare cancellation required for this package or guard against missing negotiation and expose only drop intent.
- **P0 — The committed basic case still supplies downstream_p95_ms, which the current implementation ignores, while expected output still requires cancellation.**: Replace it with all six equation-3 components and add boundary cases for equality, P10, and DAG maximum path.
- **P1 — No sliding queue window, sampled batch-wait distribution, load smoothing, DAG-path maximum, or estimator update occurs in feedback.**: Either implement bounded estimator state in feedback or define these values as trusted adapter facts with explicit provenance and replay fixtures.
- **P1 — The performance scenario remains a one-step tautological feasibility filter.**: Replay bursty multi-module traces and report goodput, drop rate, invalid GPU time, queue delay, and module-of-drop.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Repair the stale fixture and cancellation capability declaration.
1. Implement remaining-budget DEPQ and adaptive HBF/LBF hysteresis.
1. Implement or formally source the estimator facts through feedback.
1. Add linear and DAG pipeline decision traces.
1. Run Nexus/Clipper/naive goodput and invalid-compute comparisons.

## branch-regulation

### Improve with the current model

- **The old scheduler cancelled excess branches or their parent requests.** — Scheduling now defers excess candidates by omission, matching TAPER's reversible-control direction. (Schedule already has authority to select a subset of runnable branch requests without cancelling them.)
- **Admission could exceed global capacity.** — Admission now enforces both per-group branch limits and max_accepted. (Set-oriented admission and trusted WorkGroup identity provide joint branch-count authority.)
- **Evidence and baselines were overstated.** — Metadata now labels inspired-adaptation and targets list IRP-Off, C2, C5, and Eager. (Evidence classification can separate policy decisions from absent physical branch mechanics.)

### New primitives

- **`schedule.co-execute@1`** — TAPER's predictor and slack bound apply to the exact next decode-step composition; current selections are not a simultaneous-execution guarantee. Minimal contract: Enact exactly the selected branch requests and token budgets in the next forward pass with no substitution; fail atomically if composition cannot fit; feedback reports realized step latency and members.
- **`branch.defer-retain@1`** — TAPER's cheap per-step replanning assumes deferring a branch leaves shared-prefix and branch-local KV resident and resumable without restoration; current schedule omission carries no such physical guarantee. Minimal contract: For typed branch requests in one WorkGroup, omission from a step preserves shared-prefix and branch-local KV until reconsideration; whole-parent preemption remains legal but must be reported through feedback.

### Implementation gaps

- **P0 — Admission still applies static branch_limit and scalar interference_limit rather than protected baseline composition, predicted T(S), minimum deadline slack, rho budget, and marginal utility.**: Move the core decision to per-step schedule: select one protected branch per active parent, then greedily add ready branches by marginal utility per predicted latency while within the slack budget.
- **P0 — excess_branch is a host verdict rather than a result computed from current batch composition and deadlines; protected progress is not guaranteed.**: Require branch role, parent/group, readiness, context length, deadline, and utility facts; compute excess status within the policy and never filter the sole protected branch.
- **P0 — Metadata still claims excess-branch cancellation and declares request.cancel@1 although current code no longer cancels.**: Replace cancellation language with reversible branch deferral and remove the unused optional mechanic.
- **P1 — No per-step state, realized-latency feedback, predictor refresh, or utility allocation exists.**: Persist predictor calibration and prior grants in WorkGroup/shared state and update them from step feedback.
- **P1 — The fixture and benchmark remain static branch-cap tests with an artificial score.**: Replay serial/parallel phases over low, moderate, and high load and separate decision parity from exact-batch/KV-mechanic evidence.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Correct stale cancellation metadata and branch fact schema.
1. Implement protected progress, slack budgeting, predictor evaluation, and greedy utility in schedule.
1. Add per-step decision-trace fixtures across load transitions.
1. Add exact-forward-batch and branch-KV-retention guarantees.
1. Run adapter-backed IRP-Off/C2/C5/Eager goodput and TPOT experiments.

## dualmap

### Improve with the current model

- **Routing considered arbitrary edges and cache affinity always dominated SLO risk.** — The kernel now limits routing to two ordered hash candidates, uses prefix-hit token counts, applies an SLO test, and tracks target assignment counts. (Sparse edges represent the two candidate mappings and carry candidate-specific hit and TTFT facts; direct assignments express the selected mapping.)
- **Evidence previously implied kernel reproduction.** — Metadata now pins the public artifact and classifies the implementation as inspired-adaptation with physical migration deferred. (RouteCause::Rebalance and set-oriented requests can express queued-request reassignment separately from enactment.)

### New primitives

- **`target.provision@1`** — TargetChanged lets a policy react to membership changes but cannot request add, drain, or remove operations needed for DualMap's elastic dual-hash-ring claim. Minimal contract: Action containing operation add|drain|remove, target or deployment specification, expected membership revision, idempotency key, and terminal feedback with resulting revision.

### Implementation gaps

- **P0 — Hotspot handling still emits one rebalance action for the current request; it does not rank queued requests by TTFT benefit or restrict movement to each request's alternate candidate.**: On RouteCause::Rebalance, accept the overloaded queue as the request set, compute B(i→j), and directly reassign positive-benefit requests in descending order until SLO-safe.
- **P1 — The SLO fallback minimizes predicted TTFT rather than explicitly selecting the less-loaded candidate using pending prefill tokens.**: Expose pending_prefill_tokens and select by that metric after the affinity candidate exceeds the SLO threshold.
- **P1 — Adaptive prefix hotness and hash-key refinement are delegated to unvalidated input facts.**: Maintain sliding-window prefix ratios in state and test the 2/n hot and 1/n cold transitions, or validate parity of a pinned host-side candidate producer.
- **P2 — The benchmark supplies four candidates, truncates them indirectly, and disables hotspot behavior.**: Use exactly two artifact-derived candidates and add equal-hit, SLO-switch, alternate-candidate, and queued-rebalance traces.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement exact load-aware SLO fallback.
1. Implement set-oriented queued rebalancing without relying on physical movement.
1. Add adaptive-prefix candidate parity tests.
1. Specify target membership control only if claiming elasticity enactment.

## llumnix

### Improve with the current model

- **Routing minimized a raw virtual-usage scalar.** — The kernel now computes freeness from memory capacity, virtual usage, and batch size, selects maximum freeness, and tracks assignment counts. (Target facts encode instance memory and batch state; edge facts encode aggregate virtual usage; set-oriented routing provides candidate instances and capacities.)
- **Artifact provenance and physical migration scope were unclear.** — The artifact commit is pinned and the result is labeled inspired-adaptation with physical migration deferred. (The generic rebalance action and action feedback separate migration intent from engine enactment.)

### New primitives

- **`request.migrate@1`** — request.rebalance@1 exposes only request and destination and cannot represent preallocation, incremental KV-copy rounds, abort conditions, commit, or measured downtime. Minimal contract: Typed migration action with request, source, destination, transfer mode, generation, expected source/destination revisions, idempotency key, and progress/terminal feedback for preallocated, copying, committed, aborted, bytes, rounds, and downtime.
- **`target.provision@1`** — The policy can observe targets but cannot create, drain, or terminate instances. Minimal contract: Idempotent add|drain|remove action over deployment and target IDs with expected membership revision and terminal feedback.

### Implementation gaps

- **P0 — saturating_sub clamps negative freeness to zero, although negative freeness is semantically important for queued, priority, and draining cases.**: Compute signed freeness using i128 or an ordered signed rational and preserve negative values.
- **P1 — HOL demand, priority headroom, fake-infinity draining usage, and per-priority division are not computed or validated.**: Implement the virtual-usage rules in policy/state or differential-test a pinned fact producer.
- **P0 — Migration remains a request Boolean routed to the freest target; source/destination threshold pairing and low-priority/short-sequence victim selection are absent.**: Use RouteCause::Rebalance with a set of running requests, pair extreme-freeness instances, and select victims by priority and length before staging actions.
- **P1 — Feedback only counts records and does not reconcile migration success/failure or load state.**: Consume Action feedback and update request/instance migration scratch state.
- **P2 — The benchmark has no negative freeness, migration round, fragmentation, priority, or evolving queue.**: Add artifact-derived multi-epoch dispatch and pairing traces; keep downtime and cost claims separate until lifecycle enactment exists.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Fix signed freeness.
1. Implement all virtual-usage rules.
1. Implement set-oriented source/destination pairing and victim selection.
1. Reconcile rebalance outcomes in feedback.
1. Add migration and membership primitives only for physical/cost claims.

## smetric

### Improve with the current model

- **Follow-ups always selected cache affinity and staged active migration.** — The kernel now implements first-turn load balancing, overload and likely-eviction guards, and load-only fallback without an active rebalance action. (Trusted generation_id identifies turns; sparse edges carry hit, history-hit, and load facts; direct assignments express fallback placement.)
- **Global-tier behavior was mislabeled as physical migration.** — Metadata now defers global-tier KV fetch and calls the kernel an inspired adaptation. (The route decision itself is fully representable; cache movement can remain a separately negotiated mechanic.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P1 — The route kernel is close to the paper pseudocode, but no test proves overload and eviction guards independently or across a persistent session.**: Add first-turn, sticky follow-up, overload fallback, likely-eviction fallback, and post-fallback session traces.
- **P1 — Global-tier availability, capacity, and fetch latency are assumed rather than represented in benchmark facts or cache actions.**: Expose global-hit availability and expected fetch cost on edges; optionally stage existing cache.prefetch@1 when a concrete cache object is known.
- **P1 — Only max_assignments is tracked; aggregate resource demand is not accounted across the request set.**: Use shared ResourceLimit accounting while constructing assignments.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Add guard-specific persistent-session fixtures.
1. Add aggregate capacity accounting.
1. Model global-fetch availability/cost in edge facts.
1. Run trace parity, then separately validate cache-transfer mechanics.

## thunderagent

### Improve with the current model

- **Dead-program cache plans previously reclaimed every eligible object.** — Cache reclaim now stops after the required capacity is freed. (CacheContext supplies complete byte accounting in one snapshot.)
- **Migration/cancellation outcomes and paused identity were previously underspecified.** — The current model has RequestStatus::Paused, request.rebalance@1, request.cancel@1, action IDs, and enacted action feedback. (Migration and cancellation intent can be correlated safely; only policy-requested pause/resume and tool-environment authority remain absent.)

### New primitives

- **`request.pause-resume@1`** — Omitting a request from SchedulePlan does not transition active to paused or release its backend/KV resources. Minimal contract: Action with request_id, release_kv, reason, expires_at_ms, and idempotency_key; feedback reports resulting paused status, prior target, released bytes, and terminal status.
- **`tool.resource@1`** — Facts can report tool readiness, but no existing authority can request asynchronous environment preparation or release sandboxes, disk, and ports. Minimal contract: Action with group_id or request_id, tool_spec_key, desired_state prepare|release, ready_by_ms, resource limits, and idempotency_key; feedback reports handle, resource usage, and status.

### Implementation gaps

- **P0 — The scheduler still filters caller-labelled tool states instead of detecting thrashing and scoring pause/restore candidates from phase, context, and capacity.**: After adding lifecycle mechanics, implement reasoning/acting state, decay, periodic capacity checks, shortest-context victim selection, and restore scoring.
- **P0 — No policy state machine records active, paused, restored, or terminated transitions.**: Use WorkGroup/request scratch and action feedback to update transitions only after enactment.
- **P1 — Migration may be staged while the same request is also scheduled, and feedback merely counts records.**: Make migration mutually exclusive with service and correlate each action result before updating assignment state.
- **P1 — program_live does not distinguish active retention from pause-time KV release.**: Drive cache decisions from enacted program phase/lifecycle and context-recomputation cost.
- **P2 — The benchmark never sets migrate_target or creates pressure and lifecycle transitions.**: Use multi-backend reasoning/acting traces and measure throughput, recomputation, hit rate, imbalance, and tool resources.

**Best evidence without new primitives:** `inspired-only`

**Recommended sequence:**
1. Standardize pause, resume, and tool-environment mechanics with adapter tests.
1. Implement artifact-equivalent program and backend state machines.
1. Implement periodic pause/restore and migration decisions.
1. Add action-success/failure, duplicate-feedback, and restart traces.
1. Run the paper's multi-backend and tool-resource benchmarks.

## pythia

### Improve with the current model

- **Cache plans over-reclaimed and independent routing ignored target assignment counts.** — Pythia now uses capacity-bounded reclaim and the shared route helper reserves max_assignments. (RouteTarget capacities, edge demands, resident/prospective cache objects, and CachePlan already provide the required bounded decision surfaces.)
- **Prefetch was treated as absent physical behavior.** — cache.prefetch@1 is an existing negotiated action with target, urgency, expiry, idempotency, and action feedback. (Algorithms 1-3 can use host facts plus existing route, schedule, cache, feedback, and prefetch mechanics; only autoscaling authority is absent.)

### New primitives

- **`target.provision@1`** — Pythia's phase-adaptive autoscaler chooses future replica counts, while PLEX explicitly lacks provisioning, replica creation, and model-loading authority. Minimal contract: Action with model_or_pool_id, desired_replicas or delta, placement/resource constraints, ready_by_ms, and idempotency_key; feedback reports resulting target IDs/revisions, readiness, and failure.

### Implementation gaps

- **P0 — lookahead_cost routing does not implement union-bound OOM safety, output bounds, maximum headroom, or cache-affinity tie-breaking.**: Implement Algorithm 2 using trusted distribution bounds, edge demands, target resource vectors, and aggregate per-target reservations.
- **P0 — workflow_rank ordering does not implement expected remaining distance, downstream-idle risk, weighted priority, local aging, or lowest-priority preemption.**: Implement Algorithm 3 and maintain queue/aging state through timer and feedback opportunities.
- **P0 — next_use_step eviction is not Algorithm 1's future-lineage keep/drop and tiered staging policy.**: Represent future lineages, exact prompt composition, cache tiers, dead-token reclamation, host staging, and idle-only background prefill.
- **P0 — The route helper reserves counts but not aggregate named resource demands, so multi-request outputs can still fail validation.**: Track target resource totals while selecting edges or implement a Pythia-specific capacity-safe assignment.
- **P1 — Feedback is an unused counter and cannot drive cache/prefetch/priority transitions.**: Correlate action, request, route, and schedule outcomes into scoped state.
- **P2 — The benchmark independently minimizes the same random scalars and ignores cache results.**: Use workflow regexes, output distributions, multiple model queues/cache tiers, and JCT, throughput, TTFT, and OOM metrics.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement a paper-reference fact producer for workflow regexes and output bounds.
1. Replace route, schedule, and cache kernels with Algorithms 1-3.
1. Add action/outcome state transitions and resource-safe multi-request routing.
1. Establish differential and stateful decision-trace parity.
1. Add model.scale@1 only before claiming the full-system autoscaling results.

## goodserve

### Improve with the current model

- **The prior kernel optimized unrelated quality-gap and cost fields.** — It now evaluates q+p(Lin-H)+dLout, selects the least-capable feasible GPU, falls back to the most capable GPU, tracks assignment capacity, and implements feedback. (Requests and sparse edges carry fixed predictor outputs and concrete heterogeneous targets; state/feedback can retain monitor results; rebalance stages corrective intent.)
- **The benchmark used quality-adjusted cost.** — It now generates deadline, queue, cache, prefill/decode, capability, and output-length facts and reports an E2E-SLO proxy. (The policy model accepts prediction facts without embedding predictor training or deployment in the ABI.)

### New primitives

- **`request.migrate@1`** — Generic rebalance cannot require GoodServe's token-ID transfer and destination recomputation semantics or verify that corrective migration preserves generation state. Minimal contract: Migration action with request, source, destination, transfer_mode=token-ids, generation/RNG continuation data or preservation guarantee, expected revisions, idempotency key, and progress/terminal feedback including recompute and transfer latency.

### Implementation gaps

- **P0 — A high-risk request is sent to the ordinary selected target; the code does not require a target stronger than the current GPU or use refreshed remaining-time estimates.**: On RouteCause::Rebalance, include current target/capability and remaining deadline, then restrict candidates to stronger GPUs and apply just-enough selection to refreshed predictions.
- **P1 — The 50-iteration reevaluation loop and EMA/request predictor state are absent; feedback only counts records.**: Have the host trigger periodic feedback/rebalance opportunities and update per-target q/p/d and per-request output predictions in state.
- **P1 — capability_rank ordering is implicit and lacks schema validation.**: Define lower rank as less capable, reject inconsistent ranks, and add feasible/no-feasible boundary tests.
- **P2 — The benchmark still compares only with a cost-only baseline and never triggers migration.**: Add random, least-request, round-robin, lowest-TPM, Preble, and Llumnix proxies plus refreshed-risk migration traces.
- **P1 — Aggregate ResourceLimit usage is not tracked across the request set.**: Apply common demand accounting in addition to max_assignments.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Constrain corrective routes to stronger just-enough targets.
1. Implement periodic monitor state through feedback.
1. Add capacity and rank validation.
1. Expand decision baselines and migration traces.
1. Add token-ID migration lifecycle only before claiming migration latency.

## conserve

### Improve with the current model

- **Initial routing used arbitrary capacity and follow-ups merely preferred affinity.** — Initial requests now require a prefiller target; follow-ups require a decoder, honor bound_target_id, and otherwise minimize active KV occupancy; assignments track max_assignments. (Target role facts, trusted generations, request facts, sparse edges, and mutable request/group scratch can represent phase-aware placement and persistent decoder identity.)

### New primitives

- **`request.phase-handoff@1`** — Direct route assignments and generic cache prefetch do not specify the exactly-once transition from first-turn prefill to first-turn decode, bind that decoder for the conversation, or correlate KV transfer with phase completion. Minimal contract: Typed action containing request, generation, from_target, to_target, phase prefill-to-decode, KV object/reference, exactly-once idempotency key, expected revisions, and success/failure feedback including transferred bytes and committed binding.

### Implementation gaps

- **P0 — Generation 0 is always routed to the prefiller, so first-turn decode is never assigned to the decoder with lowest active KV occupancy.**: Distinguish first-prefill from first-decode using a trusted phase fact/cause; after prefill completion select and bind a decoder.
- **P0 — bound_target_id is supplied externally rather than persisted by the policy, so restart, missing-target, and stale-binding behavior is unproven.**: Write the chosen decoder to request or group scratch and define fallback/rebinding on TargetChanged.
- **P1 — Exactly-once KV transfer is absent but metadata lists no deferred mechanics.**: Declare phase handoff/KV transfer deferred until the new lifecycle action and adapter exist.
- **P1 — N·Td≥R·Ld and N·B≥R·W provisioning checks are absent.**: Validate deployment facts offline or expose them in admission/control-plane tooling; do not fold replica creation into route.
- **P2 — The benchmark alternates independent requests with a preselected binding and does not test first-decode binding persistence.**: Replay one conversation across prefill completion and multiple generations, including restart and unavailable decoder.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Add trusted phase-specific route opportunities.
1. Persist decoder binding and test TargetChanged recovery.
1. Add first-prefill/first-decode/multi-generation traces.
1. Declare transfer deferred.
1. Introduce phase-handoff lifecycle before claiming exactly-once movement.

## parrot

### Improve with the current model

- **Batch routing could exceed target assignment counts and artifact metadata was wrong.** — The shared route helper now reserves max_assignments; metadata identifies OSDI 2024, pins ParrotServe commit 2e1825ee, and labels evidence inspired-adaptation. (Set-oriented route/schedule, target capacities, WorkGroups, multi-request selections, and host-provided DAG facts can express Parrot's scheduler and placement decisions.)

### New primitives

- **`workflow.graph@1`** — Trusted DAG facts can describe readiness, but the current lifecycle does not portably guarantee server-side activation of successor requests, dependency-failure propagation, or Semantic Variable submit/get execution. Minimal contract: Negotiated guarantee for a bounded WorkGroup-scoped DAG with host-issued node/request IDs, versioned edges, automatic readiness activation, terminal propagation, and trusted depth/criteria facts.

### Implementation gaps

- **P0 — dependency_distance placement is not Parrot's context-aware, criteria-aware, capacity-constrained engine selection.**: Use propagated criteria, task/token capacities, resident context/prefix affinity, and interference cost.
- **P0 — Scheduling only filters dependency_ready and otherwise preserves input order.**: Order by trusted graph depth/criteria, construct whole task-group selections, then apply shared-prefix and resident-context grouping.
- **P0 — The route helper still ignores aggregate named resource demand.**: Reserve each selected edge's resources while assigning or use a Parrot-specific capacity-aware placement loop.
- **P1 — No artifact-backed fact producer performs backward criteria/depth/task-group propagation.**: Integrate a bounded ParrotServe-equivalent graph traversal in the host adapter and version its output schema.
- **P2 — The benchmark scores readiness only and discards route outcomes.**: Run chain, map-reduce, and multi-agent DAGs with multiple engines, prefixes, and capacities; measure application makespan and sustainable throughput.

**Best evidence without new primitives:** `decision-trace-parity`

**Recommended sequence:**
1. Implement and validate the artifact-backed DAG fact producer.
1. Replace scheduling with depth, criteria, task-group, and prefix-aware logic.
1. Replace placement with resource- and interference-aware assignment.
1. Establish differential decision-trace parity against ParrotServe.
1. Add workflow.dependency-runtime@1 before claiming the complete Semantic Variable runtime.

## saga

### Improve with the current model

- **Routing could overfill targets and cache plans reclaimed every expired object.** — SAGA now reserves target max_assignments and uses capacity-bounded reclaim. (Route, schedule, cache, WorkGroup state, target facts, and cache-object facts already expose the coordinated decision inputs.)
- **Physical stealing was previously treated as a missing core operation.** — Metadata now correctly treats request.rebalance@1 as the optional movement intent and physical enactment as adapter evidence. (WorkGroup identity can represent the session, route can select the target, request.rebalance@1 can stage movement, and feedback can update affinity after success.)
- **Headline metrics were imprecise.** — Performance metadata now reports TCT speedups and separate approximate memory-utilization ratios. (Claim scope is separated from the currently implemented reduced kernel.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — Routing always maximizes locality and caller-supplied steal replaces the paper's 0.8 affinity gate, idle/load-ratio guards, victim selection, and oldest-session choice.**: Implement thresholded affinity routing and stateful work-steal selection from trusted worker/session facts; stage rebalance only for the chosen victim.
- **P0 — Cache admission implements only positive-TTL retention; WA-LRU, AEG reuse probability, adaptive tool TTL, pressure scaling, and expiry are absent.**: Implement the exact weighted eviction score, transition-probability overlap, per-tool duration state, timestamped TTL, and pressure rules.
- **P0 — Least group_service is not AFS and there is no feedback implementation for remaining work, deadlines, allocation epochs, or blocking preemption.**: Implement AFS scores, proportional token allocation, timer-driven epochs, and feedback-maintained group progress/deadline state.
- **P1 — Route assignment reserves only counts, not aggregate resource demands.**: Track named target-resource totals during assignment.
- **P1 — Rebalance action outcomes do not update session affinity or migrated-cache state.**: Add feedback, correlate action IDs, and update affinity only after successful enactment; verify adapter KV migration separately.
- **P2 — The benchmark disables stealing, ignores cache output, and hardcodes the implementation's locality-dominant utility.**: Use multi-step sessions under pressure and deadlines; compare vLLM+APC and component ablations using TCT, memory utilization, regeneration, and SLO attainment.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Define trusted AEG, tool-duration, worker-load, deadline, and session-queue schemas.
1. Implement WA-LRU and adaptive timestamped TTL.
1. Implement affinity routing, guarded work stealing, and action-feedback affinity updates.
1. Implement AFS epochs and feedback accounting.
1. Validate coordinated traces, then run vLLM+APC and ablation benchmarks.

## routebalance

### Improve with the current model

- **The previous implementation used exponential static global matching with a 12-request bound.** — It now performs LPT ordering, cost-budget filtering, normalized quality/cost/latency scoring, per-assignment dead-reckoned load updates, and capacity-aware direct assignments. (A set-oriented route invocation exposes the whole waiting batch, concrete model instances as targets, sparse request-instance edges, target state, capacities, and direct assignments; no model-selection primitive is needed.)
- **Artifact and dataset provenance were missing.** — The official artifact commit is pinned and metadata names the paper equations and decision flow. (Model identity, predicted quality, cost, and latency are ordinary target/edge facts over concrete deployed instances.)

### New primitives

- No new primitive is required for the stated evidence ceiling.

### Implementation gaps

- **P0 — The paper uses raw quality plus 1−cost/max(cost) and 1−latency/max(latency); the kernel min-max normalizes all three signals.**: Implement Eq. 1 exactly, preserving raw quality in [0,1] and maximum-only normalization for cost and latency.
- **P0 — Dead-reckoned latency is dimensionally different from T=tpot·(d/b+L): it adds latency_ms to tpot×queued_tokens and lacks decode batch size.**: Expose pending decode tokens and decode batch size per target, compute d/b iterations plus predicted output length, and update d after each dispatch.
- **P2 — Weights are not validated or normalized to the three-simplex.**: Require the three ppm weights to sum to 1,000,000 or normalize deterministically.
- **P1 — Only max_assignments is enforced; aggregate ResourceLimit demand is ignored.**: Apply shared target resource accounting during the LPT pass.
- **P2 — The benchmark scores realized assignments using initial queue state, omitting the policy's sequential dead-reckoning, and uses only a shortest-queue proxy baseline.**: Replay the official artifact's per-step state updates and add Avengers-Pro, BEST-Route, passthrough, and Semantic-Router decision traces.

**Best evidence without new primitives:** `faithful`

**Recommended sequence:**
1. Correct Eq. 1 normalization.
1. Correct the d/b+L latency model and update rule.
1. Validate weights and aggregate capacities.
1. Add artifact differential traces and paper baselines.
1. Upgrade evidence only after exact assignment parity.
