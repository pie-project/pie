# PLEX 0.6 Redesign and Delivery Plan

## Document status

- Status: implementation plan
- Target contract: `pie:plex@0.6.0`
- Normative contract: [`plex_0.6_contract.md`](plex_0.6_contract.md)
- Migration guide: [`plex_0.5_to_0.6.md`](plex_0.5_to_0.6.md)
- Implementation status:
  - Phase 0 complete: contract, Rust semantic types, WIT, schemas, and registry
  - Phase 1 complete: trusted group/request state, lifecycle, revisions, quotas,
    cleanup, deduplication, and scope-conflict metrics
  - Phase 2 complete: typed working sets, hard protocol bounds, direct plan
    validation, snapshot binding, opportunity retry tracking, cache episodes,
    and replay-order records
  - Phase 3 complete: package format 6, typed component ABI, active Rust
    runtime/API, guest SDK, and Python binding
  - Phase 4 complete: required/optional negotiation, standard action schemas,
    subject authorization, idempotent staging, outcome correlation, and
    cancellation cleanup
  - Phase 5 complete: vLLM/SGLang templates, generic/negative fixtures,
    unified build/replay runner, v0.5 trace archive, and the rebuilt existing
    five paper artifacts
- Inputs:
  - [`plex.md`](plex.md), the implemented v0.5 contract
  - [`plex_gap.md`](plex_gap.md), the design-gap audit
  - [`plex_serving_policy_report.md`](plex_serving_policy_report.md), the
    serving-policy replication survey
  - the design decisions recorded in the 0.6 discussion
- Compatibility: v0.6 is intentionally breaking

This document defines the 0.6 implementation sequence, paper-replication
program, validation strategy, and release gates. It is a delivery plan rather
than the normative contract. Normative semantics are frozen in
[`plex_0.6_contract.md`](plex_0.6_contract.md).

## 1. Goals

PLEX 0.6 will:

1. preserve a five-operation policy waist;
2. make `WorkGroup` and `Request` first-class, trusted subjects;
3. make every decision operation set-oriented, including admission;
4. replace the pressure-only `evict` operation with the broader `cache`
   operation;
5. expose typed structural APIs while retaining extensible policy facts;
6. distinguish policy-state commit from engine enactment;
7. reserve capability negotiation for optional engine mechanics, not for
   semantic variants already expressible by the core operations;
8. implement and validate every paper-replication candidate explicitly
   identified by the current literature report and gap audit; and
9. report replication strength without overstating deferred engine mechanics.

## 2. Non-goals

The 0.6 core will not provide:

- autoscaling, provisioning, or replica creation;
- model loading or parallelism-layout changes;
- physical KV transfer, swap, or migration implementations;
- attention kernels or tensor-batch construction;
- predictor training or deployment;
- a general workflow-DAG runtime;
- simultaneous physical execution of grouped requests;
- crash-atomic coordination with an external engine; or
- rollback or two-phase commit across PLEX state and engine state.

These may be represented by facts, versioned actions, or adapter-specific
mechanics, but they are not new core hooks.

## 3. Accepted design decisions

### 3.1 Five operations remain the stable waist

The Rust authoring model will be based on:

```rust
pub trait Policy {
    fn admit(
        ctx: AdmitContext,
        state: &mut PolicyState,
    ) -> Result<AdmitPlan, PolicyError>;

    fn route(
        ctx: RouteContext,
        state: &mut PolicyState,
    ) -> Result<RoutePlan, PolicyError>;

    fn schedule(
        ctx: ScheduleContext,
        state: &mut PolicyState,
    ) -> Result<SchedulePlan, PolicyError>;

    fn cache(
        ctx: CacheContext,
        state: &mut PolicyState,
    ) -> Result<CachePlan, PolicyError>;

    fn feedback(
        ctx: FeedbackContext,
        state: &mut PolicyState,
    ) -> Result<(), PolicyError>;
}
```

The operation names denote policy authority:

| Operation | Authority |
|---|---|
| `admit` | Which proposed requests may enter serving |
| `route` | Where admitted requests should execute |
| `schedule` | Which runnable requests receive service and how much |
| `cache` | Which resident or prospective objects remain cached |
| `feedback` | How enacted outcomes update policy state |

There is no sixth core hook. Prefetch, cancellation, swap, and migration are
versioned engine actions or guarantees.

### 3.2 The normal request path is admission before placement

The conceptual lifecycle becomes:

```text
admit -> route -> schedule ... -> feedback
```

`cache` is invoked independently when objects are created, memory pressure
changes, objects expire, or a dependency-constrained reclaim episode advances.

The adapter remains responsible for invoking operations at valid lifecycle
points. PLEX validates each invocation and transition but does not run the
engine lifecycle itself.

### 3.3 Every decision operation is set-oriented

The host always supplies arrays. A singleton decision is represented by an
array of length one; there is no separate single-request ABI.

```text
admit(candidates[])
route(requests[], targets[], feasible_edges[])
schedule(runnable[])
cache(resident[], prospective[])
feedback(records[])
```

An invocation is a bounded decision opportunity chosen by the host. It is not
defined as "everything that arrived at exactly the same time." The host must
record the opportunity boundary, deterministic input order, and snapshot
revision so replay observes the same decision.

Set-oriented core inputs make joint admission, joint assignment, and
set-dependent scheduling ordinary operation modes. They do not require
capabilities such as `route.batch-assignment` or
`schedule.bundle.selection`.

### 3.4 `WorkGroup` and `Request` are orthogonal first-class subjects

The subject model is:

```text
WorkGroup
`-- Request
    `-- Generation 0, 1, 2, ...
```

`Request` replaces the v0.5 term `LogicalRequest`. It is the independently
admitted, routed, scheduled, and completed serving flow. A generation is a
continuation of the same request.

`WorkGroup` is an optional trusted coordination scope shared by several
requests. It provides:

- principal ownership and authorization;
- group lifecycle;
- quotas and maximum fan-out;
- group deadlines and host facts;
- aggregate accounting; and
- group-private policy scratch.

A work group is not itself:

- a schedulable request;
- a workflow DAG;
- a scheduling bundle;
- a guarantee of co-location; or
- a guarantee of concurrent execution.

The relationship rules are:

1. A request belongs to zero or one work group.
2. Membership is fixed when the request identity is created.
3. The host issues and validates both identities and membership.
4. Copying a group ID into application metadata does not establish membership.
5. A request ID is also its group-member identity; there is no separate
   `MemberId`.
6. An independent request does not receive an implicit singleton group.
7. A request finishing does not finish its group.
8. A group may temporarily have no live requests.
9. A closed, cancelled, expired, or otherwise terminal group cannot accept a
   new request.

The minimum trusted lifecycle is:

```text
create_group
create_request(group_id?)
continue_request
finish_request
close_group
expire_group
```

Cancellation is requested through a versioned action and becomes terminal only
after enacted feedback.

### 3.5 The policy state model gains a group scope

The guest-visible state is:

```text
PolicyState
|-- shared
|-- groups<GroupId, GroupState>
`-- requests<RequestId, RequestState>
```

The minimum SDK model is:

```rust
pub struct GroupState {
    pub principal_id: PrincipalId,
    pub status: GroupStatus,
    pub limits: GroupLimits,
    pub member_count: u32,
    facts: Document,
    pub scratch: Document,
}

pub struct RequestState {
    pub status: RequestStatus,
    facts: Document,
    pub fields: Document,
    pub scratch: Document,
}
```

Host-owned identities, lifecycle state, principal, membership, quotas, and
observations are read-only facts. Policy scratch is mutable. Request fields
remain canonical mutable request data. Group fields are omitted from the
minimum model; host lifecycle updates can merge new group observations into
facts.

Each shared, group, and request scope has a host-private revision. When an
operation references a request, its work group, if any, is automatically
included once in the working set. Sibling requests are not automatically
loaded.

The implementation must avoid turning group state into an unconditional global
serialization point. High-frequency service measurements should normally be
delivered as host facts and reduced in bounded feedback batches. Benchmarks
must measure revision conflicts among concurrent requests in the same group.

### 3.6 PLEX provides conditional policy-state commit, not an external transaction

The SDK uses `PolicyState`, not `Transaction`, because PLEX does not provide an
end-to-end transaction with the engine.

For a new invocation, the runtime performs:

```text
load one coherent policy-state working set
-> invoke policy
-> validate the result and state update
-> compare revisions and conditionally commit policy state
-> return the normalized plan
-> engine revalidates and enacts the plan
-> engine reports enacted outcomes through feedback
```

Policy-state changes are committed only if:

- the policy returns successfully;
- the result and update are structurally and semantically valid; and
- every revision in the required read/write set still matches.

The commit does not imply that the engine enacted the returned plan. Therefore:

- decision hooks may record intent or attempt state;
- success-dependent counters are updated only from feedback;
- action success must not be recorded optimistically;
- no engine failure rolls policy state back automatically; and
- the specification must not claim two-phase enactment.

Feedback deliveries remain idempotent by delivery ID. Their policy-state
updates, terminal cleanup, and deduplication record commit together.

### 3.7 Capabilities describe optional mechanics only

The following are core semantics and therefore are not capabilities:

- batch admission;
- joint route assignment;
- direct set selection by the scheduler;
- selection of an all-or-none scheduling unit; and
- cache admission of prospective objects.

Capabilities are reserved for mechanics that an engine may genuinely lack,
for example:

```text
schedule.atomic-enqueue@1
request.cancel@1
group.cancel@1
cache.prefetch@1
cache.swap@1
request.rebalance@1
```

The manifest will distinguish:

```text
implements
requires
optional
```

Missing required mechanics fail attachment or invocation. PLEX must never
silently translate an unsupported operation into an approximate behavior.

## 4. Frozen operation contract summary

### 4.1 `admit`

`admit` receives all request candidates in one bounded admission opportunity:

```rust
pub struct AdmitContext {
    pub meta: DecisionMeta,
    pub cause: AdmitCause,
    pub candidates: Vec<AdmissionCandidate>,
    pub capacity: AdmissionCapacity,
}

pub struct AdmitPlan {
    pub decisions: Vec<AdmissionDecision>,
}

pub enum AdmissionDecision {
    Accept,
    Defer,
    Reject,
}
```

Required semantics:

- `decisions` is dense and aligned with `candidates`;
- every candidate has a host-issued pending request identity;
- group membership is already trusted and immutable;
- accepted candidates collectively satisfy the presented admission limits;
- deferred identities remain pending and may be presented again;
- rejected or expired pending identities are cleaned up idempotently;
- a batch view does not imply all-or-none admission; and
- candidate order and batching policy are replay-visible host behavior.

The host creates a pending request state before admission. Only accepted
requests may proceed to `route`. This permits a policy to retain bounded state
across deferrals without pretending that the request has entered execution.

### 4.2 `route`

`route` receives a sparse feasible assignment graph:

```rust
pub struct RouteContext {
    pub meta: DecisionMeta,
    pub requests: Vec<RouteRequest>,
    pub targets: Vec<RouteTarget>,
    pub feasible_edges: Vec<RouteEdge>,
}

pub struct RoutePlan {
    pub decisions: Vec<RouteDecision>,
}

pub enum RouteDecision {
    Assign { edge_index: u32 },
    Defer,
}
```

Required semantics:

- requests may be assigned jointly;
- a singleton request uses the same shape;
- `decisions` is dense and aligned with `requests`;
- every assignment references one supplied feasible edge for the aligned
  request;
- target count, token, byte, model, hardware, and locality limits are validated;
- assignments are tied to the supplied target/capacity revision;
- `Defer` keeps the request admitted but unplaced for a new opportunity; and
- the adapter owns queue insertion and physical placement.

The 0.6 plan is a direct bounded assignment, not only per-edge scores. This is
necessary to express policies whose objective depends on combinations of
request-target edges. If enactment fails, feedback records the failure and a
later route opportunity recomputes the assignment.

### 4.3 `schedule`

`schedule` receives the complete bounded runnable set for one service
opportunity and returns explicit selections:

```rust
pub struct SchedulePlan {
    pub selections: Vec<ScheduleSelection>,
}

pub struct ScheduleSelection {
    pub requests: Vec<RequestIndex>,
    pub token_budgets: Vec<u32>,
}
```

A selection with one request is ordinary scheduling. A selection containing
several non-overlapping requests is an all-or-none normalized selection unit.

Required semantics:

- every referenced request is runnable;
- a request appears in at most one selection;
- per-request and aggregate token limits are respected;
- a multi-request selection is accepted only if every member fits;
- work-group membership does not automatically create a selection unit;
- trusted workflow or adapter facts justify any required grouping;
- selection atomicity is part of the normalized core plan; and
- atomic queue insertion requires `schedule.atomic-enqueue@1`.

PLEX does not promise simultaneous GPU start or joint success.

### 4.4 `cache`

`cache` replaces `evict` and evaluates resident and prospective objects in one
snapshot:

```rust
pub struct CacheContext {
    pub meta: DecisionMeta,
    pub cause: CacheCause,
    pub resident: Vec<CacheObject>,
    pub prospective: Vec<CacheObject>,
    pub capacity: CacheCapacity,
    pub episode: Option<CacheEpisode>,
}

pub struct CachePlan {
    pub admissions: Vec<CacheAdmission>,
    pub reclaim: Vec<CacheObjectRef>,
}

pub enum CacheAdmission {
    Cache,
    Bypass,
}
```

The initial causes include:

```text
insertion
pressure
expiry
dependency-progress
```

Required semantics:

- `admissions` is dense and aligned with `prospective`;
- `Bypass` prevents persistent insertion even when free capacity exists;
- `reclaim` is an explicit legal reclaim order;
- the resulting retained set fits the presented capacity;
- a resident object may reference bounded request and group beneficiaries;
- the host validates dependency and capacity transitions;
- allocation and eviction remain engine mechanics; and
- enacted insertion, bypass, eviction, or failure is reported through feedback.

Dependency-constrained caches use a bounded iterative episode:

1. the host supplies the currently eligible frontier;
2. the policy chooses one victim or legal frontier;
3. the engine enacts the step;
4. newly eligible objects are presented with the same episode ID; and
5. the episode ends on satisfied capacity, explicit failure, or a hard
   iteration/time limit.

### 4.5 `feedback`

Feedback becomes a typed envelope around open, versioned outcome facts:

```rust
pub struct FeedbackContext {
    pub delivery_id: DeliveryId,
    pub records: Vec<FeedbackRecord>,
}

pub struct FeedbackRecord {
    pub subject: FeedbackSubject,
    pub outcome: OutcomeKind,
    pub facts: Document,
}
```

The initial subjects include request, work group, cache object, route
assignment, schedule selection, and action ID. Initial outcome kinds include
progress, completed, failed, cancelled, expired, action-succeeded, and
action-failed.

Required semantics:

- a successful delivery ID is applied exactly once;
- duplicate delivery returns the recorded semantic result;
- request and group accounting can update in the same policy-state commit;
- terminal cleanup is idempotent;
- a requested action is distinct from its enacted outcome; and
- success-dependent accounting occurs only here.

## 5. ABI, SDK, and package plan

### 5.1 Typed structure with extensible facts

WIT 0.6 will type the structural records, lists, indices, variants, and
operation results. Engine-specific `facts`, mutable request `fields`, and
policy `scratch` remain bounded JSON documents.

This avoids exposing the safety-critical shape as untyped `Document` while
preserving extensibility for paper-specific metrics and adapter observations.

The accepted component continues to:

- export exactly five policy functions;
- import only the PLEX host interface;
- use explicit input state and output updates;
- have no ambient state-loading API; and
- run with bounded memory, fuel, deadline, input, output, and host calls.

### 5.2 Host imports

The host interface remains method-oriented:

```text
query(method, args)
action(method, args)
```

Method names are independently versioned. The Rust SDK provides typed helpers
for standard methods while retaining a raw escape hatch whose use is visible
in package review and test fixtures.

### 5.3 Manifest and package

The 0.6 manifest will contain:

- exact contract version;
- package name and version;
- implemented operations;
- required and optional engine mechanics;
- resource limits; and
- fact/action schema requirements used by attachment validation.

The package format advances to version 6. A v0.5 component is never
reinterpreted as v0.6. Migration tooling may inspect both formats, but the
runtime must dispatch by exact contract and package version.

## 6. Implementation workstreams

### Phase 0: Freeze the normative 0.6 contract

Deliverables:

- [`plex_0.6_contract.md`](plex_0.6_contract.md), the normative versioned 0.6
  specification;
- canonical Rust semantic types in `interface/plex`;
- canonical WIT files for `pie:plex@0.6.0`;
- JSON schemas for extensible documents and replay fixtures;
- an action/capability registry; and
- [`plex_0.5_to_0.6.md`](plex_0.5_to_0.6.md), the explicit migration table.

Exit criteria:

- every context and plan has one unambiguous validation algorithm;
- lifecycle and failure state machines are documented;
- no operation relies on an undefined batching or retry rule; and
- the paper and implementation documents describe the same commit boundary.

### Phase 1: Implement subjects and state

Primary areas:

- `interface/plex`
- `runtime/policy/src/state_store.rs`
- `runtime/policy/src/lifecycle.rs`
- Rust and Python engine-facing APIs

Tasks:

1. Rename `logical_request_id` to `request_id`.
2. Add trusted work-group identity, principal, lifecycle, quotas, and facts.
3. Add immutable request-to-group membership.
4. Add pending, admitted, terminal, and expired request transitions.
5. Add group-scoped scratch and revisions.
6. Auto-join referenced groups into operation working sets.
7. Commit shared, group, and request updates with feedback deduplication and
   terminal cleanup.
8. Add deterministic group and request lifecycle replay.
9. Add conflict and contention metrics by state scope.

Required tests:

- a group outlives a finished child request;
- copied metadata cannot forge membership;
- a terminal group rejects new requests;
- request membership cannot be changed by a policy;
- duplicate group-terminal feedback has no additional effect;
- concurrent child feedback does not lose group accounting;
- unrelated groups do not conflict; and
- group state remains bounded by principal and manifest quotas.

### Phase 2: Implement set-oriented semantic validation

Primary areas:

- `interface/plex/src/operation.rs`
- `runtime/policy/src/protocol.rs`
- `runtime/policy/src/engine_api.rs`
- replay normalization and fixture tools

Tasks:

1. Replace all singleton decision schemas with array-based opportunities.
2. Implement dense admission validation.
3. Implement sparse-graph route assignment validation.
4. Replace score-only scheduling normalization with direct selection
   validation.
5. Implement multi-request selection-unit validation.
6. Replace eviction scoring with cache admission and reclaim validation.
7. Implement bounded dependency-constrained cache episodes.
8. Add typed feedback subjects and outcomes.
9. Record opportunity IDs, deterministic order, and snapshot revisions in
   replay traces.

Required tests include every counterexample in `plex_gap.md`.

### Phase 3: Implement WIT, runtime, and SDK 0.6

Tasks:

1. Generate and verify the new WIT surface.
2. Implement the typed Rust guest SDK.
3. Preserve read-only facts and private working-set membership.
4. Compute sparse shared/group/request state updates.
5. Update package validation, attachment ownership, and atomic replacement.
6. Update Wasmtime invocation, bounds, trap classification, and fallback.
7. Update Python host bindings and adapter helpers.
8. Add compile-fail or runtime-negative tests for invalid typed output,
   unknown subjects, membership mutation, and unsupported mechanics.

Exit criteria:

- all five operations compile and execute as v0.6 components;
- raw-WIT guests cannot bypass host semantic validation;
- v0.5 packages fail with an explicit version error; and
- singleton opportunities remain simple for policy authors.

### Phase 4: Implement standard optional mechanics

Define portable action requests and enacted feedback for:

- request and group cancellation;
- cache prefetch;
- cache swap or tier movement;
- request rebalancing or migration; and
- atomic scheduler enqueue where an adapter supports it.

Each action contract must define:

- authorization;
- idempotency;
- required engine capability;
- accepted, succeeded, failed, and already-terminal outcomes;
- timeout and expiry behavior;
- feedback correlation; and
- what remains engine-specific.

Mock adapters must implement success, failure, duplicate, stale-revision, and
unsupported-capability paths. A generic action name without an end-to-end
adapter test does not count as support.

### Phase 5: Migrate adapters and existing fixtures

Tasks:

1. Migrate vLLM and SGLang snapshot builders to set-oriented contexts.
2. Migrate all generic policy fixtures.
3. Rename `evict` fixtures and helpers to `cache`.
4. Migrate the existing five paper policies before adding new replicas.
5. Split the monolithic paper case file into independently reviewable,
   versioned cases.
6. Keep a v0.5 trace corpus to verify that intentional changes are documented.

## 7. Paper-replication program

### 7.1 Replication evidence levels

Every paper implementation must be assigned exactly one evidence level:

1. **End-to-end source replication**: the source policy and required mechanics
   are reproduced.
2. **Decision-trace parity with deferred mechanics**: identical snapshots
   produce the same decisions and policy-state transitions, while named
   physical mechanics remain in the adapter.
3. **Policy-kernel reproduction**: the central formula or bounded algorithm is
   reproduced, but the source system exposes a materially richer subject or
   mechanism.
4. **Inspired adaptation**: the implementation changes the subject or
   algorithm enough that parity is not claimed.

Only levels 1-3 are replication results. Level 4 remains useful as an example
but must not be counted as a replica.

### 7.2 Required artifact for every candidate

Each candidate gets:

```text
tests/policies/paper-<slug>/
tests/policies/replications/<slug>/metadata.json
tests/policies/replications/<slug>/cases/*.json
tests/policies/replications/<slug>/expected/*.json
```

`metadata.json` records:

- paper title and pinned version;
- source artifact URL, commit, license, and engine version when available;
- implemented policy kernel;
- required input facts and units;
- required engine mechanics;
- deliberately deferred mechanics;
- target evidence level;
- fixture provenance; and
- validation status.

Source code is not copied from an artifact unless its license permits it.
Golden traces may be generated from a pinned source checkout, but CI consumes
committed fixtures and does not depend on a mutable external repository.

### 7.3 Mandatory validation for every candidate

Each implementation must pass:

1. **Structural conformance**: exact 0.6 contexts, plans, limits, and manifest.
2. **Algorithm vectors**: hand-checked examples, boundary values, ties, empty
   sets, and overflow-safe arithmetic.
3. **Differential replay**: PLEX versus the pinned source artifact when one is
   available, otherwise versus an independently implemented paper reference.
4. **Stateful replay**: multi-step decisions, feedback, duplicate feedback,
   group lifecycle, and restart/replay.
5. **Metamorphic tests**: permutation stability where required, monotonicity,
   conservation of capacity/service, and deterministic tie handling.
6. **Negative tests**: missing facts, stale revisions, invalid plans,
   unsupported required mechanics, and action failure.
7. **Mechanics separation**: tests demonstrate which behavior is policy parity
   and which behavior is adapter-specific.
8. **Claim audit**: documentation states the achieved evidence level and names
   every deferred mechanic.

Reproducing headline paper performance numbers is a separate evaluation task.
It is required only for an end-to-end source-replication claim.

## 8. Exhaustive candidate matrix

The following list is exhaustive for candidates explicitly surfaced by the
current literature report and gap audit. Existing fixtures are not accepted as
complete until they meet the new implementation and validation targets.

### 8.1 Existing replicas to rebuild

| Candidate | 0.6 surface | Required implementation | Validation target |
|---|---|---|---|
| Autellix / Agentix | `schedule`, `feedback`, `WorkGroup` | Replace request-local LAS with group-level PLAS accounting; add ATLAS-compatible trusted critical-path/thread facts and anti-starvation behavior. | Paper-reference trace parity for PLAS and ATLAS; relabel any reduced implementation as inspired. |
| Continuum | `schedule`, `cache`, `feedback`, `WorkGroup` | Tool-duration-aware KV TTL, preempted/pinned priority, and group-level FCFS. | Differential replay against the pinned public artifact; verify TTL expiry and failed cache actions. |
| KVFlow | `schedule`, `cache`, optional `cache.prefetch@1` | Steps-to-execution retention, varying-suffix-first reclaim, cache-loading-aware scheduling, and prefetch intent. | Differential policy traces; separately test prefetch action support and report transfer overlap as deferred unless reproduced. |
| Preble | `route` | Complete E2 exploit/explore branches over set-oriented route inputs, including prefix locality, load, and eviction cost. | Differential replay against the public artifact; prefix replication and autoscaling are separate mechanics. |
| Helium | `schedule`, optional `cache.prefetch@1`, `WorkGroup` | Ready-operator filtering, critical-path priority, cache reuse, forced progress, and profiled token cost. | Artifact/reference trace parity; remove the incorrect bundle-fairness claim and test warming separately. |

### 8.2 Primary additions

| Candidate | 0.6 surface | Required implementation | Validation target |
|---|---|---|---|
| VTC | `schedule`, `feedback` | Exact per-client virtual-token counters and observed input/output-token charging. | Differential replay against the public artifact plus fairness invariants; target strong decision-trace parity. |
| FairServe | `admit`, `schedule`, `feedback` | Batch interaction-aware throttling and weighted service accounting. | Independent paper-reference model, overload sweeps, and no-starvation/service-conservation properties. |
| DLPM / D2LPM | `route`, `schedule`, `feedback` | Prefix-locality ordering gated by per-client deficit credits. | Reference trace parity and adversarial tests showing locality cannot violate bounded fairness. |
| InferCept | `schedule`, `cache`, `feedback`, optional `cache.swap@1` | Preserve, swap, or discard at API/tool boundaries using expected waste. | Differential replay against the public artifact; action success/failure and deferred swap mechanics tested separately. |
| PEEK | `schedule`, `cache` | Shared pending-prefix structure, cluster-aware LPM, fairness lane, and demand-depth retention. | Differential replay against the public artifact, including coordinated state transitions across both operations. |
| LMetric | `route` | Exact `new_prefill_tokens * current_batch_size` objective and hotspot guard. | Differential replay against the public router and exhaustive boundary tests around hotspot transitions. |
| DualMap | `route`, optional `request.rebalance@1` | Two-choice prefix hashing, SLO-aware target selection, and hotspot migration intent. | Differential replay against the public artifact; migration enactment is adapter evidence, not route parity. |
| QLM | `admit`, `route`, `schedule`, `feedback`, `WorkGroup` | Request groups, wait estimation, virtual queues, and bounded queue operations. | Independent reference model with multi-step queue and group lifecycle traces. |

### 8.3 Extended additions

| Candidate | 0.6 surface | Required implementation | Validation target |
|---|---|---|---|
| Llumnix | `route`, `feedback`, optional `request.rebalance@1` | Virtual-usage policy and live-rescheduling decisions. | Differential replay against the public artifact; separately validate migration request/outcome semantics. |
| SLOs-Serve | `admit`, `route`, `schedule` | Multi-stage SLO token planning, soft admission, token allocation, and replica routing. | Independent reference model across admission and routing epochs; explicitly defer provisioning. |
| Certaindex / Dynasor | `schedule`, `feedback`, optional `request.cancel@1` | Progress-aware reasoning allocation and early-stop intent. | Differential replay against the public artifact plus cancellation outcome tests. |
| Justitia | `schedule`, `feedback`, `WorkGroup` | Fair completion-order scheduling for task-parallel agent requests. | Independent reference model across branch arrivals and completions; no implicit atomic-execution claim. |
| RAGCache | `cache` | Prefix-aware GDSF, legal leaf selection, iterative eligibility, and clock updates. | Paper-reference trace parity including parent/child legality and bounded multi-step reclaim episodes. |
| Chameleon | `admit`, `schedule`, `cache` | Weighted-size queues, per-queue token quotas, bypass, adapter caching, and starvation control. | Independent cross-operation reference model with heterogeneous request and adapter sizes. |

### 8.4 Temporal holdout and composition candidates

| Candidate | 0.6 surface | Required implementation | Validation target |
|---|---|---|---|
| SMetric | `route`, optional `request.rebalance@1` | Load-balance first turns, cache-affine follow-ups, and tail-outlier migration intent. | Independent reference traces proving the five-operation waist does not change. |
| ThunderAgent | `schedule`, `cache`, `feedback`, optional rebalance/cancel actions, `WorkGroup` | State-aware pausing, program migration intent, and tool-resource lifecycle. | Differential policy traces against the public artifact; physical pause/migration is separately classified. |
| Pythia | `route`, `schedule`, `cache`, `feedback`, optional prefetch | Workflow lookahead, Belady-like cache choices, and scheduling composition. | Independent reference model; proactive scaling remains explicitly outside PLEX core. |
| GoodServe | `route`, `feedback`, optional `request.rebalance@1` | Just-enough heterogeneous target selection and risk-triggered migration using recorded predictor facts. | Reference replay with fixed predictor outputs; predictor training and provisioning are excluded. |
| Conversation-level ConServe | `route`, `Request` generations | Place the heavy initial prefill and preserve conversation-tail affinity. | Reference traces over multiple generations, restart, and target-unavailable fallback. |
| Parrot | `route`, `schedule`, `WorkGroup` | Consume trusted bounded DAG/readiness facts for dependency-aware placement and execution. | Reference traces demonstrating group identity without making a general DAG runtime part of PLEX. |
| SAGA | `route`, `schedule`, `cache`, optional rebalance, `WorkGroup` | Workflow TTL, group fairness, and cache-local work stealing. | Independent coordinated replay; source-maturity limitations remain explicit. |
| RouteBalance | `route` | Joint model/target assignment and load balancing over a sparse feasible graph. | Paper-reference matching cases, including the non-greedy counterexample from the gap audit. |

### 8.5 Surface-boundary candidates

| Candidate | 0.6 surface | Required implementation | Validation target |
|---|---|---|---|
| Marconi | `cache`, `feedback` | Prospective cache admission using reuse likelihood and FLOPs per byte, plus resident comparison. | Differential replay against the public artifact; verify bypass with free capacity and high-value replacement. |
| HotPrefix | `cache`, `feedback`, optional `cache.prefetch@1` | Hotness tracking, selective admission, and promotion intent. | Independent reference traces for cache pollution prevention and tier-action outcomes. |
| PARD | `schedule`, `feedback`, optional `request.cancel@1` | Mid-lifecycle drop intent from upstream elapsed time and downstream latency distributions. | Reference traces proving cancellation is not misrepresented as initial admission. |
| Regulating Branch Parallelism | `admit`, `schedule`, optional cancellation, `WorkGroup` | Jointly admit and limit reasoning branches against co-batched latency. | Reference batch-admission and branch-control traces; no forged group or branch identity. |

### 8.6 Mapping-only systems

The following systems remain corpus mappings rather than committed full-system
replicas because their central contribution is control-plane or physical
mechanics: Helix, InferLine, INFaaS, MArk, Aladdin, KAIROS, PolyServe, and
Dyserve. A compact online policy kernel may be added later, but PLEX 0.6 must
not imply that it reproduces provisioning, model loading, GPU layout, or
kernel behavior.

## 9. Replication implementation waves

### Wave A: contract sentinels

Implement first:

- VTC for feedback-based service accounting;
- LMetric for set-oriented routing;
- FairServe for batch admission;
- Marconi for prospective cache admission; and
- RAGCache for iterative cache legality.

These policies each isolate one critical 0.6 semantic and become contract
regression tests.

### Wave B: rebuild the existing five

Rebuild Agentix, Continuum, KVFlow, Preble, and Helium without carrying forward
their known v0.5 overclaims.

### Wave C: coordinated multi-operation policies

Implement DLPM/D2LPM, InferCept, PEEK, QLM, SLOs-Serve, Certaindex/Dynasor,
Justitia, Chameleon, HotPrefix, PARD, and Regulating Branch Parallelism.

### Wave D: movement and temporal holdout

Implement DualMap, Llumnix, SMetric, ThunderAgent, Pythia, GoodServe, ConServe,
Parrot, SAGA, and RouteBalance.

No wave is complete until its fixtures, reference oracle, negative tests,
evidence classification, and documentation are committed.

## 10. Validation infrastructure

### 10.1 Unified runner

Extend the existing policy fixture pipeline to:

1. build every v0.6 guest component;
2. package and attach it with declared mechanics;
3. replay all deterministic cases;
4. compare raw policy plans, normalized plans, state updates, actions, and
   feedback effects;
5. run each case with supported and deliberately missing mechanics;
6. verify duplicate feedback and restart replay; and
7. emit a machine-readable replication report.

The existing commands remain the starting point:

```bash
cargo test --locked -p pie-plex -p pie-policy --all-targets
cargo test --manifest-path sdk/rust/plex/Cargo.toml --locked
scripts/build-plex-policies.sh
cargo run --locked -p pie-policy --example check_fixtures -- \
  tests/policies/built tests/policies
```

They will be updated rather than replaced by a new test ecosystem.

### 10.2 Cross-cutting contract suites

Add dedicated suites for:

- batch boundaries and deterministic ordering;
- dense admission output;
- sparse route assignment and capacity revisions;
- direct schedule selection and bundle overlap;
- prospective cache admission;
- iterative dependency-constrained reclaim;
- work-group authorization and lifecycle;
- group contention and revision conflicts;
- policy-state commit versus engine enactment;
- action failure and feedback correlation;
- package capability negotiation; and
- v0.5 rejection and migration diagnostics.

### 10.3 Performance validation

Benchmark:

- singleton versus batched admission;
- route cost versus request, target, and sparse-edge counts;
- scheduling cost versus runnable and selection-unit counts;
- cache cost versus resident, prospective, and episode depth;
- group pre-join and state-commit cost;
- same-group conflict rate under concurrent feedback;
- JSON extension-field serialization;
- Wasm crossing, fuel, and memory; and
- full adapter decision latency.

Before implementation, record the v0.5 singleton baseline and approve explicit
0.6 regression budgets. Batch size, edge count, context bytes, output bytes,
fuel, iteration count, and deadlines all require hard host limits.

## 11. Documentation and claim updates

Update:

- `plex.md` to the normative 0.6 contract;
- `plex_paper.md` to match the implemented commit boundary and subject model;
- README and SDK examples;
- Python adapter documentation;
- the serving-policy report and machine-readable catalog;
- each paper page with achieved evidence level; and
- fixture metadata with source and license provenance.

Required terminology:

- use `Request`, not `LogicalRequest`;
- use `cache`, not `evict`, for the 0.6 authority;
- use "policy-state commit," not "engine transaction";
- distinguish selection atomicity from atomic enqueue and simultaneous
  execution; and
- distinguish source replication, decision-trace parity, policy-kernel
  reproduction, and inspired adaptation.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Batch windows add latency or adapter-dependent behavior | Make opportunity boundaries explicit, bounded, deterministic, and trace-visible. |
| Joint route graphs grow as requests times targets | Require sparse feasible edges and hard request, target, edge, byte, fuel, and deadline limits. |
| Work-group revisions become contention hotspots | Avoid unnecessary group writes, batch feedback, measure conflicts, and keep siblings out of the implicit working set. |
| Direct plans move more work into guest policies | Keep host feasibility validation complete and provide SDK combinators for common ranking, matching, and fill strategies. |
| `WorkGroup` expands into a workflow runtime | Keep DAG, bundle, and physical execution semantics separate and fact/action based. |
| `cache` becomes engine-specific | Stabilize only object identity, capacity, prospective/resident status, beneficiaries, and plan semantics; leave movement mechanisms to actions. |
| Optional actions are mistaken for supported mechanics | Require declared capabilities and end-to-end mock or real adapter tests. |
| Public artifacts are unavailable or incompatible | Use an independent reference model, record provenance, and cap the evidence claim accordingly. |
| Thirty-one paper implementations create scope pressure | Deliver in waves, with contract sentinels first, but do not remove candidates from the release matrix silently. |
| Paper and implementation semantics diverge again | Generate the replication report from committed fixture metadata and audit it as a release gate. |

## 13. Release gates

PLEX 0.6 is complete only when:

1. all five typed operations are implemented end to end;
2. every decision operation accepts arrays and passes singleton and batch tests;
3. trusted work-group identity, lifecycle, quotas, and state are implemented;
4. request membership cannot be forged or mutated;
5. cache admission compares prospective and resident objects before persistent
   insertion;
6. policy-state commit and engine enactment are documented and tested as
   separate boundaries;
7. required mechanics fail explicitly when unsupported;
8. every candidate in Section 8 has an implementation, reference oracle,
   deterministic fixtures, negative tests, and evidence classification;
9. the paper, design document, SDK, runtime, and adapters use consistent
   terminology and semantics;
10. the full Rust, SDK, fixture, Python, replay, and adapter test suites pass;
11. performance and state-conflict results are recorded against the approved
    budgets; and
12. no replication claim is stronger than its committed evidence.

## 14. Final target

PLEX 0.6 should demonstrate that a small five-operation waist can express:

- independent and grouped request lifecycles;
- batch admission;
- joint placement;
- direct set and bundle selection;
- cache admission and eviction in one authority;
- stateful cross-operation policies; and
- enacted-outcome accounting,

without absorbing control-plane provisioning or engine mechanisms into the
portable policy ABI.

The release succeeds when the programming model is simpler for ordinary
single-request policies, strictly more expressive for grouped and set-dependent
policies, and validated by the complete replication matrix rather than by
surface-level examples alone.
