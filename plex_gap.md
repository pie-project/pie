# PLEX Design Gap Audit and Remediation Plan

## 1. Purpose

This document compares the current PLEX v0.5 contract against a corpus of 87
LLM serving-policy papers. It identifies important policy semantics that the
current contract cannot faithfully express, observe, or enforce.

The accepted v0.6 resolution is frozen in
[`plex_0.6_contract.md`](plex_0.6_contract.md). Recommendations below remain the
design rationale for that contract, not an alternative normative surface.

The audit does not treat every mechanism implemented by a paper as something
PLEX must absorb. For each apparent gap, it asks:

1. Is a core policy authority missing?
2. Does an existing authority have the wrong subject, input snapshot, or result
   shape?
3. Would a standard optional capability or action be sufficient?
4. Is the missing behavior physical engine or control-plane mechanics that
   should remain outside PLEX?
5. Is the feature already present in the conceptual paper design but absent
   only from the v0.5 prototype?

The audit cutoff is 2026-07-23. Where an ACM or USENIX full text was not
accessible, the review used official abstracts, author copies, presentations,
and public artifacts.

## 2. Executive conclusion

The five PLEX core authorities remain defensible:

- `admit`: whether serving work should enter the system;
- `route`: where that work should execute;
- `schedule`: which work receives service and how much;
- `evict`: which state remains resident; and
- `feedback`: how enacted outcomes update policy state.

The literature does not justify adding a sixth core hook. Instead, the current
contract is too narrow along three dimensions:

- **Subject**: PLEX has sequential logical requests, but no trusted program or
  work-group subject that can outlive several child requests.
- **Invocation mode**: `route` sees one request at a time, and `evict` sees one
  fixed resident set.
- **Result constraint**: `schedule` returns decomposable per-candidate scores,
  so it cannot enforce an all-or-none bundle.

The resulting priorities are:

| Priority | Item | Assessment |
|---|---|---|
| P0 | Trusted work-group lifecycle | Required to make broad agent/program-serving claims |
| P0 | Residency admission/insertion | Required to make broad cache-policy coverage claims |
| P0 | Align paper and v0.5 transaction semantics | Required before publication |
| P1 | Joint batch assignment | Optional route capability or explicit limitation |
| P1 | Atomic workflow bundle | Optional schedule capability or explicit limitation |
| P1 | Standard cancel/drop and branch control | Optional standard action |
| P1 | Iterative dependency-constrained eviction | Needed for strong RAGCache-class reproduction |
| P2 | Autoscaling, KV movement, and kernel changes | Adapter or control-plane mechanics, not a PLEX core gap |

## 3. Current PLEX baseline

The relevant constraints in [`plex.md`](plex.md) are:

1. A logical request may span several generations, but its generation IDs form
   the exact sequence `0, 1, 2, ...`.
2. Request-state lifecycle and transaction working sets are keyed by
   `logical_request_id`.
3. `route` receives one logical request and that request's candidate targets.
4. `schedule` sees the complete runnable set, but returns only a `score` and an
   optional `token_budget` for each candidate.
5. The host applies a fixed greedy fill rule in descending score order until
   capacity is exhausted.
6. `evict` receives an already-resident, fixed candidate set and reclaims the
   lowest retention scores until `bytes_needed` is satisfied.
7. Generic staged actions exist, but portable cancel, drop, fork, and join
   semantics do not.
8. v0.5 commits policy state before the engine enacts a returned decision or
   action. Actual success or failure arrives in later feedback.

These constraints work well for per-request routing, priority scheduling,
scalar retention, and feedback accounting. They lose semantics when a decision
depends on a longer-lived group, a prospective cache object, a matching across
several requests, or a non-decomposable group constraint.

---

## 4. Gap 1: Trusted work-group lifecycle

### 4.1 Logical requests and work groups are different subjects

A current PLEX logical request represents one admitted unit of serving work. It
may span multiple generations and tool pauses. That model is appropriate for a
single sequential execution flow, including a multi-turn conversation or an
InferCept-style pause and continuation.

Agentix, Parrot, Justitia, and similar program-serving systems can create
several LLM calls in parallel:

```text
Program G
|-- Logical request A
|   |-- generation 0
|   `-- generation 1
|-- Logical request B
`-- Logical request C
```

The following properties can all hold:

- A, B, and C route, schedule, and finish independently.
- G remains live after A finishes because B or C is still live.
- Fairness and deadlines are accounted to G rather than to an individual
  child.
- A branch result may create or cancel other branches.
- The program has terminal semantics distinct from the terminal event of any
  one child request.

A work group is therefore above logical requests and can outlive several of
them.

### 4.2 Why logical-request scratch is insufficient

A policy can write an object like this into one child's scratch:

```json
{
  "program_id": "G",
  "program_service": 100
}
```

That does not create a proper group lifecycle.

#### Lifetime

When the owner request finishes, its request state is removed. The group must
remain live while another child is still active.

#### Replication and consistency

If A, B, and C each hold a copy of the group aggregate, the contract does not
identify the authoritative copy. Concurrent feedback may conflict or lose an
aggregate update.

#### Working-set completeness

A hook loads only the request states directly referenced by its validated
context. Scheduling A does not automatically join B, C, or a group aggregate
into the same transaction working set.

#### Trust

An application-provided `program_id` is a declaration, not identity proof. A
tenant must not be able to attach itself to another program to share fairness
credit, deadlines, or cache residency.

#### Lifecycle

Request scratch has no standard fork, join, member-finish, group-finish,
cancellation, or expiry transition. It cannot define when stale members or
abandoned groups are collected.

#### Isolation and quotas

There is no standard host-enforced location for group membership limits, state
size limits, maximum fan-out, or principal ownership.

### 4.3 Why a map in `State.shared` is still insufficient

A prototype can store group state under
`State.shared["groups"][group_id]`. This avoids tying the data to one child's
lifetime, but `shared` is policy-backend-wide global scratch.

It still lacks:

- host-authenticated membership;
- group creation and terminal transitions;
- automatic group-scoped cleanup;
- authorization rules for reading a group record;
- a group-specific revision and working set;
- scalable conflict isolation between unrelated groups; and
- explicit principal and quota semantics.

Such a map is a useful emulation for an experiment, but it is not a portable
group contract.

### 4.4 Literature impact

- [Agentix/Autellix](plex-serving-policy-wiki/papers/autellix-agentix.md):
  PLAS uses cumulative program service, while ATLAS uses program/thread
  structure and critical-path service.
- [Parrot](plex-serving-policy-wiki/papers/parrot.md):
  scheduling and placement use a semantic-variable DAG.
- [Justitia](plex-serving-policy-wiki/papers/justitia.md):
  completion order is managed for task-parallel agents.
- [Helium](plex-serving-policy-wiki/papers/helium.md):
  scheduling uses ready operators and critical paths in workflow DAGs.
- [SAGA](plex-serving-policy-wiki/papers/saga.md):
  fairness and lifecycle are defined over workflows.

Per-request state can approximate formulas from these papers, but it cannot
provide source-faithful program semantics.

### 4.5 Recommended design: a separate trusted `WorkGroup`

PLEX should not redefine a logical request to mean an entire program. Doing so
would make independently routed, scheduled, and completed parallel children
difficult to represent. It should add a separate layer:

```text
WorkGroup
  group_id
  principal_id
  kind
  parent_group_id?
  lifecycle_state
  facts
  fields
  scratch
  revision

LogicalRequest
  logical_request_id
  group_id?
  member_id?
  parent_member_id?
  generation_id
  ...
```

The minimum rules should be:

1. A host-issued capability creates `group_id` and group membership.
2. A matching string in caller metadata is not identity evidence.
3. A group may have zero or more live logical requests.
4. Finishing a child does not imply finishing the group.
5. No new member may be attached after a terminal group event.
6. Group terminal cleanup is idempotent and coordinated with terminal
   feedback and state updates.
7. Group state has its own revision, byte quota, and member-count quota.
8. When a hook references a member request, the trusted group state is
   pre-joined into the working set according to declared rules.

### 4.6 Minimum lifecycle events

The contract needs at least the following meanings:

| Event | Meaning |
|---|---|
| `group-create` | Create a trusted group identity and initial state |
| `member-attach` | Bind a logical request to a group |
| `fork` | Create one or more child members from an existing member |
| `join` | Observe dependency completion or a join barrier |
| `member-finish` | Record a child's terminal outcome |
| `group-finish` | Record group completion, cancellation, or expiry |

An ABI could combine `fork` and `member-attach`. The exact event names are less
important than these invariants:

- membership cannot be forged;
- parent/child relationships are host-observed;
- a group can outlive a child; and
- group terminal effects occur exactly once.

### 4.7 Interaction with existing hooks

No new core policy hook is required:

- `route` sees member identity and group placement history.
- `admit` distinguishes admission of a new group from admission of a new
  member.
- `schedule` sees group-level attained service, deadlines, and ready
  dependencies.
- `evict` sees group beneficiaries and future reuse.
- `feedback` updates member service and group aggregates in one transaction.

Work groups add a higher-level subject and state scope. They do not create a
new resource authority.

---

## 5. Gap 2: Residency admission and insertion

### 5.1 Request admission and residency admission are different decisions

The overloaded word "admission" can hide an important distinction:

| Decision | Question | Time |
|---|---|---|
| Request admission | Should this serving work execute? | Before execution |
| Residency admission | Should newly created KV/cache state remain cached? | When state is created |
| Eviction | Which already-resident state should be removed? | Under memory pressure |

A request may execute successfully while the KV state it creates is deliberately
not inserted into the persistent cache. Accepting a request therefore does not
imply accepting all resulting state into residency.

### 5.2 Concrete example

Suppose GPU cache objects A, B, and C are already resident, and execution
creates a new prefix-KV object P:

```text
Current residents: A, B, C
Prospective object: P
```

The residency policy must choose between:

```text
Do not retain P.
```

and:

```text
Retain P and remove a lower-value object among A, B, and C.
```

[Marconi](plex-serving-policy-wiki/papers/marconi.md) uses reuse likelihood and
FLOPs per byte. [HotPrefix](plex-serving-policy-wiki/papers/hotprefix.md) uses
hotness to make selective admission decisions for new entries.

### 5.3 Why the current `evict` contract cannot express this

The current input is effectively:

```text
evict(
  resident = [A, B, C],
  bytes_needed = N
)
```

P is not resident yet, so it is absent from the candidate set. The policy
cannot compare P against existing objects. Before invoking PLEX, the engine
must already have selected one of these behaviors:

1. always insert P;
2. always discard P; or
3. apply a separate fixed admission rule in the adapter.

In every case, part of the residency policy remains outside PLEX.

### 5.4 Why "insert first, then immediately evict" is not equivalent

An adapter might insert P and then include it in a later eviction call. That is
not generally semantics-preserving:

- a valuable existing object may already have been displaced to allocate P;
- unnecessary allocation, copy, and metadata work has occurred;
- recency, frequency, or GDSF clock state may change;
- no eviction hook may run when free memory is currently sufficient; and
- admission rejection is reported as a mechanism event rather than a policy
  decision.

Selective admission may reject P even with free capacity to prevent future
cache pollution. A pressure-only eviction hook cannot express that behavior.

### 5.5 Recommended design: extend the existing residency authority

PLEX does not need a sixth core hook. Conceptually, `evict` already controls
which state remains resident. The ABI can preserve that authority while adding
an insertion cause and prospective candidates:

```json
{
  "cause": "insertion",
  "resident": [
    {
      "id": "A",
      "size_bytes": 5368709120,
      "facts": {"reuse": 0.8}
    }
  ],
  "prospective": [
    {
      "id": "P",
      "size_bytes": 2147483648,
      "facts": {
        "reuse": 0.2,
        "recompute_flops": 1000000
      }
    }
  ],
  "capacity": {
    "available_bytes": 1073741824
  }
}
```

A prospective decision needs at least two values:

```json
{
  "admit": false,
  "retention_score": 10.0
}
```

- `admit: false` prevents persistent insertion even when capacity is free.
- `admit: true` allows P to compete for residency.
- `retention_score` compares an admitted prospective object with existing
  residents when pressure must be resolved.

### 5.6 Deterministic normalization

One compatible deterministic rule is:

1. Remove every prospective object with `admit: false` from the insertion set.
2. Compute the byte deficit assuming all remaining prospective objects enter.
3. Sort existing residents and admitted prospective objects by ascending
   retention score.
4. Reclaim the lowest scores until the deficit is satisfied.
5. Reclaiming an existing resident means physical eviction.
6. Reclaiming a prospective object means avoiding its allocation.

For example:

| Object | Kind | Size | Retention score |
|---|---|---:|---:|
| A | Resident | 5 GB | 80 |
| B | Resident | 3 GB | 20 |
| P | Prospective | 4 GB | 70 |

If inserting P creates a 2 GB deficit, the host evicts B and admits P. If P's
score is 10, the host reclaims P itself, meaning that P is never inserted.

### 5.7 Enactment semantics

The policy should evaluate one coherent snapshot, but PLEX cannot require
physical eviction and allocation to be crash-atomic on every engine.

The minimum portable guarantees are:

- the adapter presents P before persistent allocation;
- PLEX returns one normalized residency plan;
- the adapter revalidates the plan before enactment;
- actual eviction and insertion outcomes return through feedback; and
- the policy does not record success optimistically.

An engine that supports insertion reservations or rollback may advertise a
stronger optional atomicity capability.

### 5.8 Alternative: a retention-intent map

A policy could write a bounded intent before insertion:

```text
prefix P -> do-not-cache
```

This can reduce hot-path callbacks, but it has important limitations:

- P and current residents are not compared in one snapshot;
- intent expiry and stale predictions require explicit semantics; and
- the system must know a stable cache-object identity before creation.

A retention-intent map is a useful optimization, but it is not a general
replacement for residency-admission semantics.

---

## 6. Gap 3: Joint batch assignment

### 6.1 Seeing all targets is not the same as assigning all requests jointly

Current `route` sees all targets for one request:

```text
route(request=A, candidates=[X, Y])
route(request=B, candidates=[X, Y])
```

This is sufficient for per-request routing. It is not sufficient when several
requests compete for the same target capacity and the optimal result depends
on the combination of request-target edges.

### 6.2 Counterexample

Suppose X and Y can each accept one request, with these utilities:

| | X | Y |
|---|---:|---:|
| Request A | 10 | 9 |
| Request B | 8 | 0 |

Both requests independently prefer X. If A is processed first:

```text
A -> X: 10
B -> Y: 0
Total utility: 10
```

A joint matching over the same snapshot yields:

```text
A -> Y: 9
B -> X: 8
Total utility: 17
```

The correct result depends on a combination of edges, not on one request's
target ranking.

### 6.3 Why sequential `route` plus shared state is not equivalent

A policy can record A's placement in shared state before routing B. That
implements an order-dependent online greedy algorithm, not a joint
optimization:

- A and B do not observe one immutable capacity snapshot.
- A's decision cannot be revised after B becomes visible.
- cluster state can change between invocations;
- both assignments and their state updates do not commit as one transaction;
  and
- changing replay order may change the result.

The distinction matters for work such as
[RouteBalance](plex-serving-policy-wiki/papers/routebalance.md), which combines
model routing with load balancing, and for policies that compute assignment
epochs.

### 6.4 Recommended design: an optional batch mode

This is still placement authority, so it should not become a new core hook. It
should be an optional `route` invocation mode:

```text
route.mode =
  per-request
  | batch-assignment
```

An input could be:

```json
{
  "cause": "assignment-epoch",
  "requests": [
    {"request_id": "A"},
    {"request_id": "B"}
  ],
  "targets": [
    {"id": "X", "capacity": 1},
    {"id": "Y", "capacity": 1}
  ],
  "feasible_edges": [
    {
      "request_index": 0,
      "target_index": 0,
      "facts": {"utility": 10}
    },
    {
      "request_index": 0,
      "target_index": 1,
      "facts": {"utility": 9}
    },
    {
      "request_index": 1,
      "target_index": 0,
      "facts": {"utility": 8}
    },
    {
      "request_index": 1,
      "target_index": 1,
      "facts": {"utility": 0}
    }
  ]
}
```

A dense, request-aligned result avoids repeating identifiers:

```json
{
  "targets": [1, 0]
}
```

If `null` is allowed, its defer or unassigned meaning and retry behavior must
be explicit.

### 6.5 Host validation

The host must validate at least:

- every selected pair exists in `feasible_edges`;
- each request receives at most one target;
- target count, byte, token, and GPU capacities are respected;
- model, region, hardware, and stage compatibility are respected; and
- the result references the same target/capacity revision as the input.

The policy should never return raw cluster mutations. It selects only a
feasible assignment; the adapter owns queue insertion and placement enactment.

### 6.6 Edge scores versus direct assignments

There are two plausible result designs:

1. the policy returns edge scores and the host runs a standard matching solver;
2. the policy returns a complete feasible assignment.

The first option is easier to validate and can improve portability, but it
restricts every policy to one host-defined matching objective and solver. The
second option is more general but moves combinatorial work into the policy and
increases validation cost.

For a policy-research interface, a bounded direct assignment is more honest.
PLEX can control cost through batch-size, edge-count, memory, and deadline
limits.

### 6.7 Boundary between route and the control plane

Reasonable online route targets include:

- an existing replica;
- an already-loaded model variant; and
- a feasible prefill/decode execution plan.

The following normally belong to the control plane:

- creating a new replica;
- loading or unloading a model;
- changing GPU count or parallelism layout; and
- long-horizon autoscaling and provisioning.

The full contributions of
[Aladdin](plex-serving-policy-wiki/papers/aladdin.md), Helix, InferLine,
INFaaS, and MArk must not be presented as if they were only batch routing.
PLEX can express an online assignment kernel while provisioning mechanics
remain outside the core.

---

## 7. Gap 4: All-or-none workflow bundles

### 7.1 Strength and limitation of current scheduling

Current `schedule` sees the complete runnable set for one scheduling
opportunity. It can compute set-dependent priorities from waiting time,
relative rank, tenant service, locality, or other candidate facts.

Its result is nevertheless decomposed into one score and token budget per
candidate:

```text
candidate 0 -> score, budget
candidate 1 -> score, budget
candidate 2 -> score, budget
```

The host sorts individual candidates and greedily fills capacity. A policy
cannot declare that several candidates form one indivisible scheduling unit.

### 7.2 Counterexample

Suppose workflow W has parallel branches W1 and W2 that are useful only when
both run in the same opportunity. X is an independent request:

```text
Bundle W = {W1, W2}
Capacity = 2 requests
```

The policy returns:

| Candidate | Score |
|---|---:|
| W1 | 100 |
| W2 | 99 |
| X | 99.5 |

The current host rule selects W1 and X:

```text
selected = [W1, X]
```

The policy may instead require one of these legal outcomes:

```text
selected = [W1, W2]
```

or:

```text
selected = [X]
```

Per-candidate scores cannot communicate that `{W1, W2}` is an all-or-none
constraint.

### 7.3 Why equal scores do not provide a guarantee

A policy can approximate a bundle by assigning W1 and W2 the same very large
score. The contract still does not guarantee atomic selection:

- stable tie order may interleave another candidate;
- aggregate token capacity may fit only one member;
- members may have different maximum budgets;
- the adapter may enqueue one member and fail to enqueue another; and
- no reservation protects the next scheduling step.

Scores express preference. They do not create structural constraints.

### 7.4 Why scratch cannot solve it

Scratch can store a `bundle_id` and expected member count, but host
normalization does not consume those values. Even if the policy records that
W1 implies W2, the engine retains selection and enactment authority.

Policy state cannot create a new engine feasibility constraint.

### 7.5 Recommended design: an optional structured scheduling unit

The minimum useful extension is a non-overlapping all-or-none bundle:

```json
{
  "runnable": [
    {
      "request_id": "W1",
      "max_token_budget": 4
    },
    {
      "request_id": "W2",
      "max_token_budget": 4
    },
    {
      "request_id": "X",
      "max_token_budget": 8
    }
  ],
  "bundles": [
    {
      "id": "W",
      "members": [0, 1],
      "mode": "all-or-none"
    }
  ],
  "capacity": {
    "max_selected": 2,
    "max_total_tokens": 8
  }
}
```

Bundle membership must come from the trusted work-group lifecycle or from
adapter-validated feasible structure, not from untrusted application metadata.

A result can treat the bundle itself as one scheduling unit:

```json
{
  "bundle_decisions": [
    {
      "score": 100,
      "member_token_budgets": [4, 4]
    }
  ],
  "single_decisions": [
    {
      "candidate_index": 2,
      "score": 90,
      "token_budget": 8
    }
  ]
}
```

The host selects the bundle only when all of its members fit.

### 7.6 Distinguishing guarantee levels

"Atomic bundle" can refer to three different guarantees:

#### Selection atomicity

Every member appears in the normalized decision, or none does.

#### Admission/enqueue atomicity

The adapter accepts all scheduler enqueues, or rejects all of them.

#### Simultaneous physical execution

All members start on GPUs at the same time and succeed together.

Only selection atomicity is broadly portable in the PLEX core. Atomic enqueue
requires an adapter capability. Simultaneous execution is engine mechanics and
is not generally available in LLM schedulers.

The manifest should therefore distinguish capabilities such as:

```text
schedule.bundle.selection@1
schedule.bundle.atomic-enqueue@1
```

If a policy requires atomic enqueue and an adapter lacks it, attachment or
invocation must fail. PLEX must not silently approximate the requirement with
large equal scores.

### 7.7 Cross-step reservations

Some workflows do not require all members to run in the current step, but do
require future capacity to be reserved. That is distinct from all-or-none
selection.

An optional host-enforced reservation could be:

```text
service.reserve(group_id, amount, ttl)
```

It requires a TTL, quota, expiry feedback, and principal accounting. A
policy-only indefinite reservation in scratch can leak capacity.

---

## 8. Adjacent important gaps

### 8.1 Standard cancel, drop, and branch-control capabilities

[PARD](plex-serving-policy-wiki/papers/pard.md), Certaindex/Dynasor, and
branch-parallelism policies may stop work after initial admission. A generic
raw action can invoke engine-specific cancellation, but it has no portable
semantics.

Minimum standard actions should distinguish:

```text
request.cancel@1
group.cancel@1
branch.limit@1
```

Required rules include:

- a cancellation request is distinct from an enacted terminal outcome;
- request or group state does not become terminal before action success;
- actual `cancelled`, `completed`, `expired`, or `failed` outcomes arrive in
  feedback;
- duplicate cancellation is idempotent or returns an explicit
  already-terminal outcome; and
- group cancellation defines its child-propagation scope.

A new `drop` hook is unnecessary. Standard actions from existing scheduling
and feedback paths are sufficient.

### 8.2 Iterative dependency-constrained eviction

Current `evict` scores one fixed candidate set. The host sorts those scores
once.

Prefix-tree policies such as RAGCache introduce dependency constraints:

- a parent cannot be removed while a child remains;
- removing a current leaf can make its parent a new leaf;
- a GDSF aging or clock value can change after each victim; and
- one victim may not satisfy the byte deficit, requiring a newly eligible set
  and newly computed scores.

If the adapter presents only the initial leaves, a newly eligible parent is
absent from the invocation. If it presents every node, a simple score sort can
produce an illegal parent-before-child order.

Two extensions are possible.

#### Iterative eviction episode

- The adapter enacts one victim or one frontier.
- It invokes `evict` again with the same episode ID.
- Each invocation contains the updated eligible set and clock.
- The episode repeats until the deficit is resolved.
- Iteration, bytes, and elapsed time are bounded.

#### Dependency graph

- The adapter supplies candidates and dependency edges once.
- The policy or host produces a legal topological reclaim sequence.
- The host validates every dependency and capacity transition.

The iterative episode is simpler and better preserves a source policy's
stepwise selection. It must still define partial enactment, replay, and engine
changes between invocations.

Consequently,
[RAGCache](plex-serving-policy-wiki/papers/ragcache.md) can be a useful
score-kernel demonstration without this extension, but it is not a strong
source reproduction unless leaf legality and iterative clock updates are also
implemented.

### 8.3 Multiple beneficiaries of shared resident state

A current resident unit has one `request_id` or `null`. A prefix, retrieved
document KV, adapter, or model object may benefit many requests or work groups.

A minimum representation is a bounded beneficiary reference:

```json
{
  "beneficiaries": [
    {
      "kind": "request",
      "id": "A"
    },
    {
      "kind": "group",
      "id": "G"
    }
  ],
  "beneficiary_count": 17
}
```

PLEX need not expose arbitrary live-request enumeration. An adapter can provide
bounded top-K beneficiaries plus aggregate count and reuse facts. Manifest
quotas can bound working-set expansion.

---

## 9. Transaction-semantics mismatch

### 9.1 What the conceptual paper describes

[`plex_paper.md`](plex_paper.md) describes a decision transaction in which the
host prepares policy writes, the adapter revalidates and enacts the decision,
and policy effects commit only after successful enactment:

```text
policy decision
-> validate
-> adapter enact
-> state commit
```

### 9.2 What v0.5 currently does

The implemented order in [`plex.md`](plex.md) is closer to:

```text
produce policy decision, state, and actions
-> validate result
-> commit policy state with CAS
-> return normalized decision and actions
-> engine enact
-> later feedback
```

If an engine action fails, v0.5 does not automatically roll back or compensate
the already committed policy state.

### 9.3 Why this matters

Suppose a routing policy returns target X and simultaneously writes:

```text
requests placed on X += 1
```

If placement fails after policy-state commit, that accounting is false. The
same problem applies to residency insertion, cancellation, and bundle enqueue.

### 9.4 Recommended choice

PLEX must choose and document one of two models.

#### Option A: actual two-phase enactment

- Keep policy writes in a prepared state.
- Ask the adapter to revalidate and enact the decision.
- Commit on success and discard on failure.

This provides stronger semantics, but requires an engine callback, timeout
rules, crash recovery, and a clear answer for irreversible partial enactment.

#### Option B: retain v0.5 semantics and narrow the claim

- Commit only intent or attempt state during a decision hook.
- Update all success-dependent accounting from later feedback.
- Remove claims that state commits with successful enactment.
- State explicitly that PLEX does not provide crash-atomic coordination with
  external engine state.

Option B is more realistic for the current prototype. Whichever model is
chosen, `plex_paper.md` and `plex.md` must describe the same transaction
boundary.

---

## 10. Impact on current reproduction claims

| Policy | Currently defensible description | Additional requirements for strong parity |
|---|---|---|
| Agentix | Request-level LAS-inspired policy | Trusted program/thread hierarchy, PLAS aggregate, and ATLAS critical path |
| Preble | Compact E2 policy-kernel adaptation | Complete E2 branches, prefix replication, and source mechanics |
| Helium | Cache-aware critical-path scheduler | Remove the bundle-fairness label; proactive warming needs a separate capability |
| RAGCache | GDSF score-kernel reproduction | Leaf legality, iterative eviction, and clock update |
| InferCept | Preserve/swap/discard decision kernel | Standard swap/discard capabilities and enacted outcomes |
| VTC | Strong reproduction candidate | Host-observed service charging and trace-parity validation |
| LMetric | Strong reproduction candidate | Exact candidate facts and hotspot-guard validation |

The paper and artifact should distinguish:

- **Paper replication**: source mechanics and end-to-end behavior are
  reproduced.
- **Policy-kernel reproduction**: the central score or decision algorithm is
  reproduced.
- **Decision-trace parity with deferred mechanics**: identical snapshots
  produce the same decisions and state transitions, but physical enactment
  remains in the adapter.
- **Inspired adaptation**: the subject or algorithm differs enough that source
  parity is not claimed.

The current
[`tests/policies/paper-agentix/src/lib.rs`](tests/policies/paper-agentix/src/lib.rs)
uses each candidate request's individual `attained_service`. It has no program
aggregation, thread hierarchy, or ATLAS. It should be described as an
Agentix-inspired request-level LAS adaptation, not a direct Agentix
reproduction.

---

## 11. What is not a PLEX core gap

Several important paper contributions should remain outside the stable
engine-policy waist.

### 11.1 Control-plane policy

Examples include:

- autoscaling and replica count;
- GPU provisioning;
- model loading and unloading;
- long-horizon placement optimization;
- parallelism-layout changes; and
- admission capacity planning.

The complete systems described by Helix, InferLine, INFaaS, MArk, and Aladdin
include this control-plane work. If PLEX targets an engine-local online policy
interface, those functions should remain optional control-plane extensions or
explicit non-goals.

### 11.2 Physical mechanics

Examples include:

- CPU/GPU KV swap kernels;
- live KV transfer and migration recovery;
- overlap between prefetch and computation;
- tensor-batch construction;
- attention-kernel changes;
- disaggregated prefill/decode transport; and
- model-parallel execution.

PLEX does not need to implement these mechanics. It must instead preserve four
rules:

1. An adapter exposes only capabilities it actually supports.
2. Missing required capabilities reject attachment or invocation.
3. An unsupported action is not silently translated into a "nearby" behavior.
4. Evaluation reports deferred mechanics separately from policy reproduction.

### 11.3 Offline models and predictors

Arrival predictors, output-length models, TTL estimators, and cost models are
not PLEX hooks. PLEX only needs to consume their outputs through versioned host
facts or bounded maps. Training and deploying the predictor remain separate
system concerns.

---

## 12. Position of `prefetch` and `rebalance`

The conceptual paper describes `prefetch` and `rebalance` as optional auxiliary
operations. v0.5 has a generic staged prefetch action, but it does not implement
complete typed invocation contracts for both auxiliary operations.

This does not invalidate the five core authorities:

- `prefetch` initiates active movement for expected future residency;
- `rebalance` migrates already-running work or state.

Both operations depend heavily on engine-specific mechanics and should not
carry a universal portability promise. If the paper presents them as a
contribution, however, the prototype support level must be reported accurately:

```text
native
amortized
explicitly emulated
absent
```

The existence of a generic action method name is not evidence of end-to-end
support.

---

## 13. Recommended minimum changes

### 13.1 P0: before publication

1. Define `WorkGroup` identity, state, and lifecycle.
2. Make group membership and fork/join relationships host-trusted facts.
3. Add a residency insertion cause and prospective units.
4. Add an explicit `admit` decision for selective cache insertion.
5. Correct Agentix, Helium, Preble, RAGCache, and InferCept claims to match the
   implemented semantics.
6. Align transaction ordering in the paper and the v0.5 design document.

If an item is not implemented, PLEX should state the limitation explicitly and
mark the affected policy coverage as partial.

### 13.2 P1: optional capabilities

1. `route.batch-assignment@1`
2. `schedule.bundle.selection@1`
3. `schedule.bundle.atomic-enqueue@1`
4. `request.cancel@1` and `group.cancel@1`
5. iterative eviction episodes
6. shared-resident beneficiary facts

Each capability should follow required/optional attachment semantics. Missing
support must never trigger silent emulation.

### 13.3 P2: explicit non-goals

1. autoscaling and provisioning;
2. physical KV transfer and swap;
3. kernels and tensor-batch construction;
4. offline predictor training; and
5. crash-atomic coordination with external engine state.

---

## 14. Acceptance tests

The design is not complete when only types are added. Each extension needs
behavioral tests.

### 14.1 Work groups

- Group state remains live after child A finishes while child B is still live.
- Another principal cannot attach by copying the group ID string.
- Concurrent feedback from two children increments group service exactly once
  per successful delivery.
- Replaying a group-terminal delivery does not duplicate cleanup or
  accounting.
- A terminal group cannot fork a new child.

### 14.2 Residency admission

- A prospective object with `admit: false` is not cached even when free
  capacity exists.
- A low-value prospective object does not displace a higher-value resident.
- A high-value prospective object displaces a lower-value resident when
  capacity requires it.
- Insert or eviction failure does not update success accounting before
  feedback.

### 14.3 Batch routing

For the utility matrix in Section 6, the policy must be able to produce:

```text
A -> Y
B -> X
```

A result that exceeds target capacity or selects an infeasible edge must
fallback or fail validation.

### 14.4 Atomic bundles

- If capacity fits both W1 and W2, both may be selected.
- If capacity fits only one, neither is selected.
- If atomic enqueue is required and unsupported, attachment or invocation
  fails.

### 14.5 Iterative eviction

- A parent is never selected while a child remains.
- Removing a child makes its parent eligible when appropriate.
- The episode terminates when the byte deficit is resolved.
- Exceeding the iteration bound produces an explicit fallback or error.

---

## 15. Final assessment

PLEX can express a large fraction of the literature's central online-policy
logic. It aligns especially well with per-request routing, fairness
scheduling, token budgeting, scalar retention, and feedback accounting.

It would nevertheless be an overclaim to say that unchanged v0.5 covers agent
programs and cache policy broadly.

The two most important gaps are:

1. **A trusted work-group lifecycle that outlives individual logical requests**
2. **Residency admission that decides whether new state enters the resident
   set**

Joint batch assignment and all-or-none bundles should not be mandatory on every
engine. They should be optional capabilities, and policy families that require
them should be classified as residual when the capabilities are absent.

The fundamental problem is not the number of hooks. It is that the current
hooks have a subject, invocation granularity, and output constraint narrower
than several important policy families. Extending those three dimensions
preserves the five-authority stable waist while substantially improving
literature fidelity.
