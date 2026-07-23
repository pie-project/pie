# PLEX v0.6 Normative Contract

## Document status

- Status: normative implemented contract
- Contract: `pie:plex@0.6.0`
- Package format: `.plexpkg` version 6
- Engine envelope: a future version of `pie.plex.engine@1`
- Supersedes for v0.6 semantics: [`plex.md`](plex.md)
- Delivery plan: [`plex_0.6.md`](plex_0.6.md)
- Migration guide: [`plex_0.5_to_0.6.md`](plex_0.5_to_0.6.md)

This document uses the key words **MUST**, **MUST NOT**, **REQUIRED**,
**SHOULD**, **SHOULD NOT**, and **MAY** as normative requirements.

Phase 0 originally froze the contract beside v0.5. The active interface crate,
package loader, runtime, Rust SDK, and Python binding now use v0.6. The
canonical Rust types remain under `pie_plex::v0_6`, and the canonical WIT files
live under `interface/plex/wit-0.6/`.

## 1. Stable waist and scope

PLEX v0.6 defines exactly five policy authorities:

| Operation | Authority |
|---|---|
| `admit` | Which proposed requests may enter serving |
| `route` | Where admitted requests should execute |
| `schedule` | Which runnable requests receive service and how much |
| `cache` | Which resident or prospective objects remain cached |
| `feedback` | How enacted outcomes update policy state |

The normal request path is:

```text
admit -> route -> schedule ... -> feedback
```

`cache` is independent of that linear path. It may run for insertion, pressure,
expiry, or dependency-progress events.

PLEX v0.6 does not add core operations for cancellation, prefetch, swap,
migration, autoscaling, provisioning, model loading, kernel selection, or
workflow execution. Portable instances of those mechanics are versioned
capabilities and actions.

## 2. Subjects and identity

### 2.1 Request

A `Request` is the independently admitted, routed, scheduled, and completed
serving flow. It has:

- a host-issued `request_id`;
- a host-issued `principal_id`;
- a `generation_id` beginning at zero and increasing by exactly one for each
  continuation; and
- optional immutable membership in one `WorkGroup`.

The v0.5 term `LogicalRequest` is removed. A request ID is also the member
identity inside its work group; v0.6 has no separate `MemberId`.

### 2.2 WorkGroup

A `WorkGroup` is an optional trusted coordination scope shared by several
requests. It provides:

- principal ownership and authorization;
- group lifecycle;
- member and state quotas;
- group deadlines and host facts;
- aggregate accounting; and
- group-private policy scratch.

A work group is not a schedulable request, a workflow-DAG runtime, a scheduling
bundle, a placement guarantee, or a simultaneous-execution guarantee.

### 2.3 Identity requirements

All IDs are opaque, host-issued UTF-8 strings. An ID MUST be non-empty, MUST be
at most 128 bytes, and is compared byte-for-byte. Copying an ID into metadata or
an extensible fact does not establish identity, ownership, or membership.

The host MUST validate these rules:

1. A request belongs to zero or one work group.
2. Membership is fixed when the request identity is created.
3. An independent request does not receive an implicit singleton group.
4. Finishing a request does not finish its group.
5. A group may temporarily have no live requests.
6. A terminal group cannot accept a new request.
7. A request continuation preserves request, group, and principal identity.
8. A generation advances by exactly one.
9. A grouped request and its group have the same trusted principal.

## 3. Host lifecycle state machines

Policy plans do not directly mutate host lifecycle state. The adapter
revalidates and enacts a plan, records the host transition, and reports enacted
outcomes through feedback.

### 3.1 Work-group lifecycle

The minimum group states are:

```text
open
closed
cancelled
expired
```

The allowed transitions are:

```text
absent -> open
open -> closed
open -> cancelled
open -> expired
```

`closed`, `cancelled`, and `expired` are terminal. A terminal group MUST reject
new requests. Closing a group does not implicitly terminate existing requests.
The host MAY retain terminal group state until all members and feedback records
needed for cleanup are resolved.

### 3.2 Request lifecycle

The minimum request states are:

```text
pending
admitted
active
paused
completed
failed
cancelled
expired
rejected
```

The allowed transitions are:

```text
absent -> pending
pending -> admitted | rejected | expired
admitted -> active | failed | cancelled | expired
active -> paused | completed | failed | cancelled | expired
paused -> active | completed | failed | cancelled | expired
```

`completed`, `failed`, `cancelled`, `expired`, and `rejected` are terminal.
Route and schedule plans do not by themselves transition a request. Admission
acceptance becomes `admitted` only after adapter revalidation and enactment.
Cancellation remains an action request until terminal feedback is enacted.

### 3.3 Required host operations

The engine-facing lifecycle MUST support the meanings of:

```text
create_group
create_request(group_id?)
continue_request
finish_request
close_group
expire_group
```

Cancellation is requested through a versioned action and becomes terminal only
after feedback. All create, terminal, and cleanup operations MUST be
idempotent under replay.

## 4. Opportunity and snapshot model

Every decision operation is set-oriented. A singleton decision uses an array of
length one and the same ABI as a batch.

An `Opportunity` is a bounded host-selected decision boundary. It is not
defined as all events with an identical arrival timestamp. Each decision
context contains:

- `opportunity_id`: stable for retries of the same boundary;
- `snapshot`: an opaque snapshot ID and monotonically interpreted revision;
- `attempt`: zero for the first invocation and incremented for a policy-state
  retry of the same boundary; and
- `mechanics`: the negotiated optional mechanics available to this invocation.

For one opportunity the host MUST preserve:

- candidate membership;
- candidate order;
- target and feasible-edge order;
- capacity interpretation; and
- adapter facts that are declared immutable for the attempt.

A compare-and-swap conflict MAY cause a retry with the same opportunity ID, a
new snapshot reference, and an incremented attempt. A changed feasible set,
changed opportunity boundary, or later deferred retry MUST use a new
opportunity ID.

The host MUST record opportunity boundaries, input order, snapshot references,
attempts, and normalized results in replay traces.

## 5. Documents and policy state

### 5.1 Extensible documents

The WIT `document` type is a UTF-8 JSON string. Every document at the Rust and
engine APIs MUST decode to a JSON object. Arrays, scalars, and null are invalid
documents.

Structural safety fields are typed by WIT and Rust records. Engine-specific
facts, mutable request fields, and policy scratch remain documents.

### 5.2 Policy state

The guest-visible state is:

```text
PolicyState
|-- shared
|-- groups[]
`-- requests[]
```

Each group state contains:

```text
group_id
principal_id
status
limits { max_members, max_scratch_bytes }
member_count
facts
scratch
```

Each request state contains:

```text
request { request_id, generation_id, group_id?, principal_id }
status
facts
fields
scratch
```

Writers are:

| Namespace | Writer |
|---|---|
| shared | policy |
| group facts | host |
| group scratch | policy |
| request facts | host |
| request fields | host lifecycle and policy |
| request scratch | policy |

Facts are read-only to the policy. State membership is read-only. A raw-WIT
guest MUST NOT add, remove, or rename scopes or mutate lifecycle status through
a state update.

The host MUST enforce `member_count <= max_members` and the serialized group
scratch size against `max_scratch_bytes` on both load and update. Both limits
are positive. A group may have a zero member count.

### 5.3 State updates

State updates are sparse by scope:

- absent `shared` means no shared update;
- each listed group update replaces the complete group `scratch` document;
- each listed request update may replace the complete `fields` document, the
  complete `scratch` document, or both; and
- an update MUST NOT list the same scope twice.

The contract does not define JSON Patch, JSON Merge Patch, or partial document
mutation. SDKs MAY compute these scope-level replacements by diffing the input
and output authoring model.

### 5.4 Working set

The host constructs the exact working set before invoking the guest:

1. shared state is included once;
2. every request referenced by the validated context is included once;
3. the trusted group of every referenced member request is included once; and
4. unrelated sibling requests are not implicitly included.

Every included shared, group, and request scope has a host-private revision.
The guest does not observe those revisions.

High-frequency measurements SHOULD be supplied as bounded host facts and
reduced in feedback batches. Policies SHOULD avoid unnecessary group or shared
writes because those scopes can increase revision conflicts.

## 6. Conditional policy-state commit

PLEX v0.6 provides conditional policy-state commit, not an external engine
transaction.

For a new invocation, the runtime performs:

```text
load one coherent policy-state working set
-> invoke policy
-> validate the plan and state update
-> compare revisions and conditionally commit policy state
-> expose the normalized plan and staged actions
-> adapter revalidates and enacts
-> adapter reports enacted outcomes through feedback
```

Policy state commits only when:

1. the guest returns success;
2. the plan and state update pass structural and semantic validation; and
3. every required working-set revision still matches.

Commit does not imply enactment. Therefore:

- decision operations MAY record intent or attempt state;
- success-dependent counters MUST be updated from feedback;
- action success MUST NOT be recorded optimistically;
- engine failure does not roll policy state back automatically; and
- PLEX MUST NOT claim two-phase commit or crash-atomic coordination with the
  engine.

All staged actions are discarded on guest error, trap, invalid output, limit
failure, or state conflict. They become visible to the adapter only after a
successful policy-state commit.

## 7. Common resource model

Admission and routing use named resource vectors:

```text
ResourceAmount { name, unit, amount }
ResourceLimit  { name, unit, maximum }
```

Names and units MUST be 1-64 bytes and contain only lowercase ASCII letters,
digits, `.`, `_`, or `-`, beginning with a lowercase letter. A vector MUST NOT
contain duplicate `(name, unit)` keys.

Every demand key MUST have a corresponding limit. Validation sums demands with
checked unsigned arithmetic and rejects overflow. A zero demand or zero limit
is valid.

Compatibility constraints such as model, region, hardware, stage, and locality
are represented by the host-supplied feasible set. A policy cannot create a
new feasible edge.

## 8. `admit`

### 8.1 Context

```text
AdmitContext {
  meta,
  cause,
  candidates[],
  capacity {
    max_accepted,
    limits[],
    facts
  }
}
```

Each candidate contains a trusted request reference, a resource-demand vector,
and facts. Every candidate MUST name a pending request in the working set.

Initial causes are:

```text
arrival
retry
capacity-changed
```

### 8.2 Plan

```text
AdmitPlan {
  decisions[] // accept | defer | reject
}
```

The decisions array MUST be dense and aligned with candidates.

### 8.3 Validation

The host MUST:

1. validate context and working-set identity;
2. require one decision per candidate;
3. count accepted candidates against `max_accepted`;
4. sum accepted resource demands against every capacity limit;
5. reject arithmetic overflow; and
6. reject unknown decisions.

`defer` preserves the pending identity for a later, new opportunity. `reject`
is terminal after adapter enactment and idempotent cleanup. A batch view does
not imply all-or-none admission.

## 9. `route`

### 9.1 Context

```text
RouteContext {
  meta,
  cause,
  requests[],
  targets[],
  feasible_edges[]
}
```

Each target has a stable target ID, a maximum assignment count, named resource
limits, a target revision, and facts. Each feasible edge references one request
index and one target index and carries the resource demand for that assignment.

Initial causes are:

```text
admission
retry
rebalance
target-changed
```

The host MUST reject duplicate request-target edges and out-of-range indices.

### 9.2 Plan

```text
RoutePlan {
  decisions[] // assign(edge_index) | defer
}
```

The decisions array is dense and request-aligned. An assignment references one
supplied feasible edge. `defer` means the admitted request remains unplaced and
may be presented in a later, new route opportunity.

### 9.3 Validation

The host MUST:

1. require one decision per request;
2. require an assigned edge to belong to that request index;
3. assign each request at most once;
4. enforce each target's maximum assignment count;
5. sum edge demands against target resource limits;
6. reject infeasible, duplicate, stale, or out-of-range references; and
7. tie the result to the supplied snapshot and target revisions.

The output is a direct bounded assignment. The host does not reinterpret edge
scores through a fixed matching solver. Queue insertion, placement, replica
creation, model loading, and physical migration remain adapter mechanics.

## 10. `schedule`

### 10.1 Context

```text
ScheduleContext {
  meta,
  cause,
  runnable[],
  capacity {
    max_selections,
    max_requests,
    max_total_tokens,
    facts
  }
}
```

Each runnable candidate contains a trusted request reference, a positive
maximum token budget, and facts.

Initial causes are:

```text
arrival
completion
capacity-changed
timer
feedback
```

### 10.2 Plan

```text
SchedulePlan {
  selections[] {
    requests[],
    token_budgets[]
  }
}
```

A one-request selection is ordinary scheduling. A multi-request selection is
one all-or-none normalized selection unit.

### 10.3 Validation

The host MUST:

1. reject an empty selection;
2. require request and token-budget arrays to have equal length;
3. require every request index to be runnable;
4. require every token budget to be positive and no greater than the
   candidate maximum;
5. reject a request appearing in more than one selection;
6. enforce `max_selections`, `max_requests`, and `max_total_tokens`; and
7. reject arithmetic overflow.

Work-group membership does not automatically create a selection unit. A policy
may construct a unit from trusted group, workflow, or adapter facts.

The core guarantees selection atomicity only: all members are in the normalized
plan or none are. Atomic adapter enqueue requires
`schedule.atomic-enqueue@1`. Simultaneous GPU start and joint success are not
portable guarantees.

## 11. `cache`

### 11.1 Context

```text
CacheContext {
  meta,
  cause,
  resident[],
  prospective[],
  capacity {
    max_bytes,
    fixed_bytes,
    facts
  },
  episode?
}
```

Each cache object has a stable object ID, size in bytes, bounded beneficiary
references, total beneficiary count, and facts. A resident object also has a
`reclaimable` flag for the current invocation.

Beneficiaries are typed request or group references. The bounded list MAY be a
top-K subset; `beneficiary_count` records the total and MUST be at least the
number of listed references.

Initial causes are:

```text
insertion
pressure
expiry
dependency-progress
```

An optional episode contains an episode ID, zero-based iteration, and a
positive maximum iteration count.

### 11.2 Plan

```text
CachePlan {
  admissions[] // cache | bypass, dense over prospective[]
  reclaim[]    // ordered resident indices
}
```

`bypass` prevents persistent insertion even when free capacity exists.

### 11.3 Validation

The host MUST:

1. require one admission decision per prospective object;
2. reject duplicate object IDs across resident and prospective sets;
3. reject duplicate or out-of-range reclaim indices;
4. reject reclaim of an object not marked reclaimable;
5. compute retained bytes as:

   ```text
   fixed_bytes
   + resident bytes not reclaimed
   + prospective bytes marked cache
   ```

6. require retained bytes to be no greater than `max_bytes`; and
7. reject arithmetic overflow.

The reclaim list is an explicit legal order, not a score vector.

### 11.4 Dependency-constrained episodes

For a dependency-constrained cache, the host uses a bounded iterative episode:

1. supply the current eligible frontier by marking only legal residents
   reclaimable;
2. accept one legal victim or legal frontier from the ordered reclaim list;
3. enact and report the step;
4. invoke a new opportunity with the same episode ID, incremented iteration,
   and newly eligible objects; and
5. stop on satisfied capacity, explicit failure, or the hard iteration/time
   bound.

Partial enactment is reported honestly. Exceeding the bound is an explicit
failure; the host MUST NOT continue with an unvalidated approximate order.

Allocation, eviction, swap, and movement are engine mechanics. Their outcomes
arrive through feedback.

## 12. `feedback`

### 12.1 Context

```text
FeedbackContext {
  delivery_id,
  records[] {
    subject,
    outcome,
    facts
  }
}
```

Initial subjects are:

```text
request
work-group
cache-object
route-assignment
schedule-selection
action
```

Initial outcomes are:

```text
progress
completed
failed
cancelled
expired
action-succeeded
action-failed
```

Assignment and selection subjects include the originating opportunity ID and
their normalized index. Action subjects use the staged action ID.

### 12.2 Validation and deduplication

The host MUST:

1. require a non-empty delivery ID and at least one record;
2. validate every subject identity and facts document;
3. allow `action-succeeded` and `action-failed` only for action subjects, while
   action subjects accept only those outcomes or `progress`;
4. apply a successful delivery ID exactly once;
5. commit state updates, terminal cleanup, and the deduplication record in one
   policy-state commit; and
6. return the recorded semantic result for a duplicate successful delivery
   without reinvoking the guest or replaying actions.

A cancelled request or group cleanup MUST correlate either with a successful
PLEX cancellation action or with a matching terminal record whose
`facts.initiator` is `host`. This separates policy-requested cancellation from
an authenticated engine/client cancellation.

Unavailable feedback and non-retryable policy fallback still commit an empty
state update, terminal cleanup, and the returned semantic result atomically.
State conflicts do not commit or record the delivery and MAY be retried with
the same delivery ID.

## 13. Host imports and mechanics

The only guest imports are:

```text
query(method, args) -> result<document, string>
action(method, args) -> result<action-id, string>
```

`query` is immediate, synchronous, and read-only. `action` stages a descriptor
and returns an invocation-local monotonic action ID. The host MUST account for
call count and aggregate request/response bytes.

Method names are independently versioned. Raw methods remain an extension
escape hatch, but their use is visible in manifests and fixture metadata.

The initial standard mechanics registry is:

| Mechanic | Kind | Standard method |
|---|---|---|
| `schedule.atomic-enqueue@1` | guarantee | none |
| `request.cancel@1` | action | `pie.request.cancel@1` |
| `group.cancel@1` | action | `pie.group.cancel@1` |
| `cache.prefetch@1` | action | `pie.cache.prefetch@1` |
| `cache.swap@1` | action | `pie.cache.swap@1` |
| `request.rebalance@1` | action | `pie.request.rebalance@1` |

Each registry entry names the operations from which the mechanic may be used.
The host MUST reject a standard action staged from an unlisted operation.

Action acceptance is not action success. Standard action feedback MUST
distinguish succeeded, failed, already-terminal, expired, and unsupported
outcomes as defined by each action schema. The typed action subject supplies
the invocation-local action ID; feedback facts supply the originating
opportunity ID, method, idempotency key, and terminal status for unambiguous
correlation.

## 14. Manifest and capability negotiation

A v0.6 manifest contains:

```json
{
  "contract": {"major": 0, "minor": 6},
  "package_name": "coordinated-policy",
  "package_version": "0.6.0",
  "implements": ["admit", "route", "schedule", "cache", "feedback"],
  "requires": ["request.cancel@1"],
  "optional": ["cache.prefetch@1"],
  "schemas": [
    {
      "kind": "fact",
      "id": "pie.example.queue-facts@1",
      "required": true
    }
  ],
  "limits": {
    "memory_bytes": 4194304,
    "deadline_ms": 100,
    "input_bytes": 1048576,
    "output_bytes": 1048576,
    "host_calls": 64,
    "host_call_bytes": 1048576
  }
}
```

Manifest rules are:

1. unknown fields are rejected;
2. the contract is exactly `{major: 0, minor: 6}`;
3. package name and semantic version follow the v0.5 lexical limits;
4. `implements` is a non-empty subset of the five operations;
5. `requires` and `optional` contain unique versioned mechanic IDs and are
   disjoint;
6. schema requirements contain unique `(kind, id)` pairs;
7. every limit is non-zero; and
8. every requested limit fits within the host maximum.

At attachment, every required mechanic and required schema MUST be available.
Missing required support fails attachment. Optional support is negotiated and
reported in each decision context. Missing support MUST NOT trigger silent
emulation.

Every accepted component exports all five policy functions. The manifest
declares which operations it owns. An operation has at most one attached owner.

## 15. WIT and component surface

The canonical component package is:

```text
pie:plex@0.6.0
```

The policy exports typed operation-specific invocation and output records. Only
extensible documents remain JSON strings. The component imports exactly the
PLEX host interface and MUST NOT import WASI or another external interface.

Guest failures use:

```text
PolicyError { code, message, details }
```

`code` is 1-64 bytes using lowercase ASCII letters, digits, `.`, `_`, or `-`,
beginning with a letter. `message` is at most 1024 bytes. `details` is a
document.

There is no ambient state-loading interface. A guest cannot scan backend keys,
change working-set membership, or observe host-private revisions.

A raw-WIT guest is subject to the same host validation as a guest using the
Rust SDK.

## 16. Failure, retry, and fallback

Failures are classified at least as:

```text
policy-error
invalid-context
invalid-output
state-conflict
unsupported-mechanic
resource-limit
deadline-exceeded
host-saturated
trap
enactment-failed
```

Guest-visible policy errors do not commit state or actions. Host validation
errors do not commit state or actions. State conflicts do not commit state,
feedback deduplication, cleanup, or actions.

An adapter MAY use its native policy after an unavailable or fallback outcome,
but it MUST record that PLEX did not produce an enacted success. Unsupported
required mechanics are attachment or invocation failures, not fallback
approximations.

PLEX does not require automatic retry. If the host retries:

- a state conflict for the same bounded opportunity uses the same opportunity
  ID and a higher attempt;
- a deferred request uses a new opportunity ID;
- an enactment failure is reported through feedback and any recomputation uses
  a new opportunity ID; and
- a failed feedback delivery retains its delivery ID.

## 17. Isolation and limits

The v0.6 execution model retains:

- a fresh bounded Wasmtime store and component instance per invocation;
- epoch deadline interruption;
- memory, table, instance, and input/output byte bounds;
- host-call count and byte bounds;
- disabled Wasm threads;
- no WASI;
- exact import/export verification; and
- a host-wide concurrent invocation limit.

Batch size, target count, edge count, selection count, cache-object count,
beneficiary count, context bytes, output bytes, episode iterations, and
deadline MUST have hard host bounds.

Replay MUST retain an active epoch deadline so a non-terminating guest cannot
hang the runner. Structural, byte, count, and semantic validation remains
active. A trace whose semantic result depends on crossing the wall-clock
deadline is not deterministic replay evidence.

## 18. Replay contract

A replay fixture records:

- fixture schema version;
- contract version;
- package identity and attachment generation;
- operation;
- opportunity or delivery identity;
- complete typed invocation;
- staged query and action responses;
- raw guest output;
- normalized plan;
- state update and commit result;
- enacted feedback when applicable; and
- expected success or classified failure.

Replay comparison is byte-order independent for JSON objects but order
sensitive for all typed lists. The first divergent command, plan, state update,
action, or failure classification MUST be reportable.

## 19. Package format and version dispatch

The v0.6 `.plexpkg` header is:

```text
8 bytes   magic: "PLEXPKG\0"
2 bytes   little-endian format version: 6
2 bytes   flags: 0
4 bytes   little-endian manifest length
8 bytes   little-endian component length
32 bytes  BLAKE3 digest
N bytes   manifest JSON
M bytes   component bytes
```

The digest remains length-delimited manifest plus component content and is not
a publisher signature.

A v0.5 package MUST NOT be reinterpreted as v0.6. The loader dispatches by
package format and exact contract version. Unsupported versions return an
explicit version error before component compilation.

## 20. Security invariants

The host MUST enforce:

- trusted request and group identity;
- immutable request-to-group membership;
- principal authorization for group operations;
- bounded group fan-out and state;
- exact working-set membership;
- read-only host facts;
- feasible-edge and capacity validation;
- non-overlapping schedule selections;
- reclaim eligibility and cache capacity;
- mechanic and schema negotiation;
- staged-action rollback on failed commit;
- feedback deduplication and idempotent cleanup; and
- bounded execution and host calls.

Policy-provided facts, fields, scratch, scores, costs, and hints are untrusted
inputs to host validation.

## 21. Normative completion criteria

The v0.6 contract is implemented only when:

1. all five typed operations execute end to end;
2. every decision operation passes singleton and batch tests;
3. work-group identity, lifecycle, quotas, and state are host-trusted;
4. request membership cannot be forged or mutated;
5. prospective cache admission and resident reclaim share one snapshot;
6. direct route assignments and schedule selections are semantically validated;
7. policy-state commit and engine enactment remain separate;
8. required mechanics fail explicitly when unsupported;
9. feedback state, cleanup, and deduplication commit atomically;
10. v0.5 packages fail with an explicit version error on the v0.6 path; and
11. replay fixtures cover every lifecycle, validation, retry, and failure rule
    in this document.
