# PLEX v0.5 to v0.6 Migration

## Status

This is the normative migration table for
[`pie:plex@0.6.0`](plex_0.6_contract.md). v0.6 is intentionally breaking. A
v0.5 component or `.plexpkg` is never reinterpreted as v0.6.

The interface crate, package format, runtime, Rust guest SDK, engine API,
Python binding, adapter templates, fixture corpus, and 31-candidate replication
matrix have migrated. The v0.5 JSON fixtures remain archived under
`tests/policies/v0.5/` for intentional-difference review.

## 1. Version dispatch

| Surface | v0.5 | v0.6 | Migration |
|---|---|---|---|
| WIT package | `pie:plex@0.5.0` | `pie:plex@0.6.0` | Rebuild the guest against the typed v0.6 world. |
| Package format | 5 | 6 | Repackage with a v0.6 manifest and component. |
| Contract manifest | `{major: 0, minor: 5}` | `{major: 0, minor: 6}` | Exact-match validation; no compatibility mode. |
| Active operation names | `route`, `admit`, `schedule`, `evict`, `feedback` | `admit`, `route`, `schedule`, `cache`, `feedback` | Rename `evict` ownership and exports to `cache`. |
| Normal request path | `route -> admit` | `admit -> route` | Create pending request state before admission; route only enacted accepts. |

## 2. Programming model

| Area | v0.5 | v0.6 | Required change |
|---|---|---|---|
| Request name | `LogicalRequest` / `logical_request_id` | `Request` / `request_id` | Rename APIs, facts, fixtures, and documentation. |
| Request grouping | No trusted group subject | Optional trusted `WorkGroup` | Use host-issued group identity, principal, quotas, and immutable membership. |
| Generations | Sequential on a logical request | Sequential on a request | Preserve exact `0, 1, 2, ...` advancement. |
| State scopes | `shared`, requests | `shared`, groups, requests | Add group facts/scratch and group revisions. |
| Request state | facts, fields, scratch | facts, fields, scratch | Preserve namespaces; identity is typed rather than inferred from JSON. |
| Group state | Emulated in shared/request scratch | Typed principal, status, quotas, member count, facts, and scratch | Remove authoritative group copies from untrusted documents. |
| Lifecycle status | Inferred from request events and facts | Typed host-owned group/request status in policy state | Treat status as read-only structural state. |
| Working set | Referenced requests | Referenced requests plus each trusted group | Auto-join groups once; do not auto-load siblings. |
| State update | Sparse shared/request JSON | Sparse scope replacement for shared/group/request mutable namespaces | Emit complete replacement documents for each changed namespace. |
| Commit claim | Often described as an engine transaction | Conditional policy-state commit before enactment | Move success-dependent accounting to feedback. |

## 3. Operation migration

### 3.1 `admit`

| v0.5 | v0.6 |
|---|---|
| One request decision | Dense decisions over `candidates[]` |
| Request may be routed before admission | Pending request is admitted before routing |
| Capacity mostly adapter-specific | Typed `max_accepted` plus named resource limits |
| Result `{decision}` | `AdmitPlan { decisions[] }` |

Migration steps:

1. Build one bounded candidate array.
2. Preserve host-determined order in replay.
3. Return exactly one `accept`, `defer`, or `reject` per candidate.
4. Keep deferred identities pending for a later opportunity.

### 3.2 `route`

| v0.5 | v0.6 |
|---|---|
| One request and target candidates | Request set, target set, sparse feasible-edge graph |
| Policy returns target scores | Policy returns dense direct decisions |
| Host sorts one request's targets | Host validates joint assignments |
| Unassigned behavior implicit | `defer` is explicit |

Migration steps:

1. Replace per-request candidate arrays with indexed requests, targets, and
   feasible edges.
2. Return `assign(edge_index)` or `defer` for every request.
3. Move replica creation, model loading, and physical migration out of route
   output.
4. Record target revisions and the opportunity snapshot.

### 3.3 `schedule`

| v0.5 | v0.6 |
|---|---|
| Dense score and optional token budget per runnable request | Explicit non-overlapping selections |
| Host performs greedy score fill | Policy chooses the bounded set directly |
| No all-or-none unit | A selection may contain one or several requests |
| Token budget capability | Token budgets are core plan fields |

Migration steps:

1. Replace score output with `SchedulePlan.selections`.
2. Put each request in at most one selection.
3. Return aligned positive token budgets.
4. Declare `schedule.atomic-enqueue@1` only when adapter enqueue atomicity is
   required; do not claim simultaneous execution.

### 3.4 `evict` to `cache`

| v0.5 | v0.6 |
|---|---|
| Resident objects only | Resident and prospective objects |
| Pressure-only byte deficit | Insertion, pressure, expiry, and dependency progress |
| Retention scores normalized by host | Dense admission plus explicit ordered reclaim |
| One fixed candidate set | Optional bounded iterative episode |
| One request owner or none | Bounded request/group beneficiaries plus total count |

Migration steps:

1. Rename the operation, manifest ownership, SDK method, fixtures, and replay
   outputs to `cache`.
2. Present prospective objects before persistent allocation.
3. Return `cache` or `bypass` for every prospective object.
4. Return only currently reclaimable resident indices in legal order.
5. Carry an episode ID and iteration for dependency-constrained reclaim.

### 3.5 `feedback`

| v0.5 | v0.6 |
|---|---|
| Open JSON feedback context | Typed delivery envelope and subjects/outcomes |
| Successful delivery deduplicated | Same guarantee, extended to group state and cleanup |
| Terminal request cleanup | Request and group cleanup as applicable |
| Action outcome encoded in facts | Typed action subject and success/failure outcome |

Migration steps:

1. Wrap records in one non-empty delivery.
2. Use typed subjects and outcome kinds.
3. Keep delivery IDs stable across retry.
4. Update enacted-success counters only from feedback.

## 4. WIT and SDK migration

| Area | v0.5 | v0.6 |
|---|---|---|
| Invocation transport | `context-json`, `state-json` | Typed context and state records |
| Output transport | `result-json`, `state-update-json` | Typed plan and state update records |
| Extensibility | Most structures are JSON | Only facts, fields, scratch, query/action payloads are JSON documents |
| Policy error | String | Typed code, message, and details document |
| State import | None | None |
| Host imports | JSON query/action | Document query/action with the same staged semantics |

SDK authors must:

1. generate bindings from `interface/plex/wit-0.6/`;
2. expose the five typed policy methods;
3. preserve read-only facts and immutable state membership;
4. compute sparse scope replacements;
5. reject non-object documents before crossing WIT; and
6. rely on host validation even when SDK validation succeeds.

## 5. Manifest migration

| v0.5 field | v0.6 field | Rule |
|---|---|---|
| `operations` | `implements` | Non-empty operation ownership set |
| none | `requires` | Required mechanic IDs; unavailable support rejects attachment |
| none | `optional` | Optional mechanic IDs negotiated into contexts |
| none | `schemas` | Required/optional fact and action schema requirements |
| `limits.memory_bytes` | same | Non-zero and host-bounded |
| `limits.fuel` | removed | Retained only as an ignored v0.5 format-5 compatibility field |
| `limits.deadline_ms` | same | Non-zero and host-bounded |
| `limits.input_bytes` | same | Non-zero and host-bounded |
| `limits.output_bytes` | same | Non-zero and host-bounded |
| host-only call count | `limits.host_calls` | Package request bounded by host maximum |
| host-only call bytes | `limits.host_call_bytes` | Package request bounded by host maximum |

`requires` and `optional` must be disjoint. Missing required support never
falls back to an approximate core behavior.

## 6. Engine-adapter migration

Adapters must change in this order:

1. Add trusted group and pending-request lifecycle APIs.
2. Snapshot set-oriented typed contexts with deterministic ordering.
3. Build exact working sets including trusted groups.
4. Invoke the typed v0.6 component.
5. Validate direct plans against the same snapshot.
6. Conditionally commit policy state.
7. Revalidate and enact the normalized plan.
8. Report actual outcomes through typed feedback.
9. Record opportunity IDs, attempts, snapshots, and enactment failures in
   replay.

An adapter must not:

- infer membership from metadata;
- silently split a joint route opportunity into sequential calls;
- approximate a multi-request selection with large equal scores;
- insert a prospective cache object before the `cache` decision when bypass is
  still feasible;
- mark cancellation complete when the action is merely accepted; or
- claim that policy-state commit means engine enactment.

## 7. State-data migration

Persistent v0.5 state is not automatically compatible because keys, subjects,
and update semantics changed.

An explicit migrator may:

1. rename request identity facts;
2. preserve request fields and scratch;
3. create host-authenticated groups from trusted external records only;
4. move valid group aggregates from shared/request scratch into group scratch;
5. discard forged, ambiguous, or unbounded membership data;
6. reset feedback deduplication into a versioned ledger; and
7. record the source contract and migration tool version.

The default safe behavior is a fresh v0.6 state backend. A migrator must never
promote an application-provided group ID into trusted membership without host
authorization.

## 8. Fixture and claim migration

Every fixture must record:

- contract and package version;
- deterministic opportunity boundaries;
- exact typed input;
- raw and normalized output;
- state update and feedback effects;
- required and optional mechanics;
- expected failure classification; and
- replication evidence level.

Existing paper fixtures require these corrections:

| Fixture | Required v0.6 correction |
|---|---|
| Agentix / Autellix | Use `WorkGroup` accounting rather than request-local program emulation. |
| Continuum | Use `cache`, enacted TTL feedback, and group-level FCFS where claimed. |
| KVFlow | Separate cache policy parity from physical prefetch mechanics. |
| Preble | Implement both exploit/explore routing branches over set-oriented inputs. |
| Helium | Remove bundle-fairness wording; implement critical-path scheduling and forced progress. |

No v0.5 result may be relabeled as v0.6 parity without rebuilding the fixture
against the v0.6 contract.

## 9. Rollout gate

The active runtime may switch from v0.5 to v0.6 only after:

1. typed WIT host and guest bindings compile;
2. v0.6 package format and manifest validation are implemented;
3. group/request lifecycle and state revisions pass replay tests;
4. all five semantic validators pass positive and negative suites;
5. the Rust SDK and engine adapters use the same generated WIT;
6. v0.5 rejection produces an explicit diagnostic; and
7. the paper and implementation documentation describe the v0.6 commit and
   enactment boundary consistently.
