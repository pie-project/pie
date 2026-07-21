# PLEX Design Issues

## Status

The independent policy kernel now implements the decisions below. The ABI
remains `0.x` until live Pie adapters validate the semantic boundary. Deferred
items are listed explicitly and are not silently emulated.

## 1. Independent Milestone Boundary

The independent milestone cannot make later integration an adapter-only task.
Logical-request identity propagation, feedback production, scheduler
arbitration, and reclaim candidate extraction require substantive changes to
Pie RPC and runtime internals.

**Initial decision:** describe the independent result as a policy kernel, not
an integration-complete subsystem. Keep the ABI at `0.x` until at least one
gateway operation and one engine operation have been integrated.

## 2. Attachment and State Scope

PLEX operations execute in different processes:

- admission and routing normally run in a gateway;
- scheduling and eviction run in a worker engine;
- feedback can originate in both.

One in-memory attachment registry and map store cannot coordinate these
operations across processes. Doing so correctly would introduce a distributed
state and deployment protocol.

**Initial decision:** make attachments and writable state process-local.
Demonstrate cross-operation coordination only for operations hosted in the same
process. Treat cross-gateway/worker writable state as a separate distributed
systems milestone.

## 3. Decision Commit Versus Enactment

Validating a decision does not mean the decision was enacted. A selected worker
can reject a route, a scheduler candidate can become infeasible, and an
eviction candidate can fail revalidation. Committing decision-hook writes
before enactment can therefore record state for an action that never happened.

Preserving atomicity across policy evaluation and native enactment would
require a two-phase decision/enactment protocol.

**Revised decision:** permit decision hooks to stage policy-map effects. After
the hook returns, the host validates the decision and prepares the transaction
against the revisions it read. The adapter then revalidates and enacts the
decision. Successful enactment commits the prepared effects; rejection or
failure aborts them. Operations that cannot report enactment synchronously must
record outcome-dependent state from feedback instead. This is a process-local
two-phase protocol, not crash-atomic coordination with an external engine.

## 4. Hidden Guest State

Reusing a Wasm instance allows mutable globals and linear memory to become
undeclared policy state. That state is not transactional, can survive a failed
invocation, and can invalidate rollback and replay guarantees.

**Initial decision:** cache the compiled `Component` and `InstancePre`, but
create a fresh `Store` and component instance for every invocation. Discard the
instance after the call. Optimize this only after measuring instantiation cost.

## 5. WIT Operation Shape

A component world is statically typed, while the design allows a package to
implement any subset of the five operations. Optional exports and generated
host/guest bindings need one explicit representation.

**Initial decision:** use one versioned world in which every component exports
all five entry points. The manifest declares operation ownership, and the SDK
generates unavailable stubs for unimplemented operations. Optional fields,
maps, events, and capabilities should be represented through linked handles,
not optional WIT imports.

## 6. Map Lookup and Boundary-Crossing Model

The paper proposes pre-joining all map values so each operation enters Wasm
once. Arbitrary guest code such as `map.get(computed_key)` cannot be
pre-materialized unless the host knows the lookup key before invocation.

The contract supports both:

1. declarative joins whose map and key-source field are declared in the
  manifest; and
2. host map calls during the invocation.

**Revised decision:** host-backed lookup is the general semantic model, so
guest code may call `map.get(computed_key)`. Every call uses an
attachment-linked handle and is charged against fuel, call-count, and byte
limits. Declarative joins are an optional hot-path optimization with identical
lookup semantics, not a restriction on which keys policy code may compute.

## 7. Map Transaction and Concurrency Semantics

Concurrent invocations require explicit snapshot, stale-read, lost-update, and
conflict semantics. Serializing every stateful invocation would avoid conflicts
but would also put one attachment-wide lock on the hot path.

**Revised decision:** use optimistic transactions. An invocation reads from a
stable snapshot and records the revisions of entries it observes while staging
writes locally. Prepare succeeds only if those revisions are still current and
temporarily fences the write set through enactment. A conflict aborts the
decision and all staged effects; the adapter may retry from a fresh snapshot
within its deadline, otherwise it must use the native fallback. Define revision
granularity, overflow, equality, delete, TTL, and replay ordering precisely.
Guest-visible general CAS remains separate from revision validation at commit.

## 8. Feedback Delivery and Deduplication

An in-memory map backend cannot provide crash-durable exactly-once delivery.
Deduplication also needs an explicit key scope, retention policy, capacity, and
replacement behavior.

**Initial decision:** specify at-least-once feedback delivery with
process-lifetime idempotence. Key deduplication by policy identity and delivery
ID, retain committed IDs for the process lifetime, reject new IDs when the
bounded ledger is full, and return a typed acknowledgement. Do not claim
exactly-once behavior across process restart until a durable transactional log
exists.

## 9. Replacement and Pinned Maps

Publishing a new package while old invocations finish is unsafe when both
generations can write a transferred pinned map. An old invocation may commit
after the new generation is active. Schema equality also does not prove
semantic compatibility.

Correct live transfer requires generation fencing, aborting old commits, or a
drain protocol.

**Simplest initial tradeoff:** do not transfer writable pinned maps during live
replacement. Either wait for old invocations to drain before transfer or defer
pinned-map migration entirely.

## 10. Candidate and Enactment Semantics

Dense scores require contract-level rules for:

- stable candidate input order and tie handling;
- score direction and total ordering;
- capacity and fixed-fill behavior;
- zero or oversubscribed token budgets;
- stale candidate revalidation;
- route retry ordering;
- independently reclaimable eviction units.

These rules are especially important for Pie's scheduler. Its current batch
construction couples pipeline order, pre-launch copies, quorum, same-instance
deduplication, and driver capacity. Scheduler candidates are not independently
feasible merely because each candidate is feasible in isolation.

**Initial decision:** support only set-dependent batch invocation and a stable,
documented greedy fill rule. The adapter must preserve all mechanical
constraints and revalidate before enactment. Candidate-local standing indexes
remain deferred.

## 11. Replay Determinism

Replay determinism depends on more than deterministic Wasm. It also requires a
recorded invocation order, stable candidate order, explicit external-map
revisions, deterministic time, bounded entropy, defined floating-point
ordering, and no hidden guest state.

**Initial decision:** omit policy-visible time and entropy from the first ABI,
serialize stateful invocations, reject non-finite scores, retain input order for
ties, and record every host-visible input. Replay adapter decisions separately
from Pie-specific enactment.

## 12. Host-Consumed Intent Maps

Host-consumed maps introduce authority beyond ordinary policy state: writing a
value can reserve service or influence retention. This is effectively another
control surface and requires quotas, TTLs, consumer acknowledgement, and
failure semantics.

**Initial decision:** defer host-consumed intent maps. Add them only with the
first real consumer and an explicit effect schema. External read-only maps and
policy-owned feedback state are sufficient for the first milestone.

## Recommended First Vertical Slice

Build a single-process `schedule` plus `feedback` slice with:

- one fixed, versioned WIT world;
- set-dependent dense decisions with stable ordering;
- authoritative facts and separately typed untrusted metadata;
- strict fuel, epoch deadline, memory, output, and telemetry limits;
- a fresh Wasm instance per invocation;
- scheduling decisions with enactment-coupled staged effects;
- one bounded policy-owned map accessed through metered host calls;
- optimistic transaction conflict, retry, abort, and fallback fixtures;
- process-lifetime feedback deduplication;
- explicit fallback-required failures;
- a minimal Rust SDK, failure fixtures, and replay runner developed together.

After this slice is stable, generalize the operation contract and add the
remaining reference policies. The first live Pie integration should be
gateway routing, whose current admission and routing seams are already narrow.
Scheduler and reclaim integration should follow only after their candidate and
revalidation contracts are proven.

## Deferred Until Proven Necessary

- distributed writable maps;
- candidate-local or standing-index execution;
- host-consumed intent maps;
- guest-visible general CAS beyond revision-checked commits;
- live writable pinned-map migration;
- cross-process atomic replacement;
- `prefetch`, `rebalance`, and automatic policy composition.

## Repository Notes

- A policy-specific Wasmtime engine is required. The inferlet linker exposes
  WASI, HTTP, filesystem policy, and dynamic dependencies that must not be part
  of the PLEX authority surface.
- `runtime/policy` should remain below gateway and engine adapters and must not
  depend on `pie-engine`, `pie-gateway`, or `pie-worker`.
- `sdk/rust` is excluded from the root Cargo workspace, so the PLEX SDK and
  reference policies need explicit CI build steps.
- `tests/policies` should be a nested Wasm workspace and be excluded from the
  host workspace.
- `scripts/sync-wit.sh` is the canonical drift path for inferlet and PLEX WIT
  mirrors; CI must run it whenever either interface changes.

## Serving-Policy Stress Results

The reference corpus includes direct policy adaptations from recent primary
sources:

| Policy | PLEX coverage | Intentionally outside the core |
| --- | --- | --- |
| [Agentix (NSDI 2026)](https://www.usenix.org/system/files/nsdi26-luo.pdf) | program-level attained-service scheduling, discretized priorities, anti-starvation, feedback accounting | KV swap kernels and engine time quanta |
| [Continuum](https://arxiv.org/abs/2511.02230) | TTL-backed retention, preempted/TTL/program-FCFS scheduling, eviction | tool parsing, TTL estimation, physical KV pinning |
| [KVFlow](https://arxiv.org/abs/2507.07400) | steps-to-execution eviction and cache-loading-aware scheduling | proactive CPU-to-GPU prefetch |
| [Preble](https://arxiv.org/abs/2407.00023) | E2 exploit/explore placement using prefix reuse, load, and eviction cost | prefix replication and autoscaling |
| [Helium](https://arxiv.org/abs/2603.16104) | ready-operator, critical-path, cache-reuse, and forced-progress scheduling | query-plan rewrite and proactive cache warming |

These policies execute as `wasm32-unknown-unknown` components through the same
host and replay path. The stress corpus also exercises operation-scoped fields,
computed-key map reads, TTL expiry, feedback deduplication, replacement state
transfer, conflict retry, malformed output, fuel exhaustion, traps, quotas,
and bounded lossy telemetry.
