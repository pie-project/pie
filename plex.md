# PLEX v0.5 Engine API and Explicit Policy Transactions

## Status

PLEX is an intentionally untyped serving-policy research subsystem. The
implemented stack is:

```text
PIE / vLLM adapter / SGLang adapter
        |
        | pie.plex.engine@1 JSON event -> outcome
        v
PlexRuntime + PolicyStateBackend
        |
        | pie:plex@0.5.0 explicit invocation/output
        v
Wasm policy
        |
        | query() / staged action()
        v
PLEX host
```

The implementation includes:

- one strict engine-facing JSON operation in Rust and Python;
- five explicit policy hooks;
- mutable shared and request state;
- explicit state snapshots and updates in WIT;
- immediate read-only engine queries;
- staged actions returned to the underlying engine;
- normalized route, admit, schedule, evict, and feedback decisions;
- revisioned compare-and-swap state commits;
- feedback idempotency;
- bounded, replaceable Wasm execution;
- a standalone PyO3/Maturin Python package; and
- mock PIE, vLLM, and SGLang conformance paths.

PLEX does not yet include etcd or version-pinned live vLLM/SGLang integration.
The Python adapters are thin engine-neutral compatibility templates tested with
mock schedulers.

## Stable Policy Waist

The policy hooks remain:

```text
route
admit
schedule
evict
feedback
```

Every generation follows:

```text
route -> admit -> schedule
```

This applies to an initial generation, a tool continuation, a later generation,
and a stage transition that performs a new placement/admission decision.

The engine integration API is deliberately different: it has one operation,
`PlexRuntime::invoke`, and carries the hook name as data.

## Trust Model

Operator-installed policy code is trusted to inspect and rewrite complete
request fields, including prompts, messages, sampling parameters, tools,
output constraints, and user metadata.

Wasm provides:

- a portable ABI;
- memory isolation;
- bounded execution;
- trap containment; and
- atomic package replacement.

Wasm is not an authorization boundary between a policy and request contents.
Host-owned request facts remain immutable so policy bugs cannot forge identity
or enacted observations.

## Engine-Facing API

### Rust

```rust
pub struct PlexRuntime {
    // Registry, backend, query handler, supported actions, and invocation gate.
}

impl PlexRuntime {
    pub fn invoke(
        &self,
        event: serde_json::Value,
    ) -> Result<serde_json::Value, PlexError>;

    pub fn invoke_json(
        &self,
        event_json: &str,
    ) -> Result<String, PlexError>;
}
```

`invoke()` is canonical. `invoke_json()` parses, calls the same implementation,
and serializes its result.

Engines do not implement WIT, manipulate backend revisions, or call component
exports directly.

### Event envelope

```json
{
  "api_version": "pie.plex.engine@1",
  "hook": "schedule",
  "context": {
    "runnable": [],
    "capacity": {},
    "context": {}
  },
  "request_events": []
}
```

The top-level key set is exact:

- `api_version`: exactly `pie.plex.engine@1`;
- `hook`: `route`, `admit`, `schedule`, `evict`, or `feedback`;
- `context`: hook-specific transient context;
- `request_events`: authoritative lifecycle updates.

Unknown keys, versions, hooks, or malformed event shapes are API errors. They
are not fallback-shaped outcomes.

PLEX inserts a default cause when a non-feedback context omits one and
advertises the runtime's supported action methods through
`context.capabilities.actions`.

## Request Lifecycle Events

The unified API carries infrequent state-boundary events.

### Create

```json
{
  "op": "create",
  "request_id": "L",
  "facts": {
    "generation_id": 0
  },
  "fields": {
    "body": {},
    "metadata": {}
  }
}
```

### Continue

```json
{
  "op": "continue",
  "request_id": "L",
  "facts": {
    "generation_id": 1
  },
  "fields": {
    "body": {},
    "metadata": {}
  }
}
```

### Merge facts

```json
{
  "op": "merge-facts",
  "request_id": "L",
  "facts": {
    "attained_service": 128,
    "previous_target": "node-a"
  }
}
```

### Finish

```json
{
  "op": "finish",
  "request_id": "L"
}
```

Rules:

- all event shapes and their simulated request generations validate before
  mutation;
- create, continue, and fact updates apply before policy state is loaded;
- authoritative observations remain even if the policy later falls back;
- finish is valid only with feedback and only for a request referenced by it;
- successful feedback commits mutation, deduplication, and removal atomically;
- unavailable or fallback feedback still removes finished requests;
- `previous_target` changes only after engine-confirmed placement.

The in-memory backend applies validated pre-hook events sequentially. A future
distributed backend may provide a batch transaction for backend-failure
atomicity.

## Engine Outcomes

### Success

```json
{
  "status": "success",
  "decision": {},
  "request_fields": {
    "L": {}
  },
  "actions": [
    {
      "id": 0,
      "method": "pie.kv.prefetch@1",
      "args": {
        "request_id": "L",
        "target": "node-a"
      }
    }
  ]
}
```

- `decision` is normalized and engine-actionable;
- `request_fields` contains complete fields only when fields actually changed;
- shared and request scratch never leak to the engine;
- actions preserve policy call order;
- action IDs start at zero for each invocation;
- actions appear only after result validation and state commit.

The underlying engine applies request fields, enacts actions, and reports
enacted outcomes through feedback. PLEX has no engine-side action callback.

### Normalized decisions

```text
route
  -> {"order": [candidate indexes]}

admit
  -> {"decision": "accept" | "defer" | "reject"}

schedule
  -> {"selected": [{"candidate_index": ..., "token_budget": ...}]}

evict
  -> {"selected": [{"candidate_index": ..., "size_bytes": ...}]}

feedback
  -> {}
```

Stable tie-breaking, token-capacity fill, and eviction fill happen exactly once
inside PLEX. Engine adapters do not reimplement them.

### Unavailable

```json
{"status": "unavailable"}
```

The engine uses its native policy.

### Fallback

```json
{
  "status": "fallback",
  "failure": {
    "kind": "invalid-output",
    "message": "..."
  }
}
```

No policy state update, request-field update, or action is exposed.

Invalid engine events, backend failures, invalid packages, and runtime
construction failures are `PlexError` API errors.

## Persistent State

```text
State
├── shared
└── requests
    └── <logical-request-id>
        ├── facts
        ├── fields
        └── scratch
```

### Shared scratch

`State.shared` is one mutable JSON dictionary shared by all active operation
owners using a backend:

```json
{
  "tenant_service": {
    "acme": 120
  },
  "learned_statistics": {
    "decode_tokens_p50": 18
  }
}
```

There are no shared facts or fields. Shared observations belong in hook context
or `query()`. Engine effects are staged through `action()`.

### Request state

```json
{
  "facts": {
    "logical_request_id": "L",
    "generation_id": 1,
    "previous_target": "node-a",
    "attained_service": 128
  },
  "fields": {
    "body": {
      "prompt": "hello"
    },
    "metadata": {
      "workflow_id": "wf-1"
    }
  },
  "scratch": {
    "predicted_tokens": 64,
    "admission_count": 2
  }
}
```

| Namespace | Writer | Meaning |
|---|---|---|
| `facts` | Engine/host | Identity and enacted observations |
| `fields` | Policy | Canonical request data interpreted by the engine |
| `scratch` | Policy | Request-local policy state opaque to the engine |

Facts are immutable in the author-facing SDK and cannot appear in a returned
state update. Fields and scratch must remain JSON objects.

## Working Sets

The host derives request IDs from validated hook context and loads only those
requests.

- route/admit load one request;
- schedule loads each unique runnable request;
- duplicate runnable entries share one mutable request object;
- evict loads only non-null attributed requests;
- feedback records with one ID share one mutable object;
- unreferenced live requests are invisible;
- missing state is a backend API error through `PlexRuntime`.

Schedule remains set-dependent: one invocation sees the complete feasible
runnable set for one scheduling opportunity.

## Hook Contexts

Hook contexts retain the v0.4 JSON conventions.

### Route

```json
{
  "cause": "continuation",
  "request_id": "L",
  "candidates": [
    {
      "id": "node-a",
      "facts": {
        "queue_depth": 10,
        "cached_tokens": 2048,
        "has_request_kv": true
      }
    }
  ],
  "context": {
    "capabilities": {}
  }
}
```

Return one finite score per candidate. Higher ranks first; ties are stable.

### Admit

```json
{
  "cause": "continuation",
  "request_id": "L",
  "target": {
    "id": "node-a",
    "facts": {
      "queue_depth": 10,
      "free_kv_bytes": 1048576
    }
  },
  "context": {
    "capabilities": {}
  }
}
```

Return `accept`, `defer`, or `reject`.

### Schedule

```json
{
  "cause": "service-step",
  "runnable": [
    {
      "request_id": "L",
      "facts": {
        "waiting_ms": 12
      },
      "max_token_budget": 8
    }
  ],
  "capacity": {
    "max_selected": 1,
    "max_total_tokens": 8,
    "max_token_budget": 8
  },
  "context": {
    "capabilities": {
      "token_budget": true
    }
  }
}
```

Return one aligned finite score and optional token budget per runnable entry.
Explicit budgets require capability and may not exceed candidate/host bounds.

### Evict

Resident units carry an ID, size, facts, and nullable request ID. Return one
finite retention score per unit. Lower scores reclaim first with stable ties
until the requested byte deficit is met.

### Feedback

Feedback carries a delivery ID and open-string event records describing enacted
outcomes. Duplicate deliveries skip Wasm, state updates, and actions.

## Continuations and Placement

Continuation creation:

1. preserves request scratch;
2. preserves fields except body/metadata handling;
3. replaces body;
4. shallow-merges metadata;
5. increments generation;
6. preserves durable facts;
7. clears generation-local `current_target`.

Route/admit proposals do not change `previous_target`. Engines merge that fact
only after placement is enacted.

## WIT v0.5

```wit
package pie:plex@0.5.0;

interface policy {
    record invocation {
        context-json: string,
        state-json: string,
    }

    record policy-output {
        result-json: string,
        state-update-json: string,
    }

    route: func(input: invocation)
        -> result<policy-output, string>;
    admit: func(input: invocation)
        -> result<policy-output, string>;
    schedule: func(input: invocation)
        -> result<policy-output, string>;
    evict: func(input: invocation)
        -> result<policy-output, string>;
    feedback: func(input: invocation)
        -> result<policy-output, string>;
}

interface host {
    type action-id = u64;

    query: func(method: string, args-json: string)
        -> result<string, string>;
    action: func(method: string, args-json: string)
        -> result<action-id, string>;
}

world plex-policy {
    import host;
    export policy;
}
```

Every fixture imports exactly `pie:plex/host@0.5.0`, exports exactly
`pie:plex/policy@0.5.0`, and imports no state interface, WASI, or other
capability.

### Explicit state input

`invocation.state-json` is:

```json
{
  "shared": {},
  "requests": {
    "L": {
      "facts": {},
      "fields": {},
      "scratch": {}
    }
  }
}
```

Backend revisions never cross WIT.

### Explicit state output

`policy-output.state-update-json` is:

```json
{
  "shared": {},
  "requests": {
    "L": {
      "fields": {},
      "scratch": {}
    }
  }
}
```

`{}` means no change. Unchanged scopes are omitted. Facts, unknown request IDs,
and unknown namespaces are rejected. Changed fields/scratch are returned in
full; JSON Patch is deferred.

There is no ambient state import or load/stage call-order protocol.

## Rust Guest SDK

```rust
pub type Document = serde_json::Value;
pub type Result<T> = std::result::Result<T, String>;
pub type ActionId = u64;

pub struct State {
    pub shared: Document,
    // request map is private
}

pub struct Request {
    // facts is private
    pub fields: Document,
    pub scratch: Document,
}
```

The five trait methods receive:

```rust
fn schedule(
    ctx: &Document,
    state: &mut State,
    host: &Host,
) -> Result<Document>;
```

Glue parses explicit context/state, clones initial mutable state, invokes the
policy, validates facts and membership, computes a minimal update, and returns
`PolicyOutput`.

### Direct query helpers

```rust
host.kv_lookup(request_id, target)?;
host.cluster_capacity(model)?;
host.model_config()?;
host.now_ms()?;
```

### Direct action helpers

```rust
host.prefetch_kv(request_id, target)?;
host.preempt(request_id)?;
host.replicate(request_id, targets)?;
host.set_retention(request_id, ttl_ms)?;
host.arm_timer(request_id, delay_ms)?;
```

### Raw extensions

```rust
host.query_raw("engine.custom-query@1", &args)?;
host.action_raw("engine.custom-action@1", &args)?;
```

Direct methods are SDK sugar over versioned raw methods. Adding sugar or a new
method name does not change WIT.

## Query Handling

The only engine callback is:

```rust
pub trait QueryHandler: Send + Sync {
    fn query(
        &self,
        method: &str,
        args: &Document,
    ) -> Result<Document, QueryError>;
}
```

Queries are synchronous, bounded, and side-effect-free. Unsupported methods
return a policy-visible error. A policy may handle that error or propagate it
as fallback.

Candidate-scale queue, waiting, capacity, and cache snapshots remain bulk hook
facts; normal paths do not query once per candidate.

## Action Staging

`host.action()`:

1. checks a non-empty numeric-versioned method name;
2. checks runtime support registration;
3. parses object-valued JSON arguments;
4. enforces call/byte limits;
5. assigns an invocation-local monotonic ID;
6. appends a descriptor.

Invalid policy result/update or state conflict discards every descriptor.
Successful commit returns descriptors to the engine. PLEX never enacts them.

Outcome-dependent policy state changes occur from feedback, not optimistically
when staging an action.

## Backend Transaction

`PlexRuntime` uses `Arc<dyn PolicyStateBackend>`. The implemented in-memory
backend stores shared state, request state, revisions, and feedback delivery
results.

```text
validate engine event
-> validate/preflight request events
-> apply authoritative pre-hook events
-> derive working set
-> load state + revisions
-> invoke Wasm with explicit context/state
-> validate result and update
-> backend CAS commit
-> normalize decision
-> return changed fields and actions
```

State conflict returns fallback with no engine-visible mutation or action.
Other backend failures are API errors.

Multiple runtimes may share a backend. The current backend is process-local. A
future etcd mapping can use:

```text
/plex/shared
/plex/requests/<logical-request-id>
/plex/feedback/<delivery-id>
```

Wasm must continue receiving a preloaded state snapshot; it must not perform
an etcd call itself.

## Python Binding

The standalone package is:

```text
sdk/python-plex/
├── Cargo.toml
├── pyproject.toml
├── src/lib.rs
├── python/pie_plex/
└── tests/
```

Distribution/module names:

```text
distribution: pie-plex
package: pie_plex
native module: pie_plex._native
```

Usage:

```python
from pie_plex import Runtime

runtime = Runtime(
    policy="policy.plexpkg",
    query=engine_query,
    actions=[
        "pie.kv.prefetch@1",
        "pie.schedule.preempt@1",
    ],
)

outcome = runtime.invoke(event)
```

The native seam is:

```text
NativeRuntime.invoke_json(event_json: str) -> str
```

The Python facade uses `json.dumps`/`json.loads`, so Rust and Python share one
contract and no second recursive domain conversion.

The query callback is stored as `Py<PyAny>`. Wasmtime execution detaches from
the GIL; query calls attach only while invoking Python. Exceptions and
non-JSON results become policy-visible query errors. Recursive invocation on
the same runtime while a query is active is rejected. Independent runtimes may
execute concurrently.

Python exceptions are:

```text
PlexError
├── InvalidEvent
├── BackendError
├── PolicyPackageError
└── QueryCallbackError
```

Policy unavailable/fallback remains a dictionary outcome, not an exception.

## Engine Adapters

The Rust conformance harness runs identical traces through mock PIE, vLLM, and
SGLang adapters.

The Python package includes thin vLLM/SGLang scheduler templates that:

1. snapshot runnable/capacity/context/request events;
2. call `Runtime.invoke`;
3. select native scheduling on non-success;
4. apply normalized decisions and changed fields;
5. enact returned actions in order.

They import no vLLM or SGLang internals. Version-specific snapshot/apply code
and version-pinned live integration remain separate work. No live support is
claimed by this milestone.

## Runtime Protections

- exact one-import/one-export component validation;
- no WASI;
- fresh Wasmtime store and component instance per invocation;
- disabled Wasm threads/shared memory;
- package integrity digest;
- memory and fuel limits;
- epoch deadline;
- explicit context/state and result/update byte limits;
- host-call count and byte limits;
- bounded concurrent invocation slots;
- bounded feedback ledger;
- atomic package publication.

## Replay and Fixtures

Low-level deterministic replay retains attach/replace/detach, request lifecycle,
state inspection, fact updates, placement recording, hook invocation, combined
route/admit, and terminal feedback.

The fixture harness additionally covers the engine JSON API, Rust/JSON parity,
strict event validation, normalized outcomes, scratch hiding, returned actions,
request-event preflight, fallback cleanup, query/action helpers, raw extensions,
state conflicts, feedback deduplication, and mock adapter conformance.

Research stress policies remain for Agentix, Continuum, KVFlow, Preble, and
Helium. Their physical engine mechanisms remain out of scope.

## Intentional Limitations

This milestone does not provide:

- typed schemas for every context/fact/query/action;
- generated request models;
- field/map/event/capability handles;
- JSON Patch;
- actual etcd integration;
- per-scheduler-step etcd round trips;
- arbitrary live-request scanning from policy code;
- candidate-local scheduling;
- an engine-side action callback;
- rollback after an engine enacts an action;
- sidecar/RPC scheduler hops;
- Python policy authoring;
- live version-pinned vLLM/SGLang plugins;
- automatic rerouting after field mutation;
- shared-unit multi-owner mutation.

`plex_paper.md` remains unchanged. Paper reconciliation and measurements are a
separate task.
