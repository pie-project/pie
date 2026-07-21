# PLEX JSON State-Map PoC

## Status

PLEX is an intentionally untyped policy research subsystem with a breaking
`pie:plex@0.3.0` ABI. It validates:

- five serving-policy operations;
- `route -> admit -> schedule` for every generation;
- one process-local global map;
- one process-local map per logical request;
- explicit host/policy ownership within those maps;
- atomic JSON mutation and native fallback;
- feedback idempotency;
- deterministic score and token-budget mechanics; and
- bounded, replaceable Wasm policy execution.

PLEX remains isolated from live Pie gateway, worker, scheduler, KV-reclaim,
and inferlet mechanics. Adapters for those systems are future work.

## Design Boundary

The five operations are:

```text
route
admit
schedule
evict
feedback
```

All five use one mutable JSON object. The ABI has no WIT records, schemas,
handles, typed maps, host calls, or imported capabilities.

The persistent state model has two scopes:

1. one global map shared by all active operation owners in a `LifecycleHost`;
2. one request map for each logical request.

Operation-specific candidates, facts, capacity, records, and causes are
transient host-owned context.

## Trust Model

Operator-installed policy code is trusted to inspect request contents and to
write policy-owned namespaces. Wasm provides:

- a portable ABI;
- memory isolation;
- bounded execution;
- trap containment; and
- package replacement.

Wasm is not an authorization boundary between a policy and request data.
Namespace ownership is nevertheless validated to catch policy bugs and protect
host observations.

## Persistent Map Shape

Both persistent scopes contain exactly three object-valued namespaces:

```json
{
  "facts": {},
  "fields": {},
  "scratch": {}
}
```

Values below those namespaces are arbitrary JSON.

### Ownership

| Namespace | Writer | Meaning |
|---|---|---|
| `facts` | Host only | Configuration, identity, and enacted observations |
| `fields` | Policy | Values an adapter or engine may interpret |
| `scratch` | Policy | State persisted by the host but opaque to the engine |

Policies may change `fields` and `scratch`. Any attempted mutation of `facts`
invalidates the whole response and triggers fallback.

## Global Map

There is one global map per `LifecycleHost`:

```json
{
  "facts": {
    "config": {},
    "model": "example-model",
    "replica_count": 4
  },
  "fields": {
    "scheduler": {}
  },
  "scratch": {
    "acme.routing": {},
    "acme.scheduler": {}
  }
}
```

Semantics:

- all active PLEX packages in the host share the same map;
- state survives invocations, request completion, and package replacement;
- `facts` may change through explicit host methods;
- `fields` may carry engine-interpreted policy output;
- `scratch` is arbitrary cross-request policy state;
- `reset_global` clears `fields` and `scratch` but preserves facts;
- process restart clears the entire in-memory store.

There is no per-package isolation. Independently authored policies must
namespace shared keys by convention.

## Request Maps

Each logical request has:

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
    "route_count": 2,
    "predicted_tokens": 64
  }
}
```

`facts.logical_request_id` must match the key in the invocation's `requests`
object. `facts.generation_id` is host-owned.

`fields.body` is the mutable engine request. `fields.metadata` contains
arbitrary user/application metadata and is also policy-mutable. `scratch`
holds request-local policy state.

The host creates and removes request maps. A policy cannot create, delete, or
rename them.

## Uniform Invocation Envelope

Every actual Wasm invocation receives:

```json
{
  "global": {
    "facts": {},
    "fields": {},
    "scratch": {}
  },
  "requests": {
    "L": {
      "facts": {},
      "fields": {},
      "scratch": {}
    }
  },
  "... operation-specific context ...": {}
}
```

`requests` contains each referenced request exactly once. Transient objects
refer to a request with `request_id`.

Callers of `LifecycleHost` provide only transient context and request IDs.
`PolicyStateStore` hydrates canonical global/request maps before invoking Wasm.
Caller-supplied `global` or `requests` state is rejected.

Extra canonical request maps that are not referenced are not exposed.
Duplicate references share one object rather than receiving duplicated copies.

## Operation Contracts

These contracts are JSON conventions validated by the host, not typed WIT
schemas.

### `route`

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

Result:

```json
{"scores": [790.0]}
```

Rules:

- return exactly one finite score per candidate;
- higher scores rank first;
- ties retain input order;
- candidate count, order, identity, and facts are read-only;
- the referenced request and global map may mutate in writable namespaces.

### `admit`

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

Result:

```json
{"decision": "accept"}
```

`decision` is `accept`, `defer`, or `reject`. The target and all transient
context are read-only. Writable global/request mutations commit even for a
valid defer or reject decision.

### `schedule`

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

Result:

```json
{
  "decisions": [
    {
      "score": 10.0,
      "token_budget": 8
    }
  ]
}
```

Rules:

- return one decision per runnable entry;
- scores are finite, descending, and stably ordered;
- missing or null budget requests the candidate/host maximum;
- zero budget requests no service;
- explicit budgets require `token_budget` capability;
- budgets may not exceed candidate or host bounds;
- stable greedy fill enforces selected-count and total-token capacity.

The same request ID may occur more than once. It remains one shared request map
while operation decisions stay aligned with runnable entries.

### `evict`

```json
{
  "cause": "allocation-deficit",
  "bytes_needed": 4096,
  "resident": [
    {
      "id": "unit-1",
      "request_id": "L",
      "size_bytes": 4096,
      "facts": {
        "reload_cost": 200.0
      }
    },
    {
      "id": "shared-unit",
      "request_id": null,
      "size_bytes": 4096,
      "facts": {}
    }
  ],
  "context": {
    "capabilities": {}
  }
}
```

Result:

```json
{"scores": [200.0, 100.0]}
```

Scores are retention scores. Lower scores are reclaimed first with stable ties
until `bytes_needed` is met. `request_id: null` represents unattributed or
shared host state. Shared-unit multi-owner mutation is not implemented.

### `feedback`

```json
{
  "delivery_id": "delivery-1",
  "records": [
    {
      "event": "progress",
      "request_id": "L",
      "facts": {
        "committed_tokens": 8,
        "service_us": 1250
      }
    },
    {
      "event": "tool-boundary",
      "request_id": "L",
      "facts": {
        "tool_name": "search"
      }
    }
  ],
  "context": {
    "capabilities": {}
  }
}
```

Result:

```json
{}
```

Event names are open strings. Records describe enacted outcomes. Multiple
records referencing `L` mutate the same `requests["L"]` object, so both updates
survive without last-write-wins loss.

## Per-Generation Lifecycle

Every initial generation, tool continuation, later generation, and
placement-changing stage transition follows:

```text
route -> admit -> schedule
```

Routing produces a stable descending target order. Admission then runs against
the preferred target:

- `accept`: enqueue;
- `reject`: try the next ranked target;
- `defer`: retry the same target up to the configured bound.

A tool boundary is:

```text
feedback(tool-boundary)
-> external tool execution
-> create continuation
-> route
-> admit
-> schedule
```

The PoC does not implement a distributed retry protocol.

## Continuations

Creating a continuation:

1. preserves request `scratch`;
2. preserves request `fields` except for body/metadata handling;
3. replaces `fields.body`;
4. shallow-merges new metadata into `fields.metadata`;
5. preserves durable host facts;
6. increments `facts.generation_id`;
7. clears generation-local facts.

The new metadata value wins on duplicate keys.

The implemented generation-local fact list currently contains only:

```text
current_target
```

`previous_target` is durable and survives continuations.

## Enacted Placement

Routing and admission decisions are proposals, not observations. Neither a
route score nor `admit: accept` changes `previous_target`.

After an adapter confirms placement, it calls:

```text
record_enacted_placement(logical_request_id, target_id)
```

This sets host-owned `facts.previous_target`. Continuation routing may combine
that history with current candidate cache facts. The reference route retains
locality only when the historical target still reports useful request KV.

## Host Fact API

`PolicyStateStore` and serialized `LifecycleHost` wrappers provide:

```text
replace_global_facts(facts)
merge_global_facts(facts)
merge_request_facts(logical_request_id, facts)
record_enacted_placement(logical_request_id, target_id)
```

Request identity facts cannot be changed through generic fact merge. Host fact
updates describe reality and are never rolled back by a later policy failure.

## State Transaction

`LifecycleHost` has one stateful-invocation mutex shared by all clones. The
critical section is:

```text
hydrate canonical global/request state
-> invoke Wasm
-> validate the full response
-> atomically commit global/request fields and scratch
```

This deliberately serializes stateful policy invocations. It avoids revisions,
compare-and-swap loops, and distributed transactions in the PoC.

`route -> admit` holds the same gate across the sequence while committing each
valid hook before the next hook runs.

`PolicyStateStore` publishes mutations by cloning its complete in-memory state,
updating writable namespaces, and replacing the canonical state under one
mutex. Global and all referenced request changes therefore commit together.

Scalable concurrent state transactions are deferred.

## Mutation Validation

A valid policy response must preserve:

- the exact top-level input key set;
- the global scope and request-map scope shapes;
- `global.facts`;
- every request's `facts`;
- the exact request-map key set;
- operation cause;
- candidates, targets, resident units, and runnable entries;
- candidate/event facts;
- capacity and budgets;
- feedback delivery ID and record order;
- capabilities and all other transient context.

The host does not silently restore attempted read-only mutations. Any mismatch
rejects the complete response and requests native fallback.

Only `global.fields`, `global.scratch`, and each referenced request's `fields`
and `scratch` may change.

## Failure and Rollback

Invocation outcomes are:

```text
Success
Unavailable
FallbackRequired
```

Fallback classes include invalid input, instantiation failure, explicit policy
fallback, trap, fuel exhaustion, deadline, host saturation, and invalid output.

For any failure:

- no global policy mutation commits;
- no request policy mutation commits;
- feedback delivery is not marked committed;
- the caller uses native/default behavior.

There is no patch log or general rollback protocol. Atomicity comes from
keeping canonical state untouched until the complete cloned JSON response has
validated.

## Feedback Deduplication

Deduplication belongs to `PolicyStateStore`, not a package instance.

For a new delivery:

1. hydrate canonical state;
2. invoke and validate feedback;
3. atomically commit fields/scratch and the delivery ID;
4. cache the operation result.

For a duplicate:

- Wasm is not invoked;
- no mutation is applied;
- the cached result is returned.

The ledger is bounded, process-local, and survives package replacement because
it is part of the host state store. A failed state commit does not reserve or
commit the delivery ID. Deduplication is not crash durable.

## Wasm ABI

The complete WIT is:

```wit
package pie:plex@0.3.0;

interface policy {
    route: func(input-json: string) -> result<string, string>;
    admit: func(input-json: string) -> result<string, string>;
    schedule: func(input-json: string) -> result<string, string>;
    evict: func(input-json: string) -> result<string, string>;
    feedback: func(input-json: string) -> result<string, string>;
}

world plex-policy {
    export policy;
}
```

Components import nothing and export only `pie:plex/policy@0.3.0`.

The manifest contains only:

- contract version;
- package name and version;
- owned operations;
- memory limit;
- fuel limit;
- deadline;
- input byte limit;
- output byte limit.

Unknown fields and older contracts are rejected.

## Guest SDK

The Rust SDK remains:

```rust
pub type Document = serde_json::Value;

pub trait Policy {
    fn route(input: &mut Document) -> Result<Document, String>;
    fn admit(input: &mut Document) -> Result<Document, String>;
    fn schedule(input: &mut Document) -> Result<Document, String>;
    fn evict(input: &mut Document) -> Result<Document, String>;
    fn feedback(input: &mut Document) -> Result<Document, String>;
}
```

Every hook defaults to `fallback-required`.

`export_policy!` parses object JSON, invokes the mutable hook, and returns:

```json
{
  "input": {},
  "result": {}
}
```

Returning the full input is intentionally inefficient. It keeps mutation
obvious and avoids patches, diffs, generated accessors, and typed models.

## Package and Runtime

The package envelope has:

- magic and format version;
- bounded manifest/component lengths;
- BLAKE3 integrity digest;
- manifest bytes;
- component bytes.

Attachment validates the manifest and host limits, compiles the component,
rejects every import and unexpected export, prelinks it, probes bounded
instantiation, and only then publishes.

The immutable attachment registry retains one owner per operation. Replacement
prepares first, waits for old snapshots to drain, and atomically publishes a
new generation. Persistent state does not move between package objects because
it already belongs to `LifecycleHost`.

Each call receives a fresh Wasmtime store and component instance. Runtime
protection includes:

- no WASI or PLEX imports;
- disabled Wasm threads/shared memory;
- memory limit;
- fuel;
- epoch deadline;
- input/output limits;
- bounded Wasmtime invocation slots;
- bounded feedback ledger.

## Replay

The deterministic replay runner supports:

- attach, replace, and detach;
- create and continue request;
- read global/request maps;
- replace/merge global facts;
- merge request facts;
- record enacted placement;
- invoke any operation from transient context;
- combined route/admit placement;
- terminal feedback and removal.

Replay uses no realtime epoch ticker. Identical traces on fresh stores must
produce identical reports.

## Policy Fixtures

Reference policies include:

- continuation-aware least-loaded/locality route;
- rewrite-and-admit;
- host-observed least-attained-service schedule;
- retention-score eviction;
- feedback accounting;
- coordinated five-operation policy.

Failure fixtures cover:

- unimplemented fallback;
- malformed result;
- NaN/infinity;
- invalid token budget;
- global/request fact mutation;
- candidate identity/fact mutation;
- feedback fact mutation;
- request-map insertion;
- mutate-then-fallback;
- trap after mutation;
- infinite loop.

Research stress policies remain for Agentix, Continuum, KVFlow, Preble, and
Helium. Their physical engine mechanisms remain out of scope.

## Process Lifetime and Deferred Work

Global maps, request maps, and feedback deduplication are in-memory and
process-local. This milestone does not provide:

- typed schemas or generated models;
- field/map/event/capability handles;
- host map imports;
- columnar encoding;
- patches or diffs;
- distributed global/request state;
- persistent disk state;
- crash-durable deduplication;
- scalable concurrent transactions;
- per-package global isolation;
- automatic fact derivation from arbitrary event JSON;
- automatic rerouting after downstream mutation;
- candidate-local indexes;
- shared-unit multi-owner mutation;
- `prefetch` or `rebalance`;
- live serving-engine adapters.

Efficient encodings, schema validation, and distributed integration remain
future work. `plex_paper.md` is unchanged; reconciling paper claims is a
separate task.
