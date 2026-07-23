# PLEX v0.5: Comprehensive Programming Model and Design

## Document Status

This document describes the PLEX programming model implemented by the current
v0.5 codebase. It is the design reference for:

- the engine API `pie.plex.engine@1`;
- the WebAssembly component contract `pie:plex@0.5.0`;
- the `pie-policy` host runtime;
- the Rust guest SDK;
- the Python host binding; and
- the mock conformance paths and live version-pinned vLLM and SGLang
  integrations.

The implementation is a research prototype, but the behavior documented here
is implemented and tested unless a section explicitly says otherwise. Older
planning documents describe intermediate designs and are not normative for
v0.5.

PLEX v0.5 is intentionally JSON-oriented. "Untyped" in this document means
that application-specific values are JSON rather than generated schema types.
It does not mean that the runtime accepts arbitrary structure: envelopes,
ownership boundaries, lifecycle transitions, decisions, state updates, and
resource limits are validated by the host.

## 1. Motivation

LLM serving engines own the mechanisms that admit work, place requests,
schedule service, retain KV state, and reclaim memory. Applications know
workflow structure that an engine cannot reliably infer: which generations
belong to one logical task, whether a continuation is likely, which requests
share a tenant or workflow, and which cached prefix will be reused.

Passing that information as metadata is not enough when the engine still owns
a fixed policy. Forking an engine to test every new policy gives the policy
access to the right mechanisms, but couples a small policy hypothesis to a
large and rapidly changing codebase.

PLEX separates policy from mechanism:

- the engine reports feasible choices and authoritative observations;
- a portable policy scores or selects among those choices;
- the host validates the policy result and commits policy state;
- the engine enacts the normalized decision and staged actions; and
- feedback reports what actually happened.

The central abstraction is therefore not a remote scheduler and not a metadata
schema. It is a programmable request lifecycle embedded at a stable boundary
inside a serving engine.

## 2. Design Principles

### 2.1 A five-operation stable waist

PLEX exposes five policy operations:

```text
route
admit
schedule
evict
feedback
```

They divide the serving lifecycle into:

- entry and placement: `route`, `admit`;
- recurring service arbitration: `schedule`;
- recurring residency arbitration: `evict`; and
- enacted outcomes: `feedback`.

The expected lifecycle for each generation is:

```text
route -> admit -> schedule ... -> feedback
```

`evict` can occur whenever the engine needs to reclaim resident state.
Continuations and later generations re-enter through `route` and `admit`.
The runtime validates each individual invocation, but it does not enforce this
global call ordering; the engine adapter owns the lifecycle.

### 2.2 Policy proposes; the host constrains; the engine enacts

The policy never receives raw engine pointers or mutable scheduler internals.
The engine constructs a finite feasible set. The policy returns scores,
admission intent, state changes, and staged action descriptors. PLEX then:

1. validates the result against the feasible set;
2. normalizes scores into a concrete decision;
3. atomically commits policy state; and
4. returns the decision and actions to the engine.

The engine remains the final authority over physical mechanisms.

### 2.3 One engine seam, explicit policy exports

Engine integrations use one operation:

```text
PlexRuntime::invoke(event) -> outcome
```

The hook name is data in the event. Policies, however, export five explicit
WIT functions. This gives engines a small integration surface while keeping
the policy ABI discoverable and statically linked.

### 2.4 Explicit transactions, not ambient state calls

Every policy invocation receives:

- a transient hook context; and
- an explicit snapshot of its persistent working set.

It returns:

- an operation result; and
- an explicit state update.

There is no `state.load`, `state.stage`, map handle, or required host-call
ordering protocol. The complete transaction is visible at the WIT boundary.

### 2.5 Generic helpers, stable ABI

The only host imports are:

```text
query(method, args-json)
action(method, args-json)
```

Method names are independently versioned, such as
`pie.kv.prefetch@1`. Adding a helper or Rust SDK convenience method does not
change WIT.

### 2.6 State follows the logical request

Persistent state belongs to a logical request, not to one HTTP call, one model
generation, one worker, or one engine process. A continuation increments the
generation while preserving durable facts, policy fields, and request-local
scratch.

### 2.7 Feedback closes the loop

Actions are only proposals until the engine enacts them. Policies learn about
completed work, failed actions, tool boundaries, and other outcomes through
feedback. Successful feedback delivery is deduplicated by a delivery ID.

### 2.8 eBPF-inspired, not eBPF-compatible

PLEX borrows the structural ideas that make eBPF a durable extension model:

| eBPF idea | PLEX analogue |
|---|---|
| Stable attachment points | Five lifecycle hooks |
| Maps | Explicit shared and logical-request state |
| Helpers | Versioned `query` and `action` methods |
| Verifier and loader | Package, manifest, ABI, and result validation |
| Kernel-enforced mechanism | Host-normalized decisions and engine enactment |
| Bounded execution | Wasm isolation, deadlines, memory, and call limits |

PLEX is not an eBPF ISA, verifier, map API, or compatibility layer. It uses the
WebAssembly component model and JSON transactions because serving policy is
set-dependent, request state is structured, and policy packages must remain
portable across user-space engines.

## 3. Goals and Non-Goals

### 3.1 Goals

PLEX v0.5 aims to provide:

- one engine-neutral JSON integration contract;
- portable policies compiled as WebAssembly components;
- bounded and isolated policy execution;
- five explicit lifecycle hooks;
- persistent shared and logical-request state;
- host-owned immutable facts and policy-owned mutable fields/scratch;
- complete feasible sets for set-dependent scheduling;
- immediate read-only host queries;
- staged, post-commit engine actions;
- revisioned compare-and-swap state commits;
- idempotent successful feedback delivery;
- atomic policy attachment and replacement;
- deterministic low-level replay; and
- equivalent Rust and Python host behavior.

### 3.2 Non-goals

PLEX v0.5 does not attempt to provide:

- a schema language for every engine or application field;
- a remote scheduling service or sidecar;
- direct guest access to engine internals;
- arbitrary scans over all live requests;
- one state backend round trip per decoding step;
- automatic action execution inside PLEX;
- rollback after an engine has enacted an action;
- automatic rerouting after a policy rewrites request fields;
- Python policy authoring;
- a production distributed state backend;
- signed policy packages; or
- a cluster-router implementation for the `route` hook.

## 4. Core Concepts

### 4.1 Logical request

A logical request is one admitted unit of serving work that can span multiple
model generations, tool calls, pauses, and placements. It has a stable,
non-empty `logical_request_id`.

### 4.2 Generation

A generation is one model invocation within a logical request. It has an
unsigned `generation_id`. Creation starts at generation zero. Each continuation
must advance by exactly one.

### 4.3 Hook invocation

A hook invocation is one policy transaction over:

- one transient context;
- one shared state object; and
- the exact set of request objects referenced by that context.

### 4.4 Working set

The working set is the set of logical request IDs derived from validated hook
context. It is:

- complete for that invocation;
- deduplicated by request ID;
- loaded before Wasm execution;
- immutable in membership; and
- the unit covered by revision checks.

### 4.5 Facts, fields, and scratch

Each request has three namespaces:

| Namespace | Writer | Consumer | Purpose |
|---|---|---|---|
| `facts` | Engine/host | Policy | Identity and enacted observations |
| `fields` | Engine lifecycle and policy | Policy and engine | Canonical request data |
| `scratch` | Policy | Policy only | Request-local policy memory |

The global `shared` object is policy-owned scratch shared across requests and
operation owners using the same backend.

### 4.6 Query

A query is an immediate, synchronous, read-only request to an engine-provided
`QueryHandler`. Its result is visible during the current policy invocation.

### 4.7 Action

An action is a staged descriptor for a side effect. PLEX validates and records
it but never enacts it. Actions become visible to the engine only after the
policy result and state commit succeed.

### 4.8 Feedback delivery

A feedback delivery is a batch of open-string event records identified by one
non-empty `delivery_id`. A successfully committed delivery is recorded in the
feedback ledger so retries do not execute the policy or its actions again.

## 5. System Architecture

```text
+--------------------------------------------------------------+
| Serving engine                                               |
| PIE / vLLM adapter / SGLang adapter / another integration    |
+----------------------------+---------------------------------+
                             |
                             | pie.plex.engine@1 JSON event
                             v
+--------------------------------------------------------------+
| PlexRuntime                                                  |
| - strict event validation                                    |
| - lifecycle event processing                                 |
| - working-set derivation                                     |
| - normalized outcomes                                        |
+----------------------------+---------------------------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
+---------------------------+  +-------------------------------+
| PolicyStateBackend        |  | AttachmentRegistry            |
| - shared/request state    |  | - one owner per operation     |
| - revisions               |  | - atomic attach/replace       |
| - feedback ledger         |  | - stable invocation snapshot  |
+---------------------------+  +---------------+---------------+
                                                  |
                                                  v
                                  +-------------------------------+
                                  | Wasmtime policy host          |
                                  | pie:plex@0.5.0                 |
                                  | - explicit state transaction  |
                                  | - query/action imports        |
                                  | - limits and validation       |
                                  +---------------+---------------+
                                                  |
                                                  v
                                  +-------------------------------+
                                  | Operator policy component     |
                                  | Rust SDK or raw WIT guest     |
                                  +-------------------------------+
```

### 5.1 Responsibility split

| Layer | Responsibilities |
|---|---|
| Engine adapter | Construct feasible context, report lifecycle events, apply fields, enact decisions/actions, send feedback |
| `PlexRuntime` | Validate engine API, sequence lifecycle work, map runtime failures to outcomes |
| Semantic core | Derive working set, validate policy results, normalize decisions |
| State backend | Load snapshots, enforce revisions, commit updates, deduplicate feedback |
| Wasm host | Verify package surface, meter execution, provide queries/actions |
| Policy | Implement workload-specific policy and maintain policy-owned state |

### 5.2 Trust model

Policy packages are operator-installed. They are allowed to inspect and rewrite
complete request fields, including prompts, messages, sampling parameters,
tool descriptions, output constraints, and metadata. Wasm is therefore not a
data-authorization boundary between a policy and request contents.

Wasm is used for:

- a portable component ABI;
- memory isolation from the host;
- bounded execution;
- trap containment;
- capability restriction; and
- replaceable deployment.

The host treats policy code as potentially faulty. Facts, working-set
membership, feasible choices, revisions, and action support remain
host-controlled.

## 6. Engine-Facing API

### 6.1 Rust surface

```rust
pub struct PlexRuntime {
    // Registry, state backend, query handler, action set, and invocation gate.
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

`invoke()` is canonical. `invoke_json()` only performs JSON
deserialization/serialization around the same implementation.

`PlexRuntime::from_package_bytes` constructs a runtime with:

- one attached package;
- an in-memory state backend;
- the default engine limits;
- a rejecting query handler when none is supplied; and
- two admission defer retries.

`PlexRuntime::with_parts` is the composition path for a custom registry,
backend, query handler, supported action set, and retry count.

Engine adapters do not implement WIT, instantiate Wasm, manipulate backend
revisions, or interpret raw policy scores.

### 6.2 Event envelope

```json
{
  "api_version": "pie.plex.engine@1",
  "hook": "schedule",
  "context": {
    "runnable": [],
    "capacity": {
      "max_selected": 0,
      "max_total_tokens": 0,
      "max_token_budget": 0
    },
    "context": {}
  },
  "request_events": []
}
```

The top-level key set is exact:

| Key | Requirement |
|---|---|
| `api_version` | Exactly `pie.plex.engine@1` |
| `hook` | `route`, `admit`, `schedule`, `evict`, or `feedback` |
| `context` | A hook-specific JSON object |
| `request_events` | An array of lifecycle event objects |

Missing keys, unknown top-level keys, an unknown version, an unknown hook, or
the wrong JSON type produce `PlexError::InvalidEvent`. They do not produce a
fallback-shaped outcome.

### 6.3 Context preparation

Before semantic validation, `PlexRuntime` normalizes context:

- if a non-feedback hook omits `cause`, PLEX inserts:
  - `engine-event` for `route` and `admit`;
  - `service-step` for `schedule`;
  - `allocation-deficit` for `evict`;
- if nested `context` is absent, PLEX inserts `{}`;
- if `context.capabilities` is absent, PLEX inserts `{}`; and
- `context.capabilities.actions` is set to the runtime's registered action
  method names.

The action capability list is host-authoritative. Other capability entries,
including a policy-visible `queries` list, are supplied by the engine adapter.
The actual query authority is still the configured `QueryHandler`.

## 7. Request Lifecycle Events

Lifecycle events carry authoritative, infrequent changes at the boundary
between the engine and persistent request state.

Each event has an exact key set:

| `op` | Exact keys |
|---|---|
| `create` | `op`, `request_id`, `facts`, `fields` |
| `continue` | `op`, `request_id`, `facts`, `fields` |
| `merge-facts` | `op`, `request_id`, `facts` |
| `finish` | `op`, `request_id` |

### 7.1 Create

```json
{
  "op": "create",
  "request_id": "L",
  "facts": {
    "generation_id": 0
  },
  "fields": {
    "body": {
      "prompt": "Hello"
    },
    "metadata": {
      "tenant": "acme"
    }
  }
}
```

Rules:

- `request_id` is non-empty;
- the event key set is exact;
- `facts`, `fields`, `fields.body`, and `fields.metadata` are objects;
- `generation_id` is exactly zero;
- `facts.logical_request_id`, if present, matches `request_id`; and
- the request must not already exist.

Creation installs host identity facts, the supplied complete fields object,
additional supplied facts, and empty request scratch.

### 7.2 Continue

```json
{
  "op": "continue",
  "request_id": "L",
  "facts": {
    "generation_id": 1
  },
  "fields": {
    "body": {
      "prompt": "Hello\nTool result: ..."
    },
    "metadata": {
      "step": 2
    }
  }
}
```

Rules:

- the event shape is the same as `create`;
- the request must exist;
- `generation_id` must be exactly the current generation plus one; and
- host identity facts cannot change.

Continuation semantics are:

1. preserve shared state;
2. preserve request scratch;
3. preserve durable request facts;
4. increment `generation_id`;
5. clear generation-local `current_target`;
6. replace `fields.body`;
7. shallow-merge the supplied `fields.metadata` into existing metadata;
8. preserve existing extra field keys omitted by the continuation;
9. overwrite extra field keys supplied by the continuation; and
10. merge additional supplied facts.

### 7.3 Merge facts

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

The request must exist. `logical_request_id` cannot change.
`generation_id`, if present, must equal the current generation.

`previous_target` is an enacted placement fact. A route preference or admit
decision does not update it. The engine reports it only after placement is
actually enacted.

### 7.4 Finish

```json
{
  "op": "finish",
  "request_id": "L"
}
```

Finish rules:

- it is valid only on the `feedback` hook;
- each request can appear in at most one finish event per invocation;
- the request must be referenced by a feedback record; and
- on a new delivery, the request must exist at preflight time.

Finish is not applied before the policy. The feedback policy receives the
request state first.

### 7.5 Preflight and application order

PLEX parses all lifecycle event shapes and simulates their generation
transitions before applying any event. Once preflight succeeds:

- `create`, `continue`, and `merge-facts` apply in event order;
- those updates are authoritative and remain if policy execution later
  becomes unavailable or falls back;
- `finish` is deferred until feedback processing; and
- a duplicate successful feedback delivery skips every request event.

The in-memory backend currently applies pre-hook events as sequential backend
operations rather than one batch transaction. Logical validation is atomic,
but a backend failure after an earlier event has been written can leave a
partial authoritative event batch. A distributed backend can add a
backend-specific batch API in a later contract revision.

## 8. Hook Contracts

### 8.1 Common conventions

Every hook context is a JSON object. Persistent state is never embedded in
context; top-level `shared` and `requests` keys in policy context are rejected.

Candidate and event `facts` objects are engine-defined. PLEX validates the
stable structural fields needed by its semantics and leaves workload-specific
facts open. Hook context is also open beyond the required structural fields.

Policy results must be top-level JSON objects. The host validates the
operation-specific fields before committing state. Extra result keys are
tolerated but omitted from the normalized decision. Feedback requires an
object result but defines no required result fields.

### 8.2 Working-set derivation

| Hook | Requests loaded |
|---|---|
| `route` | `request_id` |
| `admit` | `request_id` |
| `schedule` | Every unique `runnable[].request_id` |
| `evict` | Every unique non-null `resident[].request_id` |
| `feedback` | Every unique `records[].request_id` |

Repeated references share one mutable request object. Unreferenced live
requests are invisible to the policy.

### 8.3 Route

Context:

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
    },
    {
      "id": "node-b",
      "facts": {
        "queue_depth": 2,
        "cached_tokens": 0,
        "has_request_kv": false
      }
    }
  ],
  "context": {
    "model": "example-model",
    "capabilities": {}
  }
}
```

Policy result:

```json
{
  "scores": [100.0, 10.0]
}
```

Rules:

- there is exactly one finite numeric score per candidate;
- higher scores rank first; and
- ties preserve candidate order.

Normalized decision:

```json
{
  "order": [0, 1]
}
```

PLEX does not route by itself. It returns a preference order over the exact
feasible candidates supplied by the engine.

### 8.4 Admit

Context:

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

Policy result and normalized decision:

```json
{
  "decision": "accept"
}
```

The only valid decisions are:

- `accept`;
- `defer`; and
- `reject`.

The lower-level `LifecycleHost::route_and_admit` helper can compose route and
admit: it visits candidates in route order, retries `defer` up to the configured
limit, moves to the next candidate after `reject`, and returns an accepted,
deferred, or rejected placement. The primary engine JSON API deliberately does
not hide this sequencing; adapters can invoke the two hooks as their engine
requires.

### 8.5 Schedule

Context:

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
    },
    {
      "request_id": "M",
      "facts": {
        "waiting_ms": 4
      },
      "max_token_budget": 4
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

Policy result:

```json
{
  "decisions": [
    {
      "score": 20.0,
      "token_budget": 8
    },
    {
      "score": 10.0,
      "token_budget": 4
    }
  ]
}
```

Rules:

- `decisions` is dense and aligned with `runnable`;
- every decision has a finite numeric `score`;
- higher scores are selected first with stable ties;
- an omitted or `null` token budget uses the candidate maximum;
- a non-null token budget requires
  `context.capabilities.token_budget == true`;
- a budget cannot exceed the candidate maximum or host maximum;
- no more than `max_selected` entries are selected;
- total selected tokens cannot exceed `max_total_tokens`; and
- an entry whose resulting budget is zero is skipped.

When a valid candidate budget is larger than the aggregate capacity remaining,
the normalized budget is truncated to that remaining capacity.

Normalized decision:

```json
{
  "selected": [
    {
      "candidate_index": 0,
      "token_budget": 8
    }
  ]
}
```

Schedule is intentionally set-dependent. One invocation receives the complete
feasible runnable set for one scheduling opportunity. PLEX does not define a
candidate-local schedule callback.

### 8.6 Evict

Context:

```json
{
  "cause": "allocation-deficit",
  "resident": [
    {
      "id": "kv-0",
      "request_id": "L",
      "size_bytes": 4096,
      "facts": {
        "reload_cost": 200.0
      }
    },
    {
      "id": "kv-1",
      "request_id": null,
      "size_bytes": 2048,
      "facts": {
        "reload_cost": 5.0
      }
    }
  ],
  "bytes_needed": 2048,
  "context": {
    "capabilities": {}
  }
}
```

Policy result:

```json
{
  "scores": [200.0, 5.0]
}
```

Rules:

- there is exactly one finite retention score per resident unit;
- lower scores are reclaimed first;
- ties preserve resident order; and
- selection stops after cumulative `size_bytes` reaches `bytes_needed`.

Normalized decision:

```json
{
  "selected": [
    {
      "candidate_index": 1,
      "size_bytes": 2048
    }
  ]
}
```

A resident unit can be unattributed (`request_id: null`). It participates in
eviction scoring but does not add request state to the working set.

### 8.7 Feedback

Context:

```json
{
  "delivery_id": "delivery-42",
  "records": [
    {
      "event": "progress",
      "request_id": "L",
      "facts": {
        "committed_tokens": 8
      }
    },
    {
      "event": "action-succeeded",
      "request_id": "L",
      "facts": {
        "method": "pie.kv.prefetch@1"
      }
    }
  ],
  "context": {
    "capabilities": {}
  }
}
```

Feedback event names are open strings. Policies choose which events they
understand. A feedback policy still returns a JSON object, but the engine-facing
normalized decision is always:

```json
{}
```

The raw policy result is retained in the feedback ledger and is used to replay
the semantic result of duplicate successful deliveries.

## 9. Persistent State Model

### 9.1 State shape

```text
State
|-- shared
`-- requests
    `-- <logical-request-id>
        |-- facts
        |-- fields
        `-- scratch
```

WIT-visible JSON:

```json
{
  "shared": {
    "tenant_service": {
      "acme": 120
    }
  },
  "requests": {
    "L": {
      "facts": {
        "logical_request_id": "L",
        "generation_id": 1,
        "previous_target": "node-a",
        "attained_service": 128
      },
      "fields": {
        "body": {
          "prompt": "Hello"
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
  }
}
```

Every state namespace is a JSON object.

### 9.2 Shared state

`State.shared` is one mutable policy-owned object per backend. There are no
shared facts or shared fields. Host observations belong in transient context
or a query. Engine effects belong in staged actions.

Shared state lets independently attached operation owners coordinate through a
common policy memory, for example:

- tenant-level attained service;
- learned decode statistics;
- workflow counters;
- cache-policy summaries; or
- cross-hook control state.

### 9.3 Request facts

Facts are host-owned. Required identity facts are:

```json
{
  "logical_request_id": "L",
  "generation_id": 1
}
```

The engine can merge additional enacted observations. Policies can read facts
but cannot update them through the guest SDK or WIT state-update envelope.

### 9.4 Request fields

Fields are canonical request data and can contain engine-defined keys.
Engine `create` and `continue` events require object-valued `body` and
`metadata`. A policy state update is only required to keep the complete
`fields` namespace itself as an object; v0.5 does not revalidate nested
`body`/`metadata` after a policy rewrite. Policies should preserve those two
object-valued conventions so a later continuation remains valid.

Policies can rewrite complete fields. A successful engine outcome returns the
complete new fields object for each request whose fields changed relative to
the exact snapshot supplied to Wasm. The engine is responsible for applying
that object.

### 9.5 Request scratch

Scratch is private policy memory attached to the logical request. It can track
values such as:

- route and admission counts;
- predicted token demand;
- tool-call count;
- workflow-local fairness state; or
- action outcomes learned from feedback.

Scratch is persisted but never returned in the engine outcome.

### 9.6 Visibility and membership invariants

The policy cannot:

- read requests outside the working set;
- add or remove request IDs;
- mutate request identity facts;
- return updates for unknown requests; or
- change a namespace from an object to another JSON type.

The Rust SDK checks these invariants before producing WIT output, and the host
validates the returned update again.

## 10. Explicit State Transactions

### 10.1 WIT input

`invocation.state-json` is the complete visible snapshot:

```json
{
  "shared": {},
  "requests": {
    "L": {
      "facts": {
        "logical_request_id": "L",
        "generation_id": 0
      },
      "fields": {
        "body": {},
        "metadata": {}
      },
      "scratch": {}
    }
  }
}
```

Backend revisions do not cross WIT.

### 10.2 WIT output

`policy-output.state-update-json` is a sparse update envelope:

```json
{
  "shared": {
    "route_calls": 1
  },
  "requests": {
    "L": {
      "fields": {
        "body": {},
        "metadata": {
          "last_hook": "route"
        }
      },
      "scratch": {
        "route_count": 1
      }
    }
  }
}
```

Rules:

- `{}` means no state change;
- unchanged top-level scopes are omitted;
- `shared`, when present, is a complete replacement object;
- a request update contains exactly `fields` and `scratch`;
- request fields and scratch are complete replacement objects;
- facts cannot appear;
- unknown top-level namespaces are rejected; and
- unknown request IDs are rejected.

JSON Patch is not used in v0.5.

### 10.3 Backend snapshot

Internally, `StateSnapshot` contains:

- the shared value and revision;
- each working-set request value; and
- each working-set request revision.

The backend revision data remains host-private.

### 10.4 Compare-and-swap commit

The in-memory backend validates and commits under one lock. It checks:

1. snapshot structure;
2. update structure and membership;
3. terminal request membership;
4. duplicate feedback delivery;
5. shared revision equality; and
6. every working-set request revision.

PLEX checks shared and all working-set revisions even when the policy did not
modify each object. The transaction therefore represents one coherent snapshot
of the shared object and complete request working set.

On success, one atomic backend transition can:

- replace shared policy state;
- replace changed request fields and scratch;
- record a feedback result; and
- remove terminal requests.

The in-memory implementation clones the current state, applies all changes to
the clone, and publishes it only after every check succeeds.

### 10.5 State conflict

If any revision changed after the snapshot was loaded:

- no policy state update is committed;
- no feedback delivery is recorded by that attempted commit;
- no staged action is returned;
- the engine outcome is fallback with `kind: "state-conflict"` after any
  required terminal cleanup succeeds; and
- authoritative pre-hook lifecycle events already applied are not rolled back.

For terminal feedback, fallback cleanup still removes the finished request
after the failed policy transaction.

## 11. End-to-End Invocation Semantics

For a non-duplicate invocation, `PlexRuntime` executes:

```text
1. parse strict engine envelope
2. parse request events
3. prepare defaults and action capabilities
4. validate hook context and derive request working set
5. validate finish-to-feedback membership
6. check the successful-feedback ledger
7. preflight all request lifecycle transitions
8. apply authoritative non-terminal events, rechecking feedback races
9. recheck feedback and load state values/revisions
10. snapshot the active operation owner
11. create a fresh bounded Wasmtime store and component instance from the
    engine's preallocated Wasmtime pool
12. invoke the explicit WIT export
13. parse and validate result/state update
14. collect staged actions
15. compare-and-swap commit
16. normalize the operation decision
17. compute changed request fields from the exact Wasm input snapshot
18. return decision, changed fields, and actions
```

When step 6 finds a duplicate, steps 7-8 are skipped and the lifecycle recheck
in step 9 returns the cached result without loading state. Steps 10-15 are
therefore only part of the new-delivery path.

The ordering creates three important boundaries.

### 11.1 Authoritative lifecycle boundary

Engine lifecycle observations happen before policy execution and survive
policy fallback. A policy cannot make an observed continuation or fact update
disappear by failing.

### 11.2 Policy transaction boundary

Policy-owned shared state, fields, scratch, feedback ledger state, and terminal
removal commit together when the backend supports the v0.5 transaction.

### 11.3 Engine side-effect boundary

PLEX returns staged actions only after successful validation and commit. The
engine enacts them afterward. Their real outcome is reported in a later
feedback delivery.

## 12. Feedback Idempotency and Terminal Cleanup

### 12.1 Successful feedback

For a new successful feedback delivery, PLEX atomically commits:

- the policy state update;
- the raw policy result under `delivery_id`; and
- all requested terminal removals.

### 12.2 Duplicate successful feedback

For a previously committed `delivery_id`, PLEX:

- skips request lifecycle events;
- skips state loading when possible;
- skips Wasm;
- skips policy state updates;
- returns no actions;
- reuses the recorded policy result; and
- returns a normal successful engine outcome.

This remains true for terminal retries after the original request has already
been removed.

Duplicate checks occur at three levels to close races:

1. before engine lifecycle-event preflight;
2. before lifecycle state loading; and
3. under the backend commit lock.

The commit path checks duplicate delivery before revision conflicts. If two
runtimes race on the same delivery, the loser converges to the committed
duplicate result rather than reporting a state conflict.

### 12.3 Unavailable and fallback feedback

If feedback is unavailable or falls back, requested terminal requests are
still removed by `PlexRuntime` so completed serving work does not leak
persistent state. This fallback cleanup is a sequence of idempotent removals,
not part of the failed policy transaction. `NotFound` is tolerated; another
backend failure is surfaced as `PlexError::Backend` and a multi-request cleanup
can then be partial.

Unavailable or failed feedback is not inserted into the successful-delivery
ledger. Only a successfully committed delivery receives retry deduplication.

### 12.4 Bounded ledger

The in-memory ledger has a configured maximum number of successful deliveries.
It does not currently expire or evict entries. A new delivery after the ledger
reaches its limit is a backend failure; a duplicate existing delivery still
resolves as a duplicate.

## 13. Engine Outcomes and Failure Model

### 13.1 Success

```json
{
  "status": "success",
  "decision": {
    "order": [0]
  },
  "request_fields": {
    "L": {
      "body": {
        "prompt": "rewritten"
      },
      "metadata": {}
    }
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

Success always contains:

- `decision`: one normalized, engine-actionable decision;
- `request_fields`: complete fields for requests whose fields changed;
- `actions`: staged descriptors in policy call order; and
- `status: "success"`.

Shared state and request scratch never leak into the engine outcome.
Invocation-local action IDs begin at zero.

### 13.2 Normalized decision summary

| Hook | Engine decision |
|---|---|
| `route` | `{"order": [candidate indexes]}` |
| `admit` | `{"decision": "accept" | "defer" | "reject"}` |
| `schedule` | `{"selected": [{"candidate_index": N, "token_budget": N}]}` |
| `evict` | `{"selected": [{"candidate_index": N, "size_bytes": N}]}` |
| `feedback` | `{}` |

Stable ordering, capacity fill, and byte-deficit fill happen exactly once in
the semantic core. Adapters do not reimplement these algorithms.

### 13.3 Unavailable

```json
{
  "status": "unavailable"
}
```

Unavailable means no attached package owns the requested operation. The engine
uses its native policy. Terminal feedback cleanup still occurs.

### 13.4 Fallback

```json
{
  "status": "fallback",
  "failure": {
    "kind": "invalid-output",
    "message": "..."
  }
}
```

Fallback kinds are:

```text
invalid-input
instantiation
policy-fallback
trap
deadline-exceeded
host-saturated
query
action-validation
state-conflict
backend-failure
invalid-output
```

The engine API exposes all ordinary policy/runtime failures as fallback except
`backend-failure`, which is promoted to `PlexError::Backend`.

A fallback outcome contains no request fields or actions. Policy-owned updates
are not committed. Authoritative lifecycle events applied before Wasm remain.

### 13.5 API errors

The Rust error hierarchy is:

```text
PlexError
|-- InvalidEvent
|-- Backend
|-- PolicyPackage
`-- Runtime
```

Examples:

- malformed JSON or an invalid event shape -> `InvalidEvent`;
- missing referenced state or backend I/O failure -> `Backend`;
- invalid package bytes or manifest -> `PolicyPackage`;
- runtime construction or impossible post-validation normalization failure ->
  `Runtime`.

API errors are not native-policy fallback instructions. The engine integration
must surface or handle them explicitly.

## 14. Host Queries

The Rust host interface is:

```rust
pub trait QueryHandler: Send + Sync {
    fn query(
        &self,
        method: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, QueryError>;
}
```

Query rules:

- method names use `name@numeric-version`;
- arguments must be a JSON object;
- calls are synchronous;
- calls and bytes are metered;
- the configured handler determines actual support;
- the result can be any JSON value; and
- handler errors are returned to the guest as policy-visible errors.

A policy can handle a failed query and continue. If it propagates the host
error, PLEX classifies the fallback as `query`.

Queries are contractually read-only. Candidate-scale observations such as queue
depth, waiting time, cache locality, and capacity should normally be supplied
in bulk hook context instead of queried once per candidate.

### 14.1 Rust SDK query helpers

| Helper | Method |
|---|---|
| `host.kv_lookup(request_id, target)` | `pie.kv.lookup@1` |
| `host.cluster_capacity(model)` | `pie.cluster.capacity@1` |
| `host.model_config()` | `pie.model.config@1` |
| `host.now_ms()` | `pie.clock.now@1` |

`host.query_raw(method, args)` is the extension escape hatch.

## 15. Staged Actions

The generic action call:

```text
action(method, args-json) -> action-id
```

performs:

1. host-call count and byte accounting;
2. numeric-versioned method validation;
3. runtime support-set validation;
4. object-valued JSON argument parsing;
5. invocation-local monotonic ID assignment; and
6. descriptor staging.

### 15.1 Rust SDK action helpers

| Helper | Method |
|---|---|
| `host.prefetch_kv(request_id, target)` | `pie.kv.prefetch@1` |
| `host.preempt(request_id)` | `pie.schedule.preempt@1` |
| `host.replicate(request_id, targets)` | `pie.route.replicate@1` |
| `host.set_retention(request_id, ttl_ms)` | `pie.retention.set@1` |
| `host.arm_timer(request_id, delay_ms)` | `pie.timer.arm@1` |

`host.action_raw(method, args)` stages extension methods.

### 15.2 Action guarantees

- PLEX never executes an action.
- Action order matches guest call order.
- IDs are local to one invocation and begin at zero.
- Unsupported actions are policy-visible errors.
- Invalid results or state updates discard every staged action.
- State conflicts discard every staged action.
- Duplicate feedback does not replay actions.
- Only a successful state commit exposes actions to the engine.

Policies should not optimistically record that an action succeeded. They should
update outcome-dependent state from later feedback.

## 16. WIT v0.5 Contract

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

### 16.1 Surface restrictions

An accepted component imports exactly:

```text
pie:plex/host@0.5.0
```

and exports exactly:

```text
pie:plex/policy@0.5.0
```

WASI and every other external import are rejected.

### 16.2 Why JSON is inside typed WIT records

WIT stabilizes:

- operation names;
- transaction boundaries;
- host capabilities;
- success/error transport; and
- component linking.

JSON keeps engine- and workload-specific facts extensible without rebuilding a
large generated type graph. The host still validates the structural semantics
that affect safety and correctness.

### 16.3 No ambient state interface

There is deliberately no state import. A guest cannot:

- load a different request midway through execution;
- scan backend keys;
- depend on an exactly-once host call sequence; or
- observe backend revisions.

The explicit input/output pair is the transaction.

## 17. Rust Guest SDK

The `plex` crate presents:

```rust
pub type Document = serde_json::Value;
pub type Result<T> = std::result::Result<T, String>;
pub type ActionId = u64;

pub struct State {
    pub shared: Document,
    // Request membership is private.
}

pub struct Request {
    // Facts are private and exposed by facts().
    pub fields: Document,
    pub scratch: Document,
}

pub struct Host {
    // Generic imports plus direct helper methods.
}
```

Request access is explicit:

```rust
state.request("L")?;
state.request_mut("L")?;
state.request_ids();
```

Facts are read-only:

```rust
let generation = state.request("L")?.facts()["generation_id"]
    .as_u64()
    .unwrap_or(0);
```

### 17.1 Policy trait

```rust
pub trait Policy {
    fn route(
        ctx: &Document,
        state: &mut State,
        host: &Host,
    ) -> Result<Document>;

    fn admit(
        ctx: &Document,
        state: &mut State,
        host: &Host,
    ) -> Result<Document>;

    fn schedule(
        ctx: &Document,
        state: &mut State,
        host: &Host,
    ) -> Result<Document>;

    fn evict(
        ctx: &Document,
        state: &mut State,
        host: &Host,
    ) -> Result<Document>;

    fn feedback(
        ctx: &Document,
        state: &mut State,
        host: &Host,
    ) -> Result<Document>;
}
```

Default methods return a policy fallback. The SDK exports all five WIT
functions, while the package manifest declares which operations the package
owns.

### 17.2 Minimal policy

```rust
use plex::serde_json::json;
use plex::{Document, Host, Policy, State};

struct LeastLoaded;

impl Policy for LeastLoaded {
    fn route(
        ctx: &Document,
        state: &mut State,
        _host: &Host,
    ) -> Result<Document, String> {
        let request_id = ctx["request_id"]
            .as_str()
            .ok_or("request_id must be a string")?;

        let request = state.request_mut(request_id)?;
        request.scratch["route_count"] =
            json!(request.scratch["route_count"].as_u64().unwrap_or(0) + 1);

        let scores = ctx["candidates"]
            .as_array()
            .ok_or("candidates must be an array")?
            .iter()
            .map(|candidate| {
                -candidate["facts"]["queue_depth"]
                    .as_f64()
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();

        Ok(json!({"scores": scores}))
    }
}

plex::export_policy!(LeastLoaded);
```

### 17.3 Generated glue behavior

For each WIT export, SDK glue:

1. parses context JSON;
2. validates exact state namespaces;
3. validates request identity facts;
4. clones the initial state;
5. invokes the author method;
6. verifies shared/fields/scratch remain objects;
7. verifies facts did not change;
8. verifies request membership did not change;
9. computes a sparse state update;
10. serializes result and update JSON; and
11. returns explicit policy errors.

The host independently validates the output, so raw-WIT guests cannot bypass
the semantic checks performed by the Rust SDK.

## 18. Policy Packaging and Attachment

### 18.1 Manifest

A package manifest contains:

```json
{
  "contract": {
    "major": 0,
    "minor": 5
  },
  "package_name": "coordinated-policy",
  "package_version": "0.5.0",
  "operations": [
    "route",
    "admit",
    "schedule",
    "evict",
    "feedback"
  ],
  "limits": {
    "memory_bytes": 4194304,
    "deadline_ms": 100,
    "input_bytes": 1048576,
    "output_bytes": 1048576
  }
}
```

Manifest rules:

- unknown fields are rejected;
- contract must be exactly v0.5;
- package name is 1-64 ASCII alphanumeric, `-`, or `_`, beginning
  alphanumeric;
- package version is at most 32 bytes and exactly three numeric components;
- at least one operation is declared;
- every limit is non-zero; and
- every requested limit must fit within the host maximum.

### 18.2 `.plexpkg` format

The package format is:

```text
8 bytes   magic: "PLEXPKG\0"
2 bytes   little-endian format version: 5
2 bytes   flags: 0
4 bytes   little-endian manifest length
8 bytes   little-endian component length
32 bytes  BLAKE3 digest
N bytes   manifest JSON
M bytes   component bytes
```

The digest covers length-delimited manifest and component bytes. It detects
corruption and mismatched contents; it is not a signature or publisher
authentication mechanism.

The decoder checks package, manifest, and component length limits before
compilation.

### 18.3 Attachment verification

Preparing a package:

1. decodes and verifies `.plexpkg`;
2. validates the manifest;
3. checks requested limits against host limits;
4. compiles the component;
5. verifies the exact WIT import/export surface;
6. links only the PLEX host interface; and
7. probes instantiation within declared limits.

Only a prepared package can be published.

### 18.4 Operation ownership

Each operation has at most one attached owner. A package can own any non-empty
subset of the five operations. Attaching a package whose operations overlap
another package fails.

This permits:

- one coordinated package owning all five hooks;
- separate packages owning route, schedule, and eviction; or
- incremental experiments that replace one operation family.

All owners sharing a backend also share `State.shared`.

### 18.5 Atomic replacement

The registry maintains a monotonically increasing attachment generation.
Replacement:

- prepares the new package before publication;
- targets the same package name;
- waits for active registry snapshots to drain;
- verifies operation ownership against other packages;
- swaps the complete attachment set atomically; and
- increments the generation.

An invocation observes one stable attachment snapshot. It never switches
policy packages midway through execution.

Detach is supported by operation or package.

## 19. Concurrency Model

### 19.1 One `PlexRuntime`

`PlexRuntime::invoke` is serialized by an invocation mutex. This protects
lifecycle event application and policy commit ordering inside one runtime.
Rust callers must not recursively invoke the same runtime from its query
handler; the mutex is non-reentrant. The Python binding detects and rejects
that same-thread case before entering Rust.

`LifecycleHost` also serializes its lower-level stateful methods. The primary
engine integration should use `PlexRuntime`.

### 19.2 Multiple runtimes

Multiple runtimes can share one `PolicyStateBackend`. They can execute
concurrently and coordinate through:

- backend revisions;
- compare-and-swap commit;
- the feedback ledger; and
- atomic backend operations.

Conflicting non-duplicate transactions produce a state-conflict fallback.
Concurrent attempts for the same successful feedback delivery converge to the
cached duplicate result.

### 19.3 Global host capacity

`PolicyEngine` bounds concurrent Wasmtime invocations across runtimes that
share the engine. If no permit is available, the invocation falls back with
`host-saturated`.

### 19.4 Python concurrency

The Python binding:

- releases the GIL while Rust/Wasmtime executes;
- reacquires it only to call the Python query callback;
- rejects same-thread recursive invocation of the same runtime; and
- allows different Python threads to call the same runtime, where calls
  serialize through the Rust runtime mutex.

Independent Python runtimes can execute concurrently.

## 20. Python Host Binding

The standalone distribution is:

```text
distribution: pie-plex
package:      pie_plex
native module: pie_plex._native
Python:       >= 3.10
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

The Python package also provides an asynchronous worker seam:

```python
from pie_plex import AsyncRuntime

runtime = AsyncRuntime("policy.plexpkg", queue_capacity=256)
runtime.try_submit("schedule", epoch, event)
latest = runtime.latest("schedule", after_epoch)
```

`try_submit` never waits for policy execution. A Rust-owned worker serially
executes PLEX transactions and atomically publishes the latest outcome for each
hook channel. Submission uses a bounded queue and returns `False` rather than
blocking when the worker cannot accept more work.

The native seam is:

```text
NativeRuntime.invoke_json(event_json: str) -> str
```

The Python facade implements dictionary input/output with
`json.dumps`/`json.loads`. There is no independent Python conversion of the
PLEX domain model, so Rust and Python consume the same JSON contract.

### 20.1 Query callback

The optional callback receives:

```python
def engine_query(method: str, args: dict) -> object:
    ...
```

Python exceptions become `QueryError::Handler` and are visible to policy code.
Callback results must be JSON-serializable. A propagated callback failure
produces a query fallback outcome.

### 20.2 Python exceptions

```text
PlexError
|-- InvalidEvent
|-- BackendError
|-- PolicyPackageError
`-- QueryCallbackError
```

Policy unavailable and fallback remain ordinary outcome dictionaries, not
Python exceptions.

`QueryCallbackError` is used for detected same-thread recursive invocation of
the same runtime. An ordinary exception raised by a query callback is converted
to a policy-visible query error, not raised directly from `Runtime.invoke`.

## 21. Engine Adapter Pattern

An adapter performs five tasks:

1. snapshot engine state into canonical hook context;
2. report authoritative request lifecycle events;
3. invoke `PlexRuntime`;
4. apply a successful normalized decision and changed request fields; and
5. enact returned actions in order.

For `unavailable` or `fallback`, the adapter selects the engine's native
policy. API errors are handled separately.

The Python package includes equivalent vLLM and SGLang scheduler templates:

```python
class PlexSchedulerAdapter:
    def __init__(self, scheduler, runtime):
        self.scheduler = scheduler
        self.runtime = runtime

    def schedule(self):
        outcome = self.runtime.invoke(schedule_event(self.scheduler))
        if outcome["status"] != "success":
            return self.scheduler.native_schedule()
        return schedule_outcome(self.scheduler, outcome)
```

The template expects an engine-specific scheduler facade that provides:

```text
plex_runnable()
plex_capacity()
plex_context()
plex_request_events()
apply_plex_schedule(decision, request_fields)
apply_plex_action(action)
native_schedule()
plex_query(method, args)
```

These adapters do not import vLLM or SGLang internals and do not claim live,
version-pinned compatibility. They are conformance templates that isolate
version-specific snapshot and apply logic from the PLEX contract.

The live version-pinned integrations are implemented in the `ingim/vllm` and
`ingim/sglang` forks. Both load `pie-plex` only when a policy path is configured
and run Wasm/state processing on a Rust-owned asynchronous worker. Engine hot
paths consume only immutable cached plans and immediately use native behavior
when a plan is missing, stale, unavailable, or failed. Admission is optimistic
and handled outside the engine adapter; feedback is coalesced at completion,
abort, and preemption boundaries. vLLM caches standing runnable and
request-retention plans. SGLang
caches prefill-admission and decode-retraction plans because an already-resident
decode batch executes as one native unit.

Snapshot publication occurs when request membership or residency changes, not
on every decode token. Consequently, policy facts and decisions can be stale
for a bounded lifecycle interval. This is an intentional production tradeoff:
fresh same-step set-dependent decisions require blocking and are available only
through the synchronous `Runtime` API used by tests and offline experiments.

## 22. Runtime Isolation and Limits

### 22.1 Default host limits

| Limit | Default |
|---|---:|
| Package bytes | 5 MiB |
| Manifest bytes | 64 KiB |
| Component bytes | 4 MiB |
| Wasm memory | 16 MiB |
| Deadline | 100 ms |
| Context + state input | 4 MiB |
| Result + update output | 4 MiB |
| Host calls | 64 |
| Aggregate host-call bytes | 4 MiB |
| Concurrent Wasm invocations | 128 |
| Successful feedback ledger entries | 4,096 |

Package manifests request per-policy memory, deadline, input, and output
limits at or below the host maxima.

### 22.2 Wasmtime protections

Each invocation uses:

- a fresh store and component instance returned to a Wasmtime allocation pool
  after the call;
- epoch interruption;
- a memory limit;
- bounded table, instance, and memory counts;
- disabled Wasm threads;
- no WASI;
- exact import/export verification; and
- a global invocation permit.

The pool is fixed when `PolicyEngine` is created. It contains one top-level
component slot, up to four core-instance and table slots, and one linear-memory
slot per `max_concurrent_invocations`. Its memory reservation and growth limit
match the host `max_memory_bytes` limit. Pooling reuses allocation slots only:
guest memory and globals are reset by dropping each instance, so hidden state
cannot cross invocations. Reusing a live instance would be faster but would
weaken that isolation and deterministic-replay property; it is measured only
as an explicitly labeled profiler upper bound. The global invocation permit
outlives the Store, ensuring a pooled slot is fully returned before another
thread can acquire the corresponding concurrency slot.

Trap classification distinguishes deadline expiration from ordinary traps.

### 22.3 Host-call protections

Query and action calls share:

- one call-count budget; and
- one aggregate request/response byte budget.

Exceeding either budget is a fatal invocation failure even if guest code tries
to ignore the returned host error.

The configured `QueryHandler` is trusted host code. v0.5 does not impose an
independent timeout on time spent inside that callback, so an engine query
implementation must bound its own work.

### 22.4 Deterministic replay mode

Deterministic replay disables the real-time epoch ticker. Structural, memory,
and host-call validation remain active. The epoch deadline is effectively
disabled in this mode, removing wall-clock timing as a replay source of
nondeterminism.

## 23. Replay and Validation

### 23.1 Replay model

`ReplayRunner` supports deterministic traces containing:

- package attach, replace, and detach;
- request create and continue;
- shared-state inspection and replacement;
- fact merges and enacted placement recording;
- individual hook invocation;
- combined route/admit;
- terminal feedback; and
- request-state inspection.

Replay reports include attachment generations, state reads, invocation
results, normalized selections, placement outcomes, unavailable operations,
and fallback kinds. Verification reports the first divergent command.

### 23.2 Fixture coverage

The fixture suite covers:

- strict engine event validation;
- all five explicit WIT exports;
- Rust/JSON contract parity;
- request lifecycle preflight;
- continuations and placement facts;
- exact working sets;
- dense score validation;
- stable route, schedule, and eviction decisions;
- token-budget capability enforcement;
- state update validation;
- state conflict rollback;
- changed-field reporting from the exact Wasm snapshot;
- query and action helpers;
- raw extension methods;
- staged-action rollback;
- fallback and terminal cleanup;
- feedback deduplication;
- terminal feedback retry after request removal;
- package corruption rejection;
- deterministic replay;
- mock PIE/vLLM/SGLang conformance; and
- research policies modeled after Agentix, Continuum, KVFlow, Preble, and
  Helium.

### 23.3 Validation commands

```bash
cargo test --locked -p pie-plex -p pie-policy --all-targets
cargo test --manifest-path sdk/rust/plex/Cargo.toml --locked
./scripts/build-plex-policies.sh
cargo run --locked -p pie-policy --example check_fixtures -- \
  tests/policies/target/components
./scripts/check-plex-layering.sh
```

The Python package is built with Maturin and tested with Pytest.

## 24. Versioning and Extension Model

PLEX has four independent version axes:

| Surface | Current version | Purpose |
|---|---|---|
| Engine JSON API | `pie.plex.engine@1` | Engine/runtime envelope |
| WIT contract | `pie:plex@0.5.0` | Guest/host ABI |
| Package format | `6` | Binary `.plexpkg` layout |
| Helper method | e.g. `pie.kv.prefetch@1` | Query/action semantics |

This separation is intentional:

- adding an SDK helper for a new method does not change WIT;
- changing package encoding does not require changing engine events;
- engine adapters can evolve independently of guest SDK ergonomics; and
- a breaking transaction or hook change can advance WIT explicitly.

Engine envelope and request-event key sets are strict to catch adapter mistakes.
Hook-specific facts, fields, metadata, capabilities, feedback event names, and
versioned helper methods are extension points.

## 25. Implemented Scope and Known Limitations

### 25.1 Implemented

The current implementation provides:

- strict Rust and Python engine APIs;
- all five lifecycle hooks;
- explicit WIT state snapshots and updates;
- shared/request state with revisions;
- successful-feedback idempotency;
- terminal cleanup;
- query/action host imports;
- method-oriented Rust SDK helpers;
- bounded Wasmtime execution;
- package integrity and atomic publication;
- deterministic replay;
- Python binding and concurrency handling;
- mock engine adapter conformance; and
- live version-pinned vLLM and SGLang integrations.

### 25.2 Known limitations

The current milestone does not provide:

- an etcd or other distributed `PolicyStateBackend`;
- batch atomicity for pre-hook request events on backend failure;
- feedback-ledger expiry or compaction;
- cryptographic package signatures;
- typed schemas for application-specific JSON;
- JSON Patch or partial field updates;
- arbitrary live-request enumeration;
- automatic retries for state conflicts;
- an engine action callback inside PLEX;
- compensation after an engine action fails;
- an independent timeout around host query callbacks;
- shared physical units with multiple mutable request owners;
- automatic route invalidation after field mutation;
- Python guest policy authoring;
- a live cluster-router `route` integration;
- cross-process policy state without a distributed backend;
- PLEX actions in the current vLLM/SGLang adapters;
- SGLang PLEX support in disaggregated serving modes; or
- exact same-step policy decisions in the asynchronous engine adapters.

The architecture keeps these mechanisms outside the v0.5 stable waist so they
can evolve without expanding the policy ABI prematurely.

## 26. Implementation Map

| Area | Path |
|---|---|
| JSON semantic types and decision normalization | `interface/plex/src/` |
| WIT contract | `interface/plex/wit/` |
| Engine API | `runtime/policy/src/engine_api.rs` |
| Lifecycle transaction coordinator | `runtime/policy/src/lifecycle.rs` |
| State backend and revisions | `runtime/policy/src/state_store.rs` |
| Wasmtime invocation and component verification | `runtime/policy/src/package.rs` |
| Query/action host implementation | `runtime/policy/src/context.rs` |
| Attachment registry | `runtime/policy/src/registry.rs` |
| Package encoding | `runtime/policy/src/package_format.rs` |
| Deterministic replay | `runtime/policy/src/replay.rs` |
| Rust guest SDK | `sdk/rust/plex/` |
| Python host binding | `sdk/python-plex/` |
| Live vLLM adapter | `ingim/vllm:vllm/v1/core/sched/async_plex.py` |
| Live SGLang adapter | `ingim/sglang:python/sglang/srt/managers/async_plex.py` |
| Policy fixtures | `tests/policies/` |
| Fixture build pipeline | `scripts/build-plex-policies.sh` |

## 27. Design Summary

PLEX v0.5 reduces the serving-policy boundary to a small set of durable ideas:

```text
five lifecycle hooks
+ explicit JSON context
+ explicit persistent working set
+ immutable host facts
+ mutable policy fields and scratch
+ synchronous read-only queries
+ staged post-commit actions
+ normalized feasible decisions
+ feedback-driven state
+ bounded Wasm execution
+ revisioned atomic commit
```

The result is a programming model in which policy code can express coordinated
request-lifecycle behavior without becoming part of an engine fork, while the
engine retains authority over feasibility, resources, and physical execution.
