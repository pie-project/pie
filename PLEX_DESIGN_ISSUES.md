# PLEX v0.5 Design Record

## Current Decision

PLEX uses:

```text
engine API: pie.plex.engine@1
policy ABI: pie:plex@0.5.0
```

The typed v0.1 design, embedded-request v0.2 design, complete-input v0.3
responses, and ambient v0.4 state imports are not retained behind
compatibility adapters.

## Two Different Waists

Engines use one JSON operation:

```text
invoke(event) -> outcome
```

Wasm policies keep five explicit exports:

```text
route admit schedule evict feedback
```

Engines never implement or consume WIT. Hook names remain explicit in WIT
because they are the stable policy waist.

## Explicit State Transaction

Each hook receives a WIT record containing context JSON and state JSON. It
returns result JSON and state-update JSON.

There is no state import and no exactly-once load/stage protocol. Backend
revisions remain host-private.

The SDK still exposes mutable `State` and computes:

```text
{}                         no change
shared                     changed shared scratch
requests[id].fields        changed engine-visible fields
requests[id].scratch       changed policy scratch
```

Facts and request membership cannot be returned.

## Engine Outcomes

Successful outcomes expose:

```text
normalized decision
changed request fields
ordered staged actions
```

Shared and request scratch stay internal. Ranking/fill is normalized once in
the semantic core.

Unavailable and policy failure are outcomes. Invalid engine events, backend
failure, package failure, and runtime construction failure are API errors.

## Queries and Actions

Query is the only engine callback. It is synchronous, bounded, and read-only.
Candidate-scale facts remain in hook context.

Action is not a callback. PLEX validates a supported versioned method, assigns
an invocation-local ID, buffers the descriptor, commits state, and returns the
descriptor to the engine.

The engine enacts actions and reports outcomes through feedback. PLEX cannot
roll back a physical action and policy state must not assume success before
feedback.

## Request Events

Create, continue, and fact events are authoritative and happen before policy
state loading. Finish happens after feedback access. Event shapes and
generation transitions preflight before mutation.

Successful feedback can combine state update, deduplication, and finish in one
backend commit. Unavailable/fallback feedback still performs cleanup.

## State and Concurrency

Persistent state remains:

```text
shared
requests[id].facts
requests[id].fields
requests[id].scratch
```

The in-memory backend uses revisions and compare-and-swap. Multiple runtimes can
share it. State conflict returns fallback and no action.

Validated pre-hook request events are sequential in the in-memory backend.
Backend-failure atomic batching and etcd ownership/handoff are deferred rather
than hidden behind complex machinery.

## SDK Style

Normal Rust policy code uses methods such as:

```text
kv_lookup
cluster_capacity
prefetch_kv
preempt
replicate
set_retention
arm_timer
```

Raw query/action methods remain extension points. There are no one-use argument
structs or typed hook models.

## Python Boundary

`pie_plex.Runtime` is a standalone PyO3/Maturin package depending on the PLEX
runtime, not `pie-worker`.

The native seam is one JSON string method. The facade uses Python JSON
serialization. Wasmtime runs detached from the GIL; query callbacks attach only
for the callback. Same-runtime recursive invocation is rejected.

Mock vLLM/SGLang adapters are implemented and tested. They are not claims of
live version-pinned integration.

## Retained Guarantees

- exact v0.5 host import and policy export;
- no state or WASI import;
- package integrity and atomic replacement;
- fresh bounded Wasm instances;
- finite aligned scores and budget bounds;
- full-set scheduling;
- revisioned commits;
- feedback deduplication;
- deterministic replay;
- Rust/Python JSON parity;
- native fallback.

## Deferred

- typed schemas and efficient encodings;
- generated request models;
- JSON Patch;
- etcd implementation and ownership handoff;
- crash-durable deduplication;
- distributed authoritative-event batching;
- automatic conflict retry;
- candidate-local indexes;
- physical action rollback;
- live PIE hook wiring;
- version-pinned vLLM/SGLang integrations;
- Python policy authoring.

`plex_paper.md` remains unchanged.
