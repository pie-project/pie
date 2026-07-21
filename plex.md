# PLEX JSON Policy PoC

## Status

PLEX is currently implemented as an intentionally untyped JSON proof of
concept. Its purpose is to validate:

- the five-operation policy boundary;
- `route -> admit -> schedule` for every generation;
- trusted policy mutation of complete request documents;
- request-local state across hooks and tool continuations;
- deterministic score and token-budget mechanics;
- failure rollback and native fallback;
- portable, bounded Wasm execution.

The breaking ABI is `pie:plex@0.2.0`. It is a research ABI, not the final
efficient or typed representation.

PLEX remains independent from live Pie mechanics. It does not yet modify
gateway routing, worker RPC, the engine scheduler, KV reclaim, or inferlet
lifecycle code.

## Lifecycle

Every generation follows:

```text
route -> admit -> schedule
```

This applies to:

- the initial generation;
- a tool continuation;
- any later generation under the same logical request;
- a stage transition that performs a new placement/admission decision.

A tool boundary is:

```text
feedback(tool-boundary)
-> external tool execution
-> create continuation generation
-> route(continuation)
-> admit(continuation)
-> schedule
```

`route` ranks host-provided feasible targets. The host invokes `admit` on the
highest-ranked target:

- `accept`: enqueue the generation;
- `reject`: try the next ranked target;
- `defer`: retry the same target up to a small host-defined bound.

The PoC does not implement a distributed placement retry protocol.

## Trust Model

Operator-installed policy code is trusted to inspect and rewrite request
content. Wasm is used for:

- a portable ABI;
- memory isolation;
- bounded execution;
- trap containment;
- package replacement.

Wasm is not an authorization boundary between a policy and a request.

Every hook may read and modify:

- prompts and messages;
- token IDs;
- sampling parameters;
- model or adapter selection;
- tool definitions;
- output constraints;
- user metadata;
- request-local policy state;
- any additional request dictionary keys.

Only `request.identity` is immutable.

## Repository Layout

```text
interface/plex/          Minimal JSON contract and host fill validation
interface/plex/wit/      Canonical pie:plex@0.2.0 WIT
runtime/policy/          Package, Wasmtime host, registry, request store, replay
sdk/rust/plex/           serde_json::Value guest SDK
tests/policies/          JSON reference, failure, and research policies
```

`pie-policy` depends on `pie-plex`, never on Pie gateway, worker, or engine
mechanics. `scripts/check-plex-layering.sh` enforces the boundary.

## Wasm ABI

The complete WIT surface is:

```wit
package pie:plex@0.2.0;

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

There are no PLEX imports and no WASI imports.

All five exports are statically present. The manifest declares which
operations the package owns. An unimplemented operation returns
`fallback-required`.

## Manifest

The manifest contains only:

- exact contract version;
- package name;
- package version;
- owned operations;
- memory limit;
- fuel limit;
- deadline;
- maximum input bytes;
- maximum output bytes.

Unknown manifest fields are rejected. There are no field, map, event,
metadata, capability, or invocation-mode schemas.

Set-dependent invocation is the only mode.

## Request Document

The canonical request shape is:

```json
{
  "identity": {
    "logical_request_id": "opaque-host-id",
    "generation_id": 0
  },
  "body": {},
  "metadata": {},
  "state": {}
}
```

Semantics:

- `identity` is host-owned and immutable;
- `body` is an arbitrary mutable object;
- `metadata` is arbitrary mutable user/application input;
- `state` is arbitrary mutable policy-owned request state;
- extra request keys are preserved;
- valid hook mutations become visible to later hooks.

The in-process `CanonicalRequestStore` owns one current request document per
logical request.

### Continuations

Creating a continuation:

1. increments `generation_id`;
2. installs the newly supplied body;
3. preserves prior `state`;
4. preserves prior metadata;
5. shallow-merges new metadata over prior metadata;
6. lets the new value win on duplicate keys.

Terminal feedback may be followed by removal of the canonical request.

Distributed request-state propagation is not implemented.

## Policy Response

The SDK wrapper returns:

```json
{
  "input": {},
  "result": {}
}
```

`input` is the complete mutated operation input. `result` is the
operation-specific decision.

Returning the entire input is intentionally inefficient. The PoC has no patch
language, diff representation, column encoding, or typed accessor layer.

The host commits request mutation only after:

1. Wasm returns successfully;
2. output JSON parses;
3. the wrapper shape validates;
4. immutable host fields validate;
5. the operation result validates.

Any failure discards the returned mutation.

## Operation Conventions

These are host-validated JSON conventions, not WIT schemas. Additional keys
are allowed and preserved unless they belong to host-owned candidate/context
data.

### `route`

Input:

```json
{
  "cause": "generation-arrival",
  "request": {},
  "candidates": [
    {
      "id": "node-a",
      "facts": {
        "queue_depth": 10,
        "cached_tokens": 800
      }
    }
  ],
  "context": {
    "config": {}
  }
}
```

Result:

```json
{"scores": [790.0]}
```

Rules:

- exactly one finite score per candidate;
- higher score is preferred;
- ties retain input order;
- candidate count, order, and ID are immutable;
- candidate/context mutations have no effect;
- `request` is fully mutable.

### `admit`

Input:

```json
{
  "cause": "continuation",
  "request": {},
  "target": {
    "id": "node-a",
    "facts": {
      "queue_depth": 10,
      "free_kv_bytes": 1048576
    }
  },
  "context": {
    "config": {}
  }
}
```

Result:

```json
{"decision": "accept"}
```

`decision` must be `accept`, `defer`, or `reject`. Target identity and context
are host-owned. The complete request remains mutable for every decision,
including defer and reject.

### `schedule`

Input:

```json
{
  "cause": "service-step",
  "runnable": [
    {
      "request": {},
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
    "config": {},
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

- one decision per runnable entry;
- finite scores only;
- higher score first;
- stable ties;
- missing/null budget requests the candidate/host maximum;
- zero requests no service;
- explicit budgets require capability;
- explicit budgets may not exceed candidate or host bounds;
- `runnable[i].request` is mutable;
- runnable order, identity, facts, limits, capacity, and context are host-owned.

The host uses a stable descending greedy fill under `max_selected` and
`max_total_tokens`.

### `evict`

Input:

```json
{
  "cause": "allocation-deficit",
  "bytes_needed": 4096,
  "resident": [
    {
      "id": "host-unit",
      "size_bytes": 4096,
      "request": {},
      "facts": {
        "reload_cost": 200.0
      }
    }
  ],
  "context": {
    "config": {}
  }
}
```

Result:

```json
{"scores": [200.0]}
```

Scores are retention scores:

- one finite score per resident unit;
- lower scores are reclaimed first;
- ties retain input order;
- reclaim continues until `bytes_needed` is met;
- the adapter supplies only legally reclaimable units;
- non-null attributed requests are mutable;
- a null request remains unattributed.

Shared-unit multi-owner request mutation is out of scope.

### `feedback`

Input:

```json
{
  "delivery_id": "opaque-delivery-id",
  "records": [
    {
      "event": "progress",
      "request": {},
      "facts": {
        "committed_tokens": 8,
        "service_us": 1250
      }
    }
  ],
  "context": {
    "config": {}
  }
}
```

Result:

```json
{}
```

Rules:

- events are open strings;
- unknown events may be ignored;
- records contain enacted outcomes;
- every record request is mutable;
- delivery ID, event order, facts, and context are host-owned.

## Mutation Validation

For every request returned by a policy:

- the top-level request remains an object;
- `identity`, `body`, `metadata`, and `state` remain objects as required;
- `logical_request_id` remains unchanged;
- `generation_id` remains unchanged.

For route, schedule, and evict:

- candidate count and order remain unchanged;
- candidate identity remains unchanged;
- host facts and capacity are restored from the original input.

For feedback:

- delivery ID remains unchanged;
- record count and event order remain unchanged;
- enacted facts are restored from the original input.

Policy-added top-level and request-local keys are otherwise preserved.

There is no automatic reroute when admit or schedule rewrites a prompt. The
trusted policy is responsible for coordinating mutation and decision.

## Failure and Rollback

The host returns:

```text
Success
Unavailable
FallbackRequired
```

Fallback classes include:

- invalid input;
- instantiation failure;
- explicit policy fallback;
- trap;
- fuel exhaustion;
- deadline;
- host saturation;
- invalid output.

For malformed JSON, malformed decision shape, invalid score, invalid budget,
identity mutation, candidate mutation, policy error, trap, fuel exhaustion, or
deadline:

- no returned request mutation is applied;
- canonical request state remains unchanged;
- the caller invokes native/default behavior.

The PoC does not implement map transactions or a general rollback protocol.
Atomicity comes from validating a full cloned JSON response before replacing
the in-process canonical request.

## Feedback Deduplication

Feedback deduplication is process-local and keyed by delivery ID within an
attached package lineage.

- duplicate committed delivery: do not execute the policy;
- duplicate mutation: do not apply it again;
- in-flight duplicate: fallback;
- full bounded ledger: fallback;
- failed first delivery: reservation is removed;
- replacement transfers committed delivery IDs.

Deduplication is not crash durable.

## Package and Attachment

The package envelope retains:

- magic and package-format version;
- JSON manifest length;
- component length;
- BLAKE3 integrity digest;
- manifest bytes;
- component bytes.

The host validates size limits and integrity before compiling.

Attachment preparation:

1. decodes the package;
2. validates the v0.2 manifest;
3. compiles the component;
4. verifies zero imports;
5. verifies only `pie:plex/policy@0.2.0` is exported;
6. prelinks the component;
7. instantiates once inside declared limits;
8. publishes only after preparation succeeds.

The immutable attachment registry retains one owner per operation. Replacement
prepares first, waits for old invocation snapshots to drain, transfers feedback
dedup state, and atomically publishes a new generation.

There is no attachment-time request schema or capability linking.

## Runtime Bounds

The PoC retains:

- one policy-specific Wasmtime engine;
- no WASI imports;
- disabled Wasm threads/shared memory;
- one fresh store and component instance per invocation;
- one linear-memory limit;
- fuel;
- epoch deadline;
- input/output byte limits;
- bounded concurrent invocations;
- bounded feedback ledger.

The policy manifest may request less authority than the host maximum, never
more.

## Rust Guest SDK

The author-facing API is:

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

Every method defaults to `fallback-required`.

`export_policy!`:

1. parses the input string;
2. requires a top-level object;
3. calls the mutable policy hook;
4. wraps mutated input and result;
5. serializes JSON;
6. returns clear parse/policy/serialization errors.

Policy code uses ordinary dictionary access. There are no handles, generated
request structs, typed maps, record batches, or typed accessors.

## Canonical Request Store

`CanonicalRequestStore` provides:

- logical-request creation;
- canonical lookup;
- atomic application of validated operation mutations;
- continuation creation;
- terminal removal.

Applying an operation collects all associated request documents, validates
their identities against current canonical state, updates a cloned store, then
publishes all updates together.

This store is deliberately in-process. Gateway/worker distribution is future
integration work.

## Replay

The replay runner supports:

- package attach/replace/detach;
- request creation;
- continuation creation;
- generic operation invocation;
- combined route/admit placement;
- terminal feedback and request removal;
- canonical request reads.

Replay uses a deterministic engine without real-time epoch ticks. Reports
include operation responses, host selections, placement outcomes, fallback
classes, and attachment generations. Running an identical trace twice must
produce an identical report.

## Reference Policies

The JSON corpus includes:

- least-loaded route;
- rewrite-and-admit;
- least-attained-service schedule;
- retention-score eviction;
- feedback accounting;
- one coordinated five-operation policy.

Failure fixtures include:

- unimplemented fallback;
- malformed result length;
- raw NaN and infinity output;
- invalid token budget;
- identity mutation;
- candidate mutation;
- mutate-then-fallback;
- trap after mutation;
- infinite loop/fuel exhaustion.

Research stress policies remain as ordinary JSON policies:

- Agentix;
- Continuum;
- KVFlow;
- Preble;
- Helium.

Their physical engine mechanisms remain out of scope.

## Required Lifecycle Coverage

The fixture harness verifies:

1. create logical request `L`, generation 0;
2. route mutation becomes visible to admit;
3. admit runs and mutates the prompt/state;
4. schedule sees both prior mutations;
5. eviction mutates an attributed request;
6. progress feedback updates attained service;
7. duplicate feedback does not double-count;
8. tool-boundary feedback updates state;
9. continuation generation 1 preserves state and merged metadata;
10. route/admit/schedule run again;
11. admission count proves admit ran for both generations;
12. terminal feedback removes the request.

It also verifies stable score order, defer/reject bounds, invalid mutation
rollback, token budgets, non-finite output, fallback, and deterministic replay.

## Validation and CI

The repository uses existing Cargo, `wasm-tools`, and shell tooling:

- host contract/runtime tests;
- guest SDK check for `wasm32-unknown-unknown`;
- all policy component builds;
- fixture/replay harness;
- generated component WIT audit;
- zero component imports;
- PLEX layering check;
- workspace formatting/clippy/checks.

Canonical WIT remains in `interface/plex/wit` and is synchronized to the SDK
with `scripts/sync-wit.sh`.

## Explicit Non-goals

The JSON PoC does not implement:

- typed schemas;
- generated request models;
- field, map, event, or capability handles;
- columnar encoding;
- patches or diffs;
- mutable package-global maps;
- distributed request state;
- distributed transactions;
- crash-durable deduplication;
- automatic reroute after downstream mutation;
- candidate-local scheduler indexes;
- shared-unit multi-owner mutation;
- `prefetch`;
- `rebalance`;
- live adapters for every serving engine.

Efficient encoding, schema validation, and distributed integration are future
work after the five-operation and mutation semantics are validated.
