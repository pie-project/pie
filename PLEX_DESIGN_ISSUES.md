# PLEX JSON PoC Design Record

## Current Decision

PLEX now uses a breaking, intentionally untyped `pie:plex@0.2.0` research ABI.
One JSON string crosses the Wasm boundary in each direction. The prior v0.1
handle/schema/column/map design has been removed rather than retained behind
compatibility adapters.

## Lifecycle Boundary

The five operations remain:

- `route`
- `admit`
- `schedule`
- `evict`
- `feedback`

`admit` is target-local generation admission and follows `route` for every
generation:

```text
route -> admit -> schedule
```

Initial generations, continuations, and placement-changing stage transitions
all follow this sequence.

## Trust and Mutation

Operator-installed policy code is trusted with full request contents.

- `identity` is host-owned and immutable.
- `body`, `metadata`, and `state` are mutable dictionaries.
- request mutation becomes visible to subsequent hooks only after the complete
  policy response validates.
- failure discards the invocation's mutation.

User metadata is arbitrary input, not a typed or trusted fact vocabulary.

## State

Policy state retained across hooks in this milestone is request-local:

```text
request.state
```

The in-process canonical request store preserves state and shallow-merges
metadata across continuations. Mutable package-global maps, typed map schemas,
revision transactions, and cross-process state are removed.

Feedback retains bounded process-local delivery deduplication so duplicate
events do not apply request mutation twice.

## ABI

All five component exports accept `string` and return `result<string, string>`.
Components import no PLEX interfaces and no WASI interfaces.

The simplified manifest contains only package identity, owned operations, and
memory/fuel/deadline/input/output limits.

## Retained Host Guarantees

The PoC still enforces:

- package integrity and size limits;
- zero component imports;
- fresh Wasm instances;
- memory, fuel, deadline, input, and output bounds;
- finite aligned scores;
- token-budget capability and bounds;
- request identity immutability;
- candidate count/order/identity immutability;
- native fallback on any invalid result or failure;
- atomic package publication and replacement.

## Intentional Tradeoffs

Returning the complete mutated input is inefficient but makes mutation
semantics obvious. There is no patch language or typed accessor layer.

Missing keys and unexpected JSON types are ordinary policy cases. A policy may
default them or return `fallback-required`.

The host does not automatically reroute when admit or schedule changes a
prompt. Trusted policy code owns consistency between mutation and decision.

## Deferred

- typed schemas and efficient encodings;
- capability and field linking;
- provenance-enforced metadata/facts;
- mutable cross-request global state;
- distributed request state;
- crash-durable deduplication;
- automatic mutation dependency tracking;
- candidate-local indexes;
- shared-unit multi-owner mutation;
- prefetch/rebalance;
- production Pie adapters.

`plex_paper.md` remains unchanged; reconciling paper claims with the JSON PoC
is a separate task.
