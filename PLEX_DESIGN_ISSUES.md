# PLEX JSON State-Map Design Record

## Current Decision

PLEX uses the breaking, intentionally untyped `pie:plex@0.3.0` research ABI.
One JSON string crosses the Wasm boundary in each direction. The removed v0.1
handle/schema/column/map system and the v0.2 embedded request document are not
retained behind compatibility adapters.

## State Model

Every invocation receives:

- one process-local global map;
- one map per referenced logical request;
- transient operation context.

Both persistent scopes contain exactly:

```text
facts
fields
scratch
```

`facts` is host-written and policy-read-only. `fields` is policy-writable and
engine-interpreted. `scratch` is policy-writable and engine-opaque.

The global map is shared by every active operation owner in a
`LifecycleHost`. Request maps survive hooks and continuations and are removed
at terminal completion.

## Identity and Observations

Request identity moved to host facts:

```text
requests[id].facts.logical_request_id
requests[id].facts.generation_id
```

The map key must match `logical_request_id`.

Actual placement is recorded only after enactment through
`record_enacted_placement`. Route scores and admission acceptance do not
speculatively change `previous_target`.

Host-observed service belongs in facts. Policy predictions, debt models, and
derived accounting belong in scratch.

## Mutation Rule

A response may change only:

```text
global.fields
global.scratch
requests[id].fields
requests[id].scratch
```

Facts, request-map membership, candidate/event facts, capacity, identities,
causes, delivery IDs, capabilities, and all other transient context are
read-only.

Attempted read-only mutation causes fallback. The host does not silently
restore it.

## Atomicity and Concurrency

One `LifecycleHost` mutex serializes:

```text
hydrate -> invoke -> validate -> commit
```

Global and request mutations commit atomically only after complete response
validation. `route -> admit` holds the same gate across the sequence while
making each successful mutation visible to the next hook.

This is the simplest intentional PoC tradeoff. Revisions, CAS loops,
distributed transactions, and scalable concurrent state updates are deferred.

## Feedback

Feedback deduplication is part of the state store. Mutation and delivery-ID
commit happen atomically. Duplicates skip Wasm and return the cached result.
The ledger survives package replacement but not process restart.

## Continuations

Continuation creation:

- preserves request fields and scratch;
- replaces `fields.body`;
- shallow-merges `fields.metadata`;
- increments host-owned generation;
- preserves durable facts such as `previous_target`;
- clears the explicit generation-local key `current_target`.

## Retained Runtime Guarantees

The PoC retains package integrity, zero imports, fresh Wasm instances,
memory/fuel/deadline/input/output bounds, aligned finite scores, budget bounds,
native fallback, deterministic replay, and atomic package publication.

## Intentional Tradeoffs

- Full mutated JSON is returned instead of a patch.
- Policies share one global namespace and coordinate key ownership by
  convention.
- JSON shape validation is minimal and untyped.
- State and deduplication are process-local.
- Stateful invocations are serialized.
- The host does not automatically reroute after policy request mutation.

## Deferred

- typed schemas and efficient encodings;
- generated request models;
- field/map/event/capability handles;
- provenance-enforced facts;
- per-package global isolation;
- distributed/persistent state;
- crash-durable deduplication;
- scalable state transactions;
- automatic mutation dependency tracking;
- candidate-local indexes;
- shared-unit multi-owner mutation;
- prefetch/rebalance;
- production Pie adapters.

`plex_paper.md` remains unchanged. Paper reconciliation and measurements are
separate work.
