# Session Abandonment, Retry, and Duplicate Delivery

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- BurstGPT failures
- WildChat timestamps
- agent task harnesses

## Generator factors

- client timeout
- cancel delay
- automatic retry
- duplicate request
- idempotency
- late completion

## Required lifecycle events

- cancel
- timeout
- retry
- duplicate
- finish

## PLEX operations exercised

- admit
- schedule
- evict
- feedback

## Metrics

- wasted tokens
- duplicate work
- cleanup latency
- successful retry latency
- state leaks

## Why this is new

Public traces rarely expose cancellation/retry semantics even though they determine wasted compute and cleanup correctness.
