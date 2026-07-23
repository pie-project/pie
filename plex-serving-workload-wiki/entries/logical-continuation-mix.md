# Logical Continuation Mix

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- Qwen-Bailian
- BurstGPT
- LongMemEval
- mini-SWE-agent

## Generator factors

- fraction of one-shot/chat/tool/agent requests
- turn count
- pause duration
- context growth
- continuation expiry

## Required lifecycle events

- create
- boundary
- tool-start
- tool-finish
- continue
- progress
- finish

## PLEX operations exercised

- admit
- route
- schedule
- evict
- feedback

## Metrics

- logical-request completion latency
- resume queueing
- recompute tokens
- KV residency time
- throughput

## Why this is new

No public trace combines real arrivals with explicit logical-request continuation and physical/accounting shadows.
