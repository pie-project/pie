# Speculative Fanout with Cancellation

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- Math500
- LiveCodeBench
- MCTS/Best-of-N agents

## Generator factors

- branch count
- quality distribution
- early winner detection
- cancel latency
- shared-prefix depth

## Required lifecycle events

- fork
- branch-progress
- winner
- cancel
- terminal

## PLEX operations exercised

- admit
- schedule
- evict
- feedback

## Metrics

- time-to-correct-answer
- wasted tokens
- cancel response time
- cache reuse
- goodput

## Why this is new

Current request traces almost never include branch utility or cancellation, despite rising test-time compute workloads.
