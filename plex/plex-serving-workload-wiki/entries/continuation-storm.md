# Continuation Storm

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- Mooncake ToolAgent
- tau3-bench
- BurstGPT burst model

## Generator factors

- synchronized tool completions
- storm width
- resident-KV fraction
- resume priority
- memory pressure

## Required lifecycle events

- tool-finish bursts
- continuation enqueue
- cache hit/miss
- preemption

## PLEX operations exercised

- route
- schedule
- evict
- feedback

## Metrics

- resume p99
- SLO goodput
- cache hit
- preemption count
- fairness

## Why this is new

Prior work studies individual tool pauses but rarely correlated resumptions after an external service recovers.
