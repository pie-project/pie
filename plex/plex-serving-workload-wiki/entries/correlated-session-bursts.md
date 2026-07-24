# Correlated Session and Workflow Bursts

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- BurstGPT sessions
- Bailian parent links
- ServeGen client pools

## Generator factors

- session fanout
- shared external trigger
- tenant correlation
- burst scale
- workflow synchronization

## Required lifecycle events

- session start
- fanout
- join
- burst
- finish

## PLEX operations exercised

- admit
- route
- schedule
- feedback
- rebalance

## Metrics

- queue growth
- tail JCT
- fairness
- autoscaling reaction
- cache locality

## Why this is new

Independent Poisson replay misses correlation among requests from the same session, tenant, or workflow.
