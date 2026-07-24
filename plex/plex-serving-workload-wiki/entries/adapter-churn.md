# Adapter Popularity Churn

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- S-LoRA synthetic workload
- Chameleon
- dLoRA

## Generator factors

- adapter Zipf skew
- rank/size
- tenant affinity
- phase change
- load/unload time
- base-model mix

## Required lifecycle events

- adapter request
- load
- evict
- merge/unmerge
- migration

## PLEX operations exercised

- admit
- route
- schedule
- evict
- prefetch
- rebalance

## Metrics

- cold-start TTFT
- adapter hit rate
- fairness
- memory utilization
- migration cost

## Why this is new

No public production multi-LoRA request trace exists; all major systems rely on synthetic popularity assumptions.
