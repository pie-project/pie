# Mixed Human, Agent, Batch, and Reasoning Service Classes

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- Azure code/conversation
- BurstGPT
- Bailian thinking/coder
- ServeGen categories

## Generator factors

- TTFT/TPOT/JCT SLO tiers
- human reading rate
- agent full-response dependency
- batch deadlines
- price weights

## Required lifecycle events

- arrival
- SLO declaration
- progress
- deadline miss
- finish

## PLEX operations exercised

- admit
- route
- schedule
- feedback

## Metrics

- per-class goodput
- revenue/cost
- starvation
- GPU utilization
- QoE

## Why this is new

Prior systems usually optimize one objective at a time; production fleets mix incompatible latency semantics.
