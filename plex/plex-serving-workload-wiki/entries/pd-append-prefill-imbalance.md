# Prefill/Decode/Append-Prefill Imbalance

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- Azure lengths
- Mooncake agent trace
- ConServe
- DistServe

## Generator factors

- initial prefill
- append-prefill per turn
- decode length
- P/D pool ratio
- KV transfer cost
- conversation pinning

## Required lifecycle events

- route-stage
- transfer
- append-prefill
- decode
- continue

## PLEX operations exercised

- route
- schedule
- feedback
- rebalance

## Metrics

- TTFT
- tail TBT
- conversation completion
- transfer bytes
- pool imbalance

## Why this is new

Standard P/D traces treat turns independently and underrepresent agent conversations with one heavy prefill and a long tail.
