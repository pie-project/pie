# Memory-Tier Bandwidth Degradation

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- LLMServingSim
- SuperInfer
- Mooncake

## Generator factors

- HBM/DRAM/CXL/SSD capacity
- bandwidth jitter
- tier outage
- prefetch lead time
- recompute-vs-load

## Required lifecycle events

- memory pressure
- offload
- prefetch
- load complete/fail
- evict

## PLEX operations exercised

- schedule
- evict
- prefetch
- feedback

## Metrics

- SLO goodput
- stall time
- bytes moved
- wasted prefetch
- recompute

## Why this is new

Most cache policies assume stable transfer cost; real shared tiers and networks experience congestion and failure.
