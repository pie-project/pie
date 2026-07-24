# Prefix Hotspot and Popularity Shift

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- Qwen-Bailian block hashes
- Mooncake hashes
- ServeGen rate shifts

## Generator factors

- Zipf skew
- hot-prefix coverage
- phase shift
- new application launch
- cache replication

## Required lifecycle events

- arrival
- prefix map
- cache insert/evict
- route
- scale event

## PLEX operations exercised

- route
- evict
- prefetch
- rebalance

## Metrics

- TTFT
- load imbalance
- hit rate
- migration
- recovery time after shift

## Why this is new

Static prefix distributions favor cache-aware policies and miss their failure mode under rapid popularity inversion.
