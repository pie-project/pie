# Fairness versus Locality Conflict

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- VTC synthetic clients
- DLPM
- Qwen-Bailian prefixes

## Generator factors

- tenant demand skew
- prefix overlap per tenant
- weights
- honest/malicious hot prefixes
- quantum

## Required lifecycle events

- tenant arrival
- service charge
- cache hit
- deadline

## PLEX operations exercised

- route
- admit
- schedule
- feedback

## Metrics

- service gap
- Jain fairness
- throughput
- victim p99
- cache hit

## Why this is new

Existing fairness and locality papers use different workloads; one factorial trace is needed for direct policy comparison.
