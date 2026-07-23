# Multi-Region Failover, Cost, and Carbon

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- BurstGPT periodicity
- SkyWalker
- Azure Functions
- ServeGen

## Generator factors

- regional demand
- RTT/bandwidth
- energy/carbon price
- spot loss
- outage
- data residency

## Required lifecycle events

- route
- region failure
- drain
- migration
- recover

## PLEX operations exercised

- route
- admit
- rebalance
- feedback

## Metrics

- SLO goodput
- cost
- carbon
- failover recovery
- state-transfer bytes

## Why this is new

Existing routing evaluations rarely combine locality, live state, failure, and time-varying energy/cost.
