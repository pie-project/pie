# Autoscaling with Stateful Cold Start

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- BurstGPT
- ServeGen
- Azure Functions
- ServerlessLLM

## Generator factors

- forecast error
- model load time
- KV warmup
- scale unit
- drain policy
- burst duration

## Required lifecycle events

- scale request
- instance ready
- warm cache
- drain
- terminate

## PLEX operations exercised

- admit
- route
- rebalance
- feedback

## Metrics

- burst goodput
- cold-start misses
- overprovisioning cost
- drain time
- cache loss

## Why this is new

Serverless traces model cold starts, while LLM traces model tokens; their interaction is rarely evaluated together.
