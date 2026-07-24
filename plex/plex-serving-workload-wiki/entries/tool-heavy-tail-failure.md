# Heavy-Tailed Tool Latency and Failure

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- tau3-bench
- InferCept API classes
- Continuum

## Generator factors

- lognormal/Pareto tool delay
- bimodal tools
- timeouts
- partial failure
- retry/backoff
- prediction error

## Required lifecycle events

- tool-start
- tool-result
- tool-timeout
- retry
- cancel
- continue

## PLEX operations exercised

- schedule
- evict
- feedback
- prefetch

## Metrics

- JCT
- wasted KV-byte-ms
- retry amplification
- queueing after miss
- deadlock/progress

## Why this is new

Public agent benchmarks expose tool semantics but not realistic tool-duration and outage distributions.
