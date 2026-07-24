# Tool Resource Lifecycle and Cross-Resource Scheduling

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- ThunderAgent
- Terminal-Bench
- TheAgentCompany
- tau3-bench

## Generator factors

- sandbox startup
- port/disk limits
- environment reuse
- garbage collection
- LLM/tool overlap

## Required lifecycle events

- tool-resource-create
- ready
- use
- idle
- release
- leak

## PLEX operations exercised

- admit
- schedule
- feedback
- prefetch

## Metrics

- workflow throughput
- startup overlap
- resource leaks
- port/disk exhaustion
- JCT

## Why this is new

Serving papers usually schedule GPU/KV only, while sustained agent throughput is limited by tool environments too.
