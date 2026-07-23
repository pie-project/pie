# Online-Revealed Agent DAG

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- AgentBench
- TheAgentCompany
- Autellix
- Helium

## Generator factors

- chain/fanout/join/loop
- online node revelation
- critical path
- dynamic branch probability
- node model/tool type

## Required lifecycle events

- node-ready
- dependency-resolved
- fork
- join
- node-finish

## PLEX operations exercised

- route
- admit
- schedule
- feedback

## Metrics

- workflow makespan
- critical-path stall
- parallelism
- fairness
- SLO goodput

## Why this is new

Most serving evaluations either know a static DAG or ignore graph structure; dynamic revelation remains weakly covered.
