# Reasoning Progress and Marginal Utility

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- Math500
- LiveCodeBench
- Dynasor

## Generator factors

- token budget
- confidence trajectory
- answer diversity
- difficulty
- early stop
- utility calibration

## Required lifecycle events

- branch progress
- candidate answer
- confidence
- stop
- verdict

## PLEX operations exercised

- admit
- schedule
- feedback

## Metrics

- accuracy-goodput
- tokens per solved task
- tail JCT
- fairness
- premature-stop rate

## Why this is new

Request length alone cannot validate reasoning schedulers; policies need online marginal-utility signals and final correctness.
