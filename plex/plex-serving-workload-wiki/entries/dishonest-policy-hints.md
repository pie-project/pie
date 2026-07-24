# Dishonest Metadata and Learned Credibility

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | core |

## Seed sources

- BurstGPT sessions
- Qwen-Bailian lengths
- PLEX declared/observed fields

## Generator factors

- inflated predicted length
- urgency
- reuse probability
- tool TTL
- tenant attack fraction
- credibility learning

## Required lifecycle events

- declare
- observed-progress
- outcome
- credibility-update

## PLEX operations exercised

- admit
- route
- schedule
- evict
- feedback

## Metrics

- honest-tenant share/p99
- attacker gain
- recovery samples
- SLO goodput

## Why this is new

No standard serving dataset contains both caller-declared and host-observed values, so trust-boundary policies remain unvalidated.
