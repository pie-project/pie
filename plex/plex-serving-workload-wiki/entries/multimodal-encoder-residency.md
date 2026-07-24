# Multimodal Encoder-State Residency

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- MMMU
- Video-MME
- ServeGen multimodal

## Generator factors

- image/video/audio size
- encoder cost
- reuse across turns
- encoder-output size
- modality-specific SLO

## Required lifecycle events

- upload
- encode
- resident-state
- reuse
- evict

## PLEX operations exercised

- admit
- route
- schedule
- evict
- prefetch
- feedback

## Metrics

- TTFT
- encoder reuse
- resident bytes
- modality fairness
- energy

## Why this is new

KV-centric workloads omit reusable image/audio encoder state and heterogeneous preprocessing bottlenecks.
