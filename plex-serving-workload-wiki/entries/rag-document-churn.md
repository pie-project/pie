# RAG Document Churn and Shared Retrieval

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- BEIR
- RAGBench
- HotpotQA
- LongBench

## Generator factors

- document popularity
- top-k overlap
- corpus update
- position/order
- retrieval fanout
- quality sensitivity

## Required lifecycle events

- retrieve
- document set
- prefetch
- generation
- quality feedback

## PLEX operations exercised

- route
- schedule
- evict
- prefetch
- feedback

## Metrics

- TTFT
- document/KV hit rate
- quality
- stale-hit rate
- memory

## Why this is new

Static RAG benchmarks do not model changing document popularity or cache-invalidating corpus updates.
