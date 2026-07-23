# RAGCache

> Canonical title: **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2404.12457](https://arxiv.org/abs/2404.12457) |
| Venue / status | arXiv preprint, 2024 |
| Year | 2024 |
| Authors | Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; no official public artifact was confirmed. |
| OpenAlex cited-by count | 5 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.48550/arxiv.2404.12457 |
| arXiv | Not resolved |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. Its central policy contribution is: Prefix-aware GDSF score: recency + frequency × recompute cost / size.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Prefix-aware GDSF score: recency + frequency × recompute cost / size.

## PLEX mapping

Very clean direct `E` replica.

## Datasets and workloads

- Retrieval-augmented generation workloads with ordered document sequences

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, et al.. “RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation.” arXiv preprint, 2024, 2024. https://doi.org/10.48550/arxiv.2404.12457.

## Sources

- [Primary paper](https://arxiv.org/abs/2404.12457)
- [OpenAlex record](https://openalex.org/W4395021895)
