# RAGCache

> Canonical title: **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2404.12457](https://arxiv.org/abs/2404.12457) |
| Venue / status | arXiv preprint, 2024 |
| Year | 2024 |
| Authors | Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, Xin Jin |
| Institutions / group context | Peking University |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations include Peking University; no official public artifact was confirmed. |
| Citation count | 5 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2404.12457 |
| arXiv | 2404.12457 |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on prefix-aware gdsf score: recency + frequency × recompute cost / size. The reported evaluation context includes Retrieval-augmented generation workloads with ordered document sequences.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Prefix-aware GDSF score: recency + frequency × recompute cost / size.

## PLEX mapping

Very clean direct `E` replica.

## Datasets and evaluation workloads

- Retrieval-augmented generation workloads with ordered document sequences

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, et al. “RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation.” arXiv preprint, 2024, 2024. https://doi.org/10.48550/arxiv.2404.12457.

## Sources

- [Primary paper](https://arxiv.org/abs/2404.12457)
- [OpenAlex record](https://openalex.org/W4395021895)
