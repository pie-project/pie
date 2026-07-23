# PEEK

> Canonical title: **PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2607.02525](https://arxiv.org/abs/2607.02525) |
| Venue / status | arXiv (Cornell University) |
| Year | 2026 |
| Authors | Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zheng Zhen |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | 0 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | Not resolved |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/xiexbing/peek) |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. Its central policy contribution is: Queue-structure-aware scheduling and eviction.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Queue-structure-aware scheduling and eviction.

## PLEX mapping

Direct coordinated `S+E`.

## Datasets and workloads

- High-prefix-sharing online workloads
- Long-document RAG
- Mixed and no-sharing controls on SGLang and vLLM

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zheng Zhen. “PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving.” arXiv (Cornell University), 2026. https://arxiv.org/abs/2607.02525.

## Sources

- [Primary paper](https://arxiv.org/abs/2607.02525)
- [OpenAlex record](https://openalex.org/W7167747234)
- [Artifact](https://github.com/xiexbing/peek)
