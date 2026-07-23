# PEEK

> Canonical title: **PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2607.02525](https://arxiv.org/abs/2607.02525) |
| Venue / status | arXiv preprint |
| Year | 2026 |
| Authors | Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zheng Zhen |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2607.02525 |
| Artifact | [Public artifact](https://github.com/xiexbing/peek) |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on queue-structure-aware scheduling and eviction. The reported evaluation context includes High-prefix-sharing online workloads, Long-document RAG, Mixed and no-sharing controls on SGLang and vLLM.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Queue-structure-aware scheduling and eviction.

## PLEX mapping

Direct coordinated `S+E`.

## Datasets and evaluation workloads

- High-prefix-sharing online workloads
- Long-document RAG
- Mixed and no-sharing controls on SGLang and vLLM

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zheng Zhen. “PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving.” arXiv preprint, 2026. https://arxiv.org/abs/2607.02525.

## Sources

- [Primary paper](https://arxiv.org/abs/2607.02525)
- [OpenAlex record](https://openalex.org/W7167747234)
- [Artifact](https://github.com/xiexbing/peek)
