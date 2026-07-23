# CachedAttention

> Canonical title: **Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2403.19708](https://arxiv.org/abs/2403.19708) |
| Venue / status | USENIX ATC 2024 |
| Year | 2024 |
| Authors | Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (USENIX ATC 2024); author affiliations were not reliably resolved; no official public artifact was confirmed. |
| OpenAlex cited-by count | 1 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.48550/arxiv.2403.19708 |
| arXiv | Not resolved |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. Its central policy contribution is: Session-scoped hierarchical KV retention and preload.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Session-scoped hierarchical KV retention and preload.

## PLEX mapping

`E+P`; much of the contribution is storage mechanics.

## Datasets and workloads

- ShareGPT multi-turn conversations

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, et al.. “Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention.” USENIX ATC 2024, 2024. https://doi.org/10.48550/arxiv.2403.19708.

## Sources

- [Primary paper](https://arxiv.org/abs/2403.19708)
- [OpenAlex record](https://openalex.org/W4393853379)
