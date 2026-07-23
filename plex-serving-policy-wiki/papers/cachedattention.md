# CachedAttention

> Canonical title: **Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2403.19708](https://arxiv.org/abs/2403.19708) |
| Venue / status | USENIX ATC 2024 |
| Year | 2024 |
| Authors | Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | strong publication signal from a selective systems/ML venue (USENIX ATC 2024); author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 1 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2403.19708 |
| arXiv | 2403.19708 |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on session-scoped hierarchical kv retention and preload. The reported evaluation context includes ShareGPT multi-turn conversations.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Session-scoped hierarchical KV retention and preload.

## PLEX mapping

`E+P`; much of the contribution is storage mechanics.

## Datasets and evaluation workloads

- ShareGPT multi-turn conversations

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, et al. “Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention.” USENIX ATC 2024, 2024. https://doi.org/10.48550/arxiv.2403.19708.

## Sources

- [Primary paper](https://arxiv.org/abs/2403.19708)
- [OpenAlex record](https://openalex.org/W4393853379)
