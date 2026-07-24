# SGLang / RadixAttention

> Canonical title: **SGLang: Efficient Execution of Structured Language Model Programs**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104) |
| Venue / status | NeurIPS 2024 |
| Year | 2024 |
| Authors | Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng |
| Institutions / group context | UC Berkeley LMSYS Org |
| Reputation evidence | strong publication signal from a selective systems/ML venue (NeurIPS 2024); author affiliations include UC Berkeley LMSYS Org; a public implementation or artifact is linked. |
| Citation count | 10 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2312.07104 |
| arXiv | 2312.07104 |
| Artifact | [Public artifact](https://github.com/sgl-project/sglang) |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on longest-prefix-match scheduling plus radix-tree lru eviction. The reported evaluation context includes Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Longest-prefix-match scheduling plus radix-tree LRU eviction.

## PLEX mapping

Canonical `S+E`; public implementation.

## Datasets and evaluation workloads

- Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, et al. “SGLang: Efficient Execution of Structured Language Model Programs.” NeurIPS 2024, 2024. https://doi.org/10.48550/arxiv.2312.07104.

## Sources

- [Primary paper](https://arxiv.org/abs/2312.07104)
- [OpenAlex record](https://openalex.org/W4389755588)
- [Artifact](https://github.com/sgl-project/sglang)
