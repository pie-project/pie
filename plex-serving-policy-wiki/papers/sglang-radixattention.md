# SGLang / RadixAttention

> Canonical title: **SGLang: Efficient Execution of Structured Language Model Programs**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104) |
| Venue / status | NeurIPS 2024 |
| Year | 2024 |
| Authors | Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (NeurIPS 2024); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | 34 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.52202/079017-2000 |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/sgl-project/sglang) |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. Its central policy contribution is: Longest-prefix-match scheduling plus radix-tree LRU eviction.

The primary source does not have an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Longest-prefix-match scheduling plus radix-tree LRU eviction.

## PLEX mapping

Canonical `S+E`; public implementation.

## Datasets and workloads

- No named dataset was reliably identified from accessible metadata.

No named dataset was reliably identified from the accessible metadata; consult the evaluation section.

## Suggested citation

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Yu, et al.. “SGLang: Efficient Execution of Structured Language Model Programs.” NeurIPS 2024, 2024. https://doi.org/10.52202/079017-2000.

## Sources

- [Primary paper](https://arxiv.org/abs/2312.07104)
- [OpenAlex record](https://openalex.org/W4415797413)
- [Artifact](https://github.com/sgl-project/sglang)
