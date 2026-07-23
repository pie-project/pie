# InfiniGen

> Canonical title: **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/osdi24/presentation/lee](https://www.usenix.org/conference/osdi24/presentation/lee) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Wonbeom Lee, Jungi Lee, Jung-Hwan Seo, Jaewoong Sim |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2024); author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 1 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2406.19707 |
| arXiv | 2406.19707 |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on dynamic offload and speculative kv prefetch. The reported evaluation context includes Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Dynamic offload and speculative KV prefetch.

## PLEX mapping

Optional `P`; primarily mechanism.

## Datasets and evaluation workloads

- Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Wonbeom Lee, Jungi Lee, Jung-Hwan Seo, Jaewoong Sim. “InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management.” OSDI 2024, 2024. https://doi.org/10.48550/arxiv.2406.19707.

## Sources

- [Primary paper](https://www.usenix.org/conference/osdi24/presentation/lee)
- [OpenAlex record](https://openalex.org/W4400222625)
