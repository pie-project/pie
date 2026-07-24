# HCache

> Canonical title: **Fast State Restoration in LLM Serving with HCache**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2410.05004](https://arxiv.org/abs/2410.05004) |
| Venue / status | EuroSys 2025 |
| Year | 2025 |
| Authors | S.Y. Gao, Youmin Chen, Jiwu Shu |
| Institutions / group context | Tsinghua University |
| Reputation evidence | strong publication signal from a selective systems/ML venue (EuroSys 2025); author affiliations include Tsinghua University; no official public artifact was confirmed. |
| Citation count | 13 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.1145/3689031.3696072 |
| arXiv | 2410.05004 |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on restore state by choosing compute, load, or both. The reported evaluation context includes Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Restore state by choosing compute, load, or both.

## PLEX mapping

`E+P`; action/mechanics-heavy.

## Datasets and evaluation workloads

- Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

S.Y. Gao, Youmin Chen, Jiwu Shu. “Fast State Restoration in LLM Serving with HCache.” EuroSys 2025, 2025. https://doi.org/10.1145/3689031.3696072.

## Sources

- [Primary paper](https://arxiv.org/abs/2410.05004)
- [OpenAlex record](https://openalex.org/W4408844835)
