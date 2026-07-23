# HotPrefix

> Canonical title: **HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://doi.org/10.1145/3749168](https://doi.org/10.1145/3749168) |
| Venue / status | PACMMOD / SIGMOD 2026 |
| Year | 2026 |
| Authors | Yuhang Li, Rong Gu, Chengying Huan, Zhibin Wang, Renjie Yao, Chen Tian, Guihai Chen |
| Institutions / group context | Nanjing University |
| Reputation evidence | strong publication signal from a selective systems/ML venue (PACMMOD / SIGMOD 2026); author affiliations include Nanjing University; no official public artifact was confirmed. |
| Citation count | 5 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | Manual primary-source audit |
| DOI | https://doi.org/10.1145/3749168 |
| arXiv | Not resolved |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on hotness tracking, selective admission, cpu/gpu promotion. The reported evaluation context includes Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Hotness tracking, selective admission, CPU/GPU promotion.

## PLEX mapping

`E+P+F`; cache admission exposes the same gap.

## Datasets and evaluation workloads

- Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_hotprefix`
- Operations: `cache`, `feedback`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/hotprefix/metadata.json`](../../tests/policies/replications/hotprefix/metadata.json)
- Deferred mechanics: physical tier promotion
<!-- plex-v0.6-replication:end -->

## Suggested citation

Yuhang Li, Rong Gu, Chengying Huan, Zhibin Wang, Renjie Yao, Chen Tian, et al. “HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems.” PACMMOD / SIGMOD 2026, 2026. https://doi.org/10.1145/3749168.

## Sources

- [Primary paper](https://doi.org/10.1145/3749168)
- [OpenAlex record](https://openalex.org/W7082968261)
