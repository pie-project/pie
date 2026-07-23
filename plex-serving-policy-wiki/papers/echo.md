# ECHO

> Canonical title: **ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/osdi26/presentation/liu-guangda](https://www.usenix.org/conference/osdi26/presentation/liu-guangda) |
| Venue / status | OSDI 2026 |
| Year | 2026 |
| Authors | Guangda Liu, Wenhao Chen, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, Quan Chen, Shixuan Sun, Jieru Zhao, Minyi Guo |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2026); author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | Not resolved |
| Artifact | No official public artifact confirmed |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on lossless prefetching for sparse-attention kv offload. The reported evaluation context includes Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered.

An abstract was not available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Lossless prefetching for sparse-attention KV offload.

## PLEX mapping

Optional `P`; mechanism-heavy.

## Datasets and evaluation workloads

- Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Guangda Liu, Wenhao Chen, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, et al. “ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs.” OSDI 2026, 2026. https://www.usenix.org/conference/osdi26/presentation/liu-guangda.

## Sources

- [Primary paper](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
