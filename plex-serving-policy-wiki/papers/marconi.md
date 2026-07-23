# Marconi

> Canonical title: **Marconi: Prefix Caching for the Era of Hybrid LLMs**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2411.19379](https://arxiv.org/abs/2411.19379) |
| Venue / status | MLSys 2025 |
| Year | 2025 |
| Authors | Rui Pan, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zancato, Tri Dao, Yida Wang, Ravi Netravali |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | strong publication signal from a selective systems/ML venue (MLSys 2025); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2411.19379 |
| Artifact | [Public artifact](https://github.com/ruipeterpan/marconi) |
| Corpus category | Residency, cache admission, eviction, and prefetch |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies which KV, model, or session state should remain resident, be evicted, or be prefetched. The proposed policy centers on reuse-likelihood cache admission and flop-per-byte eviction for hybrid models. The reported evaluation context includes Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Reuse-likelihood cache admission and FLOP-per-byte eviction for hybrid models.

## PLEX mapping

`E+F`; cache admission exposes a surface gap.

## Datasets and evaluation workloads

- Prefix/cache-pressure and memory-tier workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_marconi`
- Operations: `cache`, `feedback`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/marconi/metadata.json`](../../tests/policies/replications/marconi/metadata.json)
- Deferred mechanics: hybrid-model kernel integration
<!-- plex-v0.6-replication:end -->

## Suggested citation

Rui Pan, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zancato, Tri Dao, et al. “Marconi: Prefix Caching for the Era of Hybrid LLMs.” MLSys 2025, 2025. https://arxiv.org/abs/2411.19379.

## Sources

- [Primary paper](https://arxiv.org/abs/2411.19379)
- [Artifact](https://github.com/ruipeterpan/marconi)
