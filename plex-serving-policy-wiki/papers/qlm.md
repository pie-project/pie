# QLM

> Canonical title: **Queue Management for SLO-Oriented Large Language Model Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2407.00047](https://arxiv.org/abs/2407.00047) |
| Venue / status | SoCC 2024 |
| Year | 2024 |
| Authors | Archit Patke, Dhemath Reddy, Saurabh Jha, Haoran Qiu, Christian Pinto, Chandra Narayanaswami, Zbigniew Kalbarczyk, Ravishankar K. Iyer |
| Institutions / group context | University of Illinois Urbana-Champaign, IBM Research, IBM Research - Ireland |
| Reputation evidence | strong publication signal from a selective systems/ML venue (SoCC 2024); author affiliations include University of Illinois Urbana-Champaign, IBM Research, IBM Research - Ireland; no official public artifact was confirmed. |
| Citation count | 11 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.1145/3698038.3698523 |
| arXiv | 2407.00047 |
| Artifact | No official public artifact confirmed |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on slo virtual queues and global queue operations. The reported evaluation context includes ShareGPT, Production-derived interactive and batch SLO configurations.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

SLO virtual queues and global queue operations.

## PLEX mapping

Direct multi-hook case.

## Datasets and evaluation workloads

- ShareGPT
- Production-derived interactive and batch SLO configurations

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_qlm`
- Operations: `admit`, `route`, `schedule`, `feedback`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/qlm/metadata.json`](../../tests/policies/replications/qlm/metadata.json)
- Deferred mechanics: None
<!-- plex-v0.6-replication:end -->

## Suggested citation

Archit Patke, Dhemath Reddy, Saurabh Jha, Haoran Qiu, Christian Pinto, Chandra Narayanaswami, et al. “Queue Management for SLO-Oriented Large Language Model Serving.” SoCC 2024, 2024. https://doi.org/10.1145/3698038.3698523.

## Sources

- [Primary paper](https://arxiv.org/abs/2407.00047)
- [OpenAlex record](https://openalex.org/W4404386015)
