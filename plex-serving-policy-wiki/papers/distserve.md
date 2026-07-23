# DistServe

> Canonical title: **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang |
| Institutions / group context | Peking University, UC San Diego, StepFun |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2024); author affiliations include Peking University, UC San Diego, StepFun; a public implementation or artifact is linked. |
| Citation count | 15 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2401.09670 |
| arXiv | 2401.09670 |
| Artifact | [Public artifact](https://github.com/LLMServe/DistServe) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. The proposed policy centers on goodput-based p/d placement and provisioning. The reported evaluation context includes Multi-replica or heterogeneous-cluster trace-driven workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Goodput-based P/D placement and provisioning.

## PLEX mapping

Corpus/adapter stress; much is offline mechanism.

## Datasets and evaluation workloads

- Multi-replica or heterogeneous-cluster trace-driven workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, et al. “DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.” OSDI 2024, 2024. https://doi.org/10.48550/arxiv.2401.09670.

## Sources

- [Primary paper](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [OpenAlex record](https://openalex.org/W4391045969)
- [Artifact](https://github.com/LLMServe/DistServe)
