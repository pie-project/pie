# Llumnix

> Canonical title: **Llumnix: Dynamic Scheduling for Large Language Model Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2406.03243](https://arxiv.org/abs/2406.03243) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin |
| Institutions / group context | Shanghai Jiao Tong University, Microsoft Research |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2024); author affiliations include Shanghai Jiao Tong University, Microsoft Research; a public implementation or artifact is linked. |
| Citation count | 4 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2406.03243 |
| arXiv | 2406.03243 |
| Artifact | [Public artifact](https://github.com/AlibabaPAI/llumnix) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. The proposed policy centers on virtual-usage-driven live rescheduling. The reported evaluation context includes ShareGPT-derived heterogeneous request traces, Priority, fragmentation, load-balancing, and autoscaling scenarios.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Virtual-usage-driven live rescheduling.

## PLEX mapping

`R+B+F`; canonical rebalance paper.

## Datasets and evaluation workloads

- ShareGPT-derived heterogeneous request traces
- Priority, fragmentation, load-balancing, and autoscaling scenarios

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, et al. “Llumnix: Dynamic Scheduling for Large Language Model Serving.” OSDI 2024, 2024. https://doi.org/10.48550/arxiv.2406.03243.

## Sources

- [Primary paper](https://arxiv.org/abs/2406.03243)
- [OpenAlex record](https://openalex.org/W4399453677)
- [Artifact](https://github.com/AlibabaPAI/llumnix)
