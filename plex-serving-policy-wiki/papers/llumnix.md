# Llumnix

> Canonical title: **Llumnix: Dynamic Scheduling for Large Language Model Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2406.03243](https://arxiv.org/abs/2406.03243) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2024); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | 4 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.48550/arxiv.2406.03243 |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/AlibabaPAI/llumnix) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. Its central policy contribution is: Virtual-usage-driven live rescheduling.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Virtual-usage-driven live rescheduling.

## PLEX mapping

`R+B+F`; canonical rebalance paper.

## Datasets and workloads

- ShareGPT-derived heterogeneous request traces
- Priority, fragmentation, load-balancing, and autoscaling scenarios

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, et al.. “Llumnix: Dynamic Scheduling for Large Language Model Serving.” OSDI 2024, 2024. https://doi.org/10.48550/arxiv.2406.03243.

## Sources

- [Primary paper](https://arxiv.org/abs/2406.03243)
- [OpenAlex record](https://openalex.org/W4399453677)
- [Artifact](https://github.com/AlibabaPAI/llumnix)
