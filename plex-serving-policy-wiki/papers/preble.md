# Preble

> Canonical title: **Preble: Efficient Distributed Prompt Scheduling for LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2407.00023](https://arxiv.org/abs/2407.00023) |
| Venue / status | ICLR 2025 |
| Year | 2025 |
| Authors | Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (ICLR 2025); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | 1 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.48550/arxiv.2407.00023 |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/WukLab/preble) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. Its central policy contribution is: Prefix reuse versus load/eviction-cost routing.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Prefix reuse versus load/eviction-cost routing.

## PLEX mapping

Existing direct `R`.

## Datasets and workloads

- Long shared-prompt and prefix-reuse workloads
- Distributed prompt-serving traces used by the Preble artifact

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang. “Preble: Efficient Distributed Prompt Scheduling for LLM Serving.” ICLR 2025, 2025. https://doi.org/10.48550/arxiv.2407.00023.

## Sources

- [Primary paper](https://arxiv.org/abs/2407.00023)
- [OpenAlex record](https://openalex.org/W4400267057)
- [Artifact](https://github.com/WukLab/preble)
