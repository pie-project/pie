# LMetric

> Canonical title: **LMetric: Simple is Better - Multiplication May Be All You Need for LLM Request Scheduling**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2603.15202](https://arxiv.org/abs/2603.15202) |
| Venue / status | OSDI 2026 |
| Year | 2026 |
| Authors | Dingyan Zhang, Jinbo Han, Kaixi Zhang, Xingda Wei, Sijie Shen, Chenguang Fang, WenYuan Yu, Jingren Zhou, Rong Chen |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2026); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | 0 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | Not resolved |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/blitz-serving/blitz-router) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. Its central policy contribution is: Multiplicative cache/load score.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Multiplicative cache/load score.

## PLEX mapping

Direct `R`; strongest simple route example.

## Datasets and workloads

- Alibaba Bailian Qwen ChatBot, Agent, and Coder traces
- Kimi ToolAgent trace
- Azure LLM inference trace

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Dingyan Zhang, Jinbo Han, Kaixi Zhang, Xingda Wei, Sijie Shen, Chenguang Fang, et al.. “LMetric: Simple is Better - Multiplication May Be All You Need for LLM Request Scheduling.” OSDI 2026, 2026. https://arxiv.org/abs/2603.15202.

## Sources

- [Primary paper](https://arxiv.org/abs/2603.15202)
- [OpenAlex record](https://openalex.org/W7139148132)
- [Artifact](https://github.com/blitz-serving/blitz-router)
