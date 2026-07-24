# LMetric

> Canonical title: **Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2603.15202](https://arxiv.org/abs/2603.15202) |
| Venue / status | OSDI 2026 |
| Year | 2026 |
| Authors | Dingyan Zhang, Jinbo Han, Kaixi Zhang, Xingda Wei, Sijie Shen, Chenguang Fang, WenYuan Yu, Jingren Zhou, Rong Chen |
| Institutions / group context | Shanghai Jiao Tong University, Alibaba Cloud |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2026); author affiliations include Shanghai Jiao Tong University, Alibaba Cloud; a public implementation or artifact is linked. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2603.15202 |
| Artifact | [Public artifact](https://github.com/blitz-serving/blitz-router) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. The proposed policy centers on multiplicative cache/load score. The reported evaluation context includes Alibaba Bailian Qwen ChatBot, Agent, and Coder traces and the Kimi ToolAgent trace.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Multiplicative cache/load score.

## PLEX mapping

Direct `R`; strongest simple route example.

## Datasets and evaluation workloads

- Alibaba Bailian Qwen ChatBot, Agent, and Coder traces
- Kimi ToolAgent trace

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_lmetric`
- Operations: `route`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/lmetric/metadata.json`](../../tests/policies/replications/lmetric/metadata.json)
- Deferred mechanics: None
<!-- plex-v0.6-replication:end -->

## Suggested citation

Dingyan Zhang, Jinbo Han, Kaixi Zhang, Xingda Wei, Sijie Shen, Chenguang Fang, et al. “Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling.” OSDI 2026, 2026. https://arxiv.org/abs/2603.15202.

## Sources

- [Primary paper](https://arxiv.org/abs/2603.15202)
- [OpenAlex record](https://openalex.org/W7139148132)
- [Artifact](https://github.com/blitz-serving/blitz-router)
