# DualMap

> Canonical title: **DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2602.06502](https://arxiv.org/abs/2602.06502) |
| Venue / status | ICLR 2026 |
| Year | 2026 |
| Authors | Ying Yuan, Pengfei Zuo, Bo Wang, Zhangyu Chen, Zhipeng Tan, Zhou Yu |
| Institutions / group context | Huazhong University of Science and Technology, Huawei |
| Reputation evidence | strong publication signal from a selective systems/ML venue (ICLR 2026); author affiliations include Huazhong University of Science and Technology, Huawei; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2602.06502 |
| Artifact | [Public artifact](https://github.com/ASISys/DualMap) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. The proposed policy centers on two-choice cache-affinity routing and hotspot migration. The reported evaluation context includes Conversation and Tool/Agent shared-prefix workloads, Real-world-style vLLM request traces.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Two-choice cache-affinity routing and hotspot migration.

## PLEX mapping

Direct `R+B`.

## Datasets and evaluation workloads

- Conversation and Tool/Agent shared-prefix workloads
- Real-world-style vLLM request traces

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_dualmap`
- Operations: `route`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/dualmap/metadata.json`](../../tests/policies/replications/dualmap/metadata.json)
- Deferred mechanics: physical migration
<!-- plex-v0.6-replication:end -->

## Suggested citation

Ying Yuan, Pengfei Zuo, Bo Wang, Zhangyu Chen, Zhipeng Tan, Zhou Yu. “DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving.” ICLR 2026, 2026. https://arxiv.org/abs/2602.06502.

## Sources

- [Primary paper](https://arxiv.org/abs/2602.06502)
- [Artifact](https://github.com/ASISys/DualMap)
