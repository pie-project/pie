# dLoRA

> Canonical title: **dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/osdi24/presentation/wu-bingyang](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Bingyang Wu, Ruidong Zhu, Zili Zhang, Peng Sun, Xuanzhe Liu, Xin Jin |
| Institutions / group context | Peking University, Shanghai AI Laboratory |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2024); author affiliations include Peking University, Shanghai AI Laboratory; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/LLMServe/dLoRA-artifact) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. The proposed policy centers on request/adapter co-placement and migration. The reported evaluation context includes Multi-replica or heterogeneous-cluster trace-driven workloads; no named public dataset was reliably recovered.

An abstract was not available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Request/adapter co-placement and migration.

## PLEX mapping

`R+B`; excellent LoRA stress case.

## Datasets and evaluation workloads

- Multi-replica or heterogeneous-cluster trace-driven workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Bingyang Wu, Ruidong Zhu, Zili Zhang, Peng Sun, Xuanzhe Liu, Xin Jin. “dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving.” OSDI 2024, 2024. https://www.usenix.org/conference/osdi24/presentation/wu-bingyang.

## Sources

- [Primary paper](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
- [Artifact](https://github.com/LLMServe/dLoRA-artifact)
