# Chameleon

> Canonical title: **Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2411.17741](https://arxiv.org/abs/2411.17741) |
| Venue / status | MICRO 2025 |
| Year | 2025 |
| Authors | Nikoleta Iliakopoulou, Jovan Stojkovic, Chloe Alverti, Tianyin Xu, Hubertus Franke, Josep Torrellas |
| Institutions / group context | University of Illinois Urbana-Champaign, IBM Research |
| Reputation evidence | strong publication signal from a selective systems/ML venue (MICRO 2025); author affiliations include University of Illinois Urbana-Champaign, IBM Research; no official public artifact was confirmed. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2411.17741 |
| arXiv | 2411.17741 |
| Artifact | No official public artifact confirmed |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on weighted-size queues, per-queue token quotas, bypass, adapter caching. The reported evaluation context includes Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Weighted-size queues, per-queue token quotas, bypass, adapter caching.

## PLEX mapping

`A+S+E`; strong LoRA case.

## Datasets and evaluation workloads

- Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_chameleon`
- Operations: `admit`, `schedule`, `cache`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/chameleon/metadata.json`](../../tests/policies/replications/chameleon/metadata.json)
- Deferred mechanics: None
<!-- plex-v0.6-replication:end -->

## Suggested citation

Nikoleta Iliakopoulou, Jovan Stojkovic, Chloe Alverti, Tianyin Xu, Hubertus Franke, Josep Torrellas. “Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments.” MICRO 2025, 2025. https://doi.org/10.48550/arxiv.2411.17741.

## Sources

- [Primary paper](https://arxiv.org/abs/2411.17741)
- [OpenAlex record](https://openalex.org/W4404990137)
