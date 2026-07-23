# VTC

> Canonical title: **Fairness in Serving Large Language Models**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2401.00588](https://arxiv.org/abs/2401.00588) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica |
| Institutions / group context | UC Berkeley Sky Computing Lab |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2024); author affiliations include UC Berkeley Sky Computing Lab; a public implementation or artifact is linked. |
| Citation count | 4 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2401.00588 |
| arXiv | 2401.00588 |
| Artifact | [Public artifact](https://github.com/Ying1123/VTC-artifact) |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on provable token-cost fairness. The reported evaluation context includes ShareGPT-derived request traces, Synthetic multi-client fairness traces.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Provable token-cost fairness.

## PLEX mapping

Direct `S+F`; must-have baseline.

## Datasets and evaluation workloads

- ShareGPT-derived request traces
- Synthetic multi-client fairness traces

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_vtc`
- Operations: `schedule`, `feedback`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/vtc/metadata.json`](../../tests/policies/replications/vtc/metadata.json)
- Deferred mechanics: None
<!-- plex-v0.6-replication:end -->

## Suggested citation

Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, et al. “Fairness in Serving Large Language Models.” OSDI 2024, 2024. https://doi.org/10.48550/arxiv.2401.00588.

## Sources

- [Primary paper](https://arxiv.org/abs/2401.00588)
- [OpenAlex record](https://openalex.org/W4390529219)
- [Artifact](https://github.com/Ying1123/VTC-artifact)
