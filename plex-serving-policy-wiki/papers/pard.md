# PARD

> Canonical title: **PARD: Enhancing Goodput for Inference Pipeline via Proactive Request Dropping**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2602.08747](https://arxiv.org/abs/2602.08747) |
| Venue / status | EuroSys 2026 |
| Year | 2026 |
| Authors | Zhixin Zhao, Yitao Hu, Simin Chen, Mingfang Ji, Wei Yang, Yuhao Zhang, Laiping Zhao, Wenxin Li, Xiulong Liu, Wenyu Qu, Hao Wang |
| Institutions / group context | Tianjin University, The University of Texas at Dallas, Stevens Institute of Technology |
| Reputation evidence | strong publication signal from a selective systems/ML venue (EuroSys 2026); author affiliations include Tianjin University, The University of Texas at Dallas, Stevens Institute of Technology; no official public artifact was confirmed. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2602.08747 |
| Artifact | No official public artifact confirmed |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on proactively drop work using upstream elapsed time and downstream latency distributions. The reported evaluation context includes Wiki-derived traces, Azure Functions trace, Multi-module inference-pipeline workloads.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Proactively drop work using upstream elapsed time and downstream latency distributions.

## PLEX mapping

Important boundary test: mid-lifecycle drop is not cleanly `admit`-once.

## Datasets and evaluation workloads

- Wiki-derived traces
- Azure Functions trace
- Multi-module inference-pipeline workloads

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_pard`
- Operations: `schedule`, `feedback`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/pard/metadata.json`](../../tests/policies/replications/pard/metadata.json)
- Deferred mechanics: latency-distribution estimator
<!-- plex-v0.6-replication:end -->

## Suggested citation

Zhixin Zhao, Yitao Hu, Simin Chen, Mingfang Ji, Wei Yang, Yuhao Zhang, et al. “PARD: Enhancing Goodput for Inference Pipeline via Proactive Request Dropping.” EuroSys 2026, 2026. https://arxiv.org/abs/2602.08747.

## Sources

- [Primary paper](https://arxiv.org/abs/2602.08747)
- [OpenAlex record](https://openalex.org/W7128556383)
