# PARD

> Canonical title: **PARDES**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2602.08747](https://arxiv.org/abs/2602.08747) |
| Venue / status | EuroSys 2026 |
| Year | 2026 |
| Authors | Opher Etzion |
| Institutions / group context | Technion – Israel Institute of Technology |
| Reputation evidence | strong publication signal from a selective systems/ML venue (EuroSys 2026); author affiliations include Technion – Israel Institute of Technology; no official public artifact was confirmed. |
| OpenAlex cited-by count | 34 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.1145/156883.156884 |
| arXiv | Not resolved |
| Artifact | No official public artifact confirmed |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. Its central policy contribution is: Proactively drop work using upstream elapsed time and downstream latency distributions.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Proactively drop work using upstream elapsed time and downstream latency distributions.

## PLEX mapping

Important boundary test: mid-lifecycle drop is not cleanly `admit`-once.

## Datasets and workloads

- Wiki-derived traces
- Azure Functions trace
- Multi-module inference-pipeline workloads

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Opher Etzion. “PARDES.” EuroSys 2026, 2026. https://doi.org/10.1145/156883.156884.

## Sources

- [Primary paper](https://arxiv.org/abs/2602.08747)
- [OpenAlex record](https://openalex.org/W2037356910)
