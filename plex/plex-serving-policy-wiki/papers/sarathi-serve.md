# Sarathi-Serve

> Canonical title: **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/osdi24/presentation/agrawal](https://www.usenix.org/conference/osdi24/presentation/agrawal) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | strong publication signal from a selective systems/ML venue (OSDI 2024); author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 15 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2403.02310 |
| arXiv | 2403.02310 |
| Artifact | No official public artifact confirmed |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on stall-free chunked-prefill token budgeting. The reported evaluation context includes Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Stall-free chunked-prefill token budgeting.

## PLEX mapping

`S`; mechanics-heavy but useful baseline.

## Datasets and evaluation workloads

- Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, et al. “Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.” OSDI 2024, 2024. https://doi.org/10.48550/arxiv.2403.02310.

## Sources

- [Primary paper](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- [OpenAlex record](https://openalex.org/W4392489911)
