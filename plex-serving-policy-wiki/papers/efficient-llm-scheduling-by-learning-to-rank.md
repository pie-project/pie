# Efficient LLM Scheduling by Learning to Rank

> Canonical title: **Efficient LLM Scheduling by Learning to Rank**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2408.15792](https://arxiv.org/abs/2408.15792) |
| Venue / status | arXiv preprint |
| Year | 2024 |
| Authors | Yichao Fu, Siqi Zhu, Runlong Su, Aurick Qiao, Ion Stoica, Hao Zhang |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| Citation count | 1 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2408.15792 |
| arXiv | 2408.15792 |
| Artifact | [Public artifact](https://github.com/hao-ai-lab/vllm-ltr) |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on relative length ranking approximating sjf. The reported evaluation context includes Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Relative length ranking approximating SJF.

## PLEX mapping

Direct `S`; [code](https://github.com/hao-ai-lab/vllm-ltr).

## Datasets and evaluation workloads

- Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Yichao Fu, Siqi Zhu, Runlong Su, Aurick Qiao, Ion Stoica, Hao Zhang. “Efficient LLM Scheduling by Learning to Rank.” arXiv preprint, 2024. https://doi.org/10.48550/arxiv.2408.15792.

## Sources

- [Primary paper](https://arxiv.org/abs/2408.15792)
- [OpenAlex record](https://openalex.org/W4402706052)
- [Artifact](https://github.com/hao-ai-lab/vllm-ltr)
