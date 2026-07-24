# FastServe

> Canonical title: **FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang) |
| Venue / status | NSDI 2026; arXiv first posted 2023 |
| Year | 2026 |
| Authors | Bingyang Wu, Yinmin Zhong, Zili Zhang, Shengyu Liu, Fangyue Liu, Yuanhang Sun, Gang Huang, Xuanzhe Liu, Xin Jin |
| Institutions / group context | Peking University |
| Reputation evidence | strong publication signal from a selective systems/ML venue (NSDI 2026; arXiv first posted 2023); author affiliations include Peking University; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2305.05920 |
| Artifact | [Public artifact](https://github.com/LLMServe/FastServe) |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on skip-join mlfq, iteration preemption, starvation promotion. The reported evaluation context includes Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered.

An abstract was not available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Skip-join MLFQ, iteration preemption, starvation promotion.

## PLEX mapping

Direct `S`; [code](https://github.com/LLMServe/FastServe).

## Datasets and evaluation workloads

- Synthetic and trace-driven shared-serving workloads; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

## Suggested citation

Bingyang Wu, Yinmin Zhong, Zili Zhang, Shengyu Liu, Fangyue Liu, Yuanhang Sun, et al. “FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference.” NSDI 2026; arXiv first posted 2023, 2026. https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang.

## Sources

- [Primary paper](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)
- [Artifact](https://github.com/LLMServe/FastServe)
