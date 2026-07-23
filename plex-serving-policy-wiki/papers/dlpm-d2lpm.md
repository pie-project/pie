# DLPM/D²LPM

> Canonical title: **Locality-aware Fair Scheduling in LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2501.14312](https://arxiv.org/abs/2501.14312) |
| Venue / status | arXiv preprint, 2025 |
| Year | 2025 |
| Authors | Shiyi Cao, Yichuan Wang, Ziming Mao, Pin-Lun Hsu, Liangsheng Yin, Tian Xia, Dacheng Li, Shu Liu, Yineng Zhang, Yang Zhou, Ying Sheng, Joseph Gonzalez, Ion Stoica |
| Institutions / group context | UC Berkeley, LMSYS Org |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations include UC Berkeley, LMSYS Org; no official public artifact was confirmed. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2501.14312 |
| Artifact | No official public artifact confirmed |
| Corpus category | Scheduling, fairness, SLOs, and admission |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies request ordering, token allocation, fairness, admission, or latency-SLO control in shared inference serving. The proposed policy centers on prefix locality with bounded fair service. The reported evaluation context includes ShareGPT-derived workloads, Multi-turn and long-context QA traces, Synthetic dominant/victim-client fairness scenarios.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Prefix locality with bounded fair service.

## PLEX mapping

Direct `R+S+F`.

## Datasets and evaluation workloads

- ShareGPT-derived workloads
- Multi-turn and long-context QA traces
- Synthetic dominant/victim-client fairness scenarios

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_dlpm`
- Operations: `route`, `schedule`, `feedback`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/dlpm/metadata.json`](../../tests/policies/replications/dlpm/metadata.json)
- Deferred mechanics: None
<!-- plex-v0.6-replication:end -->

## Suggested citation

Shiyi Cao, Yichuan Wang, Ziming Mao, Pin-Lun Hsu, Liangsheng Yin, Tian Xia, et al. “Locality-aware Fair Scheduling in LLM Serving.” arXiv preprint, 2025, 2025. https://arxiv.org/abs/2501.14312.

## Sources

- [Primary paper](https://arxiv.org/abs/2501.14312)
