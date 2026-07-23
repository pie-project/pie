# Certaindex / Dynasor

> Canonical title: **Efficiently Scaling LLM Reasoning with Certaindex**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2412.20993](https://arxiv.org/abs/2412.20993) |
| Venue / status | arXiv preprint, 2024; later revisions in 2025 |
| Year | 2024 |
| Authors | Yichao Fu, Junda Chen, Siqi Zhu, Zhongheng Fu, Zhongdongming Dai, Yonghao Zhuang, Yian Ma, Aurick Qiao, Tajana Rosing, Ion Stoica, Hao Zhang |
| Institutions / group context | UC San Diego Hao AI Lab |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations include UC San Diego Hao AI Lab; a public implementation or artifact is linked. |
| Citation count | 1 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2412.20993 |
| arXiv | 2412.20993 |
| Artifact | [Public artifact](https://github.com/hao-ai-lab/Dynasor) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on progress-aware reasoning allocation and early stop. The reported evaluation context includes Mathematical and code-reasoning workloads used to evaluate self-consistency and search-style reasoning programs.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Progress-aware reasoning allocation and early stop.

## PLEX mapping

Direct `S+F`; best bundle/task candidate.

## Datasets and evaluation workloads

- Mathematical and code-reasoning workloads used to evaluate self-consistency and search-style reasoning programs

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_dynasor`
- Operations: `schedule`, `feedback`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/dynasor/metadata.json`](../../tests/policies/replications/dynasor/metadata.json)
- Deferred mechanics: predictor training
<!-- plex-v0.6-replication:end -->

## Suggested citation

Yichao Fu, Junda Chen, Siqi Zhu, Zhongheng Fu, Zhongdongming Dai, Yonghao Zhuang, et al. “Efficiently Scaling LLM Reasoning with Certaindex.” arXiv preprint, 2024; later revisions in 2025, 2024. https://doi.org/10.48550/arxiv.2412.20993.

## Sources

- [Primary paper](https://arxiv.org/abs/2412.20993)
- [OpenAlex record](https://openalex.org/W4405957588)
- [Artifact](https://github.com/hao-ai-lab/Dynasor)
