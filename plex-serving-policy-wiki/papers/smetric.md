# SMetric

> Canonical title: **SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2607.08565](https://arxiv.org/abs/2607.08565) |
| Venue / status | arXiv (Cornell University) |
| Year | 2026 |
| Authors | Jiahao Wang, Kaizhan Lin, Kaixi Zhang, Jinbo Han, Xingda Wei, Sijie Shen, Chenguang Fang, W K Yu, Rong Chen, Haibo Chen |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; no official public artifact was confirmed. |
| OpenAlex cited-by count | 0 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | Not resolved |
| arXiv | Not resolved |
| Artifact | No official public artifact confirmed |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. Its central policy contribution is: Load-balance first turns, use cache affinity for follow-ups, migrate tail outliers.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Load-balance first turns, use cache affinity for follow-ups, migrate tail outliers.

## PLEX mapping

`R+B`; highly relevant holdout.

## Datasets and workloads

- Two large-scale production agentic-serving traces, including Alibaba Bailian

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Jiahao Wang, Kaizhan Lin, Kaixi Zhang, Jinbo Han, Xingda Wei, Sijie Shen, et al.. “SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling.” arXiv (Cornell University), 2026. https://arxiv.org/abs/2607.08565.

## Sources

- [Primary paper](https://arxiv.org/abs/2607.08565)
- [OpenAlex record](https://openalex.org/W7168054856)
