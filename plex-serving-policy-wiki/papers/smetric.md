# SMetric

> Canonical title: **SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2607.08565](https://arxiv.org/abs/2607.08565) |
| Venue / status | arXiv preprint, 2026 |
| Year | 2026 |
| Authors | Jiahao Wang, Kaizhan Lin, Kaixi Zhang, Jinbo Han, Xingda Wei, Sijie Shen, Chenguang Fang, W K Yu, Rong Chen, Haibo Chen |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2607.08565 |
| Artifact | No official public artifact confirmed |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on load-balance first turns, use cache affinity for follow-ups, migrate tail outliers. The reported evaluation context includes Two large-scale production agentic-serving traces, including Alibaba Bailian.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Load-balance first turns, use cache affinity for follow-ups, migrate tail outliers.

## PLEX mapping

`R+B`; highly relevant holdout.

## Datasets and evaluation workloads

- Two large-scale production agentic-serving traces, including Alibaba Bailian

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_smetric`
- Operations: `route`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/smetric/metadata.json`](../../tests/policies/replications/smetric/metadata.json)
- Deferred mechanics: physical migration
<!-- plex-v0.6-replication:end -->

## Suggested citation

Jiahao Wang, Kaizhan Lin, Kaixi Zhang, Jinbo Han, Xingda Wei, Sijie Shen, et al. “SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling.” arXiv preprint, 2026, 2026. https://arxiv.org/abs/2607.08565.

## Sources

- [Primary paper](https://arxiv.org/abs/2607.08565)
- [OpenAlex record](https://openalex.org/W7168054856)
