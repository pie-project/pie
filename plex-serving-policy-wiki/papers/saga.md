# SAGA

> Canonical title: **SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2605.00528](https://arxiv.org/abs/2605.00528) |
| Venue / status | arXiv preprint, 2026 |
| Year | 2026 |
| Authors | Dongxin Guo, Jikun Wu, Siu Ming Yiu |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2605.00528 |
| Artifact | No official public artifact confirmed |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on workflow-aware ttl, agent-level fairness, cache-local work stealing. The reported evaluation context includes SWE-bench.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Workflow-aware TTL, agent-level fairness, cache-local work stealing.

## PLEX mapping

`R+S+E+B`; holdout until artifact/source maturity improves.

## Datasets and evaluation workloads

- SWE-bench

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_saga`
- Operations: `route`, `schedule`, `cache`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/saga/metadata.json`](../../tests/policies/replications/saga/metadata.json)
- Deferred mechanics: physical work stealing
<!-- plex-v0.6-replication:end -->

## Suggested citation

Dongxin Guo, Jikun Wu, Siu Ming Yiu. “SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters.” arXiv preprint, 2026, 2026. https://arxiv.org/abs/2605.00528.

## Sources

- [Primary paper](https://arxiv.org/abs/2605.00528)
- [OpenAlex record](https://openalex.org/W7160329734)
