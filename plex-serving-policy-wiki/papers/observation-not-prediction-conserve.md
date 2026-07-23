# Observation, Not Prediction / ConServe

> Canonical title: **Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2606.01839](https://arxiv.org/abs/2606.01839) |
| Venue / status | arXiv preprint |
| Year | 2026 |
| Authors | Jianru Ding, Ryien Hosseini, Pouya Mahdi Gholami, Mingyuan Xiang, Henry Hoffmann |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2606.01839 |
| Artifact | No official public artifact confirmed |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on place one heavy initial prefill and pin the conversation tail. The reported evaluation context includes SWE-bench-derived agentic workloads, Qwen3-0.6B experiments on NVIDIA A40 clusters.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Place one heavy initial prefill and pin the conversation tail.

## PLEX mapping

Simple `R`; strong logical-request example.

## Datasets and evaluation workloads

- SWE-bench-derived agentic workloads
- Qwen3-0.6B experiments on NVIDIA A40 clusters

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_conserve`
- Operations: `route`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/conserve/metadata.json`](../../tests/policies/replications/conserve/metadata.json)
- Deferred mechanics: None
<!-- plex-v0.6-replication:end -->

## Suggested citation

Jianru Ding, Ryien Hosseini, Pouya Mahdi Gholami, Mingyuan Xiang, Henry Hoffmann. “Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving.” arXiv preprint, 2026. https://arxiv.org/abs/2606.01839.

## Sources

- [Primary paper](https://arxiv.org/abs/2606.01839)
- [OpenAlex record](https://openalex.org/W7163594946)
