# GoodServe

> Canonical title: **GoodServe: Towards High-Goodput Serving of Agentic LLM Inferences over Heterogeneous Resources**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2605.16867](https://arxiv.org/abs/2605.16867) |
| Venue / status | arXiv preprint, 2026 |
| Year | 2026 |
| Authors | Boxiao Du, Boning Huangfu, Yizhou Luo, Chen Chen, Zijun Li, Minchen Yu, Xiaoyi Fan, Minyi Guo |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2605.16867 |
| Artifact | No official public artifact confirmed |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on “just-enough” heterogeneous gpu selection and risk-triggered migration. The reported evaluation context includes Popular agentic LLM workloads, Heterogeneous-GPU testbed and 512-instance simulations.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

“Just-enough” heterogeneous GPU selection and risk-triggered migration.

## PLEX mapping

`R+B+F`; prediction-heavy.

## Datasets and evaluation workloads

- Popular agentic LLM workloads
- Heterogeneous-GPU testbed and 512-instance simulations

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_goodserve`
- Operations: `route`, `feedback`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/goodserve/metadata.json`](../../tests/policies/replications/goodserve/metadata.json)
- Deferred mechanics: predictor training; provisioning
<!-- plex-v0.6-replication:end -->

## Suggested citation

Boxiao Du, Boning Huangfu, Yizhou Luo, Chen Chen, Zijun Li, Minchen Yu, et al. “GoodServe: Towards High-Goodput Serving of Agentic LLM Inferences over Heterogeneous Resources.” arXiv preprint, 2026, 2026. https://arxiv.org/abs/2605.16867.

## Sources

- [Primary paper](https://arxiv.org/abs/2605.16867)
- [OpenAlex record](https://openalex.org/W7161915257)
