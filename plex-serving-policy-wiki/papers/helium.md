# Helium

> Canonical title: **Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2603.16104](https://arxiv.org/abs/2603.16104) |
| Venue / status | arXiv preprint, 2026 |
| Year | 2026 |
| Authors | Noppanat Wadlom, Junyi Shen, Yao Lu |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2603.16104 |
| Artifact | [Public artifact](https://github.com/mlsys-io/helium_demo) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on cache-aware critical-path operator scheduling and proactive warming. The reported evaluation context includes Agentic workflow DAGs and repeated-batch workflow executions reported by the paper.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Cache-aware critical-path operator scheduling and proactive warming.

## PLEX mapping

Existing `S`; prefetch is optional/mechanical.

## Datasets and evaluation workloads

- Agentic workflow DAGs and repeated-batch workflow executions reported by the paper

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_helium`
- Operations: `schedule`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/helium/metadata.json`](../../tests/policies/replications/helium/metadata.json)
- Deferred mechanics: query-plan rewrite; proactive cache warming
<!-- plex-v0.6-replication:end -->

## Suggested citation

Noppanat Wadlom, Junyi Shen, Yao Lu. “Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective.” arXiv preprint, 2026, 2026. https://arxiv.org/abs/2603.16104.

## Sources

- [Primary paper](https://arxiv.org/abs/2603.16104)
- [Artifact](https://github.com/mlsys-io/helium_demo)
