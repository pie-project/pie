# Helium

> Canonical title: **Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2603.16104](https://arxiv.org/abs/2603.16104) |
| Venue / status | arXiv preprint, 2026 |
| Year | 2026 |
| Authors | Noppanat Wadlom, Junyi Shen, Yao Lu |
| Institutions / group context | School of Computing, National University of Singapore, Singapore, Singapore |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations include School of Computing, National University of Singapore, Singapore, Singapore; a public implementation or artifact is linked. |
| OpenAlex cited-by count | 0 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.1145/3802046 |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/mlsys-io/helium_demo) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. Its central policy contribution is: Cache-aware critical-path operator scheduling and proactive warming.

The primary source does not have an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Cache-aware critical-path operator scheduling and proactive warming.

## PLEX mapping

Existing `S`; prefetch is optional/mechanical.

## Datasets and workloads

- Agentic workflow DAGs and repeated-batch workflow executions reported by the paper

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Noppanat Wadlom, Junyi Shen, Yao Lu. “Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective.” arXiv preprint, 2026, 2026. https://doi.org/10.1145/3802046.

## Sources

- [Primary paper](https://arxiv.org/abs/2603.16104)
- [OpenAlex record](https://doi.org/10.1145/3802046)
- [Artifact](https://github.com/mlsys-io/helium_demo)
