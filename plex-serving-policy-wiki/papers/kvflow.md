# KVFlow

> Canonical title: **KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2507.07400](https://arxiv.org/abs/2507.07400) |
| Venue / status | NeurIPS 2025 |
| Year | 2025 |
| Authors | Not reliably resolved |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (NeurIPS 2025); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | Not resolved (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | Not resolved |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/PanZaifeng/KVFlow) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. Its central policy contribution is: Steps-to-execution eviction and overlapped prefetch.

The primary source does not have an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Steps-to-execution eviction and overlapped prefetch.

## PLEX mapping

Existing direct replica plus optional `P`.

## Datasets and workloads

- Multi-agent workflow DAGs, including PEER-style cyclic/sequential workflows
- SGLang-based concurrent workflow traces

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Authors not resolved. “KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows.” NeurIPS 2025, 2025. https://arxiv.org/abs/2507.07400.

## Sources

- [Primary paper](https://arxiv.org/abs/2507.07400)
- [Artifact](https://github.com/PanZaifeng/KVFlow)
