# InferCept

> Canonical title: **InferCept: Efficient Intercept Support for Augmented Large Language Model Inference**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2402.01869](https://arxiv.org/abs/2402.01869) |
| Venue / status | ICML 2024 |
| Year | 2024 |
| Authors | Not reliably resolved |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (ICML 2024); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | Not resolved (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | Not resolved |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/WukLab/InferCept) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. Its central policy contribution is: Preserve/swap/discard intercepted KV state.

The primary source does not have an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Preserve/swap/discard intercepted KV state.

## PLEX mapping

Direct `S+E+F` replica.

## Datasets and workloads

- GSM8K-XL
- HotpotQA / Wikipedia QA
- ALFWorld
- ShareGPT
- Image-generation and text-to-speech API workloads

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Authors not resolved. “InferCept: Efficient Intercept Support for Augmented Large Language Model Inference.” ICML 2024, 2024. https://arxiv.org/abs/2402.01869.

## Sources

- [Primary paper](https://arxiv.org/abs/2402.01869)
- [Artifact](https://github.com/WukLab/InferCept)
