# InferCept

> Canonical title: **InferCept: Efficient Intercept Support for Augmented Large Language Model Inference**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2402.01869](https://arxiv.org/abs/2402.01869) |
| Venue / status | ICML 2024 |
| Year | 2024 |
| Authors | Reyna Abhyankar, Zijian He, Vikranth Srivatsa, Hao Zhang, Yiying Zhang |
| Institutions / group context | UC San Diego WukLab |
| Reputation evidence | strong publication signal from a selective systems/ML venue (ICML 2024); author affiliations include UC San Diego WukLab; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2402.01869 |
| Artifact | [Public artifact](https://github.com/WukLab/InferCept) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on preserve/swap/discard intercepted kv state. The reported evaluation context includes GSM8K-XL, HotpotQA / Wikipedia QA, ALFWorld, ShareGPT, among other workloads.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Preserve/swap/discard intercepted KV state.

## PLEX mapping

Direct `S+E+F` replica.

## Datasets and evaluation workloads

- GSM8K-XL
- HotpotQA / Wikipedia QA
- ALFWorld
- ShareGPT
- Image-generation and text-to-speech API workloads

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_infercept`
- Operations: `schedule`, `cache`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/infercept/metadata.json`](../../tests/policies/replications/infercept/metadata.json)
- Deferred mechanics: physical swap implementation
<!-- plex-v0.6-replication:end -->

## Suggested citation

Reyna Abhyankar, Zijian He, Vikranth Srivatsa, Hao Zhang, Yiying Zhang. “InferCept: Efficient Intercept Support for Augmented Large Language Model Inference.” ICML 2024, 2024. https://arxiv.org/abs/2402.01869.

## Sources

- [Primary paper](https://arxiv.org/abs/2402.01869)
- [Artifact](https://github.com/WukLab/InferCept)
