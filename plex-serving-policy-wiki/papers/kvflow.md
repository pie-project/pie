# KVFlow

> Canonical title: **KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2507.07400](https://arxiv.org/abs/2507.07400) |
| Venue / status | NeurIPS 2025 |
| Year | 2025 |
| Authors | Zaifeng Pan, Ajjkumar Patel, Zhengding Hu, Yipeng Shen, Yue Guan, Wan-Lu Li, Lianhui Qin, Yida Wang, Yufei Ding |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | strong publication signal from a selective systems/ML venue (NeurIPS 2025); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2507.07400 |
| Artifact | [Public artifact](https://github.com/PanZaifeng/KVFlow) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on steps-to-execution eviction and overlapped prefetch. The reported evaluation context includes Multi-agent workflow DAGs, including PEER-style cyclic/sequential workflows, SGLang-based concurrent workflow traces.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Steps-to-execution eviction and overlapped prefetch.

## PLEX mapping

Existing direct replica plus optional `P`.

## Datasets and evaluation workloads

- Multi-agent workflow DAGs, including PEER-style cyclic/sequential workflows
- SGLang-based concurrent workflow traces

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_kvflow`
- Operations: `schedule`, `cache`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/kvflow/metadata.json`](../../tests/policies/replications/kvflow/metadata.json)
- Deferred mechanics: CPU-GPU transfer overlap
<!-- plex-v0.6-replication:end -->

## Suggested citation

Zaifeng Pan, Ajjkumar Patel, Zhengding Hu, Yipeng Shen, Yue Guan, Wan-Lu Li, et al. “KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows.” NeurIPS 2025, 2025. https://arxiv.org/abs/2507.07400.

## Sources

- [Primary paper](https://arxiv.org/abs/2507.07400)
- [Artifact](https://github.com/PanZaifeng/KVFlow)
