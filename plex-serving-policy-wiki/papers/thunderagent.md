# ThunderAgent

> Canonical title: **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2602.13692](https://arxiv.org/abs/2602.13692) |
| Venue / status | arXiv preprint, 2026 |
| Year | 2026 |
| Authors | Hao Kang, Ziyang Li, Weili Xu, Xinyu Yang, Yinfang Chen, Junxiong Wang, Beidi Chen, Tushar Krishna, Chenfeng Xu, Simran Arora |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2602.13692 |
| Artifact | [Public artifact](https://github.com/ThunderAgent-org/ThunderAgent) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on state-aware pausing, program migration, tool-resource lifecycle. The reported evaluation context includes SWE-Bench Lite / SWE-bench, OpenHands, SWE-Agent, ToolOrchestra, among other workloads.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

State-aware pausing, program migration, tool-resource lifecycle.

## PLEX mapping

`S+E+B+F`; strong coordinated case.

## Datasets and evaluation workloads

- SWE-Bench Lite / SWE-bench
- OpenHands
- SWE-Agent
- ToolOrchestra
- ScienceAgentBench

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_thunderagent`
- Operations: `schedule`, `cache`, `feedback`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/thunderagent/metadata.json`](../../tests/policies/replications/thunderagent/metadata.json)
- Deferred mechanics: physical pause and migration
<!-- plex-v0.6-replication:end -->

## Suggested citation

Hao Kang, Ziyang Li, Weili Xu, Xinyu Yang, Yinfang Chen, Junxiong Wang, et al. “ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System.” arXiv preprint, 2026, 2026. https://arxiv.org/abs/2602.13692.

## Sources

- [Primary paper](https://arxiv.org/abs/2602.13692)
- [OpenAlex record](https://openalex.org/W7130237524)
- [Artifact](https://github.com/ThunderAgent-org/ThunderAgent)
