# Pythia

> Canonical title: **Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2604.25899](https://arxiv.org/abs/2604.25899) |
| Venue / status | arXiv preprint, 2026 |
| Year | 2026 |
| Authors | Shan Yu, Junyi Shu, Yuanjiang Ni, Kun Qian, Xue Li, Yang Wang, Jinyuan Zhang, Ziyi Xu, Shuo Yang, Lingjun Zhu, Ennan Zhai, Qingda Lu, Jiarong Xing, Youyou Lu, Xin Jin, Xuanzhe Liu, Harry Xu |
| Institutions / group context | Not reliably resolved; venue and artifact evidence used instead |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations were not reliably resolved; no official public artifact was confirmed. |
| Citation count | 0 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | Not resolved |
| arXiv | 2604.25899 |
| Artifact | No official public artifact confirmed |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on workflow synthesis, belady-like cache policy, lookahead scheduling, proactive scaling. The reported evaluation context includes Production multi-agent coding-assistant traces, Representative multi-agent workflow applications.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Workflow synthesis, Belady-like cache policy, lookahead scheduling, proactive scaling.

## PLEX mapping

`R+S+E+P+F`; excellent composition, high effort.

## Datasets and evaluation workloads

- Production multi-agent coding-assistant traces
- Representative multi-agent workflow applications

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_pythia`
- Operations: `route`, `schedule`, `cache`, `feedback`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/pythia/metadata.json`](../../tests/policies/replications/pythia/metadata.json)
- Deferred mechanics: proactive scaling
<!-- plex-v0.6-replication:end -->

## Suggested citation

Shan Yu, Junyi Shu, Yuanjiang Ni, Kun Qian, Xue Li, Yang Wang, et al. “Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving.” arXiv preprint, 2026, 2026. https://arxiv.org/abs/2604.25899.

## Sources

- [Primary paper](https://arxiv.org/abs/2604.25899)
- [OpenAlex record](https://openalex.org/W7159547485)
