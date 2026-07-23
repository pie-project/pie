# Continuum

> Canonical title: **Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2511.02230](https://arxiv.org/abs/2511.02230) |
| Venue / status | arXiv preprint / UC Berkeley technical report, revised 2026 |
| Year | 2026 |
| Authors | Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Alvin Cheung, Joseph Gonzalez, Ion Stoica |
| Institutions / group context | UC Berkeley Sky Computing Lab |
| Reputation evidence | recent preprint; peer-review status is not confirmed by this catalog; author affiliations include UC Berkeley Sky Computing Lab; a public implementation or artifact is linked. |
| Citation count | Not resolved (checked 2026-07-23) |
| Metadata provenance | Primary-page citation metadata |
| DOI | Not resolved |
| arXiv | 2511.02230 |
| Artifact | [Public artifact](https://github.com/Hanchenli/vllm-continuum) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on tool-duration-aware kv ttl and program fcfs. The reported evaluation context includes BFCL, mini-SWE-agent / SWE-agent workloads, TensorMesh internal agent-serving testbed.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Tool-duration-aware KV TTL and program FCFS.

## PLEX mapping

Existing direct replica.

## Datasets and evaluation workloads

- BFCL
- mini-SWE-agent / SWE-agent workloads
- TensorMesh internal agent-serving testbed

Named datasets/workloads identified from the primary text or manual audit.

## Suggested citation

Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, et al. “Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live.” arXiv preprint / UC Berkeley technical report, revised 2026, 2026. https://arxiv.org/abs/2511.02230.

## Sources

- [Primary paper](https://arxiv.org/abs/2511.02230)
- [Artifact](https://github.com/Hanchenli/vllm-continuum)
