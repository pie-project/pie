# Autellix / Agentix

> Canonical title: **Autellix: An Efficient Serving Engine for LLM Agents as General Programs**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://arxiv.org/abs/2502.13965](https://arxiv.org/abs/2502.13965) |
| Venue / status | NSDI 2026 (final paper: Agentix); arXiv 2025 preprint: Autellix |
| Year | 2026 |
| Authors | Michael Luo, Xiaoxiang Shi, Colin Cai, Tianjun Zhang, Justin Wong, Yichuan Wang, Chi Chiu Wang, Yanping Huang, Zhifeng Chen, Joseph E. Gonzalez, Ion Stoica |
| Institutions / group context | UC Berkeley Sky Computing Lab, Google DeepMind |
| Reputation evidence | strong publication signal from a selective systems/ML venue (NSDI 2026 (final paper: Agentix); arXiv 2025 preprint: Autellix); author affiliations include UC Berkeley Sky Computing Lab, Google DeepMind; no official public artifact was confirmed. |
| Citation count | 1 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2502.13965 |
| arXiv | 2502.13965 |
| Artifact | No official public artifact confirmed |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on plas and atlas attained-service scheduling for programs/dags. The reported evaluation context includes Agentic program workloads including chatbot/ReAct loops, MapReduce-style programs, and MCTS-style parallel reasoning.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

PLAS and ATLAS attained-service scheduling for programs/DAGs.

## PLEX mapping

Existing direct replica; count once.

## Datasets and evaluation workloads

- Agentic program workloads including chatbot/ReAct loops, MapReduce-style programs, and MCTS-style parallel reasoning

Named datasets/workloads identified from the primary text or manual audit.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_agentix`
- Operations: `schedule`, `feedback`
- Evidence: `policy-kernel-reproduction`
- Validation: `passing`
- Metadata: [`tests/policies/replications/agentix/metadata.json`](../../tests/policies/replications/agentix/metadata.json)
- Deferred mechanics: KV swap kernel; multi-step engine execution; full ATLAS DAG runtime
<!-- plex-v0.6-replication:end -->

## Suggested citation

Michael Luo, Xiaoxiang Shi, Colin Cai, Tianjun Zhang, Justin Wong, Yichuan Wang, et al. “Autellix: An Efficient Serving Engine for LLM Agents as General Programs.” NSDI 2026 (final paper: Agentix); arXiv 2025 preprint: Autellix, 2026. https://doi.org/10.48550/arxiv.2502.13965.

## Sources

- [Primary paper](https://arxiv.org/abs/2502.13965)
- [OpenAlex record](https://openalex.org/W4407800405)
