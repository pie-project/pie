# PLEX

This directory is the single entry point for PLEX contracts, reproduction
evidence, reports, and literature resources.

## Start here

| Artifact | Purpose |
|---|---|
| [`plex_current_model_presentation.html`](plex_current_model_presentation.html) | Visual presentation of the 17 current-model policy reproductions |
| [`plex_policy_performance_report.md`](plex_policy_performance_report.md) | Human-readable offline and live performance evidence |
| [`plex_policy_performance_report.json`](plex_policy_performance_report.json) | Canonical machine-readable performance dump |
| [`plex_replication_report.md`](plex_replication_report.md) | Per-policy reproduction evidence and claim boundaries |
| [`plex_policy_reproducibility_roadmap.md`](plex_policy_reproducibility_roadmap.md) | Remaining gaps and required primitive roadmap |
| [`plex_policy_reproducibility_roadmap.json`](plex_policy_reproducibility_roadmap.json) | Machine-readable roadmap |

## Contracts and design

| Artifact | Purpose |
|---|---|
| [`plex_0.6_contract.md`](plex_0.6_contract.md) | Normative PLEX v0.6 contract |
| [`plex_0.6.md`](plex_0.6.md) | PLEX v0.6 implementation and delivery record |
| [`plex_0.5_to_0.6.md`](plex_0.5_to_0.6.md) | Migration guide |
| [`plex.md`](plex.md) | Original implemented contract |
| [`plex_gap.md`](plex_gap.md) | Design-gap audit |
| [`plex_paper.md`](plex_paper.md) | Paper-oriented architecture description |

## Surveys and validation

| Artifact | Purpose |
|---|---|
| [`plex_serving_policy_report.md`](plex_serving_policy_report.md) | Serving-policy literature survey |
| [`plex-serving-policy-wiki/`](plex-serving-policy-wiki/README.md) | Per-paper wiki and machine-readable catalog |
| [`plex_serving_workload_report.md`](plex_serving_workload_report.md) | Serving-workload and dataset survey |
| [`plex-serving-workload-wiki/`](plex-serving-workload-wiki/README.md) | Per-workload wiki and machine-readable catalog |
| [`plex_vllm_validation.md`](plex_vllm_validation.md) | vLLM integration validation |
| [`plex_vllm_validation.json`](plex_vllm_validation.json) | Machine-readable vLLM validation result |

Executable policy evidence remains under
[`tests/policies/`](../tests/policies/), including fidelity audits,
reproducibility gaps, replication metadata, cases, and expected traces.
