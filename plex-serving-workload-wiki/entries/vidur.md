# Vidur

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | High-fidelity serving simulator and workload generator |
| Official source | [https://github.com/microsoft/vidur](https://github.com/microsoft/vidur) |
| Associated paper | [https://arxiv.org/abs/2405.05465](https://arxiv.org/abs/2405.05465) |
| License | MIT |
| Access | Open-source simulator with processed traces and synthetic generators |
| Scale | Supports Azure traces, synthetic/Poisson arrivals, multiple models and TP/PP configurations |
| Format | YAML config, CSV traces, simulator output/chrome traces |
| Recommended tier | core |

## Available fields

- arrival generator
- length generator
- model/device profile
- scheduler config
- TTFT/TPOT/E2E/batch metrics

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | yes |
| Input/output token lengths | yes |
| Prompt/response content | no |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | config |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | yes |

## Confirmed or representative use

- Used broadly for scheduling and capacity studies

## PLEX operations exercised

- route
- admit
- schedule
- feedback

## Strengths

- Fast counterfactual policy and deployment exploration with calibrated execution models

## Limitations

- Mainline has limited semantic session/tool/DAG/failure support; prefix caching is canary/experimental
