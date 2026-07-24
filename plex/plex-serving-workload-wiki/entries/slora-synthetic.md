# S-LoRA Synthetic Adapter Workload

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Synthetic multi-adapter workload |
| Official source | [https://github.com/S-LoRA/S-LoRA](https://github.com/S-LoRA/S-LoRA) |
| Associated paper | [https://arxiv.org/abs/2311.03285](https://arxiv.org/abs/2311.03285) |
| License | Apache-2.0 |
| Access | Artifact scripts; workload details in paper |
| Scale | Thousands of concurrent LoRA adapters under synthetic request/popularity patterns |
| Format | Artifact-defined synthetic generator |
| Recommended tier | extension |

## Available fields

- adapter id/rank
- request arrival
- input/output lengths
- batching configuration

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | generated |
| Input/output token lengths | generated |
| Prompt/response content | no |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | adapter |
| Hardware / topology state | no |

## Confirmed or representative use

- S-LoRA
- basis for Punica/dLoRA/CaraServe/Chameleon comparisons

## PLEX operations exercised

- admit
- route
- schedule
- evict
- prefetch

## Strengths

- Exercises adapter memory, heterogeneous ranks, batching, and popularity

## Limitations

- No public production adapter-request trace; popularity and arrivals are synthetic
