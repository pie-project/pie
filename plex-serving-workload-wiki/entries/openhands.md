# OpenHands Evaluation Trajectories

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Agent execution framework / trajectory generator |
| Official source | [https://github.com/All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) |
| Associated paper | [https://arxiv.org/abs/2407.16741](https://arxiv.org/abs/2407.16741) |
| License | Repository-specific terms |
| Access | Open source; benchmark outputs generated locally |
| Scale | Supports SWE-bench, ScienceAgentBench, and other agent benchmarks |
| Format | JSONL/event logs and sandbox actions |
| Recommended tier | extension |

## Available fields

- model messages
- actions
- observations
- tool/sandbox events
- task result

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | yes |
| Workflow graph / dependencies | partial |
| SLO / priority | no |
| Failure / cancel / retry | yes |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- ThunderAgent
- ScienceAgentBench integrations

## PLEX operations exercised

- schedule
- evict
- feedback
- prefetch

## Strengths

- Rich executable trajectories and multiple benchmark adapters

## Limitations

- No standardized public serving trace; arrival concurrency and latency instrumentation must be added
