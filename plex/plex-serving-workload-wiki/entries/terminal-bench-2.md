# Terminal-Bench 2.0

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Executable agent task benchmark |
| Official source | [https://github.com/laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench) |
| Associated paper | [https://arxiv.org/abs/2601.11868](https://arxiv.org/abs/2601.11868) |
| License | Apache-2.0 |
| Access | Public tasks and Harbor execution harness |
| Scale | Approximately 100 hard terminal tasks in the original core; evolving registry in 2.0 |
| Format | Task directories, Docker environments, test scripts, agent trajectories |
| Recommended tier | core |

## Available fields

- instruction
- environment
- test oracle
- terminal actions/output
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
| Tool boundaries and durations | terminal |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | yes |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- KAIROS

## PLEX operations exercised

- schedule
- evict
- feedback

## Strengths

- Long-horizon executable tasks with objective validation and realistic tool intervals

## Limitations

- No native concurrent arrivals, prefix lineage, SLO, tenant, or standardized model-call timing
