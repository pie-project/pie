# AgentBench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Multi-environment agent benchmark |
| Official source | [https://github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench) |
| Associated paper | [https://arxiv.org/abs/2308.03688](https://arxiv.org/abs/2308.03688) |
| License | Apache-2.0 |
| Access | Public tasks and containerized environments |
| Scale | Eight environments including OS, DB, KG, card game, lateral thinking, WebShop, Mind2Web, and web browsing |
| Format | Task splits and environment interaction logs |
| Recommended tier | extension |

## Available fields

- task
- environment state
- actions/observations
- multi-turn interaction
- result

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
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended breadth workload

## PLEX operations exercised

- schedule
- feedback

## Strengths

- Broad environment diversity and multi-turn behavior

## Limitations

- Old tool/function interfaces; no production arrivals, tokens, SLO, prefix, or hardware fields
