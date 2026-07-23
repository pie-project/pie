# ToolBench / StableToolBench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Large-scale tool-use trajectory corpus |
| Official source | [https://github.com/OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) |
| Associated paper | [https://arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789) |
| License | Apache-2.0 |
| Access | Public dataset; stable local API simulator available separately |
| Scale | 3,451 tools, 16,464 APIs, 126K instructions, 469K API calls |
| Format | JSON instructions, API schemas/environments, DFS reasoning/tool trajectories |
| Recommended tier | extension |

## Available fields

- single/multi-tool instructions
- API definitions
- reasoning tree
- tool execution and results

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | partial |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | yes |
| Workflow graph / dependencies | tree |
| SLO / priority | no |
| Failure / cancel / retry | partial |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- ToolLLM-style workloads in SLOs-Serve and agent papers

## PLEX operations exercised

- schedule
- feedback
- prefetch

## Strengths

- Large and diverse multi-tool decision trees with released execution results

## Limitations

- Automatically generated; live RapidAPI timing is unstable; no serving arrival/SLO/tenant/cache metadata
