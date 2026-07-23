# mini-SWE-agent Trajectories

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Agent trajectory generator |
| Official source | [https://github.com/SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) |
| Associated paper | [https://mini-swe-agent.com/](https://mini-swe-agent.com/) |
| License | MIT |
| Access | Open-source agent and trajectory browser |
| Scale | Runs SWE-bench and related coding benchmarks with a linear message history |
| Format | Message/trajectory logs |
| Recommended tier | core |

## Available fields

- linear message history
- bash actions
- tool outputs
- model calls
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
| Workflow graph / dependencies | chain |
| SLO / priority | no |
| Failure / cancel / retry | yes |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Continuum and KAIROS use mini-SWE-agent-style workloads

## PLEX operations exercised

- schedule
- evict
- feedback
- prefetch

## Strengths

- Simple one-to-one relation between message history and trajectory; easy to instrument for serving events

## Limitations

- Single bash tool and linear control flow do not cover parallel DAG agents or rich tool ecosystems
