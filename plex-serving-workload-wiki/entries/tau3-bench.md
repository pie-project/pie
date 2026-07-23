# tau3-bench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Stateful tool-agent-user benchmark |
| Official source | [https://github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) |
| Associated paper | [https://arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045) |
| License | MIT |
| Access | Open-source tasks, environments, simulators, and historical trajectories |
| Scale | Airline, retail, telecom, banking knowledge/RAG, mock, and full-duplex voice modes |
| Format | Task/config files and trajectory result files |
| Recommended tier | core |

## Available fields

- domain state
- agent tools
- user tools
- tasks
- policy rules
- multi-turn trajectories
- voice/audio events

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
| Multimodal payload | audio |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended new workload; not broadly used in first-wave serving papers

## PLEX operations exercised

- admit
- schedule
- evict
- feedback
- prefetch

## Strengths

- Best public source for stateful tool/user interaction, policy constraints, RAG, and full-duplex voice

## Limitations

- Generated user behavior and tool timing; no production arrival process, tenant/SLO, or cache lineage
