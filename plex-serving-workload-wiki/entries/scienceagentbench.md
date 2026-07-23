# ScienceAgentBench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Scientific coding-agent benchmark |
| Official source | [https://github.com/OSU-NLP-Group/ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) |
| Associated paper | [https://arxiv.org/abs/2410.05080](https://arxiv.org/abs/2410.05080) |
| License | Mostly CC-BY-4.0 data; MIT code; task-specific exceptions |
| Access | Public annotation and passworded full benchmark artifacts |
| Scale | 102 tasks from 44 peer-reviewed papers across four disciplines |
| Format | Task annotations, scientific datasets, generated program and JSONL evaluation logs |
| Recommended tier | extension |

## Available fields

- task input
- data files
- generated code
- execution result
- cost
- trajectory logs

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | code |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | yes |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- ThunderAgent
- OpenHands integration

## PLEX operations exercised

- schedule
- feedback
- prefetch

## Strengths

- Authentic long-running scientific tasks and executable outcomes

## Limitations

- No production arrival model, prefix/cache state, SLO, or standardized tool duration
