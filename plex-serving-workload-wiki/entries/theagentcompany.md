# TheAgentCompany

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Enterprise workflow-agent benchmark |
| Official source | [https://github.com/TheAgentCompany/TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) |
| Associated paper | [https://arxiv.org/abs/2412.14161](https://arxiv.org/abs/2412.14161) |
| License | MIT |
| Access | Public Dockerized company environment and result trajectories |
| Scale | Professional tasks spanning web, code, programs, and coworker communication |
| Format | Docker task images and trajectory files |
| Recommended tier | extension |

## Available fields

- task
- company applications
- web/code actions
- coworker messages
- subcheckpoints
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

- Recommended new enterprise workflow workload

## PLEX operations exercised

- route
- schedule
- feedback

## Strengths

- Multi-application workflows and communication resembling compound enterprise agents

## Limitations

- No production arrival/token/cache/SLO fields; environment setup is heavyweight
