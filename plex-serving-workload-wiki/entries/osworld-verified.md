# OSWorld-Verified

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Multimodal computer-use benchmark |
| Official source | [https://github.com/xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) |
| Associated paper | [https://arxiv.org/abs/2404.07972](https://arxiv.org/abs/2404.07972) |
| License | Apache-2.0 |
| Access | Public benchmark, VM images, examples, screenshots and videos |
| Scale | Open-ended tasks across Ubuntu/Windows applications; verified revision available |
| Format | JSON task metadata plus screenshots, actions, videos, and result logs |
| Recommended tier | extension |

## Available fields

- task
- OS/application
- screenshots
- actions
- video
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
| Tool boundaries and durations | GUI |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | yes |
| Multimodal payload | yes |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended multimodal agent workload

## PLEX operations exercised

- schedule
- feedback
- evict

## Strengths

- Real GUI state, multimodal payloads, long trajectories, cloud-parallel harness

## Limitations

- No production arrival/session population, token/cache lineage, SLO, or external tool timing
