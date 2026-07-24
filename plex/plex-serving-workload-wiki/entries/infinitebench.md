# InfiniteBench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | 100K+ long-context benchmark |
| Official source | [https://github.com/OpenBMB/InfiniteBench](https://github.com/OpenBMB/InfiniteBench) |
| Associated paper | [https://arxiv.org/abs/2402.13718](https://arxiv.org/abs/2402.13718) |
| License | MIT |
| Access | Public Hugging Face data and JSONL files |
| Scale | 12 tasks; many inputs 75K-200K tokens, with some Chinese book QA above 2M tokens |
| Format | JSONL |
| Recommended tier | extension |

## Available fields

- task
- context
- question/input
- answer/output

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | derived |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended extreme long-context workload

## PLEX operations exercised

- admit
- schedule
- evict
- prefetch

## Strengths

- Extreme context pressure, long dialogue, code, retrieval, and synthetic stress tasks

## Limitations

- Content-only; no arrival/session/cache/SLO/tenant/hardware events
