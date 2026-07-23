# Dynasor / Math500 Reasoning Workload

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Reasoning trajectory benchmark |
| Official source | [https://github.com/hao-ai-lab/Dynasor](https://github.com/hao-ai-lab/Dynasor) |
| Associated paper | [https://arxiv.org/abs/2412.20993](https://arxiv.org/abs/2412.20993) |
| License | Repository-specific open-source terms |
| Access | Public implementation and Math500 benchmark scripts |
| Scale | Math500 questions; configurable parallel solution sampling and token deprivation |
| Format | Prompt plus multiple reasoning trajectories and progress/certainty signals |
| Recommended tier | core |

## Available fields

- problem
- sample/branch
- generated tokens
- candidate answers
- Certaindex/progress
- stop decision

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | branches |
| SLO / priority | no |
| Failure / cancel / retry | partial |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Certaindex / Dynasor

## PLEX operations exercised

- admit
- schedule
- feedback

## Strengths

- Directly exercises dynamic compute allocation, early termination, and branch-level utility

## Limitations

- No production arrival or tenant/SLO/cache/hardware trace
