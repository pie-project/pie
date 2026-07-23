# GSM8K / MATH / AIME / Math500 Suite

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Reasoning content suite |
| Official source | [https://huggingface.co/datasets/HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| Associated paper | [https://arxiv.org/abs/2412.20993](https://arxiv.org/abs/2412.20993) |
| License | Dataset-specific |
| Access | Public benchmark datasets |
| Scale | Thousands of grade-school through competition-level math questions |
| Format | Question/answer records |
| Recommended tier | supporting |

## Available fields

- problem
- answer
- difficulty/category

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- InferCept GSM8K-XL
- Certaindex
- many reasoning schedulers

## PLEX operations exercised

- admit
- schedule
- feedback

## Strengths

- Easy to generate Best-of-N, self-consistency, long-reasoning, and branch-parallel workloads

## Limitations

- Content-only; no arrivals, real tool waits, session identity, cache, SLO, or production failures
