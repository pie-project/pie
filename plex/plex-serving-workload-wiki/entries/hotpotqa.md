# HotpotQA

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Multi-hop QA content benchmark |
| Official source | [https://hotpotqa.github.io/](https://hotpotqa.github.io/) |
| Associated paper | [https://arxiv.org/abs/1809.09600](https://arxiv.org/abs/1809.09600) |
| License | See official dataset terms |
| Access | Public dataset |
| Scale | About 113K multi-hop questions with supporting facts and distractors |
| Format | JSON |
| Recommended tier | supporting |

## Available fields

- question
- answer
- context paragraphs
- supporting facts
- type/level

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
| Workflow graph / dependencies | derived |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- InferCept
- BEIR/RAG workloads

## PLEX operations exercised

- route
- schedule
- evict
- prefetch

## Strengths

- Multi-document dependencies and reusable corpus/document prefixes

## Limitations

- No arrivals, sessions, token timing, SLO, tool duration, or cache events
