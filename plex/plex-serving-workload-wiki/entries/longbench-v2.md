# LongBench v2

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Long-context content benchmark |
| Official source | [https://github.com/THUDM/LongBench](https://github.com/THUDM/LongBench) |
| Associated paper | [https://arxiv.org/abs/2412.15204](https://arxiv.org/abs/2412.15204) |
| License | MIT |
| Access | Public Hugging Face dataset |
| Scale | 503 questions with 8K to 2M-word contexts across six task categories |
| Format | JSON/Hugging Face rows |
| Recommended tier | extension |

## Available fields

- id
- instruction
- input
- context
- answer
- length
- domain
- sub_domain

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

- Recommended new long-context workload

## PLEX operations exercised

- admit
- schedule
- evict
- prefetch

## Strengths

- Realistic long documents, code repositories, dialogues, structured data, and controllable length

## Limitations

- No arrivals/session population, prefix hashes, SLO, tool boundaries, or infrastructure state
