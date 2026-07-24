# MMMU / MMMU-Pro

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Multimodal reasoning benchmark |
| Official source | [https://github.com/MMMU-Benchmark/MMMU](https://github.com/MMMU-Benchmark/MMMU) |
| Associated paper | [https://arxiv.org/abs/2311.16502](https://arxiv.org/abs/2311.16502) |
| License | Apache-2.0 code; dataset terms apply |
| Access | Public Hugging Face datasets |
| Scale | 11.5K questions across 30 subjects, 183 subfields, and 32 image types |
| Format | Hugging Face rows with text/images/answers |
| Recommended tier | extension |

## Available fields

- question
- images
- options/answer
- subject/subfield
- explanation

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
| Multimodal payload | image |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended multimodal serving extension

## PLEX operations exercised

- admit
- route
- schedule
- evict
- feedback

## Strengths

- Heterogeneous encoder workloads, image counts/types, and reasoning lengths

## Limitations

- No arrival/session/cache/SLO/tenant/tool fields
