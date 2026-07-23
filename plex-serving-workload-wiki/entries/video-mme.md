# Video-MME

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Long multimodal/video benchmark |
| Official source | [https://github.com/MME-Benchmarks/Video-MME](https://github.com/MME-Benchmarks/Video-MME) |
| Associated paper | [https://arxiv.org/abs/2405.21075](https://arxiv.org/abs/2405.21075) |
| License | Video copyrights remain with owners; annotation terms apply |
| Access | Public Hugging Face data/annotations |
| Scale | 900 videos, 254 hours, 2,700 QA pairs; 11 seconds to 1 hour; subtitles/audio |
| Format | Video/audio/subtitle assets plus JSON answers |
| Recommended tier | extension |

## Available fields

- video
- duration class
- subtitle
- audio
- question/answer
- domain/task type

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
| Multimodal payload | video/audio |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended encoder-residency and long-prefill workload

## PLEX operations exercised

- admit
- route
- schedule
- evict
- prefetch

## Strengths

- Wide payload size and modality mix; long encoder outputs and context

## Limitations

- No serving arrivals, session, SLO, cache lineage, failure, or tenant fields
