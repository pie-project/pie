# GAIA

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | General assistant benchmark |
| Official source | [https://huggingface.co/datasets/gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) |
| Associated paper | [https://arxiv.org/abs/2311.12983](https://arxiv.org/abs/2311.12983) |
| License | Gated dataset; no redistribution |
| Access | Public dev; gated validation/test |
| Scale | More than 450 questions across three difficulty levels, often with attachments |
| Format | Parquet metadata plus PDFs/media/attachments |
| Recommended tier | extension |

## Available fields

- task_id
- Question
- Level
- Final answer
- file_path
- annotator metadata

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | derived |
| Workflow graph / dependencies | derived |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | partial |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Dyserve

## PLEX operations exercised

- route
- schedule
- feedback

## Strengths

- Diverse tool/search/file workloads and clear task-level quality target

## Limitations

- Content benchmark only; no trajectories, arrivals, token timing, session, cache, or SLO
