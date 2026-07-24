# RouteLLM / Chatbot Arena Preference Data

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Preference-based model-routing workload |
| Official source | [https://github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) |
| Associated paper | [https://arxiv.org/abs/2406.18665](https://arxiv.org/abs/2406.18665) |
| License | Apache-2.0 code; LMSYS data agreement applies |
| Access | Public code, routers, cached benchmark outputs, gated preference data |
| Scale | 55K Arena preferences plus MMLU, GSM8K, and MT-Bench evaluations |
| Format | Preference pairs and precomputed model outcomes |
| Recommended tier | supporting |

## Available fields

- prompt
- strong/weak model responses
- preference/win
- cost threshold
- benchmark

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
| Failure / cancel / retry | quality |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Router-tier control papers

## PLEX operations exercised

- route
- feedback

## Strengths

- Direct cost-quality routing objective and calibration shift experiments

## Limitations

- No serving load, cache, session, hardware, SLO latency, or resource contention
