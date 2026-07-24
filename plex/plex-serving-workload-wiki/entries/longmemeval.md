# LongMemEval

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Timestamped long-term conversation benchmark |
| Official source | [https://github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) |
| Associated paper | [https://arxiv.org/abs/2410.10813](https://arxiv.org/abs/2410.10813) |
| License | MIT |
| Access | Public cleaned JSON datasets and construction pipeline |
| Scale | 500 questions; 40-session/115K-token and 500-session variants |
| Format | JSON histories with timestamped sessions and turns |
| Recommended tier | extension |

## Available fields

- question_type
- question
- answer
- haystack_session_ids
- haystack_dates
- haystack_sessions
- answer_session_ids

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | derived |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | chain |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended logical-request and long-session workload

## PLEX operations exercised

- route
- schedule
- evict
- feedback

## Strengths

- Explicit timestamps, hundreds of sessions, knowledge updates, temporal reasoning, controllable history length

## Limitations

- Synthetic session construction; no concurrent arrivals, tenant/SLO, tool events, or serving outcomes
