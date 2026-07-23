# BurstGPT v2

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Production request/session trace |
| Official source | [https://github.com/HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT) |
| Associated paper | [https://doi.org/10.1145/3711896.3737413](https://doi.org/10.1145/3711896.3737413) |
| License | CC-BY-4.0 |
| Access | GitHub release CSVs |
| Scale | Three releases spanning 110-121 days; roughly 5.3M records per major trace |
| Format | CSV |
| Recommended tier | core |

## Available fields

- Timestamp
- Session ID
- Elapsed time
- Model
- Request tokens
- Response tokens
- Total tokens
- Log Type

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | yes |
| Input/output token lengths | yes |
| Prompt/response content | no |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | partial |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- BurstGPT workload study; suitable replacement for Poisson arrivals in serving papers

## PLEX operations exercised

- admit
- route
- schedule
- feedback
- rebalance

## Strengths

- Long duration, daily/weekly periodicity, burstiness, session IDs, end-to-end latency, and zero-token failures

## Limitations

- No raw content, prefix hashes, tenant IDs, TTFT/TPOT, explicit cancellation cause, tools, or SLOs
