# Mooncake FAST'25 Traces

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Production and synthetic KV-prefix traces |
| Official source | [https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release) |
| Associated paper | [https://www.usenix.org/conference/fast25/presentation/qin](https://www.usenix.org/conference/fast25/presentation/qin) |
| License | Apache-2.0 repository |
| Access | Direct JSONL files |
| Scale | 12,031 conversation, 23,608 tool/agent, and 3,993 synthetic requests |
| Format | JSONL |
| Recommended tier | core |

## Available fields

- timestamp
- input_length
- output_length
- hash_ids

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | yes |
| Input/output token lengths | yes |
| Prompt/response content | no |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | yes |
| Tool boundaries and durations | derived |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Mooncake
- LMetric ToolAgent workload

## PLEX operations exercised

- route
- schedule
- evict
- prefetch

## Strengths

- Real one-hour conversation and tool/agent arrival patterns
- 512-token prefix hashes expose reusable KV

## Limitations

- Tool boundaries and sessions are not explicit; no SLO, tenant, failures, or raw content
