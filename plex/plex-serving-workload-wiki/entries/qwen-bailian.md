# Qwen-Bailian Anonymous Usage Traces

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Production session and KV-prefix trace |
| Official source | [https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) |
| Associated paper | [https://www.usenix.org/conference/atc25/presentation/wang-jiahao](https://www.usenix.org/conference/atc25/presentation/wang-jiahao) |
| License | Apache-2.0 |
| Access | Direct JSONL files and official Rust replayer |
| Scale | Two-hour samples for To-C, To-B, thinking, and coder workloads |
| Format | JSONL |
| Recommended tier | core |

## Available fields

- chat_id
- parent_chat_id
- timestamp
- input_length
- output_length
- type
- turn
- hash_ids

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | yes |
| Input/output token lengths | yes |
| Prompt/response content | no |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | yes |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | partial |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- KVCache Cache in the Wild
- LMetric
- SMetric

## PLEX operations exercised

- route
- schedule
- evict
- feedback
- rebalance

## Strengths

- Best public combination of real timestamps, session lineage, turn number, workload type, and prefix-block hashes
- High-fidelity official replayer

## Limitations

- No tenant identity, SLO, tool timing/outcome, workflow DAG, cancellation, or raw content
