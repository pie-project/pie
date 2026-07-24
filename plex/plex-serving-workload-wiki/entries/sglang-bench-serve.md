# SGLang Serving Benchmarks

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Serving benchmark harness |
| Official source | [https://docs.sglang.ai/](https://docs.sglang.ai/) |
| Associated paper | [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) |
| License | Apache-2.0 |
| Access | Included with SGLang |
| Scale | ShareGPT, random, GSM8K, multi-turn, prefix-sharing, and structured-program benchmarks |
| Format | CLI, JSON/JSONL inputs and result logs |
| Recommended tier | supporting |

## Available fields

- request rate
- prompts
- lengths
- prefix-sharing setup
- latencies
- throughput

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | generated |
| Input/output token lengths | yes |
| Prompt/response content | yes |
| Session / logical-request lineage | partial |
| Tenant / principal identity | no |
| Prefix/cache lineage | yes |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | measured |

## Confirmed or representative use

- SGLang/RadixAttention
- KVFlow
- PEEK
- many prefix-cache papers

## PLEX operations exercised

- schedule
- evict
- feedback

## Strengths

- Useful real-engine prefix/cache and structured-generation controls

## Limitations

- No standard tenant/SLO/tool-duration/failure/DAG-lifecycle schema across benchmarks
