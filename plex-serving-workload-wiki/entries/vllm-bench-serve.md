# vLLM Bench Serve

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Serving benchmark harness |
| Official source | [https://docs.vllm.ai/en/latest/cli/bench/serve.html](https://docs.vllm.ai/en/latest/cli/bench/serve.html) |
| Associated paper | [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| License | Apache-2.0 |
| Access | Included with vLLM |
| Scale | Configurable request count/rate using random, ShareGPT, sonnet, and other dataset adapters |
| Format | CLI and JSON result records |
| Recommended tier | supporting |

## Available fields

- dataset adapter
- request rate
- input/output lengths
- TTFT/TPOT/E2E/throughput

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | generated |
| Input/output token lengths | yes |
| Prompt/response content | partial |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | measured |

## Confirmed or representative use

- Common evaluation harness in serving artifacts

## PLEX operations exercised

- admit
- schedule
- feedback

## Strengths

- Low-friction real-engine measurement and standardized latency outputs

## Limitations

- Weak session/tool/DAG/tenant/SLO/cache event semantics unless extended
