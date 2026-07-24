# SWE-bench Verified

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Agent task benchmark |
| Official source | [https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) |
| Associated paper | [https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770) |
| License | MIT |
| Access | Hugging Face plus Docker evaluation harness |
| Scale | 500 human-validated GitHub issue/PR instances |
| Format | Dataset rows plus repository snapshots and tests |
| Recommended tier | core |

## Available fields

- instance_id
- repo
- base_commit
- problem_statement
- patch
- test_patch
- FAIL_TO_PASS
- PASS_TO_PASS

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | derived |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | generated |
| Workflow graph / dependencies | derived |
| SLO / priority | no |
| Failure / cancel / retry | generated |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Continuum
- ThunderAgent
- ConServe
- KAIROS
- Dyserve
- SAGA

## PLEX operations exercised

- schedule
- evict
- feedback
- prefetch

## Strengths

- Real software tasks, reproducible environment, objective completion tests, long agent trajectories

## Limitations

- Dataset itself has no LLM calls, arrivals, tokens, tool latencies, SLOs, or cancellations; requires an instrumented agent run
