# HumanEval / MBPP / APPS Code Suite

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Code-generation content suite |
| Official source | [https://github.com/openai/human-eval](https://github.com/openai/human-eval) |
| Associated paper | [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374) |
| License | Dataset-specific |
| Access | Public datasets and test harnesses |
| Scale | Hundreds to thousands of code problems with executable tests |
| Format | Problem text, function signatures, tests, generated code |
| Recommended tier | supporting |

## Available fields

- problem
- tests
- solution
- execution result

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
| Failure / cancel / retry | yes |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- RouterBench includes MBPP; common serving and routing benchmarks

## PLEX operations exercised

- route
- schedule
- feedback

## Strengths

- Objective quality and variable output lengths; supports sampling/repair workloads

## Limitations

- No request timing, session, cache, SLO, tenant, or tool environment
