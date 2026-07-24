# LiveCodeBench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Temporal code-reasoning benchmark |
| Official source | [https://github.com/LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) |
| Associated paper | [https://arxiv.org/abs/2403.07974](https://arxiv.org/abs/2403.07974) |
| License | MIT |
| Access | Public continuously updated dataset and evaluator |
| Scale | 400 problems in v1, 1,055 by v6; dated contest problems from LeetCode, AtCoder, Codeforces |
| Format | Hugging Face datasets and generated solution files |
| Recommended tier | extension |

## Available fields

- problem
- release date
- tests
- scenario
- generated solutions
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
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Dyserve

## PLEX operations exercised

- route
- schedule
- feedback

## Strengths

- Temporal holdout, executable quality, variable difficulty, self-repair and execution scenarios

## Limitations

- No serving arrivals, sessions, prefix/cache, SLO, or branch trajectories unless generated
