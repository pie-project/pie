# ServeGen

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Production-fitted workload generator |
| Official source | [https://github.com/alibaba/ServeGen](https://github.com/alibaba/ServeGen) |
| Associated paper | [https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen](https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen) |
| License | Apache-2.0 |
| Access | Open-source generator plus fitted distributions and hashed conversations |
| Scale | Models billions of requests across 12 production models |
| Format | JSON distributions, CSV rate traces, hashed conversation JSON |
| Recommended tier | core |

## Available fields

- client rate/CV distributions
- input/output length CDFs over time
- language/reasoning/multimodal categories
- hashed conversations

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | generated |
| Input/output token lengths | generated |
| Prompt/response content | no |
| Session / logical-request lineage | partial |
| Tenant / principal identity | no |
| Prefix/cache lineage | partial |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | yes |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- ServeGen; temporal holdout beyond most surveyed policies

## PLEX operations exercised

- admit
- route
- schedule
- feedback
- rebalance

## Strengths

- Captures non-Poisson burstiness, shifting distributions, multimodal composition, and bimodal reasoning lengths
- Controllable and scalable

## Limitations

- No tool DAG, SLO, tenant fairness, cancellation, or explicit cache-residency events
