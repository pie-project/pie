# LLMServingSim 2.0

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Cycle-level heterogeneous/disaggregated simulator |
| Official source | [https://github.com/casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim) |
| Associated paper | [https://doi.org/10.1109/ISPASS69572.2026.00012](https://doi.org/10.1109/ISPASS69572.2026.00012) |
| License | MIT |
| Access | Open-source Dockerized simulator |
| Scale | Models heterogeneous accelerators, CPU/CXL/PIM memory, MoE routing, TP/PP/EP/DP |
| Format | Config files, Chakra/ASTRA-Sim traces, profiler data |
| Recommended tier | extension |

## Available fields

- request workload
- hardware topology
- parallelism
- memory tiers
- network
- scheduler

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | generated |
| Input/output token lengths | generated |
| Prompt/response content | no |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | yes |

## Confirmed or representative use

- Recommended infrastructure stress backend

## PLEX operations exercised

- route
- schedule
- evict
- prefetch
- rebalance

## Strengths

- Strong hardware, network, disaggregation, and memory-tier modeling

## Limitations

- No native agent semantics, tenant/SLO honesty, tool lifecycle, or workflow DAG trace
