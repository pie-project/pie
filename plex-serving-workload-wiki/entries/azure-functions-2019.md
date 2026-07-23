# Azure Functions Trace 2019

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Legacy production serverless trace |
| Official source | [https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) |
| Associated paper | [https://www.usenix.org/conference/atc20/presentation/shahrad](https://www.usenix.org/conference/atc20/presentation/shahrad) |
| License | CC-BY |
| Access | Direct compressed CSV release |
| Scale | 14 days of sampled applications/functions; per-minute counts plus duration and memory distributions |
| Format | CSV |
| Recommended tier | supporting |

## Available fields

- HashOwner
- HashApp
- HashFunction
- Trigger
- per-minute invocations
- duration percentiles
- memory percentiles

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | aggregated |
| Input/output token lengths | no |
| Prompt/response content | no |
| Session / logical-request lineage | no |
| Tenant / principal identity | yes |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | partial |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | partial |

## Confirmed or representative use

- PARD uses an Azure Functions trace; useful legacy baseline for burst and cold-start policies

## PLEX operations exercised

- admit
- route
- rebalance
- feedback

## Strengths

- Long-lived owner/app/function identities, burst patterns, orchestration triggers, memory and duration distributions

## Limitations

- Minute aggregation; no per-request tokens/content/session/cache/SLO; not an LLM trace
