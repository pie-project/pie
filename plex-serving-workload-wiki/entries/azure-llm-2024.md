# Azure LLM Inference Trace 2024

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Production request trace |
| Official source | [https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) |
| Associated paper | [https://arxiv.org/abs/2408.00741](https://arxiv.org/abs/2408.00741) |
| License | CC-BY |
| Access | Direct one-week CSV downloads |
| Scale | May 10-19, 2024 sample; code and conversation services |
| Format | CSV |
| Recommended tier | core |

## Available fields

- TIMESTAMP
- ContextTokens
- GeneratedTokens

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | yes |
| Input/output token lengths | yes |
| Prompt/response content | no |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- DynamoLLM
- LMetric comparison
- multiple serving simulators

## PLEX operations exercised

- route
- admit
- schedule
- rebalance

## Strengths

- Longer production window than the 2023 trace
- Good for burst, diurnal, energy, and capacity studies

## Limitations

- No content, session identity, prefix reuse, tenant, SLO, or outcome fields
