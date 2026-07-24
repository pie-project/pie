# Azure LLM Inference Trace 2023

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Production request trace |
| Official source | [https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) |
| Associated paper | [https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/) |
| License | CC-BY |
| Access | Direct CSV download |
| Scale | One-day sample from multiple Azure LLM services; separate code and conversation traces |
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

- Splitwise
- Vidur examples

## PLEX operations exercised

- route
- admit
- schedule

## Strengths

- Real production arrival and joint input/output length distributions
- Simple, portable replay format

## Limitations

- No prompt content, sessions, tenants, prefix hashes, SLOs, failures, or tool events
