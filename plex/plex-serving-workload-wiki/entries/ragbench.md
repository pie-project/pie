# RAGBench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | RAG evaluation corpus |
| Official source | [https://huggingface.co/datasets/galileo-ai/ragbench](https://huggingface.co/datasets/galileo-ai/ragbench) |
| Associated paper | [https://arxiv.org/abs/2407.11005](https://arxiv.org/abs/2407.11005) |
| License | See dataset card |
| Access | Public Hugging Face dataset |
| Scale | Component datasets include HotpotQA, MSMARCO, HAGRID, and ExpertQA |
| Format | Question, retrieved context, response, and RAG quality labels |
| Recommended tier | extension |

## Available fields

- question
- documents/context
- response
- relevance/faithfulness/utilization labels

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | no |
| Tenant / principal identity | no |
| Prefix/cache lineage | derived |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | quality |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended RAG policy workload

## PLEX operations exercised

- route
- schedule
- evict
- prefetch
- feedback

## Strengths

- Can couple cache/resource policy to answer quality and context utilization

## Limitations

- No serving arrivals/session/SLO/tenant/tool timing; generated responses may be model-specific
