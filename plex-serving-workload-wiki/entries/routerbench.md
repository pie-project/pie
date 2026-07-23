# RouterBench

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Multi-model routing outcome dataset |
| Official source | [https://huggingface.co/datasets/withmartian/routerbench](https://huggingface.co/datasets/withmartian/routerbench) |
| Associated paper | [https://arxiv.org/abs/2403.12031](https://arxiv.org/abs/2403.12031) |
| License | See dataset card |
| Access | Public Hugging Face data |
| Scale | Over 30K prompts, responses from 11 LLMs, zero-shot and five-shot variants |
| Format | Dataset rows |
| Recommended tier | extension |

## Available fields

- prompt
- model
- response
- estimated cost
- correctness/performance score
- source benchmark

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
| Failure / cancel / retry | quality |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Model-routing related work

## PLEX operations exercised

- route
- feedback

## Strengths

- Counterfactual quality/cost outcomes across many models for the same prompt

## Limitations

- No queue/load/hardware/cache/arrival/SLO/session data; model routing is not replica placement
