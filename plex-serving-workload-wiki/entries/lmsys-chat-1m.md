# LMSYS-Chat-1M

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Real conversation content corpus |
| Official source | [https://huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) |
| Associated paper | [https://arxiv.org/abs/2309.11998](https://arxiv.org/abs/2309.11998) |
| License | Custom LMSYS dataset agreement |
| Access | Gated agreement; redistribution restricted |
| Scale | 1M conversations, 25 models, 210K IPs, 154 languages; average 2 turns |
| Format | OpenAI-style conversation JSON |
| Recommended tier | supporting |

## Available fields

- conversation_id
- model
- conversation
- language
- moderation
- redacted

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | partial |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Common content source for ShareGPT-like serving workloads and model routing

## PLEX operations exercised

- route
- schedule
- evict

## Strengths

- Real prompts/responses, model identity, multilingual and moderation metadata

## Limitations

- No request-level arrival timing, token timing, prefix hashes, SLO, tool events, or system outcomes
