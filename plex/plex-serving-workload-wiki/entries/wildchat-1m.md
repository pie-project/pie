# WildChat-1M

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Timestamped real conversation corpus |
| Official source | [https://huggingface.co/datasets/allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) |
| Associated paper | [https://arxiv.org/abs/2405.01470](https://arxiv.org/abs/2405.01470) |
| License | ODC-BY |
| Access | Hugging Face download; toxic full version is gated |
| Scale | 837,989 released conversations; GPT-3.5/GPT-4; 68 languages |
| Format | Parquet / Hugging Face dataset |
| Recommended tier | extension |

## Available fields

- conversation_hash
- model
- timestamp
- per-turn timestamps
- content
- hashed_ip
- country/state
- turn_identifier
- moderation

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | partial |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | partial |
| Prefix/cache lineage | no |
| Tool boundaries and durations | no |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | partial |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Not a common legacy serving trace; recommended content/session overlay

## PLEX operations exercised

- route
- admit
- schedule
- feedback

## Strengths

- Per-turn timestamps, user linkage, model identity, content, geography, and multi-turn structure

## Limitations

- No token lengths, TTFT/TPOT, prefix hashes, SLO, tool/DAG metadata, or infrastructure state
