# ShareGPT / Vicuna Conversation Corpus

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Community conversation content corpus |
| Official source | [https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| Associated paper | [https://lmsys.org/blog/2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/) |
| License | Provenance and redistribution terms require care |
| Access | Community mirrors; original release is unstable |
| Scale | Tens of thousands of multi-turn shared ChatGPT conversations |
| Format | Conversation JSON |
| Recommended tier | supporting |

## Available fields

- conversation id
- user/assistant turns
- text

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
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

- InferCept
- VTC-derived traces
- QLM
- DLPM
- Llumnix
- CachedAttention
- many serving baselines

## PLEX operations exercised

- schedule
- evict
- feedback

## Strengths

- Historically dominant source for realistic length and multi-turn content

## Limitations

- No reliable arrivals, tenant, model, SLO, prefix hashes, tools, failures; licensing/provenance is weaker than newer corpora
