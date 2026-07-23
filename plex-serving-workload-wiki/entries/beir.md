# BEIR

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Retrieval benchmark suite |
| Official source | [https://github.com/beir-cellar/beir](https://github.com/beir-cellar/beir) |
| Associated paper | [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) |
| License | Apache-2.0 code; component dataset licenses vary |
| Access | Public datasets and loader |
| Scale | 17-18 retrieval datasets; corpora from thousands to millions of documents |
| Format | corpus/queries/qrels |
| Recommended tier | supporting |

## Available fields

- document id/text
- query id/text
- relevance labels

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
| Failure / cancel / retry | no |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- RAG and prefix-cache papers; HotpotQA used by InferCept

## PLEX operations exercised

- route
- schedule
- evict
- prefetch

## Strengths

- Diverse domain and corpus sizes; easy to synthesize document-overlap and retrieval fanout

## Limitations

- No generator-side answer tokens, arrivals, sessions, SLO, or cache lineage without a RAG pipeline
