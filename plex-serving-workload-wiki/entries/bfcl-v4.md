# Berkeley Function Calling Leaderboard V4

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Tool/function-calling benchmark |
| Official source | [https://gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Associated paper | [https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html) |
| License | Apache-2.0 code; dataset terms in Gorilla repository |
| Access | Public code, data, and model responses |
| Scale | Real-world function data with single-turn, multi-turn, and web-search categories |
| Format | JSON function schemas, prompts, expected calls, model responses |
| Recommended tier | core |

## Available fields

- functions/tools
- multi-turn state
- expected calls
- latency/cost leaderboard outputs

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | yes |
| Workflow graph / dependencies | no |
| SLO / priority | no |
| Failure / cancel / retry | partial |
| Multimodal payload | no |
| Model / adapter identity | yes |
| Hardware / topology state | no |

## Confirmed or representative use

- Continuum

## PLEX operations exercised

- schedule
- evict
- feedback

## Strengths

- Canonical multi-turn tool benchmark; released model responses and measured cost/latency

## Limitations

- No production arrivals, true external tool durations, tenant/SLO, prefix hashes, or infrastructure state
