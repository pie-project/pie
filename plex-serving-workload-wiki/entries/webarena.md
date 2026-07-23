# WebArena / BrowserGym

## Metadata

| Field | Value |
|---|---|
| Status | Existing public workload |
| Kind | Web-agent benchmark and trajectory corpus |
| Official source | [https://github.com/web-arena-x/webarena](https://github.com/web-arena-x/webarena) |
| Associated paper | [https://arxiv.org/abs/2307.13854](https://arxiv.org/abs/2307.13854) |
| License | Apache-2.0 |
| Access | Self-hosted websites, 812 tasks, execution and human trajectories |
| Scale | 812 benchmark examples; about 170 released human trajectories |
| Format | JSON task configs and HTML/action trajectories |
| Recommended tier | extension |

## Available fields

- instruction
- website state
- actions/observations
- human/agent trajectory
- success

## Policy-relevant coverage

| Dimension | Coverage |
|---|---|
| Arrival timestamps / rate | no |
| Input/output token lengths | no |
| Prompt/response content | yes |
| Session / logical-request lineage | yes |
| Tenant / principal identity | no |
| Prefix/cache lineage | no |
| Tool boundaries and durations | web |
| Workflow graph / dependencies | partial |
| SLO / priority | no |
| Failure / cancel / retry | yes |
| Multimodal payload | no |
| Model / adapter identity | no |
| Hardware / topology state | no |

## Confirmed or representative use

- Recommended browser-agent workload

## PLEX operations exercised

- schedule
- feedback
- prefetch

## Strengths

- Reproducible web environment and real trajectory structure

## Limitations

- No serving timing, token/cache lineage, SLO, tenant, or concurrent workflow arrivals
