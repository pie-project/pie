# PLEX serving-policy paper wiki

This wiki contains one page for each of the **87** policy-bearing
papers in the PLEX literature corpus.

## Metadata conventions

- Search and citation-count cutoff: **2026-07-23**.
- Citation counts are OpenAlex `cited_by_count` values for the matched record.
  Preprint and proceedings versions can split citations, so counts are signals,
  not authoritative totals.
- “Reputation evidence” reports observable signals: publication venue, author
  affiliations, and artifact availability. It is not an endorsement.
- Abstracts are represented by editorial paraphrases. The primary source is
  linked for the authoritative abstract.
- Dataset fields are conservative. Missing names are marked rather than guessed.
- The machine-readable catalog is [`catalog.json`](catalog.json).

## Navigation

- [Methodology and limitations](methodology.md)
- [Root report](../plex_serving_policy_report.md)

## Agent, workflow, and multi-turn serving

| Paper | Year | Venue / status | Citations | PLEX |
|---|---:|---|---:|---|
| [Parrot](papers/parrot.md) | 2024 | arXiv (Cornell University) | 5 | `R+S`; good corpus, larger application contract. |
| [InferCept](papers/infercept.md) | 2024 | ICML 2024 | — | Direct `S+E+F` replica. |
| [Autellix / Agentix](papers/autellix-agentix.md) | 2026 | NSDI 2026 (final paper: Agentix); arXiv 2025 preprint: Autellix | 1 | Existing direct replica; count once. |
| [Continuum](papers/continuum.md) | 2026 | arXiv preprint / UC Berkeley technical report, revised 2026 | 9967 | Existing direct replica. |
| [KVFlow](papers/kvflow.md) | 2025 | NeurIPS 2025 | — | Existing direct replica plus optional `P`. |
| [Certaindex / Dynasor](papers/certaindex-dynasor.md) | 2024 | arXiv preprint, 2024; later revisions in 2025 | 1 | Direct `S+F`; best bundle/task candidate. |
| [Helium](papers/helium.md) | 2026 | arXiv preprint, 2026 | 0 | Existing `S`; prefetch is optional/mechanical. |
| [Justitia](papers/justitia.md) | 2025 | arXiv preprint, 2025 | — | Direct `S+F`. |
| [ThunderAgent](papers/thunderagent.md) | 2026 | arXiv (Cornell University) | 0 | `S+E+B+F`; strong coordinated case. |
| [Pythia](papers/pythia.md) | 2026 | arXiv (Cornell University) | 0 | `R+S+E+P+F`; excellent composition, high effort. |
| [HexAGenT](papers/hexagent.md) | 2026 | arXiv (Cornell University) | 0 | `R+S`; temporal-holdout candidate. |
| [Observation, Not Prediction / ConServe](papers/observation-not-prediction-conserve.md) | 2026 | arXiv (Cornell University) | 0 | Simple `R`; strong logical-request example. |
| [SMetric](papers/smetric.md) | 2026 | arXiv (Cornell University) | 0 | `R+B`; highly relevant holdout. |
| [GoodServe](papers/goodserve.md) | 2026 | arXiv (Cornell University) | 0 | `R+B+F`; prediction-heavy. |
| [KAIROS](papers/kairos.md) | 2005 | Proceedings of the twentieth ACM symposium on Operating systems principles | 54 | `A+R+F`; DVFS is an optional host action. |
| [A Workflow-Aware Serving Layer for Agentic Applications (Dyserve)](papers/a-workflow-aware-serving-layer-for-agentic-applications-dyserve.md) | 2026 | arXiv (Cornell University) | 0 | `A+R+F` only if service plans are legal route candidates. |
| [SAGA](papers/saga.md) | 1987 | Proceedings of the 1987 ACM SIGMOD international conference on Management of data - SIGMOD '87 | 390 | `R+S+E+B`; holdout until artifact/source maturity improves. |
## Scheduling, fairness, SLOs, and admission

| Paper | Year | Venue / status | Citations | PLEX |
|---|---:|---|---:|---|
| [Orca](papers/orca.md) | 2022 | OSDI 2022 | — | Foundational `S`; policy intertwined with execution model. |
| [FastServe](papers/fastserve.md) | 2026 | NSDI 2026; arXiv first posted 2023 | — | Direct `S`; [code](https://github.com/LLMServe/FastServe). |
| [VTC](papers/vtc.md) | 2024 | OSDI 2024 | 4 | Direct `S+F`; must-have baseline. |
| [FairServe](papers/fairserve.md) | 2024 | arXiv (Cornell University) | 0 | Direct `A+S+F`. |
| [DLPM/D²LPM](papers/dlpm-d2lpm.md) | 2025 | arXiv (Cornell University) | 0 | Direct `R+S+F`. |
| [Andes](papers/andes.md) | 2024 | arXiv (Cornell University) | 6 | `S+F`; client token pacer is outside PLEX. |
| [Sarathi-Serve](papers/sarathi-serve.md) | 2024 | OSDI 2024 | 15 | `S`; mechanics-heavy but useful baseline. |
| [Response Length Perception and Sequence Scheduling](papers/response-length-perception-and-sequence-scheduling.md) | 2023 | arXiv (Cornell University) | 9 | Direct `S`. |
| [S³](papers/s3.md) | 2023 | NeurIPS 2023 | 7 | Direct `S`; compare with LTR. |
| [Efficient LLM Scheduling by Learning to Rank](papers/efficient-llm-scheduling-by-learning-to-rank.md) | 2024 | Advances in Neural Information Processing Systems 37 | 9 | Direct `S`; [code](https://github.com/hao-ai-lab/vllm-ltr). |
| [QLM](papers/qlm.md) | 2024 | SoCC 2024 | 11 | Direct multi-hook case. |
| [SOLA](papers/sola.md) | 2025 | MLSys 2025 | — | Direct `S+F`; no public code found. |
| [SLOs-Serve](papers/slos-serve.md) | 2025 | arXiv preprint, 2025 | — | Direct but high-effort `A+R+S`. |
| [Apt-Serve](papers/apt-serve.md) | 2025 | Proceedings of the ACM on Management of Data | 5 | `S+E`; [code](https://github.com/eddiegaoo/Apt-Serve). |
| [Chameleon](papers/chameleon.md) | 2018 | Proceedings of the 2018 Conference of the ACM Special Interest Group on Data Communication | 455 | `A+S+E`; strong LoRA case. |
| [PEEK](papers/peek.md) | 2026 | arXiv (Cornell University) | 0 | Direct coordinated `S+E`. |
| [LLM Query Scheduling with Prefix Reuse and Latency Constraints](papers/llm-query-scheduling-with-prefix-reuse-and-latency-constraints.md) | 2025 | arXiv (Cornell University) | 0 | Direct `S`; theory-friendly baseline. |
| [FastSwitch](papers/fastswitch.md) | 2024 | arXiv (Cornell University) | 1 | `S+E/P`; mechanisms are significant. |
| [PolyServe](papers/polyserve.md) | Unknown | Metadata not resolved | — | `A+R+S`; autoscaling is control-plane scope. |
| [SuperInfer](papers/superinfer.md) | 2026 | Zenodo (CERN European Organization for Nuclear Research) | 0 | `S+E+P`; Superchip-specific mechanics. |
| [PASCAL](papers/pascal.md) | 2026 | HPCA 2026 | 0 | `S`; useful reasoning holdout. |
| [Fairness-Aware and Latency-Controllable Scheduling for Chunked-Prefill LLM Serving](papers/fairness-aware-and-latency-controllable-scheduling-for-chunked-prefill-llm-serving.md) | 2026 | arXiv (Cornell University) | 0 | `S+F`; recent holdout. |
| [Regulating Branch Parallelism in LLM Serving](papers/regulating-branch-parallelism-in-llm-serving.md) | 2026 | arXiv (Cornell University) | 0 | Boundary test for branch-level `A/S`. |
| [PARD](papers/pard.md) | 2026 | EuroSys 2026 | 34 | Important boundary test: mid-lifecycle drop is not cleanly `admit`-once. |
## Routing, placement, and rebalancing

| Paper | Year | Venue / status | Citations | PLEX |
|---|---:|---|---:|---|
| [Preble](papers/preble.md) | 2025 | ICLR 2025 | 1 | Existing direct `R`. |
| [LMetric](papers/lmetric.md) | 2026 | OSDI 2026 | 0 | Direct `R`; strongest simple route example. |
| [DualMap](papers/dualmap.md) | 2026 | ICLR 2026 | — | Direct `R+B`. |
| [Llumnix](papers/llumnix.md) | 2024 | OSDI 2024 | 4 | `R+B+F`; canonical rebalance paper. |
| [Intelligent Router for LLM Workloads](papers/intelligent-router-for-llm-workloads.md) | 2024 | arXiv (Cornell University) | 0 | `R`; learned/predictive alternative. |
| [SkyWalker](papers/skywalker.md) | 2009 | 2009 IEEE International Conference on Rehabilitation Robotics | 27 | `R`; network/cost facts required. |
| [RouteBalance](papers/routebalance.md) | 2026 | arXiv (Cornell University) | 0 | `R`; include only if model choice is in route scope. |
| [DistServe](papers/distserve.md) | 2024 | OSDI 2024 | 15 | Corpus/adapter stress; much is offline mechanism. |
| [Splitwise](papers/splitwise.md) | 2024 | ISCA 2024 | 186 | Corpus, not a compact online policy replica. |
| [Mooncake](papers/mooncake.md) | 2025 | FAST 2025 | 13 | `R+E/P`; separate policy from cache transport. |
| [MemServe](papers/memserve.md) | 2024 | arXiv (Cornell University) | 2 | `R+E/P`; mechanics-heavy. |
| [P/D-Serve](papers/p-d-serve.md) | 2024 | arXiv (Cornell University) | 1 | `R`; corpus-only unless policy is isolated. |
| [DynaServe](papers/dynaserve.md) | 2025 | arXiv (Cornell University) | 0 | `R+B`; architecture-heavy. |
| [WindServe](papers/windserve.md) | 2025 | ISCA 2025 | 10 | `R+S`; corpus/holdout. |
| [Helix](papers/helix.md) | 2025 | Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 | 15 | Offline `R`; code available. |
| [Aladdin](papers/aladdin.md) | 2014 | ACM SIGARCH Computer Architecture News | 243 | `R`; control-plane corpus. |
| [DynamoLLM](papers/dynamollm.md) | 2025 | 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA) | 76 | `R+B`; control-plane corpus. |
| [AIBrix](papers/aibrix.md) | Unknown | Metadata not resolved | — | Deployment seam/target, not one source policy. |
| [dLoRA](papers/dlora.md) | 2024 | OSDI 2024 | — | `R+B`; excellent LoRA stress case. |
| [CaraServe](papers/caraserve.md) | 2024 | arXiv (Cornell University) | 1 | Direct `R+S`; CPU assist is mechanism. |
| [POLAR](papers/polar.md) | 2026 | arXiv (Cornell University) | 0 | `R+E+F`; recent holdout. |
## Residency, cache admission, eviction, and prefetch

| Paper | Year | Venue / status | Citations | PLEX |
|---|---:|---|---:|---|
| [SGLang / RadixAttention](papers/sglang-radixattention.md) | 2024 | NeurIPS 2024 | 34 | Canonical `S+E`; public implementation. |
| [RAGCache](papers/ragcache.md) | 2024 | arXiv preprint, 2024 | 5 | Very clean direct `E` replica. |
| [Marconi](papers/marconi.md) | 2025 | MLSys 2025 | — | `E+F`; cache admission exposes a surface gap. |
| [HotPrefix](papers/hotprefix.md) | 2026 | PACMMOD / SIGMOD 2026 | 5 | `E+P+F`; cache admission exposes the same gap. |
| [UniCache](papers/unicache.md) | 2026 | ASPLOS 2026 | 0 | Direct `E`; recent holdout. |
| [CachedAttention](papers/cachedattention.md) | 2024 | USENIX ATC 2024 | 1 | `E+P`; much of the contribution is storage mechanics. |
| [Pensieve](papers/pensieve.md) | 2010 | Proceedings of the SIGCHI Conference on Human Factors in Computing Systems | 170 | `E+P`; corpus/adapter stress. |
| [HCache](papers/hcache.md) | 2025 | EuroSys 2025 | 13 | `E+P`; action/mechanics-heavy. |
| [PRESERVE](papers/preserve.md) | 2025 | arXiv (Cornell University) | 0 | Optional `P`; corpus. |
| [InfiniGen](papers/infinigen.md) | 2024 | OSDI 2024 | 1 | Optional `P`; primarily mechanism. |
| [IC-Cache](papers/ic-cache.md) | 2025 | SOSP 2025 | 0 | `E/P`; corpus. |
| [Strata](papers/strata.md) | Unknown | Metadata not resolved | — | `E/P`; corpus. |
| [ECHO](papers/echo.md) | 2026 | OSDI 2026 | — | Optional `P`; mechanism-heavy. |
## Foundational pre-LLM serving policies

| Paper | Year | Venue / status | Citations | PLEX |
|---|---:|---|---:|---|
| [Clipper](papers/clipper.md) | 2017 | NSDI 2017 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [Nexus](papers/nexus.md) | 2019 | SOSP 2019 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [Clockwork](papers/clockwork.md) | 2020 | OSDI 2020 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [InferLine](papers/inferline.md) | 2020 | SoCC 2020 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [INFaaS](papers/infaas.md) | 2021 | USENIX ATC 2021 | 27 | Foundational related-work baseline; pre-LLM request/state model. |
| [SHEPHERD](papers/shepherd.md) | 2023 | NSDI 2023 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [AlpaServe](papers/alpaserve.md) | 2023 | OSDI 2023 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [GSLICE](papers/gslice.md) | 2020 | SoCC 2020 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [REEF](papers/reef.md) | 2022 | OSDI 2022 | — | Foundational related-work baseline; pre-LLM request/state model. |
| [MArk](papers/mark.md) | 2019 | USENIX ATC 2019 | 36 | Foundational related-work baseline; pre-LLM request/state model. |
| [PREMA](papers/prema.md) | 2020 | HPCA 2020 | 115 | Foundational related-work baseline; pre-LLM request/state model. |
| [Abacus](papers/abacus.md) | 2021 | SC 2021 | 57 | Foundational related-work baseline; pre-LLM request/state model. |
