# Methodology and scoring

## Scope

The survey starts from the 87 serving-policy papers in
`plex-serving-policy-wiki/`, then expands to public production traces,
conversation corpora, agent benchmarks, reasoning/RAG/multimodal datasets, and
serving simulators available by 2026-07-23.

Catalog size:

- Existing public sources: 42
- Proposed PLEX workloads: 21

## Workload classes

1. **Production request traces** preserve arrivals and lengths but often remove
   content, tenants, SLOs, failures, and application structure.
2. **Conversation/content corpora** preserve semantics and sessions but lack
   serving timing and infrastructure outcomes.
3. **Agent benchmarks** preserve tools, state, and task success but require an
   instrumented run to become serving traces.
4. **Reasoning/RAG/multimodal benchmarks** expose compute and quality diversity
   but are content-only.
5. **Simulators/generators** provide scalable counterfactual experiments but are
   only as faithful as their fitted dimensions.

## Coverage labels

- `yes`: explicitly present in the public source.
- `partial`: present but incomplete or only indirectly useful.
- `derived`: can be reconstructed from content or execution.
- `generated`: produced by a workload generator rather than observed.
- domain labels such as `chain`, `tree`, `audio`, or `GUI`: explicit structure
  of that kind.
- `no`: absent.

## Selection principle

A core PLEX workload must add a policy-relevant dimension that another core
source does not provide. No single source is treated as representative of all
serving traffic.
