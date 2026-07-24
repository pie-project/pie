# Methodology and limitations

## Inclusion rule

A work is included when it contains an explicit serving decision rule for
admission, routing, scheduling, token budgeting, residency, eviction, prefetch,
feedback-driven adaptation, migration, or rebalancing. Mechanism papers remain
only when a separable control policy can be identified.

## Sources

- Current PLEX paper and policy fixtures.
- Eight maintained LLM-serving, scheduling, KV-cache, and agent-system lists.
- Fifteen OpenAlex searches yielding 974 unique candidate records.
- Primary arXiv, proceedings, conference, DOI, and artifact pages.

## Coverage

- Paper pages: 87
- Metadata records resolved: 87
- Indexed abstracts available: 71
- Papers with named datasets/workloads identified: 34
- Papers with a categorical workload description: 87

## Citation counts

Counts are exact-record OpenAlex values when available and exact-record Crossref
`is-referenced-by-count` values otherwise, retrieved on 2026-07-23. They can
lag, merge imperfectly, or split between arXiv and proceedings records. Do not
use them as the sole quality signal. Entries without an exact count are marked
unresolved rather than assigned a fuzzy-search value.

## Group reputation

The wiki does not assign an unqualified “reputable/not reputable” label.
Instead it records:

1. peer-reviewed venue or preprint status;
2. author institutions when resolvable; and
3. public artifact availability.

The resulting “reputation evidence” is a transparent signal, not an endorsement
of correctness or reproducibility.

## Abstracts and summaries

Each page contains an editorial, non-verbatim abstract synopsis plus a separate
serving-policy summary. The primary paper remains authoritative.

## Datasets

Dataset/workload names come from a manual audit or indexed primary text.
When no named public dataset is confirmed, the page supplies a conservative
categorical workload description and marks it as such. References mentioned
only in related work are not intentionally treated as evaluation datasets.
