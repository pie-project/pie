# Minimum PLEX trace schema

The public workload gap is primarily a schema gap. A reusable PLEX trace should
represent both application intent and enacted serving outcomes.

## Required envelope

```json
{
  "trace_id": "trace",
  "timestamp_ms": 0,
  "event": "create|continue|enqueue|progress|boundary|preempt|cancel|finish",
  "tenant_id": "tenant",
  "logical_request_id": "logical",
  "generation_id": "generation",
  "workflow_id": "workflow-or-null",
  "node_id": "node-or-null",
  "parent_ids": [],
  "model": "model",
  "adapter": null,
  "stage": "prefill|decode|tool|encoder",
  "input_tokens": 0,
  "output_tokens": 0,
  "prefix_block_hashes": [],
  "declared": {},
  "observed": {},
  "slo": {},
  "outcome": {}
}
```

## Required event families

1. Lifecycle: create, continue, finish, cancel, expiry.
2. Service: enqueue, selected, token budget, progress, preempt.
3. Placement: route candidates, chosen target, migration.
4. Residency: cache insert, hit, evict, prefetch, load failure.
5. External boundaries: tool start/result/timeout, user wait, retrieval.
6. Workflow: fork, join, node-ready, dependency-resolved.
7. Trust: declared versus observed length, urgency, reuse, and tool duration.

## Privacy-preserving content

Raw text is optional. Token-block hashes, lengths, content class, and stable
session/workflow identities are sufficient for most serving-policy studies.
Hashes must be salted and domain-remapped as in the Qwen-Bailian release.

## Outcome requirements

Every trace should retain final task success or quality where meaningful, not
only latency. Reasoning, model-routing, RAG, and agent policies otherwise cannot
be evaluated for goodput or marginal utility.
