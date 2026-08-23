# json-schema-constrained-decoding

JSON generation that masks every step to the tokens allowed by a caller-supplied
JSON Schema.

## Source

Geng et al., ***Grammar-Constrained Decoding for Structured NLP Tasks without
Finetuning*** (NeurIPS 2023) — <https://arxiv.org/abs/2305.13971>. Applies the
per-step valid-token masking rule to JSON Schema constraints.

**Faithfulness: Faithful to the constrained-decoding mechanism.** The inferlet
enforces an incremental grammar mask and validates the final JSON, but delegates
the schema automaton construction to the host helper rather than expressing it in
PTIR.

## What it does

Plain prompting can ask for JSON, but it cannot make invalid tokens impossible.
Grammar-constrained decoding does: after each accepted token, an incremental
matcher computes the set of tokens that can still lead to a valid completion, and
all other logits are masked away before selection.

This inferlet builds a JSON Schema constraint on the host, asks it for the
current allowed-token mask, and uses a PTIR epilogue to choose the argmax among
allowed logits. Generation stops only when the constraint reports termination;
the decoded text is then parsed as JSON as a final sanity check.

## The rule

```
state_0 = build_constraint(schema)

allowed_t = valid_next_tokens(state_t)
token_t   = argmax(mask(logits_t, allowed_t))
state_{t+1} = advance(state_t, token_t)

accept when state_t is terminated and decode(tokens) parses as JSON
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"Generate a profile for a fictional software engineer named Alice."` | User prompt to send to the model |
| `schema` | string | *(profile object schema)* | JSON Schema used to constrain the output |
| `max_tokens` | int | `512` | Maximum number of constrained tokens to generate |

## Implementation notes

The grammar mask is host-derived from the immediately preceding output token, so
this inferlet is structurally **depth-1**. It must submit one fire, wait for its
token, advance the matcher, publish the next mask, and only then submit the next
fire. The earlier run-ahead version silently reused a stale grammar mask, which
meant the schema was not actually enforced on those steps; the audit documents
that failure mode in
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

The device side is intentionally simple: `masked_argmax(intrinsics::logits(),
allowed)` in the epilogue. All schema semantics live in the host `JsonSchema`
constraint object.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```
