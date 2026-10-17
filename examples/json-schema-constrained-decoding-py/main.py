"""JSON generation constrained by a JSON Schema — the Python twin of
`json-schema-constrained-decoding`.

The host grammar matcher advances after every accepted token and supplies
the next allowed-token mask to an ETA `masked_argmax` epilogue. Exercises the
`grammar` + `mask` modules and a host-writer mask channel.
"""

import json

from inferlet import chat, grammar, mask, model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    channel_capacity,
    dtype,
    intrinsics,
    masked_argmax,
    reshape,
)

DEFAULT_SCHEMA = """{
    "type": "object",
    "properties": {
        "name": { "type": "string", "minLength": 1 },
        "age": { "type": "integer", "minimum": 0, "maximum": 150 },
        "skills": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        }
    },
    "required": ["name", "age", "skills"],
    "additionalProperties": false
}"""


async def main(input: dict) -> str:
    prompt_text = input.get("prompt", "Generate a profile for a fictional software engineer named Alice.")
    schema = input.get("schema", DEFAULT_SCHEMA)
    max_tokens = int(input.get("max_tokens", 512))
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    vocab = model.output_vocab_size()
    ws = WorkingSet()
    constraint = grammar.Matcher(grammar.Grammar.from_json_schema(schema))

    prompt = chat.system_user(
        "Generate only the requested JSON value, with no markdown or explanation.", prompt_text
    ) + chat.cue()
    if not prompt:
        prompt = [0]
    n = len(prompt)

    prompt_tokens = Channel.from_(prompt, dtype.i32)
    prefill_indptr = Channel.from_([0, n], dtype.u32).named("prefill_indptr")
    prefill_mask = Channel([vocab], dtype.bool).named("prefill_mask")
    first_out = Channel([1], dtype.i32).named("first_token")

    prefill = ForwardPass()
    prefill.embed(prompt_tokens, prefill_indptr)
    prefill.state([ws], StateGeometry.folding())

    @prefill.epilogue
    def _prefill():
        allowed = prefill_mask.take()
        first_out.put(reshape(masked_argmax(intrinsics.logits(), allowed), [1]))

    prefill_mask.put(mask.unpack_mask(constraint.mask(), vocab))
    pipeline = Pipeline()
    pipeline.submit(prefill)
    first = await first_out.take_scalar()

    generated = [first]
    constraint.accept_tokens([first])

    if not constraint.is_terminated() and len(generated) < max_tokens:
        token_in = Channel.from_([first], dtype.i32).named("token_in")
        grammar_mask = Channel([vocab], dtype.bool).named("grammar_mask")
        embed_indptr = Channel.from_([0, 1], dtype.u32).named("embed_indptr")
        token_out = Channel([1], dtype.i32).capacity(channel_capacity()).named("token_out")

        decode = ForwardPass()
        decode.embed(token_in, embed_indptr)
        decode.state([ws], StateGeometry.folding())

        @decode.epilogue
        def _decode():
            allowed = grammar_mask.take()
            token = reshape(masked_argmax(intrinsics.logits(), allowed), [1])
            token_in.put(token)
            token_out.put(token)

        # The grammar mask for fire k+1 is only known once fire k's token has
        # advanced the matcher, so this loop is inherently depth-1.
        budget = max_tokens - len(generated)
        submitted = 0
        while submitted < budget:
            grammar_mask.put(mask.unpack_mask(constraint.mask(), vocab))
            pipeline.submit(decode)
            submitted += 1
            token = await token_out.take_scalar()
            generated.append(token)
            constraint.accept_tokens([token])
            if constraint.is_terminated() or len(generated) == max_tokens:
                break
    pipeline.close()

    if not constraint.is_terminated():
        raise RuntimeError(f"JSON generation did not terminate within {max_tokens} tokens")

    text = model.decode(generated)
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"constraint terminated with invalid JSON: {e}; output={text!r}") from None
    return text
