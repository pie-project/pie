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

    vocab = model.output_vocab_size()
    ws = WorkingSet()
    constraint = grammar.Matcher(grammar.Grammar.from_json_schema(schema))

    prompt = chat.system_user(
        "Generate only the requested JSON value, with no markdown or explanation.", prompt_text
    ) + chat.cue()
    if not prompt:
        raise ValueError("empty prompt")
    *head, last = prompt

    pipeline = Pipeline()
    if head:
        fill = ForwardPass()
        fill.embed(Channel.from_(head, dtype.i32), Channel.from_([0, len(head)], dtype.u32))
        fill.state([ws], StateGeometry.folding())
        pipeline.submit(fill)

    token_in = Channel.from_([last], dtype.i32)
    grammar_mask = Channel([vocab], dtype.bool)
    embed_indptr = Channel.from_([0, 1], dtype.u32)
    token_out = Channel([1], dtype.i32).capacity(channel_capacity())

    decode = ForwardPass()
    decode.embed(token_in, embed_indptr)
    decode.state([ws], StateGeometry.folding())

    @decode.epilogue
    def _decode():
        allowed = grammar_mask.take()
        token = reshape(masked_argmax(intrinsics.output(0, "logits"), allowed), [1])
        token_in.put(token)
        token_out.put(token)

    # The grammar mask for fire k+1 is only known once fire k's token has
    # advanced the matcher, so this loop is inherently depth-1.
    generated: list[int] = []
    while not constraint.is_terminated() and len(generated) < max_tokens:
        grammar_mask.put(mask.unpack_mask(constraint.mask(), vocab))
        pipeline.submit(decode)
        token = await token_out.take_scalar()
        generated.append(token)
        constraint.accept_tokens([token])
    pipeline.close()

    if not constraint.is_terminated():
        raise RuntimeError(f"JSON generation did not terminate within {max_tokens} tokens")

    text = model.decode(generated)
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"constraint terminated with invalid JSON: {e}; output={text!r}") from None
    return text
