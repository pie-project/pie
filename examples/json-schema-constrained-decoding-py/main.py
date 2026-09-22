"""JSON generation constrained by a JSON Schema — the Python twin of
`json-schema-constrained-decoding`.

The host grammar matcher advances after every accepted token and supplies
the next allowed-token mask to an ETA `masked_argmax` epilogue. Exercises the
`grammar` + `mask` modules and a host-writer mask channel, and runs on any
pass kind through `bind_state`.
"""

import json

from inferlet import chat, grammar, mask, model
from inferlet.eta import (
    Channel,
    ForwardKind,
    ForwardPass,
    KvGeometry,
    Pipeline,
    RsWorkingSet,
    WorkingSet,
    channel_capacity,
    dtype,
    indptr,
    intrinsics,
    kv_page_size,
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
    kind = model.pass_kind()
    ws = WorkingSet()
    rs_ws = [RsWorkingSet()] if kind != ForwardKind.ATTENTION else []
    page_size = kv_page_size()
    constraint = grammar.Matcher(grammar.Grammar.from_json_schema(schema))

    prompt = chat.system_user(
        "Generate only the requested JSON value, with no markdown or explanation.", prompt_text
    ) + chat.cue()
    if not prompt:
        prompt = [0]
    n = len(prompt)
    max_pages = max(-(-(n + max_tokens + 1) // page_size), 1)
    ws.reserve(max_pages)

    prompt_tokens = Channel.from_(prompt, dtype.i32)
    prefill_indptr = Channel.from_([0, n], dtype.u32).named("prefill_indptr")
    prefill_positions = Channel.from_(range(n), dtype.u32).named("prefill_positions")
    prefill_pages = Channel.from_(range(max_pages), dtype.u32).named("prefill_pages")
    prefill_page_indptr = Channel.from_([0, -(-n // page_size)], dtype.u32).named("prefill_page_indptr")
    prefill_w_slot = Channel.from_([p // page_size for p in range(n)], dtype.u32).named("prefill_w_slot")
    prefill_w_off = Channel.from_([p % page_size for p in range(n)], dtype.u32).named("prefill_w_off")
    prefill_mask = Channel([vocab], dtype.bool).named("prefill_mask")
    first_out = Channel([1], dtype.i32).named("first_token")
    prefill_kv_len = Channel.from_([n], dtype.u32).named("prefill_kv_len")

    prefill = ForwardPass(kind)
    prefill.embed(prompt_tokens, prefill_indptr)
    prefill.bind_state(
        ws,
        KvGeometry(
            kv_len=prefill_kv_len,
            pages=prefill_pages,
            page_indptr=prefill_page_indptr,
            w_slot=prefill_w_slot,
            w_off=prefill_w_off,
            positions=prefill_positions,
        ),
        rs_ws,
    )

    @prefill.epilogue
    def _prefill():
        allowed = prefill_mask.take()
        first_out.put(reshape(masked_argmax(intrinsics.logits(), allowed), [1]))

    prefill_mask.put(mask.unpack_mask(constraint.mask(), vocab))
    pipeline = Pipeline()
    prefill.submit(pipeline)
    first = await first_out.take_scalar()

    generated = [first]
    constraint.accept_tokens([first])

    if not constraint.is_terminated() and len(generated) < max_tokens:
        token_in = Channel.from_([first], dtype.i32).named("token_in")
        grammar_mask = Channel([vocab], dtype.bool).named("grammar_mask")
        embed_indptr = Channel.from_([0, 1], dtype.u32).named("embed_indptr")
        positions = Channel.from_([n], dtype.u32).named("positions")
        pages = Channel.from_(range(max_pages), dtype.u32).named("pages")
        page_indptr = Channel.from_([0, -(-(n + 1) // page_size)], dtype.u32).named("page_indptr")
        w_slot = Channel.from_([n // page_size], dtype.u32).named("w_slot")
        w_off = Channel.from_([n % page_size], dtype.u32).named("w_off")
        token_out = Channel([1], dtype.i32).capacity(channel_capacity()).named("token_out")
        kv_len = Channel.from_([n + 1], dtype.u32).named("kv_len")

        decode = ForwardPass(kind)
        decode.embed(token_in, embed_indptr)
        decode.bind_state(
            ws,
            KvGeometry(
                kv_len=kv_len,
                pages=pages,
                page_indptr=page_indptr,
                w_slot=w_slot,
                w_off=w_off,
                positions=positions,
                writable_pages=n // page_size,
            ),
            rs_ws,
        )

        @decode.epilogue
        def _decode():
            length = kv_len.take()
            allowed = grammar_mask.take()
            token = reshape(masked_argmax(intrinsics.logits(), allowed), [1])
            next_length = length + 1
            page_count = next_length.div_ceil(page_size)
            token_in.put(token)
            kv_len.put(next_length)
            positions.put(length)
            w_slot.put(length // page_size)
            w_off.put(length % page_size)
            page_indptr.put(indptr(1, page_count))
            token_out.put(token)

        # The grammar mask for fire k+1 is only known once fire k's token has
        # advanced the matcher, so this loop is inherently depth-1.
        budget = max_tokens - len(generated)
        submitted = 0
        while submitted < budget:
            grammar_mask.put(mask.unpack_mask(constraint.mask(), vocab))
            decode.submit(pipeline)
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
