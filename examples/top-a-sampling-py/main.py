"""Top-a sampling — the Python twin of `top-a-sampling`.

`keep(x) iff p(x) >= a · p_max²`: one softmax, one `reduce_max`, one
multiply, one comparison over the vocabulary, then a Gumbel-max draw over the
masked logits. Exercises `softmax`/`reduce_max`/`ge`/`select`/`cast`/
`broadcast` and the two f32 stat drains beside the token channel.
"""

from inferlet import chat, model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    broadcast,
    cast,
    channel_capacity,
    dtype,
    ge,
    gumbel_max,
    intrinsics,
    iota,
    reduce_max,
    reduce_sum,
    reshape,
    select,
    softmax,
)


def top_a_keep(logits, vocab: int, a: float):
    """The top-a keep-mask: `(keep, kept_count, kept_mass)`."""
    probs = softmax(logits)
    p_max = reduce_max(probs)
    threshold = a * (p_max * p_max)
    keep = ge(probs, broadcast(threshold, [vocab]))
    zeros = broadcast(0.0, [vocab])
    kept_mass = reshape(reduce_sum(select(keep, probs, zeros)), [1])
    kept = reshape(reduce_sum(cast(keep, dtype.f32)), [1])
    return keep, kept, kept_mass


def step(logits, vocab: int, a: float, temperature: float, rng_state):
    scaled = logits if temperature == 1.0 else logits / temperature
    keep, kept, kept_mass = top_a_keep(scaled, vocab, a)
    neg_inf = broadcast(float("-inf"), [vocab])
    masked = select(keep, scaled, neg_inf)
    return gumbel_max(masked, rng_state), kept, kept_mass


async def main(input: dict) -> dict:
    prompt_text = input.get("prompt", "Write a short paragraph about top-a sampling.")
    a = float(input.get("a", 0.2))
    temperature = float(input.get("temperature", 1.0))
    max_tokens = int(input.get("max_tokens", 32))
    seed = int(input.get("seed", 0x7CE1))
    if not (0.0 < a <= 1.0):
        raise ValueError("a must be finite and in (0, 1]")
    if not (temperature > 0.0) or temperature == float("inf"):
        raise ValueError("temperature must be finite and greater than 0")

    vocab = model.output_vocab_size()
    ws = WorkingSet()
    pipe = Pipeline()

    # The model's opening (`<bos>` where it has one) before the raw text: a
    # gemma without it answers noise.
    prompt = chat.prefix() + model.encode(prompt_text)
    if not prompt:
        raise ValueError("empty prompt")
    *head, last = prompt
    if head:
        fill = ForwardPass()
        fill.embed(Channel.from_(head, dtype.i32), Channel.from_([0, len(head)], dtype.u32))
        fill.state([ws], StateGeometry.folding())
        pipe.submit(fill)

    cap = channel_capacity()
    tok_in = Channel.from_([last], dtype.i32)
    rng = Channel.from_([seed, 0], dtype.u32)
    tok_out = Channel([1], dtype.i32).capacity(cap)
    s1_out = Channel([1], dtype.f32).capacity(cap)
    s2_out = Channel([1], dtype.f32).capacity(cap)

    fwd = ForwardPass()
    fwd.embed(tok_in, Channel.from_([0, 1], dtype.u32))
    fwd.state([ws], StateGeometry.folding())

    @fwd.epilogue
    def _():
        r = rng.take()
        logits = intrinsics.output(0, "logits")
        token, kept, mass = step(logits, vocab, a, temperature, r)
        r_next = r + iota(2)
        tok_in.put(token)
        tok_out.put(token)
        s1_out.put(kept)
        s2_out.put(mass)
        rng.put(r_next)

    generated: list[int] = []
    s1: list[float] = []
    s2: list[float] = []

    async def on_token() -> bool:
        generated.append(await tok_out.take_scalar())
        s1.append(await s1_out.take_scalar())
        s2.append(await s2_out.take_scalar())
        return True

    await pipe.run_ahead(fwd, max_tokens, on_token)
    pipe.close()

    def mean(v):
        return sum(v) / max(len(v), 1)

    min_s1 = max(min(s1, default=0.0), 0.0)
    if s1 and min_s1 == 0.0:
        raise RuntimeError("top-a keep-set was empty — the peak token was masked out")
    return {
        "sampler": "top-a",
        "text": model.decode(generated),
        "count": len(generated),
        "a": a,
        "mean_kept": mean(s1),
        "min_kept": int(min_s1),
        "mean_mass": mean(s2),
        "tokens": generated,
    }
