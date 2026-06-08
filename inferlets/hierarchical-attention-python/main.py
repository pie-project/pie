"""Hierarchical-attention inferlet in Python.

Mirrors the canonical Rust version (`hierarchical-attention-rust`). A long prompt
is split into chunks; during a manual decode loop, each step builds a BRLE
attention mask that keeps a *hierarchy* visible:

    global instructions      (sink tokens at the start)
      chunk summaries        (a header range per chunk)
        selected chunk(s)    (the full body of the most relevant chunk)
          recent window      (a sliding window at the end)

This uses manual `ctx.forward()` passes because `ctx.generate()` does not expose
a different attention mask per generation step. See the README for how this
differs from `attention-sink` / `windowed-attention`. The pure helpers
(`split_words`, `select_relevant_chunks`, `merge_ranges`, `build_brle_mask`) are
model-free and unit-tested in `test_hierarchical_attention.py`.
"""

from inferlet import Context, Model, Sampler, chat, runtime


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================


def split_words(text: str, chunk_words: int) -> list[str]:
    """Split text into chunks of at most ``chunk_words`` words. An empty prompt
    yields a single empty chunk so range bookkeeping always has a target."""
    words = text.split()
    if not words:
        return [""]
    n = max(1, chunk_words)
    return [" ".join(words[i : i + n]) for i in range(0, len(words), n)]


def summarize_words(text: str, n: int) -> str:
    """First ``n`` words of ``text`` — the stand-in 'summary' for a chunk."""
    return " ".join(text.split()[:n])


def select_relevant_chunks(chunks: list[str], query: str, k: int) -> list[int]:
    """Pick the ``k`` chunks with the highest lexical overlap with ``query``.

    Overlap = count of query words (>3 chars, lowercased) appearing in the
    chunk. Ties break toward the earlier chunk. Returned indices are in
    positional order. Always returns at least one index when chunks exist.
    """
    if not chunks:
        return []
    q = {w.lower() for w in query.split() if len(w) > 3}
    scored = []
    for i, chunk in enumerate(chunks):
        score = sum(1 for w in chunk.split() if w.lower() in q)
        scored.append((i, score))
    # Sort by score desc, then index asc.
    scored.sort(key=lambda t: (-t[1], t[0]))
    take = max(1, min(k, len(chunks)))
    return sorted(i for i, _ in scored[:take])


def merge_ranges(ranges: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    """Clip ranges to ``[0, total)``, drop empties, sort, and merge overlapping
    or touching ranges into a minimal disjoint set."""
    clipped = []
    for start, end in ranges:
        start = max(0, min(total, start))
        end = max(0, min(total, end))
        if start < end:
            clipped.append((start, end))
    clipped.sort()

    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:  # `<=` merges touching ranges
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_brle_mask(total: int, ranges: list[tuple[int, int]]) -> list[int]:
    """BRLE attention mask over ``[0, total)`` keeping ``ranges``.

    BRLE alternates run lengths starting with a **false** run:
    ``[false, true, false, true, ...]``. ``[0, total]`` means 'all true'.

    An empty / fully-clipped keep-set returns ``[0, total]`` (all-true / no
    restriction), never an all-false mask (which the runtime rejects as
    attending to nothing).
    """
    merged = merge_ranges(ranges, total)
    if not merged:
        return [0, total]

    out: list[int] = []
    cursor = 0
    for start, end in merged:
        out.append(start - cursor)  # false run up to this range
        out.append(end - start)     # true run covering this range
        cursor = end
    if cursor < total:
        out.append(total - cursor)  # trailing false run
    return out


def mask_true_count(mask: list[int]) -> int:
    """Number of attended positions = sum of the true runs (odd indices)."""
    return sum(mask[1::2])


def _fmt_ranges(ranges: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"[{s},{e})" for s, e in ranges) + "]"


# =============================================================================
# Entry point
# =============================================================================


async def main(input: dict) -> str:
    model = Model.load(runtime.models()[0])
    tokenizer = model.tokenizer()

    prompt = input.get(
        "prompt",
        "Explain how LLM serving systems use KV cache, batching, scheduling, and "
        "attention masks. Include one practical example.",
    )
    max_tokens = int(input.get("max_tokens", 128))
    chunk_words = max(8, int(input.get("chunk_size_words", 80)))
    sink_tokens = int(input.get("sink_tokens", 64))
    summary_tokens_per_chunk = int(input.get("summary_tokens_per_chunk", 24))
    local_window_tokens = int(input.get("local_window_tokens", 128))
    selected_chunks = max(1, int(input.get("selected_chunks", 1)))
    selection_mode = input.get("selection_mode", "lexical")

    chunks = split_words(prompt, chunk_words)
    selected = select_relevant_chunks(chunks, prompt, selected_chunks)

    prompt_tokens: list[int] = []
    summary_ranges: list[tuple[int, int]] = []
    full_ranges: list[tuple[int, int]] = []

    prompt_tokens.extend(
        chat.system(
            model,
            "You are a concise assistant. Use the visible hierarchy: global "
            "instructions, the chunk summaries, and the selected local chunk.",
        )
    )

    for i, chunk in enumerate(chunks):
        header = f"Chunk {i} summary: {summarize_words(chunk, 20)}\n"
        body = f"Chunk {i} full text:\n{chunk}\n"

        header_start = len(prompt_tokens)
        prompt_tokens.extend(chat.user(model, header))
        header_end = len(prompt_tokens)
        summary_ranges.append(
            (header_start, min(header_end, header_start + summary_tokens_per_chunk))
        )

        body_start = len(prompt_tokens)
        prompt_tokens.extend(chat.user(model, body))
        body_end = len(prompt_tokens)
        full_ranges.append((body_start, body_end))

    prompt_tokens.extend(
        chat.user(
            model,
            "Answer the original request using the selected local chunk(s) and the "
            "global chunk summaries.",
        )
    )
    prompt_tokens.extend(chat.cue(model))

    print("--- hierarchical-attention-python ---")
    print(f"chunks={len(chunks)}")
    print(f"selected_chunk={selected} (mode={selection_mode})")
    print(f"summary_ranges={_fmt_ranges(summary_ranges)}")
    print(f"full_ranges={_fmt_ranges(full_ranges)}")

    ctx = Context(model)
    pending = prompt_tokens
    generated: list[int] = []
    stop_tokens = set(chat.stop_tokens(model))
    logged_mask = False

    for _ in range(max_tokens):
        if not pending:
            break

        fwd = ctx.forward()
        total_after = fwd.start_position() + len(pending)

        keep: list[tuple[int, int]] = []
        keep.append((0, min(sink_tokens, total_after)))           # 1. sink
        keep.extend(summary_ranges)                               # 2. summaries
        for idx in selected:                                      # 3. selected bodies
            if idx < len(full_ranges):
                keep.append(full_ranges[idx])
        keep.append((max(0, total_after - local_window_tokens), total_after))  # 4. window

        mask = build_brle_mask(total_after, keep)

        if not logged_mask:
            print(f"mask_true_tokens={mask_true_count(mask)} / total={total_after}")
            logged_mask = True

        fwd.input(pending)
        # One shared mask per query token in this pass (MVP simplification).
        fwd.attention_mask([list(mask) for _ in pending])

        h = fwd.sample([len(pending) - 1], Sampler.argmax())
        out = await fwd.execute()
        token = out.token(h)

        if token is None or token in stop_tokens:
            break

        generated.append(int(token))
        pending = [int(token)]

    print(f"generated_tokens={len(generated)}")
    return tokenizer.decode(generated)
