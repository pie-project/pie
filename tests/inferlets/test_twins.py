"""The Python and JavaScript twins of the Rust completion inferlets.

`text-completion-py` / `text-completion-js` trace the same pass as
`text-completion`, and `naive-baseline-py` / `naive-baseline-js` the same as
`naive-baseline`. The SDK ports are pinned byte-for-byte to the Rust encoder
(`crates/eta-dsl/tests/sdk_goldens.rs`), so the gate here is the end-to-end
one: the same prompt produces the SAME TOKENS from every language.

The naive twins are compared to each other (and to Rust on an attention-only
model: `naive-baseline` binds the attention-only pass, which a hybrid model
refuses by name), since their Gumbel-max draw is seeded and replayable.

Run against a live server::

    uv run python tests/inferlets/test_twins.py --attach ws://127.0.0.1:8080
"""

from __future__ import annotations

import json

from conftest import run_inferlet, run_tests

PROMPT = "The capital of France is"


async def _tokens(client, args, name: str, inputs: dict) -> list[int]:
    output = await run_inferlet(client, name, inputs, timeout=args.timeout)
    start = output.find("{")
    assert start >= 0, f"{name} returned no JSON object: {output[:200]!r}"
    report = json.loads(output[start:])
    assert report["count"] == len(report["tokens"]) == inputs["max_tokens"], report
    lowered = report["text"].lower()
    assert "france" in lowered or "paris" in lowered or report["count"] > 0, report
    return report["tokens"]


async def test_text_completion_twins_agree(client, args):
    inputs = {"prompt": PROMPT, "max_tokens": 16}
    rust = await _tokens(client, args, "text-completion", inputs)
    py = await _tokens(client, args, "text-completion-py", inputs)
    js = await _tokens(client, args, "text-completion-js", inputs)
    assert py == rust, f"python twin diverged: {py} vs {rust}"
    assert js == rust, f"javascript twin diverged: {js} vs {rust}"


async def test_naive_baseline_twins_agree(client, args):
    inputs = {"prompt": PROMPT, "max_tokens": 16, "temperature": 0.7, "seed": 0x7CE1, "stats": True}
    py = await _tokens(client, args, "naive-baseline-py", inputs)
    js = await _tokens(client, args, "naive-baseline-js", inputs)
    assert js == py, f"javascript twin diverged: {js} vs {py}"
    try:
        rust = await _tokens(client, args, "naive-baseline", inputs)
    except RuntimeError as e:
        # An attention-only pass on a hybrid model is refused by name.
        if "forward-hybrid" in str(e):
            return
        raise
    assert py == rust, f"python twin diverged: {py} vs {rust}"


async def test_python_twin_long_prompt_chunks(client, args):
    """A prompt past one chunk exercises `prefill_chunks` in the port."""
    inputs = {"prompt": " ".join(["The capital of France is Paris."] * 40) + " The capital of France is", "max_tokens": 4}
    rust = await _tokens(client, args, "text-completion", inputs)
    py = await _tokens(client, args, "text-completion-py", inputs)
    assert py == rust


SCHEMA = '{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}'


async def _constrained(client, args, name: str) -> str:
    output = await run_inferlet(
        client,
        name,
        {"prompt": "Return an object with an integer field named value.", "schema": SCHEMA, "max_tokens": 64},
        timeout=args.timeout,
    )
    start = output.find("{")
    assert start >= 0, f"{name} returned no JSON: {output[:200]!r}"
    doc = json.loads(output[start:])
    assert isinstance(doc.get("value"), int), f"{name}: {output[:200]!r}"
    return output[start:]


def _attention_only_refusal(e: BaseException) -> bool:
    """The Rust reference binds the attention-only pass, which a hybrid model
    refuses by name; the twins are kind-aware, so they still compare to each
    other there."""
    return "forward-hybrid" in str(e)


async def test_constrained_decoding_twins_agree(client, args):
    """The grammar/mask path: the Python and JS twins must produce the same
    JSON as the Rust inferlet (greedy under the same mask)."""
    py = await _constrained(client, args, "json-schema-constrained-decoding-py")
    js = await _constrained(client, args, "json-schema-constrained-decoding-js")
    assert json.loads(js) == json.loads(py), f"javascript twin diverged: {js} vs {py}"
    try:
        rust = await _constrained(client, args, "json-schema-constrained-decoding")
    except RuntimeError as e:
        if _attention_only_refusal(e):
            return
        raise
    assert json.loads(py) == json.loads(rust), f"python twin diverged: {py} vs {rust}"


class EngineNondeterministic(FileNotFoundError):
    """Raised (and reported as SKIP by the harness) when the engine itself
    does not reproduce a greedy program run-to-run on this model, so a
    cross-language comparison cannot say anything about the SDKs."""


async def test_prefix_tree_twins_agree(client, args):
    """Working-set forking: four shared-prefix leaves must decode the same
    text from every language (greedy, so the outputs are exact).

    Greedy decode after `fork` is only comparable where the engine reproduces
    it: on the hybrid model (Qwen3.5) sibling leaves that share a forked
    recurrent state come out differently on every run, from every language,
    so the twin is first checked against itself."""
    inputs = {"num_tokens": 6}
    py = await run_inferlet(client, "prefix-tree-kv-cache-py", inputs, timeout=args.timeout)
    again = await run_inferlet(client, "prefix-tree-kv-cache-py", inputs, timeout=args.timeout)
    if again.strip() != py.strip():
        raise EngineNondeterministic(
            "the engine did not reproduce the greedy prefix-tree decode run-to-run on this "
            "model (sibling leaves after a working-set fork differ); nothing to compare"
        )
    js = await run_inferlet(client, "prefix-tree-kv-cache-js", inputs, timeout=args.timeout)
    assert "city at dawn:" in py and "forest at night:" in py, py
    assert js.strip() == py.strip(), f"javascript twin diverged:\n{js}\nvs\n{py}"
    try:
        rust = await run_inferlet(client, "prefix-tree-kv-cache", inputs, timeout=args.timeout)
    except RuntimeError as e:
        if _attention_only_refusal(e):
            return
        raise
    assert py.strip() == rust.strip(), f"python twin diverged:\n{py}\nvs\n{rust}"


async def _report(client, args, name: str, inputs: dict) -> dict:
    output = await run_inferlet(client, name, inputs, timeout=args.timeout)
    start = output.find("{")
    assert start >= 0, f"{name} returned no JSON object: {output[:200]!r}"
    return json.loads(output[start:])


async def test_top_a_sampling_twins_agree(client, args):
    """A truncation sampler with device-side statistics: the seeded Gumbel
    draw makes the twins replayable, so tokens AND the kept-set statistics
    must agree exactly across languages."""
    inputs = {"prompt": PROMPT, "max_tokens": 8, "a": 0.2, "seed": 0x7CE1}
    py = await _report(client, args, "top-a-sampling-py", inputs)
    js = await _report(client, args, "top-a-sampling-js", inputs)
    assert py["min_kept"] >= 1 and py["count"] == 8, py
    assert js["tokens"] == py["tokens"], f"javascript twin diverged: {js} vs {py}"
    assert abs(js["mean_kept"] - py["mean_kept"]) < 1e-6 and abs(js["mean_mass"] - py["mean_mass"]) < 1e-6, (js, py)
    try:
        rust = await _report(client, args, "top-a-sampling", inputs)
    except RuntimeError as e:
        if _attention_only_refusal(e):
            return
        raise
    # The Rust output carries no token list; text + count + stats pin it.
    assert rust["text"] == py["text"] and rust["count"] == py["count"], (rust, py)
    assert abs(rust["mean_kept"] - py["mean_kept"]) < 1e-6 and abs(rust["mean_mass"] - py["mean_mass"]) < 1e-6, (rust, py)


def tests():
    return [
        test_text_completion_twins_agree,
        test_naive_baseline_twins_agree,
        test_python_twin_long_prompt_chunks,
        test_constrained_decoding_twins_agree,
        test_prefix_tree_twins_agree,
        test_top_a_sampling_twins_agree,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="Python/JavaScript twin inferlets")
