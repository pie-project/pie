"""E2E test for js-constrained-decoding inferlet.

Verifies that the JavaScript SDK drives a stateful grammar matcher per
generated token (regression test for the constrained-decoding port).
Without per-step matcher driving, output is unconstrained after the first
token and would fail JSON parsing or schema conformance.

Also exercises the bakery wrapper's structured-return support: the
inferlet returns a `Person` object directly and the wrapper auto-stringifies.
"""
import json

from conftest import run_inferlet, run_tests


async def test_js_constrained_decoding(client, args):
    output = await run_inferlet(
        client, "js-constrained-decoding",
        {"max_tokens": 512},
        timeout=max(args.timeout, 300),
    )
    assert "Hello " in output, "Missing greeting (typed-field access)"
    assert "Skills:" in output, "Missing skills line"
    assert "[done]" in output, "Missing [done] marker"

    # Trailing return value: structured object auto-stringified by the
    # bakery wrapper. Walk back from the end through "{" candidates and
    # pick one that parses cleanly.
    decoder = json.JSONDecoder()
    parsed = None
    start = len(output)
    while True:
        start = output.rfind("{", 0, start)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(output[start:])
            if end == len(output) - start:
                parsed = obj
                break
        except json.JSONDecodeError:
            continue
    assert parsed is not None, "No trailing JSON object in output"
    for field in ("name", "age", "email", "skills"):
        assert field in parsed, f"Missing '{field}' in output: {parsed}"

    # Schema conformance — enforced by the grammar.
    assert isinstance(parsed["name"], str) and parsed["name"], "name must be non-empty string"
    assert isinstance(parsed["age"], int) and 0 <= parsed["age"] <= 150, \
        f"age must be int in [0,150], got {parsed['age']!r}"
    assert isinstance(parsed["email"], str), "email must be string"
    assert isinstance(parsed["skills"], list) and len(parsed["skills"]) >= 1, \
        "skills must be non-empty list"
    assert all(isinstance(s, str) for s in parsed["skills"]), \
        "every skill must be a string"


if __name__ == "__main__":
    run_tests([test_js_constrained_decoding])
