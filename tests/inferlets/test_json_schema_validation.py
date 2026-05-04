"""E2E test for json-schema-validation inferlet."""
import json

from conftest import run_inferlet, run_tests


async def test_json_schema_validation(client, args):
    output = await run_inferlet(
        client, "json-schema-validation",
        {"max_tokens": 1024},
        timeout=max(args.timeout, 300),
    )
    assert "Generated:" in output, "Missing 'Generated:' header"

    # The inferlet returns the validated JSON as its return value, which
    # `run_inferlet` appends at the end of `output`. Walk backwards from
    # the end through "{" candidates and pick the one that parses as JSON
    # consuming the rest of the output (= the trailing return-value JSON).
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
    for field in ("name", "age", "email", "skills", "address"):
        assert field in parsed, f"Missing '{field}' field in output"


if __name__ == "__main__":
    run_tests([test_json_schema_validation])
