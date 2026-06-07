"""E2E test for reflexion inferlet modes."""
import json

from conftest import run_inferlet, run_tests


def parse_result(output, mode):
    """Find the final structured result in the mixed stdout/return stream."""
    decoder = json.JSONDecoder()
    matches = []

    for start, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[start:])
        except json.JSONDecodeError:
            continue

        if not isinstance(value, dict):
            continue
        if mode == "both":
            if "baseline" in value and "reflexion" in value:
                matches.append(value)
        elif value.get("mode") == mode:
            matches.append(value)

    assert matches, f"Missing structured {mode} result"
    return matches[-1]


def assert_run_result(result, mode):
    assert result["mode"] == mode
    assert isinstance(result["success"], bool)
    assert isinstance(result["iterations"], int)
    assert isinstance(result["schedule"], str)
    assert isinstance(result["violations"], list)


async def test_reflexion(client, args):
    common = {
        "max_iterations": 1,
        "max_tokens": 64,
        "reflection_tokens": 64,
    }

    baseline = await run_inferlet(
        client,
        "reflexion",
        {**common, "mode": "baseline"},
        timeout=args.timeout,
    )
    assert_run_result(parse_result(baseline, "baseline"), "baseline")

    reflexion = await run_inferlet(
        client,
        "reflexion",
        {**common, "mode": "reflexion"},
        timeout=args.timeout,
    )
    assert_run_result(parse_result(reflexion, "reflexion"), "reflexion")

    both = await run_inferlet(
        client,
        "reflexion",
        {**common, "mode": "both"},
        timeout=args.timeout,
    )
    combined = parse_result(both, "both")
    assert_run_result(combined["baseline"], "baseline")
    assert_run_result(combined["reflexion"], "reflexion")


if __name__ == "__main__":
    run_tests([test_reflexion])
