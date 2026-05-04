"""E2E test for js-example inferlet."""
from conftest import run_inferlet, run_tests


async def test_js_example(client, args):
    output = await run_inferlet(
        client, "js-example",
        timeout=args.timeout,
    )
    assert len(output) > 0, "Output is empty"
    assert "[done]" in output, "Missing [done] marker"


if __name__ == "__main__":
    run_tests([test_js_example])
