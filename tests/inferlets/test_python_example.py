"""E2E test for python-example inferlet."""
from conftest import run_inferlet, run_tests


async def test_python_example(client, args):
    output = await run_inferlet(
        client, "python-example",
        timeout=args.timeout,
    )
    assert len(output) > 0, "Output is empty"
    assert "[done]" in output, "Missing [done] marker"


if __name__ == "__main__":
    run_tests([test_python_example])
