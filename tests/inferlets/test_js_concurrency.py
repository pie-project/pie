"""E2E test for js-concurrency-test inferlet."""
from conftest import run_inferlet, run_tests


async def test_js_concurrency(client, args):
    output = await run_inferlet(
        client, "js-concurrency-test",
        timeout=args.timeout,
    )
    assert "[CTX1] START" in output, "Missing CTX1 START"
    assert "[CTX2] START" in output, "Missing CTX2 START"
    assert "[CTX1] END" in output, "Missing CTX1 END"
    assert "[CTX2] END" in output, "Missing CTX2 END"
    assert "[test] verdict:" in output, "Missing verdict"


if __name__ == "__main__":
    run_tests([test_js_concurrency])
