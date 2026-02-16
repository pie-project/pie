"""E2E test for helloworld inferlet."""
from conftest import run_inferlet, run_tests


async def test_helloworld(client, args):
    output = await run_inferlet(client, "helloworld", timeout=args.timeout)
    assert "Hello World!!" in output, f"Missing 'Hello World!!' in output"
    assert "running in the Pie runtime" in output, "Missing runtime info"


if __name__ == "__main__":
    run_tests([test_helloworld])
