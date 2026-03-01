"""E2E test for prefix-tree inferlet."""
import re

from conftest import run_inferlet, run_tests


async def test_prefix_tree(client, args):
    output = await run_inferlet(
        client, "prefix-tree",
        {"num_tokens": 32},
        timeout=args.timeout,
    )
    found = {int(m.group(1)) for m in re.finditer(r"Prompt #(\d+)", output)}
    missing = set(range(1, 9)) - found
    assert not missing, f"Missing prompt outputs: {sorted(missing)}"
    assert "All 8 generations completed" in output, "Missing completion summary"


if __name__ == "__main__":
    run_tests([test_prefix_tree])
