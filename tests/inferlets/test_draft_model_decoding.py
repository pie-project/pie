"""E2E test for draft-model-decoding inferlet."""
import re

from conftest import run_inferlet, run_tests


async def test_draft_model_decoding(client, args):
    output = await run_inferlet(
        client, "draft-model-decoding",
        {"max_tokens": 64},
        timeout=args.timeout,
    )
    assert "Generated in" in output, "Missing timing line"
    assert "Output:" in output, "Missing 'Output:' section"

    match = re.search(r"Output:\s*\n(.+)", output, re.DOTALL)
    assert match and match.group(1).strip(), "Output text is empty"


if __name__ == "__main__":
    run_tests([test_draft_model_decoding])
