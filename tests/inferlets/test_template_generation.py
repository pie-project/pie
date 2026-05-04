"""E2E test for template-generation inferlet."""
from conftest import run_inferlet, run_tests


async def test_template_generation(client, args):
    output = await run_inferlet(
        client, "template-generation",
        {"max_tokens": 2048},
        timeout=max(args.timeout, 300),
    )
    assert "PRODUCT ANNOUNCEMENT" in output, "Missing rendered template header"
    assert "OVERVIEW" in output, "Missing OVERVIEW section"
    assert "KEY FEATURES" in output, "Missing KEY FEATURES section"
    assert "PRICING & AVAILABILITY" in output, "Missing pricing section"


if __name__ == "__main__":
    run_tests([test_template_generation])
