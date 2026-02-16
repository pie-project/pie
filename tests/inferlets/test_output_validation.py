"""E2E test for output-validation inferlet."""
import re

from conftest import run_inferlet, run_tests


async def test_output_validation(client, args):
    output = await run_inferlet(
        client, "output-validation",
        timeout=args.timeout,
    )
    assert "Validation Results" in output, "Missing 'Validation Results' section"

    probs = re.findall(r"Probability:\s*([\d.]+)%", output)
    assert len(probs) >= 4, f"Expected 4 candidates, found {len(probs)}"

    total = sum(float(p) for p in probs)
    assert abs(total - 100.0) < 1.0, f"Probabilities sum to {total:.2f}%, expected ~100%"


if __name__ == "__main__":
    run_tests([test_output_validation])
