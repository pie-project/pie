"""E2E regression test for the runtime's empty-input rejection.

Verifies that `runtime/src/api/inference.rs::ForwardPass::execute()`
returns an error when both `input_tokens` and `speculative_tokens`
are empty. This is a regression test for fix #6 in BRIDGE.md: an
ill-formed forward pass (zero query positions but an attached
sampler) must be rejected at the API boundary, not silently no-op'd
at the driver layer.

The companion inferlet at `inferlets/empty-forward-test/` builds an
empty `ForwardPass` and asserts the runtime rejects it with a
descriptive error.
"""
from conftest import run_inferlet, run_tests


async def test_empty_forward(client, args):
    output = await run_inferlet(
        client, "empty-forward-test", {},
        timeout=args.timeout,
    )
    assert "Got expected rejection" in output, (
        "Inferlet didn't see the runtime's empty-input rejection. "
        f"Output was: {output}"
    )
    assert "empty input" in output, (
        "Rejection message doesn't mention 'empty input' — fix #6 may "
        f"have been silently reverted. Output was: {output}"
    )


if __name__ == "__main__":
    run_tests([test_empty_forward])
