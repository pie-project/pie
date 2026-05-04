"""E2E test for image-fetch inferlet."""
from conftest import run_inferlet, run_tests


async def test_image_fetch(client, args):
    output = await run_inferlet(
        client, "image-fetch",
        timeout=args.timeout,
    )
    # May fail if no network; we only check that it runs without crashing
    assert "Fetching image from:" in output, "Missing 'Fetching image from:' message"


if __name__ == "__main__":
    run_tests([test_image_fetch])
