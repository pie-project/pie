"""Connect to a serving pie, draw one picture, save the file it sends back.

Run a pie that has a generative model bound, then:

    pie --config ~/.pie/config.flux2.toml serve &
    python python/client/examples/text_to_image.py \
        --prompt "a red bicycle leaning on a blue wall" --out ./out

The whole point of this file is the FILE event. `text-to-image` streams a
progress line or two and then sends its output as a file; what arrives is a
`ReceivedFile` -- bytes, with the name the inferlet suggested attached. That
name is what makes the difference between writing `file-0000.bin` and writing
`image.latent.f32` (or `image.png`, on a model whose VAE decode reading the
guest can drive), so it is what this example is about.

`text-to-image` must already be on the server: `pie inferlet install
<text_to_image.wasm>`, or `client.install_program(wasm)` as below,
or a `pie run text-to-image` from a source checkout, which uploads it.
"""

import argparse
import asyncio
import json
from pathlib import Path

from pie_client import Event, PieClient

INFERLET = "text-to-image@0.1.0"


async def draw(uri: str, prompt: str, out: Path, steps: int, size: int,
               wasm: str | None) -> int:
    async with PieClient(uri) as client:
        await client.authenticate("example")

        # Only when pointed at a local build. A published inferlet is already
        # there and this is skipped.
        if wasm:
            await client.install_program(wasm, force_overwrite=True)

        process = await client.launch_process(
            INFERLET,
            {"prompt": prompt, "steps": steps, "width": size, "height": size,
             "out": "image"},
        )

        saved: list[Path] = []
        while True:
            event, value = await process.recv()

            if event is Event.File:
                # `value` is a `ReceivedFile`: still bytes, plus `.name`.
                # `file_name()` sanitises what came off the wire -- a name is
                # a suggestion, not a path this program has to trust.
                path = value.save(out, fallback="output.bin")
                saved.append(path)
                print(f"wrote {path} ({len(value):,} bytes)")

            elif event is Event.Return:
                # The guest's report: the geometry, the sigmas it stepped,
                # and the name it gave the file above.
                report = json.loads(value) if value.startswith("{") else {}
                if report:
                    print(f"{report['width']}x{report['height']} in "
                          f"{report['steps']} steps, {report['rows']} latent rows")
                return 0 if saved else 1

            elif event is Event.Error:
                print(f"the inferlet failed: {value}")
                return 1

            elif event in (Event.Stdout, Event.Stderr, Event.Message):
                print(value, end="" if event is not Event.Message else "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", default="ws://127.0.0.1:8231/v1/ws")
    ap.add_argument("--prompt", default="a red bicycle leaning on a blue wall")
    ap.add_argument("--out", type=Path, default=Path("./out"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--wasm", default=None, help="a local build to upload first")
    args = ap.parse_args()
    return asyncio.run(draw(args.uri, args.prompt, args.out, args.steps,
                            args.size, args.wasm))


if __name__ == "__main__":
    raise SystemExit(main())
