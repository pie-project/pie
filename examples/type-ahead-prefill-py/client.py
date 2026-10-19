"""Type into a running pie and watch the prompt prefill as you go.

    pie serve &
    python examples/type-ahead-prefill-py/client.py            # type, Enter sends
    python examples/type-ahead-prefill-py/client.py --compare  # scripted typing vs. pasting it all at once

Every keystroke sends the whole draft to the inferlet; the dim counter on the
right is how many prompt tokens are already in the KV cache.
"""

import argparse
import asyncio
import json
import random
import sys
import termios
import tty
from pathlib import Path

from pie_client import Event, PieClient

SOURCE = Path(__file__).with_name("main.py")
NAME, VERSION = "type-ahead-prefill", "0.1.0"

SAMPLE = (
    "I am planning a three day trip to Kyoto in late autumn with my parents, who walk slowly "
    "and do not like crowds. We want to see temples, gardens and maple leaves, eat good tofu "
    "and noodles, and stay somewhere quiet near a train station. Please suggest a day by day "
    "plan with times, how to avoid the busiest places, and one rainy day alternative."
)


async def start(client: PieClient, max_tokens: int):
    program = await client.install_program_bytes(SOURCE.read_bytes(), f"{NAME}.py", VERSION, force_overwrite=True)
    return await client.launch_process(program, {"max_tokens": max_tokens})


async def answer(process, echo: bool) -> dict:
    """Drain replies until the return; print the answer as it streams."""
    stats: dict = {}
    while True:
        event, value = await process.recv()
        if event is Event.Message:
            reply = json.loads(value)
            if "token" in reply and echo:
                print(reply["token"], end="", flush=True)
            elif "ttft_ms" in reply:
                stats.update(reply)
        elif event is Event.Return:
            stats.update(json.loads(value))
            return stats
        elif event is Event.Error:
            raise RuntimeError(value)


async def scripted(client: PieClient, text: str, cps: float, type_ahead: bool, max_tokens: int) -> dict:
    process = await start(client, max_tokens)
    if type_ahead:
        for i in range(1, len(text) + 1):
            await process.signal(json.dumps({"draft": text[:i]}))
            await asyncio.sleep(1 / cps)
    await process.signal(json.dumps({"send": text}))
    return await answer(process, echo=False)


async def compare(client: PieClient, args) -> None:
    text = Path(args.text).read_text() if args.text else SAMPLE
    print(f"{len(text)} characters typed at {args.cps:g} per second\n")
    for label, type_ahead in (("pasted at Enter", False), ("prefilled while typing", True)):
        # a fresh opening each run, or the prefix cache would hand the second run the first one's work
        stats = await scripted(client, f"[{random.getrandbits(32):08x}] {text}", args.cps, type_ahead, args.max_tokens)
        print(f"{label:24} first token {stats['ttft_ms']:7.1f} ms   "
              f"{stats['tail']:4d} of {stats['prompt_tokens']} prompt tokens left at Enter")


async def interactive(client: PieClient, max_tokens: int) -> None:
    process = await start(client, max_tokens)
    draft, prefilled = "", 0
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    loop = asyncio.get_running_loop()
    keys: asyncio.Queue[str] = asyncio.Queue()
    loop.add_reader(fd, lambda: keys.put_nowait(sys.stdin.read(1)))

    def show() -> None:
        sys.stdout.write(f"\r\x1b[K> {draft}  \x1b[2m{prefilled} tokens cached\x1b[0m")
        sys.stdout.write(f"\x1b[{len(str(prefilled)) + 17}D")
        sys.stdout.flush()

    async def replies() -> None:
        nonlocal prefilled
        while True:
            event, value = await process.recv()
            if event is Event.Message and "prefilled" in (reply := json.loads(value)):
                prefilled = reply["prefilled"]
                show()
            elif event is Event.Message:
                await stream.put(reply)
            else:
                await stream.put({"_event": event, "value": value})

    stream: asyncio.Queue[dict] = asyncio.Queue()
    reader = asyncio.create_task(replies())
    try:
        tty.setcbreak(fd)
        show()
        while True:
            key = await keys.get()
            if key in ("\r", "\n"):
                break
            if key in ("\x7f", "\b"):
                draft = draft[:-1]
            elif key.isprintable():
                draft += key
            else:
                continue
            show()
            await process.signal(json.dumps({"draft": draft}))
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    sys.stdout.write("\r\x1b[K> " + draft + "\n\n")
    await process.signal(json.dumps({"send": draft}))
    while True:
        reply = await stream.get()
        if "token" in reply:
            print(reply["token"], end="", flush=True)
        elif "ttft_ms" in reply:
            ttft = reply
        elif reply.get("_event") is Event.Return:
            stats = json.loads(reply["value"])
            break
        elif reply.get("_event") is Event.Error:
            raise RuntimeError(reply["value"])
    reader.cancel()
    print(f"\n\n\x1b[2mfirst token {ttft['ttft_ms']} ms, {ttft['tail']} of "
          f"{stats['prompt_tokens']} prompt tokens left at Enter\x1b[0m")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ws://127.0.0.1:8231/v1/ws")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--compare", action="store_true", help="time scripted typing against pasting")
    ap.add_argument("--text", help="file with the text --compare types (a built-in paragraph otherwise)")
    ap.add_argument("--cps", type=float, default=12, help="characters per second for --compare")
    args = ap.parse_args()
    async with PieClient(args.uri) as client:
        await client.authenticate("type-ahead")
        if args.compare:
            await compare(client, args)
        elif not sys.stdin.isatty():
            sys.exit("typing needs a terminal: run this in Terminal, or pass --compare")
        else:
            await interactive(client, args.max_tokens)


if __name__ == "__main__":
    asyncio.run(main())
