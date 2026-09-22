#!/usr/bin/env python3
"""Launch one inferlet on a running server and print its result.

    launch.py <ws-url> <inferlet> [<json input>]

Prints `ok <result>` (exit 0) or `error <message>` (exit 1)."""

import asyncio
import json
import sys

from pie_client import PieClient


async def main() -> int:
    url, name = sys.argv[1], sys.argv[2]
    given = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"prompt": "The capital of France is", "max_tokens": 4}
    async with PieClient(url) as client:
        await client.authenticate("ci", None)
        try:
            process = await client.launch_process(name, given)
            print("ok", str(await process.result())[:200])
            return 0
        except Exception as error:
            print("error", str(error).splitlines()[0][:200])
            return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
