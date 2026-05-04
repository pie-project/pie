# pie-client

Python WebSocket client for a running `pie serve` engine.

## Install

```bash
pip install pie-client
```

From this checkout:

```bash
pip install -e client/python
```

## Example

```python
import asyncio
from pie_client import Event, PieClient

async def main():
    async with PieClient("ws://127.0.0.1:8080") as client:
        await client.authenticate("local-dev")

        proc = await client.launch_process(
            "text-completion",
            input={"prompt": "The capital of France is"},
        )

        while True:
            event, value = await proc.recv()
            if event in {Event.Return, Event.Error}:
                print(value)
                break

asyncio.run(main())
```

Use public-key authentication when server auth is enabled. For local
development, start the server with `pie serve --no-auth`.
