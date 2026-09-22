# pie-client

The Python client for pie: `PieClient` speaks to a `pie serve` (or an embedded
`pie-server`) over its WebSocket, and `pie-client` is the command line on top.

```bash
pip install pie-client
pie-client submit text-completion -- --prompt "The capital of France is"
pie-client ping
```

```python
from pie_client import PieClient

client = PieClient("ws://127.0.0.1:8080")
await client.connect()
await client.authenticate("me")
process = await client.launch_process("text-completion", {"prompt": "The capital of France is"})
event, value = await process.recv()
```

`pie-client config init` writes `~/.pie/cli_config.toml` (host, port,
username); `--host`, `--port` and `--username` on any command override it.
