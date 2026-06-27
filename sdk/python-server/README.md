# pie-server

Python wrapper for embedding the Pie engine in tests, notebooks, or scripts.
For normal use, prefer the `pie` CLI from `worker/`.

```python
import asyncio
from pie.config import AuthConfig, Config, DriverConfig, ModelConfig, ServerConfig
from pie.server import Server

cfg = Config(
    server=ServerConfig(port=0),
    auth=AuthConfig(enabled=False),
    model=ModelConfig(
        name="default",
        hf_repo="Qwen/Qwen3-0.6B",
        driver=DriverConfig(type="dev", device=["cuda:0"]),
    ),
)

async def main():
    async with Server(cfg) as server:
        client = await server.connect()
        proc = await client.launch_process(
            "text-completion",
            input={"prompt": "Hello"},
        )
        print(await proc.recv())

asyncio.run(main())
```

## Install

```bash
uv pip install pie-server
```

From this checkout:

```bash
uv venv /tmp/pie-python-server-venv
uv pip install --python /tmp/pie-python-server-venv/bin/python -e sdk/python-server/
```

## Local embedded-driver build

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
  uv --project sdk/python-server sync --reinstall-package pie-server
```
