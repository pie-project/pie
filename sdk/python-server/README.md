# pie-server

Embed the pie engine inside a Python script.

```python
import asyncio
from pie.server import Server
from pie.config import (
    Config, ServerConfig, AuthConfig, ModelConfig, DriverConfig,
)

cfg = Config(
    server=ServerConfig(port=0),                     # 0 = auto-pick
    auth=AuthConfig(enabled=False),
    models=[ModelConfig(
        name="default",
        hf_repo="Qwen/Qwen3-0.6B",
        driver=DriverConfig(type="dev", device=["cuda:0"]),
    )],
)

async def main():
    async with Server(cfg) as server:
        client = await server.connect()
        proc = await client.launch_process(
            "text-completion@0.2.11", input={"prompt": "Hello"})
        event, value = await proc.recv()
        print(value)

asyncio.run(main())
```

Drop-in replacement for the legacy `pie-server` Python wheel — same
`pie.server.Server` async context manager, same `pie.config.*`
dataclass shapes, same `connect() → PieClient`. Existing test fixtures
under `tests/inferlets/`, `benches/`, and `sdk/demo/` keep working
unchanged.

## How it differs from `pie serve` (the CLI)

Same engine; different driver. The Rust binary `pie` parses argv and
calls `serve::start_engine`; this wheel parses a Python `Config` (or
TOML string) and calls the same `start_engine`. Single source of truth
on the Rust side — both surfaces can never drift in their boot
behavior.

The pyo3 layer (`pie._engine`) is intentionally tiny: `bootstrap(toml)
-> EngineHandle`. Everything user-facing lives in pure Python (`pie.server`,
`pie.config`).

## Lifecycle guarantees

When the Python script exits, the engine exits. Three layers of
cleanup:

1. `Server.__aexit__` calls the engine handle's `shutdown()`.
2. If `__aexit__` doesn't run (interpreter SIGKILL, segfault),
   `EngineHandle.__del__` (Rust `Drop`) takes the same path on GC /
   interpreter shutdown.
3. Subprocess drivers (`dev` / `vllm` / `sglang`) set
   `PR_SET_PDEATHSIG` on Linux, so even a hard parent kill kills
   the launcher → no orphan Python workers, no leaked GPU memory.

## Install

```sh
uv pip install pie-server
```

(or, from this checkout, `uv pip install --python <venv>/bin/python -e
sdk/python-server/`)
