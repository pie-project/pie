# pie-server

What `pie serve` boots (controller, gateway, worker, the built-in inferlets),
inside your Python process on the native GPU engines. Clients connect as they
would to a `pie serve`: over WebSocket with `pie-client`, or over the OpenAI /
Anthropic / Gemini HTTP routes.

```python
import asyncio
from pie.config import Config, ModelConfig, ServerConfig
from pie.server import Server

async def main():
    cfg = Config(server=ServerConfig(port=0), model=ModelConfig(hf_repo="Qwen/Qwen3.5-0.8B"))
    async with Server(cfg) as server:          # resolves once the model is loaded
        client = await server.connect()        # a PieClient over ws
        print(server.http_url)                 # the OpenAI / Anthropic / Gemini routes

asyncio.run(main())
```

`Config` is what `pie serve --config` reads; `Config.to_toml()` renders it.
A `Server` garbage-collected without leaving the context still stops the
engine, on a background thread.

The wheel carries every engine its platform supports (Linux: CUDA, Vulkan,
wgpu; macOS: Metal, wgpu; Windows: CUDA, Vulkan); `EngineConfig.type` picks
one at boot. The Node.js twin is `@pie-project/server` (`javascript/server`).

## Building

```bash
cd python/server
uv run --with maturin maturin develop --features cuda      # or metal, vulkan, wgpu; several at once
```

The extension is abi3 (Python 3.10+), so one wheel per platform.
