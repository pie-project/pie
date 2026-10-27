# pie-server

Embeds `pie serve` inside a Python process on the native GPU backends.

```python
import asyncio
from pie.config import Config, EngineConfig, ModelConfig, ServerConfig
from pie.server import Server

async def main():
    cfg = Config(
        server=ServerConfig(listen="127.0.0.1:0"),
        model=ModelConfig(source="Qwen/Qwen3.5-0.8B"),
        engine=EngineConfig(type="cuda", device=["cuda:0"]),
    )
    async with Server(cfg) as server:
        client = await server.connect()   # PieClient over WebSocket
        print(server.http_url)            # OpenAI / Anthropic / Gemini routes

asyncio.run(main())
```

## Build from source

```bash
uv run --with maturin maturin develop --features cuda   # or metal, vulkan, wgpu
```
