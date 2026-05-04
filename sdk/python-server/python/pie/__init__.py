"""`pie` — Python bindings for the pie engine.

Embed `pie serve` inside a Python script:

    import asyncio
    from pie.server import Server
    from pie.config import (
        Config, ServerConfig, AuthConfig, ModelConfig, DriverConfig,
    )

    cfg = Config(
        server=ServerConfig(port=0),     # 0 = auto-pick a free port
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

Mirrors the legacy `pie-server` Python wheel surface so existing tests
(`tests/inferlets/`, `benches/`, `sdk/demo/`) keep working unchanged.
"""

from pie.server import Server  # noqa: F401
from pie import config  # noqa: F401
