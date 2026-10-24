"""`pie`: what `pie serve` boots, inside a Python script — the Python twin
of `@pie-project/server`.

    import asyncio
    from pie.server import Server
    from pie.config import Config, ServerConfig, ModelConfig, EngineConfig

    cfg = Config(
        server=ServerConfig(listen="127.0.0.1:0"),  # port 0: a free one; see server.url
        model=ModelConfig(source="Qwen/Qwen3.5-0.8B"),
        engine=EngineConfig(type="cuda", device=["cuda:0"]),
    )

    async def main():
        async with Server(cfg) as server:
            client = await server.connect()
            proc = await client.launch_process("text-completion", input={"prompt": "Hello"})
            event, value = await proc.recv()
            print(value)

    asyncio.run(main())
"""

from pie.server import Server  # noqa: F401
from pie import config  # noqa: F401
