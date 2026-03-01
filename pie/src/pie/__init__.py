"""Pie: programmable inference engine.

Public API::

    from pie import Server
    from pie.config import Config, ModelConfig

    cfg = Config(models=[ModelConfig(hf_repo="Qwen/Qwen3-0.6B")])
    async with Server(cfg) as server:
        client = await server.connect()
        process = await client.launch_process(
            "text-completion@0.2.11",
            input={"prompt": "Hello"},
        )
        event, value = await process.recv()
        print(value)
"""

from pie.server import Server  # noqa: F401

# Rust extension bindings (pie._runtime)
from pie import _runtime  # noqa: F401
