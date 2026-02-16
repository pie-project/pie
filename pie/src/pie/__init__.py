"""Pie: programmable inference engine.

Public API::

    from pie import Server

    async with Server(model="Qwen/Qwen3-0.6B") as client:
        process = await client.launch_process(
            "text-completion@0.2.11",
            arguments=["--prompt", "Hello"],
        )
        event, value = await process.recv()
        print(value)
"""

from pie.server import Server  # noqa: F401

# Rust extension bindings (pie._runtime)
from pie import _runtime  # noqa: F401
