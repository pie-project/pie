"""Pie: programmable inference engine.

Public API::

    from pie import Server
    from pie.config import Config, ModelConfig, ServerConfig, DriverConfig

    cfg = Config(
        server=ServerConfig(),
        models={"default": ModelConfig(
            name="default",
            hf_repo="Qwen/Qwen3-0.6B",
            driver=DriverConfig(type="native", device=["cuda:0"]),
        )},
    )
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


# ---------------------------------------------------------------------------
# Eager driver registration
# ---------------------------------------------------------------------------
# Each driver package registers itself with `pie.drivers` on import. We try
# every known driver module here so the registry is populated before any
# config is parsed. Optional drivers (e.g., vllm) silently skip if their
# extras aren't installed.

def _register_known_drivers() -> None:
    for module_name in ("pie_driver", "pie_driver_cuda_native", "pie_driver_portable", "pie_driver_dummy", "pie_driver_vllm", "pie_driver_sgl"):
        try:
            __import__(module_name)
        except ImportError:
            # Driver's extras not installed — that's fine; only the configured
            # driver(s) need to be importable. The registry lookup will raise
            # a clear error if a TOML references an unregistered driver.
            pass


_register_known_drivers()
