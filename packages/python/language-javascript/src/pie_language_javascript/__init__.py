"""The javascript language component of pie: one wasm, found by `pie.server.Server`
when this package is installed (`pip install "pie-server[javascript]"`)."""

from importlib.resources import files

LANGUAGE = "javascript"


def read() -> bytes:
    """The component's bytes."""
    return (files(__package__) / "javascript.wasm").read_bytes()
