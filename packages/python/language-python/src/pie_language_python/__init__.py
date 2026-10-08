"""The python language component of pie: one wasm, found by `pie.server.Server`
when this package is installed (`pip install "pie-server[python]"`)."""

from importlib.resources import files

LANGUAGE = "python"


def read() -> bytes:
    """The component's bytes."""
    return (files(__package__) / "python.wasm").read_bytes()
