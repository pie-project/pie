"""Compatibility shim for importing the Metal handler module.

The primary implementation lives in ``handler.py``. This file is retained so
existing imports (e.g. ``from metal_handler import MetalHandler``) continue to
work while the codebase migrates to the new module name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_handler_path = Path(__file__).with_name("handler.py")
_spec = importlib.util.spec_from_file_location("metal_backend_handler", _handler_path)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive guard
    raise ImportError(f"Unable to load Metal handler implementation at {_handler_path}")

_handler_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_handler_module)

MetalHandler = _handler_module.MetalHandler
Handler = _handler_module.Handler

__all__ = ["MetalHandler", "Handler"]
