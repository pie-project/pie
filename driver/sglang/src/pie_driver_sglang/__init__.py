# Pie Driver (SGLang) — `sglang` driver. Uses sglang's model definitions,
# attention kernels, and KV cache plumbing under pie's RPC surface.
#
# After Phase 8, the standalone discovers drivers via `python -m
# pie_driver_sglang` directly; no in-process registry to populate.

from __future__ import annotations


_SGLANG_INSTALL_HINT = (
    "pie_driver_sglang requires SGLang. Install with `uv pip install pie-driver-sglang`."
)


def _require_sglang():
    try:
        import sglang  # noqa: F401
    except ImportError as e:
        raise ImportError(_SGLANG_INSTALL_HINT) from e
