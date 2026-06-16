# Pie Driver (vLLM) — `vllm` driver. Uses vllm's model definitions and
# kernels under pie's RPC surface.
#
# After Phase 8, the standalone discovers drivers via `python -m
# pie_driver_vllm` directly; no in-process registry to populate.

from __future__ import annotations

import os


# Defensive: align flashinfer-python / flashinfer-cubin / flashinfer-jit-cache
# pin alignment is enforced in pyproject.toml. This bypass exists for early
# dev environments that haven't re-synced.
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

# vLLM's AOT compile cache key does not distinguish the model call surface
# that produced the artifact. Keep Pie's default roots separate from
# standalone vLLM and from Pie's own input_ids-vs-input_embeds modes.
def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


_cache_surface = (
    "vllm-inputids"
    if _env_flag("PIE_VLLM_INPUT_IDS_FORWARD", True)
    else "vllm-embeds"
)
os.environ.setdefault(
    "VLLM_CACHE_ROOT",
    os.path.expanduser(f"~/.cache/pie/{_cache_surface}"),
)


_VLLM_INSTALL_HINT = (
    "pie_driver_vllm requires vLLM. Install with `uv pip install pie-driver-vllm`."
)


def _require_vllm():
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        raise ImportError(_VLLM_INSTALL_HINT) from e
