"""Torch-free helpers shared by the bridge.

Anything that touches `torch`, `torch.distributed`, or `torch.cuda`
belongs in a per-flavor `utils.py` (see `pie_driver_dev.utils`,
`pie_driver_sglang.utils`, `pie_driver_vllm.utils`) so the bridge wheel
stays torch-free.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback


def is_apple_silicon() -> bool:
    """True when running on macOS with Apple Silicon (M1/M2/M3/M4)."""
    return platform.system() == "Darwin" and platform.processor() == "arm"


def terminate(msg: str) -> None:
    """Terminate the program with a message."""
    print(f"\n[!!!] {msg} Terminating.", file=sys.stderr)
    traceback.print_exc()
    os._exit(1)
