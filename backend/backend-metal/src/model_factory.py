"""Metal backend model factory helpers."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_PYTHON_PATH = Path(__file__).resolve().parents[2] / "backend-python"
COMMON_PYTHON_PATH = Path(__file__).resolve().parents[2] / "common_python"
if str(BACKEND_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PYTHON_PATH))
if str(COMMON_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(COMMON_PYTHON_PATH))

from dataclasses import asdict

from config.common import ModelInfo
from common_model.l4ma import L4maForCausalLM, create_fusion_map as create_l4ma_fusion_map

from l4ma_runtime import MetalL4maBackend
from metal_backend import MetalBackend


def create_model_and_fusion_map(model_info: ModelInfo):
    """Instantiate a Metal-backed model and fusion map for the requested architecture."""

    arch_type = model_info.architecture.type.lower()

    if arch_type == "l4ma":
        architecture_dict = asdict(model_info.architecture)
        metal_backend = MetalBackend(model_metadata={"architecture": architecture_dict})
        if not metal_backend.initialize():
            raise RuntimeError("Metal backend initialization failed for model factory")
        backend = MetalL4maBackend(metal_backend=metal_backend)
        model = L4maForCausalLM(model_info.architecture, backend=backend)
        fusion_map = create_l4ma_fusion_map(model)
        return model, fusion_map

    raise ValueError(f"Unsupported architecture type for Metal backend: {model_info.architecture.type}")


__all__ = ["create_model_and_fusion_map"]
