"""Factory helpers for constructing backend models and fusion maps."""

from __future__ import annotations

import sys
from pathlib import Path

# Use repo_utils for consistent path setup
_repo_utils_path = Path(__file__).parent.parent.parent / "repo_utils.py"
sys.path.insert(0, str(_repo_utils_path.parent))

import repo_utils  # pylint: disable=wrong-import-position

repo_utils.setup_pie_imports()

# pylint: disable=wrong-import-position,wrong-import-order  # Must come after repo_utils setup
from common import (
    ModelInfo,
    L4maArch,
    Qwen3Arch,
    GptOssArch,
)

# Import model components from common_python (base classes)
from common_model.l4ma import (
    L4maForCausalLM,
    create_fusion_map as create_l4ma_fusion_map,
)

# Import model components from local backend-python model directory
from model.l4ma_flashinfer import FlashInferL4maBackend
from model.qwen3 import Qwen3ForCausalLM, create_fusion_map as create_qwen3_fusion_map
from model.gptoss import (
    GptOssForCausalLM,
    create_fusion_map as create_gptoss_fusion_map,
)


def create_model_and_fusion_map(model_info: ModelInfo):
    """Instantiate a model and its fusion map based on the architecture."""
    arch_type = model_info.architecture.type.lower()

    if arch_type == "l4ma":
        if not FlashInferL4maBackend.is_available():
            raise RuntimeError(
                "FlashInfer backend is not available; cannot instantiate L4MA model."
            )

        backend = FlashInferL4maBackend()
        l4ma_arch = L4maArch(**model_info.architecture.__dict__)
        model = L4maForCausalLM(l4ma_arch, backend=backend)
        fusion_map = create_l4ma_fusion_map(model)
        return model, fusion_map

    if arch_type == "qwen3":
        qwen3_arch = Qwen3Arch(**model_info.architecture.__dict__)
        model = Qwen3ForCausalLM(qwen3_arch)
        fusion_map = create_qwen3_fusion_map(model)
        return model, fusion_map

    if arch_type == "gptoss":
        gptoss_arch = GptOssArch(**model_info.architecture.__dict__)
        model = GptOssForCausalLM(gptoss_arch)
        fusion_map = create_gptoss_fusion_map(model)
        return model, fusion_map

    raise ValueError(f"Unsupported architecture type: {model_info.architecture.type}")


__all__ = ["create_model_and_fusion_map"]
