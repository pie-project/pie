"""Standalone launcher entry point for the TensorRT-LLM driver."""

from .bootstrap import ensure_cuda_library_path

ensure_cuda_library_path(module="pie_driver_tensorrt_llm")

from ._bridge._launcher import launch

from . import worker
from .config import TensorRTLLMDriverConfig
from .utils import validate_cuda_devices


if __name__ == "__main__":
    raise SystemExit(
        launch(
            prog="pie_driver_tensorrt_llm",
            config_cls=TensorRTLLMDriverConfig,
            worker=worker,
            validate_devices=validate_cuda_devices,
        )
    )
