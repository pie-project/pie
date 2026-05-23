"""`pie driver tensorrt-llm doctor` probe for TensorRT-LLM imports."""

from __future__ import annotations

from .bootstrap import ensure_cuda_library_path


def main() -> None:
    ensure_cuda_library_path(module="pie_driver_tensorrt_llm._trt_probe")

    import tensorrt_llm

    print(getattr(tensorrt_llm, "__version__", "unknown"))


if __name__ == "__main__":
    main()
