# Python Backend

Pie's Python backend requires manual installation of several dependencies with specific hardware requirements (e.g., CUDA). For this reason, dependencies like `torch` and `triton` are not specified in `pyproject.toml` to allow for system-specific builds.

-----

## Prerequisites

Before you begin, ensure you have the `uv` installed:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

-----

## Installation Guide

Follow these steps in order to set up your environment correctly.

### 1\. Create and Activate a Virtual Environment

First, create a new virtual environment using `uv` and activate it.

```sh
uv venv
```

### 2\. Install FlashInfer

Next, install the `flashinfer` library.

```sh
uv pip install flashinfer-python==0.5.3
```

> **Note:** Depending on your system configuration, you might need to build `flashinfer` from the source. Please refer to the official [FlashInfer repository](https://github.com/flashinfer-ai/flashinfer) for detailed instructions.

### 3\. Install PyTorch with CUDA Support

Install the correct version of PyTorch that matches your system's CUDA toolkit.

1.  Visit the official [PyTorch website](https://pytorch.org/get-started/locally/) to get the precise installation command for your setup (e.g., specific CUDA version, OS).
2.  Run the command provided by the website. You **must** add the `--force-reinstall` flag to ensure the correct GPU-enabled version of PyTorch overwrites any CPU-only version that `flashinfer` may have installed as a dependency.

For example, a typical command might look like this:

```sh
# Example command - get the correct one for your system from pytorch.org!
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129 --force-reinstall
```

### 4\. Install Triton

The final step is to install Triton. The package differs based on your system's architecture.

* **For `x86_64` Systems:**

  You can typically install Triton directly from PyPI.

  ```sh
  uv pip install triton
  ```

* **For `aarch64` Systems (e.g., NVIDIA Jetson and GH200):**

  The standard Triton wheel is not available for `aarch64`. You can install the version provided by PyTorch's index.

  ```sh
  uv pip install pytorch_triton --index-url https://download.pytorch.org/whl --force-reinstall
  ```

-----

## Docker Support

Official Docker images for the backend are planned and will be available in the future to simplify the setup process. 