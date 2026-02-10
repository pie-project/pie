"""Configuration utilities for Pie."""

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def get_default_device() -> str:
    """Get the default device based on the platform."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_default_config_content() -> str:
    """Create the default configuration file content."""
    device = get_default_device()
    formatted_device = f'"{device}"'

    return f"""\
host = "127.0.0.1"
port = 8080
verbose = false
registry = "https://registry.pie-project.org/"

[auth]
enabled = false

[telemetry]
enabled = false
endpoint = "http://localhost:4317"
service_name = "pie"

# Model configuration (can have multiple [[model]] sections)
[[model]]
hf_repo = "{DEFAULT_MODEL}"

# Device assignment (single GPU or list for tensor parallel)
device = [{formatted_device}]

# Tensor Parallelism degree (splits model across GPUs within a group)
# Set to 1 for Data Parallelism only (each GPU runs full model independently)
# Set to len(device) for Tensor Parallelism only (all GPUs share one model)
# Example: 4 GPUs with tensor_parallel_size=2 → 2 DP groups of 2 TP GPUs
tensor_parallel_size = 1

# Precision settings
activation_dtype = "bfloat16"
weight_dtype = "bfloat16"

# KV cache configuration
kv_page_size = 16

# Distribution/sampling
max_dist_size = 32

# Embedding limits
max_num_embeds = 128

# Adapter (LoRA) settings
max_num_adapters = 32
max_adapter_rank = 8

# Memory management
gpu_mem_utilization = 0.8

# CUDA graphs (experimental)
use_cuda_graphs = false

# Random seed
random_seed = 42

"""
