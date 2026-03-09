# Pie

## Installation

### From PyPI (Recommended)

```bash
# For NVIDIA GPUs (includes CUDA dependencies)
pip install "pie-server[cuda]"

# For Apple Silicon (includes Metal dependencies)  
pip install "pie-server[metal]"

# Base installation (no GPU-specific packages)
pip install pie-server
```

### From Source

```bash
uv sync --extra cu128 # cu126, cu128 supported
```

### Verify Installation

```bash
pie doctor
```
