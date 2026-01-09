# Pie Control

The **Pie Control** (`pie`) is the main CLI for the Pie Inference Engine. It provides commands to manage the engine, models, configuration, and run inferlets.

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
cd server
pip install -e ".[cuda]"   # or [metal]
```

### Verify Installation

```bash
pie doctor
```

This checks Python version, PyTorch, GPU availability, and required dependencies.

## Quick Start

```bash
# 1. Initialize configuration
pie config init

# 2. Download a model
pie model download Qwen/Qwen3-0.6B

# 3. Run an inferlet
pie run ./my_inferlet.wasm -- --prompt "Hello, world!"
```

## Commands

### `pie serve`

Start the Pie engine and optionally enter an interactive session.

```bash
pie serve [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to config file (default: `~/.pie/config.toml`) |
| `--host` | Override host address |
| `--port` | Override port |
| `--no-auth` | Disable authentication |
| `--verbose`, `-v` | Enable verbose logging |
| `--log` | Log file path |
| `--interactive`, `-i` | Enter interactive shell after starting |

**Interactive Mode:**
```bash
pie serve -i
# pie> run ./inferlet.wasm --arg1 value
# pie> stat
# pie> exit
```

---

### `pie run`

Run an inferlet with a one-shot engine (starts, runs, shuts down).

```bash
pie run <INFERLET> [OPTIONS] [-- ARGS...]
```

| Argument | Description |
|----------|-------------|
| `INFERLET` | Path to the `.wasm` inferlet file |
| `ARGS...` | Arguments passed to the inferlet (after `--`) |

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to config file |
| `--log` | Log file path |

**Example:**
```bash
# Run from local file
pie run --path ./text_completion.wasm -- --prompt "Explain quantum computing"

# Run from registry
pie run std/text-completion@0.1.0 -- --prompt "Explain quantum computing"
```

---

### `pie doctor`

Run environment health check.

```bash
pie doctor
```

Checks:
- Python version (requires 3.11+)
- PyTorch installation
- CUDA/Metal availability
- FlashInfer, FBGEMM (CUDA)
- PyObjC (Metal)
- pie-backend, pie-client packages

---

### `pie config`

Manage engine configuration.

#### `pie config init`

Create a default configuration file.

```bash
pie config init [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--dummy` | Initialize with dummy backend (for testing) |
| `--path` | Custom config file path |

#### `pie config show`

Display the current configuration.

```bash
pie config show [--path PATH]
```

#### `pie config update`

Update configuration values.

```bash
pie config update [OPTIONS]
```

**Engine Options:**
| Option | Description |
|--------|-------------|
| `--host` | Engine host address |
| `--port` | Engine port |
| `--enable-auth` | Enable/disable authentication |
| `--verbose` | Enable verbose logging |
| `--cache-dir` | Cache directory path |
| `--log-dir` | Log directory path |
| `--registry` | Inferlet registry URL |

**Model Options:**
| Option | Description |
|--------|-------------|
| `--hf-repo` | HuggingFace repository (e.g., `Qwen/Qwen3-0.6B`) |
| `--device` | Device (e.g., `cuda:0`, `mps`) |
| `--kv-page-size` | KV cache page size |
| `--max-batch-tokens` | Max tokens per batch |
| `--gpu-mem-utilization` | GPU memory utilization (0.0-1.0) |
| `--activation-dtype` | Activation dtype (`bfloat16`, `float16`) |
| `--weight-dtype` | Weight dtype (`auto`, `float32`, `float16`, `bfloat16`, `int4`, `int8`) |
| `--max-num-adapters` | Max number of adapters |
| `--max-adapter-rank` | Max adapter rank |
| `--enable-profiling` | Enable profiling |
| `--random-seed` | Random seed for model initialization |

---

### `pie model`

Manage local models.

#### `pie model list`

List downloaded models.

```bash
pie model list
```

#### `pie model download`

Download a model from HuggingFace.

```bash
pie model download <REPO_ID>
```

Example:
```bash
pie model download Qwen/Qwen3-0.6B
```

#### `pie model remove`

Remove a locally cached model.

```bash
pie model remove <REPO_ID>
```

---

### `pie auth`

Manage authenticated users (public key authentication).

#### `pie auth add`

Add a user or public key.

```bash
# Add user with key from stdin
cat ~/.ssh/id_ed25519.pub | pie auth add <USERNAME> <KEY_NAME>

# Add user without key (key can be added later)
pie auth add <USERNAME>
```

#### `pie auth remove`

Remove a user or specific key.

```bash
# Remove entire user
pie auth remove <USERNAME>

# Remove specific key
pie auth remove <USERNAME> --key <KEY_NAME>
```

#### `pie auth list`

List authorized users and their keys.

```bash
pie auth list
```

---

## Configuration File

Default location: `~/.pie/config.toml`

```toml
[engine]
host = "127.0.0.1"
port = 8080
enable_auth = false
verbose = false
cache_dir = "/home/user/.pie/cache"
log_dir = "/home/user/.pie/logs"
registry = "https://registry.pie-project.org/"

[[model]]
hf_repo = "Qwen/Qwen3-0.6B"
device = ["cuda:0"]
activation_dtype = "bfloat16"
weight_dtype = "auto"
kv_page_size = 16
max_batch_tokens = 10240
max_dist_size = 32
max_num_embeds = 128
max_num_adapters = 32
max_adapter_rank = 8
gpu_mem_utilization = 0.9
enable_profiling = false
random_seed = 42
```

| Type | Description |
|------|-------------|
| `python` | Built-in Python backend (pie-backend) |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PIE_HOME` | Override default Pie home directory (`~/.pie`) |
| `PIE_CACHE_HOME` | Override cache directory |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      pie serve                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Pie Engine (Rust/Tokio)                  │    │
│  │  • WebSocket server for clients                     │    │
│  │  • Instance management                              │    │
│  │  • Program caching                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │ ZMQ                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Python Backend (pie-backend)              │    │
│  │  • Model loading (Hugging Face)                     │    │
│  │  • Forward pass / batching                          │    │
│  │  • KV cache management                              │    │
│  │  • CUDA/Metal acceleration                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Development

### Run Tests

```bash
cd server
pip install -e ".[test]"
pytest tests/ -v
```

### Build from Source

```bash
cd server
maturin develop  # Debug build
maturin develop --release  # Release build
```
