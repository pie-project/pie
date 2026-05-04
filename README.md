<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/img/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/img/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/img/pie-light.svg"
         width="30%">
    <p></p>
  </picture>
</div>


**Pie** is a high-performance, programmable LLM serving system that empowers you to design and deploy custom inference logic and optimization strategies.

> **Note** 🧪
>
> This software is in a **pre-release** stage and under active development. It's recommended for testing and research purposes only.



## How it works

Pie ships as **one Rust binary** (`pie`) plus **driver wheels** for the
inference backends. The binary owns the WS scheduler, runtime, and CLI;
drivers come in two flavors:

| Driver | Where it runs | Bundled in `pie`? |
|---|---|---|
| `portable` (ggml CPU/CUDA), `cuda_native` (CUDA C++), `dummy` (Rust) | linked into `pie` as a static lib | yes (Cargo feature) |
| `dev` (reference PyTorch + flashinfer), `vllm`, `sglang` | spawned as a Python subprocess from a venv you control | no — install separately |

The Rust binary is Python-free; Python only enters the picture when a
model picks one of `dev` / `vllm` / `sglang`. The standalone supervises
the subprocess via a handshake pipe and the same shmem fast path the
embedded drivers use.

## Getting Started

### Step 1 — install the `pie` binary

```bash
git clone https://github.com/pie-project/pie.git && cd pie

# Default features: embedded `portable` + `dummy` drivers.
cargo install --path server

# To also include the embedded CUDA driver:
cargo install --path server --features driver-portable,driver-cuda,driver-dummy
```

That's the entire install for the embedded drivers. To check:

```bash
pie doctor          # platform / GPUs / compiled-in drivers / venv readiness
```

### Step 2 — only if using a Python driver

`pie driver <type> install` prints (and `--run` executes) the canonical
recipe for setting up a venv with the matching wheel:

```bash
pie driver dev install ~/.pie/venvs/dev --run
# → uv venv ~/.pie/venvs/dev --python 3.12
# → uv pip install --python ~/.pie/venvs/dev/bin/python pie-driver-dev[cu128]

pie driver dev set venv ~/.pie/venvs/dev      # persist as the default
pie driver dev doctor                         # confirm imports + torch.cuda
```

`pie` looks up which Python interpreter to invoke for each model via this
precedence chain (highest wins):

1. `[model.driver.options].venv` or `python` (per-model override)
2. `$PIE_PYTHON`
3. `$VIRTUAL_ENV/bin/python` (an activated venv)
4. `~/.pie/drivers.toml [driver.<type>]` (set via `pie driver <type> set`)
5. `~/.pie/drivers.toml [python]` (shared default)
6. `which python3`

`pie driver <type> show` prints the resolved path **and** which step
matched, so you can debug a wrong choice without having to rederive the
chain.

### Quick Start

```bash
pie config init                                       # one-shot: ~/.pie/config.toml
pie serve                                             # long-running engine
# or, in another terminal:
pie run text-completion -- --prompt "Hello world!"    # one-shot inferlet
```

A minimal serve config (`~/.pie/config.toml`):

```toml
[server]
host = "127.0.0.1"
port = 8080

[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "dev"           # or "portable" / "cuda_native" / "vllm" / "sglang"
device = ["cuda:0"]

[model.driver.options]
gpu_mem_utilization = 0.85
```

For the full subcommand surface, run `pie --help` (e.g. `pie model
list`, `pie auth add`, `pie driver list`).

### Embed inside a Python script

The `pie-server` wheel exposes the same engine via an async context
manager. Useful for notebooks, integration tests, or research scripts
where you want the engine's lifetime tied to your script's:

```python
import asyncio
from pie.server import Server
from pie.config import Config, ServerConfig, AuthConfig, ModelConfig, DriverConfig

cfg = Config(
    server=ServerConfig(port=0),                # 0 = auto-pick
    auth=AuthConfig(enabled=False),
    models=[ModelConfig(
        name="default",
        hf_repo="Qwen/Qwen3-0.6B",
        driver=DriverConfig(type="dev", device=["cuda:0"]),
    )],
)

async def main():
    async with Server(cfg) as server:
        client = await server.connect()
        proc = await client.launch_process(
            "text-completion@0.2.11", input={"prompt": "Hello"})
        event, value = await proc.recv()
        print(value)

asyncio.run(main())
```

When the script exits, the engine exits — `Server.__aexit__` shuts
drivers down, and on Linux `PR_SET_PDEATHSIG` ensures subprocess
drivers die even if the parent is hard-killed (no orphan workers, no
leaked GPU memory). Drop-in replacement for the legacy `pie-server`
wheel: existing `tests/inferlets/`, `benches/`, and `sdk/demo/`
fixtures continue to work without modification.

Install: `uv pip install pie-server` (or `uv pip install -e
sdk/python-server/` from this checkout).

Check out [pie-project.org](https://pie-project.org/) for design docs and
the inferlet authoring guide.

## Community

**Issues & Bugs**: Please report bugs on [GitHub Issues](https://github.com/pie-project/pie/issues).

**Discussions**: Have a question or feedback? Join us on [GitHub Discussions](https://github.com/pie-project/pie/discussions).




## License

[Apache License 2.0](LICENSE)
