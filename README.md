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



## Getting Started

### Installation

**Option 1: PyPI**

```bash
pip install "pie-server[cuda]"   # Linux/Windows
pip install "pie-server[metal]"  # macOS
```

**Option 2: Build from Source (Recommended)**

```bash
git clone https://github.com/pie-project/pie.git && cd pie/pie

# Recommended: use uv to sync (options: cu126, cu128, cu130, metal)
uv sync --extra cu128
```

### Quick Start

Run a test prompt (you will be prompted for configuration and model download if this is your first time):

```bash
pie run text-completion -- --prompt "Hello world!"
pie run beam-search -- --prompt "What is the capital of France?" --beam-size 2
```

> **Note:** The first run may take longer due to JIT compilation.
> *If built from source, prefix commands with `uv run` (e.g., `uv run pie config init`).*




Check out the [https://pie-project.org/docs](https://pie-project.org/) for more information.

## Community

**Issues & Bugs**: Please report bugs on [GitHub Issues](https://github.com/pie-project/pie/issues).

**Discussions**: Have a question or feedback? Join us on [GitHub Discussions](https://github.com/pie-project/pie/discussions).




## License

[Apache License 2.0](LICENSE)