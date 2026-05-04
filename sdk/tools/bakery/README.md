# Bakery

Build tool for Pie inferlets.

The `pie` CLI wraps the common commands:

```bash
pie new my-inferlet
pie new my-inferlet --ts
pie build ./my-inferlet -o out.wasm
```

The standalone `bakery` command is still available for registry workflows:

```bash
bakery login
bakery inferlet publish ./my-inferlet
bakery inferlet search text
```

## Supported inputs

- Rust crates with `Cargo.toml`
- Python projects with `pyproject.toml` or `main.py`
- JavaScript/TypeScript projects with `package.json`, `.js`, or `.ts` input

Rust builds require `wasm32-wasip2`. JavaScript/TypeScript builds require
Node.js. Python builds use the cached Pie Python Wasm runtime.

## Repository development

Bakery auto-detects the SDK layout when run inside this repo. Override it
when needed:

```bash
export PIE_SDK=/path/to/pie/sdk
```
