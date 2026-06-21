# driver/dummy

Rust dummy driver for smoke tests. It links into `pie`, uses the same
shared-memory driver ABI as the real drivers, and returns shape-correct random
or zero-valued outputs without loading weights.

Use it for CLI, scheduler, client, and constrained-decoding plumbing tests.
Do not use it for numerical behavior.

## Build

The dummy driver is always linked into `pie`.

```bash
cargo build -p pie-worker --release
```

Driver-only compile check:

```bash
cargo build -p pie-driver-dummy
```

## Config

```toml
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
vocab_size = 151936
arch_name = "qwen3"
```

`hf_repo` must still resolve to a repo or local snapshot with a tokenizer.
Only weight loading and compute are skipped.
The dummy driver derives its synthetic forward limits, context length, KV page
pool, and internal `kv_page_size`, then reports them in its capability
handshake.

## Behavior

- Samplers return random token IDs.
- Raw logits, logprobs, distributions, and entropy are shape-correct but not
  meaningful.
- Constraint masks are honored, so grammar/JSON examples still produce
  parseable output.
- Adapter operations are accepted as no-ops.
