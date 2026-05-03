# driver/dummy

Rust dummy driver for `pie-standalone`. Same C ABI as
`driver/portable` and `driver/cuda` (`pie_driver_dummy_run` /
`pie_driver_dummy_request_stop`), same shmem wire protocol, no model
load and no real compute. Each `fire_batch` request returns a vector of
random token ids, one per sampler slot.

The Python `pie_driver_dummy` worker (under `pie/src/pie_driver_dummy/`)
shells through `pie_driver/worker.run_worker` whose shmem fast path
hard-imports `librt.so.1` at module load — Linux-only. This Rust dummy
links into `pie-standalone` directly, calls POSIX `shm_open` /
`mmap` via `libc`, and works on Linux **and** macOS.

## Build

Standalone smoke (just compile the crate):

```bash
cargo build -p pie-driver-dummy
```

End-to-end with `pie-standalone`:

```bash
cargo build -p pie-standalone --no-default-features --features driver-dummy --release
./target/release/pie --config /path/to/dummy.toml
```

`driver-dummy` is mutually exclusive with `driver-portable` /
`driver-cuda`. One driver per binary.

## Config

```toml
[[model]]
name = "default"
hf_repo = "/path/to/dir/with/tokenizer.json"

[model.driver]
type = "dummy"
device = ["cpu"]              # required by validate(); ignored by the dummy

[model.driver.options]
kv_page_size       = 16
max_num_kv_pages   = 256
max_batch_tokens   = 4096
max_batch_size     = 128
vocab_size         = 32000    # no model file to introspect
arch_name          = "qwen3"  # runtime uses this for chat-template lookups
max_model_len      = 4096
```

`hf_repo` still needs to point at a directory containing a real
`tokenizer.json`. The runtime instantiates the tokenizer host-side; only
weight loading is skipped.

## Supported sampler / probe types

The dummy decodes `A_SAMPLER_TYPES` per slot and emits a shape-correct
placeholder for each:

| Type | Output |
|---|---|
| `Argmax` / `TopP` / `TopK` / `MinP` / `TopKTopP` / `Multinomial` | random `u32` in `[0, vocab_size)` |
| `RawLogits` | `vocab_size * 4` bytes of native-endian f32 zeros |
| `Distribution` | empty `(ids, probs)` pair |
| `Logprob` | `[0.0]` |
| `Logprobs` | `[0.0; K]` (K from `sampler_label_indptr`) |
| `Entropy` | `0.0` |

Responses always use `RESP_MODE_MSGPACK` so probe slots round-trip with
their proper shape.

## Limitations

- **Probe outputs are dimensionally correct but numerically meaningless.**
  Watermarking, log-prob reranking, etc. compile and run, but the values
  they read are zeros — the inferlet's downstream math will produce
  deterministic-but-not-meaningful results. For real numerics, run a real
  driver.
- **Speculative decoding without verifier semantics.** The dummy emits
  one random token per sampler slot, including spec-verifier slots, so
  custom `Speculator` impls don't crash but every draft is effectively
  rejected (random ≠ predicted).
- **Adapter ops are no-ops.** `Adapter::create` / `lora.load(path)` /
  swap / page-copy succeed without doing anything. The dummy ignores
  `adapter_indices` in `fire_batch` requests entirely.

If you need real numbers, real spec acceptance, or real adapters, run a
real driver — the dummy is for plumbing tests.

## Tip — running the marketing examples in `what-is-pie.mdx`

Tabs 2 (watermark) and 3 (LoRA + spec decode) need a couple of runtime
knobs set:

```toml
[runtime]
allow_fs = true          # tab 3 writes the LoRA blob to /scratch
allow_network = true     # tab 3 fetches the LoRA from an HTTPS URL (default true)

[[model]]
name = "default"
hf_repo = "/path/to/dir/with/tokenizer.json"

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
vocab_size = 151936
arch_name = "qwen3"
```
