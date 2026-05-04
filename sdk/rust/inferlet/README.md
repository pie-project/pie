# inferlet Rust SDK

Rust API for writing Pie inferlets.

```rust
use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};

#[inferlet::main]
async fn main(_: ()) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let mut ctx = Context::new(&model)?;

    ctx.system("You are helpful.")
        .user("What's 2 + 2?")
        .cue();

    ctx.generate(Sampler::Argmax)
        .max_tokens(64)
        .collect_text()
        .await
}
```

## Main pieces

- `Context`: owns KV-cache state and chat/raw token buffers.
- `Forward`: runs one explicit forward pass with samplers, probes, masks, and
  manual page control.
- `Generator`: multi-step generation loop with stop conditions, constraints,
  speculation, adapters, and JSON collection.
- `chat`, `reasoning`, `tools`: optional decoders and helpers for model-native
  formats.
- `runtime`, `scheduling`, `messaging`: host services exposed to inferlets.

## Build

```bash
rustup target add wasm32-wasip2
cargo build --target wasm32-wasip2 --release
```

Most examples live under `inferlets/`. For constrained decoding details, see
[`sdk/CONSTRAINED_DECODING.md`](../../CONSTRAINED_DECODING.md).
