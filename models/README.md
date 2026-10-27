# Models

Each `.poem` file (or directory of `.poem` files) here is a model package,
written in Starlark against [`crates/poem`](../crates/poem). Shared helpers
live in [`lib/`](lib) and are loaded as `//lib/…`.

| Entry | Purpose |
|---|---|
| `MODELS`, `DEPLOYMENTS`, `PUBLISHED` | Model metadata (`template`, `tokenizer`, parts, drafters), deployments in import preference order, and published draft heads |
| `layout(id, deploy)` | Weight struct; each field's path (`layers.3.q`) names its parameter in the trace and its tensor in the `.zt` artifact |
| `media(id, deploy)` | Image framing (`patch`, `block`, `mrope`, delimiters, `frame(h, w, budget)`) for vision models |
| `caches(m, c)`, `forward(m, inputs)` | KV / recurrent state caches and the traced forward pass |
| `formats(m)` | Checkpoint format mappings onto the declared layout |

Deployments follow the naming convention
`{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-yarn{f}][-tap.{key}][-tp{n}]`.
Quantized checkpoint formats (block FP8, per-channel FP8, MXFP8, NVFP4) are
decoded by `//pie/codec.poem`.

## Tokenizers, templates, and media

No Rust code names a model family:

- `template(format, …)` configures a built-in chat format (`chatml`, `harmony`,
  `gemma`, `deepseek`, `dots`, `glm`, `kimi`, `kimi3`, `inkling`, `atem`,
  `mimo`, `minimax`, `hy`, `lines`, `raw`).
- `tokenizer(markers, pinned, parts)` declares required vocabulary markers and
  pinned token IDs.
- `media(id, deploy)` declares how the host resizes and cuts images into RGB
  patch rows before the forward pass normalizes and positions them.

## Runtime catalog

All packages are embedded at build time and seeded into `$PIE_HOME/models/` on
startup. Unmodified files update automatically when the binary changes;
hand-edited or newly added `.poem` files are preserved and served without
rebuilding Pie.
