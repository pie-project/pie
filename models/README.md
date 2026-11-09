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
| `adopt(id, config, special)` | Optional: how a checkpoint this package lists no model of is read from its `config.json` (see below) |

Deployments follow the naming convention
`{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-yarn{f}][-tap.{key}][-tp{n}]`.
Quantized checkpoint formats (block FP8, per-channel FP8, MXFP8, NVFP4) are
decoded by `//pie/codec.poem`.

## Adopting a checkpoint

A package lists its models by hand, and a checkpoint stating any other width,
depth or rope is another model's. A family whose forward does not change with
those numbers may also say how to read them from a checkpoint instead:

```starlark
def adopt(id, config, special):
    # config: config.json as a dict; special: tokenizer_config.json's
    # bos/eos/pad/unk tokens as text. fail() names what the forward does not run.
    if config.get("rope_scaling") != None:
        fail("this family turns its rope unscaled")
    return struct(
        MODELS = [model(id, ...)],
        DIMS = {id: struct(hidden = config["hidden_size"], ...)},
        DEPLOYMENTS = [deployment(id, weights = dtype.bf16, kv = dtype.bf16)],
    )
```

Every field of the returned struct rebinds the package global of that name.
`pie model adopt <owner/name | dir>` finds the family whose formats recognize
the checkpoint's tensor names, shows where it departs from the nearest listed
model, and writes `$PIE_HOME/models/adopted_<id>.poem`: the family's text as
it shipped, the configuration it read, and the rebinding. The written package
is checked as an import would check it (every tensor read, the plan
compiled, the tokenizer compiled) before it is written, and it serves without
a rebuild. Every package but `minimax_h3` adopts; each hook refuses, by
name, what its forward does not run.

For a diffusers pipeline, `config` is each component's `config.json` under the
role its tensors are prefixed by (`dit`, `te`, `vae`, …), beside the pipeline's
`model_index.json` as `pipeline` and its scheduler's as `scheduler`.

Every adopting package is checked against the models it lists:
`every_adopting_family_reproduces_its_listed_models` hands each hook a
release's own configuration (fixtures under
`crates/poem-compiler/tests/poem_compiler/adopted/`, fetched by
`scripts/adopt/fixture.py <repo> <model id>`) and requires the very trace the
listed model has, at every deployment it lists.

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
