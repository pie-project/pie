# Models

Each directory here is a model package, written in Starlark against the
builtins of `crates/poem`. A package is whatever `.poem` files its directory
holds; what they are named says nothing. It is read by the entries it
defines, each defined once across its files:

| entry      | states                                                          |
| ---------- | --------------------------------------------------------------- |
| `MODELS`, `DEPLOYMENTS`, `PUBLISHED` | the models (ids, miniatures, parts, drafters, template, tokenizer); the deployments, in the order an import tries them; and the draft heads published for a checkpoint repository, the first listed being what `pie model import` drafts with unless told `--drafter none` |
| `layout(id, deploy)` | a deployment's dims and the weights they lay out |
| `media(id, deploy)` | if it reads stills: `struct(image = struct(patch, block, mrope, prefix, placeholder, suffix, frame))` |
| `caches(m, c)`, `forward(m, inputs)` | the caches it holds and the forward its rows run |
| `formats(m)` | the checkpoint formats it is read in, and what lands where |

One file may hold them all. A package split across files still names each
entry once; a name a file only `load()`s is not its definition. `lib/`
holds the helpers every package may load, as `//lib/…`; what a library
defines is never a package's entry.

A weight is named by where it sits in the struct `layout` returns:
`layers.3.q` is the `q` of the fourth of `layers`, `head` the field `head`.
That name is its parameter in the trace, its tensor in an artifact and what
the host finds an adapter bank by (`layers.{l}.lora_a`). A weight reached
at two paths is one weight, named by the first.

How a release stores its quantized weights is the importer's, not a
package's: `//pie/codec.poem`, which ships with pie, reads block FP8,
per-channel FP8, MXFP8 and NVFP4 as the release stores them. A package's
formats name a weight's legs (`weight(reads, w, legs)`) or a bank's experts'
(`bank(reads, w, experts)`), and the weight's declared dtype says which
encoding it keeps; a quantized leg of a weight read plainly is refused,
or, with `widen = True`, dequantized to bf16.

A deployment is named `{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-yarn{f}][-tap.{key}][-tp{n}]`; `yarn{f}` stretches the
model's native context `f` times by YaRN, in a package that turns it
(`deploy.yarn`, read into `rope_scaling(...)`); `tap.{key}`, for a parity
check, ends the forward at the intermediate `key` and reads it out instead,
in a model that states `taps = True` (`deploy.tap`).
Each of its weights is a dtype or a precision: a stored dtype whose products
quantize their activations, spelled with what they quantize to (`nvfp4a4`,
NVFP4 codes against NVFP4 activations; `mxfp4a8`, MXFP4 codes against e4m3
in 1x128 blocks). A weight declared at a precision stores its dtype's planes
and its products run on CUDA's tcgen05 GEMMs. Where there are none (Hopper,
Ampere, sm120), CUDA reads the same planes against bf16 activations: an
MXFP4 or NVFP4 bank as W4A16, a block-FP8 one as W8A16
(`crates/engine-cuda/src/dispatch/linear.rs`, `select_a16` and `matmul_a16`).

## What a model says of its tokens

A model is spoken through a template and a tokenizer, both stated as data in
`MODELS`, so no Rust names a model:

- `template(format, …)` picks a format the runtime ships (`chatml`, `harmony`,
  `gemma`, `deepseek`, `dots`, `glm`, `kimi`, `kimi3`, `inkling`, `atem`, `mimo`, `minimax`,
  `hy`, `lines`, `raw`) and sets what the format leaves open: ChatML its `thinking`,
  `preserve_thinking`, `tools`, `generation_suffix` and `stop`; `lines` its
  `stop`, `bos` and `eos`; `raw` its `stop`; `hy` its `thinking`. A setting the format does not
  read is refused.
- `tokenizer(markers, pinned, parts)` states what the model asks of the
  tokenizer an artifact carries: `markers`, groups of tokens the vocabulary
  must hold; `pinned`, markers at the id they must hold; and `parts`, the
  marker groups each part (`vision`, say) adds when a deployment serves it.

## How a still is read

Nothing about a model's pictures is in Rust. `media(id, deploy)` in
`model.poem` states how a still is spelled and framed: `patch` and
`block` (the tower's patch size and the side of the block its row fold
takes, one token a block), `mrope` (tokens placed on the merged grid or in
sequence), the delimiter tokens, and `frame(h, w, budget)`, the resize the
family's processor applies (`//lib/media.poem` has Qwen's and Gemma's).
The host decodes, frames, resamples and cuts the still into patch rows of
raw RGB bytes in block order; the tower does the rest in the forward:
`ops.layout.pixels(inputs.pixels(3 · patch²), mean, std, order, temporal)`
normalizes and lays the rows out, and `ops.layout.grid_taps(positions,
inputs.image_grids(), segments, kind, side)` taps the position table.

## Where the packages go

The runtime embeds every package here at build time (`crates/runtime/build.rs`)
and, on every start, seeds `$PIE_HOME/models/` with the same tree and reads
the catalog from there, so the directory is always complete and is what
serves. Each seeded directory records its seed's digest in `.seeded`: a
directory nobody edited is brought up to the binary's version when the
binary changes; one edited by hand is kept, and a warning says when the
built-in it started from has moved on. A new directory there is a new
model, with no build. The tests read this directory
(`poem_compiler::catalog::repository()`), so editing a package rebuilds
nothing below the runtime.

An artifact (`$PIE_HOME/artifacts/<model>/*.zt`) names the package it was
imported with and that package's digest in its serving stamp; it is served
by the package of that name and nothing else. A package edited since import
still serves if its layout is the same (the artifact's planes are checked
against the trace), and `pie model import` sees the digest differ and
writes the artifact again.
