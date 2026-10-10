# Models

Each directory here is a model package, written in Starlark against the
builtins of `crates/poem`. A package is four kinds of file, each evaluated
with only the builtins of its stage:

| file           | states                                                          |
| -------------- | --------------------------------------------------------------- |
| `package.poem` | `MODELS` (ids, miniatures, parts, drafters, template, tokenizer) and `DEPLOYMENTS`, in the order an import tries them |
| `model.poem`   | `layout(id, deploy)`: a deployment's dims and the weights they lay out |
| `forward.poem` | `caches(m, c)` and `forward(m, inputs)`: the caches it holds and the forward its rows run |
| `formats.poem` | `formats(m)`: the checkpoint formats it is read in, and what lands where |

Any other `.poem` file is a helper the stages may `load()`; `lib/` holds
the helpers every package may load, as `//lib/…`.

A deployment is named `{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-tp{n}]`.

## What a model says of its tokens

A model is spoken through a template and a tokenizer, both stated as data in
`package.poem`, so no Rust names a model:

- `template(format, …)` picks a format the runtime ships (`chatml`, `harmony`,
  `gemma`, `deepseek`, `glm`, `kimi`, `kimi3`, `inkling`, `atem`, `lines`,
  `raw`) and sets what the format leaves open: ChatML its `thinking`,
  `preserve_thinking`, `tools`, `generation_suffix` and `stop`; `lines` its
  `stop`, `bos` and `eos`; `raw` its `stop`. A setting the format does not
  read is refused.
- `tokenizer(markers, pinned, parts)` states what the model asks of the
  tokenizer an artifact carries: `markers`, groups of tokens the vocabulary
  must hold; `pinned`, markers at the id they must hold; and `parts`, the
  marker groups each part (`vision`, say) adds when a deployment serves it.

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
