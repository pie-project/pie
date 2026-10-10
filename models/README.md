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
as the catalog a build ships; the tests read them from this directory
(`poem_compiler::catalog::repository()`), so editing a package rebuilds
nothing below the runtime.

`pie model import` writes the package into the artifact it produces (under
the `pie.package/` attributes, with the builtins version it was written
against), and serving traces the artifact's own package: an artifact keeps
meaning what it meant when it was imported, whatever this tree holds since.
