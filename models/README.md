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

Any other `.poem` file is a helper the stages may `load()`.

A deployment is named `{id}[-{part}…][-{drafter}]-{weights…}-kv-{kv}[-tp{n}]`.

`pie model import` writes the package into the artifact it produces (under
the `pie.package/` attributes, with the builtins version it was written
against), and serving traces the artifact's own package: an artifact keeps
meaning what it meant when it was imported, whatever this tree holds since.
