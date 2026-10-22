# prefix-tree-kv-cache

A small prompt tree that shares model state across common prefixes, then
forks into independent leaves for generation.

## Source

Zheng et al., ***SGLang: Efficient Execution of Structured Language Model
Programs*** — <https://arxiv.org/abs/2312.07104>. Demonstrates the RadixAttention
prefix-sharing mechanism in explicit ETA working sets.

**Faithfulness: Structural — explicit fork tree, not runtime RadixAttention.**
It shows prefix KV reuse over a fixed two-level tree, but the tree is authored by
the inferlet rather than discovered automatically by the scheduler.

## What it does

If several requests share a prefix, recomputing that prefix for every request is
waste. RadixAttention stores common token prefixes once and lets multiple
continuations point into the shared cache. The benefit is largest for structured
programs that fan out from a prompt template into many suffixes.

This inferlet builds that shape directly. It pre-fills `Write a short scene set`,
forks two branches (`in a city`, `in a forest`), forks each of those into two
leaves (`at dawn`, `at night`), and then generates from all four leaves. The
interesting operation is `WorkingSet::fork`: a child shares its parent's frozen
slots and appends its own tokens after them.

## The rule

```
root = prefill("Write a short scene set")

for a in [" in a city", " in a forest"]:
    branch = fork(root)
    append(branch, a)
    for b in [" at dawn", " at night"]:
        leaf = fork(branch)
        first = append(leaf, b)
        generate(leaf, first, num_tokens)
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `num_tokens` | int | `32` | Maximum number of tokens generated from each leaf |

## Implementation notes

Every append folds what it wrote (`StateGeometry::folding()`), so the prefix
is frozen before it is forked and the branches share it rather than copy it.
The tree is built on one pipeline, which orders each fork after the append it
forks from; each leaf then decodes on its own pipeline with the usual
device-carried greedy loop.

## Run

```bash
cargo build --release --target wasm32-wasip3
python tests/examples/run_all.py --engine cuda --model <model-path>
```
