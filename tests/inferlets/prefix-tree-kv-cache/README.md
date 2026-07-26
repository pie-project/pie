# prefix-tree-kv-cache

A small prompt tree that shares KV cache pages across common prefixes, then
forks into independent leaves for generation.

## Source

Zheng et al., ***SGLang: Efficient Execution of Structured Language Model
Programs*** — <https://arxiv.org/abs/2312.07104>. Demonstrates the RadixAttention
prefix-sharing mechanism in explicit PTIR working sets.

**Faithfulness: Structural — explicit copy-on-write tree, not runtime RadixAttention.**
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
interesting operation is `WorkingSet::fork`: descendants inherit the parent's KV
pages and append their own tokens copy-on-write.

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

A KV working set is scoped to the first pipeline that fires it, so the whole tree
build and all four leaf generations stay on one pipeline. The append and decode
passes keep the page CSR aligned with the true sequence length using
`page_count = ceil(kv_len / page_size)`, then run the usual device-carried greedy
decode loop for each leaf.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```
