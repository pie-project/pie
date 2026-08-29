# top-a-sampling

Truncation with a floor that scales with the *square* of the peak probability,
so it tightens sharply when the model is confident and relaxes when it is not.

## Source

No canonical paper. Reference implementations:
[KoboldAI](https://github.com/KoboldAI/KoboldAI-Client) and
oobabooga/text-generation-webui.

**Faithfulness: Exact.** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

Min-p uses a floor linear in `p_max`. Top-a squares it. The quadratic term is
the entire idea: when the model is certain (`p_max ≈ 1`) the floor is `a`, but
when it is uncertain (`p_max ≈ 0.1`) the floor collapses to `a/100`, admitting
a much wider field. The response to confidence is therefore far more
aggressive than min-p's, with one parameter and no sort.

## The rule

```
keep = {x : p(x) >= a · p_max²}
```

The mask is never empty — the argmax satisfies `p_max >= a · p_max²` whenever
`a · p_max <= 1`, which holds for every `a <= 1`.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `a` | float | `0.2` | Floor coefficient in `p >= a·p_max²` |
| `temperature` | float | `1.0` | Temperature applied before truncation |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |

## Cost

**3.81 ms/token, 1.15× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B — one `ReduceMax` and one compare. This is the cheapest
truncation here precisely because it needs no ranking: contrast the 5.3× paid
by `tail-free-sampling`, which must sort.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
