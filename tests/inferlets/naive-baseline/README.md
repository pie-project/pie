# naive-baseline

Plain text completion. This is not an algorithm — it is the **performance
control** the other inferlets in this set are measured against.

## Why it exists

Every algorithm inferlet here reports a cost as a multiple of "naive
decoding", and that multiple is only meaningful if the naive case is measured
through the *identical* code path. This crate is that case: same skeleton, same
prefill, same channel wiring, same decode loop as every sampler inferlet. The
epilogue does only temperature scaling and a Gumbel-max draw.

The repo's two existing completion inferlets could not serve this role, both
for reasons predating this work:

- `text-completion-bench` stalls the driver on `ResizePool`.
- `chat-completion` fails with `EmbedTokens is not host-derivable: channel 0
  has no host-known value`.

## The `stats` flag

The algorithm inferlets each carry two instrumentation channels that report the
statistic proving their rule fired. Reading those channels costs something on
its own, independent of the algorithm's arithmetic. Setting `stats = true`
drains both channels while doing no extra algorithm work, which separates the
two:

| Configuration | ms/token |
| --- | --- |
| `naive-baseline` | 3.30 |
| `naive-baseline` with `stats = true` | 3.61 |

So roughly 0.3 ms/token of every algorithm's measured cost is instrumentation,
not algorithm.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `temperature` | float | `1.0` | Temperature applied before the Gumbel-max draw |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |
| `stats` | bool | `false` | Drain the two instrumentation channels the algorithm inferlets carry |

## How the numbers were produced

Two-point regression. Each configuration runs at a 32- and a 160-token budget,
7 repetitions each after a discarded warm-up, and

```
per_token_ms = (median t(160) − median t(32)) / 128
```

Differencing cancels install, JIT, prefill and teardown; the intercept recovers
them. Medians rather than minima, because
[`synthid-tournament-sampling`](../synthid-tournament-sampling) turned out to
be bimodal and `min` reports its lucky mode.

All configurations were measured **in one server session**, so the ratios are
comparable. The absolute baseline drifts about ±10 % between sessions —
2.70 to 3.60 ms/token was observed — so quote the ratios, not the absolutes.

## Results

| Inferlet | ms/token | × naive |
| --- | --- | --- |
| [`token-healing`](../token-healing) | 2.48 | 0.75× |
| [`gumbel-watermark`](../gumbel-watermark) | 2.78 | 0.84× |
| **`naive-baseline`** | **3.30** | **1.00×** |
| [`entropy-adaptive-temperature`](../entropy-adaptive-temperature) | 3.53 | 1.07× |
| `naive-baseline` + 2 stat channels | 3.61 | 1.09× |
| [`top-a-sampling`](../top-a-sampling) | 3.81 | 1.15× |
| [`eta-epsilon-sampling`](../eta-epsilon-sampling) | 4.31 | 1.31× |
| [`xtc-sampling`](../xtc-sampling) | 4.48 | 1.36× |
| [`repetition-penalty`](../repetition-penalty) | 4.89 | 1.48× |
| [`dry-repetition-penalty`](../dry-repetition-penalty) | 7.08 | 2.15× |
| [`context-aware-decoding`](../context-aware-decoding) | 13.12 | 3.98× |
| [`classifier-free-guidance`](../classifier-free-guidance) | 13.87 | 4.21× |
| [`tail-free-sampling`](../tail-free-sampling) | 17.47 | 5.30× |
| [`locally-typical-sampling`](../locally-typical-sampling) | 17.71 | 5.37× |
| [`synthid-tournament-sampling`](../synthid-tournament-sampling) | 33.55 | 10.2× |

[`asap-grammar-aligned-decoding`](../asap-grammar-aligned-decoding) is absent:
its multi-round shape does not fit a marginal-per-token model.

Three conclusions, spelled out in the audit's `## Runtime cost` section:

- **Overhead is entirely marginal, never fixed.** Intercepts land at 87–165 ms
  for every configuration including the baseline. No inferlet has heavy
  one-time setup.
- **The 5× cliff is `top_k`, not the algorithm.** `tail-free-sampling` and
  `locally-typical-sampling` compute completely unrelated statistics yet cost
  the same, because the tier-0 `k_topk_rows` is an incremental-threshold
  selection that rescans the row once per pick — `O(k · vocab)`, or 33.5 M
  element visits per token at `k_max = 128`. `top-a-sampling`, which needs no
  ranking at all, sits at 1.15×.
- **The 4× on the contrastive pair is two effects.** Two forward passes by
  construction, *plus* loss of run-ahead because the next input depends on both
  passes' combined output.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
