# naive-baseline

Plain text completion. This is not an algorithm — it is the **performance
control** the other inferlets in this set are measured against.

## Why it exists

Every algorithm inferlet here reports a cost as a multiple of "naive
decoding", and that multiple is only meaningful if the naive case is measured
through the *identical* code path. This crate is that case: same skeleton, same
prefill, same channel wiring, same decode loop as every sampler inferlet. The
epilogue does only temperature scaling and a Gumbel-max draw.

The repo's two existing completion inferlets do not serve this role. When this
crate was written the reason given was that neither one ran — `text-completion-
bench` was said to stall the engine on `ResizePool`, and `chat-completion` to
fail with `EmbedTokens is not host-derivable: channel 0 has no host-known
value`. **Both of those failures are gone, and neither was ever the real
reason.** `cuda_forward` drives `text-completion-bench` to a coherent sixteen
tokens on `cuda_native` in about two and a half seconds, and `chat-completion`
passes the curated suite on `cuda_native` and on `vulkan` alike.

The reason that survives is the one this control actually needs, and it is a
structural one that no bug fix can change: a baseline is only a baseline if it
is measured through the *identical* code path. `chat-completion` builds a chat
template and drains through the client edge; `text-completion-bench` is a
throughput harness with its own budget loop. This crate has the algorithm
inferlets' exact skeleton — one N-wide prefill fire, a `eta::run_ahead` decode
loop, the same channel wiring — and differs from them in the epilogue and
nowhere else. Subtracting it therefore leaves the algorithm and nothing else.

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
- **The 5× `top_k` cliff was a kernel defect, and it is fixed on Metal.**
  `tail-free-sampling` and `locally-typical-sampling` compute completely
  unrelated statistics yet cost the same, which identified `top_k` rather than
  the algorithms as the cause. The kernel rescanned the row once per pick —
  `O(k · vocab)`, or 19.4 M element visits per token at `k_max = 128` — on a
  single 256-thread block. A radix select plus a bitonic sort of the survivors
  brings them to 1.49× and 1.54× and makes the cost flat in `k`. What remains
  is the schedule barrier; `top-a-sampling`, which needs no ranking at all,
  sits at 1.15×.

  **The two rows in the table above are the PRE-fix numbers**, and they are
  the last ones anyone can quote, because the fix is
  `codegen/metal/topk.rs::emit_grouped_topk` — one backend's grouped library
  kernel. `codegen/cuda` has written no library kernels at all, so on the
  `cuda_native` engine the `## Run` line below invokes, these two inferlets do
  not run slowly: they do not run. A single-op `top_k` lift reaches the CUDA
  emitter, which declines it with `generated region contains a non-generated
  boundary (top_k)`, and the engine refuses the program (`crates/worker/tests/
  cuda_canaries.rs` records the same three fixtures under the same class).
  Re-measuring the pair against this baseline needs either that CUDA kernel or
  a re-run of the whole table on a backend that has one.
- **The 4× on the contrastive pair is two effects.** Two forward passes by
  construction, *plus* loss of run-ahead because the next input depends on both
  passes' combined output.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
