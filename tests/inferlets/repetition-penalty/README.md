# repetition-penalty

The three standard anti-repetition penalties — frequency, presence and
CTRL-style repetition — computed from a device-resident token histogram.

## Source

Keskar, McCann, Varshney, Xiong and Socher, ***CTRL: A Conditional Transformer
Language Model for Controllable Generation*** —
<https://arxiv.org/abs/1909.05858>, §4.1, for the multiplicative repetition
penalty. The frequency and presence penalties follow OpenAI's API semantics.

The **scope split** — repetition reads `prompt ∪ output`, while frequency and
presence read the output only — follows
[vLLM](https://github.com/vllm-project/vllm)'s sampler.

**Faithfulness: Exact.** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

Maintains a count of how often each vocabulary token has already appeared, then
discounts the logits of tokens in that history before sampling. The three
penalties differ in shape: frequency scales with the count, presence is a flat
one-off charge, and repetition is multiplicative.

The multiplicative one is sign-aware, which is the detail implementations get
wrong. Dividing a negative logit by `penalty > 1` makes it *larger*, so CTRL
multiplies negative logits instead.

## The rule

```
logit(x) -= frequency  · count_output(x)
logit(x) -= presence   · [count_output(x) > 0]
logit(x)  = logit(x) / repetition   if logit(x) > 0 and x ∈ prompt ∪ output
            logit(x) * repetition   if logit(x) <= 0 and x ∈ prompt ∪ output
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `frequency_penalty` | float | `0.0` | Subtracted per prior occurrence, OpenAI semantics |
| `presence_penalty` | float | `0.0` | Subtracted once if the token occurred at all |
| `repetition_penalty` | float | `1.1` | CTRL-style multiplicative penalty; `1.0` disables |
| `temperature` | float | `1.0` | Temperature applied after the penalties |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |

## Cost

**4.89 ms/token, 1.48× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B. The cost is the scatter into the seen-token vector,
which runs once per step.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
