# text-completion-bench

Benchmarking and reference text-completion harness. This is not an algorithm
inferlet; it exists to drive the same PTIR completion path under benchmark
controls and to return exact token-count envelopes.

## Purpose

The crate runs an N-wide prompt prefill followed by a device-carried decode loop
with a frame-quantized run-ahead window. It is the reference implementation used
when other inferlets need output-equivalence checks against a plain completion
path, and it is also shaped for throughput benchmarks: optional batch inputs,
token counts, first-token timing, inter-token timing, and per-token WASM delay.

Unlike `chat-completion`, this harness does not stream text while generating.
It drains token ids, returns counts as the authoritative result, and decodes text
only when requested.

## What it validates

The harness validates completion plumbing rather than a decoding paper:

- explicit string prompts or pre-tokenized prompts;
- single-request and batched request shapes;
- exact output-token accounting with optional EOS ignoring;
- argmax at `temperature <= 0` and top-p Gumbel-max sampling otherwise;
- launch-inclusive timing handshakes used by benchmark clients.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `""` | Prompt for single-request mode, or fallback prompt for batched string inputs |
| `prompt_tokens` | array of int, optional | `null` | Pre-tokenized prompt for single-request mode |
| `prompts` | array of string | `[]` | Batched string prompts |
| `prompt_tokens_batch` | array of arrays of int | `[]` | Batched pre-tokenized prompts |
| `max_tokens` | int | `256` | Maximum sampler emissions per request |
| `system` | string | `"You are a helpful, respectful and honest assistant."` | System message used when tokenizing string prompts |
| `temperature` | float | `0.6` | Sampling temperature; `<= 0.0` selects argmax |
| `top_p` | float | `0.95` | Nucleus sampling threshold used when temperature is positive |
| `ignore_eos` | bool | `false` | Ignore chat stop tokens and run to the full `max_tokens` budget |
| `wasm_delay_us` | int | `0` | Busy-wait per drained token to simulate guest-side token work |
| `return_text` | bool | `true` | Decode and return generated text; benchmarks can disable it and use counts only |
| `wait_for_start` | bool | `false` | Send a `ready` session message and wait for the benchmark client to start |
| `system_speculation` | bool, optional | `null` | Accepted for input compatibility; speculation is driver-side now |
| `batch_concurrency` | int, optional | `null` | Maximum number of batched requests to run concurrently; defaults to the batch size |
| `report_timing` | bool | `false` | Return guest first-token latency, inter-token gaps, and prologue timings |
| `report_arrivals` | bool | `false` | Return shared-host monotonic arrival stamps without the live `t0` message |

## Implementation notes

The prefill sample is already the first output token. Decode fires are submitted
ahead of the host drain, and the same top-up rule is used for every frame size.
When `ignore_eos = false`, already staged fires may still settle after a stop
token is observed; their outputs are drained and ignored so counts remain exact.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
