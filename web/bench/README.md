# Native wgpu vs. the browser — the same benchmark on both

`web/site/bench.mjs` holds one benchmark. It takes a connected client from the
JavaScript SDK and a program name, so the identical code measures:

- **native**: `pie serve` (wgpu engine, Vulkan) over its WebSocket —
  `node web/bench/native.mjs ws://127.0.0.1:28417/v1/ws`;
- **browser**: the tab, through the SDK's in-page transport —
  `node web/tools/headless.mjs web/site "bench.html?config=…"`.

Scenarios: warm-up; time to first token (short prompt, `max_tokens=1`, median
of 5); decode cost per token from 64-token runs; prefill of 200- and 600-word
prompts; 1/2/4/8 concurrent processes; 30 repeated short runs; a
device-carried inferlet (`naive-baseline`) beside the host-driven one; 200
extra runs with memory before/after (tab only). Corner cases: empty and
Korean prompts, `max_tokens` 0 and 256, malformed input, an unknown program,
terminate-then-relaunch, more processes than lanes, a prompt near and past
`max_context`, two clients at once.

`text-completion` returns all tokens at once, so per-token decode latency is
derived: (t(64) − t(1)) / 63.

Raw logs live in `results/`. The numbers in the tables below are from
2026-09-10 on an RTX 4090 (native: Vulkan; browser: Chrome 153 headless via
Dawn/Vulkan), engine limits matched (`max_forward_tokens 1024,
max_forward_requests 8, max_total_pages 2048`).

## Results

`native-0.8b-unified` is the native server on the engine as it now is: one
code path for both hosts, the browser's — every GPU wait is a callback, the
rows of a step land through `landing.rs`, and native drives the callbacks
from one poll thread (`device/host.rs`). The other `native-*` columns are
the earlier engine, which blocked the engine lane on `Device::poll` for
every frame. Not blocking the lane is what lifts eight lanes from 293 to
472 tokens/s and the 30-run maximum from 278 to 65 ms; one lane goes from
5.5–6.4 to 5.3 ms a token. `browser-0.8b-unified` is the tab on the same
build: within run-to-run noise of `browser-0.8b-matched` (decode 7.1–7.3 ms
a token across three runs, eight lanes 380–415 tokens/s), as it should be,
since the tab's path is the one that was kept.


Columns: `browser-*-matched` and `native-*` share the engine limits above;
`browser-0.8b` is the page's defaults (512 tokens, 4 lanes, 512 pages).
`native-0.8b-run2` is the same server measured a second time, freshly
restarted — its 50 s warm-up is the native engine compiling its pipelines on
the first frames (the tab compiles the same WGSL through Tint in about a
second). Browser columns are the final build (on-device guests, asynchronous
attach, the runtime in a Web Worker); the earlier browser generations are
described below.

| | browser-0.8b-matched | browser-0.8b-unified | browser-0.8b | browser-2b-matched | browser-4b-matched | browser-9b-matched | native-0.8b-run2 | native-0.8b-unified | native-0.8b | native-2b | native-4b | native-9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | qwen35-d0.8b-u4g64-kv-bf16 | qwen35-d0.8b-u4g64-kv-bf16 | qwen35-d0.8b-u4g64-kv-bf16 | qwen35-d2b-u4g64-kv-bf16 | qwen35-d4b-u4g64-kv-bf16 | qwen35-d9b-u4g64-kv-bf16 | qwen35-d0.8b-u4g64 | qwen35-d0.8b-u4g64 | qwen35-d0.8b-u4g64 | qwen35-d2b-u4g64 | qwen35-d4b-u4g64 | qwen35-d9b-u4g64 |
| boot (ms) | 964 | 73 | 1127 | 2201 | 2438 | 88 | – | – | – | – | – | – |
| memory after boot (MiB) | 677 | 285 | 677 | 1524 | 2897 | 1500 | – | – | – | – | – | – |
| warm-up run (ms) | 1617 | 1740.9 | 1474.6 | 1368.7 | 1522.5 | 1234.5 | 50036.6 | 64178.1 | 200.8 | 468.4 | 326.6 | 316.4 |
| TTFT, short prompt (ms) | 14.2 | 39.6 | 14.5 | 16.4 | 25.8 | 29.1 | 45.1 | 46.4 | 91.6 | 41.9 | 59.4 | 49.9 |
| decode (ms/token) | 7.22 | 7.1 | 7.74 | 8.41 | 11.96 | 15.11 | 6.42 | 5.25 | 5.48 | 7.94 | 10.66 | 15.57 |
| decode (tok/s, 1 lane) | 138.5 | 140.8 | 129.2 | 118.9 | 83.6 | 66.2 | 155.8 | 190.5 | 182.5 | 125.9 | 93.8 | 64.2 |
| TTFT, 200-word prompt (ms) | 435.2 | 436.2 | 453.9 | 517.7 | 872.3 | 1181.9 | 433.1 | 429.3 | 434.1 | 533.3 | 894.1 | 1245.3 |
| TTFT, 600-word prompt (ms) | 1412.1 | 1417.1 | 1478.6 | 1678.6 | 2829.9 | 3848.4 | 1456.2 | 1455.5 | 1451.3 | 1758.6 | 3009.8 | 4072.6 |
| 2 concurrent, aggregate tok/s | 192.3 | 176.5 | 180.4 | 157.3 | 114.3 | 93 | 177.2 | 265.6 | 187.3 | 153.5 | 112.2 | 69.5 |
| 4 concurrent, aggregate tok/s | 297.3 | 284 | 285 | 242.6 | 171 | 136.1 | 229.9 | 360.6 | 243.2 | 178.5 | 129 | 83.9 |
| 8 concurrent, aggregate tok/s | 394.9 | 377.8 | 300.2 | 342.9 | 228.8 | 171.8 | 293.1 | 471.8 | 170.2 | 180.8 | 143.1 | 92.2 |
| 8 concurrent, latency (ms) | 648 | 677.2 | 841 | 746.4 | 1118.8 | 1489.9 | 873.2 | 542.4 | 1504.2 | 1415.9 | 1788.4 | 2776.8 |
| 30 × 8 tokens, median (ms) | 65 | 67.4 | 64 | 74.5 | 105.8 | 134 | 99.7 | 53.1 | 96.9 | 114.9 | 111 | 156.2 |
| 30 × 8 tokens, max (ms) | 73.1 | 73.7 | 67.5 | 83.6 | 108.9 | 140 | 277.7 | 64.5 | 110.2 | 251.1 | 164.3 | 183.3 |
| device-carried decode (ms/token) | 7.22 | 6.79 | 6.75 | 7.85 | 11.61 | 14.9 | 6.2 | 5.22 | 6.56 | 7.63 | 11.22 | 15.44 |
| device-carried, 4 concurrent tok/s | 307.3 | 298.3 | 305.7 | 257.9 | 174.8 | 143 | 224 | 318.9 | 215.6 | 150.4 | 121.3 | 77.8 |
| memory after +200 runs (MiB) | 677 | 285 | 677 | 1524 | 2897 | 1500 | – | – | – | – | – | – |
| corner cases passed | 11/11 | – | 11/11 | 11/11 | – | – | 11/11 | 11/11 | 11/11 | 11/11 | – | – |

What the numbers say:

- **Time to first token** is lower in the tab (14–18 ms) than through the
  native server (42–92 ms): the tab's client is in-process, the native path
  is client → gateway WebSocket → TCP → worker, and it varies run to run.
- **Single-lane decode** is within 1–2 ms/token of native (7.2 vs 5.5–6.4 ms
  on 0.8B; 8.4 vs 7.9 on 2B). The GPU work is the same WGSL; what
  remains is WebGPU's completion model — the `mapAsync` of the readout is an
  event-loop hop that a native `Device::poll` does not pay.
- **Prefill** of long prompts is identical: GPU-bound, same kernels.
- **Aggregate throughput** under concurrency is set by how large a batch the
  scheduler fuses (up to `max_forward_requests` ready requests per fire,
  shared runtime code in `scheduler/batch.rs`) and by how fast the single
  worker thread encodes each fire. A tab fire is ~394 compute dispatches,
  each a wasm→JS→Dawn crossing, so encoding costs ~5 ms of worker CPU (native
  records the same dispatches in-process in ~0 ms and its `engine_fire` is
  mostly GPU). Profiling (`--features profile-fire`, `query("model_status")`)
  found the browser GPU idle ~7 ms between fires — starved while the worker
  encodes — at ~44 % utilization. The browser's batch caps were also lower
  than native's: `max_forward_requests` defaulted to 4 (native 8) and
  `frame_size` to 2. Raising the defaults to `max_forward_requests = 8` and
  `frame_size = 8` (both in `crates/web/src/boot.rs`) lifts 8-concurrent
  0.8B from 295 to 386 tok/s (~78 % of native's 496) with 1-lane latency
  unchanged; batch memory only grows when concurrent requests actually
  exist. What remains (386 vs 496) is the per-fire wasm→JS encode cost, which
  only fewer dispatches (kernel fusion) would close — a separate effort.
- **Device-carried loops** cost about the same as host-driven ones in the
  tab (7.2 ms/token vs 6.2 natively; 4 lanes ~300 vs 224 tok/s).
- **The client transport is not a cost.** `web/site/transport.html` boots
  the runtime without an engine and times a ping round trip: 0.01 ms through
  the JSON session API, 0.28 ms through the SDK's MessagePack shim, 2 µs for
  an empty executor tick.
- **Memory** in the tab does not grow: 677 MiB (0.8B) and 1524 MiB (2B)
  before and after 200 further runs, and across 50 sessions.
- **Corner cases** pass on both hosts (`results/*.log`, `corners` arrays);
  the tab additionally passes its host-level cases
  (`results/browser-host-corners.log`: double boot refused, garbage program
  fails at launch and the runtime survives, closed and unknown sessions
  refused, malformed messages refused, a session closed mid-run, 50 sessions
  without growth) and the page-lifecycle checks
  (`results/browser-lifecycle.log`: a first visit serves the loader's ranges
  from the network in 1.9 s and fills the OPFS cache in the background in
  1.2 s, a reload serves them from the cache in 0.8 s, a hidden tab keeps
  generating, memory stays at 285 MiB).

How the browser got here (four builds, same bench):

| | host-interpreted guests | on-device, synchronous attach | on-device, asynchronous attach | + map right behind submit |
|---|---:|---:|---:|---:|
| decode, 1 lane (ms/token) | 8.56 | 8.74 | 7.43 | 7.43 |
| device-carried (ms/token) | 16.54 | 13.84 | 8.14 | 6.93 |
| 8 concurrent (tok/s) | 358.8 | 246.3 | 408.9 | 421.7 |
| 30 × 8 tokens, max (ms) | 93.7 | 347.3 | 69.2 | 74.6 |

The first build ran attached guest programs on the wasm CPU inside the
completion callback (7 ms of Gumbel + argmax over a 248k vocabulary per
step). The second lowered them to the device but waited for each step to
land inside `submit`, which serialised frames. The third runs them from the
landing callback on a green-thread worker pool, so `submit` returns at once
and only a step that feeds on the previous token waits; it also stopped
minting a GPU buffer per step, which had been reaching the page's garbage
collector and causing the 200–400 ms stalls behind the earlier maxima. The
fourth issues the readout's `mapAsync` in the same task as the submit and
treats its resolution as the landing, saving the `onSubmittedWorkDone` hop
on every step.

The runtime then moved into a Web Worker (`web/site/worker.mjs`): per-token
cost unchanged within noise (7.2 ms/token, 396–409 tok/s at 8 lanes,
device-carried 6.8–7.1), and the page's event loop stays free — its maximum
gap during the inferlet's first launch (Cranelift + Tint compiles) is 12 ms
against 639 ms on the main thread, and 10 ms during a 200-token generation
(`corners.html`).

Two tabs at once (`web/tools/twotabs.mjs`): each boots its own runtime on
the same GPU (677 MiB each), both return the reference completion, and a
warm 16-token run takes 210 ms per tab against 130 ms alone — the GPU is
shared, nothing else is.

Memory. wasm32 grows to the full 4 GiB in Chrome, but one Rust allocation
cannot pass 2 GiB (`isize::MAX` on a 32-bit target), and an artifact copied
into the tab was bounded by that (the first design topped out around the
4B at 2.9 GiB in use). The mount is therefore *lazy*: the loader's reads are served as 8 MiB-aligned byte ranges by the
page while the loading green thread parks, so the tab never holds the
artifact — 0.8B boots with 285 MiB, 4B (`qwen35-d4b-u4g64`, 2.39 GB; the
catalog row was added for this) with 650 MiB, and **Qwen3.5-9B in 4 bits
(5.06 GB) with 1.5 GiB**, 15.1 ms/token on one lane and 172 tok/s over
eight, the same greedy tokens as native, no growth over 200 runs
(`browser-9b-matched`). WebGPU does not report device memory, so a large
model needs `device_memory_mb` in the tab's config (the engine assumes 8 GiB
otherwise). The native wgpu server's 9B column (`native-9b`) needed one more
knob: `max_state_slots = 16`. With the default of 256 slots the recurrent-state
pool alone is 12 GiB (24 linear-attention layers × 2 MiB per slot), which
leaves the dense weights less device memory than they need, so the engine
silently rotates them through a host-tier ring — every token pulls about
5 GB over PCIe (3.5–6 GB/s on `nvidia-smi dmon`) and decode runs at
840 ms/token, first minute and fifth minute alike. That is what the earlier
"no clean number" was. With the pool sized for the lanes actually served
(16 slots seat 8 lanes at dispatch depth 2) the two hosts decode 9B at the
same speed; the engine now warns at boot when any dense plane leaves the
device (`engine_wgpu::weights`). The tab cannot hit this silently: its
build refuses a layout that would stream anything, so a pool too large for
the model is a boot error there, not a slow decode (the page defaults to
64 slots).

## How pie compares to other in-browser engines

The same RTX 4090 and the same headless Chrome (WebGPU on the real
`nvidia`/`lovelace` adapter, confirmed by `navigator.gpu.requestAdapter`),
single-stream greedy decode, 4-bit weights. Different engines can only run
the models their toolchains ship, so this is not one model across all
engines — read it as a per-engine snapshot, not a controlled A/B.

| engine | model | decode (tok/s) |
| --- | --- | --- |
| pie | Qwen3.5-0.8B | 144 |
| pie | Qwen3.5-2B | 95 |
| WebLLM / MLC | Qwen2.5-0.5B (q4f32) | 72 |
| WebLLM / MLC | Qwen2.5-1.5B (q4f32) | 41 |
| llama.cpp WebGPU | Qwen3.5-2B (Q4_K_M) | 1.8 |

- **WebLLM / MLC** is pie's only close peer. It could not run pie's exact
  model (Qwen3.5 is not in MLC's prebuilt catalog, and compiling one needs
  the `mlc_llm` + TVM toolchain), so the comparison is by size: pie's 0.8B
  beats WebLLM's 0.5B and pie's 2B more than doubles WebLLM's 1.5B — roughly
  2× at a comparable size. WebLLM ran q4f32 because q4f16 failed to compile
  in this headless Chrome. That is not a WebLLM bug: the WebGPU adapter here
  does not expose the `shader-f16` feature at all (a Dawn/driver limit on
  this 4090 config — `requestDevice({requiredFeatures:["shader-f16"]})` is
  refused, and a developer-features flag does not change it), so WebLLM's
  native-f16 models cannot run. pie is unaffected because its kernels carry
  bf16 packed into u32 by hand rather than using WGSL's `enable f16`, so they
  run without the extension — a portability edge in its own right. f16 would
  add perhaps 20–30% to WebLLM where available, still behind pie.
- **llama.cpp WebGPU** is the May-2026 WebGPU backend for llama.cpp (the
  "Llamas on the Web" demo), driven through its own benchmark UI on the same
  Qwen3.5-2B pie runs. It reported 2.7 tok/s prefill and 1.8 tok/s decode on
  the confirmed real 4090 — about 50× behind pie on the identical model.
  This reflects how new and unoptimized that backend is ("no mature WebGPU
  BLAS libraries yet"), not llama.cpp's CPU or CUDA paths, and it will
  improve; the project's own paper reports absolute numbers several times
  below a per-engine native benchmark. `wllama` (llama.cpp on WASM) is
  CPU-only and not a WebGPU comparison; `transformers.js` (ONNX Runtime Web)
  would not finish importing in this headless setup.

The shared ceiling for every WebGPU engine here is the CPU-side cost of
submitting each fire's compute dispatches across the wasm→JS→Dawn boundary
(measured ~24–36 µs per dispatch, matching the published characterizations),
so the lever that lifts all of them is fewer dispatches per token through
kernel fusion. pie's runahead already overlaps that submission with GPU
execution and it batches concurrent requests, which is why it leads here
despite fusing less aggressively than TVM.
