# pie in the browser

The same runtime, WebGPU engine and inferlet sandbox as `pie serve`, compiled
to `wasm32-unknown-unknown` and running in one tab. No server-side inference:
the page fetches a `.wgpu.zt` artifact, the weights land in WebGPU buffers,
and inferlets run under wasmtime's Pulley interpreter inside the same module.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked   # must match Cargo.lock
./web/build.sh                      # release; `min` for the 23 MB shipping build, `dev` for debug
```

`build.sh` builds `crates/web` (the host), runs wasm-bindgen into
`web/site/pkg/`, and builds the sample inferlet into `web/site/inferlets/`.

## Run

Import a model for the wgpu engine with the native binary, then serve the site
and the artifact from any static server that sends `application/wasm`:

```bash
PIE_HOME=~/.pie-web pie model import Qwen/Qwen3.5-0.8B     # writes <slug>.<sku>.wgpu.zt
ln -s ~/.pie-web/models/Qwen--Qwen3.5-0.8B web/site/models
node web/tools/serve.mjs web/site 8765
```

The artifact is read by byte range as the loader asks for it (`bootLazy`
in `pie.mjs`; see "Big models"), and a copy is kept in the origin's private
file system, keyed by URL, size and ETag, so a reload reads it from there.

Open <http://127.0.0.1:8765/> in Chrome 137+ (JSPI and WebGPU are on by
default there). The runtime runs in a dedicated Web Worker, so the page
stays responsive through the inferlet's in-browser compile and every
forward pass (`?worker=0` keeps it on the main thread for debugging). The page loads the model, installs `text-completion`, and runs
a prompt. Query parameters drive it unattended: `?auto` runs all three steps,
`&prompt=…&max_tokens=8`, `&config=key%20%3D%20value,…` (see `BootConfig` in
`crates/web/src/boot.rs`), `&log=runtime%3Ddebug` for the console filter,
`&debug` prints the scheduler's state every 20 s, `&trace` logs every fiber
switch, `&ticks` logs executor ticks.

`sdk.html` does the same through the unmodified JavaScript client SDK
(`sdk/client/javascript`): `web/site/client-shim.mjs` routes `pie://` URIs to an
in-page transport that speaks the SDK's MessagePack frames, so
`new PieClient("pie://local")` launches and streams exactly as against
`pie serve`. (The page pulls `msgpack-lite` and `@noble/hashes` from esm.sh.)

Measured on an RTX 4090 through Dawn/Vulkan, Qwen3.5-0.8B 4-bit: boot in about
2 s from the network (the artifact served by byte range), a cold first
completion (pipelines compile on the way) in about 1.5 s, then 7.1–7.3 ms
per token on one lane (about 140 tokens/s) and 380–415 tokens/s aggregate
over eight lanes; time to first token 14 ms. The native wgpu server on the
same GPU, on the same engine code, does 5.3 ms per token on one lane and
472 tokens/s over eight. `web/bench/README.md` has the full comparison. The
tab's wasm memory sits at 285 MiB after boot (650 MiB for the 4B, 1.5 GiB
for the 9B); the artifact is never held in the tab, and wasm memory never
shrinks.

Headless, with the Chromium Playwright installs (`web/tools/browsers.mjs`
finds Playwright in `$PIE_PLAYWRIGHT`, then `web/tools/node_modules`, then
the usual places):

```bash
npm install --prefix web/tools playwright@1.63.0
npx --prefix web/tools playwright install chromium   # and firefox, for the Firefox runner
node web/tools/headless.mjs web/site "index.html?auto&max_tokens=8"
```

It exits 0 when the inferlet returns. `web/test.sh` runs the whole GPU
suite the same way: the kernels through Tint, a build, a completion, the
host corner cases, the transport cost, the client SDK page, the SDK corner
cases, a streaming chat completion, nineteen deterministic inferlets from
the matrix below (device-carried loops and masks, watermarks,
grammar-constrained decoding, token healing, a beam of one, the
epilogue-op probe), and the page lifecycle;
`PIE_WEB_MODELS="models4b/… models9b/…"` boots more artifacts.

## Benchmarks and corner cases

`web/bench/README.md` compares the tab with the native wgpu server using one
benchmark over the JavaScript SDK (`web/site/bench.mjs`; drivers
`web/bench/native.mjs` and `web/site/bench.html`), plus the tab-only host
cases in `web/site/corners.html`. `node web/bench/compare.mjs` prints the
table from `web/bench/results/`.

## The inferlet matrix

`web/tools/matrix.sh` runs every general-purpose inferlet in `tests/inferlets`
(built for `wasm32-wasip2`) once in the tab and once against a native
`pie serve`, with the inputs in `web/site/matrix.json`, and prints which ran
and which answered the same (`web/site/matrix.html` is the tab side, one
fresh tab per entry; `web/tools/matrix-native.mjs` the native side). What it
found on the way: a page-side bug (an event arriving in the same batch as
the launch response was dropped), a Chrome one (Tint folds the WGSL literal
`-0.0` to `+0.0`; `web/tools/negzero.html` shows it, and the guest runtime
library now builds negative zero from bits), and the expected: samplers at
a real temperature may pick differently on the two hosts, since Tint and
naga emit floating point differently and a Gumbel-max draw over two
near-tied candidates goes either way — at temperature 0.05 every sampler
answers the same on both. Entries that fail identically on both hosts are
not the port's: some are built for other model families or engine
interfaces (`forward` on a hybrid model), and the rest were the runtime's
and the guest runtime library's, found through this matrix and fixed on
the way:

- `attention-sink`, `sliding-window-attention`, `naive-masked` and
  `consensus-decoding` carry a loop-carried token channel written by the
  device pass beside a channel-bound dense attention mask. The runtime used
  to classify such a program neither as a decode envelope (its envelope had
  no per-lane mask) nor as pool-owned device geometry (its page channel is
  one-dimensional), and fell back to evaluating the next fire's geometry on
  the host, where a device-decided token is unknown. A channel-bound boolean
  mask of one row per token now rides the decode envelope as a
  device-resolved port beside the tokens: the wgpu engine publishes its
  host-role channel rings as host mirrors, honours the runtime's channel
  tickets (`device_channel_commit`, the way CUDA does), and reads the
  tokens and the mask rows of each lane off the rings at the fire —
  including one lane per beam or candidate for a multi-lane instance,
  whose read-out rows are joined into the one `[lanes, vocab]` rectangle
  the epilogue reads.
- `beam-search`, `consensus-decoding` and `sampling-primitives` then ran
  but answered nonsense: the WGSL guest runtime implemented `broadcast` as
  a flat copy (right for a scalar, and for a `[1, v]` row widened to
  `[2, v]` it read straight past the source into the next value), and
  read a nucleus pivot's predicate payload from the wrong parameter word
  (the channel slot). `tests/inferlets/sort-probe` now recomputes every
  epilogue op a beam or a candidate set uses (`log_softmax`, `cumsum`,
  `sort_desc`, `top_k` over `[v]` and over a broadcast `[2, v]`,
  `reduce_argmax` over two rows, a `pivot_threshold` nucleus) on the host
  from the published logits, and is part of the matrix.
- `contrastive-decoding`'s amateur prefill has an epilogue that only puts
  the logits on a channel; it lowers to no dispatch at all, which the
  wgpu guest session refused. It now runs as the copies around it.
- `text-completion-bench` submits its decode fire while the prefill whose
  epilogue seeds the decode's loop-carried token is still airborne. The
  engine reads a device-resolved envelope off the ring at the fire, so a
  ring that is empty now settles every airborne pass of any instance bound
  to that channel before it is declared empty.
- `prefill-rows` reads every row of a five-token prefill out through one
  lane; the readout seat was sized by lanes and is now sized by the rows
  named.
- `consensus-decoding-1` (one candidate) failed to bind on either host
  (`incompatible operand shapes [v] vs [1, v]` in the epilogue): its
  batched-decode epilogue read `intrinsics::logits()` as `[B, vocab]`, but
  a fire that reads out a single row squeezes the logits to rank-1 `[vocab]`.
  The inferlet now reshapes back to `[B, vocab]` for one candidate, the way
  the beam search does, and leaves the bare intrinsic for `B >= 2` (a reshape
  there defeats the library's fused nucleus fast path). The bind error was
  raised by the runtime's shape checker, which reported the same message on
  both hosts before the fix, so it was a program bug, not the port's, and the
  reshape that satisfies the checker is engine-independent.

A CUDA `pie serve` on the same GPU (`cargo build --release -p pie
--features cuda`, engine type `cuda_native`) is the reference for what the
matrix cannot recompute on the host: its logits are bf16 and tie where the
wgpu engine's f32 logits do not, so a greedy pick over two near-tied tokens
can differ between the two (`sampling-primitives` picks token 11751 on
wgpu and 303 on CUDA, tied at 14.4375 in bf16), while `sort-probe`'s
single-row checks pass on both and the 0.9 nucleus keeps 249 tokens on
wgpu and 254 on CUDA.

The run of 2026-09-11 (RTX 4090, Qwen3.5-0.8B in 4 bits): 41 entries, 35
ran on both hosts, 23 answered the same, and every entry that answers
differently samples at a real temperature. The five that fail on both
hosts are all inferlets built for the `pie:inferlet/forward` interface,
which a hybrid model refuses. (`dry-repetition-penalty` at its sampled
temperature can also trip its own "nothing was penalized" self-check on
one host when the short completion happens not to repeat; the cold variant
matches on both, and `sort-probe` checks the DRY device ops directly.) One
entry may
take minutes (`synthid`'s compile, a beam search rebinding every step), so
the per-entry timeout is four minutes (`PIE_MATRIX_ENTRY_TIMEOUT_MS`) and a
busy GPU still turns the longest into timeouts to rerun alone.
A guest program whose device code is very large (`synthid`, two 4,400-line
kernels) takes a while to compile, and while the engine lane waits for the
compile nothing else runs. Each op of a guest stage is its own dispatch,
and a driver compiles the code an entry point reaches. On the desktop
Vulkan/Metal backends and lavapipe that is done afresh per pipeline, so an
entry that reaches the whole `ptir_step` switch drags every op into every
one of a stage's hundreds of compiles (the epilogue probe took four minutes
on lavapipe); the native emitter names the one op each step runs instead,
which cuts that to seconds. Chrome's Dawn reuses the compile of an entry
that reaches identical code, so there the shared switch is faster and the
browser emitter keeps it. The split is by target (`codegen::wgsl::specialize`,
a `cfg!(target_arch)` gate), so the browser WGSL is unchanged.

## Big models

The artifact is not held in the tab at all: the mount is *lazy* (`bootLazy`
in `pie.mjs`, the only boot there is). The loader's reads are
served on demand as 8 MiB-aligned HTTP `Range` requests (or slices of the
OPFS copy when one exists) by the page, while the loading green thread parks
inside the wasm; a small window cache keeps the manifest and the loader's
neighbourhood resident. After a boot from the network the page copies the
artifact into the origin's private file system in the background (files up
to about 2 GB), so the next visit serves its ranges locally; `awaitCache()`
in `pie.mjs` waits for that copy. Peak wasm memory is then a few times the largest
tensor, not the artifact: 0.8B boots with 285 MiB, 4B with 650 MiB, and
**Qwen3.5-9B in 4 bits (5.06 GB) with 1.5 GiB**, generating the same tokens
as the native engine. (Copying the whole artifact into wasm memory was the
first design; one Rust allocation on wasm32 cannot pass 2 GiB and the tab's
memory ends at 4 GiB, so it topped out around the 4B, and the lazy mount
replaced it.) Two things
to know: the engine cannot ask WebGPU how much device memory there is and
assumes 8 GiB, so a large model with the default pools is refused
("raise the device weight budget") — pass `device_memory_mb = 20480` (or
smaller `max_total_pages`) in the boot config; and the origin's private file
system may refuse to cache a file past about 2 GB in a fresh profile, in
which case the model is fetched again on the next visit. wasm-bindgen
0.2.127 writes a string-returning import's result through signed 32-bit
pointer arithmetic, which breaks above the 2 GiB mark; `build.sh` rewrites
those stores unsigned in the generated glue, and the runtime also frees the
artifact before it boots so guest fibers land low.

## Other browsers

Firefox 155 (Playwright's build, `web/tools/headless-firefox.mjs`) runs the
whole substrate — wasmtime compiling a component in the tab, JSPI-backed
fibers, green threads, channels and timers
(`javascript.options.wasm_js_promise_integration` must be on) — and
booted without an engine (`config=engine%20%3D%20false`) the runtime comes
up in about 1 s, installs an inferlet, and refuses a launch by name ("no
engine is serving this model"). What it lacks on Linux is WebGPU: `navigator.gpu` is absent ("WebGPU is disabled
by blocklist"), so the engine cannot open a device and boot stops with a
clear error. Safari is untested (no JSPI as of this writing).

## Gates

`./web/check.sh` needs no GPU: clippy with `-D warnings` for every crate the
tab links, on `wasm32-unknown-unknown` and natively, the native tests of the
touched crates, rustfmt on them, and `node --check` over the page scripts.
CI runs it as the `browser-check` job. `./web/test.sh` is the GPU half:
every kernel through Tint, then a real completion in headless Chrome.

Two more CI jobs run the engine itself without a GPU, on a random-weight
miniature of Qwen3.5 (`web/tools/tiny_qwen35.py` writes the snapshot the
`qwen35-tiny` catalog row imports: 4 layers, 256 wide, the real tokenizer,
a 43 MiB artifact that says nothing in particular but says the same thing
on every host):

- `wgpu-lavapipe` (`web/tools/ci-lavapipe.sh`): the native wgpu server over
  Mesa's lavapipe, the SDK corner cases against it, and `sort-probe` (the
  device epilogue ops recomputed on the host). It is the same
  engine code the tab runs, with a poll thread standing in for the page's
  event loop. Locally: `sudo apt install mesa-vulkan-drivers` and run the
  script; `PIE_TOKENIZER_DIR` points at a directory that already holds
  `tokenizer.json`, else it is fetched.
- `tint-swiftshader`: every kernel variant through Chrome's Tint on
  SwiftShader (`PIE_CHROME_ARGS="--use-webgpu-adapter=swiftshader
  --enable-unsafe-swiftshader"`). Chrome will not use lavapipe, and
  SwiftShader allows 10 storage buffers per stage where the paged-attention
  kernels bind 11 or 12, so those are reported as beyond the adapter's
  limits rather than refused; the model itself does not run there.

The miniature also runs in the tab on a real GPU
(`index.html?model=models-tiny/…`), and the tokens it produces there are
the ones lavapipe produces natively.

## Check the kernels against Tint

Chrome compiles WGSL with Tint, which refuses some things naga accepts. After
any kernel change:

```bash
cargo run -p kernels-wgpu --example dump_wgsl -- web/tools/wgsl
node web/tools/headless.mjs web/tools wgsl-check.html      # "312 ok, 0 refused by Tint, 0 beyond this adapter's limits"
```

## How it fits together

| piece | what it does |
|---|---|
| `crates/wasmtime-web` | wasmtime's custom platform: heap-backed "virtual memory", TLS slots, and fibers implemented by the page with JSPI (`web/site/platform.mjs`). Guests are compiled by Cranelift *in the tab* to Pulley bytecode. |
| `crates/web-rt` | The executor the page ticks, timers over `performance.now()`, crossbeam-shaped channels, and green threads on fibers. The runtime's scheduler and engine lanes run on those unchanged. |
| `crates/wasi-web` | WASI host for the inferlet's imports: cli, io, clocks, random (0.2 and the 0.3 clocks). No files, http or sockets: a guest importing those is refused at launch, by name. |
| `crates/runtime` (`crate::rt`) | One facade over tokio/std/crossbeam natively and `web-rt` on wasm32. |
| `crates/engine-wgpu` | The same engine on both hosts, written for the browser: no GPU wait blocks, every step lands by callback (`landing.rs`), guests run from the landing. What differs by host sits in `device/host.rs` — where a wait parks (a green thread in the tab, an OS thread natively) and who fires the callbacks (the page's event loop, or native's one poll thread). |
| `crates/web` | The wasm-bindgen host: boot, install, sessions speaking `ClientMessage`/`ServerMessage` as JSON, `pie_tick`. |
| `web/site` | `worker.mjs` (the host: wasm init, tick loop, sessions, the artifact served by byte range; runs in a Web Worker), `pie.mjs` (the page-side proxy with the same API), `platform.mjs` (fibers, clock, wake), `client-shim.mjs` (the SDK's in-page WebSocket), `index.html`; the test pages share `harness.mjs`. |

Device-carried decode loops (`naive-baseline`, `chat-completion`, …) run too:
the engine lane parks until the landing callback posts the step, and a step
that feeds on the previous step's sampled token waits for that step to land
first. The tab and the native wgpu engine run the same engine code and
produce identical tokens (greedy and seeded sampling alike). Pass extra
inferlet input fields with `&input=%7B…%7D`.

The native wgpu server is the engine's test host rather than a production
backend: it runs the browser's code path over Vulkan, Metal or DX12 with a
poll thread standing in for the page's event loop, so what the browser suite
exercises is exercised natively too, and vice versa. Unifying the two paths
made native faster (`web/bench/README.md`, `native-0.8b-unified`).

Not available in a tab: MoE host-tier expert streaming (it maps the artifact
file), Python inferlet snapshots, telemetry, the program registry, outbound
HTTP from inferlets.
