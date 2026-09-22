# @pie-project/server-web

The browser half of `@pie-project/server`: the same runtime, WebGPU engine
and inferlet sandbox as `pie serve`, compiled to `wasm32-unknown-unknown`.
No server-side inference: the page fetches a `.wgpu.zt` artifact, the
weights land in WebGPU buffers, and inferlets run under wasmtime's Pulley
interpreter inside the same module. A bundler reaches it through
`@pie-project/server`'s `browser` export condition; nothing imports it by
name.

## Use

The package ships the wasm and a self-contained bundle (`dist/pie.mjs`,
`dist/worker.mjs`, `dist/pie_browser_bg.wasm`); the client
(`@pie-project/client`) is bundled in and re-exported. The worker and the
wasm are fetched relative to the module, so serve them from your own origin
(a bundler that understands `new URL(…, import.meta.url)` — Vite, webpack 5 —
does this by itself; without one, copy `dist/` next to your page).

```js
import { Server } from "@pie-project/server";      // resolves here under a bundler
import "@pie-project/language-python";

const server = await Server.start({ model: "/models/qwen3.5-0.8b.wgpu.zt" });   // served with Range support
const program = await server.install(inferletBytes, manifestToml);
const client = await server.connect();              // a PieClient, as for `pie serve`
const proc = await client.launchProcess(program, { prompt: "The capital of France is", max_tokens: 16 });
for (;;) {
  const { event, value } = await proc.recv();
  if (event === "return" || event === "error") break;
}
await server.shutdown();
```

The page-level functions behind the class (`load`, `bootLazy`, `install`,
`installLanguage`, `connect`, `awaitCache`, `memoryBytes`) are exported as
well; `tests/browser` drives them directly.


- The page must be cross-origin isolated (`Cross-Origin-Opener-Policy:
  same-origin`, `Cross-Origin-Embedder-Policy: require-corp`), and the
  browser needs WebGPU and JSPI: Chrome 137+, Firefox 153+ (Firefox on
  Linux has no WebGPU, so the engine cannot boot there).
- `bootLazy` never holds the artifact in the tab: the loader's reads are
  served as 8 MiB-aligned `Range` requests, and after a boot from the
  network the artifact is copied into the origin's private file system
  (files up to about 2 GB) so the next visit reads it from there;
  `awaitCache()` waits for that copy. Peak wasm memory is a few times the
  largest tensor: 0.8B boots in 285 MiB, 9B in 4 bits in 1.5 GiB.
- The engine cannot ask WebGPU how much device memory there is and assumes
  8 GiB; a large model with the default pools is refused ("raise the device
  weight budget") — pass `device_memory_mb = 20480` (or a smaller
  `max_total_pages`) in the boot config. `BootConfig` in
  `crates/browser/src/boot.rs` lists the keys.
- Rust, JavaScript and Python inferlets run. Not available in a tab: MoE
  host-tier expert streaming, Python inferlet snapshots, telemetry, and
  outbound HTTP or files from inferlets (the WASI interfaces link, a call
  fails).
- `index.d.ts` carries the types.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked   # must match Cargo.lock
./javascript/server-web/build.sh          # release; `min` for the 23 MB shipping build, `dev` for debug
```

`build.sh` builds `crates/browser`, runs wasm-bindgen into `pkg/`, and bundles
`src/` with esbuild into `dist/` (`npm run bundle`). The package's own
dependencies are installed on the first run. Tests, benchmarks and CI gates
live under `tests/browser` (see its README).

## How it fits together

| piece | what it does |
|---|---|
| `crates/browser` | The wasm-bindgen host: boot, install, sessions speaking the MessagePack frames `pie serve`'s WebSocket carries, `pie_tick`. |
| `crates/wasmtime-web` | wasmtime in a tab: heap-backed "virtual memory", TLS slots, fibers implemented by the page's JSPI glue (`src/platform.mjs`), and the WASI host for the inferlet's imports (cli, io, clocks, random; 0.2 linked by hand, the 0.3 clocks from `crates/inferlet/wit`). The 0.2 filesystem and http interfaces a JavaScript or Python guest runtime imports regardless are stubbed per component, so they link and only a call fails. |
| `crates/web-std` | The executor the page ticks, timers over `performance.now()`, crossbeam-shaped channels, and green threads on fibers. The runtime's scheduler and engine lanes run on those unchanged; `crates/runtime`'s `rt` facade names it on wasm32. |
| `crates/engine-wgpu` | The same engine on both hosts, written for the browser: no GPU wait blocks, every step lands by callback, guests run from the landing. What differs by host sits in `device/host.rs`. |
| `src/worker.mjs` | The host: wasm init, the tick loop, sessions, the artifact served by byte range. Runs in a Web Worker so the page stays responsive through the inferlet's in-browser compile and every forward pass; `load({ worker: false })` runs it on the page for debugging. |
| `src/pie.mjs` | The page-side API: the `Server` class over `load`, `bootLazy`, `install`, `connect`. |
| `src/languages.mjs` | The language registry `@pie-project/server/languages` re-exports: what a language package calls on import. |
| `src/platform.mjs` | Fibers (JSPI), the clock, the wake hook the wasm imports. |
| `src/transport.mjs` | The client's in-page socket over the runtime's frame API. |

The native wgpu server is the engine's test host rather than a production
backend: it runs the browser's code path over Vulkan, Metal or DX12 with a
poll thread standing in for the page's event loop, so what the browser suite
exercises is exercised natively too, and the two produce identical tokens.
