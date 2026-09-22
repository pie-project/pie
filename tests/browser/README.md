# The browser suite

Everything that runs `@pie-project/browser` (`javascript/browser`) in a real browser.
Pages are served from the repository root, so they import the package's
`dist/` bundle by relative path.

## Set up

```bash
PIE_HOME=~/.pie-browser pie model import Qwen/Qwen3.5-0.8B     # a wgpu build of pie writes <slug>.<sku>.wgpu.zt
ln -s ~/.pie-browser/models/Qwen--Qwen3.5-0.8B tests/browser/models
npm install --prefix tests/browser/tools playwright@1.63.0
npx --prefix tests/browser/tools playwright install chromium
./javascript/browser/build.sh
./tests/browser/tools/inferlets.sh      # the inferlets matrix.json names, beside the pages
```

The JavaScript and Python twins are scripts, not builds: `inferlets.sh`
copies their source beside the page, and the language component from
`~/.pie/languages` into `tests/browser/languages/`, which
the page installs from bytes before the first script in that language.

## Run

```bash
./tests/browser/run.sh                                                    # the whole GPU suite
node tests/browser/tools/headless.mjs . "tests/browser/index.html?auto&max_tokens=8"   # one page
node tests/browser/tools/serve.mjs . 8765                                 # then open /tests/browser/ in Chrome 137+
```

`headless.mjs` opens a page in Playwright's Chromium with WebGPU and JSPI
on, streams its console, and exits with the status the page reports.
`run.sh` runs, in order: every kernel through Chrome's Tint
(`crates/kernels-wgpu/tools/wgsl-check.html`), the build, and the pages:

| page | what it checks |
|---|---|
| `index.html?auto` | boot, install, a completion; `&prompt=`, `&max_tokens=`, `&config=key%20%3D%20value,…`, `&log=` (console filter), `&worker=0` (runtime on the page thread), `&input=%7B…%7D` (extra inferlet input) |
| `corners.html` | host corner cases: a missing artifact, launch before boot, a second boot, the UI thread during compile and a 200-token run, re-install, garbage bytes, closed and malformed connections, many connections, memory over 50 runs |
| `transport.html` | the cost of a client ping, and a launch without an engine (`engine = false`) |
| `client.html` | the JavaScript client against the tab |
| `bench.html?what=corners` | the inferlet library corner cases shared with the native benchmark (`tests/browser/bench.mjs`) |
| `matrix.html?only=…` | the inferlets in `matrix.json` (device-carried loops and masks, watermarks, grammar-constrained decoding, token healing, beam search, the epilogue-op probe, the JavaScript twin); `tools/matrix.sh` runs them here and against a native `pie serve` and compares |
| `lifecycle.html` (`tools/lifecycle.mjs`) | a first visit from the network, a reload from the private file system, generation in a hidden tab |

`PIE_WEB_MODELS="models4b/… models9b/…"` makes `run.sh` boot more artifacts
once. Samplers at a real temperature may pick differently on the two hosts
(Tint and naga emit floating point differently, and a Gumbel-max draw over
two near-tied candidates goes either way); at temperature 0.05 every matrix
entry answers the same on both.

## Benchmarks

`tests/browser/bench.mjs` is one benchmark for both hosts: `bench.html?what=bench`
runs it in the tab and `node scripts/bench/browser/native.mjs ws://127.0.0.1:28417/v1/ws`
against a native wgpu `pie serve`; `node scripts/bench/browser/compare.mjs` prints
the table from `scripts/bench/browser/results/`.

## Gates without a GPU

CI's `browser:` job runs `./scripts/browser-check.sh` (clippy with `-D warnings`
for every crate the tab links on `wasm32-unknown-unknown`, the wgpu feature
natively, `node --check` over the scripts), every kernel variant through Tint
on SwiftShader, and `index.html`/`transport.html` in headless Chromium on
SwiftShader with a random-weight miniature of Qwen3.5 (`scripts/tiny_qwen35.py`).
The native engines run the same miniature on Mesa's lavapipe in the
`wgpu:`/`vulkan:` jobs (`scripts/ci/lavapipe.sh`: the library corner cases,
`sort-probe`, one compat API call, a Python and a JavaScript inferlet).
