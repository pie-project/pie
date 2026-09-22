# @pie-project/server

What `pie serve` boots (controller, gateway, worker, the built-in
inferlets), inside your process. Under Node.js it runs on the native GPU
engines; under a bundler, the same import is pie compiled for the browser,
on WebGPU. Clients connect as they would to a `pie serve`: with
`@pie-project/client` over the server's socket, or (Node) over the OpenAI /
Anthropic / Gemini HTTP routes.

```js
import { Server } from '@pie-project/server';
import '@pie-project/language-python';           // Python inferlets run; one line per language

const server = await Server.start({
  model: 'Qwen/Qwen3.5-0.8B',                    // Node: what `pie model import` fetched
  // model: '/models/qwen3.5-0.8b.wgpu.zt',      // browser: the artifact's URL, served with Range support
  config: { server: { port: 0 } },               // the rest of `pie serve`'s config
});

await server.install(inferletBytes, manifestToml);   // an inferlet of your own, from bytes or a URL
const client = await server.connect();               // a PieClient
const process = await client.launchProcess('text-completion', { prompt: 'Hello' });
console.log(await process.result());
await server.shutdown();
```

`Server.start` resolves once the model is loaded and the server is
listening, with every registered language component installed; `config` is
the file `pie serve --config` reads, minus the model, as an object. A bare
`Server.start(config)` with `model.model` inside, as an object or TOML text,
is accepted too. Under Node, `server.httpUrl` is where the HTTP routes live.
A handle garbage-collected without `shutdown()` still stops the engine.

The Python twin is the `pie-server` wheel (`python/server`).

## Installing

```bash
npm install @pie-project/server
```

Under Node the engine is a native addon, one optional dependency per
platform (`@pie-project/server-linux-x64`, `-linux-arm64`, `-darwin-arm64`,
`-win32-x64`), each carrying every engine that platform supports (Linux:
CUDA, Vulkan, wgpu; macOS: Metal, wgpu; Windows: CUDA, Vulkan); the
config's `engine.type` picks one at boot. Under a bundler (Vite, webpack 5)
the `browser` export condition resolves to `@pie-project/server-web`: the
runtime and the WebGPU engine as wasm, run in a Web Worker. The page must be
cross-origin isolated and the browser needs WebGPU and JSPI (Chrome 137+);
`javascript/server-web/README.md` has the details.

Language components (`@pie-project/language-python`,
`@pie-project/language-javascript`) register themselves on import; a
registration after `start()` lands on the running server too.

## Building

From source, the Node addon is built from this directory with the engine
features for your GPU (`cuda` by default, `metal` on macOS):

```bash
npm run build                      # release, default features
bash build.sh release --features metal
bash build.sh dev --no-default-features   # no engine: loads, refuses to serve
```

It lands beside `index.mjs` as `pie-server.<platform>-<arch>.node`, which
the loader prefers over the platform package. The browser bundle is
`javascript/server-web/build.sh`.
