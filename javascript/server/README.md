# @pie-project/server

What `pie serve` boots (controller, gateway, worker, the built-in
inferlets), inside your Node.js process on the native GPU engines. Clients
connect as they would to a `pie serve`: over WebSocket with
`@pie-project/client`, or over the OpenAI / Anthropic / Gemini HTTP routes.

```js
import { Server } from '@pie-project/server';

const server = await Server.start({
  server: { port: 0 },                         // 0: a free port; see server.url
  model: { model: 'Qwen/Qwen3.5-0.8B' },       // what `pie model import` fetched
});

const client = await server.connect();          // a PieClient over ws
const process = await client.launchProcess('text-completion', { prompt: 'Hello' });
console.log(await process.result());

const reply = await fetch(server.httpUrl + '/v1/chat/completions', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ messages: [{ role: 'user', content: 'Hi' }] }),
});
console.log((await reply.json()).choices[0].message.content);

await server.shutdown();
```

`Server.start` takes the config `pie serve --config` reads (what `pie config
init` writes), as an object or as TOML text, and resolves once the model is
loaded and the listener is bound. A handle garbage-collected without
`shutdown()` still stops the engine, on a background thread.

The Python twin is the `pie-server` wheel (`python/server`); the browser
build of pie is `javascript/browser`.

## Installing

```bash
npm install @pie-project/server
```

The addon comes as an optional dependency per platform
(`@pie-project/server-linux-x64`, `-linux-arm64`, `-darwin-arm64`,
`-win32-x64`), each carrying every engine that platform supports (Linux:
CUDA, Vulkan, wgpu; macOS: Metal, wgpu; Windows: CUDA, Vulkan); the config's
`engine.type` picks one at boot.

## Building

From source, the addon is built from this directory with the engine features
for your GPU (`cuda` by default, `metal` on macOS):

```bash
npm run build                      # release, default features
bash build.sh release --features metal
bash build.sh dev --no-default-features   # no engine: loads, refuses to serve
```

It lands beside `index.mjs` as `pie-server.<platform>-<arch>.node`, which
the loader prefers over the platform package.
