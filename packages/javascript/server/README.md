# @pie-project/server

Embeds `pie serve` inside Node.js (native GPU backends) or the browser (WebGPU via bundler).

```bash
npm install @pie-project/server
```

```js
import { Server } from '@pie-project/server';
import '@pie-project/language-python';

const server = await Server.start({
  model: 'Qwen/Qwen3.5-0.8B',                    // Node: imported model; browser: URL to .wgpu.zt
  config: { server: { listen: '127.0.0.1:0' } },
});

const client = await server.connect();
const proc = await client.launchProcess('text-completion', { prompt: 'Hello' });
console.log(await proc.result());
await server.shutdown();
```

Under Node.js, platform packages supply CUDA, Metal, Vulkan, and WebGPU engines, and `server.httpUrl` exposes the OpenAI, Anthropic, and Gemini HTTP routes. Under a bundler (Vite, webpack 5), the `browser` export resolves to `@pie-project/server-web` (requires a cross-origin isolated page with WebGPU and JSPI, Chrome 137+).

## Build from source

```bash
npm run build                               # default GPU backend
bash build.sh release --features metal      # explicit feature
```
