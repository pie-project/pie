// `@pie-project/server`: what `pie serve` boots, inside a Node process.
//
//   import { Server } from '@pie-project/server';
//   const server = await Server.start({ server: { port: 0 }, model: { model: 'Qwen/Qwen3.5-0.8B' } });
//   const client = await server.connect();
//   ...
//   await server.shutdown();
//
// The engine is a napi-rs addon: `pie-server.<platform>-<arch>.node` beside
// this file when built from source (`build.sh`), else the platform package
// `@pie-project/server-<platform>-<arch>` npm installed as an optional
// dependency; this wrapper loads it and adds `connect()`.

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync } from 'node:fs';
import WebSocket from 'ws';
import { PieClient } from '@pie-project/client';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const target = `${process.platform}-${process.arch}`;

/** The addon this platform loads: a local build, else the platform package; null when neither is present. */
export function addonPath() {
  const local = join(here, `pie-server.${target}.node`);
  if (existsSync(local)) return local;
  try {
    return require.resolve(`@pie-project/server-${target}`);
  } catch {
    return null;
  }
}

let native = null;
function load() {
  if (native) return native;
  const path = addonPath();
  if (!path) {
    throw new Error(
      `@pie-project/server: no native addon for ${target}: npm did not install @pie-project/server-${target} ` +
        '(no build for this platform, or optional dependencies were skipped), and none is built beside ' +
        `${here} (\`npm run build\` builds one; it needs the Rust toolchain)`,
    );
  }
  native = require(path);
  return native;
}

/** A running pie engine in this process. */
export class Server {
  #handle;
  #clients = [];

  constructor(handle) {
    this.#handle = handle;
  }

  /**
   * Boot the engine from the TOML text `pie serve --config` reads, or an
   * object of the same shape (`{server: {port: 0}, model: {…}}`).
   * `server.port = 0` asks the OS for a free port; `url` says which.
   * Resolves once the engine is serving — minutes for a large model.
   */
  static async start(config) {
    const api = load();
    const handle = typeof config === 'string' ? await api.startToml(config) : await api.start(config);
    return new Server(handle);
  }

  /** `ws://host:port` the engine listens on. */
  get url() {
    return this.#handle.url;
  }

  /** `http://host:port` — the OpenAI / Anthropic / Gemini routes live here. */
  get httpUrl() {
    return this.url.replace(/^ws(s?):\/\//, 'http$1://');
  }

  /** True until `shutdown()` resolves. */
  get running() {
    return this.#handle.running;
  }

  /** A `PieClient` connected to this engine (over Node's `ws`, which can send the identity header). Closed by `shutdown()`. */
  async connect({ identity = 'default/node' } = {}) {
    if (!this.running) throw new Error('the server is shut down');
    const client = new PieClient(this.url, { WebSocket, identity });
    await client.connect();
    this.#clients.push(client);
    return client;
  }

  /** Close the clients from `connect()`, then stop the engine. Idempotent. */
  async shutdown() {
    for (const client of this.#clients.splice(0)) {
      try {
        await client.close();
      } catch {
        // already gone
      }
    }
    await this.#handle.shutdown();
  }
}
