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
import { readFile } from 'node:fs/promises';
import WebSocket from 'ws';
import { PieClient } from '@pie-project/client';
import { attachLanguages } from '@pie-project/server-web/languages';

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

/** Bytes from what `install*` accept: bytes, or a file/http URL to read. */
async function bytesOf(source) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  const url = source instanceof URL ? source : new URL(String(source), 'file://');
  if (url.protocol === 'file:') return new Uint8Array(await readFile(fileURLToPath(url)));
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return new Uint8Array(await response.arrayBuffer());
}

/** Under Node a language package that is installed need not be imported by hand: importing it here registers it. */
async function importInstalledLanguages() {
  for (const language of ['python', 'javascript']) {
    try {
      await import(`@pie-project/language-${language}`);
    } catch (error) {
      if (error?.code !== 'ERR_MODULE_NOT_FOUND') throw error;
    }
  }
}

/** A running pie engine in this process. */
export class Server {
  #handle;
  #clients = [];
  #detach = null;

  constructor(handle) {
    this.#handle = handle;
  }

  /**
   * Boot the engine. `{ model, config }` names the model (`Qwen/Qwen3.5-0.8B`,
   * a path, an artifact) and gives the rest of `pie serve`'s config as an
   * object; a bare config — TOML text, or an object of the file's shape with
   * `model.model` inside — is accepted as well. `server.port = 0` asks the OS
   * for a free port; `url` says which. Resolves once the engine is serving,
   * with every registered language component installed.
   */
  static async start(options) {
    let config = options;
    if (options && typeof options === 'object' && typeof options.model === 'string') {
      const { model, config: rest = {} } = options;
      if (typeof rest !== 'object') throw new Error('Server.start: with `model`, `config` is an object');
      config = { ...rest, model: { name: 'default', ...(rest.model ?? {}), model } };
    }
    const api = load();
    const handle = typeof config === 'string' ? await api.startToml(config) : await api.start(config);
    const server = new Server(handle);
    await importInstalledLanguages();
    server.#detach = await attachLanguages(server);
    return server;
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

  /** A language component (`python`, `javascript`) from bytes or a URL; a script inferlet in that language runs once this resolves. */
  async installLanguage(language, source) {
    const handle = this.#alive();
    return handle.installLanguage(language, Buffer.from(await bytesOf(source)));
  }

  /** An inferlet from its component bytes (or URL) and manifest text, replacing an installed version. Resolves to `name@version`. */
  async install(source, manifest) {
    const handle = this.#alive();
    return handle.install(Buffer.from(await bytesOf(source)), manifest);
  }

  /** A `PieClient` connected to this engine (over Node's `ws`, which can send the identity header). Closed by `shutdown()`. */
  async connect({ identity = 'default/node' } = {}) {
    const client = new PieClient(this.#alive().url, { WebSocket, identity });
    await client.connect();
    this.#clients.push(client);
    return client;
  }

  /** Close the clients from `connect()`, then stop the engine. Idempotent. */
  async shutdown() {
    this.#detach?.();
    this.#detach = null;
    for (const client of this.#clients.splice(0)) {
      try {
        await client.close();
      } catch {
        // already gone
      }
    }
    await this.#handle.shutdown();
  }

  #alive() {
    if (!this.running) throw new Error('the server is shut down');
    return this.#handle;
  }
}
