import { PieClient, programFile } from "@pie-project/client";

import { socketClass } from "./transport.mjs";
import { attachLanguages } from "./languages.mjs";

export { PieClient, programFile };

const WASM_PATH = typeof __PIE_WASM__ === "undefined" ? "../pkg/pie_browser_bg.wasm" : __PIE_WASM__;

class WorkerBackend {
  terminate() {
    this.worker.terminate();
  }

  constructor(worker) {
    this.worker = worker;
    this.pending = new Map();
    this.next = 0;
    worker.onmessage = ({ data }) => {
      const call = this.pending.get(data.id);
      if (!call) return;
      if (data.progress) {
        call.onProgress?.(...data.progress);
        return;
      }
      this.pending.delete(data.id);
      if (data.ok) call.resolve(data.value);
      else call.reject(new Error(data.error));
    };
    worker.onerror = (e) => {
      console.error(`pie worker: ${e.message ?? e}`);
      const error = new Error(`pie worker: ${e.message ?? e}`);
      for (const call of this.pending.values()) call.reject(error);
      this.pending.clear();
    };
    worker.onmessageerror = (e) => console.error("pie worker: message could not be deserialised", e);
  }

  call(op, args = [], { transfer = [], onProgress } = {}) {
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, op, args }, transfer);
    });
  }
}

class LocalBackend {
  constructor(host) {
    this.host = host;
  }

  async call(op, args = [], { onProgress } = {}) {
    return await this.host[op](...args, onProgress);
  }
}

let backend = null;
let Socket = null;

function ready() {
  if (!backend) throw new Error("pie: call load() first");
  return backend;
}

export async function load({ worker = true, log, wasmUrl, fresh = false } = {}) {
  if (backend && !fresh) throw new Error("pie: load() was already called");
  if (backend) {
    backend.terminate?.();
    backend = null;
  }
  const url = new URL(wasmUrl ?? WASM_PATH, wasmUrl ? location.href : import.meta.url).href;
  if (worker) {
    const w = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module", name: "pie" });
    backend = new WorkerBackend(w);
  } else {
    const { host } = await import("./worker.mjs");
    backend = new LocalBackend(host);
  }
  const info = await backend.call("load", [url, log]);
  Socket = socketClass(backend);
  return info;
}

export function client() {
  ready();
  return new PieClient("pie://local", { WebSocket: Socket });
}

export async function connect() {
  const c = client();
  await c.connect();
  return c;
}

export async function install(bytes, file, version = null) {
  const copy = new Uint8Array(bytes).slice();
  return await ready().call("install", [copy, file, version], { transfer: [copy.buffer] });
}

/** Hand the runtime a language component (`"python"`, `"javascript"`)
 * from bytes; a script inferlet in that language can run once this has. */
export async function installLanguage(language, wasmBytes) {
  const copy = new Uint8Array(wasmBytes).slice();
  return await ready().call("installLanguage", [language, copy], { transfer: [copy.buffer] });
}

export async function memoryBytes() {
  return await ready().call("memoryBytes");
}

export async function bootLazy(url, config, onProgress) {
  return await ready().call("bootLazy", [url, config, location.href], { onProgress });
}

export async function awaitCache() {
  return await ready().call("awaitCache", []);
}

function isBytes(source) {
  return source instanceof Uint8Array || source instanceof ArrayBuffer;
}

async function bytesOf(source) {
  if (isBytes(source)) return new Uint8Array(source);
  const response = await fetch(source);
  if (!response.ok) throw new Error(`pie: ${source}: ${response.status} ${response.statusText}`);
  return new Uint8Array(await response.arrayBuffer());
}

/** One running pie in this tab, with the surface of the Node host. */
export class Server {
  #summary;
  #open = true;
  #detach = null;

  constructor(summary) {
    this.#summary = summary;
  }

  /**
   * Boot the runtime and the WebGPU engine. `model` is the URL of a
   * `.wgpu.zt` artifact served with Range support; `config` the boot config
   * (an object, or TOML text). Resolves once the weights are on the device,
   * with every registered language component installed.
   */
  static async start({ model, config = {}, onProgress, worker = true, log, wasmUrl } = {}) {
    if (typeof model !== "string") throw new Error("Server.start: `model` is the URL of the artifact to serve");
    if (!backend) await load({ worker, log, wasmUrl });
    const summary = await bootLazy(model, typeof config === "string" ? config : JSON.stringify(config), onProgress);
    const server = new Server(summary);
    server.#detach = await attachLanguages(server);
    return server;
  }

  /** What the boot found: model, sku, weight bytes, kv pages. */
  get summary() {
    return this.#summary;
  }

  /** The in-page address clients connect to. */
  get url() {
    return "pie://local";
  }

  /** True until `shutdown()`. */
  get running() {
    return this.#open && backend !== null;
  }

  /** A language component (`python`, `javascript`) from bytes or a URL. */
  async installLanguage(language, source) {
    this.#alive();
    return installLanguage(language, await bytesOf(source));
  }

  async install(source, file = null, version = null) {
    this.#alive();
    if (!file) {
      if (isBytes(source)) throw new Error("Server.install: bytes need `file`, the program's file name (`x.wasm`, `x.py`, `x.js`)");
      file = programFile(new URL(source, location.href));
    }
    return install(await bytesOf(source), file, version);
  }

  /** A `PieClient` connected in-page. */
  async connect() {
    this.#alive();
    return connect();
  }

  /** Bytes of wasm memory in use. */
  async memoryBytes() {
    this.#alive();
    return memoryBytes();
  }

  /** Resolves once the artifact is copied into the origin's private file system. */
  async awaitCache() {
    this.#alive();
    return awaitCache();
  }

  /** Stop the worker and release the runtime. Idempotent. */
  async shutdown() {
    if (!this.#open) return;
    this.#open = false;
    this.#detach?.();
    backend?.terminate?.();
    backend = null;
    Socket = null;
  }

  #alive() {
    if (!this.running) throw new Error("the server is shut down");
  }
}
