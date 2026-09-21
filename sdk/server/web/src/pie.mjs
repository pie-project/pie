import { PieClient } from "@pie-project/client";

import { socketClass } from "./transport.mjs";

export { PieClient };

const WASM_PATH = typeof __PIE_WASM__ === "undefined" ? "../pkg/pie_web_bg.wasm" : __PIE_WASM__;

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

export async function install(wasmBytes, manifestToml) {
  const copy = new Uint8Array(wasmBytes).slice();
  return await ready().call("install", [copy, manifestToml], { transfer: [copy.buffer] });
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
