// pie in the tab, as the page sees it: `load`, `bootLazy`, `install`, `Session`.
//
// The runtime itself — the wasm, its executor loop, the sessions — lives in
// `worker.mjs`. By default that module runs in a dedicated Web Worker so the
// page stays responsive: the inferlet's in-browser Cranelift compile, the
// pipeline compiles on the first frames, prefill submits and every executor
// tick happen off the UI thread. Everything here is a thin proxy: one
// `postMessage` per call, the reply matched by id, `Uint8Array`s transferred
// rather than copied, progress callbacks delivered as messages. Pass
// `{ worker: false }` to `load` and the same module is imported here instead,
// for debugging with everything on one thread.

import { installInPageTransport } from "./client-shim.mjs";

/** Calls into `worker.mjs` running in a dedicated worker. */
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

/** Calls into `worker.mjs` imported on this thread. */
class LocalBackend {
  constructor(host) {
    this.host = host;
  }

  async call(op, args = [], { onProgress } = {}) {
    return await this.host[op](...args, onProgress);
  }
}

let backend = null;

function ready() {
  if (!backend) throw new Error("pie: call load() first");
  return backend;
}


/**
 * Load and instantiate the wasm. Call once. `options.worker` (default true)
 * hosts the runtime in a dedicated worker; `false` runs it on this thread.
 * Resolves with `{ worker }` saying which happened.
 */
export async function load(wasmUrl, logFilter, { worker = true, fresh = false } = {}) {
  if (backend && !fresh) throw new Error("pie: load() was already called");
  // `fresh` discards a previous runtime (its worker is terminated) and starts
  // another — for tests that need a clean module, since a runtime boots once.
  if (backend) {
    backend.terminate?.();
    backend = null;
  }
  const url = new URL(wasmUrl, location.href).href;
  if (worker) {
    const w = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module", name: "pie" });
    backend = new WorkerBackend(w);
  } else {
    const { host } = await import("./worker.mjs");
    backend = new LocalBackend(host);
  }
  const flags = { fiberTrace: !!globalThis.__pieFiberTrace, tickTrace: !!globalThis.__pieTickTrace };
  const info = await backend.call("load", [url, logFilter, flags]);
  installInPageTransport(backend);
  return info;
}

/** Executor state alone (tasks, next timer, parked waits), without touching the runtime. */
export async function executorState() {
  return await ready().call("executorState");
}

/** The scheduler's debug dump for engine 0 plus executor state. */
export async function debugDump() {
  return await ready().call("debugDump");
}

/** Install an inferlet component from bytes plus its Pie.toml manifest. The bytes are copied. */
export async function install(wasmBytes, manifestToml) {
  const copy = new Uint8Array(wasmBytes).slice();
  return await ready().call("install", [copy, manifestToml], { transfer: [copy.buffer] });
}

/** One executor tick through the promising export, for measurements. */
export async function tickOnce() {
  return await ready().call("tickOnce");
}

/** The wasm linear memory's current size in bytes. */
export async function memoryBytes() {
  return await ready().call("memoryBytes");
}

/**
 * Boot from `url` without holding the artifact in the tab: the loader's
 * reads are served as byte ranges, from the OPFS copy an earlier visit
 * left when there is one, else with `Range` requests (and the copy is then
 * made in the background; see `awaitCache`). Nothing else is
 * cached. `onProgress(served, total, from)` counts bytes served. Resolves
 * with the boot summary.
 */
export async function bootLazy(url, config, onProgress) {
  return await ready().call("bootLazy", [url, config, location.href], { onProgress });
}

/** Wait for the background artifact cache fill a lazy boot may have started. */
export async function awaitCache() {
  return await ready().call("awaitCache", []);
}

/** A session's id once the runtime has opened it (`{ id }` objects work too). */
async function sessionId(session) {
  if (session.id == null) session.id = await session.opened;
  return session.id;
}

/** A client session speaking ClientMessage / ServerMessage as JSON objects. */
export class Session {
  constructor() {
    this.id = null;
    this.opened = ready().call("openSession");
    this.corr = 0;
  }

  /** Deliver one message; rejects when the runtime refuses it. */
  async send(message) {
    await backend.call("send", [await sessionId(this), JSON.stringify(message)]);
  }

  async recv(maxWaitMs = 1000, max = 64) {
    if (this.held?.length) return this.held.splice(0, max);
    return JSON.parse(await backend.call("recv", [await sessionId(this), maxWaitMs, max]));
  }

  close() {
    return sessionId(this).then((id) => backend.call("closeSession", [id])).catch(() => {});
  }

  /**
   * Send a request and wait for its `response` by correlation id. Anything
   * else that arrives meanwhile (a process event of a process that finished
   * before its launch was answered, say) is kept for the next `recv`.
   */
  async request(message) {
    const corr_id = ++this.corr;
    await this.send({ ...message, corr_id });
    for (;;) {
      const batch = await this.recv(2000, 64);
      let response = null;
      for (const m of batch) {
        if (response === null && m.type === "response" && m.corr_id === corr_id) response = m;
        else (this.held ??= []).push(m);
      }
      if (response !== null) {
        if (!response.ok) throw new Error(`pie: ${JSON.stringify(response.result)}`);
        return response.result;
      }
    }
  }

  /**
   * Launch an installed program and stream its events. `onEvent` receives
   * every process_event; resolves when the process returns or errors.
   */
  async run(programName, inputJson, onEvent) {
    // `result` is the process id as a string.
    await this.request({
      type: "launch_process",
      inferlet: programName,
      input: inputJson,
      capture_outputs: true,
    });
    for (;;) {
      const batch = await this.recv(5000, 64);
      for (const m of batch) {
        if (m.type !== "process_event") continue;
        onEvent?.(m);
        if (m.event === "return" || m.event === "error") return m;
      }
    }
  }
}
