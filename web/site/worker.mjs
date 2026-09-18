// pie's host: the wasm, its executor loop, sessions and the streamed model
// bytes. This module runs in a dedicated worker by default — `pie.mjs` posts
// requests here and the page thread never touches the wasm — or on the page
// itself when `pie.load(…, { worker: false })` imports it there for debugging.
//
// The wasm exposes a handful of functions (see crates/web/src/page.rs); `host`
// wraps them in the shape an application wants and owns the one loop the
// host needs: whenever a task inside the wasm becomes ready, call `pie_tick`
// through a promising export so any fiber switch inside can suspend it.
//
// Worker protocol (page → worker, one message per call):
//   { id, op, args }                 call `host[op](...args)`
// and back:
//   { id, progress: [got, total, from] }   zero or more, for the streamed ops
//   { id, ok: true, value }          `Uint8Array`s in `value` are transferred
//   { id, ok: false, error }         the error's message
// `{ op: "load" }` must come first; calls are answered in any order, so every
// message carries its id.

// platform.mjs is imported from inside pkg/ so the wasm's own imports and this
// module share one instance (one fiber table, one wake hook).
import init, * as bindings from "./pkg/pie_web.js";
import * as platform from "./pkg/platform.mjs";

// The page's half of a lazy boot: the wasm's `__pieHost.fetchRange` import
// (see crates/web/src/page.rs) asks for a byte range of the artifact and is
// answered through `pie_range_ready`/`pie_range_failed`. `bootLazy` installs
// the server for the artifact it boots; outside one, every range is refused.
globalThis.__pieHost = {
  fetchRange(offset, len, request) {
    queueMicrotask(() => bindings.pie_range_failed(request, "no artifact is being served lazily"));
  },
};

let wasm = null;
let tick = null;
/** A background copy of the artifact into OPFS, started by a lazy boot. */
let cacheFill = null;
const CACHE_FILE_LIMIT = 1900 * 1048576;

async function fillCache(href, dir, key, size) {
  const res = await fetch(href);
  if (!res.ok) throw new Error(`${href}: ${res.status}`);
  const handle = await dir.getFileHandle(key, { create: true });
  const writable = await handle.createWritable();
  const reader = res.body.getReader();
  let got = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      await writable.write(value);
      got += value.length;
    }
    if (got !== size) throw new Error(`cache fill got ${got} of ${size} bytes`);
    await writable.close();
    console.log(`pie: artifact cached for the next visit (${(size / 1048576).toFixed(0)} MiB)`);
  } catch (e) {
    try { await writable.abort(); } catch { /* already gone */ }
    try { await dir.removeEntry(key); } catch { /* nothing to remove */ }
    throw e;
  }
}
let tickScheduled = false;
let tickRunning = false;
let timer = null;

function scheduleTick(delayMs = 0) {
  // A tick already queued to run now covers this request; one waiting on a
  // timer does not — a wake must not sit behind the executor's own backstop.
  if (tickScheduled && (timer === null || delayMs > 0)) return;
  tickScheduled = true;
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
  const run = async () => {
    tickScheduled = false;
    if (tickRunning) {
      // A tick is suspended in a fiber switch; it will loop when it returns.
      tickScheduled = true;
      return;
    }
    tickRunning = true;
    let next;
    try {
      next = await tick();
    } finally {
      tickRunning = false;
    }
    if (globalThis.__pieTickTrace) console.log(`[tick] next=${next}`);
    if (tickScheduled) {
      tickScheduled = false;
      scheduleTick(0);
    } else if (next >= 0) {
      scheduleTick(next);
    }
  };
  if (delayMs <= 0) queueMicrotask(run);
  else timer = setTimeout(() => { timer = null; run(); }, delayMs);
}

// A tick once every message already queued has been handled. A burst of
// sends (the SDK launching several processes at once) is one task on the
// page but one message each here; ticking between them would launch the
// first alone and split what the scheduler batches into one prefill. A
// MessageChannel task shares the posted-message task source, so it runs
// after the messages that were queued before it.
let afterMessages = null;
function scheduleTickAfterMessages() {
  if (afterMessages === null) {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      afterMessages.armed = false;
      scheduleTick(0);
    };
    afterMessages = { port: channel.port2, armed: false };
  }
  if (afterMessages.armed) return;
  afterMessages.armed = true;
  afterMessages.port.postMessage(null);
}

/** Start a promise-returning export and make sure the executor runs it. */
async function driven(promise) {
  scheduleTick(0);
  return await promise;
}

// ---- the origin's private file system as an artifact cache ----------------

/** `HEAD` the artifact for its size and ETag; the cache key is all three. */
async function describe(url) {
  const head = await fetch(url, { method: "HEAD" }).catch(() => null);
  const size = Number(head?.headers.get("content-length") ?? 0);
  const etag = head?.headers.get("etag") ?? "";
  return { size, etag };
}

function cacheKey(name, size, etag) {
  return `${name}|${size}|${etag}`.replace(/[^a-zA-Z0-9._-]/g, "_");
}

async function cacheDir() {
  try {
    return (await navigator.storage?.getDirectory?.()) ?? null;
  } catch {
    return null;
  }
}

/**
 * Everything the host does, as async methods; each is one `op` of the
 * worker protocol. `onProgress(got, total, from)` is the trailing argument of
 * `bootLazy`.
 */
export const host = {
  /** Load and instantiate the wasm. Call once. `flags` mirrors the page's `__pie*Trace` globals. */
  async load(wasmUrl, logFilter, flags = {}) {
    if (wasm) throw new Error("pie: the wasm is already loaded");
    if (flags.fiberTrace) globalThis.__pieFiberTrace = true;
    if (flags.tickTrace) globalThis.__pieTickTrace = true;
    // `init` resolves to the raw exports (pie_tick, pie_fiber_entry,
    // __stack_pointer); everything else goes through the bindgen wrappers.
    wasm = await init({ module_or_path: wasmUrl });
    platform.attach(wasm, () => scheduleTick(0));
    tick = WebAssembly.promising(wasm.pie_tick);
    bindings.pie_init(logFilter);
    return { worker: typeof WorkerGlobalScope !== "undefined" };
  },

  /**
   * Boot from `url` without holding the artifact in the tab: its size comes
   * from a `HEAD`, and the loader's reads are served as byte ranges — from
   * the copy an earlier visit left in the origin's private file system when
   * there is one (same cache key), else with `Range` requests (the server
   * must answer 206), in which case the copy is made in the background
   * after the boot (`fillCache`). Progress is bytes served so far (which may
   * pass `total`: a window read twice is served twice). Resolves with the
   * boot summary.
   */
  async bootLazy(url, config, base, onProgress) {
    cacheFill = null;
    const href = new URL(url, base ?? location.href).href;
    const { size, etag } = await describe(href);
    if (!size) throw new Error(`${url}: the server sent no content-length; a lazy boot needs the size up front`);
    let file = null;
    const dir = await cacheDir();
    if (dir) {
      try {
        const cached = await (await dir.getFileHandle(cacheKey(url, size, etag))).getFile();
        if (cached.size === size) file = cached;
      } catch {
        // not cached: ranges come from the network
      }
    }
    const from = file ? "cache" : "network";
    let served = 0;
    let requests = 0;
    const range = async (offset, len) => {
      requests += 1;
      let bytes;
      if (file) {
        bytes = new Uint8Array(await file.slice(offset, offset + len).arrayBuffer());
      } else {
        const res = await fetch(href, { headers: { range: `bytes=${offset}-${offset + len - 1}` } });
        if (res.status !== 206) {
          throw new Error(`${url}: the server answered ${res.status} to a range request; a lazy boot needs ranges (206)`);
        }
        bytes = new Uint8Array(await res.arrayBuffer());
      }
      if (bytes.length !== len) throw new Error(`${url}: range ${offset}+${len} came back as ${bytes.length} bytes`);
      served += len;
      onProgress?.(served, size, from);
      return bytes;
    };
    const refuse = globalThis.__pieHost.fetchRange;
    globalThis.__pieHost.fetchRange = (offset, len, request) => {
      range(offset, len).then(
        (bytes) => bindings.pie_range_ready(request, bytes),
        (e) => bindings.pie_range_failed(request, String(e?.message ?? e)),
      );
    };
    const t0 = performance.now();
    let summary;
    try {
      summary = JSON.parse(await driven(bindings.pie_boot_lazy(config, url.split("/").pop(), size)));
    } finally {
      globalThis.__pieHost.fetchRange = refuse;
      console.log(`pie: lazy boot served ${requests} range request(s), ${(served / 1048576).toFixed(0)} MiB from ${from}, in ${(performance.now() - t0).toFixed(0)} ms`);
    }
    // The next visit can serve its ranges from the origin's private file
    // system: fill it in the background now, when the artifact is small
    // enough for one OPFS file (about 2 GB) and came from the network.
    if (!file && dir && size <= CACHE_FILE_LIMIT) {
      cacheFill = fillCache(href, dir, cacheKey(url, size, etag), size).catch((e) => {
        console.warn("pie: background artifact cache fill failed:", e);
      });
    }
    return summary;
  },

  /** Resolves once a background cache fill started by `bootLazy` has ended. */
  async awaitCache() {
    await cacheFill;
  },

  /** Install an inferlet component from bytes plus its Pie.toml manifest. */
  async install(wasmBytes, manifestToml) {
    return await driven(bindings.pie_install_program(wasmBytes, manifestToml));
  },

  /** Executor state alone (tasks, next timer, parked waits): no request to the runtime, so a stall stays as it is. */
  async executorState() {
    return bindings.pie_executor_state();
  },

  /** The scheduler's debug dump for engine 0 plus executor state. */
  async debugDump() {
    const state = bindings.pie_executor_state();
    return `${state}\n${await driven(bindings.pie_debug(0))}`;
  },

  /** One executor tick through the promising export, for measurements. */
  async tickOnce() {
    return await tick();
  },

  /** The wasm linear memory's current size in bytes. */
  async memoryBytes() {
    return wasm.memory.buffer.byteLength;
  },

  // Sessions: the client protocol as JSON strings (`send`/`recv`) or as the
  // MessagePack frames `pie serve`'s WebSocket carries (`sendFrame`/`recvFrames`).
  async openSession() {
    return bindings.pie_open_session();
  },
  async closeSession(id) {
    bindings.pie_close_session(id);
  },
  async send(id, messageJson) {
    bindings.pie_send(id, messageJson);
    scheduleTickAfterMessages();
  },
  /** Resolves with the JSON text of an array of ServerMessages (possibly empty). */
  async recv(id, maxWaitMs, max) {
    return await driven(bindings.pie_recv(id, maxWaitMs, max));
  },
  async sendFrame(id, frame) {
    bindings.pie_send_frame(id, frame);
    scheduleTickAfterMessages();
  },
  /** Resolves with an array of `Uint8Array`, each owning its buffer. */
  async recvFrames(id, maxWaitMs, max) {
    return await driven(bindings.pie_recv_frames(id, maxWaitMs, max));
  },
};

/** The buffers a result can hand over instead of copying. */
export function transferables(value) {
  if (value instanceof Uint8Array) return [value.buffer];
  if (Array.isArray(value)) return value.filter((v) => v instanceof Uint8Array).map((v) => v.buffer);
  return [];
}

// ---- the worker's message loop -------------------------------------------

if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  self.onmessage = async ({ data }) => {
    const { id, op, args = [] } = data;
    // Progress is posted at most every 50 ms plus the final chunk, so a
    // 64 KiB-chunked download does not cost thousands of messages.
    let lastProgress = 0;
    const progress = (got, total, from) => {
      const now = performance.now();
      if (got < total && now - lastProgress < 50) return;
      lastProgress = now;
      self.postMessage({ id, progress: [got, total, from] });
    };
    try {
      const fn = host[op];
      if (typeof fn !== "function") throw new Error(`pie: unknown op ${op}`);
      const value = await fn.call(host, ...args, progress);
      self.postMessage({ id, ok: true, value }, transferables(value));
    } catch (e) {
      self.postMessage({ id, ok: false, error: String(e?.message ?? e) });
    }
  };
}
