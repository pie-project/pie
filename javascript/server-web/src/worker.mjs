import init, * as bindings from "../pkg/pie_browser.js";
import * as platform from "../pkg/platform.mjs";

globalThis.__pieHost = {
  fetchRange(offset, len, request) {
    queueMicrotask(() => bindings.pie_range_failed(request, "no artifact is being served lazily"));
  },
};

let wasm = null;
let tick = null;
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
  if (tickScheduled && (timer === null || delayMs > 0)) return;
  tickScheduled = true;
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
  const run = async () => {
    tickScheduled = false;
    if (tickRunning) {
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

async function driven(promise) {
  scheduleTick(0);
  return await promise;
}

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

export const host = {
  async load(wasmUrl, logFilter) {
    if (wasm) throw new Error("pie: the wasm is already loaded");
    wasm = await init({ module_or_path: wasmUrl });
    platform.attach(wasm, () => scheduleTick(0));
    tick = WebAssembly.promising(wasm.pie_tick);
    bindings.pie_init(logFilter);
    return { worker: typeof WorkerGlobalScope !== "undefined" };
  },

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
    if (!file && dir && size <= CACHE_FILE_LIMIT) {
      cacheFill = fillCache(href, dir, cacheKey(url, size, etag), size).catch((e) => {
        console.warn("pie: background artifact cache fill failed:", e);
      });
    }
    return summary;
  },

  async awaitCache() {
    await cacheFill;
  },

  async installLanguage(language, wasmBytes) {
    return await driven(bindings.pie_install_language(language, wasmBytes));
  },
  async install(wasmBytes, manifestToml) {
    return await driven(bindings.pie_install_program(wasmBytes, manifestToml));
  },

  async memoryBytes() {
    return wasm.memory.buffer.byteLength;
  },

  async openSession() {
    return bindings.pie_open_session();
  },
  async closeSession(id) {
    bindings.pie_close_session(id);
  },
  async sendFrame(id, frame) {
    bindings.pie_send_frame(id, frame);
    scheduleTickAfterMessages();
  },
  async recvFrames(id, maxWaitMs, max) {
    return await driven(bindings.pie_recv_frames(id, maxWaitMs, max));
  },
};

export function transferables(value) {
  if (value instanceof Uint8Array) return [value.buffer];
  if (Array.isArray(value)) return value.filter((v) => v instanceof Uint8Array).map((v) => v.buffer);
  return [];
}

if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  self.onmessage = async ({ data }) => {
    const { id, op, args = [] } = data;
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
