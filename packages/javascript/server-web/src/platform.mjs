// Fibers (wasmtime-fiber's `custom` contract, see crates/wasmtime-web):
// every context, the page's tick and each fiber, is one promising export
// invocation; a park returns a promise from the Suspending import. JSPI keeps
// the engine stack; Rust's shadow stack (`__stack_pointer`) is saved and
// restored here so the two stay paired.
let wasm = null;
let fiberEntry = null;
const slots = new Map();
let onWake = () => {};

export function attach(exports, wakeHook) {
  wasm = exports;
  fiberEntry = WebAssembly.promising(exports.pie_fiber_entry);
  onWake = wakeHook;
}

function finishFiber(top) {
  const parked = slots.get(top);
  slots.delete(top);
  if (!parked || parked.init) {
    console.error(`[fiber] ${top} finished with no parked resumer`, parked);
    return;
  }
  wasm.__stack_pointer.value = parked.sp;
  parked.resolve();
}

export function pie_fiber_init(top, entry, arg0) {
  slots.set(top >>> 0, { init: { entry, arg0 } });
}

export const pie_fiber_switch = new WebAssembly.Suspending((top) => {
  top = top >>> 0;
  const parked = slots.get(top);
  if (!parked) throw new Error(`fiber switch to unknown stack ${top}`);
  const me = { sp: wasm.__stack_pointer.value, resolve: null };
  const resumed = new Promise((r) => (me.resolve = r));
  slots.set(top, me);
  if (parked.init) {
    const { entry, arg0 } = parked.init;
    queueMicrotask(() => {
      wasm.__stack_pointer.value = (top - 16) >>> 0;
      fiberEntry(entry, arg0, top).then((ret) => finishFiber(ret >>> 0));
    });
  } else {
    queueMicrotask(() => {
      wasm.__stack_pointer.value = parked.sp;
      parked.resolve();
    });
  }
  return resumed;
});

export function pie_now() {
  return performance.now();
}

export function pie_wake() {
  onWake();
}

// Slot files (crates/web-std/src/opfs.rs). Rust owns the numbers in `slotFiles`.
// A throw must not cross into wasm: every import here answers with a status.
const slotFiles = new Map();
let nextSlotFile = 1;

// Wasm addresses can pass 2^31, and growth detaches views: a fresh one per call.
const bytesAt = (ptr, len) => new Uint8Array(wasm.memory.buffer, ptr >>> 0, len >>> 0);
const textAt = (ptr, len) => new TextDecoder().decode(bytesAt(ptr, len));

// `work` on the directory with dead tabs' files gone. Tabs take turns: a removal
// holds its file's lock for a moment, which refuses a handle to a tab that is
// opening that file. A live tab's handle refuses the removal, so its file stays.
async function sweptDir(path, work) {
  if (typeof FileSystemSyncAccessHandle === "undefined") {
    throw new Error("no synchronous OPFS handles on this thread");
  }
  return await navigator.locks.request(`pie-opfs:${path}`, async () => {
    let dir = await navigator.storage.getDirectory();
    for (const part of path.split("/").filter(Boolean)) {
      dir = await dir.getDirectoryHandle(part, { create: true });
    }
    for (const name of await Array.fromAsync(dir.keys())) {
      await dir.removeEntry(name).catch(() => {});
    }
    return await work(dir);
  });
}

function answer(request, work, what) {
  work.then(
    (value) => wasm.pie_opfs_answer(request, value),
    (e) => {
      console.warn(`pie: no OPFS disk tier: ${what}: ${e?.message ?? e}`);
      wasm.pie_opfs_answer(request, 0);
    },
  );
}

export function pie_opfs_open(dirPtr, dirLen, namePtr, nameLen, len, request) {
  const path = textAt(dirPtr, dirLen);
  const name = `${crypto.randomUUID()}.${textAt(namePtr, nameLen)}`;
  const open = sweptDir(path, async (dir) => {
    const file = await dir.getFileHandle(name, { create: true });
    const handle = await file.createSyncAccessHandle();
    try {
      handle.truncate(len);
      // Past its limit a truncate does not throw and leaves the file empty.
      if (handle.getSize() !== len) throw new Error(`the file took ${handle.getSize()} of ${len} bytes`);
    } catch (e) {
      handle.close();
      await dir.removeEntry(name).catch(() => {});
      throw e;
    }
    const id = nextSlotFile++;
    slotFiles.set(id, { handle, dir, name });
    return id;
  });
  answer(request, open, `opening ${name}`);
}

export function pie_opfs_free_space(dirPtr, dirLen, request) {
  const path = textAt(dirPtr, dirLen);
  // Swept first, so a dead tab's file does not count against this one.
  const free = sweptDir(path, async () => {
    const { quota = 0, usage = 0 } = await navigator.storage.estimate();
    return Math.max(quota - usage, 0);
  });
  answer(request, free, "reading the free space");
}

function slotIo(what, io) {
  try {
    return io();
  } catch (e) {
    console.error(`pie: OPFS ${what}: ${e?.message ?? e}`);
    return -1;
  }
}

export function pie_opfs_read(id, at, ptr, len) {
  return slotIo("read", () => slotFiles.get(id).handle.read(bytesAt(ptr, len), { at }));
}

export function pie_opfs_write(id, at, ptr, len) {
  return slotIo("write", () => slotFiles.get(id).handle.write(bytesAt(ptr, len), { at }));
}

export function pie_opfs_flush(id) {
  return slotIo("flush", () => (slotFiles.get(id).handle.flush(), 0));
}

export function pie_opfs_close(id) {
  const { handle, dir, name } = slotFiles.get(id);
  slotFiles.delete(id);
  slotIo("close", () => handle.close());
  dir.removeEntry(name).catch(() => {});
}
