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
