// The page's half of pie's platform: what the wasm imports from "./platform.mjs".
//
// Three things live here, all of them things a wasm32 module cannot do for
// itself: switch stacks (JSPI), read a clock, and ask to be called again.
//
// Fibers — wasmtime-fiber's `custom` contract, as `crates/wasmtime-web` states it:
//   pie_fiber_init(top, entry, arg0): prepare the stack whose top is `top` so the
//     first switch into it calls `entry(arg0, top)`; when `entry` returns `ret`
//     the fiber performs one final switch(ret) and is never resumed.
//   pie_fiber_switch(top): symmetric — park the current execution in the slot
//     for `top`, resume whatever was parked there (or start the fiber).
// Every execution context — the page's tick and each fiber — is one promising
// export invocation. A context parks by returning a promise from the
// Suspending import; it resumes when that promise resolves. JSPI keeps each
// invocation's engine stack; Rust's shadow stack (`__stack_pointer`, exported
// by the build) is saved and restored here so the two stay paired.

let wasm = null; // instance exports, set by attach()
let fiberEntry = null; // promising wrapper of pie_fiber_entry
const slots = new Map(); // top -> parked continuation
let onWake = () => {};
// Diagnostics: `globalThis.__pieFiberTrace = true` logs every switch.
let current = "main"; // which context runs right now (for the trace)
const trace = (...args) => {
  if (globalThis.__pieFiberTrace) console.log("[fiber]", ...args);
};
// wasmtime-fiber keeps the resumer's result pointer at top-4; read it for the trace.
const slotAt = (top) => new DataView(wasm.memory.buffer).getUint32(top - 4, true);

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
  trace(`finish ${top} -> resume ${parked.name} sp=${parked.sp} slot=${slotAt(top)}`);
  current = parked.name;
  wasm.__stack_pointer.value = parked.sp;
  parked.resolve();
}

export function pie_fiber_init(top, entry, arg0) {
  top >>>= 0;
  if (slots.has(top)) trace(`init ${top} replaces a slot`, slots.get(top));
  trace(`init ${top} entry=${entry} arg0=${arg0} from ${current}`);
  slots.set(top, { init: { entry, arg0 } });
}

export const pie_fiber_switch = new WebAssembly.Suspending((top) => {
  top = top >>> 0;
  const parked = slots.get(top);
  if (!parked) throw new Error(`fiber switch to unknown stack ${top}`);
  const me = { sp: wasm.__stack_pointer.value, resolve: null, name: current };
  const resumed = new Promise((r) => (me.resolve = r));
  slots.set(top, me);
  if (parked.init) {
    const { entry, arg0 } = parked.init;
    trace(`switch ${top}: start fiber from ${current} sp=${me.sp}`);
    queueMicrotask(() => {
      // Two 4-byte slots at the top belong to wasmtime; start below them.
      current = `fiber@${top}`;
      wasm.__stack_pointer.value = (top - 16) >>> 0;
      fiberEntry(entry, arg0, top).then((ret) => finishFiber(ret >>> 0));
    });
  } else {
    trace(`switch ${top}: ${current} sp=${me.sp} -> ${parked.name} sp=${parked.sp} slot=${slotAt(top)}`);
    queueMicrotask(() => {
      trace(`  resume ${parked.name} at ${top}: slot=${slotAt(top)}`);
      current = parked.name;
      wasm.__stack_pointer.value = parked.sp;
      parked.resolve();
    });
  }
  return resumed;
});

// Clock: milliseconds, monotonic, from performance.now().
export function pie_now() {
  return performance.now();
}

// A waker fired outside a tick: the driver schedules another tick.
export function pie_wake() {
  if (globalThis.__pieTickTrace) console.log("[wake]");
  onWake();
}
