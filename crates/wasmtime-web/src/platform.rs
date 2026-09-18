//! The link-time symbols wasmtime's custom platform layer imports.

use std::alloc::Layout;
use std::cell::RefCell;
use std::collections::HashMap;

// ---- thread-local slots ---------------------------------------------------

static mut TLS: [usize; 2] = [0; 2];

#[unsafe(no_mangle)]
extern "C" fn wasmtime_tls_get(slot: usize) -> *mut u8 {
    unsafe { TLS[slot] as *mut u8 }
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_tls_set(slot: usize, ptr: *mut u8) {
    unsafe { TLS[slot] = ptr as usize }
}

// ---- "virtual memory" over the heap ------------------------------------------

const PAGE: usize = 4096;

thread_local! {
    static MAPPINGS: RefCell<HashMap<usize, Layout>> = RefCell::new(HashMap::new());
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_mmap_new(size: usize, _prot: u32, ret: &mut *mut u8) -> i32 {
    let Ok(layout) = Layout::from_size_align(size.max(PAGE), PAGE) else {
        return 1;
    };
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return 2;
    }
    MAPPINGS.with(|m| m.borrow_mut().insert(ptr as usize, layout));
    *ret = ptr;
    0
}

/// "Replace with a blank mapping": the bytes go back to zero.
#[unsafe(no_mangle)]
extern "C" fn wasmtime_mmap_remap(addr: *mut u8, size: usize, _prot: u32) -> i32 {
    unsafe { core::ptr::write_bytes(addr, 0, size) };
    0
}

/// Only whole mappings are freed; a partial unmap has nothing to give back.
#[unsafe(no_mangle)]
extern "C" fn wasmtime_munmap(ptr: *mut u8, _size: usize) -> i32 {
    if let Some(layout) = MAPPINGS.with(|m| m.borrow_mut().remove(&(ptr as usize))) {
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
    0
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_mprotect(_ptr: *mut u8, _size: usize, _prot: u32) -> i32 {
    0
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_page_size() -> usize {
    PAGE
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_memory_image_new(_ptr: *const u8, _len: usize, ret: &mut *mut u8) -> i32 {
    *ret = core::ptr::null_mut();
    0
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_memory_image_map_at(_image: *mut u8, _addr: *mut u8, _len: usize) -> i32 {
    1
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_memory_image_free(_image: *mut u8) {}

// ---- fibers ----------------------------------------------------------------
//
// wasmtime-fiber's `custom` backend declares `wasmtime_fiber_init` and
// `wasmtime_fiber_switch` as plain C symbols. They resolve here and forward
// to imports the page provides from `./platform.mjs` (the module name is what
// wasm-bindgen turns into an ES import next to its generated glue).

#[link(wasm_import_module = "./platform.mjs")]
unsafe extern "C" {
    fn pie_fiber_init(top: *mut u8, entry: usize, arg0: *mut u8);
    fn pie_fiber_switch(top: *mut u8);
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_fiber_init(
    top: *mut u8,
    entry: extern "C" fn(*mut u8, *mut u8) -> *mut u8,
    arg0: *mut u8,
) {
    unsafe { pie_fiber_init(top, entry as usize, arg0) }
}

#[unsafe(no_mangle)]
extern "C" fn wasmtime_fiber_switch(top: *mut u8) {
    unsafe { pie_fiber_switch(top) }
}

/// The page starts a fiber by calling this promising export with the entry
/// function pointer and argument `wasmtime_fiber_init` handed it. The return
/// value is the stack top; the page uses it to perform the fiber's final
/// switch back to whoever resumed it.
#[unsafe(no_mangle)]
pub extern "C" fn fiber_entry(entry: usize, arg0: *mut u8, top: *mut u8) -> *mut u8 {
    let f: extern "C" fn(*mut u8, *mut u8) -> *mut u8 = unsafe { core::mem::transmute(entry) };
    f(arg0, top)
}
