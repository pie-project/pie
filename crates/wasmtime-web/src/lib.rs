// wasm32 has no stack-switching backend: the fiber hooks are forwarded to the
// page's JSPI glue (javascript/browser/src/platform.mjs). wasm-bindgen's own JSPI
// cannot host them: it evacuates a suspended activation's shadow-stack frames,
// and wasmtime hands the fiber raw pointers into the resumer's frame.
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
mod platform;

#[cfg(target_arch = "wasm32")]
pub use platform::fiber_entry;

mod wasi;

pub use wasi::*;

use std::sync::Arc;

struct NopCodeMemory;

impl wasmtime::CustomCodeMemory for NopCodeMemory {
    fn required_alignment(&self) -> usize {
        1
    }
    fn publish_executable(&self, _ptr: *const u8, _len: usize) -> wasmtime::Result<()> {
        Ok(())
    }
    fn unpublish_executable(&self, _ptr: *const u8, _len: usize) -> wasmtime::Result<()> {
        Ok(())
    }
}

struct NoTiming;

impl cranelift_codegen::timing::Profiler for NoTiming {
    fn start_pass(&self, _pass: cranelift_codegen::timing::Pass) -> Box<dyn std::any::Any> {
        Box::new(())
    }
}

pub fn configure(config: &mut wasmtime::Config) -> wasmtime::Result<()> {
    cranelift_codegen::timing::set_thread_profiler(Box::new(NoTiming));
    config.target("pulley32")?;
    config.wasm_component_model(true);
    config.signals_based_traps(false);
    config.memory_reservation(0);
    config.memory_reservation_for_growth(1 << 20);
    config.memory_guard_size(0);
    config.memory_init_cow(false);
    config.with_custom_code_memory(Some(Arc::new(NopCodeMemory)));
    Ok(())
}
