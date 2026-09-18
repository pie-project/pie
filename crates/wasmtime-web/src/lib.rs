//! wasmtime inside a browser tab.
//!
//! `wasm32-unknown-unknown` is neither unix nor windows, so wasmtime's `std`
//! build routes through its custom platform layer and expects the embedder to
//! supply what an OS normally would. This crate is that embedder:
//!
//! * **virtual memory** — a tab has one linear memory and no protection, so a
//!   mapping is a page-aligned zeroed allocation, protection changes are
//!   no-ops, and copy-on-write memory images are declined;
//! * **thread-local slots** — two pointers wasmtime keeps per thread; a tab has
//!   one thread, so two statics;
//! * **fibers** — every guest call runs on its own stack so an async host
//!   function can park it. wasm32 has no stack-switching backend, so the two
//!   hooks are forwarded to the page, which implements them with JSPI
//!   (`WebAssembly.Suspending` imports and promising exports), saving and
//!   restoring Rust's shadow stack pointer per context. The Rust side of that
//!   contract is [`fiber_entry`]; the page's side lives in `web/site/platform.mjs`.
//!
//! Everything here is a link-time symbol, so the crate only has to be linked
//! into the final artifact once. On any other target it compiles to nothing.
//!
//! It is a crate of its own (not part of `wasi-web`) so that its `unsafe`
//! platform hooks stay outside the workspace's `forbid(unsafe_code)` policy;
//! `wasi-web` and the rest of the tree remain unsafe-free.
#![allow(unsafe_code)]

#[cfg(target_arch = "wasm32")]
mod platform;

#[cfg(target_arch = "wasm32")]
pub use platform::fiber_entry;

use std::sync::Arc;

/// Pulley artifacts are bytecode; nothing needs to be made executable.
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

/// Cranelift times every pass with `Instant::now()`, which has no clock on
/// wasm32-unknown-unknown. This profiler never asks.
struct NoTiming;

impl cranelift_codegen::timing::Profiler for NoTiming {
    fn start_pass(&self, _pass: cranelift_codegen::timing::Pass) -> Box<dyn std::any::Any> {
        Box::new(())
    }
}

/// Point a `Config` at the Pulley interpreter with the settings a
/// signal-less, virtual-memory-less host needs. Call before `Engine::new`.
///
/// The compile-relevant settings (target, component model) are what an
/// ahead-of-time compiler must match for `Component::deserialize` to accept
/// its output; the rest is runtime-side.
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
