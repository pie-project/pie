//! Driver-fire phase breakdown probes.
//!
//! Decomposes `fire.execute.driver_fire_us` (the largest single bucket
//! in pie's hot path — ~95% of fire wall) into the phases it actually
//! consists of:
//!
//! ```text
//! DriverCudaProbes
//! ├── ipc_submit_us            Rust:  hand request to IPC ring
//! ├── gpu_wait_us              Rust:  futex park on response slot
//! ├── ipc_recv_us              Rust:  unwrap response from ring
//! ├── wire_parse_us            C++:   view parsing, span construction
//! ├── plan_us                  C++:   FlashInfer plan + sample plan
//! ├── h2d_us                   C++:   cudaMemcpyAsync of inputs + setup
//! ├── kernel_launch_us         C++:   prepare hook + forward dispatch
//! ├── sync_us                  C++:   cudaStreamSynchronize wall
//! │                                   (includes GPU compute)
//! └── response_build_us        C++:   build_token_only_dense into IPC slab
//! ```
//!
//! Gated by the `profile-driver-cuda` Cargo feature.
//!
//! ## Plumbing
//!
//! Two paths, because half the probes live on the Rust side and half
//! on the C++ side.
//!
//! **Rust-side IPC phases** (`ipc_submit`, `gpu_wait`, `ipc_recv`)
//! are recorded by `InProcChannel` / `InProcPollingChannel`'s
//! `submit_sync_for_state`. Those impls don't have access to the
//! scheduler's `SchedulerStats` arc — they're a layer below — so we
//! stash per-call timings in a thread-local `Cell<u64>` and let the
//! scheduler thread drain them after each fire via `take_*_us()`.
//! This works because `fire_batch_sync` is always called on the
//! scheduler OS thread (no thread crossings).
//!
//! **C++-side host phases** (`wire_parse`, `plan`, `h2d`,
//! `kernel_launch`, `sync`, `response_build`) are filled by the C++
//! `handle_fire_batch` via `std::chrono::steady_clock` and returned
//! through the `ForwardResponse` payload. The Rust side reads them
//! into the same `DriverCudaProbes` atomics.
//!
//! With the feature off, both paths short-circuit at the source:
//! `record_*` is empty, the response-payload read fills with zeros.

use std::sync::atomic::AtomicU64;

#[derive(Debug, Default)]
pub struct DriverCudaProbes {
    /// Rust-side: time to hand the request to the IPC ring buffer
    /// (enqueue in `pending` + push to `inbox` + `notify_one`).
    pub ipc_submit_us: AtomicU64,
    /// Rust-side: time parked on the response slot via futex / spin.
    /// Equals C++-side `sync_us` plus all the C++ host work + IPC
    /// reply transport. Most of the fire lives here.
    pub gpu_wait_us: AtomicU64,
    /// Rust-side: time from `submit_sync` returning to `fire_batch_sync`
    /// returning (response payload unwrap + ResponsePayload match).
    pub ipc_recv_us: AtomicU64,

    /// C++-side: PieForwardRequestView span construction + spec
    /// expansion (when applicable).
    pub wire_parse_us: AtomicU64,
    /// C++-side: `build_sample_plan` + `plan_attention_flashinfer_decode`
    /// (or its `plan_static_nonsplit_decode` fast path).
    pub plan_us: AtomicU64,
    /// C++-side: `pi.tokens.copy_from_host(...)` x6 + sample-input
    /// uploads + any other H2D enqueues before the forward.
    pub h2d_us: AtomicU64,
    /// C++-side: `invoke_prepare` + `run_forward_dispatch` host time —
    /// the time spent issuing the work to the GPU. Does NOT include
    /// GPU compute (which happens during `sync_us`).
    pub kernel_launch_us: AtomicU64,
    /// C++-side: wall time of `cudaStreamSynchronize(cublas.stream())`
    /// after the sampling kernel + d2h memcpy. This is effectively
    /// GPU compute time (all prior async work drains here).
    pub sync_us: AtomicU64,
    /// C++-side: `response_builder.build_token_only_dense` (writes the
    /// sampled tokens into the IPC response slab).
    pub response_build_us: AtomicU64,
}

// =============================================================================
// Thread-local IPC phase timings
// =============================================================================
//
// Filled by `InProcChannel` / `InProcPollingChannel` inside
// `submit_sync_for_state`. Drained by the scheduler thread after each
// fire (inside the `probe_fire!(driver_fire_us)` scope) via
// `take_*_us()`. Only valid because the scheduler thread is the sole
// caller of `submit_sync` — the channel impl is layered below the
// scheduler.

#[cfg(feature = "profile-driver-cuda")]
mod tls {
    use std::cell::Cell;

    thread_local! {
        pub(super) static IPC_SUBMIT_LAST_US: Cell<u64> = const { Cell::new(0) };
        pub(super) static GPU_WAIT_LAST_US:   Cell<u64> = const { Cell::new(0) };
        pub(super) static IPC_RECV_LAST_US:   Cell<u64> = const { Cell::new(0) };
    }
}

#[cfg(feature = "profile-driver-cuda")]
#[inline]
pub fn record_ipc_submit(d: std::time::Duration) {
    tls::IPC_SUBMIT_LAST_US.with(|c| c.set(d.as_micros() as u64));
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[inline]
pub fn record_ipc_submit(_d: std::time::Duration) {}

#[cfg(feature = "profile-driver-cuda")]
#[inline]
pub fn record_gpu_wait(d: std::time::Duration) {
    tls::GPU_WAIT_LAST_US.with(|c| c.set(d.as_micros() as u64));
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[inline]
pub fn record_gpu_wait(_d: std::time::Duration) {}

#[cfg(feature = "profile-driver-cuda")]
#[inline]
pub fn take_ipc_submit_us() -> u64 {
    tls::IPC_SUBMIT_LAST_US.with(|c| c.replace(0))
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[inline]
pub fn take_ipc_submit_us() -> u64 {
    0
}

#[cfg(feature = "profile-driver-cuda")]
#[inline]
pub fn take_gpu_wait_us() -> u64 {
    tls::GPU_WAIT_LAST_US.with(|c| c.replace(0))
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[inline]
pub fn take_gpu_wait_us() -> u64 {
    0
}

#[cfg(feature = "profile-driver-cuda")]
#[inline]
pub fn record_ipc_recv(d: std::time::Duration) {
    tls::IPC_RECV_LAST_US.with(|c| c.set(d.as_micros() as u64));
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[inline]
pub fn record_ipc_recv(_d: std::time::Duration) {}

#[cfg(feature = "profile-driver-cuda")]
#[inline]
pub fn take_ipc_recv_us() -> u64 {
    tls::IPC_RECV_LAST_US.with(|c| c.replace(0))
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[inline]
pub fn take_ipc_recv_us() -> u64 {
    0
}

// =============================================================================
// Macros
// =============================================================================
//
// `probe_driver_cuda!` matches the shape of `probe_fire!`: wrap body,
// accumulate elapsed micros into a target atomic, return body's
// value. With the feature off the macro is a body-only no-op.

#[cfg(feature = "profile-driver-cuda")]
#[macro_export]
macro_rules! probe_driver_cuda {
    ($target:expr, $body:expr) => {{
        let __probe_start = ::std::time::Instant::now();
        let __probe_result = $body;
        $target.fetch_add(
            __probe_start.elapsed().as_micros() as u64,
            ::std::sync::atomic::Ordering::Relaxed,
        );
        __probe_result
    }};
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[macro_export]
macro_rules! probe_driver_cuda {
    ($target:expr, $body:expr) => {{
        let _ = &$target;
        $body
    }};
}

#[cfg(feature = "profile-driver-cuda")]
#[macro_export]
macro_rules! probe_driver_cuda_record {
    ($target:expr, $duration:expr) => {{
        $target.fetch_add(
            $duration.as_micros() as u64,
            ::std::sync::atomic::Ordering::Relaxed,
        );
    }};
}
#[cfg(not(feature = "profile-driver-cuda"))]
#[macro_export]
macro_rules! probe_driver_cuda_record {
    ($target:expr, $duration:expr) => {{
        let _ = (&$target, &$duration);
    }};
}
