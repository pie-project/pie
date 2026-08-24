//! Launches a "fire": one hand-written instantiation of a `.cuh` whose host
//! side this driver owns, distinct from a point's claim body (compiled beside
//! its declaration).

use kernels_cuda::jit::{Ctx, Launch};
use kernels_cuda::{ArgValue, Refusal};

/// Launch one instantiation of the file `file` names.
///
/// # Errors
///
/// Whatever the JIT declined with — compile, load or launch. Per-symbol JIT
/// has no fallback, so there is nothing to try instead; what there is, is a
/// caller. The `panic!` that stood here made a refusal this process's exit,
/// and a driver's next line is somebody else's request.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
pub fn fire(
    file: &'static str,
    instantiation: &'static str,
    launch: Launch,
    values: &[ArgValue],
    stream: *mut std::ffi::c_void,
) -> Result<(), Refusal> {
    // SAFETY: `values` are live allocations; `stream` stays valid across the launch.
    unsafe { Ctx::on(stream).fire(kernels::Fire::at(file, instantiation).apply(launch), values) }
}
