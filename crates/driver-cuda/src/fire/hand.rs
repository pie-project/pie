//! Launches a "fire": a driver-declared `Root` instantiation, distinct from a
//! routine (compiled beside its `.cuh`).

use kernels_cuda::ArgValue;
use kernels_cuda::jit::{Ctx, Launch};

/// Launch one instantiation of the file `file` names. Panics if compile,
/// load or launch fails — per-symbol JIT has no fallback.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
pub fn fire(
    file: &'static str,
    instantiation: &'static str,
    launch: Launch,
    values: &[ArgValue],
    stream: *mut std::ffi::c_void,
) {
    // SAFETY: `values` are live allocations; `stream` stays valid across the launch.
    // `Ctx::launch` BECAME `Ctx::fire`, taking the four facts as one `Fire`.
    let fired = unsafe {
        Ctx::on(stream).fire(
            kernels::Fire::at(file, instantiation).apply(launch),
            values,
        )
    };
    if let Err(why) = fired {
        panic!("{instantiation}: {why}");
    }
}
