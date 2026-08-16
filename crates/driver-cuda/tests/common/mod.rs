//! Shared setup for the GPU test binaries.
//!
//! Lives in `tests/common/`, not a test file: `mod gpu_smoke;` would compile
//! its `#[test]` functions into every binary that includes it, running them
//! repeatedly and, worse, concurrently with graph captures.

use driver_cuda::device::{COMPILED_MAJOR, Device};
use std::sync::{Mutex, MutexGuard};

/// Serialises all GPU work in a test binary.
///
/// Broader than it looks like it needs, deliberately: `Allocator::begin_capture`
/// uses `cudaStreamCaptureModeGlobal`, which fails any potentially-unsafe CUDA
/// call on **any thread** while a capture is open -- so serialising captures
/// against each other isn't enough; an unrelated `cudaMemsetAsync` elsewhere
/// would fail too, intermittently. Locking every GPU test against every
/// other costs nothing here (under a second total) and makes a flaky suite
/// deterministic.
static GPU: Mutex<()> = Mutex::new(());

/// Take the GPU lock. The lock guards an ordering, not data, so a poisoned
/// lock is still perfectly usable.
pub fn gpu_guard() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run `f`, answering `None` if it panics, without printing a crash report.
///
/// The `Err` arms below cannot report the case this helper exists for. `cudarc`
/// is `fallback-dynamic-loading` and nothing here has a `DT_NEEDED` on
/// `libcuda`: the first call `dlopen`s it and PANICS through
/// `cudarc::panic_no_lib_found` when no candidate name resolves. A box with a
/// driver but no device answers `Ok(0)`; a box with NO CUDA AT ALL -- which is
/// what `ubuntu-latest` is, and what the doc below means by "including CI" --
/// never reaches a `Result` at all. So the skip that CI depends on was
/// unreachable from CI. Catching is what makes it reachable, and only the
/// first call needs it: once `count()` answers, the library is loaded.
fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

/// Bind device 0, or return `None` if there is nothing this build can bind.
///
/// Two things are skipped: **no device at all** (the GPU-less case,
/// including CI), and **a device this build cannot legally drive** -- the
/// loaded runtime's major version differs from the crate's, which CI builds
/// *both* ABIs for on purpose, so exactly one is always wrong on a real box.
/// [`Device::bind`] reports that refusal; a failure here would say the code
/// is broken when really this binary isn't the one for this machine.
///
/// Everything else -- device present, ABI right, bind still failing -- is a
/// **failure**: skipping then would report green for a build that cannot run.
#[allow(dead_code)] // not every test binary probes the device
pub fn device_or_skip(what: &str) -> Option<Device> {
    let Some(counted) = quietly(Device::count) else {
        eprintln!("skipping {what}: no CUDA driver library on this machine");
        return None;
    };
    match counted {
        Ok(0) => {
            eprintln!("skipping {what}: no CUDA device");
            None
        }
        Err(e) => {
            eprintln!("skipping {what}: {e}");
            None
        }
        Ok(n) => {
            if Device::runtime_major_matches() != Ok(true) {
                eprintln!(
                    "skipping {what}: built for the CUDA {}.x ABI, which is not \
                     what this machine loads",
                    COMPILED_MAJOR
                );
                return None;
            }
            match Device::bind(0) {
                Ok(d) => Some(d),
                Err(e) => panic!("{n} CUDA device(s) present but bind failed: {e}"),
            }
        }
    }
}
