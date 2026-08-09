//! Shared setup for the GPU test binaries.
//!
//! Lives in `tests/common/` rather than in one of the test files because
//! `mod gpu_smoke;` would compile that file's `#[test]` functions into every
//! binary that included it -- running them several times under confusing
//! names, and, worse, running them *concurrently with graph captures*.

use driver_cuda::cuda::{COMPILED_MAJOR, Device};
use std::sync::{Mutex, MutexGuard};

/// Serialises all GPU work in a test binary.
///
/// This is broader than it looks like it needs to be, and deliberately so.
/// `Allocator::begin_capture` uses `cudaStreamCaptureModeGlobal`, matching the
/// C++, and that mode makes any potentially-unsafe CUDA call in **any thread**
/// fail for as long as a capture is open. So it is not enough to serialise
/// captures against each other: an unrelated `cudaMemsetAsync` on another test
/// thread will fail too, and it will fail intermittently depending on how the
/// harness interleaves them.
///
/// Locking every GPU test against every other one costs nothing here -- they
/// take under a second in total -- and turns a flaky suite into a
/// deterministic one.
static GPU: Mutex<()> = Mutex::new(());

/// Take the GPU lock. The lock guards an ordering, not data, so a poisoned
/// lock is still perfectly usable.
pub fn gpu_guard() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bind device 0, or return `None` if there is nothing this build can bind.
///
/// Two things are skipped, and they are different:
///
///   * **No device at all** -- the GPU-less case, including CI.
///   * **A device this build cannot legally drive**, because the loaded
///     runtime's major version is not the one the crate was compiled against.
///     CI builds *both* ABIs on purpose, so on any real GPU box exactly one of
///     them will always be the wrong one. Refusing to run is the correct
///     outcome there and is reported by [`Device::bind`] itself; turning it
///     into a red suite would say the code is broken when the truth is that
///     this binary is not the one for this machine.
///
/// Everything else -- a device present, the ABI right, and the bind still
/// failing -- is a **failure**, because at that point something really is
/// wrong and skipping would report green for a build that cannot run.
#[allow(dead_code)] // not every test binary probes the device
pub fn device_or_skip(what: &str) -> Option<Device> {
    match Device::count() {
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
