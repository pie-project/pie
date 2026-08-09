//! A cubin, loaded: the module, the function, and the width to launch it at.
//!
//! # What `block_threads` is for, and why it is a power of two
//!
//! The generated fused kernels reduce with the standard halving tree —
//! `for (stride = blockDim.x / 2; stride > 0; stride >>= 1)`. A tree written
//! that way is correct only when `blockDim.x` is a power of two; at 768
//! threads the first halving is 384, the second 192, and the lanes at the top
//! of the block are folded twice while the ones in the middle are never folded
//! at all. The answer is wrong and nothing reports it.
//!
//! So the launch width is not a tuning constant. It is
//! `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` — what the compiled function's
//! register pressure actually permits — rounded DOWN to a power of two. Down,
//! because rounding up is a launch failure, and the attribute is a ceiling.
//!
//! # Why the module is owned rather than leaked
//!
//! `CUmodule` is a process-wide resource and a program that is registered,
//! bound, and closed a thousand times over a serving day leaks a thousand of
//! them. [`Module`] unloads in `Drop`, which is the whole reason the cubin is
//! wrapped at all instead of being loaded inline where it is compiled.

use cudarc::driver::sys as dr;

use crate::error::{Error, Result};

/// The launch width to fall back to when the attribute cannot be read.
///
/// The C++ initialises `block_threads` to this and overwrites it only on a
/// successful query, so a driver that refuses the attribute launches at 256
/// rather than not launching. Reproduced: an attribute query is not a
/// precondition for running a kernel, and 256 is a power of two and within
/// every device's limit.
const DEFAULT_BLOCK_THREADS: u32 = 256;

/// The warp width, and the floor a rounded-down launch width may not go below.
const WARP: u32 = 32;

/// The maximum threads any CUDA block may hold.
const MAX_BLOCK_THREADS: u32 = 1024;

/// A loaded cubin and one entry point inside it.
///
/// Not `Clone`: the `Drop` unloads the module, so a copy would unload it twice.
/// Sharing is what `Arc` is for, and a caller that needs it says so.
#[derive(Debug)]
pub struct Module {
    module: dr::CUmodule,
    function: dr::CUfunction,
    block_threads: u32,
    entry_name: String,
}

// SAFETY: `CUmodule` and `CUfunction` are context-scoped handles, not
// thread-scoped ones. The driver API is documented as thread-safe for module
// and function handles, and this crate binds exactly one primary context per
// device (`Device::bind`), so a handle observed on one thread names the same
// module on every other. What is NOT safe is unloading concurrently with a
// launch, and that is `&self`/`Drop` rather than a `Send` question.
unsafe impl Send for Module {}
// SAFETY: as above -- every method here is a read of an immutable handle.
unsafe impl Sync for Module {}

impl Module {
    /// Load `cubin` and resolve `entry_name` inside it.
    ///
    /// # Errors
    ///
    /// If the image is not loadable on this device, or carries no such entry
    /// point. Both are the same class of fault — a cubin that does not match
    /// the source that was supposed to produce it — which is why a failure
    /// here invalidates the disk cache entry it came from rather than being
    /// reported to the caller as a program error.
    pub fn load(cubin: &[u8], entry_name: &str) -> Result<Self> {
        if cubin.is_empty() {
            return Err(Error::invalid("cuModuleLoadData", "the cubin is empty"));
        }
        let Ok(entry_c) = std::ffi::CString::new(entry_name) else {
            return Err(Error::invalid(
                "cuModuleGetFunction",
                format!("entry name '{entry_name}' contains a NUL"),
            ));
        };

        let mut module: dr::CUmodule = std::ptr::null_mut();
        // SAFETY: `cubin` is a live byte image and `module` a live
        // out-parameter. `cuModuleLoadData` reads the image's own header for
        // its length rather than taking one, which is why the slice's length
        // is not passed -- and why an empty slice is refused above instead of
        // being handed to the driver to read past.
        let code = unsafe { dr::cuModuleLoadData(&raw mut module, cubin.as_ptr().cast()) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Error::Driver {
                call: "cuModuleLoadData",
                code,
            });
        }

        let mut function: dr::CUfunction = std::ptr::null_mut();
        // SAFETY: `module` loaded successfully above; `entry_c` outlives the call.
        let code = unsafe { dr::cuModuleGetFunction(&raw mut function, module, entry_c.as_ptr()) };
        if code != dr::CUresult::CUDA_SUCCESS {
            // The module is loaded and the entry is missing, so nothing will
            // ever use it. Unload before returning: the alternative is a leak
            // per failed lookup, and a stale disk cache produces exactly this
            // failure in a loop.
            //
            // SAFETY: `module` is loaded and no function from it is in flight.
            unsafe { dr::cuModuleUnload(module) };
            return Err(Error::Driver {
                call: "cuModuleGetFunction",
                code,
            });
        }

        Ok(Self {
            module,
            function,
            block_threads: launch_width(function),
            entry_name: entry_name.to_string(),
        })
    }

    /// The entry point handle, for `cuLaunchKernel`.
    #[must_use]
    pub const fn function(&self) -> dr::CUfunction {
        self.function
    }

    /// The width to launch this function at: a power of two, within what its
    /// register pressure permits.
    #[must_use]
    pub const fn block_threads(&self) -> u32 {
        self.block_threads
    }

    /// The entry point's name, as the host emitted it.
    #[must_use]
    pub fn entry_name(&self) -> &str {
        &self.entry_name
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: loaded in `load`, dropped once. The return code is
            // deliberately ignored: a `Drop` has nowhere to report it, and the
            // only documented failures are a dead context or a double unload,
            // both of which mean the process is already past saving.
            unsafe { dr::cuModuleUnload(self.module) };
        }
    }
}

/// The register-limited launch width, rounded down to a power of two.
fn launch_width(function: dr::CUfunction) -> u32 {
    let mut max_threads = 0i32;
    // SAFETY: `max_threads` is a live out-parameter and `function` was just
    // resolved out of a loaded module.
    let code = unsafe {
        dr::cuFuncGetAttribute(
            &raw mut max_threads,
            dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            function,
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return DEFAULT_BLOCK_THREADS;
    }
    round_down_to_power_of_two(max_threads)
}

/// `max_threads` rounded down to a power of two, within `[WARP, 1024]`.
///
/// Split out from the query so the arithmetic is testable without a GPU — it
/// is the part that can be wrong in a way no test on the device would catch,
/// because a slightly-too-small width still runs and still produces plausible
/// numbers.
fn round_down_to_power_of_two(max_threads: i32) -> u32 {
    // Below one warp there is nothing to round to and the attribute is not
    // believable; the C++ leaves `block_threads` at its default in that case
    // rather than launching a partial warp.
    let Ok(max_threads) = u32::try_from(max_threads) else {
        return DEFAULT_BLOCK_THREADS;
    };
    if max_threads < WARP {
        return DEFAULT_BLOCK_THREADS;
    }
    let mut width = WARP;
    while width * 2 <= max_threads && width < MAX_BLOCK_THREADS {
        width *= 2;
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated reductions halve `blockDim.x`, so a width that is not a
    /// power of two folds some lanes twice and others never. Every answer must
    /// be one.
    #[test]
    fn every_width_is_a_power_of_two() {
        for max in [32, 33, 63, 64, 100, 128, 512, 768, 1024, 2048] {
            let width = round_down_to_power_of_two(max);
            assert!(
                width.is_power_of_two(),
                "{max} rounded to {width}, which is not a power of two"
            );
        }
    }

    /// Down, never up: the attribute is a ceiling the register allocator set,
    /// and exceeding it is a launch failure rather than a slow kernel.
    #[test]
    fn the_width_never_exceeds_what_the_function_permits() {
        assert_eq!(round_down_to_power_of_two(768), 512);
        assert_eq!(round_down_to_power_of_two(1023), 512);
        assert_eq!(round_down_to_power_of_two(100), 64);
        assert_eq!(round_down_to_power_of_two(63), 32);
    }

    /// An exact power of two is already the answer and must not be halved.
    #[test]
    fn an_exact_power_of_two_is_kept() {
        assert_eq!(round_down_to_power_of_two(1024), 1024);
        assert_eq!(round_down_to_power_of_two(256), 256);
        assert_eq!(round_down_to_power_of_two(32), 32);
    }

    /// A block may not exceed 1024 threads whatever the attribute claims.
    #[test]
    fn the_width_is_capped_at_the_hardware_block_limit() {
        assert_eq!(round_down_to_power_of_two(4096), MAX_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(i32::MAX), MAX_BLOCK_THREADS);
    }

    /// An unbelievable attribute falls back rather than launching a partial
    /// warp -- and a negative one, which `cuFuncGetAttribute` should never
    /// produce, must not become a huge `u32`.
    #[test]
    fn an_unusable_attribute_falls_back_to_the_default_width() {
        assert_eq!(round_down_to_power_of_two(31), DEFAULT_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(0), DEFAULT_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(-1), DEFAULT_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(i32::MIN), DEFAULT_BLOCK_THREADS);
    }
}
