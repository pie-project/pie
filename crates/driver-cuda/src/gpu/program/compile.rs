//! NVRTC: one self-contained translation unit in, one cubin out.
//!
//! # Why there is no include path
//!
//! `nvrtcCreateProgram` is called with zero headers and zero include names,
//! and that is not an omission. The host emitter (`tensor-compiler`) splices
//! `ptir_m1_runtime_prologue.cuh`, `fused_block0.cuh`, the entry name,
//! `fused_block1.cuh`, `fused_block2.cuh` and the generated body into ONE
//! string before the driver ever sees it, and the runtime headers say so about
//! themselves — *"these headers are compiled with no include path at all"*.
//! So a `#include` appearing in an emitted source is a bug in the emitter, and
//! the right place to find that out is a compile error rather than a search
//! path that silently resolves it against whatever CUDA toolkit is installed.
//!
//! It is also what makes this crate's toolkit-free build survive: there is no
//! header to find at build time, and `libnvrtc.so` is resolved by `dlopen` on
//! first call like every other CUDA symbol here.
//!
//! # Why the real architecture and not a virtual one
//!
//! `--gpu-architecture=sm_XY`, from the live device, so NVRTC returns a cubin.
//! `compute_XY` would return PTX, and PTX is JIT-compiled a second time by the
//! driver — which is outside CUDA minor-version compatibility and would force
//! every host to carry a driver at least as new as the toolkit pie was built
//! against. The C++ shell says the same thing at `module_cache.hpp:196-202`
//! and this reproduces it.
//!
//! # The three float flags are a contract, not a tuning knob
//!
//! `--fmad=false --prec-div=true --prec-sqrt=true`. The channel plane's whole
//! claim is that a replay lands on the same token, and
//! `driver-pipeline`'s tolerance contract holds magnitudes to one ulp *and
//! argmax indices to zero*. Contracting a multiply-add, or taking a fast
//! reciprocal, moves a lane by more than that and turns a tie into a different
//! winner. These flags are what make the CPU interpreter and the GPU agree.

use std::ffi::{CStr, CString};

use cudarc::nvrtc::sys as nvrtc;

/// Whether a compile failure is worth remembering.
///
/// The distinction earns its keep at the negative cache: a program NVRTC
/// rejected will be rejected identically forever, and re-running a
/// multi-hundred-millisecond compile per fire to learn that again is the
/// difference between a slow model and an unusable one. A machine that ran out
/// of memory mid-compile, on the other hand, must be retried — caching that
/// answer would make one bad minute permanent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// The source is wrong. It will be wrong next time. Remember it.
    Deterministic,
    /// The machine could not, this time. Do not remember it.
    Retryable,
}

/// A compile that did not produce a cubin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileError {
    /// Whether to remember this answer.
    pub kind: FailureKind,
    /// What NVRTC said — its log for a compilation failure, its error string
    /// otherwise.
    pub message: String,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CompileError {}

/// The NVRTC library's version, as `(major, minor)`.
///
/// Part of the compile-cache key: two NVRTC versions compile one source to
/// different machine code, and a cubin that outlives a toolkit upgrade is a
/// crash rather than a wrong answer.
///
/// # Errors
///
/// If `libnvrtc` cannot be loaded or the query fails.
pub fn version() -> std::result::Result<(i32, i32), CompileError> {
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: both out-parameters are live `i32`s for the duration of the call.
    let status = unsafe { nvrtc::nvrtcVersion(&raw mut major, &raw mut minor) };
    if status == nvrtc::nvrtcResult::NVRTC_SUCCESS {
        Ok((major, minor))
    } else {
        Err(CompileError {
            kind: FailureKind::Retryable,
            message: format!("cannot query NVRTC version: {}", describe(status)),
        })
    }
}

/// Compile one emitted region to a cubin for `architecture`.
///
/// `architecture` is the `sm_XY` string from [`arch_flag`]. `source` is the
/// whole translation unit as the host emitted it; nothing is prepended.
///
/// # Errors
///
/// [`FailureKind::Deterministic`] carrying NVRTC's log when the source is
/// rejected, [`FailureKind::Retryable`] for every other failure.
pub fn compile(source: &str, architecture: &str) -> std::result::Result<Vec<u8>, CompileError> {
    let retryable = |message: String| CompileError {
        kind: FailureKind::Retryable,
        message,
    };

    // NVRTC takes C strings, and a NUL inside the source would truncate the
    // translation unit at that byte and compile a prefix -- which can succeed,
    // and produce a kernel that is missing its tail. Refusing here is the only
    // place the whole string is still in hand.
    let Ok(source_c) = CString::new(source) else {
        return Err(CompileError {
            kind: FailureKind::Deterministic,
            message: "emitted source contains an interior NUL byte".into(),
        });
    };
    let name = c"ptir_fused_region.cu";

    let mut program: nvrtc::nvrtcProgram = std::ptr::null_mut();
    // SAFETY: `program` is a live out-parameter; the two strings outlive the
    // call; zero headers means the two header arrays are legitimately null.
    let status = unsafe {
        nvrtc::nvrtcCreateProgram(
            &raw mut program,
            source_c.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(retryable(format!(
            "NVRTC program creation failed: {}",
            describe(status)
        )));
    }
    // From here on every exit must destroy `program`, so the body is a closure
    // and the destroy is unconditional. The C++ repeats the destroy at each of
    // its five early returns and forgets it at none of them -- but it is five
    // chances to forget, and this is zero.
    let outcome = compile_into(program, architecture);
    // SAFETY: `program` was created above and has not been destroyed.
    unsafe { nvrtc::nvrtcDestroyProgram(&raw mut program) };
    outcome
}

/// The compile proper, with `program` guaranteed destroyed by the caller.
fn compile_into(program: nvrtc::nvrtcProgram, architecture: &str) -> std::result::Result<Vec<u8>, CompileError> {
    let retryable = |message: String| CompileError {
        kind: FailureKind::Retryable,
        message,
    };

    let Ok(arch) = CString::new(architecture) else {
        return Err(retryable("architecture flag contains a NUL".into()));
    };
    let options: [*const std::ffi::c_char; 5] = [
        arch.as_ptr(),
        c"--std=c++17".as_ptr(),
        c"--fmad=false".as_ptr(),
        c"--prec-div=true".as_ptr(),
        c"--prec-sqrt=true".as_ptr(),
    ];
    // SAFETY: `program` is live; `options` is a five-element array of pointers
    // to C strings that outlive the call.
    let status = unsafe {
        nvrtc::nvrtcCompileProgram(
            program,
            i32::try_from(options.len()).expect("five fits an i32"),
            options.as_ptr(),
        )
    };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        // The log is the whole diagnostic; the status code alone says only
        // "compilation failed". Read it even on a non-compilation failure --
        // NVRTC fills it either way, and an empty log costs one call.
        return Err(CompileError {
            kind: if status == nvrtc::nvrtcResult::NVRTC_ERROR_COMPILATION {
                FailureKind::Deterministic
            } else {
                FailureKind::Retryable
            },
            message: format!("NVRTC fused compilation failed: {}", log(program)),
        });
    }

    let mut size = 0usize;
    // SAFETY: `program` compiled successfully; `size` is a live out-parameter.
    let status = unsafe { nvrtc::nvrtcGetCUBINSize(program, &raw mut size) };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(retryable(format!(
            "NVRTC fused cubin sizing failed: {}",
            describe(status)
        )));
    }
    if size == 0 {
        return Err(retryable("NVRTC produced an empty cubin".into()));
    }
    let mut cubin = vec![0u8; size];
    // SAFETY: `cubin` is exactly the `size` bytes NVRTC just asked for. The
    // cast is `u8` to `c_char`, which differ only in signedness.
    let status = unsafe { nvrtc::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(retryable(format!(
            "NVRTC fused cubin extraction failed: {}",
            describe(status)
        )));
    }
    Ok(cubin)
}

/// The compiler log, or a note that it could not be read.
fn log(program: nvrtc::nvrtcProgram) -> String {
    let mut size = 0usize;
    // SAFETY: `program` is live; `size` is a live out-parameter.
    let status = unsafe { nvrtc::nvrtcGetProgramLogSize(program, &raw mut size) };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS || size <= 1 {
        return "(no diagnostic)".into();
    }
    let mut buffer = vec![0u8; size];
    // SAFETY: `buffer` holds the `size` bytes NVRTC reported, NUL included.
    let status = unsafe { nvrtc::nvrtcGetProgramLog(program, buffer.as_mut_ptr().cast()) };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return "(diagnostic unavailable)".into();
    }
    // NVRTC writes a NUL-terminated string into a buffer sized to include it.
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end])
        .trim_end()
        .to_string()
}

/// The `--gpu-architecture` flag for a compute capability.
///
/// `sm_{major}{minor}` with no separator, which is the toolkit's own spelling
/// and is why 12.0 is `sm_120` and not `sm_1200`: the minor is one digit
/// because no shipped capability has ever had two.
#[must_use]
pub fn arch_flag(major: i32, minor: i32) -> String {
    format!("--gpu-architecture=sm_{major}{minor}")
}

/// NVRTC's own name for a status code.
fn describe(status: nvrtc::nvrtcResult) -> String {
    // SAFETY: `nvrtcGetErrorString` returns a pointer to a static string for
    // every enumerator, and null for none of them -- but it is checked anyway,
    // because a version mismatch could hand back a code this build does not
    // name and there is no contract covering that case.
    let message = unsafe { nvrtc::nvrtcGetErrorString(status) };
    if message.is_null() {
        return format!("{status:?}");
    }
    // SAFETY: NUL-terminated, static lifetime, owned by the library.
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag's spelling is what the toolkit parses, and getting the
    /// separator wrong is a compile failure per fire rather than a build error.
    #[test]
    fn the_architecture_flag_runs_major_and_minor_together() {
        assert_eq!(arch_flag(8, 9), "--gpu-architecture=sm_89");
        assert_eq!(arch_flag(9, 0), "--gpu-architecture=sm_90");
        assert_eq!(arch_flag(12, 0), "--gpu-architecture=sm_120");
    }

    /// A source with an interior NUL is refused rather than truncated. NVRTC
    /// takes a C string, so the bytes past the NUL would simply not be
    /// compiled -- and a prefix of a generated kernel can compile cleanly and
    /// be missing its commit.
    #[test]
    fn an_interior_nul_is_deterministic_rather_than_a_truncated_compile() {
        let error = compile("extern \"C\" __global__ void k(){}\0trailing", "sm_89")
            .expect_err("a NUL-bearing source must not reach NVRTC");
        assert_eq!(error.kind, FailureKind::Deterministic);
        assert!(
            error.message.contains("NUL"),
            "the message must name the cause: {}",
            error.message
        );
    }
}

// ── The loaded module the compile above produces ──

// A cubin, loaded: the module, the function, and the width to launch it at.
//
// # What `block_threads` is for, and why it is a power of two
//
// The generated fused kernels reduce with the standard halving tree —
// `for (stride = blockDim.x / 2; stride > 0; stride >>= 1)`. A tree written
// that way is correct only when `blockDim.x` is a power of two; at 768
// threads the first halving is 384, the second 192, and the lanes at the top
// of the block are folded twice while the ones in the middle are never folded
// at all. The answer is wrong and nothing reports it.
//
// So the launch width is not a tuning constant. It is
// `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` — what the compiled function's
// register pressure actually permits — rounded DOWN to a power of two. Down,
// because rounding up is a launch failure, and the attribute is a ceiling.
//
// # Why the module is owned rather than leaked
//
// `CUmodule` is a process-wide resource and a program that is registered,
// bound, and closed a thousand times over a serving day leaks a thousand of
// them. [`Module`] unloads in `Drop`, which is the whole reason the cubin is
// wrapped at all instead of being loaded inline where it is compiled.

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
mod tests_2 {
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
