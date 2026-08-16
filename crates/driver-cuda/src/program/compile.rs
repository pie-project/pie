//! NVRTC: one self-contained translation unit in, one cubin out.
//!
//! No include path: the emitter splices every runtime header into one string,
//! so a `#include` in an emitted source is an emitter bug, and `libnvrtc.so` is
//! `dlopen`ed like every other CUDA symbol (the crate builds with no toolkit).
//! Compiles target the device's real `sm_XY`, not `compute_XY`'s re-JIT-ed PTX.

use std::ffi::{CStr, CString};

use cudarc::nvrtc::sys as nvrtc;

/// Whether a compile failure is worth remembering: a source NVRTC rejects is
/// rejected forever (cache it), but an out-of-memory machine must be retried.
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
    /// What NVRTC said — its compilation log, or its error string otherwise.
    pub message: String,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CompileError {}

/// The NVRTC library's version `(major, minor)`; part of the cache key.
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

/// Compile one emitted region to a cubin for `architecture` (an `sm_XY` string
/// from [`arch_flag`]); `source` is the whole translation unit, nothing prepended.
///
/// # Errors
///
/// [`FailureKind::Deterministic`] with NVRTC's log when the source is rejected,
/// [`FailureKind::Retryable`] otherwise.
pub fn compile(source: &str, architecture: &str) -> std::result::Result<Vec<u8>, CompileError> {
    let retryable = |message: String| CompileError {
        kind: FailureKind::Retryable,
        message,
    };

    // A NUL in the source would truncate the C string and compile a prefix —
    // which can succeed and produce a kernel missing its tail. Refuse it here.
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
    // From here on every exit must destroy `program`, so the body is a helper
    // and the destroy is unconditional.
    let outcome = compile_into(program, architecture);
    // SAFETY: `program` was created above and has not been destroyed.
    unsafe { nvrtc::nvrtcDestroyProgram(&raw mut program) };
    outcome
}

/// The compile proper, with `program` guaranteed destroyed by the caller.
fn compile_into(
    program: nvrtc::nvrtcProgram,
    architecture: &str,
) -> std::result::Result<Vec<u8>, CompileError> {
    let retryable = |message: String| CompileError {
        kind: FailureKind::Retryable,
        message,
    };

    let Ok(arch) = CString::new(architecture) else {
        return Err(retryable("architecture flag contains a NUL".into()));
    };
    // --fmad=false/--prec-div=true/--prec-sqrt=true is a determinism contract:
    // a contracted FMA or fast reciprocal moves a lane past the tolerance.
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
        // The log is the whole diagnostic; read it even on a non-compilation
        // failure — NVRTC fills it either way.
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

/// The `--gpu-architecture` flag for a compute capability: `sm_{major}{minor}`
/// with no separator, so 12.0 is `sm_120` (the minor is always one digit).
#[must_use]
pub fn arch_flag(major: i32, minor: i32) -> String {
    format!("--gpu-architecture=sm_{major}{minor}")
}

/// NVRTC's own name for a status code.
fn describe(status: nvrtc::nvrtcResult) -> String {
    // SAFETY: `nvrtcGetErrorString` returns a static string per enumerator;
    // null is checked anyway in case a version mismatch hands back an unnamed code.
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

    /// A wrong separator is a per-fire compile failure, not a build error.
    #[test]
    fn the_architecture_flag_runs_major_and_minor_together() {
        assert_eq!(arch_flag(8, 9), "--gpu-architecture=sm_89");
        assert_eq!(arch_flag(9, 0), "--gpu-architecture=sm_90");
        assert_eq!(arch_flag(12, 0), "--gpu-architecture=sm_120");
    }

    /// An interior NUL is refused, not truncated — a prefix could miss its commit.
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

// `block_threads` must be a power of two: the fused kernels reduce with a
// halving tree (`stride = blockDim.x / 2`) that silently folds lanes wrong
// otherwise. It is `MAX_THREADS_PER_BLOCK` rounded down (up is a launch
// failure). [`Module`] unloads in `Drop`, or a `CUmodule` leaks per close.

use cudarc::driver::sys as dr;

use crate::error::{Error, Result};

/// Fallback launch width when the attribute cannot be read: 256 is a power of
/// two and within every device's limit, so a failed query still launches.
const DEFAULT_BLOCK_THREADS: u32 = 256;

/// The warp width, and the floor a rounded-down launch width may not go below.
const WARP: u32 = 32;

/// The maximum threads any CUDA block may hold.
const MAX_BLOCK_THREADS: u32 = 1024;

/// A loaded cubin and one entry point inside it. Not `Clone`: `Drop` unloads
/// the module, so a copy would unload it twice — share via `Arc`.
#[derive(Debug)]
pub struct Module {
    module: dr::CUmodule,
    function: dr::CUfunction,
    block_threads: u32,
    entry_name: String,
}

// SAFETY: `CUmodule`/`CUfunction` are context-scoped, and this crate binds one
// primary context per device, so a handle is valid on every thread; the only
// unsafe race (unload during launch) is `Drop`'s.
unsafe impl Send for Module {}
// SAFETY: as above — every method is a read of an immutable handle.
unsafe impl Sync for Module {}

impl Module {
    /// Load `cubin` and resolve `entry_name` inside it.
    ///
    /// # Errors
    ///
    /// If the image is not loadable here or carries no such entry point — a
    /// cubin/source mismatch, which is why the caller invalidates its disk entry.
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
        // SAFETY: `cubin` is a live image, `module` a live out-parameter.
        // `cuModuleLoadData` reads the length from the image's own header, so
        // the slice length is not passed and an empty slice is refused above.
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
            // The entry is missing, so unload before returning; a stale disk
            // cache produces this failure in a loop, one leak each.
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

    /// The width to launch this function at: a power of two within register limits.
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
            // ignored — a `Drop` has nowhere to report it.
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

/// `max_threads` rounded down to a power of two, within `[WARP, 1024]`. Split
/// from the query so the arithmetic is testable without a GPU.
fn round_down_to_power_of_two(max_threads: i32) -> u32 {
    // Below one warp the attribute is not believable; fall back rather than
    // launch a partial warp.
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

    /// The reductions halve `blockDim.x`, so every width must be a power of two.
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

    /// Down, never up: the attribute is a ceiling; exceeding it fails the launch.
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

    /// An unbelievable or negative attribute falls back rather than launching a
    /// partial warp or a huge `u32`.
    #[test]
    fn an_unusable_attribute_falls_back_to_the_default_width() {
        assert_eq!(round_down_to_power_of_two(31), DEFAULT_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(0), DEFAULT_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(-1), DEFAULT_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(i32::MIN), DEFAULT_BLOCK_THREADS);
    }
}
