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
pub fn version() -> Result<(i32, i32), CompileError> {
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
pub fn compile(source: &str, architecture: &str) -> Result<Vec<u8>, CompileError> {
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
fn compile_into(program: nvrtc::nvrtcProgram, architecture: &str) -> Result<Vec<u8>, CompileError> {
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
