//! NVRTC: one self-contained translation unit in, one launchable region out.
//!
//! **NO INCLUDE PATH.** `tensor-compiler`'s CUDA emitter splices every runtime
//! header into the string it hands over (`runtime/cuda/*.cuh` plus the
//! generated RNG preamble), so an `#include` in an emitted source is an
//! emitter bug rather than a search-path problem, and `libnvrtc` is
//! `dlopen`ed like every other CUDA symbol — this crate builds with no
//! toolkit. Compiles target the device's real `sm_XY`, not `compute_XY`'s
//! re-JIT-ed PTX.
//!
//! **WHY NOT `kernels-cuda`'s JIT.** That machinery exists and was the first
//! thing checked: `kernels_cuda::jit` compiles a *named, in-crate* template
//! against a specialization record, caches by
//! `(entry, specialization, arch)`, and answers with a `CUfunction` it owns
//! for the life of the process. Everything in that sentence is wrong here.
//! A guest program's source is not in any crate — it arrives per registration
//! from a host that ran `tensor-compiler` — so there is no template to name;
//! the cache key that makes a PTIR cubin reusable is
//! [`engine::cache_identity`] (backend × device × stage signature × four
//! version numbers) plus a fingerprint of the emitted bytes, not a
//! specialization struct; and a program is CLOSED, which has to unload its
//! modules, where a kernel template never is. So this is a second NVRTC
//! caller, and it shares with the first only the library.
//!
//! **THREE TIERS, AND THE NEGATIVE ONE IS NOT AN OPTIMISATION.** In memory by
//! program hash (a re-registration of a live program compiles nothing); on
//! disk by identity × source fingerprint (a compile that survives the
//! process); and a bounded negative tier that remembers only
//! [`Failure::Deterministic`] — a source NVRTC rejects is rejected forever and
//! answering from memory is what keeps a hot-looping guest from re-running
//! the compiler on every attempt, while a machine that was merely out of
//! memory must be retried.
//!
//! Nothing is installed until every region of every stage compiles: a
//! half-installed program is a wrong answer, not a slow one.

use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use engine::engine_api::program::LaunchStagePlan;
use engine::{
    Backend, Bounded, CacheStats, Emitted, EmittedKernel, ExecPlan, Failure, Lookup,
    MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES, Slot, Stages, Versions,
    cache_identity, combined_signature,
};

use crate::error::{Fault, Result};
use engine::engine_api::program::{KernelKind, LibraryOp, RegionKind};
// `Stage` is this module's own struct (one compiled stage), so PTIR's
// attachment-point enum is reached by path rather than imported.
use engine::tensor_ir::registry::Stage as Attach;

/// The only kind the CUDA emitter produces, named from the contract rather
/// than written as `1`, so a renumbering is a build break and not an empty
/// slot forever.
const KERNEL_FUSED: KernelKind = KernelKind::Fused;

/// Fallback launch width when `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`
/// cannot be read: 256 is a power of two and inside every device's limit, so
/// a failed query still launches.
const DEFAULT_BLOCK_THREADS: u32 = 256;

/// The warp width, and the floor a rounded-down launch width may not cross.
const WARP: u32 = 32;

/// The largest block CUDA permits.
const MAX_BLOCK_THREADS: u32 = 1024;

// ─────────────────────────────────────────────────────────────────────────────
// What a compile needs to know that it cannot read off the program
// ─────────────────────────────────────────────────────────────────────────────

/// The device a program is compiled *for*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// Compute capability major, for `sm_XY`.
    pub major: i32,
    /// Compute capability minor.
    pub minor: i32,
    /// A stable id for this GPU, so two devices of different families never
    /// share a cached compilation. The ordinal is enough in a one-shell
    /// process; a multi-GPU host wants the UUID.
    pub device: u64,
    /// NVRTC's own `(major, minor)`. Two NVRTC versions compile one source to
    /// different machine code, so a cubin must not outlive a toolkit upgrade —
    /// and [`engine::cache_identity`] has no seat for it (it is shared with
    /// backends that never call NVRTC), so it is folded into the memory key
    /// here and into the disk key through the source fingerprint's sibling.
    pub nvrtc: (i32, i32),
}

impl Target {
    /// The target a bound [`Context`](crate::device::Context) describes.
    ///
    /// # Errors
    ///
    /// [`Fault::Compile`] with a retryable failure when NVRTC cannot be asked
    /// its version — which is the same condition as "there is no NVRTC", and
    /// is a property of the machine rather than of the program.
    pub fn of(context: &crate::device::Context) -> Result<Target> {
        let (major, minor) = context.capability();
        Ok(Target {
            major,
            minor,
            device: context.ordinal() as u64,
            nvrtc: nvrtc_version()?,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NVRTC
// ─────────────────────────────────────────────────────────────────────────────

/// The `--gpu-architecture` flag for a compute capability: `sm_{major}{minor}`
/// with no separator, so 12.0 is `sm_120` (the minor is always one digit).
#[must_use]
pub fn arch_flag(major: i32, minor: i32) -> String {
    format!("--gpu-architecture=sm_{major}{minor}")
}

/// The NVRTC library's version, `(major, minor)`.
///
/// # Errors
///
/// [`Fault::Compile`] (retryable) when `libnvrtc` cannot be loaded or the
/// query fails: both are facts about the machine, not about a program.
pub fn nvrtc_version() -> Result<(i32, i32)> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::nvrtc::sys as nvrtc;

        let mut major = 0;
        let mut minor = 0;
        // SAFETY: both out-parameters are live `i32`s for the call's duration.
        let status = unsafe { nvrtc::nvrtcVersion(&raw mut major, &raw mut minor) };
        if status == nvrtc::nvrtcResult::NVRTC_SUCCESS {
            Ok((major, minor))
        } else {
            Err(Fault::Compile(Failure::Retryable {
                reason: format!("cannot query NVRTC's version: {}", describe(status)),
            }))
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// Compile one emitted region to a cubin for `architecture` (an `sm_XY` flag
/// from [`arch_flag`]); `source` is the whole translation unit, nothing
/// prepended.
///
/// # Errors
///
/// [`Failure::Deterministic`] carrying NVRTC's log when the source is
/// rejected — that answer is worth remembering — and [`Failure::Retryable`]
/// for everything else.
pub fn compile(source: &str, architecture: &str) -> std::result::Result<Vec<u8>, Failure> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::nvrtc::sys as nvrtc;
        use std::ffi::CString;

        let retryable = |reason: String| Failure::Retryable { reason };

        // A NUL in the source would truncate the C string and compile a
        // PREFIX — which can succeed and produce a kernel missing its tail.
        // Refuse it here, where the cause is still nameable.
        let Ok(source_c) = CString::new(source) else {
            return Err(Failure::Deterministic {
                reason: "the emitted source contains an interior NUL byte".into(),
            });
        };

        let mut program: nvrtc::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: `program` is a live out-parameter; both strings outlive the
        // call; zero headers is what makes the two header arrays legitimately
        // null.
        let status = unsafe {
            nvrtc::nvrtcCreateProgram(
                &raw mut program,
                source_c.as_ptr(),
                c"ptir_fused_region.cu".as_ptr(),
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
        // Past here every exit must destroy `program`, so the body is a helper
        // and the destroy is unconditional.
        let outcome = compile_into(program, architecture);
        // SAFETY: `program` was created above and has not been destroyed.
        unsafe { nvrtc::nvrtcDestroyProgram(&raw mut program) };
        outcome
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (source, architecture);
        Err(Failure::Retryable {
            reason: "this build carries no CUDA runtime, so there is no NVRTC".into(),
        })
    }
}

/// The compile proper, with `program` guaranteed destroyed by the caller.
#[cfg(feature = "_cuda")]
fn compile_into(
    program: cudarc::nvrtc::sys::nvrtcProgram,
    architecture: &str,
) -> std::result::Result<Vec<u8>, Failure> {
    use cudarc::nvrtc::sys as nvrtc;
    use std::ffi::CString;

    let retryable = |reason: String| Failure::Retryable { reason };

    let Ok(arch) = CString::new(architecture) else {
        return Err(retryable("the architecture flag contains a NUL".into()));
    };
    // `--fmad=false` / `--prec-div=true` / `--prec-sqrt=true` is a
    // DETERMINISM CONTRACT and not a taste: the channel plane promises
    // bit-for-bit reproducibility, and a contracted FMA or a fast reciprocal
    // moves a lane past the tolerance the host interpreter — which has
    // neither — is diffed against.
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
        // The log is the whole diagnostic; read it either way — NVRTC fills it
        // on a non-compilation failure too.
        let reason = format!("NVRTC refused the emitted region: {}", log(program));
        return Err(if status == nvrtc::nvrtcResult::NVRTC_ERROR_COMPILATION {
            Failure::Deterministic { reason }
        } else {
            Failure::Retryable { reason }
        });
    }

    let mut size = 0usize;
    // SAFETY: `program` compiled successfully; `size` is a live out-parameter.
    let status = unsafe { nvrtc::nvrtcGetCUBINSize(program, &raw mut size) };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(retryable(format!(
            "NVRTC cubin sizing failed: {}",
            describe(status)
        )));
    }
    if size == 0 {
        return Err(retryable("NVRTC produced an empty cubin".into()));
    }
    let mut cubin = vec![0u8; size];
    // SAFETY: `cubin` is exactly the `size` bytes NVRTC just asked for; the
    // cast is `u8` to `c_char`, which differ only in signedness.
    let status = unsafe { nvrtc::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
        return Err(retryable(format!(
            "NVRTC cubin extraction failed: {}",
            describe(status)
        )));
    }
    Ok(cubin)
}

/// The compiler log, or a note saying it could not be read.
#[cfg(feature = "_cuda")]
fn log(program: cudarc::nvrtc::sys::nvrtcProgram) -> String {
    use cudarc::nvrtc::sys as nvrtc;

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
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end])
        .trim_end()
        .to_string()
}

/// NVRTC's own name for a status code.
#[cfg(feature = "_cuda")]
fn describe(status: cudarc::nvrtc::sys::nvrtcResult) -> String {
    // SAFETY: `nvrtcGetErrorString` returns a static string per enumerator;
    // null is checked anyway, in case a version mismatch hands back a code it
    // has no name for.
    let message = unsafe { cudarc::nvrtc::sys::nvrtcGetErrorString(status) };
    if message.is_null() {
        return format!("{status:?}");
    }
    // SAFETY: NUL-terminated, static lifetime, owned by the library.
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

// ─────────────────────────────────────────────────────────────────────────────
// A loaded cubin
// ─────────────────────────────────────────────────────────────────────────────

/// A loaded cubin and one entry point inside it.
///
/// NOT `Clone`: [`Drop`] unloads the module, so a copy would unload it twice.
/// Share through the [`Arc`] the compile plane hands out.
#[derive(Debug)]
pub struct Module {
    #[cfg(feature = "_cuda")]
    module: cudarc::driver::sys::CUmodule,
    #[cfg(feature = "_cuda")]
    function: cudarc::driver::sys::CUfunction,
    block_threads: u32,
    entry_name: String,
}

// SAFETY: `CUmodule`/`CUfunction` are context-scoped, and this crate binds one
// primary context per device, so a handle is valid on every thread that has
// that context current. The only unsafe race — unload during launch — is
// `Drop`'s, and a `Module` is dropped only when its program is closed.
unsafe impl Send for Module {}
// SAFETY: as above; every method below reads an immutable handle.
unsafe impl Sync for Module {}

impl Module {
    /// Load `cubin` and resolve `entry_name` inside it.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the image is not loadable here or carries no
    /// such entry point — a cubin/source mismatch, which is exactly why the
    /// caller invalidates its disk entry when this fails.
    pub fn load(cubin: &[u8], entry_name: &str) -> Result<Module> {
        if cubin.is_empty() {
            return Err(Fault::program("cuModuleLoadData", "the cubin is empty"));
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::driver::sys as dr;

            let Ok(entry_c) = std::ffi::CString::new(entry_name) else {
                return Err(Fault::program(
                    "cuModuleGetFunction",
                    format!("the entry name `{entry_name}` contains a NUL"),
                ));
            };

            let mut module: dr::CUmodule = std::ptr::null_mut();
            // SAFETY: `cubin` is a live image and `module` a live
            // out-parameter. `cuModuleLoadData` reads the length out of the
            // image's own header, which is why the slice length is not passed
            // and an empty slice is refused above.
            let code = unsafe { dr::cuModuleLoadData(&raw mut module, cubin.as_ptr().cast()) };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(Fault::Device {
                    call: "cuModuleLoadData",
                    code: code as i32,
                });
            }

            let mut function: dr::CUfunction = std::ptr::null_mut();
            // SAFETY: `module` loaded above; `entry_c` outlives the call.
            let code =
                unsafe { dr::cuModuleGetFunction(&raw mut function, module, entry_c.as_ptr()) };
            if code != dr::CUresult::CUDA_SUCCESS {
                // Unload before returning: a stale disk cache produces this
                // failure in a loop, one leaked module each.
                //
                // SAFETY: `module` is loaded and no function of it is in flight.
                unsafe { dr::cuModuleUnload(module) };
                return Err(Fault::Device {
                    call: "cuModuleGetFunction",
                    code: code as i32,
                });
            }
            Ok(Module {
                module,
                function,
                block_threads: launch_width(function),
                entry_name: entry_name.to_string(),
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = entry_name;
            Err(Fault::Runtimeless)
        }
    }

    /// The entry point handle, for `cuLaunchKernel`.
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub const fn function(&self) -> cudarc::driver::sys::CUfunction {
        self.function
    }

    /// The width to launch this function at: a power of two inside its own
    /// register limit.
    ///
    /// **THE REDUCTIONS DEMAND A POWER OF TWO.** The emitted fused kernels
    /// reduce with a halving tree (`stride = blockDim.x / 2`), which folds
    /// lanes wrong — silently — at any other width.
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
        #[cfg(feature = "_cuda")]
        if !self.module.is_null() {
            // SAFETY: loaded in `load`, dropped once. The return code is
            // ignored because a `Drop` has nowhere to report it.
            unsafe { cudarc::driver::sys::cuModuleUnload(self.module) };
        }
    }
}

/// The register-limited launch width, rounded down to a power of two.
#[cfg(feature = "_cuda")]
fn launch_width(function: cudarc::driver::sys::CUfunction) -> u32 {
    use cudarc::driver::sys as dr;

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

/// `max_threads` rounded down to a power of two inside `[WARP, 1024]`. Split
/// out of the query so the arithmetic is testable with no GPU in the room.
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

// ─────────────────────────────────────────────────────────────────────────────
// The disk tier
// ─────────────────────────────────────────────────────────────────────────────

/// File magic; the `01` is a format version, bumped when the layout changes so
/// that older entries miss rather than misparse.
const MAGIC: &[u8; 8] = b"PTRCUB01";

/// Header bytes before the variable-length tail: magic + three `u32` + a `u64`.
const HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 8;

/// The largest entry that will be read: a corrupt header claiming a huge
/// length must not become an allocation the size of the claim.
const MAX_ENTRY_BYTES: u64 = 128 * 1024 * 1024;

/// Serialises the temp-file names of concurrent writers inside one process.
static NONCE: AtomicU64 = AtomicU64::new(0);

/// Where cubins are kept, or `None` when nowhere is writable.
///
/// **EVERY FAILURE HERE IS A MISS, NEVER AN ERROR.** NVRTC is always
/// available when the cache is not, and a corrupt entry is removed on the way
/// past so that it is paid for once rather than every run.
#[derive(Clone, Debug)]
pub struct Disk {
    directory: Option<PathBuf>,
}

impl Disk {
    /// The cache a deployment's stated directory roots, or nowhere when it
    /// stated none.
    ///
    /// **`Disk::from_env` STOOD HERE** and resolved `$PIE_HOME/cache/ptir-cuda`,
    /// else `$XDG_CACHE_HOME/pie/ptir-cuda`, else `$HOME/.cache/pie/ptir-cuda`.
    /// Article 9 (alto design §1) says a shell reads no environment, and this
    /// was the last read in the crate: the directory is a DEPLOYMENT fact, so
    /// it arrives typed on `Boot::program_cache_dir` — off the boot document's
    /// `[cache] dir`, which the worker has written all along — exactly as the
    /// warm-boot weight artifacts' directory does.
    ///
    /// `None` is [`Disk::disabled`], and that costs nothing but NVRTC time:
    /// every failure of this cache is a miss and never an error.
    #[must_use]
    pub fn rooted(directory: Option<impl Into<PathBuf>>) -> Disk {
        Disk {
            directory: directory.map(Into::into),
        }
    }

    /// A cache rooted at an explicit directory.
    #[must_use]
    pub fn at(directory: impl Into<PathBuf>) -> Disk {
        Disk {
            directory: Some(directory.into()),
        }
    }

    /// A cache that stores nothing: every load misses, every store is a no-op.
    #[must_use]
    pub const fn disabled() -> Disk {
        Disk { directory: None }
    }

    /// Whether anything will actually be written.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.directory.is_some()
    }

    /// The cubin stored for `(key, region_index, entry)`, if it still matches.
    /// A mismatched or malformed entry is removed before `None` is returned.
    #[must_use]
    pub fn load(&self, key: &str, region_index: u32, entry: &str) -> Option<Vec<u8>> {
        let path = self.path(key, region_index)?;
        let bytes = fs::read(&path).ok()?;
        match parse(&bytes, key, region_index, entry) {
            Some(cubin) => Some(cubin),
            None => {
                self.invalidate(key, region_index);
                None
            }
        }
    }

    /// Store `cubin` for `(key, region_index, entry)`.
    ///
    /// Written to a per-writer temp file and atomically `rename`d in:
    /// concurrent writers are normal, and a half-written cubin another process
    /// loads is a segfault in the driver.
    pub fn store(&self, key: &str, region_index: u32, entry: &str, cubin: &[u8]) {
        let Some(directory) = self.directory.as_ref() else {
            return;
        };
        if u32::try_from(key.len()).is_err() || u32::try_from(entry.len()).is_err() {
            return;
        }
        if fs::create_dir_all(directory).is_err() {
            return;
        }
        let Some(destination) = self.path(key, region_index) else {
            return;
        };

        let mut bytes = Vec::with_capacity(HEADER_BYTES + key.len() + entry.len() + cubin.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&region_index.to_le_bytes());
        bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(entry.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(cubin.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(entry.as_bytes());
        bytes.extend_from_slice(cubin);

        let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
        let temporary =
            destination.with_extension(format!("cubin.tmp-{}-{nonce}", std::process::id()));
        let written = fs::File::create(&temporary).and_then(|mut file| {
            file.write_all(&bytes)?;
            file.sync_all()
        });
        if written.is_err() || fs::rename(&temporary, &destination).is_err() {
            let _ = fs::remove_file(&temporary);
        }
    }

    /// Remove whatever is stored for `(key, region_index)`.
    pub fn invalidate(&self, key: &str, region_index: u32) {
        if let Some(path) = self.path(key, region_index) {
            let _ = fs::remove_file(path);
        }
    }

    /// The file a `(key, region_index)` pair maps to.
    fn path(&self, key: &str, region_index: u32) -> Option<PathBuf> {
        let directory = self.directory.as_ref()?;
        Some(directory.join(format!(
            "{:016x}-{region_index}.cubin",
            engine::tensor_ir::fnv1a64(key.as_bytes())
        )))
    }
}

/// The identity string plus an eight-byte fingerprint of the source, appended
/// rather than folded in so the identity stays readable inside a key.
///
/// **THE FINGERPRINT IS WHY THIS FUNCTION EXISTS.** Editing `tensor-compiler`'s
/// device templates bumps no version number, so without the source in the key
/// a stale cubin matches today's identity and every kernel edit silently does
/// nothing.
#[must_use]
pub fn disk_key(identity: &str, source: &str) -> String {
    let hash = engine::tensor_ir::fnv1a64(source.as_bytes());
    let mut key = String::with_capacity(identity.len() + 16);
    key.push_str(identity);
    for byte in hash.to_le_bytes() {
        // Hex, so a stored key stays human-readable; both sides of the
        // comparison are produced here, so the spelling is free to choose.
        let _ = write!(key, "{byte:02x}");
    }
    key
}

/// Validate a stored entry and return its cubin.
///
/// The filename is only a 64-bit hash of the key, so the key and the entry
/// name are stored and compared here too: a hash collision would otherwise
/// load one program's machine code for another's launch. Every length is
/// checked against the file's own size before any slice, so a lying header is
/// a miss and not a panic.
fn parse(bytes: &[u8], key: &str, region_index: u32, entry: &str) -> Option<Vec<u8>> {
    if bytes.len() < HEADER_BYTES || bytes.len() as u64 > MAX_ENTRY_BYTES {
        return None;
    }
    if &bytes[..8] != MAGIC {
        return None;
    }
    let stored_region = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let key_size = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let entry_size = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    let cubin_size = u64::from_le_bytes(bytes[20..28].try_into().ok()?);

    if stored_region != region_index || key_size != key.len() || entry_size != entry.len() {
        return None;
    }
    // The tail must be exactly the three pieces the header describes; a longer
    // tail means header and file disagree.
    let tail = bytes.len().checked_sub(HEADER_BYTES)?;
    let claimed = (key_size as u64)
        .checked_add(entry_size as u64)?
        .checked_add(cubin_size)?;
    if tail as u64 != claimed {
        return None;
    }

    let key_at = HEADER_BYTES;
    let entry_at = key_at + key_size;
    let cubin_at = entry_at + entry_size;
    if &bytes[key_at..entry_at] != key.as_bytes() || &bytes[entry_at..cubin_at] != entry.as_bytes()
    {
        return None;
    }
    Some(bytes[cubin_at..].to_vec())
}

// ─────────────────────────────────────────────────────────────────────────────
// The compiled program
// ─────────────────────────────────────────────────────────────────────────────

/// One compiled region: the module that holds it, and which region it is.
#[derive(Debug)]
pub struct Region {
    /// Which region of its stage this is.
    pub region_index: u32,
    /// The loaded cubin and its entry point.
    pub module: Arc<Module>,
}

/// One compiled stage: every generated region it declares, in region order.
///
/// Shared rather than owned: two programs naming the same stage share one
/// cubin, and a `CUmodule` unloaded while another program's launch is in
/// flight is a fault.
#[derive(Debug, Clone)]
pub struct Stage {
    /// The stage's signature hash, as its plan states it.
    pub signature_hash: u64,
    /// The generated regions, in ascending `region_index`.
    pub regions: Arc<Vec<Region>>,
}

impl Stage {
    /// The region with this index, if it was compiled.
    #[must_use]
    pub fn region(&self, region_index: u32) -> Option<&Region> {
        self.regions
            .iter()
            .find(|region| region.region_index == region_index)
    }
}

/// A registered program's compiled form: one [`Stage`] per stage plan.
#[derive(Debug, Clone)]
pub struct Compiled {
    /// The stages, in plan order.
    pub stages: Arc<Vec<Stage>>,
    /// The stage plans these were compiled from, in the same order. Carried
    /// rather than looked up, so a compiled program cannot drift from its plan.
    pub plans: Arc<Vec<LaunchStagePlan>>,
    /// Each stage's attachment point (`LaunchStage::kind`). Carried because
    /// `LaunchStagePlan` has no `kind`, and firing by position picks the
    /// adapter rather than the sampler once a program has a prologue.
    pub kinds: Arc<Vec<Attach>>,
}

impl Compiled {
    /// The index of the first stage with this attachment point.
    #[must_use]
    pub fn stage_of_kind(&self, kind: Attach) -> Option<usize> {
        self.kinds.iter().position(|&k| k == kind)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The cache
// ─────────────────────────────────────────────────────────────────────────────

/// The compile cache: the only thing in this crate that calls NVRTC.
#[derive(Debug)]
pub struct Cache {
    programs: Bounded<u64, Compiled>,
    stages: Stages<Stage>,
    negative: Bounded<u64, String>,
    disk: Disk,
    stats: CacheStats,
}

impl Default for Cache {
    /// A cache that stores nothing. The directory is the deployment's and
    /// arrives on the `Boot` (article 9); a `Default` that went looking for
    /// one would be the environment read this crate no longer makes.
    fn default() -> Cache {
        Cache::new(Disk::disabled())
    }
}

impl Cache {
    /// A cache backed by `disk`.
    #[must_use]
    pub fn new(disk: Disk) -> Cache {
        Cache {
            programs: Bounded::new(MAX_PROGRAM_ENTRIES),
            stages: Stages::new(MAX_STAGE_ENTRIES),
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            disk,
            stats: CacheStats::default(),
        }
    }

    /// The persistent tier this cache compiles into.
    #[must_use]
    pub const fn disk(&self) -> &Disk {
        &self.disk
    }

    /// What the tiers have been doing.
    ///
    /// **AN ABSENCE HAS NO OUTPUT**, so the one claim worth asserting about a
    /// cache — that the second bind of a program compiles nothing — is only
    /// reachable through [`CacheStats::compilations`].
    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Compile `plan`'s generated regions, or answer from a tier. `versions`
    /// carries the identity's four version numbers, so a host-side bump misses
    /// rather than reusing a stale cubin.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] when the program cannot compile here — only
    /// these are remembered — and [`Failure::Retryable`] when the machine
    /// could not.
    pub fn compile(
        &mut self,
        program_hash: u64,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Compiled, Failure> {
        if let Some(compiled) = self.programs.get(&program_hash) {
            self.stats.memory_hits += 1;
            return Ok(compiled.clone());
        }

        let program_identity = cache_identity(
            Backend::Cuda,
            target.device,
            combined_signature(&plan.package.plans),
            versions,
        );
        let program_key = engine::tensor_ir::fnv1a64(program_identity.as_bytes());
        if let Some(reason) = self.negative.get(&program_key) {
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic {
                reason: reason.clone(),
            });
        }

        match self.build(plan, kernels, versions, target) {
            Ok(compiled) => {
                // Past the last failure: only now is anything installed.
                self.stages.commit();
                self.programs.insert(program_hash, compiled.clone());
                Ok(compiled)
            }
            Err(failure) => {
                // A half-failed program leaves no half-stage behind.
                self.stages.abandon();
                if let Failure::Deterministic { reason } = &failure {
                    self.negative.insert(program_key, reason.clone());
                }
                Err(failure)
            }
        }
    }

    /// Forget `program_hash`, dropping this cache's share of its modules.
    pub fn forget(&mut self, program_hash: u64) {
        self.programs.remove(&program_hash);
    }

    /// The compile proper. Installs nothing; the caller commits or abandons.
    fn build(
        &mut self,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Compiled, Failure> {
        let index = Emitted::index(kernels).map_err(|duplicate| Failure::Deterministic {
            reason: format!(
                "the emitted kernel table names slot (kind {}, stage {}, region {}) twice; \
                 an engine cannot know which of the two the host meant",
                duplicate.kind as u32, duplicate.stage, duplicate.region
            ),
        })?;
        let architecture = arch_flag(target.major, target.minor);

        let mut stages = Vec::with_capacity(plan.package.plans.len());
        for (stage_index, stage_plan) in plan.package.plans.iter().enumerate() {
            let stage_index = u32::try_from(stage_index).map_err(|_| Failure::Deterministic {
                reason: "a program with more than four billion stages is not a program".into(),
            })?;
            let identity = cache_identity(
                Backend::Cuda,
                target.device,
                stage_plan.signature_hash,
                versions,
            );
            // NVRTC's version is not in `cache_identity` — that record is
            // shared with backends that never call NVRTC — so it is folded
            // into the memory key here.
            let key = fnv1a64_with(
                identity.as_bytes(),
                &[
                    target.nvrtc.0.to_le_bytes().as_slice(),
                    target.nvrtc.1.to_le_bytes().as_slice(),
                ],
            );
            let (lookup, hit) = self.stages.lookup(key, stage_plan.identity);
            match lookup {
                Lookup::Hit => {
                    self.stats.memory_hits += 1;
                    if let Some(stage) = hit {
                        stages.push(stage);
                        continue;
                    }
                }
                // A signature collision builds the stage unshared: two stages
                // that hash alike are still two valid stages.
                Lookup::Collided | Lookup::Miss => {}
            }

            let compiled =
                self.build_stage(stage_index, stage_plan, &index, &identity, &architecture)?;
            if lookup == Lookup::Miss {
                self.stages
                    .stage(key, stage_plan.identity, compiled.clone());
            }
            stages.push(compiled);
        }
        Ok(Compiled {
            stages: Arc::new(stages),
            plans: Arc::new(plan.package.plans.clone()),
            // `plans` is parallel to `package.stages` — `adopt_launch_package`
            // refuses a package where it is not — so kinds index the same way.
            kinds: Arc::new(plan.package.stages.iter().map(|s| s.stage).collect()),
        })
    }

    /// Every generated region of one stage.
    fn build_stage(
        &mut self,
        stage_index: u32,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
        identity: &str,
        architecture: &str,
    ) -> std::result::Result<Stage, Failure> {
        let mut regions = Vec::new();
        for region_index in 0..plan.fused.len() {
            let region_index = u32::try_from(region_index).map_err(|_| Failure::Deterministic {
                reason: "a stage with more than four billion regions is not a stage".into(),
            })?;
            // A SECOND-PARTY region has no generated kernel and never will: it
            // is a `kernel_call` or a `sink_call`, which is a NAME the shell
            // launches itself rather than a body the emitter could write. The
            // emitter declines it correctly, and reading that decline as a
            // compile failure would refuse every adapter program this shell
            // can actually run. It is the LIBRARY tag that says so — an
            // emitter that declined a genuinely generated region still has to
            // be a failure, which is what the arms below are for.
            if plan.fused.get(region_index as usize).is_some_and(|region| {
                region.kind == RegionKind::Library(LibraryOp::SecondParty)
            }) {
                continue;
            }
            let (source, entry) = match index.get(KERNEL_FUSED, stage_index, region_index) {
                Slot::Kernel { source, entry } => (source, entry),
                // NOT a `continue`. "The host declined on purpose" presumes a
                // shell with its own path for the region, and this one has
                // none — every region it runs is a compiled `KernelKind::Fused`.
                // Skipping a refusal drops the region's ops from the fire while
                // the plan still budgets their scratch, so they read back as
                // the zeros the fire memset and publish a confident wrong
                // answer. A reason nobody can act on still beats an answer
                // nobody can distinguish.
                Slot::Refused(why) => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} was declined by the \
                             emitter ({why}); this shell runs only compiled regions, so a \
                             declined one would silently not run at all"
                        ),
                    });
                }
                Slot::Absent => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} is a generated region \
                             and the host emitted nothing for it; this shell carries no \
                             emitter, so there is no slower path to fall back to"
                        ),
                    });
                }
                Slot::Malformed => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} was emitted with \
                             neither a source nor a reason for declining"
                        ),
                    });
                }
            };

            let module = self.region_module(identity, region_index, entry, source, architecture)?;
            regions.push(Region {
                region_index,
                module,
            });
        }
        Ok(Stage {
            signature_hash: plan.signature_hash,
            regions: Arc::new(regions),
        })
    }

    /// One region: disk, else NVRTC.
    fn region_module(
        &mut self,
        identity: &str,
        region_index: u32,
        entry: &str,
        source: &str,
        architecture: &str,
    ) -> std::result::Result<Arc<Module>, Failure> {
        let key = disk_key(identity, source);
        if let Some(cubin) = self.disk.load(&key, region_index, entry) {
            match Module::load(&cubin, entry) {
                Ok(module) => {
                    self.stats.persistent_hits += 1;
                    return Ok(Arc::new(module));
                }
                // A cubin that will not load must not stay on disk.
                Err(_) => self.disk.invalidate(&key, region_index),
            }
        }

        let cubin = compile(source, architecture)?;
        self.stats.compilations += 1;
        let module = Module::load(&cubin, entry).map_err(|error| Failure::Retryable {
            reason: format!("loading `{entry}`: {error}"),
        })?;
        // Stored only after it loads, so an unusable cubin never reaches disk.
        self.disk.store(&key, region_index, entry, &cubin);
        Ok(Arc::new(module))
    }
}

/// FNV-1a over `bytes` and then each of `tails`, as one stream. Folding the
/// extra fields in is what keeps `cache_identity` free of CUDA-only facts.
fn fnv1a64_with(bytes: &[u8], tails: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut fold = |slice: &[u8]| {
        for &byte in slice {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    fold(bytes);
    for tail in tails {
        fold(tail);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wrong separator is a per-registration compile failure, not a build
    /// error, so the spelling is pinned rather than reviewed.
    #[test]
    fn the_architecture_flag_runs_major_and_minor_together() {
        assert_eq!(arch_flag(8, 9), "--gpu-architecture=sm_89");
        assert_eq!(arch_flag(9, 0), "--gpu-architecture=sm_90");
        assert_eq!(arch_flag(12, 0), "--gpu-architecture=sm_120");
    }

    /// The reductions halve `blockDim.x`, so every width must be a power of two.
    #[test]
    fn every_launch_width_is_a_power_of_two() {
        for max in [32, 33, 63, 64, 100, 128, 512, 768, 1024, 2048] {
            let width = round_down_to_power_of_two(max);
            assert!(
                width.is_power_of_two(),
                "{max} rounded to {width}, which is not a power of two"
            );
        }
    }

    /// Down, never up: the attribute is a ceiling and exceeding it fails the
    /// launch. An unbelievable attribute falls back rather than launching a
    /// partial warp or a huge `u32`.
    #[test]
    fn the_launch_width_never_exceeds_what_the_function_permits() {
        assert_eq!(round_down_to_power_of_two(768), 512);
        assert_eq!(round_down_to_power_of_two(1023), 512);
        assert_eq!(round_down_to_power_of_two(1024), 1024);
        assert_eq!(round_down_to_power_of_two(4096), MAX_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(31), DEFAULT_BLOCK_THREADS);
        assert_eq!(round_down_to_power_of_two(-1), DEFAULT_BLOCK_THREADS);
    }

    /// The point of the disk key: a source edit must miss even when every
    /// version number and the identity are unchanged.
    #[test]
    fn editing_the_source_changes_the_disk_key_with_no_version_bump() {
        let identity = "0100000000000000000300000000000000000000-v0003000400000003 00000015";
        let before = disk_key(identity, "__global__ void k() { a(); }");
        let after = disk_key(identity, "__global__ void k() { b(); }");
        assert_ne!(
            before, after,
            "a template edit bumps no version, so the source itself has to be \
             in the key — otherwise yesterday's cubin answers today's launch"
        );
        assert!(before.starts_with(identity), "the identity stays readable");
    }

    fn scratch(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("pie-ptir-disk-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        path
    }

    /// A stored cubin comes back exactly, and only for the request it was
    /// stored for: the filename covers the key alone, so region and entry are
    /// compared out of the file.
    #[test]
    fn an_entry_answers_only_the_exact_request_it_was_stored_for() {
        let disk = Disk::at(scratch("exact"));
        disk.store("key-a", 2, "entry_r2", b"cubin");
        assert_eq!(disk.load("key-a", 2, "entry_r2"), Some(b"cubin".to_vec()));
        assert_eq!(disk.load("key-a", 3, "entry_r2"), None, "wrong region");
        assert_eq!(disk.load("key-a", 2, "entry_r9"), None, "wrong entry name");
        assert_eq!(disk.load("key-b", 2, "entry_r2"), None, "wrong key");
    }

    /// A truncated or corrupt file is a miss, and it is removed on the way
    /// past so the cost is paid once rather than every run.
    #[test]
    fn a_corrupt_entry_is_a_miss_and_is_deleted() {
        let directory = scratch("corrupt");
        let disk = Disk::at(&directory);
        disk.store("key-a", 0, "entry", b"cubin-bytes");
        let path = disk.path("key-a", 0).expect("enabled");
        let good = fs::read(&path).expect("stored");

        fs::write(&path, &good[..good.len() - 3]).expect("truncate");
        assert_eq!(disk.load("key-a", 0, "entry"), None, "a short tail misses");
        assert!(!path.exists(), "and the entry is removed");

        disk.store("key-a", 0, "entry", b"cubin-bytes");
        let mut wrong_magic = fs::read(&path).expect("stored");
        wrong_magic[7] = b'9';
        fs::write(&path, &wrong_magic).expect("write");
        assert_eq!(disk.load("key-a", 0, "entry"), None, "a format bump misses");
        assert!(!path.exists());
    }

    /// A header lying about its lengths is refused without panicking and
    /// without allocating the size of the claim.
    #[test]
    fn a_header_that_lies_about_its_lengths_is_refused_without_panicking() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(b"key-a");
        bytes.extend_from_slice(b"entry");
        assert_eq!(parse(&bytes, "key-a", 0, "entry"), None);
    }

    /// A cache with no home stores nothing and misses everything, and neither
    /// is a failure.
    #[test]
    fn a_disabled_cache_is_a_miss_and_not_a_failure() {
        let disk = Disk::disabled();
        assert!(!disk.is_enabled());
        disk.store("key", 0, "entry", b"cubin");
        assert_eq!(disk.load("key", 0, "entry"), None);
    }

    /// The fold must be the workspace's, since the string it folds came out of
    /// `cache_identity`.
    #[test]
    fn folding_in_tails_is_the_same_as_folding_the_concatenation() {
        assert_eq!(fnv1a64_with(b"", &[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(
            fnv1a64_with(b"ptir", &[]),
            engine::tensor_ir::fnv1a64(b"ptir")
        );
        let joined = fnv1a64_with(b"identity\x0c\x00\x00\x00\x00\x00\x00\x00", &[]);
        let split = fnv1a64_with(b"identity", &[&12u32.to_le_bytes(), &0u32.to_le_bytes()]);
        assert_eq!(joined, split);
    }

    /// An NVRTC upgrade must miss; the identity record carries no NVRTC field,
    /// so the miss has to come from the fold.
    #[test]
    fn an_nvrtc_version_bump_changes_the_stage_key() {
        let identity = b"the-same-identity";
        let before = fnv1a64_with(identity, &[&12i32.to_le_bytes(), &8i32.to_le_bytes()]);
        let after = fnv1a64_with(identity, &[&13i32.to_le_bytes(), &0i32.to_le_bytes()]);
        assert_ne!(before, after);
    }

    /// The kind looked up is the contract's, so a renumbering is a build
    /// break rather than a program whose every region is `Slot::Absent`.
    #[test]
    fn cuda_compiles_the_fused_kind_the_contract_names() {
        assert_eq!(KERNEL_FUSED, KernelKind::Fused);
    }
}
