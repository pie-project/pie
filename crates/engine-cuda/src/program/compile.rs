#![allow(unsafe_code)]
use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_compiler::codegen::program::KernelKind;
use eta_exec::{
    Backend, Bounded, CacheStats, Emitted, EmittedKernel, ExecPlan, Failure, Lookup,
    MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES, Slot, Stages, Versions,
    cache_identity, combined_signature,
};

use crate::error::{Fault, Result};
use eta_ir::registry::Stage as Attach;

const KERNEL_FUSED: KernelKind = KernelKind::Fused;

#[cfg_attr(not(feature = "device"), allow(dead_code))]
const DEFAULT_BLOCK_THREADS: u32 = 256;

#[cfg_attr(not(feature = "device"), allow(dead_code))]
const WARP: u32 = 32;

#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) const MAX_BLOCK_THREADS: u32 = 1024;

/// The narrowest block a region is emitted for on ROCm: one wavefront.
#[cfg(feature = "rocm")]
const ROCM_WAVE: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub major: i32,
    pub minor: i32,
    pub device: u64,
    /// The runtime compiler's version, NVRTC's or hiprtc's: a compile also
    /// reads its headers.
    pub rtc: (i32, i32),
    /// The device's architecture (`gfx942`), which hiprtc emits for.
    #[cfg(feature = "rocm")]
    pub arch: &'static str,
}

impl Target {
    pub fn of(context: &crate::device::Context) -> Result<Target> {
        let (major, minor) = context.capability();
        Ok(Target {
            major,
            minor,
            device: context.ordinal() as u64,
            rtc: rtc_version()?,
            #[cfg(feature = "rocm")]
            arch: kernels_cuda::jit::cache::arch(context.ordinal()).ok_or(Fault::Compile(
                Failure::Retryable {
                    reason: format!("device {} answered no gfx architecture", context.ordinal()),
                },
            ))?,
        })
    }

    /// The architecture the runtime compiler emits for.
    #[must_use]
    pub fn architecture(&self) -> String {
        #[cfg(not(feature = "rocm"))]
        {
            arch_flag(self.major, self.minor)
        }
        #[cfg(feature = "rocm")]
        {
            self.arch.to_string()
        }
    }
}

/// The codegen backend a fused region is emitted for.
#[cfg(not(feature = "rocm"))]
const BACKEND: Backend = Backend::Cuda;
#[cfg(feature = "rocm")]
const BACKEND: Backend = Backend::Hip;

#[must_use]
pub fn arch_flag(major: i32, minor: i32) -> String {
    format!("--gpu-architecture=sm_{major}{minor}")
}

pub fn rtc_version() -> Result<(i32, i32)> {
    #[cfg(feature = "rocm")]
    {
        let api = kernels_cuda::hip::sys::api()
            .map_err(|reason| Fault::Compile(Failure::Retryable { reason }))?;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: both out-parameters are live `i32`s for the call's duration.
        let status = unsafe { (api.rtc.hiprtcVersion)(&raw mut major, &raw mut minor) };
        if status == kernels_cuda::hip::sys::HIPRTC_SUCCESS {
            Ok((major, minor))
        } else {
            Err(Fault::Compile(Failure::Retryable {
                reason: format!("cannot query hiprtc's version: it answered {status}"),
            }))
        }
    }
    #[cfg(feature = "cuda")]
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
    #[cfg(not(feature = "device"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// The options a region compiles under on ROCm: no contraction of a*b+c
/// into an fma, so a region rounds as the host evaluator does.
#[cfg(feature = "rocm")]
const HIPRTC_OPTIONS: [&str; 3] = ["-std=c++17", "-O3", "-ffp-contract=off"];

/// The options a region compiles under on CUDA, but the architecture:
/// contraction and the fast divide and square root off, so a region rounds
/// as the host evaluator does.
#[cfg(feature = "cuda")]
const NVRTC_OPTIONS: [&std::ffi::CStr; 5] = [
    c"--std=c++17",
    c"--fmad=false",
    c"--prec-div=true",
    c"--prec-sqrt=true",
    c"--maxrregcount=64",
];

/// The options [`compile`] passes the runtime compiler, as bytes; none in
/// a build with no runtime compiler, which compiles nothing.
fn compile_options() -> Vec<&'static [u8]> {
    #[cfg(feature = "rocm")]
    return HIPRTC_OPTIONS.iter().map(|option| option.as_bytes()).collect();
    #[cfg(feature = "cuda")]
    return NVRTC_OPTIONS.iter().map(|option| option.to_bytes()).collect();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    Vec::new()
}

pub fn compile(source: &str, architecture: &str) -> std::result::Result<Vec<u8>, Failure> {
    #[cfg(feature = "rocm")]
    {
        use kernels_cuda::hip::{self, CompileError, code_object};
        // A region states no launch bounds, so hiprtc emits it for the
        // widest block and holds it to that block's registers; one that
        // needs more spills, and its scratch is a buffer the runtime would
        // allocate at its first launch (refused at load). As CUDA's driver
        // narrows such a kernel's block, the region is emitted again for
        // half the block till it needs none: the launch reads the width it
        // was emitted for (`launch_width`). The narrowest is one wavefront.
        let mut bound: Option<u32> = None;
        loop {
            let flag = bound.map(|threads| format!("--gpu-max-threads-per-block={threads}"));
            let options: Vec<&str> = HIPRTC_OPTIONS
                .iter()
                .copied()
                .chain(flag.as_deref())
                .collect();
            let code = match hip::compile("ptir_fused_region", source, architecture, &options, &[]) {
                Ok(built) if built.code.is_empty() => {
                    return Err(Failure::Retryable {
                        reason: "hiprtc produced an empty code object".into(),
                    });
                }
                Ok(built) => built.code,
                Err(CompileError::Rejected(log)) => {
                    return Err(Failure::Deterministic {
                        reason: format!("hiprtc refused the emitted region: {log}"),
                    });
                }
                Err(CompileError::Failed(reason)) => return Err(Failure::Retryable { reason }),
            };
            let narrower = bound.unwrap_or(MAX_BLOCK_THREADS) / 2;
            if code_object::refuse_late_buffers(&code).is_ok() || narrower < ROCM_WAVE {
                return Ok(code);
            }
            bound = Some(narrower);
        }
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::nvrtc::sys as nvrtc;
        use std::ffi::CString;

        let retryable = |reason: String| Failure::Retryable { reason };

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
        let outcome = compile_into(program, architecture);
        // SAFETY: `program` was created above and has not been destroyed.
        unsafe { nvrtc::nvrtcDestroyProgram(&raw mut program) };
        outcome
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = (source, architecture);
        Err(Failure::Retryable {
            reason: "this build carries no CUDA runtime, so there is no NVRTC".into(),
        })
    }
}

#[cfg(feature = "cuda")]
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
    let options: [*const std::ffi::c_char; 6] = [
        arch.as_ptr(),
        NVRTC_OPTIONS[0].as_ptr(),
        NVRTC_OPTIONS[1].as_ptr(),
        NVRTC_OPTIONS[2].as_ptr(),
        NVRTC_OPTIONS[3].as_ptr(),
        NVRTC_OPTIONS[4].as_ptr(),
    ];
    // SAFETY: `program` is live; `options` is a six-element array of pointers
    // to C strings that outlive the call.
    let status = unsafe {
        nvrtc::nvrtcCompileProgram(
            program,
            i32::try_from(options.len()).expect("six fits an i32"),
            options.as_ptr(),
        )
    };
    if status != nvrtc::nvrtcResult::NVRTC_SUCCESS {
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[derive(Debug)]
pub struct Module {
    #[cfg(feature = "device")]
    module: crate::device::dr::CUmodule,
    #[cfg(feature = "device")]
    function: crate::device::dr::CUfunction,
    block_threads: u32,
    /// The dynamic shared memory each launch hands it.
    dynamic_shared: u32,
    /// The blocks of it one multiprocessor holds at once (its registers and
    /// shared memory's occupancy).
    resident: u32,
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
    /// `dynamic_shared` bytes of dynamic shared memory each launch hands
    /// the entry, its allowance raised to them here.
    pub fn load(cubin: &[u8], entry_name: &str, dynamic_shared: u32) -> Result<Module> {
        if cubin.is_empty() {
            return Err(Fault::program("cuModuleLoadData", "the cubin is empty"));
        }
        #[cfg(feature = "device")]
        {
            use crate::device::dr;

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
                // SAFETY: `module` is loaded and no function of it is in flight.
                unsafe { dr::cuModuleUnload(module) };
                return Err(Fault::Device {
                    call: "cuModuleGetFunction",
                    code: code as i32,
                });
            }
            // ROCm keeps the cut's state static and never asks for any.
            #[cfg(not(feature = "rocm"))]
            if dynamic_shared > 0 {
                // SAFETY: `function` was just resolved out of the loaded module.
                let code = unsafe {
                    dr::cuFuncSetAttribute(
                        function,
                        dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        i32::try_from(dynamic_shared).unwrap_or(i32::MAX),
                    )
                };
                if code != dr::CUresult::CUDA_SUCCESS {
                    // SAFETY: as above, no function of it is in flight.
                    unsafe { dr::cuModuleUnload(module) };
                    return Err(Fault::Device {
                        call: "cuFuncSetAttribute",
                        code: code as i32,
                    });
                }
            }
            let block_threads = launch_width(function);
            let mut resident: core::ffi::c_int = 0;
            // SAFETY: `function` was just resolved out of the loaded module;
            // `resident` is a live out-parameter.
            let code = unsafe {
                dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    &raw mut resident,
                    function,
                    i32::try_from(block_threads).unwrap_or(i32::MAX),
                    dynamic_shared as usize,
                )
            };
            if code != dr::CUresult::CUDA_SUCCESS {
                // SAFETY: as above, no function of it is in flight.
                unsafe { dr::cuModuleUnload(module) };
                return Err(Fault::Device {
                    call: "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                    code: code as i32,
                });
            }
            Ok(Module {
                module,
                function,
                block_threads,
                dynamic_shared,
                resident: u32::try_from(resident).unwrap_or(0),
                entry_name: entry_name.to_string(),
            })
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (entry_name, dynamic_shared);
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub const fn dynamic_shared(&self) -> u32 {
        self.dynamic_shared
    }

    #[cfg(feature = "device")]
    #[must_use]
    pub const fn function(&self) -> crate::device::dr::CUfunction {
        self.function
    }

    #[must_use]
    pub const fn block_threads(&self) -> u32 {
        self.block_threads
    }

    /// The blocks of it one multiprocessor holds at once.
    #[must_use]
    pub const fn resident(&self) -> u32 {
        self.resident
    }

    #[must_use]
    pub fn entry_name(&self) -> &str {
        &self.entry_name
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        #[cfg(feature = "device")]
        if !self.module.is_null() {
            // SAFETY: loaded in `load`, dropped once. The return code is
            // ignored because a `Drop` has nowhere to report it.
            unsafe { crate::device::dr::cuModuleUnload(self.module) };
        }
    }
}

#[cfg(feature = "device")]
fn launch_width(function: crate::device::dr::CUfunction) -> u32 {
    use crate::device::dr;

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

#[cfg_attr(not(feature = "device"), allow(dead_code))]
fn round_down_to_power_of_two(max_threads: i32) -> u32 {
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

const MAGIC: &[u8; 8] = b"PTRCUB02";

const HEADER_BYTES: usize = 8 + 4 + 4 + 8;

const MAX_ENTRY_BYTES: u64 = 128 * 1024 * 1024;

static NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct Disk {
    directory: Option<PathBuf>,
}

impl Disk {
    #[must_use]
    pub fn rooted(directory: Option<impl Into<PathBuf>>) -> Disk {
        Disk {
            directory: directory.map(Into::into),
        }
    }

    #[must_use]
    pub fn at(directory: impl Into<PathBuf>) -> Disk {
        Disk {
            directory: Some(directory.into()),
        }
    }

    #[must_use]
    pub const fn disabled() -> Disk {
        Disk { directory: None }
    }

    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.directory.is_some()
    }

    #[must_use]
    pub fn load(&self, key: &str, entry: &str) -> Option<Vec<u8>> {
        let path = self.path(key)?;
        let bytes = fs::read(&path).ok()?;
        match parse(&bytes, key, entry) {
            Some(cubin) => Some(cubin),
            None => {
                self.invalidate(key);
                None
            }
        }
    }

    pub fn store(&self, key: &str, entry: &str, cubin: &[u8]) {
        let Some(directory) = self.directory.as_ref() else {
            return;
        };
        if u32::try_from(key.len()).is_err() || u32::try_from(entry.len()).is_err() {
            return;
        }
        if fs::create_dir_all(directory).is_err() {
            return;
        }
        let Some(destination) = self.path(key) else {
            return;
        };

        let mut bytes = Vec::with_capacity(HEADER_BYTES + key.len() + entry.len() + cubin.len());
        bytes.extend_from_slice(MAGIC);
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

    pub fn invalidate(&self, key: &str) {
        if let Some(path) = self.path(key) {
            let _ = fs::remove_file(path);
        }
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        let directory = self.directory.as_ref()?;
        Some(directory.join(format!("{:016x}.cubin", eta_ir::fnv1a64(key.as_bytes()))))
    }
}

/// What a region's code is keyed by, in memory and on disk: its text and
/// what compiles it (the backend, the architecture, the runtime compiler's
/// version and options), never the stage it sits in, so regions the same
/// but for where they sit share one build.
#[must_use]
pub fn region_key(target: &Target, source: &str) -> String {
    let hash = fnv1a64_with(source.as_bytes(), &compile_options());
    let (major, minor) = target.rtc;
    format!(
        "{BACKEND:?}-{}-rtc{major}.{minor}-{hash:016x}",
        target.architecture()
    )
}

fn parse(bytes: &[u8], key: &str, entry: &str) -> Option<Vec<u8>> {
    if bytes.len() < HEADER_BYTES || bytes.len() as u64 > MAX_ENTRY_BYTES {
        return None;
    }
    if &bytes[..8] != MAGIC {
        return None;
    }
    let key_size = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let entry_size = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let cubin_size = u64::from_le_bytes(bytes[16..24].try_into().ok()?);

    if key_size != key.len() || entry_size != entry.len() {
        return None;
    }
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

#[derive(Debug)]
pub struct Region {
    pub region_index: u32,
    pub module: Arc<Module>,
}

#[derive(Debug, Clone)]
pub struct Stage {
    pub signature_hash: u64,
    pub regions: Arc<Vec<Region>>,
}

impl Stage {
    #[must_use]
    pub fn region(&self, region_index: u32) -> Option<&Region> {
        self.regions
            .iter()
            .find(|region| region.region_index == region_index)
    }
}

#[derive(Debug, Clone)]
pub struct Compiled {
    pub stages: Arc<Vec<Stage>>,
    pub plans: Arc<Vec<LaunchStagePlan>>,
    pub kinds: Arc<Vec<Attach>>,
}

#[derive(Debug)]
pub struct Cache {
    programs: Bounded<u64, Compiled>,
    stages: Stages<Stage>,
    negative: Bounded<u64, String>,
    /// The code each region source built, by its key under the region
    /// symbol: stages whose regions are the same text but for their entry
    /// names (a sampler region in every stage that samples) build it once.
    built: Bounded<String, Arc<Module>>,
    /// The modules the load's prewarm ([`Cache::warm`]) built, by the same
    /// key: held for the load's life, out of `built`'s bound, so a guest's
    /// registrations never evict a standard sampler's code.
    warmed: HashMap<String, Arc<Module>>,
    /// The region modules a registration loaded or compiled because no
    /// prewarm ([`Cache::warm`]) or earlier registration had: each one a
    /// module built after ready.
    late: u64,
    disk: Disk,
    stats: CacheStats,
}

impl Default for Cache {
    fn default() -> Cache {
        Cache::new(Disk::disabled())
    }
}

impl Cache {
    #[must_use]
    pub fn new(disk: Disk) -> Cache {
        Cache {
            programs: Bounded::new(MAX_PROGRAM_ENTRIES),
            stages: Stages::new(MAX_STAGE_ENTRIES),
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            built: Bounded::new(MAX_STAGE_ENTRIES),
            warmed: HashMap::new(),
            late: 0,
            disk,
            stats: CacheStats::default(),
        }
    }

    #[must_use]
    pub const fn disk(&self) -> &Disk {
        &self.disk
    }

    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// A program's registration, whole: [`Cache::plan`], the build, and
    /// [`Cache::install`] on one thread.
    pub fn compile(
        &mut self,
        program_hash: u64,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Compiled, Failure> {
        match self.plan(program_hash, plan, kernels, versions, target)? {
            Planned::Ready(compiled) => Ok(compiled),
            Planned::Build(mut build) => {
                let built = build.build();
                self.install(build, built)
            }
        }
    }

    /// What a program's registration takes, on the engine's thread: the
    /// program or its stages where this cache holds them, and the regions
    /// it must build, each with its source and key. The build itself
    /// ([`Build::build`]) holds nothing of the engine's and runs on any
    /// thread.
    pub fn plan(
        &mut self,
        program_hash: u64,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Planned, Failure> {
        if let Some(compiled) = self.programs.get(&program_hash) {
            self.stats.memory_hits += 1;
            return Ok(Planned::Ready(compiled.clone()));
        }

        let program_identity = cache_identity(
            BACKEND,
            target.device,
            combined_signature(&plan.package.plans),
            versions,
        );
        let program_key = eta_ir::fnv1a64(program_identity.as_bytes());
        if let Some(reason) = self.negative.get(&program_key) {
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic {
                reason: reason.clone(),
            });
        }

        let refused = |failure: Failure, cache: &mut Cache| {
            if let Failure::Deterministic { reason } = &failure {
                cache.negative.insert(program_key, reason.clone());
            }
            failure
        };
        let index = match Emitted::index(kernels) {
            Ok(index) => index,
            Err(duplicate) => {
                let failure = Failure::Deterministic {
                    reason: format!(
                        "the emitted kernel table names slot (kind {}, stage {}, region {}) \
                         twice; an engine cannot know which of the two the host meant",
                        duplicate.kind as u32, duplicate.stage, duplicate.region
                    ),
                };
                return Err(refused(failure, self));
            }
        };
        let mut stages = Vec::with_capacity(plan.package.plans.len());
        for (stage_index, stage_plan) in plan.package.plans.iter().enumerate() {
            let planned = self.plan_stage(stage_index, stage_plan, &index, versions, &target);
            stages.push(planned.map_err(|failure| refused(failure, self))?);
        }
        Ok(Planned::Build(Build {
            program_hash,
            program_key,
            architecture: target.architecture(),
            stages,
            plans: Arc::new(plan.package.plans.clone()),
            kinds: Arc::new(plan.package.stages.iter().map(|s| s.stage).collect()),
            disk: self.disk.clone(),
            persistent_hits: 0,
            compilations: 0,
        }))
    }

    fn plan_stage(
        &mut self,
        stage_index: usize,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
        versions: Versions,
        target: &Target,
    ) -> std::result::Result<StageBuild, Failure> {
        let stage_index = u32::try_from(stage_index).map_err(|_| Failure::Deterministic {
            reason: "a program with more than four billion stages is not a program".into(),
        })?;
        let identity = cache_identity(BACKEND, target.device, plan.signature_hash, versions);
        let key = fnv1a64_with(
            identity.as_bytes(),
            &[
                target.rtc.0.to_le_bytes().as_slice(),
                target.rtc.1.to_le_bytes().as_slice(),
            ],
        );
        let (lookup, hit) = self.stages.lookup(key, plan.identity);
        if lookup == Lookup::Hit {
            self.stats.memory_hits += 1;
            if let Some(stage) = hit {
                return Ok(StageBuild::Held(stage));
            }
        }
        let mut regions = Vec::new();
        for region_index in 0..plan.fused.len() {
            let region_index = u32::try_from(region_index).map_err(|_| Failure::Deterministic {
                reason: "a stage with more than four billion regions is not a stage".into(),
            })?;
            if plan
                .fused
                .get(region_index as usize)
                .is_some_and(|region| !super::launch::shell_launches(region))
            {
                continue;
            }
            let (source, entry) = match index.get(KERNEL_FUSED, stage_index, region_index) {
                Slot::Kernel { source, entry, .. } => (source, entry),
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
            let mut region = RegionBuild::new(target, region_index, entry, source);
            region.module = self.module(&region.key);
            if region.module.is_none() {
                self.late += 1;
            }
            regions.push(region);
        }
        Ok(StageBuild::Regions {
            key,
            identity: plan.identity,
            signature_hash: plan.signature_hash,
            cached: lookup == Lookup::Miss,
            regions,
        })
    }

    /// The region modules registrations built that the load's prewarm had
    /// not ([`Cache::warm`]).
    #[must_use]
    pub const fn built_late(&self) -> u64 {
        self.late
    }

    /// Builds the region modules of `kernels` (a load's prewarm, before
    /// ready): each region's text once, from the disk where it holds it,
    /// the rest compiled side by side; a registration whose regions are the
    /// same text then loads and compiles nothing.
    pub fn warm(
        &mut self,
        target: &Target,
        kernels: &[EmittedKernel],
    ) -> std::result::Result<u64, Failure> {
        let mut wanted: Vec<(String, String, u32)> = Vec::new();
        for kernel in kernels {
            if kernel.kind != KERNEL_FUSED || kernel.source.is_empty() {
                continue;
            }
            let source = kernel.source.replace(&kernel.entry_name, REGION_SYMBOL);
            let key = region_key(target, &source);
            if let Some(module) = self.built.remove(&key) {
                self.warmed.insert(key, module);
                continue;
            }
            if self.warmed.contains_key(&key) || wanted.iter().any(|(seen, _, _)| *seen == key) {
                continue;
            }
            wanted.push((key, source, kernel.region_index));
        }
        let mut cubins: Vec<Option<Vec<u8>>> = wanted
            .iter()
            .map(|(key, _, _)| {
                let cubin = self.disk.load(key, REGION_SYMBOL)?;
                late_memory(&cubin).ok()?;
                self.stats.persistent_hits += 1;
                Some(cubin)
            })
            .collect();
        let architecture = target.architecture();
        let compiled: Vec<(usize, std::result::Result<Vec<u8>, Failure>)> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = cubins
                    .iter()
                    .enumerate()
                    .filter(|(_, cubin)| cubin.is_none())
                    .map(|(at, _)| {
                        let source = &wanted[at].1;
                        let architecture = &architecture;
                        (at, scope.spawn(move || compile(source, architecture)))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|(at, handle)| {
                        let built = handle.join().unwrap_or_else(|_| {
                            Err(Failure::Retryable {
                                reason: "a region's compile panicked".into(),
                            })
                        });
                        (at, built)
                    })
                    .collect()
            });
        for (at, cubin) in compiled {
            let (key, _, region_index) = &wanted[at];
            let cubin = cubin?;
            self.stats.compilations += 1;
            late_memory(&cubin).map_err(|why| Failure::Deterministic {
                reason: format!("region {region_index}: {why}"),
            })?;
            self.disk.store(key, REGION_SYMBOL, &cubin);
            cubins[at] = Some(cubin);
        }
        for ((key, source, _), cubin) in wanted.iter().zip(cubins) {
            let Some(cubin) = cubin else { continue };
            let module =
                Module::load(&cubin, REGION_SYMBOL, cut_state_bytes(source)).map_err(|error| {
                    Failure::Retryable {
                        reason: format!("loading a prewarmed region: {error}"),
                    }
                })?;
            self.warmed.insert(key.clone(), Arc::new(module));
        }
        Ok(wanted.len() as u64)
    }

    /// The module this cache holds for the region text under `key`: a
    /// prewarmed one, else a registration's.
    fn module(&mut self, key: &str) -> Option<Arc<Module>> {
        self.warmed
            .get(key)
            .or_else(|| self.built.get(&key.to_owned()))
            .cloned()
    }

    /// A program's build made modules, on the engine's thread (a module
    /// loads into the context current there), and kept: its regions' code,
    /// its stages and the program. A deterministic refusal is remembered.
    pub fn install(
        &mut self,
        build: Build,
        built: std::result::Result<(), Failure>,
    ) -> std::result::Result<Compiled, Failure> {
        self.stats.persistent_hits += build.persistent_hits;
        self.stats.compilations += build.compilations;
        if let Err(failure) = built {
            if let Failure::Deterministic { reason } = &failure {
                self.negative.insert(build.program_key, reason.clone());
            }
            return Err(failure);
        }
        if let Some(compiled) = self.programs.get(&build.program_hash) {
            return Ok(compiled.clone());
        }
        let mut stages = Vec::with_capacity(build.stages.len());
        for stage in build.stages {
            match stage {
                StageBuild::Held(stage) => stages.push(stage),
                StageBuild::Regions {
                    key,
                    identity,
                    signature_hash,
                    cached,
                    regions,
                } => {
                    let mut modules = Vec::with_capacity(regions.len());
                    for region in regions {
                        match self.load(&region, &build.disk) {
                            Ok(module) => {
                                modules.push(Region {
                                    region_index: region.region_index,
                                    module,
                                });
                            }
                            Err(failure) => {
                                self.stages.abandon();
                                return Err(failure);
                            }
                        }
                    }
                    let stage = Stage {
                        signature_hash,
                        regions: Arc::new(modules),
                    };
                    if cached {
                        self.stages.stage(key, identity, stage.clone());
                    }
                    stages.push(stage);
                }
            }
        }
        self.stages.commit();
        let compiled = Compiled {
            stages: Arc::new(stages),
            plans: build.plans,
            kinds: build.kinds,
        };
        self.programs.insert(build.program_hash, compiled.clone());
        Ok(compiled)
    }

    pub fn forget(&mut self, program_hash: u64) {
        self.programs.remove(&program_hash);
    }

    /// One region's module, built whole on this thread.
    #[cfg(all(test, feature = "rocm"))]
    fn region_module(
        &mut self,
        target: &Target,
        region_index: u32,
        entry: &str,
        source: &str,
    ) -> std::result::Result<Arc<Module>, Failure> {
        let mut region = RegionBuild::new(target, region_index, entry, source);
        region.module = self.module(&region.key);
        if region.module.is_none() {
            self.late += 1;
        }
        let mut build = Build {
            program_hash: 0,
            program_key: 0,
            architecture: target.architecture(),
            stages: Vec::new(),
            plans: Arc::new(Vec::new()),
            kinds: Arc::new(Vec::new()),
            disk: self.disk.clone(),
            persistent_hits: 0,
            compilations: 0,
        };
        let built = build.build_region(&mut region, &mut std::collections::HashMap::new());
        self.stats.persistent_hits += build.persistent_hits;
        self.stats.compilations += build.compilations;
        built?;
        let disk = self.disk.clone();
        self.load(&region, &disk)
    }

    /// `region`'s module: the one this cache holds for its text, else
    /// loaded from the code its build made and kept under its key.
    fn load(
        &mut self,
        region: &RegionBuild,
        disk: &Disk,
    ) -> std::result::Result<Arc<Module>, Failure> {
        if let Some(module) = &region.module {
            return Ok(Arc::clone(module));
        }
        if let Some(module) = self.module(&region.key) {
            return Ok(module);
        }
        let module = region.load(disk)?;
        self.built.insert(region.key.clone(), Arc::clone(&module));
        Ok(module)
    }
}

/// What [`Cache::plan`] found.
#[derive(Debug)]
pub enum Planned {
    /// The program, as this cache holds it.
    Ready(Compiled),
    /// What its registration builds.
    Build(Build),
}

/// A program's build: the regions [`Cache::plan`] found no code for in
/// memory, compiled or read from disk by [`Build::build`] on any thread,
/// then loaded and kept by [`Cache::install`] on the engine's.
#[derive(Debug)]
pub struct Build {
    program_hash: u64,
    program_key: u64,
    architecture: String,
    stages: Vec<StageBuild>,
    plans: Arc<Vec<LaunchStagePlan>>,
    kinds: Arc<Vec<Attach>>,
    disk: Disk,
    persistent_hits: u64,
    compilations: u64,
}

#[derive(Debug)]
enum StageBuild {
    /// A stage this cache holds.
    Held(Stage),
    Regions {
        key: u64,
        identity: u64,
        signature_hash: u64,
        /// The stage cache had no entry at its key: the stage is kept.
        cached: bool,
        regions: Vec<RegionBuild>,
    },
}

/// Where a region's code came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Memory,
    Disk,
    Compiled,
}

#[derive(Debug)]
struct RegionBuild {
    region_index: u32,
    /// The region's own name.
    entry: String,
    /// Its source under [`REGION_SYMBOL`].
    source: String,
    key: String,
    /// The dynamic shared memory each launch hands it.
    dynamic: u32,
    /// The module the cache holds for its text, where it holds one: the
    /// region builds nothing.
    module: Option<Arc<Module>>,
    code: Option<(Arc<[u8]>, Origin)>,
}

impl RegionBuild {
    fn new(target: &Target, region_index: u32, entry: &str, source: &str) -> RegionBuild {
        // The region is built under one symbol, so its code is keyed by
        // what it computes, never by the stage it sits in: stages whose
        // regions are the same text share one module.
        let source = source.replace(entry, REGION_SYMBOL);
        let key = region_key(target, &source);
        let dynamic = cut_state_bytes(&source);
        RegionBuild {
            region_index,
            entry: entry.to_owned(),
            source,
            key,
            dynamic,
            module: None,
            code: None,
        }
    }

    /// Its module, loaded into the context current on this thread from
    /// the code its build made. Code read from disk that does not load is
    /// dropped from the disk, so the next registration compiles it.
    fn load(&self, disk: &Disk) -> std::result::Result<Arc<Module>, Failure> {
        let Some((code, origin)) = self.code.clone() else {
            return Err(Failure::Retryable {
                reason: format!("region `{}` was never built", self.entry),
            });
        };
        let module = Module::load(&code, REGION_SYMBOL, self.dynamic).map_err(|error| {
            if origin == Origin::Disk {
                disk.invalidate(&self.key);
            }
            Failure::Retryable {
                reason: format!("loading `{}`: {error}", self.entry),
            }
        })?;
        Ok(Arc::new(module))
    }
}

impl Build {
    /// Compiles, or reads from disk, every region's code the plan found
    /// none for. It holds nothing of the engine's and needs no context, so
    /// it runs on any thread while the engine fires on. Regions the same
    /// but for their names build once.
    pub fn build(&mut self) -> std::result::Result<(), Failure> {
        let mut stages = std::mem::take(&mut self.stages);
        let mut made = std::collections::HashMap::new();
        let built = stages.iter_mut().try_for_each(|stage| match stage {
            StageBuild::Held(_) => Ok(()),
            StageBuild::Regions { regions, .. } => regions
                .iter_mut()
                .try_for_each(|region| self.build_region(region, &mut made)),
        });
        self.stages = stages;
        built
    }

    fn build_region(
        &mut self,
        region: &mut RegionBuild,
        made: &mut std::collections::HashMap<String, Arc<[u8]>>,
    ) -> std::result::Result<(), Failure> {
        if region.module.is_some() || region.code.is_some() {
            return Ok(());
        }
        if let Some(code) = made.get(&region.key) {
            region.code = Some((Arc::clone(code), Origin::Memory));
            return Ok(());
        }
        let code = match self.disk.load(&region.key, REGION_SYMBOL) {
            // One that asks for late memory was compiled before that was
            // refused: compiled again, and refused below if it still asks.
            Some(code) if late_memory(&code).is_ok() => {
                self.persistent_hits += 1;
                (Arc::<[u8]>::from(code), Origin::Disk)
            }
            stale => {
                if stale.is_some() {
                    self.disk.invalidate(&region.key);
                }
                let code = compile(&region.source, &self.architecture)?;
                self.compilations += 1;
                late_memory(&code).map_err(|why| Failure::Deterministic {
                    reason: format!("region {} (`{}`): {why}", region.region_index, region.entry),
                })?;
                self.disk.store(&region.key, REGION_SYMBOL, &code);
                (Arc::<[u8]>::from(code), Origin::Compiled)
            }
        };
        made.insert(region.key.clone(), Arc::clone(&code.0));
        region.code = Some(code);
        Ok(())
    }
}

/// The dynamic shared memory a region's text asks: CUDA keeps the exact
/// cut's state there; HIP's text keeps it static.
fn cut_state_bytes(source: &str) -> u32 {
    if !cfg!(feature = "rocm") && source.contains(eta_compiler::codegen::cuda::CUT_STATE_MARK) {
        eta_compiler::codegen::cuda::CUT_STATE_BYTES
    } else {
        0
    }
}

/// The symbol every region's kernel is built under.
const REGION_SYMBOL: &str = "ptir_region";

/// Refuses a region's code object whose kernel asks the HIP runtime for
/// memory it allocates late: a buffer at its first launch, or scratch, which
/// ROCr grows at the first dispatch that needs it. Either is a device
/// allocation after the load, so a region is refused at registration, as the
/// engine's own kernels are at theirs, never at its first fire. CUDA's
/// modules ask for neither.
fn late_memory(cubin: &[u8]) -> std::result::Result<(), String> {
    #[cfg(feature = "rocm")]
    kernels_cuda::hip::code_object::refuse_late_buffers(cubin)?;
    #[cfg(not(feature = "rocm"))]
    let _ = cubin;
    Ok(())
}

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

    #[test]
    fn compile_every_case() {
        the_region_key_is_its_text_and_what_compiles_it();
        a_corrupt_entry_is_a_miss_and_is_deleted();
        a_disabled_cache_is_a_miss_and_not_a_failure();
        #[cfg(feature = "rocm")]
        {
            regions_the_same_but_for_their_names_are_built_once();
            a_region_that_asks_for_scratch_is_refused_by_name();
        }
    }

    #[cfg(feature = "rocm")]
    fn device_target() -> Target {
        Target {
            major: 9,
            minor: 4,
            device: 0,
            rtc: rtc_version().expect("hiprtc answers its version"),
            arch: kernels_cuda::jit::cache::arch(0).expect("a gfx device"),
        }
    }

    #[cfg(feature = "rocm")]
    fn regions_the_same_but_for_their_names_are_built_once() {
        let region = |entry: &str| {
            format!(
                "extern \"C\" __global__ void {entry}(unsigned* out) {{ out[threadIdx.x] = 1u; }}"
            )
        };
        let directory = scratch("shared");
        let target = device_target();
        // Two stages' regions: other names, other region indices.
        let (a, b) = (
            "ptir_fused_00000000000000aa_r0",
            "ptir_fused_00000000000000bb_r3",
        );
        let mut cache = Cache::new(Disk::at(&directory));
        let first = cache
            .region_module(&target, 0, a, &region(a))
            .expect("builds");
        let second = cache
            .region_module(&target, 3, b, &region(b))
            .expect("builds");
        assert_eq!(
            cache.stats().compilations,
            1,
            "the second stage's region is the first's code"
        );
        assert!(Arc::ptr_eq(&first, &second), "and the first's module");

        // A later boot finds it on disk under a third name and index.
        let c = "ptir_fused_00000000000000cc_r5";
        let mut later = Cache::new(Disk::at(&directory));
        let _third = later
            .region_module(&target, 5, c, &region(c))
            .expect("loads");
        assert_eq!(
            (later.stats().compilations, later.stats().persistent_hits),
            (0, 1),
            "the disk entry is the region's text's, not its stage's"
        );
        let _ = fs::remove_dir_all(&directory);
    }

    #[cfg(feature = "rocm")]
    fn a_region_that_asks_for_scratch_is_refused_by_name() {
        // A local array read at a lane's own index lives in private memory.
        let entry = "ptir_fused_00000000000000cc_r1";
        let source = format!(
            "extern \"C\" __global__ void {entry}(const unsigned* in, unsigned* out) {{\n\
             \x20 unsigned local[1024];\n\
             \x20 for (unsigned i = 0; i < 1024u; ++i) local[i] = in[i];\n\
             \x20 out[threadIdx.x] = local[in[threadIdx.x] & 1023u];\n\
             }}"
        );
        let mut cache = Cache::new(Disk::disabled());
        match cache.region_module(&device_target(), 1, entry, &source) {
            Err(Failure::Deterministic { reason }) => assert!(
                reason.contains(entry) && reason.contains("scratch"),
                "the refusal names the region and what it asks: {reason}"
            ),
            Err(other) => panic!("refused as retryable, which would compile it again: {other:?}"),
            Ok(_) => panic!("a region that asks for scratch was registered"),
        }
    }

    fn the_region_key_is_its_text_and_what_compiles_it() {
        let target = Target {
            major: 8,
            minor: 9,
            device: 0,
            rtc: (12, 4),
            #[cfg(feature = "rocm")]
            arch: "gfx942",
        };
        let before = region_key(&target, "__global__ void k() { a(); }");
        let after = region_key(&target, "__global__ void k() { b(); }");
        assert_ne!(
            before, after,
            "a template edit bumps no version, so the source itself has to be \
             in the key — otherwise yesterday's cubin answers today's launch"
        );
        let other_device = Target {
            device: 1,
            ..target
        };
        assert_eq!(
            before,
            region_key(&other_device, "__global__ void k() { a(); }"),
            "a second card of the same architecture shares the build"
        );
        let newer = Target {
            rtc: (12, 5),
            ..target
        };
        assert_ne!(
            before,
            region_key(&newer, "__global__ void k() { a(); }"),
            "another runtime compiler builds it again"
        );
        assert!(
            before.contains(&target.architecture()),
            "the target stays readable"
        );
    }

    fn scratch(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("pie-ptir-disk-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        path
    }

    fn a_corrupt_entry_is_a_miss_and_is_deleted() {
        let directory = scratch("corrupt");
        let disk = Disk::at(&directory);
        disk.store("key-a", "entry", b"cubin-bytes");
        let path = disk.path("key-a").expect("enabled");
        let good = fs::read(&path).expect("stored");

        fs::write(&path, &good[..good.len() - 3]).expect("truncate");
        assert_eq!(disk.load("key-a", "entry"), None, "a short tail misses");
        assert!(!path.exists(), "and the entry is removed");

        disk.store("key-a", "entry", b"cubin-bytes");
        let mut wrong_magic = fs::read(&path).expect("stored");
        wrong_magic[7] = b'9';
        fs::write(&path, &wrong_magic).expect("write");
        assert_eq!(disk.load("key-a", "entry"), None, "a format bump misses");
        assert!(!path.exists());
    }

    fn a_disabled_cache_is_a_miss_and_not_a_failure() {
        let disk = Disk::disabled();
        assert!(!disk.is_enabled());
        disk.store("key", "entry", b"cubin");
        assert_eq!(disk.load("key", "entry"), None);
    }
}
