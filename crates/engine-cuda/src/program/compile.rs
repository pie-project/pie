#![allow(unsafe_code)]
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use eta_compiler::codegen::cuda_steps;
use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{
    Backend, Bounded, CacheStats, ExecPlan, Failure, MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES,
};

use crate::error::{Fault, Result};
use eta_ir::registry::Stage as Attach;

#[cfg_attr(not(feature = "device"), allow(dead_code))]
const DEFAULT_BLOCK_THREADS: u32 = 256;

#[cfg_attr(not(feature = "device"), allow(dead_code))]
const WARP: u32 = 32;

#[cfg_attr(not(feature = "device"), allow(dead_code))]
const MAX_BLOCK_THREADS: u32 = 1024;

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
    return HIPRTC_OPTIONS
        .iter()
        .map(|option| option.as_bytes())
        .collect();
    #[cfg(feature = "cuda")]
    return NVRTC_OPTIONS
        .iter()
        .map(|option| option.to_bytes())
        .collect();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    Vec::new()
}

pub fn compile(source: &str, architecture: &str) -> std::result::Result<Vec<u8>, Failure> {
    #[cfg(feature = "rocm")]
    {
        use kernels_cuda::hip::{self, CompileError};
        match hip::compile(
            "ptir_fused_region",
            source,
            architecture,
            &HIPRTC_OPTIONS,
            &[],
        ) {
            Ok(built) if built.code.is_empty() => Err(Failure::Retryable {
                reason: "hiprtc produced an empty code object".into(),
            }),
            Ok(built) => Ok(built.code),
            Err(CompileError::Rejected(log)) => Err(Failure::Deterministic {
                reason: format!("hiprtc refused the emitted region: {log}"),
            }),
            Err(CompileError::Failed(reason)) => Err(Failure::Retryable { reason }),
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
    /// The blocks of `RESIDENT_THREADS` threads a multiprocessor holds at
    /// once of it (at least one).
    resident: u32,
    entry_name: String,
}

/// The block width [`Module::resident`] counts blocks of: a stage
/// kernel's (`cuda_steps::BLOCK`).
#[cfg_attr(not(feature = "device"), allow(dead_code))]
const RESIDENT_THREADS: u32 = 256;

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
            Ok(Module {
                module,
                function,
                block_threads: launch_width(function),
                dynamic_shared,
                resident: resident_blocks(function, dynamic_shared),
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

    /// The blocks of 256 threads a multiprocessor holds at once of this
    /// module's entry: what a launch whose blocks wait on one another may
    /// run a unit.
    #[must_use]
    pub const fn resident(&self) -> u32 {
        self.resident
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
fn resident_blocks(function: crate::device::dr::CUfunction, dynamic_shared: u32) -> u32 {
    #[cfg(not(feature = "rocm"))]
    {
        use crate::device::dr;
        let mut blocks = 0i32;
        // SAFETY: `blocks` is a live out-parameter and `function` was just
        // resolved out of a loaded module.
        let code = unsafe {
            dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &raw mut blocks,
                function,
                RESIDENT_THREADS as i32,
                dynamic_shared as usize,
            )
        };
        if code == dr::CUresult::CUDA_SUCCESS {
            return u32::try_from(blocks).unwrap_or(1).max(1);
        }
        1
    }
    #[cfg(feature = "rocm")]
    {
        let _ = (function, dynamic_shared, RESIDENT_THREADS);
        1
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

#[derive(Debug, Clone)]
pub struct Stage {
    pub signature_hash: u64,
    /// The stage kernel, which interprets every stage of every program.
    pub module: Arc<Module>,
    /// The stage's steps, in order, as its table holds them
    /// (`cuda_steps::record`): none for a stage the device runs nothing for.
    pub steps: Arc<Vec<[u32; 4]>>,
}

impl Stage {
    /// The bytes of the stage's step table, as a bind writes it: each
    /// step's record, its node's temporary area (`temporaries`, a bind's)
    /// in it.
    #[must_use]
    pub fn table(&self, temporaries: &[u32]) -> Vec<u8> {
        self.steps
            .iter()
            .flat_map(|&[call, node, rung, selector]| {
                let temporary = temporaries.get(node as usize).copied().unwrap_or(0);
                [call, node, rung, selector, temporary, 0, 0, 0]
            })
            .flat_map(u32::to_le_bytes)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Compiled {
    pub stages: Arc<Vec<Stage>>,
    pub plans: Arc<Vec<LaunchStagePlan>>,
    pub kinds: Arc<Vec<Attach>>,
}

impl Compiled {
    /// The steps each stage runs.
    #[must_use]
    pub fn step_counts(&self) -> Vec<usize> {
        self.stages.iter().map(|stage| stage.steps.len()).collect()
    }
}

/// The closed set (`eta_compiler::codegen::cuda_steps`) as one kernel that
/// interprets a stage, built once for the engine at load ([`Cache::arm`]),
/// and the programs registered over it. A program names no kernel of its
/// own: its stages' steps index the set, so a registration builds nothing.
#[derive(Debug)]
pub struct Cache {
    programs: Bounded<u64, Compiled>,
    negative: Bounded<u64, String>,
    stage: Option<Arc<Module>>,
    disk: Disk,
    stats: CacheStats,
    /// The kernels a registration found unbuilt and built: after ready
    /// when the load armed the set, which is the rule that this reads 0.
    late: u64,
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
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            stage: None,
            disk,
            stats: CacheStats::default(),
            late: 0,
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

    /// The kernels a registration built because the load had not.
    #[must_use]
    pub const fn built_late(&self) -> u64 {
        self.late
    }

    /// Builds the stage kernel for `target`, or reads it from the disk:
    /// what a load does before ready.
    pub fn arm(&mut self, target: &Target) -> std::result::Result<(), Failure> {
        self.stage_module(target).map(|_| ())
    }

    fn stage_module(&mut self, target: &Target) -> std::result::Result<Arc<Module>, Failure> {
        if let Some(module) = &self.stage {
            return Ok(Arc::clone(module));
        }
        let entry = cuda_steps::ENTRY;
        let source = cuda_steps::source();
        let key = region_key(target, &source);
        let cubin = match self.disk.load(&key, entry) {
            Some(cubin) if late_memory(&cubin).is_ok() => {
                self.stats.persistent_hits += 1;
                cubin
            }
            stale => {
                if stale.is_some() {
                    self.disk.invalidate(&key);
                }
                let cubin = compile(&source, &target.architecture())?;
                self.stats.compilations += 1;
                late_memory(&cubin).map_err(|why| Failure::Deterministic {
                    reason: format!("`{entry}`: {why}"),
                })?;
                self.disk.store(&key, entry, &cubin);
                cubin
            }
        };
        let module = Module::load(&cubin, entry, 0).map_err(|error| Failure::Retryable {
            reason: format!("loading `{entry}`: {error}"),
        })?;
        let module = Arc::new(module);
        self.stage = Some(Arc::clone(&module));
        Ok(module)
    }

    /// The program's stages as steps of the set. A stage kernel the load
    /// did not build is built here and counted ([`Cache::built_late`]).
    pub fn compile(
        &mut self,
        program_hash: u64,
        plan: &ExecPlan,
        target: Target,
    ) -> std::result::Result<Compiled, Failure> {
        if let Some(compiled) = self.programs.get(&program_hash) {
            self.stats.memory_hits += 1;
            return Ok(compiled.clone());
        }
        if let Some(reason) = self.negative.get(&program_hash) {
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic {
                reason: reason.clone(),
            });
        }
        match self.stages(plan, &target) {
            Ok(stages) => {
                let compiled = Compiled {
                    stages: Arc::new(stages),
                    plans: Arc::new(plan.package.plans.clone()),
                    kinds: Arc::new(plan.package.stages.iter().map(|s| s.stage).collect()),
                };
                self.programs.insert(program_hash, compiled.clone());
                Ok(compiled)
            }
            Err(failure) => {
                if let Failure::Deterministic { reason } = &failure {
                    self.negative.insert(program_hash, reason.clone());
                }
                Err(failure)
            }
        }
    }

    fn stages(
        &mut self,
        plan: &ExecPlan,
        target: &Target,
    ) -> std::result::Result<Vec<Stage>, Failure> {
        if self.stage.is_none() {
            self.stage_module(target)?;
            self.late += 1;
        }
        let module = self.stage.clone().ok_or_else(|| Failure::Retryable {
            reason: "the stage kernel is not built".into(),
        })?;
        let set = cuda_steps::kernels();
        let mut stages = Vec::with_capacity(plan.package.plans.len());
        for (index, stage_plan) in plan.package.plans.iter().enumerate() {
            let refused = |tag: u8| Failure::Deterministic {
                reason: format!(
                    "stage {index} runs `{}` (tag {tag:#04x}), which no kernel of this \
                     engine's closed set serves; a guest pass runs on the device whole \
                     or not at all",
                    eta_ir::op::spec(tag).map_or("?", |row| row.name)
                ),
            };
            let steps = cuda_steps::stage_steps(stage_plan)
                .map_err(refused)?
                .iter()
                .map(|step| {
                    cuda_steps::record(step, &set)
                        .ok_or_else(|| refused(stage_plan.ops[step.uniform.node as usize].tag))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            stages.push(Stage {
                signature_hash: stage_plan.signature_hash,
                module: Arc::clone(&module),
                steps: Arc::new(steps),
            });
        }
        Ok(stages)
    }

    pub fn forget(&mut self, program_hash: u64) {
        self.programs.remove(&program_hash);
    }
}

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
            the_closed_set_builds_once_and_a_later_boot_reads_it();
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
    fn the_closed_set_builds_once_and_a_later_boot_reads_it() {
        let directory = scratch("closed");
        let target = device_target();
        let mut cache = Cache::new(Disk::at(&directory));
        cache.arm(&target).expect("the set builds");
        assert_eq!(cache.stats().compilations, 1);
        let mut later = Cache::new(Disk::at(&directory));
        later.arm(&target).expect("the set loads");
        assert_eq!(
            (later.stats().compilations, later.stats().persistent_hits),
            (0, 1),
            "a later boot compiles nothing"
        );
        let _ = fs::remove_dir_all(&directory);
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
