use std::fmt::Write as _;
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

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const DEFAULT_BLOCK_THREADS: u32 = 256;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const WARP: u32 = 32;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const MAX_BLOCK_THREADS: u32 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub major: i32,
    pub minor: i32,
    pub device: u64,
    pub nvrtc: (i32, i32),
}

impl Target {
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

#[must_use]
pub fn arch_flag(major: i32, minor: i32) -> String {
    format!("--gpu-architecture=sm_{major}{minor}")
}

pub fn nvrtc_version() -> Result<(i32, i32)> {
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
    #[cfg(not(feature = "cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

pub fn compile(source: &str, architecture: &str) -> std::result::Result<Vec<u8>, Failure> {
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
    #[cfg(not(feature = "cuda"))]
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
    #[cfg(feature = "cuda")]
    module: cudarc::driver::sys::CUmodule,
    #[cfg(feature = "cuda")]
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
    pub fn load(cubin: &[u8], entry_name: &str) -> Result<Module> {
        if cubin.is_empty() {
            return Err(Fault::program("cuModuleLoadData", "the cubin is empty"));
        }
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = entry_name;
            Err(Fault::Runtimeless)
        }
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub const fn function(&self) -> cudarc::driver::sys::CUfunction {
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
        #[cfg(feature = "cuda")]
        if !self.module.is_null() {
            // SAFETY: loaded in `load`, dropped once. The return code is
            // ignored because a `Drop` has nowhere to report it.
            unsafe { cudarc::driver::sys::cuModuleUnload(self.module) };
        }
    }
}

#[cfg(feature = "cuda")]
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

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
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

const MAGIC: &[u8; 8] = b"PTRCUB01";

const HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 8;

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

    pub fn invalidate(&self, key: &str, region_index: u32) {
        if let Some(path) = self.path(key, region_index) {
            let _ = fs::remove_file(path);
        }
    }

    fn path(&self, key: &str, region_index: u32) -> Option<PathBuf> {
        let directory = self.directory.as_ref()?;
        Some(directory.join(format!(
            "{:016x}-{region_index}.cubin",
            eta_ir::fnv1a64(key.as_bytes())
        )))
    }
}

#[must_use]
pub fn disk_key(identity: &str, source: &str) -> String {
    let hash = eta_ir::fnv1a64(source.as_bytes());
    let mut key = String::with_capacity(identity.len() + 16);
    key.push_str(identity);
    for byte in hash.to_le_bytes() {
        let _ = write!(key, "{byte:02x}");
    }
    key
}

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

impl Compiled {
    #[must_use]
    pub fn stage_of_kind(&self, kind: Attach) -> Option<usize> {
        self.kinds.iter().position(|&k| k == kind)
    }
}

#[derive(Debug)]
pub struct Cache {
    programs: Bounded<u64, Compiled>,
    stages: Stages<Stage>,
    negative: Bounded<u64, String>,
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
        let program_key = eta_ir::fnv1a64(program_identity.as_bytes());
        if let Some(reason) = self.negative.get(&program_key) {
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic {
                reason: reason.clone(),
            });
        }

        match self.build(plan, kernels, versions, target) {
            Ok(compiled) => {
                self.stages.commit();
                self.programs.insert(program_hash, compiled.clone());
                Ok(compiled)
            }
            Err(failure) => {
                self.stages.abandon();
                if let Failure::Deterministic { reason } = &failure {
                    self.negative.insert(program_key, reason.clone());
                }
                Err(failure)
            }
        }
    }

    pub fn forget(&mut self, program_hash: u64) {
        self.programs.remove(&program_hash);
    }

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
            kinds: Arc::new(plan.package.stages.iter().map(|s| s.stage).collect()),
        })
    }

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
                Err(_) => self.disk.invalidate(&key, region_index),
            }
        }

        let cubin = compile(source, architecture)?;
        self.stats.compilations += 1;
        let module = Module::load(&cubin, entry).map_err(|error| Failure::Retryable {
            reason: format!("loading `{entry}`: {error}"),
        })?;
        self.disk.store(&key, region_index, entry, &cubin);
        Ok(Arc::new(module))
    }
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
        editing_the_source_changes_the_disk_key_with_no_version_bump();
        a_corrupt_entry_is_a_miss_and_is_deleted();
        a_disabled_cache_is_a_miss_and_not_a_failure();
    }

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

    fn a_disabled_cache_is_a_miss_and_not_a_failure() {
        let disk = Disk::disabled();
        assert!(!disk.is_enabled());
        disk.store("key", 0, "entry", b"cubin");
        assert_eq!(disk.load("key", 0, "entry"), None);
    }

}
