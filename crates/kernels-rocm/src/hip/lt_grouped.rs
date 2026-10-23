//! hipBLASLt's grouped-GEMM kernels, launched from Rust without
//! `hipblaslt_ext::GroupedGemm` (a C++ class): the routed experts of a
//! prefill as one launch whose per-group sizes and pointers live in device
//! memory, so a routing kernel writes them and a graph can hold the launch.
//!
//! The logic file (zlib-compressed msgpack) names the grouped solutions and
//! their tiles; the code object (a compressed offload bundle) loads through
//! `hipModuleLoadData`. The arguments are built as hipBLASLt 1.4
//! (rocm-libraries 39d8d603) builds them for `run(deviceUserArgs)`:
//!
//! - kernarg, 144 bytes, of which the kernel reads 40 in its user-argument
//!   mode (ArgType 2): `[0]` gemm_count | 2 << 30, `[4]` internalArgs (GSU,
//!   StaggerU), `[8]` internalArgs1 (WGM, WGMXCC, WGMXCCG), `[12]` numWG,
//!   `[16]` the `UserArguments` array, `[24]` Synchronizer, `[32]` Workspace
//!   (tensilelite/src/ContractionSolution.cpp, kernelArgs and
//!   generateSingleCallGroupedGemm).
//! - `UserArguments`: 196 packed bytes a group (m, n, batch, k; d, c, a, b;
//!   eight strides; alpha, beta as 16 bytes each; epilogue fields, zero
//!   here) — library/include/hipblaslt/hipblaslt-ext.hpp.
//! - grid: one dimension of `prod(workGroup)`-lane workgroups. Each walks
//!   the groups' m/n/batch on device to find its own tile, and one that
//!   lands in a group of n = 0 ends (KernelWriterAssembly.py, the grouped
//!   linear search and `EarlyStop_if_N_is_0`). A grid sized for the worst
//!   case with an n = 0 sentinel group after the real ones needs no host
//!   knowledge of the counts.
//!
//! hipBLASLt's own grouped heuristic offers nothing for these libraries (a
//! `FreeSize` table); which tile a shape takes is the device profile's
//! (`kernel.moe.lt_tile_*`).
//!
//! Guards: the hipBLASLt version must be a known major.minor, and only
//! solutions whose argument contract matches the one above are offered.
//! `tests/lt_grouped.rs` checks the results on a ROCm bump.
#![allow(unsafe_code)]

use core::ffi::{c_int, c_void};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use dtype::Dtype;

use super::sys::{self, Stream};

/// The hipBLASLt release whose argument contract this module speaks.
const KNOWN_MAJOR: i32 = 1;
const KNOWN_MINOR: i32 = 4;

/// One `UserArguments` record.
pub const USER_ARGS_BYTES: usize = 196;
/// The grouped kernels' kernarg segment.
const KERNARG_BYTES: usize = 144;

/// One grouped solution: its kernel and what a launch derives from it.
#[derive(Clone, Debug)]
pub struct Solution {
    pub index: i64,
    pub kernel: String,
    /// MT0 (the weight's rows a tile), MT1 (the routes a tile), depth U.
    pub mt: [u32; 3],
    pub lanes: u32,
    stagger_u: u32,
    stagger_mapping: u32,
    stagger_shift: u32,
    wgm: i32,
    wgmxcc: i32,
    wgmxccg: i32,
}

impl Solution {
    /// The tile, for logs: `MT256x192x64 WGM8`.
    #[must_use]
    pub fn label(&self) -> String {
        format!(
            "#{} MT{}x{}x{} WGM{} SU{}",
            self.index, self.mt[0], self.mt[1], self.mt[2], self.wgm, self.stagger_u
        )
    }

    /// `internalArgs` and `internalArgs1` (kernelArgs at version 2, no SFC,
    /// GSU 1, the old WGMXCC layout).
    fn internal_args(&self, cus: u32) -> (u32, u32) {
        let su = (self.stagger_u & 0xff)
            | ((self.stagger_shift << 8) & 0x1f00)
            | ((self.stagger_mapping & 7) << 13);
        let arg0 = 1 | (su << 16);
        // WGM 0 asks for ⌈√(CUs / XCDs)⌉, 7 on MI300X's 304 over 8.
        let wgm = if self.wgm == 0 { 7 } else { self.wgm };
        let xccg = if self.wgmxcc >= 1 && self.wgmxccg == -1 {
            cus as i32
        } else {
            self.wgmxccg
        };
        let arg1 = ((xccg as u32) << 22) | ((self.wgmxcc as u32) << 16) | (wgm as u32 & 0xffff);
        (arg0, arg1)
    }

    /// The workgroups a launch over `rows` routes in `groups` groups of
    /// `n`-row weights can need at most: every route in whole tiles, and a
    /// partial tile a group.
    #[must_use]
    pub fn worst_grid(&self, n: u32, rows: u32, groups: u32) -> u32 {
        n.div_ceil(self.mt[0]) * (rows.div_ceil(self.mt[1]) + groups)
    }
}

/// A loaded grouped library: its launchable solutions and its code object.
pub struct Library {
    pub dtype: Dtype,
    pub solutions: Vec<Solution>,
    module: sys::Module,
    cus: u32,
    functions: Mutex<HashMap<usize, usize>>,
}

// SAFETY: the module handle is only read (hipModuleGetFunction is
// thread-safe); the function cache sits behind its lock.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

/// The grouped library for `dtype` (bf16 or f16, the weight `[n, k]` rows,
/// f32 accumulation) on the current device, opened once; `Err` says why it
/// is not offered, and keeps saying it.
pub fn library(dtype: Dtype) -> Result<&'static Library, String> {
    type Slot = &'static Result<Library, String>;
    static LIBRARIES: OnceLock<Mutex<HashMap<(c_int, Dtype), Slot>>> = OnceLock::new();
    let device = current_device();
    let mut map = LIBRARIES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lock poisoned");
    let slot = *map.entry((device, dtype)).or_insert_with(|| {
        let opened = Library::open(device, dtype);
        match &opened {
            Ok(lib) => tracing::info!(
                ?dtype,
                solutions = lib.solutions.len(),
                "opened hipBLASLt's grouped GEMM library"
            ),
            Err(why) => tracing::warn!(?dtype, why, "hipBLASLt's grouped GEMM is not offered"),
        }
        Box::leak(Box::new(opened))
    });
    slot.as_ref().map_err(Clone::clone)
}

/// The hipBLASLt version (`major·100000 + minor·100 + patch`), once.
pub fn version() -> Result<i32, String> {
    static VERSION: OnceLock<Result<i32, String>> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            let lt = sys::hipblaslt()?;
            let mut handle: sys::Lt = core::ptr::null_mut();
            let mut version: c_int = 0;
            // SAFETY: each call writes the out-pointer it is handed; the
            // handle is destroyed before returning.
            unsafe {
                if (lt.hipblasLtCreate)(&raw mut handle) != sys::LT_SUCCESS {
                    return Err("hipblasLtCreate failed".into());
                }
                let status = (lt.hipblasLtGetVersion)(handle, &raw mut version);
                (lt.hipblasLtDestroy)(handle);
                if status != sys::LT_SUCCESS {
                    return Err(format!("hipblasLtGetVersion answered {status}"));
                }
            }
            Ok(version)
        })
        .clone()
}

impl Library {
    fn open(device: c_int, dtype: Dtype) -> Result<Self, String> {
        let version = version()?;
        let (major, minor) = (version / 100_000, version / 100 % 1000);
        if (major, minor) != (KNOWN_MAJOR, KNOWN_MINOR) {
            return Err(format!(
                "hipBLASLt {major}.{minor} (version {version}); the grouped arguments are \
                 known for {KNOWN_MAJOR}.{KNOWN_MINOR}"
            ));
        }
        let tag = match dtype {
            Dtype::Bf16 => "BB",
            Dtype::F16 => "HH",
            other => return Err(format!("no grouped library for {other:?}")),
        };
        let arch = crate::jit::cache::arch(device)
            .ok_or("the device named no architecture")?
            .split(':')
            .next()
            .unwrap_or_default();
        let stem = format!(
            "TensileLibrary_{tag}_{tag}_HA_Bias_GG_SAV_UA_Type_{tag}_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_{arch}"
        );
        let dir = library_dirs()
            .into_iter()
            .map(|d| d.join(arch))
            .find(|d| d.join(format!("{stem}.co")).is_file())
            .ok_or_else(|| format!("no {stem}.co under hipBLASLt's library"))?;
        let packed = std::fs::read(dir.join(format!("{stem}.dat.zlib")))
            .map_err(|e| format!("{stem}.dat.zlib: {e}"))?;
        let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&packed)
            .map_err(|e| format!("{stem}.dat.zlib does not inflate: {e:?}"))?;
        let logic = Reader { b: &raw, at: 0 }
            .value()
            .ok_or_else(|| format!("{stem}.dat.zlib is not the msgpack expected"))?;
        let solutions = solutions(&logic, dtype);
        if solutions.is_empty() {
            return Err(format!("{stem} offers no solution of the known contract"));
        }
        let code =
            std::fs::read(dir.join(format!("{stem}.co"))).map_err(|e| format!("{stem}.co: {e}"))?;
        let api = sys::api()?;
        let mut module: sys::Module = core::ptr::null_mut();
        // SAFETY: `code` is a whole code object, alive across the call.
        let status = unsafe { (api.hip.hipModuleLoadData)(&raw mut module, code.as_ptr().cast()) };
        if status != sys::HIP_SUCCESS {
            return Err(format!("hipModuleLoadData({stem}.co) answered {status}"));
        }
        Ok(Self {
            dtype,
            solutions,
            module,
            cus: super::multiprocessors(device as u32).map_err(|e| e.to_string())?,
            functions: Mutex::new(HashMap::new()),
        })
    }

    /// The kernel of solution `at`, resolved once.
    pub fn function(&self, at: usize) -> Result<sys::Function, String> {
        let mut map = self.functions.lock().expect("functions lock poisoned");
        if let Some(&f) = map.get(&at) {
            return Ok(f as sys::Function);
        }
        let name = std::ffi::CString::new(self.solutions[at].kernel.as_str())
            .map_err(|_| "a kernel name with a NUL".to_string())?;
        let api = sys::api()?;
        let mut f: sys::Function = core::ptr::null_mut();
        // SAFETY: the module is live and the name NUL-terminated.
        let status =
            unsafe { (api.hip.hipModuleGetFunction)(&raw mut f, self.module, name.as_ptr()) };
        if status != sys::HIP_SUCCESS {
            return Err(format!(
                "hipModuleGetFunction({}) answered {status}",
                self.solutions[at].label()
            ));
        }
        map.insert(at, f as usize);
        Ok(f)
    }

    /// The solution of tile `code` (`MT0·10⁶ + MT1·10³ + depth U`, as the
    /// device profile names it: 256_192_064 is MT256x192x64), and of its
    /// runtime settings the first with WGM 1 and no StaggerU, else the first
    /// with WGM 1, else the tile's first: one solution a tile, whatever the
    /// shape.
    #[must_use]
    pub fn pick(&self, code: u32) -> Option<usize> {
        let tile = |s: &Solution| s.mt[0] * 1_000_000 + s.mt[1] * 1_000 + s.mt[2] == code;
        let of = |want: &dyn Fn(&Solution) -> bool| {
            self.solutions.iter().position(|s| tile(s) && want(s))
        };
        of(&|s| s.wgm == 1 && s.stagger_u == 0)
            .or_else(|| of(&|s| s.wgm == 1))
            .or_else(|| of(&|_| true))
    }

    /// Enqueues solution `at` over `groups` UserArguments records at
    /// `user_args` (the last an n = 0 sentinel when `grid` is a worst case).
    ///
    /// # Safety
    /// `user_args` must hold `groups` records whose pointers name live
    /// device ranges of the sizes they state, until the launch completes.
    pub unsafe fn launch(
        &self,
        at: usize,
        groups: u32,
        user_args: u64,
        grid: u32,
        stream: Stream,
    ) -> Result<(), String> {
        let sol = &self.solutions[at];
        let f = self.function(at)?;
        let (arg0, arg1) = sol.internal_args(self.cus);
        let mut ka = [0u8; KERNARG_BYTES];
        for (at, v) in [(0, groups | 2 << 30), (4, arg0), (8, arg1), (12, grid)] {
            ka[at..at + 4].copy_from_slice(&u32::to_le_bytes(v));
        }
        // GSU 1 touches neither the synchronizer nor the workspace; both
        // name the argument array, a live range, all the same.
        for (at, v) in [(16, user_args), (24, user_args), (32, user_args)] {
            ka[at..at + 8].copy_from_slice(&u64::to_le_bytes(v));
        }
        let mut size = KERNARG_BYTES;
        let mut extra: [*mut c_void; 5] = [
            core::ptr::without_provenance_mut(1), // HIP_LAUNCH_PARAM_BUFFER_POINTER
            ka.as_mut_ptr().cast(),
            core::ptr::without_provenance_mut(2), // HIP_LAUNCH_PARAM_BUFFER_SIZE
            (&raw mut size).cast(),
            core::ptr::without_provenance_mut(3), // HIP_LAUNCH_PARAM_END
        ];
        let api = sys::api()?;
        // SAFETY: HIP copies the kernarg block at the launch; the caller
        // keeps the buffers it names live.
        let status = unsafe {
            (api.hip.hipModuleLaunchKernel)(
                f,
                grid,
                1,
                1,
                sol.lanes,
                1,
                1,
                0,
                stream,
                core::ptr::null_mut(),
                extra.as_mut_ptr(),
            )
        };
        if status != sys::HIP_SUCCESS {
            return Err(format!(
                "hipModuleLaunchKernel({}) answered {status}",
                sol.label()
            ));
        }
        Ok(())
    }
}

fn current_device() -> c_int {
    let mut device: c_int = 0;
    // SAFETY: writes one int.
    unsafe { super::rt::hipGetDevice(&raw mut device) };
    device
}

/// Where hipBLASLt keeps its kernel libraries.
fn library_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(p) = std::env::var_os("HIPBLASLT_TENSILE_LIBPATH") {
        dirs.push(PathBuf::from(p));
    }
    if let Some(root) = std::env::var_os("ROCM_PATH") {
        dirs.push(PathBuf::from(root).join("lib/hipblaslt/library"));
    }
    dirs.push(PathBuf::from("/opt/rocm/lib/hipblaslt/library"));
    dirs
}

/// The grouped solutions this module launches: the user-argument contract
/// above, `[n, k]` weights (transA), the types asked for, GSU 1 and no
/// StreamK, adaptive or cluster modes.
fn solutions(logic: &Mp, dtype: Dtype) -> Vec<Solution> {
    let ty = if dtype == Dtype::Bf16 {
        "BFloat16"
    } else {
        "Half"
    };
    let Some(Mp::Arr(all)) = logic.get("solutions") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for s in all {
        let i = |p: &[&str]| s.at(p).and_then(Mp::int);
        let t = |p: &[&str]| s.at(p).is_some_and(Mp::truth);
        let txt = |p: &[&str]| s.at(p).and_then(Mp::text).unwrap_or("");
        let ok = i(&["internalArgsSupport", "version"]) == Some(2)
            && t(&["internalArgsSupport", "useUniversalArgs"])
            && !t(&["internalArgsSupport", "useSFC"])
            && t(&["problemType", "supportDeviceUserArguments"])
            && t(&["problemType", "groupedGemm"])
            && t(&["problemType", "transA"])
            && !t(&["problemType", "transB"])
            && ["aType", "bType", "cType", "dType"]
                .iter()
                .all(|k| txt(&["problemType", k]) == ty)
            && txt(&["problemType", "computeType"]) == "Float"
            && i(&["sizeMapping", "globalSplitU"]) == Some(1)
            && i(&["sizeMapping", "globalAccumulation"]) != Some(3)
            && i(&["sizeMapping", "adaptiveGemmGSUA"]) == Some(0)
            && i(&["sizeMapping", "adaptiveGemmNTAB"]) == Some(0)
            && i(&["sizeMapping", "streamK"]) == Some(0)
            && i(&["sizeMapping", "workGroupMappingXCC"]).is_some_and(|x| x >= 1)
            && s.at(&["sizeMapping", "clusterDim"]).map(Mp::ints) == Some(vec![1, 1]);
        if !ok {
            continue;
        }
        let mt = s
            .at(&["sizeMapping", "macroTile"])
            .map(Mp::ints)
            .unwrap_or_default();
        let wg = s
            .at(&["sizeMapping", "workGroup"])
            .map(Mp::ints)
            .unwrap_or_default();
        let field = |p: &str| i(&["sizeMapping", p]);
        let (Some(depth), Some(su), Some(sum), Some(sus), Some(wgm), Some(wgmxcc), Some(wgmxccg)) = (
            field("depthU"),
            field("staggerU"),
            field("staggerUMapping"),
            field("staggerStrideShift"),
            field("workGroupMapping"),
            field("workGroupMappingXCC"),
            field("workGroupMappingXCCGroup"),
        ) else {
            continue;
        };
        let lanes = wg.iter().product::<i64>();
        if mt.len() < 2 || txt(&["kernelName"]).is_empty() || !(64..=1024).contains(&lanes) {
            continue;
        }
        out.push(Solution {
            index: i(&["index"]).unwrap_or(-1),
            kernel: txt(&["kernelName"]).to_string(),
            mt: [mt[0] as u32, mt[1] as u32, depth as u32],
            lanes: lanes as u32,
            stagger_u: su as u32,
            stagger_mapping: sum as u32,
            stagger_shift: sus as u32,
            wgm: wgm as i32,
            wgmxcc: wgmxcc as i32,
            wgmxccg: wgmxccg as i32,
        });
    }
    out
}

// ---- msgpack, as much as the logic files use -------------------------------

#[derive(Debug)]
enum Mp {
    Nil,
    Bool(bool),
    Int(i64),
    Float,
    Str(String),
    Bin,
    Arr(Vec<Mp>),
    Map(Vec<(Mp, Mp)>),
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }
    fn be(&mut self, n: usize) -> Option<u64> {
        Some(self.take(n)?.iter().fold(0, |a, &x| a << 8 | u64::from(x)))
    }
    fn str(&mut self, n: usize) -> Option<Mp> {
        Some(Mp::Str(String::from_utf8_lossy(self.take(n)?).into_owned()))
    }
    fn arr(&mut self, n: usize) -> Option<Mp> {
        (0..n)
            .map(|_| self.value())
            .collect::<Option<_>>()
            .map(Mp::Arr)
    }
    fn map(&mut self, n: usize) -> Option<Mp> {
        (0..n)
            .map(|_| Some((self.value()?, self.value()?)))
            .collect::<Option<_>>()
            .map(Mp::Map)
    }
    fn value(&mut self) -> Option<Mp> {
        let t = self.take(1)?[0];
        Some(match t {
            0x00..=0x7f => Mp::Int(i64::from(t)),
            0x80..=0x8f => self.map(usize::from(t & 15))?,
            0x90..=0x9f => self.arr(usize::from(t & 15))?,
            0xa0..=0xbf => self.str(usize::from(t & 31))?,
            0xe0..=0xff => Mp::Int(i64::from(t as i8)),
            0xc0 => Mp::Nil,
            0xc2 => Mp::Bool(false),
            0xc3 => Mp::Bool(true),
            0xc4..=0xc6 => {
                let n = self.be(1 << (t - 0xc4))? as usize;
                self.take(n)?;
                Mp::Bin
            }
            0xca => {
                self.take(4)?;
                Mp::Float
            }
            0xcb => {
                self.take(8)?;
                Mp::Float
            }
            0xcc..=0xcf => Mp::Int(self.be(1 << (t - 0xcc))? as i64),
            0xd0 => Mp::Int(i64::from(self.be(1)? as u8 as i8)),
            0xd1 => Mp::Int(i64::from(self.be(2)? as u16 as i16)),
            0xd2 => Mp::Int(i64::from(self.be(4)? as u32 as i32)),
            0xd3 => Mp::Int(self.be(8)? as i64),
            0xd9..=0xdb => {
                let n = self.be(1 << (t - 0xd9))? as usize;
                self.str(n)?
            }
            0xdc | 0xdd => {
                let n = self.be(2 << (t - 0xdc))? as usize;
                self.arr(n)?
            }
            0xde | 0xdf => {
                let n = self.be(2 << (t - 0xde))? as usize;
                self.map(n)?
            }
            _ => return None,
        })
    }
}

impl Mp {
    fn get(&self, key: &str) -> Option<&Mp> {
        let Mp::Map(kv) = self else { return None };
        kv.iter()
            .find(|(k, _)| matches!(k, Mp::Str(s) if s == key))
            .map(|(_, v)| v)
    }
    fn at(&self, path: &[&str]) -> Option<&Mp> {
        path.iter().try_fold(self, |v, k| v.get(k))
    }
    fn int(&self) -> Option<i64> {
        match self {
            Mp::Int(v) => Some(*v),
            Mp::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }
    fn truth(&self) -> bool {
        matches!(self, Mp::Bool(true))
    }
    fn text(&self) -> Option<&str> {
        match self {
            Mp::Str(s) => Some(s),
            _ => None,
        }
    }
    fn ints(&self) -> Vec<i64> {
        match self {
            Mp::Arr(v) => v.iter().filter_map(Mp::int).collect(),
            _ => Vec::new(),
        }
    }
}
