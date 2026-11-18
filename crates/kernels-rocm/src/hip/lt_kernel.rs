//! The kernel a dense hipBLASLt plan launches, as hipBLASLt's own library
//! states it, for the waves it fills ([`super::lt::waves`]).
//!
//! hipBLASLt 1.4 leaves the heuristic result's `wavesCount` unwritten, so
//! the waves are read from what the library does state: the algorithm's
//! solution index (`hipblaslt_ext::getIndexFromAlgo`), and that solution in
//! the logic beside its code object (`TensileLibrary_*.dat.zlib`: its kernel,
//! macro tile, workgroup and global split). The workgroups a compute unit
//! holds at once of that kernel are HIP's occupancy calculator's over the
//! kernel itself, loaded from the code object and unloaded again. What is
//! read is kept on disk beside the compiled code objects (`lt-solutions/`),
//! keyed by the architecture, hipBLASLt's version and its logic files, so
//! only a machine's first load reads the library.
//!
//! What this cannot read is stated, never assumed: the first reason is kept
//! and [`pricing`] names it, and the estimate prices such a matmul by the
//! measured curve as timed.
#![allow(unsafe_code)]

use core::ffi::{c_int, c_void};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use super::msgpack::{self, Mp};
use super::sys;

/// `int hipblaslt_ext::getIndexFromAlgo(hipblasLtMatmulAlgo_t&)`, the one C++
/// entry read here, by its Itanium-mangled name: hipBLASLt declares no C name
/// for it. The ABI assumed is Linux x86-64's (the reference passed as a
/// pointer, the `int` returned in a register); no class crosses it.
const INDEX_FROM_ALGO: &str = "_ZN13hipblaslt_ext16getIndexFromAlgoER22_hipblasLtMatmulAlgo_t";

type IndexFromAlgo = unsafe extern "C" fn(algo: *mut c_void) -> c_int;

fn index_from_algo() -> Result<IndexFromAlgo, String> {
    static ENTRY: OnceLock<Result<(libloading::Library, IndexFromAlgo), String>> = OnceLock::new();
    ENTRY
        .get_or_init(|| {
            let lib = sys::open_library("libhipblaslt.so")?;
            // SAFETY: the symbol is resolved with the signature above, and the
            // library is kept beside the pointer for the process.
            let f = unsafe {
                *lib.get::<IndexFromAlgo>(format!("{INDEX_FROM_ALGO}\0").as_bytes())
                    .map_err(|e| format!("libhipblaslt.so has no `{INDEX_FROM_ALGO}`: {e}"))?
            };
            Ok((lib, f))
        })
        .as_ref()
        .map(|(_, f)| *f)
        .map_err(Clone::clone)
}

/// A solution as the library's logic states it.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The logic file (without `.dat.zlib`) beside the code object holding it.
    stem: String,
    pub kernel: String,
    /// MT0 (the weight's rows a tile), MT1 (the activation's rows a tile).
    pub tile: [u32; 2],
    /// Threads a workgroup.
    pub lanes: u32,
    /// The global split of K, `None` where the solution leaves it to the
    /// launch (`globalSplitU` -1: hipBLASLt picks it from the problem).
    pub split: Option<u32>,
}

/// One architecture's dense bf16 logic as far as it has been read: every
/// `TensileLibrary_BB_BB_*` logic of the contraction a plan asks for (`[n,
/// k]` weights transposed, column-major, `Alik_Bljk`), read a file at a time
/// until the index asked for is found.
struct Logic {
    dir: PathBuf,
    unread: Vec<String>,
    by_index: HashMap<i64, Solution>,
    by_stem: HashMap<String, Vec<String>>,
}

impl Logic {
    fn open(arch: &str) -> Result<Self, String> {
        let dir = super::lt_grouped::library_dirs()
            .into_iter()
            .map(|d| d.join(arch))
            .find(|d| d.is_dir())
            .ok_or_else(|| format!("no {arch} directory under hipBLASLt's library"))?;
        let mut unread: Vec<String> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().into_string().ok())
            .filter_map(|f| f.strip_suffix(".dat.zlib").map(str::to_string))
            .filter(|f| {
                f.starts_with("TensileLibrary_BB_BB_") && f.contains("_Contraction_l_Alik_Bljk_")
            })
            .collect();
        if unread.is_empty() {
            return Err(format!(
                "no TensileLibrary_BB_BB_*Alik_Bljk logic under {}",
                dir.display()
            ));
        }
        // Popped from the back: read in name order.
        unread.sort_unstable_by(|a, b| b.cmp(a));
        Ok(Self {
            dir,
            unread,
            by_index: HashMap::new(),
            by_stem: HashMap::new(),
        })
    }

    /// The solution `index`, reading the logic until it is found.
    fn find(&mut self, index: i64) -> Result<Solution, String> {
        loop {
            if let Some(s) = self.by_index.get(&index) {
                return Ok(s.clone());
            }
            let Some(stem) = self.unread.pop() else {
                return Err(format!(
                    "solution {index} is in none of the TensileLibrary_BB_BB_*Alik_Bljk logic under {}",
                    self.dir.display()
                ));
            };
            self.read(&stem)?;
        }
    }

    fn read(&mut self, stem: &str) -> Result<(), String> {
        let packed = std::fs::read(self.dir.join(format!("{stem}.dat.zlib")))
            .map_err(|e| format!("{stem}.dat.zlib: {e}"))?;
        let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&packed)
            .map_err(|e| format!("{stem}.dat.zlib does not inflate: {e:?}"))?;
        let logic = msgpack::read(&raw).ok_or_else(|| format!("{stem}.dat.zlib is not msgpack"))?;
        let Some(Mp::Arr(listed)) = logic.get("solutions") else {
            return Err(format!("{stem}.dat.zlib lists no solutions"));
        };
        let mut kernels = Vec::new();
        for s in listed {
            let i = |p: &[&str]| s.at(p).and_then(Mp::int);
            let (Some(index), Some(kernel)) =
                (i(&["index"]), s.at(&["kernelName"]).and_then(Mp::text))
            else {
                continue;
            };
            let tile = s
                .at(&["sizeMapping", "macroTile"])
                .map(Mp::ints)
                .unwrap_or_default();
            let group = s
                .at(&["sizeMapping", "workGroup"])
                .map(Mp::ints)
                .unwrap_or_default();
            let lanes = group.iter().product::<i64>();
            if tile.len() < 2 || tile[0] <= 0 || tile[1] <= 0 || group.is_empty() || lanes <= 0 {
                continue;
            }
            let split = match i(&["sizeMapping", "globalSplitU"]) {
                Some(g) if g >= 1 => Some(g as u32),
                _ => None,
            };
            kernels.push(kernel.to_string());
            self.by_index.insert(
                index,
                Solution {
                    stem: stem.to_string(),
                    kernel: kernel.to_string(),
                    tile: [tile[0] as u32, tile[1] as u32],
                    lanes: lanes as u32,
                    split,
                },
            );
        }
        self.by_stem.insert(stem.to_string(), kernels);
        Ok(())
    }
}

fn arch_of(ordinal: c_int) -> Result<String, String> {
    Ok(crate::jit::cache::arch(ordinal)
        .ok_or("the device named no architecture")?
        .split(':')
        .next()
        .unwrap_or_default()
        .to_string())
}

/// The logic of each architecture, opened once.
fn logic() -> &'static Mutex<HashMap<String, Result<Logic, String>>> {
    static LOGIC: OnceLock<Mutex<HashMap<String, Result<Logic, String>>>> = OnceLock::new();
    LOGIC.get_or_init(|| Mutex::new(HashMap::new()))
}

/// What hipBLASLt states of a solution on one architecture: the solution,
/// and the workgroups of its kernel one compute unit holds at once.
#[derive(Clone, Debug)]
pub struct Stated {
    pub solution: Solution,
    pub resident: u32,
}

/// Each architecture's stated solutions so far, by index, with the disk
/// record they are kept in (`None` where no cache directory is installed).
struct Known {
    record: Option<PathBuf>,
    by_index: HashMap<i64, Stated>,
}

fn known() -> &'static Mutex<HashMap<String, Known>> {
    static KNOWN: OnceLock<Mutex<HashMap<String, Known>>> = OnceLock::new();
    KNOWN.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The disk record of `arch`'s stated solutions: named by hipBLASLt's
/// version and a digest of its dense bf16 logic files' names and sizes. A
/// record of another version or logic is never opened (its name differs),
/// so a changed library is read again rather than trusted; where the
/// version cannot be read there is no record at all.
fn record_of(arch: &str) -> Option<PathBuf> {
    let root = crate::disk::dir("lt-solutions")?;
    let version = super::lt_grouped::version().ok()?;
    let dir = super::lt_grouped::library_dirs()
        .into_iter()
        .map(|d| d.join(arch))
        .find(|d| d.is_dir())?;
    let mut files: Vec<(String, u64)> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            (name.starts_with("TensileLibrary_BB_BB_")
                && name.contains("_Contraction_l_Alik_Bljk_"))
            .then(|| Some((name, e.metadata().ok()?.len())))?
        })
        .collect();
    files.sort();
    // FNV-1a over the names and sizes.
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    for (name, len) in &files {
        for byte in name.bytes().chain(len.to_le_bytes()) {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    Some(root.join(format!("{arch}-{version}-{digest:016x}.tsv")))
}

/// A record's lines: `index kernel mt0 mt1 lanes split resident stem`, split
/// 0 where the launch picks it.
fn read_record(path: &PathBuf) -> HashMap<i64, Stated> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            let [index, kernel, mt0, mt1, lanes, split, resident, stem] = f.as_slice() else {
                return None;
            };
            let split: u32 = split.parse().ok()?;
            Some((
                index.parse().ok()?,
                Stated {
                    solution: Solution {
                        stem: (*stem).to_string(),
                        kernel: (*kernel).to_string(),
                        tile: [mt0.parse().ok()?, mt1.parse().ok()?],
                        lanes: lanes.parse().ok()?,
                        split: (split > 0).then_some(split),
                    },
                    resident: resident.parse().ok()?,
                },
            ))
        })
        .collect()
}

/// Writes `by_index` whole to `path` (a sibling written, then renamed over
/// it). A record that cannot be written leaves the next load to read the
/// library again.
fn write_record(path: &PathBuf, by_index: &HashMap<i64, Stated>) {
    let mut lines: Vec<(i64, String)> = by_index
        .iter()
        .map(|(index, s)| {
            let x = &s.solution;
            (
                *index,
                format!(
                    "{index}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    x.kernel,
                    x.tile[0],
                    x.tile[1],
                    x.lanes,
                    x.split.unwrap_or(0),
                    s.resident,
                    x.stem
                ),
            )
        })
        .collect();
    lines.sort();
    let text: String = lines.into_iter().map(|(_, l)| l).collect();
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let partial = path.with_extension(format!("tsv.{}", std::process::id()));
    if std::fs::write(&partial, text).is_ok() && std::fs::rename(&partial, path).is_err() {
        let _ = std::fs::remove_file(&partial);
    }
}

/// What hipBLASLt states of the solution `algo` (a heuristic result's 24-byte
/// algorithm) indexes on device `ordinal`: from this process's record, the
/// disk's, or the library (its logic, and the occupancy of its kernel).
pub fn stated(ordinal: c_int, algo: &mut [u64; 3]) -> Result<Stated, String> {
    let index = index_from_algo()?;
    // SAFETY: `algo` is a whole heuristic algorithm, read by reference.
    let at = i64::from(unsafe { index(algo.as_mut_ptr().cast()) });
    let arch = arch_of(ordinal)?;
    let mut all = known().lock().expect("lock poisoned");
    let known = all.entry(arch.clone()).or_insert_with(|| {
        let record = record_of(&arch);
        let by_index = record.as_ref().map(read_record).unwrap_or_default();
        Known { record, by_index }
    });
    if let Some(s) = known.by_index.get(&at) {
        return Ok(s.clone());
    }
    let solution = {
        let mut logic = logic().lock().expect("lock poisoned");
        let read = logic
            .entry(arch.clone())
            .or_insert_with(|| Logic::open(&arch));
        read.as_mut().map_err(|e| e.clone())?.find(at)?
    };
    let resident = resident(&arch, &solution)?;
    // Every solution of the logic read whose kernel's occupancy is known
    // joins the record, not only this one.
    {
        let logic = logic().lock().expect("lock poisoned");
        let blocks = blocks().lock().expect("lock poisoned");
        if let Some(Ok(read)) = logic.get(&arch) {
            for (index, s) in &read.by_index {
                if let Some(&r) = blocks.get(&(arch.clone(), s.kernel.clone())) {
                    known.by_index.entry(*index).or_insert_with(|| Stated {
                        solution: s.clone(),
                        resident: r,
                    });
                }
            }
        }
    }
    known.by_index.entry(at).or_insert_with(|| Stated {
        solution: solution.clone(),
        resident,
    });
    if let Some(path) = &known.record {
        write_record(path, &known.by_index);
    }
    Ok(Stated { solution, resident })
}

/// The workgroups a compute unit holds of each kernel asked so far, by
/// architecture and kernel name.
fn blocks() -> &'static Mutex<HashMap<(String, String), u32>> {
    static BLOCKS: OnceLock<Mutex<HashMap<(String, String), u32>>> = OnceLock::new();
    BLOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The workgroups of `s`'s kernel one compute unit of the current device
/// (of `arch`) holds at once, by HIP's occupancy calculator over the
/// kernel. A code object loads once for every kernel of its logic read so
/// far, answers them all, and unloads.
fn resident(arch: &str, s: &Solution) -> Result<u32, String> {
    let mut known = blocks().lock().expect("lock poisoned");
    if let Some(&r) = known.get(&(arch.to_string(), s.kernel.clone())) {
        return Ok(r);
    }
    let (dir, kernels) = {
        let all = logic().lock().expect("lock poisoned");
        let Some(Ok(logic)) = all.get(arch) else {
            return Err(format!("{arch}'s logic was not read"));
        };
        let mut kernels = logic.by_stem.get(&s.stem).cloned().unwrap_or_default();
        kernels.retain(|k| k != &s.kernel);
        let lanes: HashMap<&str, u32> = logic
            .by_index
            .values()
            .filter(|x| x.stem == s.stem)
            .map(|x| (x.kernel.as_str(), x.lanes))
            .collect();
        let kernels: Vec<(String, u32)> = std::iter::once((s.kernel.clone(), s.lanes))
            .chain(
                kernels
                    .into_iter()
                    .filter_map(|k| lanes.get(k.as_str()).map(|&l| (k, l))),
            )
            .collect();
        (logic.dir.clone(), kernels)
    };
    let bundled = std::fs::read(dir.join(format!("{}.co", s.stem)))
        .map_err(|e| format!("{}.co: {e}", s.stem))?;
    let code = super::code_object::unbundle(&bundled, arch)
        .map_err(|why| format!("{}.co: {why}", s.stem))?;
    let api = sys::api()?;
    let mut module: sys::Module = core::ptr::null_mut();
    // SAFETY: `code` is a whole code object, alive across the call.
    let status = unsafe { (api.hip.hipModuleLoadData)(&raw mut module, code.as_ptr().cast()) };
    if status != sys::HIP_SUCCESS {
        return Err(format!(
            "hipModuleLoadData({}.co) answered {status}",
            s.stem
        ));
    }
    let ask = |kernel: &str, lanes: u32| -> Result<u32, String> {
        let name =
            std::ffi::CString::new(kernel).map_err(|_| "a kernel name with a NUL".to_string())?;
        let mut f: sys::Function = core::ptr::null_mut();
        // SAFETY: the module is live and the name NUL-terminated.
        let status = unsafe { (api.hip.hipModuleGetFunction)(&raw mut f, module, name.as_ptr()) };
        if status != sys::HIP_SUCCESS {
            return Err(format!("{}.co has no {kernel} ({status})", s.stem));
        }
        let mut blocks: c_int = 0;
        // SAFETY: a live function of the loaded module; `blocks` a local.
        let status = unsafe {
            (api.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)(
                &raw mut blocks,
                f,
                lanes as c_int,
                0,
            )
        };
        if status != sys::HIP_SUCCESS || blocks < 1 {
            return Err(format!(
                "the occupancy of {kernel} answered {status} ({blocks} blocks)"
            ));
        }
        Ok(blocks as u32)
    };
    let mut asked = Ok(0);
    for (at, (kernel, lanes)) in kernels.iter().enumerate() {
        match ask(kernel, *lanes) {
            Ok(r) => {
                known.insert((arch.to_string(), kernel.clone()), r);
                if at == 0 {
                    asked = Ok(r);
                }
            }
            // The one asked for answers; another of the file that does not
            // is asked again (and answers) when a plan indexes it.
            Err(why) if at == 0 => {
                asked = Err(why);
                break;
            }
            Err(_) => {}
        }
    }
    // SAFETY: the module loaded above, no function of it kept.
    unsafe { (api.hip.hipModuleUnload)(module) };
    asked
}

/// The first reason a plan's waves went unread, kept for [`pricing`].
static UNREAD: Mutex<Option<String>> = Mutex::new(None);

/// Keeps `why` as the reason waves went unread, the first one only.
pub fn unread(why: String) {
    let mut first = UNREAD.lock().expect("lock poisoned");
    if first.is_none() {
        *first = Some(why);
    }
}

/// What the dense matmuls' waves were priced by, so far in this process:
/// hipBLASLt's stated solutions, or the measured curve as timed and why.
#[must_use]
pub fn pricing() -> String {
    match UNREAD.lock().expect("lock poisoned").as_ref() {
        None => "waves from hipBLASLt's solutions (tile and occupancy of the kernel each \
                 algorithm indexes; a split left to the launch taken as one from a wave, \
                 inferred from traced launches, not stated)"
            .to_string(),
        Some(why) => format!("the measured curve as timed: hipBLASLt's solutions unread ({why})"),
    }
}
