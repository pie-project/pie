//! What a load makes before the wafer opens: the plan of the model's trace, the program it links
//! and the checkpoint's cut for it, each kept in the cache under a key of everything it follows
//! from, so a second load of the same model compiles and cuts nothing.
//!
//! ```text
//! <cache_dir>/wafer/
//!   p-<model>-<key>/   a linked program: the build (plan.json, layout/, ...), out/ (cslc's),
//!                      out/artifact/ (what the engine opens), key.json, READY
//!   t-<model>-<key>/   a cut: <name>.npy, key.json, READY
//!   .lock              held while one of them is made
//! ```
//!
//! An entry is made in a temporary directory and renamed into place, `READY` last; making one
//! drops the model's other entries of its kind, so a model keeps one program and one cut.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use checkpoint::executor::{Execution, sink::MemorySink};
use checkpoint::file::read::parse_metadata;
use checkpoint::file::zt;
use checkpoint::plan::{StorageTarget, compile_streaming};
use checkpoint::types::BackendKind;
use kernels_cerebras::compile::build::{self, EMITS, Fabric, ProgramKey};
use kernels_cerebras::compile::emit::params::Levers;
use kernels_cerebras::compile::plan::Plan;
use kernels_cerebras::compile::program::Program;
use kernels_cerebras::compile::target::{self, Ask};
use kernels_cerebras::compile::tiles::{self, Param, Params};
use kernels_cerebras::compile::toolchain::{Arch, Cslc};
use poem_ir::Trace;

use crate::artifact::Weights;
use crate::config::{Build, Config, Target, Tiles};

type R<T> = std::result::Result<T, String>;

/// The cut's format: bumped when the tiler's output for the same params changes.
const TILES: u32 = 2;

/// A load's program and tiles, with the temporary cut it keeps alive (`tiles = "memory"`).
pub struct Prepared {
    pub program: PathBuf,
    pub weights: Weights,
    _scratch: Option<tempfile::TempDir>,
}

/// The trace's params as the checkpoint lands them.
pub struct Landed {
    shapes: HashMap<String, (Vec<u64>, poem_ir::Dtype)>,
    bytes: HashMap<String, Vec<u8>>,
}

impl Params for Landed {
    fn get(&self, name: &str) -> Option<Param<'_>> {
        let (shape, dtype) = self.shapes.get(name)?;
        Some(Param {
            shape,
            dtype: *dtype,
            bytes: self.bytes.get(name)?,
        })
    }
}

impl Landed {
    /// Params already landed, as a safetensors file of the trace's names (a test checkpoint's).
    pub fn read(path: &Path) -> R<Landed> {
        let all = std::fs::read(path).map_err(io(path))?;
        let n = all
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap_or_default()) as usize)
            .ok_or_else(|| format!("{}: no safetensors header", path.display()))?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(all.get(8..8 + n).unwrap_or_default())
                .map_err(|e| format!("{}: {e}", path.display()))?;
        let (mut shapes, mut bytes) = (HashMap::new(), HashMap::new());
        for (name, t) in header.iter().filter(|(k, _)| *k != "__metadata__") {
            let dtype = match t["dtype"].as_str() {
                Some("BF16") => poem_ir::Dtype::Bf16,
                Some("F32") => poem_ir::Dtype::F32,
                Some("U8") => poem_ir::Dtype::U8,
                d => return Err(format!("{name}: a {d:?} tensor")),
            };
            let shape: Vec<u64> = t["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(serde_json::Value::as_u64).collect())
                .unwrap_or_default();
            let off: Vec<usize> = t["data_offsets"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|v| v as usize))
                        .collect()
                })
                .unwrap_or_default();
            let data = off
                .get(..2)
                .and_then(|o| all.get(8 + n + o[0]..8 + n + o[1]))
                .ok_or_else(|| format!("{name}: offsets past the file"))?;
            shapes.insert(name.clone(), (shape, dtype));
            bytes.insert(name.clone(), data.to_vec());
        }
        Ok(Landed { shapes, bytes })
    }
}

/// Lands every param of `trace` from `checkpoint` through the import contract, on the host.
pub fn land(trace: &Trace, contract: &ModelContract, checkpoint: &Path) -> R<Landed> {
    let (metadata, snapshot) = if checkpoint.is_dir() {
        (
            parse_metadata(checkpoint).map_err(|e| e.to_string())?,
            checkpoint,
        )
    } else {
        (
            zt::parse(checkpoint).map_err(|e| e.to_string())?,
            checkpoint.parent().unwrap_or(Path::new(".")),
        )
    };
    let target = StorageTarget::for_backend(BackendKind::Cerebras, 0, 1);
    let plan = compile_streaming(&metadata, contract, target).map_err(|e| e.to_string())?;
    let mut sink = MemorySink::default();
    Execution::new(&plan, snapshot)
        .streaming()
        .sink(&mut sink)
        .run()
        .map_err(|e| e.to_string())?;
    Ok(Landed {
        shapes: trace
            .params
            .iter()
            .map(|p| (p.name.clone(), (p.shape.clone(), p.dtype)))
            .collect(),
        bytes: sink.tensors,
    })
}

fn arch(t: Target) -> Arch {
    match t {
        Target::Wse2 => Arch::Wse2,
        Target::Wse3 => Arch::Wse3,
    }
}

fn io(p: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", p.display())
}

/// A file name part of the model's name.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// The cache, its lock held.
struct Cache {
    root: PathBuf,
    _lock: File,
}

impl Cache {
    fn open(cfg: &Config) -> R<Cache> {
        let base = cfg
            .cache_dir
            .clone()
            .ok_or("the cerebras engine has no cache_dir to keep its programs in")?;
        let root = base.join("wafer");
        std::fs::create_dir_all(&root).map_err(io(&root))?;
        let lock_path = root.join(".lock");
        let lock = File::create(&lock_path).map_err(io(&lock_path))?;
        lock.lock().map_err(io(&lock_path))?;
        Ok(Cache { root, _lock: lock })
    }

    /// The entry `<kind>-<model>-<key>`, made by `make` into a fresh directory unless ready; the
    /// model's other entries of the kind go once it is.
    fn entry(
        &self,
        kind: &str,
        model: &str,
        key: &str,
        parts: &serde_json::Value,
        make: impl FnOnce(&Path) -> R<()>,
    ) -> R<(PathBuf, bool)> {
        let stem = format!("{kind}-{}-", slug(model));
        let dir = self.root.join(format!("{stem}{key}"));
        if dir.join("READY").exists() {
            return Ok((dir, false));
        }
        let tmp = self.root.join(format!(".tmp-{stem}{key}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).map_err(io(&tmp))?;
        make(&tmp)?;
        let text = serde_json::to_string_pretty(parts).map_err(|e| e.to_string())?;
        std::fs::write(tmp.join("key.json"), text).map_err(io(&tmp))?;
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::rename(&tmp, &dir).map_err(io(&dir))?;
        std::fs::write(dir.join("READY"), b"").map_err(io(&dir))?;
        for e in std::fs::read_dir(&self.root)
            .map_err(io(&self.root))?
            .flatten()
        {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with(&stem) && e.path() != dir {
                tracing::info!(entry = %n, "cerebras cache: a stale entry dropped");
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
        Ok((dir, true))
    }
}

/// The plan, program and tiles of a load (`max_model_len`: the positions to serve, if stated).
pub fn prepare(
    cfg: &Config,
    trace: &Trace,
    contract: &ModelContract,
    checkpoint: &Path,
    max_model_len: Option<u32>,
) -> R<Prepared> {
    let knobs = cfg.debug();
    if cfg.debug.is_some() {
        tracing::warn!(settings = ?knobs, "cerebras: debug settings in force");
    }
    let ask = Ask {
        set: knobs.set.clone(),
        positions: max_model_len,
    };
    let cache = Cache::open(cfg)?;
    // One plan: the search's, which the memory model already holds within its
    // PEs; a link past it is an error.
    let planned = target::plan(trace, &ask).map_err(|e| format!("{}: {e}", trace.name))?;
    let plan = planned.plan;
    tracing::info!(
        kernels = kernels_cerebras::compile::kernels::KernelSet::of_shape(&plan.shape).name(),
        preset = %planned.preset.name,
        positions = plan.contract.positions,
        "cerebras: the wafer program's plan"
    );
    let program = match &knobs.program {
        Some(p) => p.clone(),
        None => compile(cfg, &cache, trace, &plan, &knobs.levers)?,
    };

    let (weights, scratch) = match &knobs.weights {
        Some(w) => (Weights::parse(w)?, None),
        None => {
            let stamp = checkpoint_stamp(checkpoint)?;
            // the cut follows from the model, the partition and the contract (not the levers)
            let parts = serde_json::json!({
                "shape": plan.shape,
                "partition": plan.partition,
                "contract": plan.contract,
                "checkpoint": stamp,
                "format": TILES,
            });
            let text = serde_json::to_string(&parts).map_err(|e| e.to_string())?;
            let key = blake3::hash(text.as_bytes()).to_hex()[..16].to_string();
            let cut = |dir: &Path| -> R<()> {
                let t0 = std::time::Instant::now();
                let landed = land(trace, contract, checkpoint)?;
                tiles::stage(&plan, &landed, dir)?;
                tracing::info!(
                    s = t0.elapsed().as_secs_f64(),
                    "cerebras: the checkpoint cut"
                );
                Ok(())
            };
            match cfg.wafer.tiles {
                Tiles::Cache => {
                    let (dir, _) = cache.entry("t", &trace.name, &key, &parts, cut)?;
                    (Weights::parse(&dir.display().to_string())?, None)
                }
                Tiles::Memory => {
                    let tmp = tempfile::Builder::new()
                        .prefix(".cut-")
                        .tempdir_in(&cache.root)
                        .map_err(|e| e.to_string())?;
                    cut(tmp.path())?;
                    (
                        Weights::parse(&tmp.path().display().to_string())?,
                        Some(tmp),
                    )
                }
            }
        }
    };
    Ok(Prepared {
        program,
        weights,
        _scratch: scratch,
    })
}

/// The program of `plan`, linked into the cache unless it holds it.
fn compile(cfg: &Config, cache: &Cache, trace: &Trace, plan: &Plan, levers: &str) -> R<PathBuf> {
    let levers = Levers::of(&plan.levers, levers)?;
    let cslc = Cslc::find(&cfg.toolchain).map_err(|e| e.to_string())?;
    // A system runs a program linked against its whole wafer.
    let fabric = if cfg.cmaddr.is_some() {
        Fabric::System
    } else {
        Fabric::Simulator
    };
    let key = ProgramKey::of(plan, &levers, arch(cfg.target), fabric, cslc.frontend())?;
    let (dir, made) = cache.entry("p", &trace.name, &key.hex, &key.parts, |tmp| {
        if cfg.wafer.build == Build::Never {
            return Err(format!(
                "no compiled program {} in the cache and [engine.wafer] build = \
                 \"never\": compile it first (wafer-compile build + link)",
                key.hex
            ));
        }
        let t0 = std::time::Instant::now();
        let prog = Program::of(plan.clone())?;
        build::emit(&prog, &levers, None, tmp, &EMITS)?;
        let room = build::link(tmp, &cslc, arch(cfg.target), fabric)?;
        tracing::info!(
            s = t0.elapsed().as_secs_f64(),
            stack_room = room,
            "cerebras: the wafer program compiled"
        );
        Ok(())
    })?;
    tracing::info!(program = %dir.display(), compiled = made, "cerebras: the wafer program");
    Ok(dir.join("out"))
}

/// The steps a request of `trace`'s generative schedule denoises over (1 where it pins none).
fn schedule_steps(trace: &Trace) -> u32 {
    poem_compiler::catalog::repository()
        .parse(&trace.name)
        .and_then(|m| m.try_generative().ok().flatten())
        .and_then(|g| g.schedule)
        .map_or(1, |s| {
            u32::try_from(s.pinned_sigmas.len()).unwrap_or(1).max(1)
        })
}

/// A stream model's programs, compiled on load into the cache unless it holds them: each reading
/// of the trace's `reading` switch the stream compiler lowers, linked for the engine's fabric.
/// Returns each program's directory and link; `None` where no reading is a stream reading. A
/// reading the compiler refuses is not served (logged). The key hashes every program's files and
/// the link's target.
pub fn stream_programs(
    cfg: &Config,
    trace: &Trace,
    host: bool,
    voxels: Option<u32>,
) -> R<Option<Vec<(PathBuf, PathBuf)>>> {
    use kernels_cerebras::compile::stream;
    if trace.facts.declared("reading").is_none() {
        return Ok(None);
    }
    let rows = cfg
        .wafer
        .stream_rows
        .unwrap_or(crate::config::DEFAULT_STREAM_ROWS);
    let cache = Cache::open(cfg)?;
    let stage = tempfile::Builder::new()
        .prefix(".stream-")
        .tempdir_in(&cache.root)
        .map_err(io(&cache.root))?;
    let mut names = Vec::new();
    let mut hasher = blake3::Hasher::new();
    // the readings' programs chosen together: the cheapest request of those that place together
    let mut wanted = stream::wanted(trace, rows, voxels, schedule_steps(trace));
    // the annealer, where the config asks (its placement in the files hashed below)
    for w in &mut wanted {
        w.o.anneal = anneal(cfg);
    }
    let t0 = std::time::Instant::now();
    let (chosen, refused) =
        stream::compile_together(trace, &wanted, &|r| stage.path().join(slug(r)))?;
    for (reading, e) in &refused {
        tracing::warn!(reading = %reading, "cerebras: a reading the stream compiler refuses is not served: {e}");
        let _ = std::fs::remove_dir_all(stage.path().join(slug(reading)));
    }
    for (reading, p) in &chosen {
        tracing::info!(
            reading = %reading,
            rows,
            regime = ?p.regime,
            w = p.w,
            h = p.h,
            cycles = p.cycles,
            s = t0.elapsed().as_secs_f64(),
            "cerebras: a stream reading compiled"
        );
    }
    for (reading, _) in &chosen {
        let dir = stage.path().join(slug(reading));
        names.push(reading.clone());
        let mut files: Vec<PathBuf> = walk(&dir)?;
        files.sort();
        hasher.update(reading.as_bytes());
        for f in files {
            hasher.update(
                f.strip_prefix(&dir)
                    .unwrap_or(&f)
                    .to_string_lossy()
                    .as_bytes(),
            );
            hasher.update(&std::fs::read(&f).map_err(io(&f))?);
        }
    }
    if names.is_empty() {
        return Ok(None);
    }
    let fabric = if host {
        "host"
    } else if cfg.cmaddr.is_some() {
        "system"
    } else {
        "simulator"
    };
    hasher.update(fabric.as_bytes());
    hasher.update(b"coresident-v1");
    for (n, src) in kernels_cerebras::sources::OPS
        .iter()
        .chain(kernels_cerebras::sources::stream_ops())
    {
        hasher.update(n.as_bytes());
        hasher.update(src.as_bytes());
    }
    let cslc = if host {
        None
    } else {
        Some(Cslc::find(&cfg.toolchain).map_err(|e| e.to_string())?)
    };
    if let Some(c) = &cslc {
        hasher.update(c.frontend().to_string_lossy().as_bytes());
    }
    let key = hasher.finalize().to_hex()[..16].to_string();
    let parts = serde_json::json!({ "readings": names, "rows": rows, "fabric": fabric });
    let (root, made) = cache.entry("s", &trace.name, &key, &parts, |tmp| {
        if cfg.wafer.build == Build::Never && !host {
            return Err(format!(
                "no compiled stream programs {key} in the cache and [engine.wafer] build = \
                 \"never\": compile them first (wafer-compile stream + cslc)"
            ));
        }
        let slugs: Vec<String> = names.iter().map(|r| slug(r)).collect();
        for reading in &names {
            let from = stage.path().join(slug(reading));
            let dir = tmp.join(slug(reading));
            std::fs::rename(&from, &dir).map_err(io(&dir))?;
        }
        // the readings co-resident: one program, a reading's launch its PEs alone
        let placed = kernels_cerebras::compile::stream::coresident::combine(tmp, &slugs)?;
        tracing::info!(readings = ?placed.iter().map(|p| (&p.name, p.x, p.w, p.h)).collect::<Vec<_>>(), "cerebras: the stream readings co-resident");
        let progs = [tmp.to_path_buf()];
        if let Some(cslc) = &cslc {
            for dir in &progs {
                let reading = dir.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
                let t0 = std::time::Instant::now();
                link_stream(cfg, cslc, dir).map_err(|e| format!("{reading}: {e}"))?;
                tracing::info!(reading = %reading, s = t0.elapsed().as_secs_f64(), "cerebras: a stream program linked");
            }
        }
        Ok(())
    })?;
    tracing::info!(programs = %root.display(), compiled = made, "cerebras: the stream programs");
    Ok(Some(
        names
            .iter()
            .map(|r| (root.join(slug(r)), root.join("out")))
            .collect(),
    ))
}

/// A stream program's link (`dir`, linked into `dir/out`) for the engine's fabric.
fn link_stream(cfg: &Config, cslc: &Cslc, dir: &Path) -> R<()> {
    let fab = std::fs::read_to_string(dir.join("fabric.txt")).map_err(io(dir))?;
    let f: Vec<u32> = fab
        .split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect();
    let (w, h, bf16) = match f[..] {
        [w, h, b] => (w, h, b == 1),
        _ => return Err(format!("{}: fabric.txt is not `w h bf16`", dir.display())),
    };
    let (mut fabric_dims, fabric_offsets) =
        kernels_cerebras::compile::toolchain::memcpy_fabric(w, h);
    if cfg.cmaddr.is_some() {
        fabric_dims = arch(cfg.target).system_fabric();
    }
    cslc.compile(&kernels_cerebras::compile::toolchain::Compile {
        arch: arch(cfg.target),
        layout: dir.join("layout.csl"),
        fabric_dims,
        fabric_offsets,
        params: Vec::new(),
        memcpy: true,
        channels: 1,
        import_paths: Vec::new(),
        out: dir.join("out"),
        bf16,
    })
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// The lanes a unit decode program runs a launch: so many, or the cheapest a token up to so many.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lanes {
    Exactly(u32),
    Upto(u32),
}

/// The positions a unit decode program is first tried at when the config states none, and the
/// fewest it is searched down to.
const UNIT_POSITIONS: (u32, u32) = (8192, 64);

/// A decode program of stream units, compiled on load into the cache unless it holds it: at
/// `positions`, else the most in `UNIT_POSITIONS` its K / V columns hold, `lanes` rows a step.
/// Returns its directory, its link and its positions.
pub fn unit_program(
    cfg: &Config,
    trace: &Trace,
    positions: Option<u32>,
    lanes: Lanes,
) -> R<(PathBuf, PathBuf, u32)> {
    use kernels_cerebras::compile::stream;
    let cache = Cache::open(cfg)?;
    let cslc = Cslc::find(&cfg.toolchain).map_err(|e| e.to_string())?;
    let fabric = if cfg.cmaddr.is_some() {
        "system"
    } else {
        "simulator"
    };
    let stage = tempfile::Builder::new()
        .prefix(".units-")
        .tempdir_in(&cache.root)
        .map_err(io(&cache.root))?;
    let dir = stage.path().join("decode");
    // the search's placement while sizing; the annealer, if asked, for the program kept alone
    let compile = |at: u32, b: u32, a: &kernels_cerebras::compile::place::anneal::Anneal| {
        let _ = std::fs::remove_dir_all(&dir);
        stream::compile_decode(trace, at, b, true, None, a, &dir)
    };
    let off = kernels_cerebras::compile::place::anneal::Anneal::off();
    // the most positions that fit one lane, by bisection (or the stated ones)
    let (mut hi, mut lo) = match positions {
        Some(p) => (p, p),
        None => UNIT_POSITIONS,
    };
    let mut at = hi;
    if let Err(e) = compile(hi, 1, &off) {
        if lo == hi {
            return Err(e);
        }
        compile(lo, 1, &off)?;
        while hi > lo + lo.max(16) / 8 {
            let mid = lo + (hi - lo) / 2;
            match compile(mid, 1, &off) {
                Ok(_) => lo = mid,
                Err(_) => hi = mid,
            }
        }
        // multiples of 16 positions
        lo -= lo % 16;
        at = lo;
    }
    // the lanes a launch: the stated, else the cheapest a token of those that place
    let lanes = match lanes {
        Lanes::Exactly(b) => b,
        Lanes::Upto(most) => {
            let mut best = (f64::MAX, 1);
            let mut b = 1;
            while b <= most.max(1) {
                if let Ok(p) = compile(at, b, &off) {
                    let per =
                        kernels_cerebras::compile::cost::stream::time(&p).cycles / f64::from(b);
                    tracing::info!(
                        lanes = b,
                        cycles_a_token = per,
                        "cerebras: a unit decode program priced"
                    );
                    if per < best.0 {
                        best = (per, b);
                    }
                }
                b *= 2;
            }
            best.1
        }
    };
    compile(at, lanes, &anneal(cfg))?;
    let mut hasher = blake3::Hasher::new();
    let mut files = walk(&dir)?;
    files.sort();
    for f in files {
        hasher.update(
            f.strip_prefix(&dir)
                .unwrap_or(&f)
                .to_string_lossy()
                .as_bytes(),
        );
        hasher.update(&std::fs::read(&f).map_err(io(&f))?);
    }
    for (n, src) in kernels_cerebras::sources::OPS
        .iter()
        .chain(kernels_cerebras::sources::stream_ops())
    {
        hasher.update(n.as_bytes());
        hasher.update(src.as_bytes());
    }
    hasher.update(fabric.as_bytes());
    hasher.update(cslc.frontend().to_string_lossy().as_bytes());
    let key = hasher.finalize().to_hex()[..16].to_string();
    let parts =
        serde_json::json!({ "decode": "units", "positions": at, "lanes": lanes, "fabric": fabric });
    let made = cache.entry("u", &trace.name, &key, &parts, |tmp| {
        if cfg.wafer.build == Build::Never {
            return Err(format!(
                "no compiled unit program {key} in the cache and [engine.wafer] build = \
                 \"never\": compile it first (wafer-compile stream --reading decode + cslc)"
            ));
        }
        for e in std::fs::read_dir(&dir).map_err(io(&dir))?.flatten() {
            let to = tmp.join(e.file_name());
            std::fs::rename(e.path(), &to).map_err(io(&to))?;
        }
        kernels_cerebras::compile::stream::coresident::ops(tmp, &[])?;
        let t0 = std::time::Instant::now();
        link_stream(cfg, &cslc, tmp)?;
        tracing::info!(
            positions = at,
            s = t0.elapsed().as_secs_f64(),
            "cerebras: the unit decode program linked"
        );
        Ok(())
    });
    let (root, made) = made?;
    tracing::info!(program = %root.display(), positions = at, compiled = made, "cerebras: the unit decode program");
    Ok((root.clone(), root.join("out"), at))
}

/// Every file under `dir`.
fn walk(dir: &Path) -> R<Vec<PathBuf>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).map_err(io(dir))?.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p)?);
        } else {
            out.push(p);
        }
    }
    Ok(out)
}

/// What identifies a checkpoint's contents: every file's name, size and modification time.
fn checkpoint_stamp(checkpoint: &Path) -> R<String> {
    let mut files: Vec<(String, u64, u64)> = Vec::new();
    let entries: Vec<PathBuf> = if checkpoint.is_dir() {
        std::fs::read_dir(checkpoint)
            .map_err(io(checkpoint))?
            .map(|e| e.map(|e| e.path()).map_err(io(checkpoint)))
            .collect::<R<Vec<PathBuf>>>()?
            .into_iter()
            .filter(|p| p.is_file())
            .collect()
    } else {
        vec![checkpoint.to_path_buf()]
    };
    for p in entries {
        let m = std::fs::metadata(&p).map_err(io(&p))?;
        // (a file whose time cannot be read is refused: a stamp of 0 would match a changed
        // checkpoint)
        let mtime = m
            .modified()
            .map_err(|e| format!("{}: its modification time: {e}", p.display()))?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("{}: its modification time: {e}", p.display()))?
            .as_secs();
        files.push((
            p.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            m.len(),
            mtime,
        ));
    }
    files.sort();
    Ok(format!("{}:{files:?}", checkpoint.display()))
}

/// [`plane`], its landed bytes then dropped: a forward built param by param holds each once.
pub fn take_plane(landed: &mut Landed, name: &str, count: usize) -> Vec<f32> {
    let v = plane(landed, name, count);
    landed.bytes.remove(name);
    v
}

/// A checkpoint plane's values as f32 (`landed` holds it bf16 or f32 as the contract lands it).
pub fn plane(landed: &Landed, name: &str, count: usize) -> Vec<f32> {
    use kernels_cerebras::compile::tiles::Params;
    let Some(p) = landed.get(name) else {
        return Vec::new();
    };
    let b = p.bytes;
    if b.len() == 2 * count {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16))
            .collect()
    } else if b.len() == 4 * count {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect()
    } else {
        Vec::new()
    }
}

/// The annealing placer as `cfg` asks (`wafer.anneal`: moves a block, seed 1; off unless stated).
fn anneal(cfg: &Config) -> kernels_cerebras::compile::place::anneal::Anneal {
    kernels_cerebras::compile::place::anneal::Anneal {
        per_block: cfg.wafer.anneal.unwrap_or(0),
        seed: 1,
        units: false,
    }
}
