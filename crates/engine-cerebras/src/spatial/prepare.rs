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
use kernels_cerebras::spatial::build::{self, EMITS, ProgramKey};
use kernels_cerebras::spatial::emit::params::Levers;
use kernels_cerebras::spatial::plan::Plan;
use kernels_cerebras::spatial::program::Program;
use kernels_cerebras::spatial::target::{self, Ask};
use kernels_cerebras::spatial::tiles::{self, Param, Params};
use kernels_cerebras::spatial::toolchain::{Arch, Cslc};
use poem_ir::Trace;

use super::artifact::Weights;
use super::config::{Build, Config, Target, Tiles};

type R<T> = std::result::Result<T, String>;

/// The cut's format: bumped when a tiler's output for the same params changes.
const TILES: u32 = 1;

/// A load's program and tiles, with the temporary cut it keeps alive (`tiles = "memory"`).
pub struct Prepared {
    pub program: PathBuf,
    pub weights: Weights,
    pub plan: Plan,
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

    /// The entry `<kind>-<model>-<key>`, made by `make` into a fresh directory unless it is
    /// ready; the model's other entries of the kind go once it is.
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
    let planned = target::plan(
        trace,
        &Ask {
            preset: cfg.wafer.preset.clone(),
            set: knobs.set.clone(),
            positions: max_model_len,
        },
    )?;
    let plan = planned.plan;
    tracing::info!(
        family = plan.family.name(),
        preset = %planned.preset.name,
        positions = plan.contract.positions,
        "cerebras: the wafer program's plan"
    );
    let cache = Cache::open(cfg)?;

    let program = match &knobs.program {
        Some(p) => p.clone(),
        None => {
            let levers = Levers::of(&plan.levers, &knobs.levers)?;
            let cslc = Cslc::find(&cfg.toolchain).map_err(|e| e.to_string())?;
            let key = ProgramKey::of(&plan, &levers, arch(cfg.target), cslc.frontend())?;
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
                let room = build::link(tmp, &cslc, arch(cfg.target))?;
                tracing::info!(
                    s = t0.elapsed().as_secs_f64(),
                    stack_room = room,
                    "cerebras: the wafer program compiled"
                );
                Ok(())
            })?;
            tracing::info!(program = %dir.display(), compiled = made, "cerebras: the wafer program");
            dir.join("out")
        }
    };

    let (weights, scratch) = match &knobs.weights {
        Some(w) => (Weights::parse(w)?, None),
        None => {
            let stamp = checkpoint_stamp(checkpoint)?;
            let mut tiling = kernels_cerebras::spatial::family::of(plan.family).tiling(&plan)?;
            tiling["checkpoint"] = stamp.into();
            tiling["format"] = TILES.into();
            let text = serde_json::to_string(&tiling).map_err(|e| e.to_string())?;
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
                    let (dir, _) = cache.entry("t", &trace.name, &key, &tiling, cut)?;
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
        plan,
        _scratch: scratch,
    })
}

/// What identifies a checkpoint's contents: every file's name, size and modification time.
fn checkpoint_stamp(checkpoint: &Path) -> R<String> {
    let mut files: Vec<(String, u64, u64)> = Vec::new();
    let entries: Vec<PathBuf> = if checkpoint.is_dir() {
        std::fs::read_dir(checkpoint)
            .map_err(io(checkpoint))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect()
    } else {
        vec![checkpoint.to_path_buf()]
    };
    for p in entries {
        let m = std::fs::metadata(&p).map_err(io(&p))?;
        let mtime = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
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
