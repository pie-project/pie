//! The device: host-resident buffers, a compile cache of CSL programs, and
//! the one way a traced program runs.
//!
//! The fabric holds nothing between fires here: every root lives on the
//! host as 32-bit words, a traced fire is one CSL program whose exported
//! buffers are those roots, and running it uploads what the program reads,
//! launches it in a `fabric-run` process (the simulator is one per process),
//! and downloads what it wrote. A dry device records program texts and runs
//! nothing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::fabric::lanes::{LanePlan, picks};
use dtype::Dtype;
use kernels_cerebras::phase::{BankSplit, ColPlane, HostOp, Reduce, Segment};

use crate::error::{Fault, Result};
use crate::fabric::{HostPhase, Manifest};
use crate::sdk::{Arch, Compile, Cslc, Target};
use crate::trace::Signature;

mod host;
mod lanes;
mod layout;
mod runner;

use host::{paste_views, run_host};
use lanes::LaneLayout;
use layout::{gather_up, lay_out};
pub use runner::fabric_run;
use runner::{Servers, phase_timeout, run_once, serving};

/// The element a buffer stores on this device: every plain dtype is one
/// 32-bit word per element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    F32,
    Bf16,
    S32,
    U32,
    U8,
}

impl ElementType {
    /// Bytes of `n` elements in the dtype's host storage form.
    #[must_use]
    pub fn bytes(self, n: usize) -> usize {
        n * self.host_bytes()
    }

    fn host_bytes(self) -> usize {
        match self {
            ElementType::F32 | ElementType::S32 | ElementType::U32 => 4,
            ElementType::Bf16 => 2,
            ElementType::U8 => 1,
        }
    }
}

/// The storage element of a plain dtype, or `None` for packed formats.
#[must_use]
pub fn element_of(dtype: Dtype) -> Option<ElementType> {
    Some(match dtype {
        Dtype::F32 => ElementType::F32,
        Dtype::Bf16 => ElementType::Bf16,
        Dtype::I32 => ElementType::S32,
        Dtype::U32 => ElementType::U32,
        Dtype::U8 | Dtype::Bool => ElementType::U8,
        _ => return None,
    })
}

/// A host-resident array: one 32-bit word per element (f32 bits, or an
/// integer zero-extended), `[rows, width]`.
#[derive(Clone, Debug)]
pub struct Buffer {
    ty: ElementType,
    dtype: Dtype,
    rows: u32,
    width: u32,
    words: Arc<Vec<u32>>,
}

impl Buffer {
    #[must_use]
    pub fn new(dtype: Dtype, rows: u32, width: u32, words: Vec<u32>) -> Option<Buffer> {
        let ty = element_of(dtype)?;
        Some(Buffer {
            ty,
            dtype,
            rows,
            width,
            words: Arc::new(words),
        })
    }

    /// From bytes in the dtype's host storage form (bf16 pairs, f32/i32
    /// quads, u8 bytes).
    #[must_use]
    pub fn from_bytes(dtype: Dtype, rows: u32, width: u32, bytes: &[u8]) -> Option<Buffer> {
        let ty = element_of(dtype)?;
        let n = rows as usize * width as usize;
        let mut words = Vec::with_capacity(n);
        match ty {
            ElementType::F32 | ElementType::S32 | ElementType::U32 => {
                words.extend(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes(*c)),
                );
            }
            ElementType::Bf16 => {
                words.extend(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| u32::from(u16::from_le_bytes(*c)) << 16),
                );
            }
            ElementType::U8 => words.extend(bytes.iter().map(|b| u32::from(*b))),
        }
        words.resize(n, 0);
        Some(Buffer {
            ty,
            dtype,
            rows,
            width,
            words: Arc::new(words),
        })
    }

    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    #[must_use]
    pub fn shape(&self) -> (u32, u32) {
        (self.rows, self.width)
    }

    /// The same words viewed as f32 (a bf16 word is its f32 bit pattern).
    #[must_use]
    pub fn as_f32(&self) -> Buffer {
        Buffer {
            ty: ElementType::F32,
            dtype: Dtype::F32,
            rows: self.rows,
            width: self.width,
            words: Arc::clone(&self.words),
        }
    }

    /// The array as bytes in its dtype's host storage form.
    pub fn download(&self) -> Result<Vec<u8>> {
        Ok(match self.ty {
            ElementType::F32 | ElementType::S32 | ElementType::U32 => {
                self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
            }
            ElementType::Bf16 => self
                .words
                .iter()
                .flat_map(|w| ((w >> 16) as u16).to_le_bytes())
                .collect(),
            ElementType::U8 => self.words.iter().map(|w| *w as u8).collect(),
        })
    }

    pub fn element_type(&self) -> Result<ElementType> {
        Ok(self.ty)
    }

    pub fn dims(&self) -> Result<Vec<i64>> {
        Ok(vec![i64::from(self.rows), i64::from(self.width)])
    }
}

/// How a program takes a parameter: shared, or donated (the program's
/// result replaces it).
pub enum Arg<'a> {
    Keep(&'a Buffer),
    Donate(Buffer),
}

impl Arg<'_> {
    fn buffer(&self) -> &Buffer {
        match self {
            Arg::Keep(b) => b,
            Arg::Donate(b) => b,
        }
    }
}

/// A compiled program and the signature it was traced with.
pub struct Program {
    /// `None` on a dry device that traced the program without compiling it.
    pub exe: Option<Compiled>,
    pub sig: Signature,
    pub text_bytes: usize,
}

/// A compiled program's artifacts: one compiled phase program after another.
pub struct Compiled {
    pub phases: Vec<(PathBuf, Manifest)>,
}

/// Where a device runs programs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Platform {
    Simulator { target: Target },
    System { cmaddr: String, target: Target },
}

impl Platform {
    fn target(&self) -> Target {
        match self {
            Platform::Simulator { target } | Platform::System { target, .. } => *target,
        }
    }
}

/// Fields drop in order: the compiled programs go before anything else.
/// A kept pool's page move the next fire's table carries: `rows` rows of
/// `export` from row `src_row` to row `dst_row` (pages of `page_size`
/// rows, strided over the row of PEs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolMove {
    pub export: String,
    pub src_row: u64,
    pub dst_row: u64,
    pub rows: u32,
    pub page_size: u32,
}

pub struct Device {
    /// Page moves queued for the next table-driven run (kept pools).
    moves: Mutex<Vec<PoolMove>>,
    cache: Mutex<HashMap<[u8; 32], Arc<Program>>>,
    platform: Platform,
    kind: String,
    /// Compiled artifacts live here ([`disk_cache`]).
    disk: PathBuf,
    /// A dry device keeps every program text it was handed, and runs
    /// nothing: pools and weights hold no buffers.
    dry: Option<Mutex<Vec<String>>>,
    /// Whether a dry device also compiles what it traces.
    dry_compiles: bool,
    /// Resident programs: a `fabric-run --serve` per compiled program that
    /// keeps it loaded and runs spec after spec (guest stages, which run
    /// the same program every step).
    servers: Mutex<Servers>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("kind", &self.kind)
            .field("platform", &self.platform)
            .finish()
    }
}

/// The compile cache the boot config names (`[cerebras] cache_dir`: the
/// worker's `$PIE_HOME/cache/cerebras`), set once when an engine opens.
static CACHE_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Sets the compile cache's default directory (the first call wins).
pub fn set_cache_dir(dir: PathBuf) {
    let _ = CACHE_DIR.set(dir);
}

/// Where compiled programs are cached: the boot config's `cache_dir`, else
/// `PIE_CEREBRAS_CACHE` (a development override where no config set one:
/// tests, examples), else `$XDG_CACHE_HOME/pie/cerebras` or
/// `~/.cache/pie/cerebras`.
pub(crate) fn disk_cache() -> PathBuf {
    if let Some(dir) = CACHE_DIR.get() {
        return dir.clone();
    }
    match std::env::var_os("PIE_CEREBRAS_CACHE") {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(std::env::temp_dir)
            .join("pie")
            .join("cerebras"),
    }
}

impl Device {
    /// A device that compiles and runs programs on `platform`.
    pub fn open(platform: Platform) -> Result<Device> {
        if !Cslc::available() {
            return Err(Fault::NoDevice {
                detail: "the Cerebras SDK compiler (cslc) is not on this box".to_string(),
            });
        }
        if !crate::sdk::Sdk::available() {
            return Err(Fault::NoDevice {
                detail: format!(
                    "no SDK runtime at {}",
                    crate::sdk::Sdk::default_lib_dir().display()
                ),
            });
        }
        let kind = match &platform {
            Platform::Simulator { target } => format!("simfab-{}", target_name(*target)),
            Platform::System { target, .. } => format!("cs-{}", target_name(*target)),
        };
        tracing::info!(kind, "cerebras device open");
        Ok(Device {
            cache: Mutex::new(HashMap::new()),
            platform,
            kind,
            disk: disk_cache(),
            dry: None,
            dry_compiles: false,
            servers: Mutex::new(Servers::default()),
            moves: Mutex::new(Vec::new()),
        })
    }

    /// A device that traces and runs nothing: every program text it is
    /// handed is kept ([`Device::dry_texts`]); with `compile`, each text is
    /// compiled with `cslc` too, but never run.
    pub fn dry(compile: bool) -> Result<Device> {
        if compile && !Cslc::available() {
            return Err(Fault::NoDevice {
                detail: "the Cerebras SDK compiler (cslc) is not on this box".to_string(),
            });
        }
        Ok(Device {
            cache: Mutex::new(HashMap::new()),
            platform: Platform::Simulator {
                target: Target::Wse3,
            },
            kind: "dry".to_string(),
            disk: disk_cache(),
            dry: Some(Mutex::new(Vec::new())),
            dry_compiles: compile,
            servers: Mutex::new(Servers::default()),
            moves: Mutex::new(Vec::new()),
        })
    }

    #[must_use]
    pub fn is_dry(&self) -> bool {
        self.dry.is_some()
    }

    #[must_use]
    pub fn dry_texts(&self) -> Vec<String> {
        self.dry.as_ref().map_or_else(Vec::new, |texts| {
            texts.lock().expect("texts lock poisoned").clone()
        })
    }

    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    #[must_use]
    pub fn platform(&self) -> &Platform {
        &self.platform
    }

    /// Device memory: one PE's (48 KiB, or the `pe_bytes` switch), until
    /// placement spans a rectangle.
    #[must_use]
    pub fn memory(&self) -> Option<u64> {
        Some(crate::switches::parse("pe_bytes").unwrap_or(48 << 10))
    }

    /// Lands `bytes` (the dtype's host storage form) as a `[rows, width]`
    /// array of `dtype`.
    pub fn upload(&self, dtype: Dtype, rows: u32, width: u32, bytes: &[u8]) -> Result<Buffer> {
        Buffer::from_bytes(dtype, rows, width, bytes).ok_or_else(|| Fault::Unbound {
            what: format!("a {dtype:?} plane, which has no storage form on this device"),
        })
    }

    /// Lands `words` as a `[len, 1]` array of `dtype`.
    pub fn upload_words(&self, dtype: Dtype, words: Vec<u32>) -> Result<Buffer> {
        let len = words.len() as u32;
        Buffer::new(dtype, len, 1, words).ok_or_else(|| Fault::Unbound {
            what: format!("a {dtype:?} vector, which has no storage form on this device"),
        })
    }

    /// Zeros of `dtype` over `[rows, width]`, or nothing on a dry device.
    pub fn zeros_or_dry(&self, dtype: Dtype, rows: u32, width: u32) -> Result<Option<Buffer>> {
        if self.is_dry() {
            return Ok(None);
        }
        self.zeros(dtype, rows, width).map(Some)
    }

    pub fn zeros(&self, dtype: Dtype, rows: u32, width: u32) -> Result<Buffer> {
        Buffer::new(dtype, rows, width, vec![0; rows as usize * width as usize]).ok_or_else(|| {
            Fault::Unbound {
                what: format!("a {dtype:?} plane, which has no storage form on this device"),
            }
        })
    }

    /// The compiled form of `text` (a rendered program, see
    /// [`crate::trace::Tracer::finish`]), from the cache or freshly built.
    pub fn program(&self, text: &str, sig: Signature) -> Result<Arc<Program>> {
        let key = *blake3::hash(text.as_bytes()).as_bytes();
        if let Some(hit) = self.cache.lock().expect("cache lock poisoned").get(&key) {
            return Ok(Arc::clone(hit));
        }
        if let Some(texts) = &self.dry {
            texts
                .lock()
                .expect("texts lock poisoned")
                .push(text.to_string());
        }
        let exe = if self.dry.is_none() || self.dry_compiles {
            Some(self.compile(text)?)
        } else {
            None
        };
        let program = Arc::new(Program {
            exe,
            sig,
            text_bytes: text.len(),
        });
        self.cache
            .lock()
            .expect("cache lock poisoned")
            .insert(key, Arc::clone(&program));
        Ok(program)
    }

    fn compile(&self, text: &str) -> Result<Compiled> {
        let mut phases = Vec::new();
        for phase in text.split(crate::trace::PHASE_SEPARATOR) {
            let rendered = crate::trace::unrender(phase).ok_or_else(|| Fault::Unbound {
                what: "a program text that is not a rendered CSL program".to_string(),
            })?;
            if rendered.manifest.host.is_some() {
                phases.push((PathBuf::new(), rendered.manifest));
                continue;
            }
            let hex = blake3::hash(phase.as_bytes()).to_hex().to_string();
            // A system's programs are laid out on the whole wafer, so they
            // are cached apart from the simulator's.
            let system = matches!(self.platform, Platform::System { .. });
            let dir = self.disk.join(format!(
                "{hex}-{}{}",
                target_name(self.platform.target()),
                if system { "-system" } else { "" }
            ));
            let out = dir.join("out");
            // A finished compile leaves the RPC manifest and the PE binary;
            // a failed one (the linker out of PE memory) may leave the
            // manifest alone, which must not pass for a program.
            let bin = out.join("bin");
            let finished = bin.join("out_rpc.json").is_file() && bin.join("out_0_0.elf").is_file();
            if !finished {
                let _ = std::fs::remove_dir_all(&out);
                std::fs::create_dir_all(&dir).map_err(|e| Fault::Unbound {
                    what: format!("the program cache at {}: {e}", dir.display()),
                })?;
                std::fs::write(dir.join("gather.csl"), &rendered.layout)
                    .map_err(|e| io_fault(&dir, e))?;
                std::fs::write(dir.join("pe.csl"), &rendered.pe).map_err(|e| io_fault(&dir, e))?;
                let started = std::time::Instant::now();
                let (w, h) = rendered.manifest.rect;
                let mut compile = Compile::memcpy(
                    Arch::from(self.platform.target()),
                    dir.join("gather.csl"),
                    w,
                    h,
                    &out,
                );
                if system {
                    compile.fabric_dims = wafer_dims(self.platform.target());
                }
                if let Err(e) = Cslc::find().and_then(|cslc| cslc.compile(&compile)) {
                    let _ = std::fs::remove_dir_all(&out);
                    return Err(Fault::Compile { why: e.to_string() });
                }
                tracing::info!(
                    bytes = phase.len(),
                    ms = started.elapsed().as_millis() as u64,
                    "cerebras compiled a phase program"
                );
            }
            phases.push((out, rendered.manifest));
        }
        Ok(Compiled { phases })
    }

    #[must_use]
    pub fn compiled(&self) -> usize {
        self.cache.lock().expect("cache lock poisoned").len()
    }

    /// Runs `program` over `args` (in its parameter order); returns its
    /// results. Every buffer of the fire lives on the host between the
    /// phases; each phase program uploads what it names and downloads it
    /// back.
    pub fn run(&self, program: &Program, args: Vec<Arg<'_>>, wait: bool) -> Result<Vec<Buffer>> {
        let _ = wait;
        self.run_with(program, args, false)
    }

    /// `run`; a `resident` program stays loaded in a `fabric-run --serve`
    /// of its own between runs (the simulator only; a system runs a phase
    /// a process as before).
    /// Queues page moves for the next run's table (kept pools): the host
    /// no longer edits those pools, the device moves the rows.
    pub fn queue_moves(&self, moves: Vec<PoolMove>) {
        self.moves
            .lock()
            .expect("moves lock poisoned")
            .extend(moves);
    }

    pub fn run_with(
        &self,
        program: &Program,
        args: Vec<Arg<'_>>,
        resident: bool,
    ) -> Result<Vec<Buffer>> {
        if self.is_dry() {
            return Ok(Vec::new());
        }
        let exe = program.exe.as_ref().ok_or(Fault::Deviceless)?;
        let sig = &program.sig;
        if args.len() != sig.params.len() {
            return Err(Fault::Unbound {
                what: format!(
                    "{} arguments for {} parameters",
                    args.len(),
                    sig.params.len()
                ),
            });
        }
        // The host state: every buffer's words, parameters first, pack
        // windows cut from the pack, everything else zeros.
        let mut state: HashMap<String, Vec<u32>> = HashMap::new();
        for (arg, name) in args.iter().zip(&sig.names) {
            state.insert(name.clone(), arg.buffer().words.to_vec());
        }
        if let Some(pack) = state.get("pack").cloned() {
            for (name, offset, n) in &sig.packed_views {
                let (from, to) = (*offset as usize, (*offset + *n) as usize);
                let mut words: Vec<u32> =
                    pack.get(from..to.min(pack.len())).unwrap_or(&[]).to_vec();
                words.resize(*n as usize, 0);
                state.insert(name.clone(), words);
            }
        }
        for e in &sig.buffers {
            let n = e.rows as usize * e.width as usize;
            let words = state.entry(e.name.clone()).or_insert_with(|| vec![0; n]);
            if words.len() != n {
                return Err(Fault::Unbound {
                    what: format!(
                        "{}: {} words for a {}x{} buffer",
                        e.name,
                        words.len(),
                        e.rows,
                        e.width
                    ),
                });
            }
        }
        let scratch = tempfile::Builder::new()
            .prefix("pie-cerebras-fire-")
            .tempdir()
            .map_err(|e| io_fault(Path::new("."), e))?;
        let cmaddr = match &self.platform {
            Platform::System { cmaddr, .. } => Some(cmaddr.as_str()),
            Platform::Simulator { .. } => None,
        };
        let started = std::time::Instant::now();
        // `PIE_CEREBRAS_SERVE_ALL=1` keeps every program resident, the
        // model's phases included (an experiment: a phase must then leave
        // its scratch as it found it).
        let all = crate::switches::on("serve_all");
        let servers = ((resident || all) && cmaddr.is_none() && serving()).then_some(&self.servers);
        let persistent: std::collections::HashSet<String> = sig
            .params
            .iter()
            .zip(&sig.names)
            .filter(|(source, _)| matches!(source, crate::trace::Source::Weight { .. }))
            .map(|(_, name)| name.clone())
            .collect();
        let moves: Vec<PoolMove> =
            std::mem::take(&mut *self.moves.lock().expect("moves lock poisoned"));
        let ran = run_phases(
            &fabric_run()?,
            target_name(self.platform.target()),
            cmaddr,
            &exe.phases,
            &mut state,
            scratch.path(),
            servers,
            &persistent,
            &moves,
        );
        if let Err(e) = ran {
            // A failed fire's scratch (specs, inputs) is the replay material.
            if crate::switches::get("keep").is_some() {
                let kept = scratch.keep();
                eprintln!("kept the failed fire's scratch at {}", kept.display());
            }
            return Err(e);
        }
        tracing::debug!(
            phases = exe.phases.len(),
            ms = started.elapsed().as_millis() as u64,
            "cerebras ran a fire"
        );
        let mut outs = Vec::with_capacity(sig.outputs.len());
        for (name, (dtype, rows, width)) in &sig.outputs {
            let words = state.remove(name).ok_or_else(|| Fault::Unbound {
                what: format!("result {name}, which no phase holds"),
            })?;
            outs.push(
                Buffer::new(*dtype, *rows, *width, words).ok_or_else(|| Fault::Unbound {
                    what: format!("result {name} of {dtype:?}, which has no storage form"),
                })?,
            );
        }
        if crate::switches::get("keep").is_some() {
            let kept = scratch.keep();
            tracing::info!(dir = %kept.display(), "kept a fire's scratch directory");
        }
        Ok(outs)
    }
}

/// Runs compiled phase programs in order over the host `state` (every
/// buffer's words by name): before each phase its views are cut from their
/// roots and its buffers laid out per PE; after it, what came back is pasted
/// into the state (views back into their roots).
#[allow(clippy::too_many_arguments)]
pub fn run_phases(
    runner: &Path,
    target: &str,
    cmaddr: Option<&str>,
    phases: &[(PathBuf, Manifest)],
    state: &mut HashMap<String, Vec<u32>>,
    scratch: &Path,
    servers: Option<&Mutex<Servers>>,
    persistent: &std::collections::HashSet<String>,
    moves: &[PoolMove],
) -> Result<()> {
    let mut moves_left: Vec<PoolMove> = moves.to_vec();
    for (at, (dir, manifest)) in phases.iter().enumerate() {
        for view in &manifest.views {
            let root = state.get(&view.root).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "phase {at} views {}, which the fire has no buffer for",
                    view.root
                ),
            })?;
            let (from, to) = (view.offset as usize, (view.offset + view.len) as usize);
            let mut words = root.get(from..to.min(root.len())).unwrap_or(&[]).to_vec();
            words.resize(view.len as usize, 0);
            state.insert(view.name.clone(), words);
        }
        if let Some(host) = &manifest.host {
            run_host(host, state, at)?;
            paste_views(manifest, state);
            continue;
        }
        let plans: Vec<LaneLayout> = manifest
            .lanes
            .iter()
            .map(|lane| LaneLayout::plan(lane, state, at))
            .collect::<Result<_>>()?;
        // A buffer belongs to the first plan that names it.
        let plan_of = |name: &str| plans.iter().find(|p| p.owns(name));
        let (w, h) = manifest.rect;
        // A whole fire over rows of PEs: row `y` runs its own table, and
        // every row's PEs mirror the fire's PEs across, so a buffer is laid
        // out for the `w` PEs of a row.
        let row_tables: Vec<&crate::fabric::Table> = if manifest.tables.is_empty() {
            manifest.table.iter().collect()
        } else {
            manifest.tables.iter().collect()
        };
        let split = !manifest.tables.is_empty();
        let lay_pes = if split {
            w as usize
        } else {
            w as usize * h as usize
        };
        // On a resident program, a persistent buffer (a weight) is uploaded
        // once and never read back: the host's copy is the truth.
        let skip: Vec<&str> = if servers.is_some() {
            manifest
                .exports
                .iter()
                .filter(|e| persistent.contains(&e.name))
                .map(|e| e.name.as_str())
                .collect()
        } else {
            Vec::new()
        };
        let mut spec = serde_json::json!({
            "target": target,
            "entry": manifest.entry,
            "rect": [w, h],
            "persistent": skip,
            "inputs": [],
            "outputs": [],
        });
        if let Some(cmaddr) = cmaddr {
            spec["cmaddr"] = serde_json::Value::String(cmaddr.to_string());
        }
        // Every export laid out per PE (what the device gets).
        let mut sent_of: HashMap<String, Vec<u32>> = HashMap::new();
        for export in &manifest.exports {
            let words = state.get(&export.name).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "phase {at} names {}, which the fire has no buffer for",
                    export.name
                ),
            })?;
            let n = export.rows as usize * export.width as usize;
            if words.len() != n {
                return Err(Fault::Unbound {
                    what: format!(
                        "{}: {} words for a {}x{} buffer",
                        export.name,
                        words.len(),
                        export.rows,
                        export.width
                    ),
                });
            }
            // A packed export: the bf16 words two halves a word, laid out
            // as a half-width array.
            let pairs: Vec<u32>;
            let (words, width) = if export.packed {
                pairs = words
                    .chunks(2)
                    .map(|c| (c[0] >> 16) | (c.get(1).copied().unwrap_or(0) >> 16 << 16))
                    .collect();
                (&pairs, export.width as usize / 2)
            } else {
                (words, export.width as usize)
            };
            let laid = match plan_of(&export.name).and_then(|p| p.lay_out(&export.name, words)) {
                Some(laid) => laid,
                None => lay_out(words, export.rows as usize, width, export.shard, lay_pes),
            };
            sent_of.insert(export.name.clone(), laid);
        }
        let pes = w as usize * h as usize;
        let packed_back =
            |e: &crate::fabric::Export| e.packed && e.role != crate::fabric::Symbol::Input;
        let comes_back = |e: &crate::fabric::Export| {
            !skip.contains(&e.name.as_str())
                && (!e.packed || packed_back(e))
                && e.role != crate::fabric::Symbol::Input
        };
        let write_in = |file: &str, words: &[u32]| -> Result<()> {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(scratch.join(file), bytes).map_err(|e| io_fault(scratch, e))
        };
        let read_out = |file: &str| -> Result<Vec<u32>> {
            let bytes = std::fs::read(scratch.join(file)).map_err(|e| io_fault(scratch, e))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect())
        };
        // The arena arrays are declared at the largest row's size; every
        // row's kept chunks must agree for a chunk to be persistent.
        let chunks = row_tables.iter().map(|t| t.chunks() as usize).max();
        let chunk_len = |c: usize| {
            row_tables
                .iter()
                .map(|t| t.chunk_words(c as u64) as usize)
                .max()
                .unwrap_or(1)
        };
        let keep_chunks = row_tables
            .iter()
            .map(|t| t.keep_chunks as usize)
            .min()
            .unwrap_or(0);
        // Each row's packed arena (its own words a PE) and ops.
        let mut packed: Vec<(Vec<u32>, usize)> = Vec::new();
        if !row_tables.is_empty() {
            // The table-driven program: one arena and one op table a PE.
            // The fire's page moves ride the first table that has rows for
            // them.
            let ops_max = row_tables
                .iter()
                .map(|t| t.ops_words as usize)
                .max()
                .unwrap_or(1);
            let mut ops_all = vec![0u32; pes * ops_max];
            for (g, table) in row_tables.iter().enumerate() {
                // The moves of the pools this row holds.
                let mut moving: Vec<PoolMove> = Vec::new();
                if table.move_rows > 0 {
                    let (mine, rest): (Vec<PoolMove>, Vec<PoolMove>) =
                        std::mem::take(&mut moves_left)
                            .into_iter()
                            .partition(|m| table.arena.iter().any(|(n, ..)| *n == m.export));
                    moving = mine;
                    moves_left = rest;
                }
                let (arena, ops) =
                    pack_table(table, &manifest.exports, &sent_of, lay_pes, at, &moving)?;
                let own = table.ops_words as usize;
                if split {
                    for x in 0..lay_pes {
                        let dst = (g * lay_pes + x) * ops_max;
                        ops_all[dst..dst + own].copy_from_slice(&ops[x * own..(x + 1) * own]);
                    }
                } else {
                    ops_all = ops;
                }
                packed.push((arena, table.arena_words as usize));
            }
            // The kept slots (weights, pools, constant tables): one layout
            // over the rectangle, each slot its own symbol, uploaded once
            // per resident server and never read back. Every row's words
            // are the fire's PEs' (a slot is laid out for the `w` PEs of a
            // row; the rows that do not name it get the same words).
            let kept_slots: Vec<(String, usize, usize)> = row_tables
                .first()
                .map(|t| {
                    t.arena
                        .iter()
                        .filter(|(_, off, _)| {
                            *off < keep_chunks as u64 * crate::fabric::ARENA_CHUNK
                        })
                        .map(|(n, o, w)| (n.clone(), *o as usize, *w as usize))
                        .collect()
                })
                .unwrap_or_default();
            if !kept_slots.is_empty() {
                spec["init"] = serde_json::Value::String("init".to_string());
            }
            for (name, off, words) in &kept_slots {
                // A slot no export or constant fills (another class's) goes
                // up as zeros only when nothing better is known: skip it, the
                // class that fills it uploads it.
                let filled = sent_of.contains_key(name)
                    || row_tables
                        .iter()
                        .any(|t| t.consts.iter().any(|(c, _)| c == name));
                if !filled {
                    continue;
                }
                // The row whose packed arena holds the slot's words: any row
                // names every kept slot, the first filled row's words serve.
                let (arena, arena_words) = &packed[0];
                let mut data = Vec::with_capacity(pes * words);
                for pe in 0..pes {
                    let x = if split { pe % lay_pes } else { pe };
                    data.extend_from_slice(
                        &arena[x * arena_words + off..x * arena_words + off + words],
                    );
                }
                write_in(&format!("{name}.in"), &data)?;
                spec["inputs"]
                    .as_array_mut()
                    .expect("array")
                    .push(serde_json::json!({ "name": name, "file": format!("{name}.in") }));
                if servers.is_some() {
                    spec["persistent"]
                        .as_array_mut()
                        .expect("array")
                        .push(serde_json::Value::String(name.clone()));
                }
            }
            for c in keep_chunks..chunks.unwrap_or(1) {
                let lo = c * crate::fabric::ARENA_CHUNK as usize;
                let len = chunk_len(c);
                let mut words = Vec::with_capacity(pes * len);
                for pe in 0..pes {
                    // PE `pe` of the rectangle: row `pe / w` when split.
                    let (g, x) = if split {
                        (pe / lay_pes, pe % lay_pes)
                    } else {
                        (0, pe)
                    };
                    let (arena, arena_words) = &packed[g];
                    let base = x * arena_words + lo;
                    let n = len.min(arena_words.saturating_sub(lo));
                    words.extend_from_slice(&arena[base..base + n]);
                    words.resize((pe + 1) * len, 0);
                }
                let name = format!("arena{c}");
                write_in(&format!("{name}.in"), &words)?;
                spec["inputs"]
                    .as_array_mut()
                    .expect("array")
                    .push(serde_json::json!({ "name": name, "file": format!("{name}.in") }));
                spec["outputs"]
                    .as_array_mut()
                    .expect("array")
                    .push(serde_json::json!({ "name": name, "len": words.len(), "file": format!("{name}.out") }));
            }
            write_in("ops.in", &ops_all)?;
            spec["inputs"]
                .as_array_mut()
                .expect("array")
                .push(serde_json::json!({ "name": "ops", "file": "ops.in" }));
        } else {
            for export in &manifest.exports {
                let laid = &sent_of[&export.name];
                let file = format!("{}.in", export.name);
                write_in(&file, laid)?;
                spec["inputs"]
                    .as_array_mut()
                    .expect("array")
                    .push(serde_json::json!({ "name": export.name, "file": file }));
                if !comes_back(export) {
                    continue;
                }
                spec["outputs"]
                    .as_array_mut()
                    .expect("array")
                    .push(serde_json::json!({
                        "name": export.name, "len": laid.len(), "file": format!("{}.out", export.name)
                    }));
            }
        }
        let spec_path = scratch.join(format!("spec{at}.json"));
        std::fs::write(&spec_path, spec.to_string()).map_err(|e| io_fault(scratch, e))?;
        let started = std::time::Instant::now();
        if let Some(servers) = servers {
            // The program stays loaded: its server runs the spec. A server
            // that fails is dropped (killed); the next run starts a new one.
            let mut pool = servers.lock().expect("servers lock poisoned");
            let ran = pool
                .get(runner, dir, target)?
                .run(&spec_path, phase_timeout(), at);
            if ran.is_err() {
                pool.drop_server(dir);
            }
            ran?;
        } else {
            run_once(runner, dir, &spec_path, scratch, at)?;
        }
        tracing::debug!(
            phase = at,
            pes = format_args!("{w}x{h}"),
            buffers = manifest.exports.len(),
            words = manifest
                .exports
                .iter()
                .map(|e| e.local() as usize * w as usize * h as usize)
                .sum::<usize>(),
            secs = started.elapsed().as_secs_f64(),
            "cerebras ran a phase"
        );
        // What came back, per export (laid out per PE as it went).
        let mut laid_of: HashMap<String, Vec<u32>> = HashMap::new();
        if !row_tables.is_empty() {
            // Every row's arena back (its own words a PE).
            let mut arenas: Vec<Vec<u32>> = packed
                .iter()
                .map(|(_, words)| vec![0u32; lay_pes * words])
                .collect();
            for c in keep_chunks..chunks.unwrap_or(1) {
                let lo = c * crate::fabric::ARENA_CHUNK as usize;
                let len = chunk_len(c);
                let words = read_out(&format!("arena{c}.out"))?;
                for pe in 0..pes {
                    let (g, x) = if split {
                        (pe / lay_pes, pe % lay_pes)
                    } else {
                        (0, pe)
                    };
                    let arena_words = packed[g].1;
                    let n = len.min(arena_words.saturating_sub(lo));
                    if let Some(src) = words.get(pe * len..pe * len + n) {
                        arenas[g][x * arena_words + lo..x * arena_words + lo + n]
                            .copy_from_slice(src);
                    }
                }
            }
            for export in &manifest.exports {
                if !comes_back(export) || export.keep {
                    continue;
                }
                let per = sent_of[&export.name].len() / lay_pes.max(1);
                // The last row holding the buffer has its final words (a
                // handoff only goes down).
                let (g, off) = row_tables
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(g, t)| {
                        t.arena
                            .iter()
                            .find(|(n, ..)| *n == export.name)
                            .map(|(_, o, _)| (g, *o as usize))
                    })
                    .ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: {} has no arena slot", export.name),
                    })?;
                let arena_words = packed[g].1;
                let mut laid = Vec::with_capacity(per * lay_pes);
                for x in 0..lay_pes {
                    laid.extend_from_slice(
                        &arenas[g][x * arena_words + off..x * arena_words + off + per],
                    );
                }
                laid_of.insert(export.name.clone(), laid);
            }
        } else {
            for export in &manifest.exports {
                if !comes_back(export) {
                    continue;
                }
                laid_of.insert(
                    export.name.clone(),
                    read_out(&format!("{}.out", export.name))?,
                );
            }
        }
        for export in &manifest.exports {
            if !comes_back(export) || (!row_tables.is_empty() && export.keep) {
                continue;
            }
            let laid = &laid_of[&export.name];
            // A packed export's host words are paired for the merge and
            // unpacked after.
            let pack = |words: &[u32]| -> Vec<u32> {
                words
                    .chunks(2)
                    .map(|c| (c[0] >> 16) | (c.get(1).copied().unwrap_or(0) >> 16 << 16))
                    .collect()
            };
            let unpack = |words: &[u32]| -> Vec<u32> {
                words
                    .iter()
                    .flat_map(|w| [w << 16, w & 0xFFFF_0000])
                    .collect()
            };
            let words = match plan_of(&export.name) {
                Some(p) => {
                    let current = state.get(&export.name).cloned().unwrap_or_default();
                    let current = if export.packed {
                        pack(&current)
                    } else {
                        current
                    };
                    let sent = &sent_of[&export.name];
                    let mut words = p.gather_up(&export.name, laid, sent, current, &laid_of);
                    if p.sums(&export.name) && export.dtype == Dtype::Bf16 {
                        for v in words.iter_mut() {
                            *v = crate::bench::round_bf16(f32::from_bits(*v)).to_bits();
                        }
                    }
                    words
                }
                _ => {
                    let mut words = gather_up(
                        laid,
                        export.rows as usize,
                        export.width as usize / if export.packed { 2 } else { 1 },
                        export.shard,
                        lay_pes,
                    );
                    // Partial sums are added in f32 and rounded here.
                    if matches!(
                        export.shard,
                        kernels_cerebras::phase::Shard::SumCols { .. }
                            | kernels_cerebras::phase::Shard::SumGrid { .. }
                    ) && export.dtype == Dtype::Bf16
                    {
                        for v in words.iter_mut() {
                            *v = crate::bench::round_bf16(f32::from_bits(*v)).to_bits();
                        }
                    }
                    words
                }
            };
            let words = if export.packed { unpack(&words) } else { words };
            state.insert(export.name.clone(), words);
        }
        paste_views(manifest, state);
    }
    // Every page move must have found the row (and table) holding its pool.
    if let Some(m) = moves_left.first() {
        return Err(Fault::Unbound {
            what: format!(
                "{} page moves found no table holding their pool (the first: {})",
                moves_left.len(),
                m.export
            ),
        });
    }
    Ok(())
}

/// A table-driven program's device data: every PE's arena (its exports at
/// their offsets, scratch zeros) back to back, and every PE's op rows
/// (opcode, then the arguments resolved to arena offsets and words).
fn pack_table(
    table: &crate::fabric::Table,
    exports: &[crate::fabric::Export],
    sent_of: &HashMap<String, Vec<u32>>,
    pes: usize,
    at: usize,
    moves: &[PoolMove],
) -> Result<(Vec<u32>, Vec<u32>)> {
    use crate::fabric::{
        MOVE_ROW_WORDS, OP_BROADCAST, OP_BROADCAST_FROM, OP_GATHER, OP_HANDOFF, OP_NOP, OP_ON,
        OP_REDUCE, OP_ROOT, TableOp, op_words,
    };
    use kernels_cerebras::csl::Arg;
    use kernels_cerebras::phase::FabricOp;
    let arena_words = table.arena_words as usize;
    let mut arena = vec![0u32; pes * arena_words];
    let offset = |name: &str| -> Option<usize> {
        table
            .arena
            .iter()
            .find(|(n, ..)| n == name)
            .map(|(_, o, _)| *o as usize)
    };
    for (name, off, words) in &table.arena {
        let Some(laid) = sent_of.get(name) else {
            continue;
        };
        // An output's slot may be a dead input's: nothing goes up for it.
        if exports
            .iter()
            .any(|e| e.name == *name && e.role == crate::fabric::Symbol::Output)
        {
            continue;
        }
        let per = laid.len() / pes.max(1);
        let n = per.min(*words as usize);
        for pe in 0..pes {
            let dst = pe * arena_words + *off as usize;
            arena[dst..dst + n].copy_from_slice(&laid[pe * per..pe * per + n]);
        }
    }
    // The constant tables, the same on every PE.
    for (name, words) in &table.consts {
        let Some(off) = offset(name) else { continue };
        let n = words.len().min(arena_words.saturating_sub(off));
        for pe in 0..pes {
            let dst = pe * arena_words + off;
            arena[dst..dst + n].copy_from_slice(&words[..n]);
        }
    }
    let unbound = |what: String| Fault::Unbound {
        what: format!("phase {at}: the op table cannot carry {what}"),
    };
    // A lane header word for one PE.
    let word = |name: &str, i: u32, pe: usize| -> Option<i32> {
        let laid = sent_of.get(name)?;
        let per = laid.len() / pes.max(1);
        laid.get(pe * per + i as usize).map(|w| *w as i32)
    };
    // `name[i]`, `name[i] * c`, `(name[a] - name[b]) * c`: a header word
    // expression as the lowerings assemble them.
    type HeaderWord = (String, u32);
    let header_expr = |e: &str| -> Option<(HeaderWord, Option<HeaderWord>, i32)> {
        let (body, mult) = match e.rsplit_once(" * ") {
            Some((lhs, rhs)) if rhs.trim().parse::<i32>().is_ok() => {
                (lhs.trim(), rhs.trim().parse::<i32>().ok()?)
            }
            _ => (e, 1),
        };
        let body = body.trim().trim_start_matches('(').trim_end_matches(')');
        let word = |t: &str| -> Option<(String, u32)> {
            let (name, idx) = t.trim().split_once('[')?;
            Some((name.to_string(), idx.trim_end_matches(']').parse().ok()?))
        };
        match body.split_once(" - ") {
            Some((l, r)) => Some((word(l)?, Some(word(r)?), mult)),
            None => Some((word(body)?, None, mult)),
        }
    };
    let resolve = |arg: &Arg, pe: usize| -> Result<i32> {
        Ok(match arg {
            Arg::Ptr(b) => {
                offset(&b.name).ok_or_else(|| unbound(format!("a pointer to {}", b.name)))? as i32
            }
            Arg::Scratch(name, _) => {
                offset(name).ok_or_else(|| unbound(format!("a pointer to {name}")))? as i32
            }
            Arg::Int(v) => *v as i32,
            Arg::Bool(v) => i32::from(*v),
            Arg::Float(v) => v.to_bits() as i32,
            Arg::PeTimes(c) => (pe as i64 * c) as i32,
            Arg::Word(name, i) => {
                word(name, *i, pe).ok_or_else(|| unbound(format!("word {i} of {name}")))?
            }
            Arg::Dummy(elem) => {
                offset("k_dummy").ok_or_else(|| unbound("the dummy slot".to_string()))? as i32
                    + i32::from(*elem != "f32")
            }
            Arg::Expr(e) => {
                let e = e.trim();
                if let Ok(v) = e.parse::<i64>() {
                    v as i32
                } else if let Some(rest) = e.strip_prefix("@ptrcast(") {
                    let name = rest.trim_end_matches(')').rsplit('&').next().unwrap_or("");
                    if name.starts_with("k_dummy") {
                        offset("k_dummy").ok_or_else(|| unbound("the dummy slot".to_string()))?
                            as i32
                            + i32::from(name != "k_dummy_f32")
                    } else {
                        offset(name).ok_or_else(|| unbound(format!("a pointer to {name}")))? as i32
                    }
                } else if let Some(((n1, i1), second, mult)) = header_expr(e) {
                    let mut v =
                        word(&n1, i1, pe).ok_or_else(|| unbound(format!("word {i1} of {n1}")))?;
                    if let Some((n2, i2)) = second {
                        v -= word(&n2, i2, pe)
                            .ok_or_else(|| unbound(format!("word {i2} of {n2}")))?;
                    }
                    v * mult
                } else {
                    return Err(unbound(format!("the expression `{e}`")));
                }
            }
        })
    };
    // The op table a PE: each row its length word then its words, a zero
    // length ending the table, zeros to the capacity.
    let ops_words = table.ops_words as usize;
    let mut ops = vec![0i32; pes * ops_words];
    // The page moves, in the reserved rows: a copy on one PE, or a copy
    // into the scratch page on the source PE, its broadcast, and a copy
    // out on the destination PE.
    let mut move_ops: Vec<Vec<i32>> = Vec::new();
    if !moves.is_empty() {
        let copy = table
            .kernels
            .iter()
            .position(|k| *k == kernels_cerebras::library::Kernel::Copy)
            .ok_or_else(|| unbound("a page move without k_copy".to_string()))?
            as i32;
        let scratch = offset("move_scratch")
            .ok_or_else(|| unbound("a page move without its scratch".to_string()))?
            as i32;
        for m in moves {
            let e = exports
                .iter()
                .find(|e| e.name == m.export)
                .ok_or_else(|| unbound(format!("a move of {}", m.export)))?;
            let slot =
                offset(&e.name).ok_or_else(|| unbound(format!("a move of {}", m.export)))? as u64;
            let row_words = u64::from(e.width) / if e.packed { 2 } else { 1 };
            let ps = u64::from(m.page_size.max(1));
            let place = |row: u64| -> (i32, i32) {
                let page = row / ps;
                let pe = (page % pes as u64) as i32;
                let local = ((page / pes as u64) * ps + row % ps) * row_words;
                (pe, (slot + local) as i32)
            };
            // Whole runs within a page, in scratch-sized pieces.
            let mut done = 0u64;
            while done < u64::from(m.rows) {
                let left = u64::from(m.rows) - done;
                let fit = (crate::fabric::MOVE_SCRATCH / row_words.max(1)).max(1);
                let n = left
                    .min(fit)
                    .min(ps - (m.src_row + done) % ps)
                    .min(ps - (m.dst_row + done) % ps);
                let (spe, soff) = place(m.src_row + done);
                let (dpe, doff) = place(m.dst_row + done);
                let words = (n * row_words) as i32;
                if spe == dpe {
                    move_ops.push(vec![OP_ON, spe, copy, doff, 0, soff, 0, words]);
                } else {
                    move_ops.push(vec![OP_ON, spe, copy, scratch, 0, soff, 0, words]);
                    move_ops.push(vec![OP_BROADCAST_FROM, spe, scratch, words]);
                    move_ops.push(vec![OP_ON, dpe, copy, doff, 0, scratch, 0, words]);
                }
                done += n;
            }
        }
        if move_ops.len() > table.move_rows as usize {
            return Err(Fault::Unbound {
                what: format!(
                    "phase {at}: {} page-move rows for a table with {} reserved",
                    move_ops.len(),
                    table.move_rows
                ),
            });
        }
    }
    for pe in 0..pes {
        let mut at_word = pe * ops_words;
        for (r, op) in table.ops.iter().enumerate() {
            let len = op_words(op) as usize;
            if at_word + 1 + len >= (pe + 1) * ops_words {
                return Err(Fault::Unbound {
                    what: format!("phase {at}: the op table overflows its {ops_words} words"),
                });
            }
            ops[at_word] = len as i32;
            let row = &mut ops[at_word + 1..at_word + 1 + len];
            at_word += 1 + len;
            if let TableOp::Nop = op {
                row[0] = OP_NOP;
                if let Some(filled) = move_ops.get(r) {
                    debug_assert!(filled.len() <= MOVE_ROW_WORDS as usize);
                    row[..filled.len()].copy_from_slice(filled);
                }
                continue;
            }
            match op {
                TableOp::Call {
                    root,
                    on,
                    kernel,
                    args,
                } => {
                    let mut at_col = 0;
                    if let Some(pe_on) = on {
                        row[0] = OP_ON;
                        row[1] = *pe_on as i32;
                        row[2] = *kernel as i32;
                        at_col = 3;
                    } else if *root {
                        row[0] = OP_ROOT;
                        row[1] = *kernel as i32;
                        at_col = 2;
                    } else {
                        row[0] = *kernel as i32;
                        at_col += 1;
                    }
                    for a in args {
                        row[at_col] = resolve(a, pe)?;
                        at_col += 1;
                    }
                }
                TableOp::Collective(FabricOp::Reduce { send, recv, count })
                | TableOp::Collective(FabricOp::Gather { send, recv, count }) => {
                    row[0] = if matches!(op, TableOp::Collective(FabricOp::Reduce { .. })) {
                        OP_REDUCE
                    } else {
                        OP_GATHER
                    };
                    row[1] =
                        offset(send).ok_or_else(|| unbound(format!("a pointer to {send}")))? as i32;
                    row[2] =
                        offset(recv).ok_or_else(|| unbound(format!("a pointer to {recv}")))? as i32;
                    row[3] = *count as i32;
                }
                TableOp::Collective(FabricOp::Broadcast { buf, count }) => {
                    row[0] = OP_BROADCAST;
                    row[1] =
                        offset(buf).ok_or_else(|| unbound(format!("a pointer to {buf}")))? as i32;
                    row[2] = *count as i32;
                }
                TableOp::Collective(FabricOp::Handoff { root, buf, count }) => {
                    row[0] = OP_HANDOFF;
                    row[1] = *root as i32;
                    row[2] =
                        offset(buf).ok_or_else(|| unbound(format!("a pointer to {buf}")))? as i32;
                    row[3] = *count as i32;
                }
                TableOp::Collective(FabricOp::BroadcastFrom { root, buf, count }) => {
                    row[0] = OP_BROADCAST_FROM;
                    row[1] = *root as i32;
                    row[2] =
                        offset(buf).ok_or_else(|| unbound(format!("a pointer to {buf}")))? as i32;
                    row[3] = *count as i32;
                }
                TableOp::Nop => {}
            }
        }
    }
    Ok((arena, ops.into_iter().map(|v| v as u32).collect()))
}

fn target_name(target: Target) -> &'static str {
    match target {
        Target::Wse2 => "wse2",
        Target::Wse3 => "wse3",
    }
}

/// The fabric a system program is compiled against: the whole wafer (the
/// `wafer_dims` switch, `WxH`, else the target's; the SDK runtime reports
/// no fabric size).
pub(crate) fn wafer_dims(target: Target) -> (u32, u32) {
    if let Some((w, h)) = crate::switches::get("wafer_dims").and_then(|v| {
        let (w, h) = v.split_once(['x', 'X'])?;
        Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
    }) {
        return (w, h);
    }
    match target {
        Target::Wse2 => (757, 996),
        Target::Wse3 => (762, 1172),
    }
}

fn io_fault(dir: &Path, e: std::io::Error) -> Fault {
    Fault::Unbound {
        what: format!("{}: {e}", dir.display()),
    }
}
