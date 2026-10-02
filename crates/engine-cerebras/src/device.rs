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

use dtype::Dtype;
use kernels_cerebras::program::{
    BankSplit, ColPlane, HostOp, HostPhase, Manifest, Reduce, Rendered, Segment, picks,
};

use crate::error::{Fault, Result};
use crate::sdk::{Arch, Compile, Cslc, Target};
use crate::trace::Signature;

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
    /// Compiled artifacts live here (`PIE_CEREBRAS_CACHE`, default
    /// `$XDG_CACHE_HOME/pie/cerebras` or `~/.cache/pie/cerebras`).
    disk: PathBuf,
    /// Where each program's sources are also written, for inspection
    /// (`PIE_CEREBRAS_DUMP`).
    dump: Option<PathBuf>,
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

/// The resident servers, at most `most` at a time: a simulator that sits
/// idle still spins, so many of them starve the one at work (on the
/// simulator; `PIE_CEREBRAS_SERVERS` sets the cap, 4 by default). The least
/// recently used goes when one more is needed.
#[derive(Default)]
pub struct Servers {
    live: HashMap<PathBuf, Served>,
    recent: Vec<PathBuf>,
}

impl Servers {
    fn most() -> usize {
        std::env::var("PIE_CEREBRAS_SERVERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(4)
    }

    /// The server of `dir`, spawned when none is live (the least recently
    /// used dropped first when the cap is reached).
    fn get(&mut self, runner: &Path, dir: &Path, target: &str) -> Result<&mut Served> {
        if !self.live.contains_key(dir) {
            while self.live.len() >= Self::most() {
                let Some(old) = self.recent.first().cloned() else {
                    break;
                };
                self.recent.remove(0);
                self.live.remove(&old);
            }
            self.live
                .insert(dir.to_path_buf(), Served::spawn(runner, dir, target)?);
        }
        self.recent.retain(|d| d != dir);
        self.recent.push(dir.to_path_buf());
        Ok(self.live.get_mut(dir).expect("just inserted"))
    }

    fn drop_server(&mut self, dir: &Path) {
        self.live.remove(dir);
        self.recent.retain(|d| d != dir);
    }
}

/// A `fabric-run --serve` child: specs go in on its stdin, `ok`/`err`
/// lines come back on a channel a reader thread feeds.
pub struct Served {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    answers: std::sync::mpsc::Receiver<String>,
    /// The simulator's log in the server's directory.
    log: PathBuf,
}

impl Served {
    fn spawn(runner: &Path, dir: &Path, target: &str) -> Result<Served> {
        let mut child = std::process::Command::new(runner)
            .arg("--serve")
            .arg(dir)
            .arg(target)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| Fault::Compile {
                why: format!("spawning fabric-run --serve: {e}"),
            })?;
        let stdin = child.stdin.take().ok_or(Fault::Compile {
            why: "fabric-run --serve has no stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or(Fault::Compile {
            why: "fabric-run --serve has no stdout".into(),
        })?;
        let (tx, answers) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Served {
            child,
            stdin,
            answers,
            log: dir.join("serve").join("sim.log"),
        })
    }

    /// Runs `spec` on the loaded program, within `timeout`.
    fn run(&mut self, spec: &Path, timeout: std::time::Duration, at: usize) -> Result<()> {
        use std::io::Write;
        writeln!(self.stdin, "{}", spec.display()).map_err(|e| Fault::Compile {
            why: format!("fabric-run --serve went away: {e}"),
        })?;
        let _ = self.stdin.flush();
        match self.answers.recv_timeout(timeout) {
            Ok(line) if line == "ok" => Ok(()),
            Ok(line) => Err(Fault::Compile {
                why: format!("phase {at} on the resident program: {line}"),
            }),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(Fault::Compile {
                why: match simulator_fatal(&self.log) {
                    Some(fatal) => {
                        format!("the resident simulator faulted on phase {at}: {fatal}")
                    }
                    None => format!("phase {at} on the resident program ran past {timeout:?}"),
                },
            }),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(Fault::Compile {
                why: format!("fabric-run --serve died on phase {at}"),
            }),
        }
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Whether compiled programs stay resident between runs
/// (`PIE_CEREBRAS_SERVE=0` turns it off).
fn serving() -> bool {
    std::env::var("PIE_CEREBRAS_SERVE")
        .ok()
        .is_none_or(|v| v != "0")
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("kind", &self.kind)
            .field("platform", &self.platform)
            .finish()
    }
}

fn disk_cache() -> PathBuf {
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
            dump: std::env::var_os("PIE_CEREBRAS_DUMP").map(PathBuf::from),
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
            dump: std::env::var_os("PIE_CEREBRAS_DUMP").map(PathBuf::from),
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
            texts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
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

    /// Device memory: one PE's 48 KB, until placement spans a rectangle.
    #[must_use]
    pub fn memory(&self) -> Option<u64> {
        Some(48 << 10)
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
        if let Some(hit) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
        {
            return Ok(Arc::clone(hit));
        }
        let hex = blake3::hash(text.as_bytes()).to_hex().to_string();
        if let Some(dir) = &self.dump {
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(format!("{hex}.csl.txt")), text);
        }
        if let Some(texts) = &self.dry {
            texts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(text.to_string());
        }
        let exe = if self.dry.is_none() || self.dry_compiles {
            Some(self.compile(text, &hex)?)
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
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, Arc::clone(&program));
        Ok(program)
    }

    fn compile(&self, text: &str, _hex: &str) -> Result<Compiled> {
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
            let dir = self
                .disk
                .join(format!("{hex}-{}", target_name(self.platform.target())));
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
                std::fs::write(dir.join("layout.csl"), &rendered.layout)
                    .map_err(|e| io_fault(&dir, e))?;
                std::fs::write(dir.join("pe.csl"), &rendered.pe).map_err(|e| io_fault(&dir, e))?;
                let started = std::time::Instant::now();
                let (w, h) = rendered.manifest.rect;
                let compile = Compile::memcpy(
                    Arch::from(self.platform.target()),
                    dir.join("layout.csl"),
                    w,
                    h,
                    &out,
                );
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
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
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
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        let all = std::env::var("PIE_CEREBRAS_SERVE_ALL").is_ok_and(|v| v != "0");
        let servers = ((resident || all) && cmaddr.is_none() && serving()).then_some(&self.servers);
        let persistent: std::collections::HashSet<String> = sig
            .params
            .iter()
            .zip(&sig.names)
            .filter(|(source, _)| matches!(source, crate::trace::Source::Weight { .. }))
            .map(|(_, name)| name.clone())
            .collect();
        let moves: Vec<PoolMove> = std::mem::take(
            &mut *self
                .moves
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
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
            if std::env::var_os("PIE_CEREBRAS_KEEP").is_some() {
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
        if std::env::var_os("PIE_CEREBRAS_KEEP").is_some() {
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
        let row_tables: Vec<&kernels_cerebras::program::Table> = if manifest.tables.is_empty() {
            manifest.table.iter().collect()
        } else {
            manifest.tables.iter().collect()
        };
        let split = !manifest.tables.is_empty();
        let lay_pes = if split { w as usize } else { w as usize * h as usize };
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
                None => lay_out(
                    words,
                    export.rows as usize,
                    width,
                    export.shard,
                    lay_pes,
                ),
            };
            sent_of.insert(export.name.clone(), laid);
        }
        let pes = w as usize * h as usize;
        let packed_back = |e: &kernels_cerebras::program::Export| e.packed && e.role != kernels_cerebras::program::Symbol::Input;
        let comes_back = |e: &kernels_cerebras::program::Export| {
            !skip.contains(&e.name.as_str())
                && (!e.packed || packed_back(e))
                && e.role != kernels_cerebras::program::Symbol::Input
        };
        let write_in = |file: &str, words: &[u32]| -> Result<()> {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(scratch.join(file), bytes).map_err(|e| io_fault(scratch, e))
        };
        let read_out = |file: &str| -> Result<Vec<u32>> {
            let bytes = std::fs::read(scratch.join(file)).map_err(|e| io_fault(scratch, e))?;
            Ok(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
        };
        // The arena arrays are declared at the largest row's size; every
        // row's kept chunks must agree for a chunk to be persistent.
        let chunks = row_tables.iter().map(|t| t.chunks() as usize).max();
        let chunk_len = |c: usize| row_tables.iter().map(|t| t.chunk_words(c as u64) as usize).max().unwrap_or(1);
        let keep_chunks = row_tables.iter().map(|t| t.keep_chunks as usize).min().unwrap_or(0);
        // Each row's packed arena (its own words a PE) and ops.
        let mut packed: Vec<(Vec<u32>, usize)> = Vec::new();
        if !row_tables.is_empty() {
            // The table-driven program: one arena and one op table a PE.
            // The fire's page moves ride the first table that has rows for
            // them.
            let ops_max = row_tables.iter().map(|t| t.ops_words as usize).max().unwrap_or(1);
            let mut ops_all = vec![0u32; pes * ops_max];
            for (g, table) in row_tables.iter().enumerate() {
                // The moves of the pools this row holds.
                let mut moving: Vec<PoolMove> = Vec::new();
                if table.move_rows > 0 {
                    let (mine, rest): (Vec<PoolMove>, Vec<PoolMove>) = std::mem::take(&mut moves_left)
                        .into_iter()
                        .partition(|m| table.arena.iter().any(|(n, ..)| *n == m.export));
                    moving = mine;
                    moves_left = rest;
                }
                let (arena, ops) = pack_table(table, &manifest.exports, &sent_of, lay_pes, at, &moving)?;
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
                        .filter(|(_, off, _)| *off < keep_chunks as u64 * kernels_cerebras::program::ARENA_CHUNK)
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
                let filled = sent_of.contains_key(name) || row_tables.iter().any(|t| t.consts.iter().any(|(c, _)| c == name));
                if !filled {
                    continue;
                }
                // The row whose packed arena holds the slot's words: any row
                // names every kept slot, the first filled row's words serve.
                let (arena, arena_words) = &packed[0];
                let mut data = Vec::with_capacity(pes * words);
                for pe in 0..pes {
                    let x = if split { pe % lay_pes } else { pe };
                    data.extend_from_slice(&arena[x * arena_words + off..x * arena_words + off + words]);
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
                let lo = c * kernels_cerebras::program::ARENA_CHUNK as usize;
                let len = chunk_len(c);
                let mut words = Vec::with_capacity(pes * len);
                for pe in 0..pes {
                    // PE `pe` of the rectangle: row `pe / w` when split.
                    let (g, x) = if split { (pe / lay_pes, pe % lay_pes) } else { (0, pe) };
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
            let mut pool = servers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        if std::env::var_os("PIE_CEREBRAS_TRACE_PHASES").is_some() {
            let words: usize = manifest
                .exports
                .iter()
                .map(|e| e.local() as usize * w as usize * h as usize)
                .sum();
            eprintln!(
                "phase {at}: {}x{} PEs, {} buffers, {} words moved, {:.1}s",
                w,
                h,
                manifest.exports.len(),
                words,
                started.elapsed().as_secs_f64()
            );
        }
        // What came back, per export (laid out per PE as it went).
        let mut laid_of: HashMap<String, Vec<u32>> = HashMap::new();
        if !row_tables.is_empty() {
            // Every row's arena back (its own words a PE).
            let mut arenas: Vec<Vec<u32>> = packed.iter().map(|(_, words)| vec![0u32; lay_pes * words]).collect();
            for c in keep_chunks..chunks.unwrap_or(1) {
                let lo = c * kernels_cerebras::program::ARENA_CHUNK as usize;
                let len = chunk_len(c);
                let words = read_out(&format!("arena{c}.out"))?;
                for pe in 0..pes {
                    let (g, x) = if split { (pe / lay_pes, pe % lay_pes) } else { (0, pe) };
                    let arena_words = packed[g].1;
                    let n = len.min(arena_words.saturating_sub(lo));
                    if let Some(src) = words.get(pe * len..pe * len + n) {
                        arenas[g][x * arena_words + lo..x * arena_words + lo + n].copy_from_slice(src);
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
                        t.arena.iter().find(|(n, ..)| *n == export.name).map(|(_, o, _)| (g, *o as usize))
                    })
                    .ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: {} has no arena slot", export.name),
                    })?;
                let arena_words = packed[g].1;
                let mut laid = Vec::with_capacity(per * lay_pes);
                for x in 0..lay_pes {
                    laid.extend_from_slice(&arenas[g][x * arena_words + off..x * arena_words + off + per]);
                }
                laid_of.insert(export.name.clone(), laid);
            }
        } else {
            for export in &manifest.exports {
                if !comes_back(export) {
                    continue;
                }
                laid_of.insert(export.name.clone(), read_out(&format!("{}.out", export.name))?);
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
                words.iter().flat_map(|w| [w << 16, w & 0xFFFF_0000]).collect()
            };
            let words = match plan_of(&export.name) {
                Some(p) => {
                    let current = state.get(&export.name).cloned().unwrap_or_default();
                    let current = if export.packed { pack(&current) } else { current };
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
                        kernels_cerebras::program::Shard::SumCols { .. }
                            | kernels_cerebras::program::Shard::SumGrid { .. }
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
    table: &kernels_cerebras::program::Table,
    exports: &[kernels_cerebras::program::Export],
    sent_of: &HashMap<String, Vec<u32>>,
    pes: usize,
    at: usize,
    moves: &[PoolMove],
) -> Result<(Vec<u32>, Vec<u32>)> {
    use kernels_cerebras::csl::Arg;
    use kernels_cerebras::program::{
        op_words, FabricOp, TableOp, MOVE_ROW_WORDS, OP_BROADCAST, OP_BROADCAST_FROM, OP_GATHER, OP_HANDOFF, OP_NOP, OP_ON,
        OP_REDUCE, OP_ROOT,
    };
    let arena_words = table.arena_words as usize;
    let mut arena = vec![0u32; pes * arena_words];
    let offset = |name: &str| -> Option<usize> {
        table.arena.iter().find(|(n, ..)| n == name).map(|(_, o, _)| *o as usize)
    };
    for (name, off, words) in &table.arena {
        let Some(laid) = sent_of.get(name) else {
            continue;
        };
        // An output's slot may be a dead input's: nothing goes up for it.
        if exports.iter().any(|e| e.name == *name && e.role == kernels_cerebras::program::Symbol::Output) {
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
            Some((lhs, rhs)) if rhs.trim().parse::<i32>().is_ok() => (lhs.trim(), rhs.trim().parse::<i32>().ok()?),
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
            Arg::Ptr(b) => offset(&b.name).ok_or_else(|| unbound(format!("a pointer to {}", b.name)))? as i32,
            Arg::Scratch(name, _) => offset(name).ok_or_else(|| unbound(format!("a pointer to {name}")))? as i32,
            Arg::Int(v) => *v as i32,
            Arg::Bool(v) => i32::from(*v),
            Arg::Float(v) => v.to_bits() as i32,
            Arg::PeTimes(c) => (pe as i64 * c) as i32,
            Arg::Word(name, i) => word(name, *i, pe).ok_or_else(|| unbound(format!("word {i} of {name}")))?,
            Arg::Dummy(elem) => {
                offset("k_dummy").ok_or_else(|| unbound("the dummy slot".to_string()))? as i32 + i32::from(*elem != "f32")
            }
            Arg::Expr(e) => {
                let e = e.trim();
                if let Ok(v) = e.parse::<i64>() {
                    v as i32
                } else if let Some(rest) = e.strip_prefix("@ptrcast(") {
                    let name = rest.trim_end_matches(')').rsplit('&').next().unwrap_or("");
                    if name.starts_with("k_dummy") {
                        offset("k_dummy").ok_or_else(|| unbound("the dummy slot".to_string()))? as i32
                            + i32::from(name != "k_dummy_f32")
                    } else {
                        offset(name).ok_or_else(|| unbound(format!("a pointer to {name}")))? as i32
                    }
                } else if let Some(((n1, i1), second, mult)) = header_expr(e) {
                    let mut v = word(&n1, i1, pe).ok_or_else(|| unbound(format!("word {i1} of {n1}")))?;
                    if let Some((n2, i2)) = second {
                        v -= word(&n2, i2, pe).ok_or_else(|| unbound(format!("word {i2} of {n2}")))?;
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
            .position(|k| k == "k_copy")
            .ok_or_else(|| unbound("a page move without k_copy".to_string()))? as i32;
        let scratch = offset("move_scratch").ok_or_else(|| unbound("a page move without its scratch".to_string()))? as i32;
        for m in moves {
            let e = exports
                .iter()
                .find(|e| e.name == m.export)
                .ok_or_else(|| unbound(format!("a move of {}", m.export)))?;
            let slot = offset(&e.name).ok_or_else(|| unbound(format!("a move of {}", m.export)))? as u64;
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
                let fit = (kernels_cerebras::program::MOVE_SCRATCH / row_words.max(1)).max(1);
                let n = left.min(fit).min(ps - (m.src_row + done) % ps).min(ps - (m.dst_row + done) % ps);
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
                TableOp::Call { root, on, kernel, args } => {
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
                    row[1] = offset(send).ok_or_else(|| unbound(format!("a pointer to {send}")))? as i32;
                    row[2] = offset(recv).ok_or_else(|| unbound(format!("a pointer to {recv}")))? as i32;
                    row[3] = *count as i32;
                }
                TableOp::Collective(FabricOp::Broadcast { buf, count }) => {
                    row[0] = OP_BROADCAST;
                    row[1] = offset(buf).ok_or_else(|| unbound(format!("a pointer to {buf}")))? as i32;
                    row[2] = *count as i32;
                }
                TableOp::Collective(FabricOp::Handoff { root, buf, count }) => {
                    row[0] = OP_HANDOFF;
                    row[1] = *root as i32;
                    row[2] = offset(buf).ok_or_else(|| unbound(format!("a pointer to {buf}")))? as i32;
                    row[3] = *count as i32;
                }
                TableOp::Collective(FabricOp::BroadcastFrom { root, buf, count }) => {
                    row[0] = OP_BROADCAST_FROM;
                    row[1] = *root as i32;
                    row[2] = offset(buf).ok_or_else(|| unbound(format!("a pointer to {buf}")))? as i32;
                    row[3] = *count as i32;
                }
                TableOp::Nop => {}
            }
        }
    }
    Ok((arena, ops.into_iter().map(|v| v as u32).collect()))
}

/// One `fabric-run` process for one phase: spawned, bounded by the phase
/// timeout, its log watched for a fault.
fn run_once(runner: &Path, dir: &Path, spec_path: &Path, scratch: &Path, at: usize) -> Result<()> {
    // A simulator that faults can hang instead of exiting: bound the wait,
    // and watch its log for a fatal error (a PE's bad memory access, say).
    // A run the simulator's own startup kills (a signal with no fault
    // logged; seen when two simulators start at once) gets one more try.
    let sim_log = scratch.join("sim.log");
    let mut attempt = 0;
    loop {
        attempt += 1;
        let _ = std::fs::remove_file(&sim_log);
        let mut child = std::process::Command::new(runner)
            .arg(dir)
            .arg(spec_path)
            .stdout(std::process::Stdio::null())
            .spawn()
            .map_err(|e| Fault::Compile {
                why: format!("spawning fabric-run: {e}"),
            })?;
        let deadline = std::time::Instant::now() + phase_timeout();
        let mut polls: u32 = 0;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    polls += 1;
                    if polls.is_multiple_of(40)
                        && let Some(fatal) = simulator_fatal(&sim_log)
                    {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(Fault::Compile {
                            why: format!("the simulator faulted on phase {at}: {fatal}"),
                        });
                    }
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Fault::Compile {
                        why: format!(
                            "fabric-run on phase {at} ran past {:?} and was killed",
                            phase_timeout()
                        ),
                    });
                }
                Err(e) => {
                    return Err(Fault::Compile {
                        why: format!("waiting for fabric-run: {e}"),
                    });
                }
            }
        };
        if status.success() {
            break;
        }
        let crashed = status.code().is_none() && simulator_fatal(&sim_log).is_none();
        if crashed && attempt < 2 {
            tracing::warn!(phase = at, %status, "fabric-run died starting up; trying once more");
            continue;
        }
        return Err(Fault::Compile {
            why: format!("fabric-run failed on phase {at}: {status}"),
        });
    }
    Ok(())
}

/// Pastes a phase's view windows back into their roots.
fn paste_views(manifest: &Manifest, state: &mut HashMap<String, Vec<u32>>) {
    for view in &manifest.views {
        let Some(words) = state.remove(&view.name) else {
            continue;
        };
        if let Some(root) = state.get_mut(&view.root) {
            let from = view.offset as usize;
            let to = (from + words.len()).min(root.len());
            root[from..to].copy_from_slice(&words[..to - from]);
        }
    }
}

/// The pool cell holding `pos` of request `req`: `indices[indptr[req] + pos
/// / ps] · ps + pos % ps`, or none when a table runs out.
fn cell_of(indices: &[i32], indptr: &[i32], req: i32, pos: i64, ps: i64) -> Option<usize> {
    let pos = pos.max(0);
    let first = i64::from(*indptr.get(usize::try_from(req).ok()?)?);
    let at = usize::try_from(first + pos / ps).ok()?;
    let page = i64::from(*indices.get(at)?);
    usize::try_from(page * ps + pos % ps).ok()
}

/// Runs a host phase over the fire's buffers.
fn run_host(host: &HostPhase, state: &mut HashMap<String, Vec<u32>>, at: usize) -> Result<()> {
    let take = |state: &HashMap<String, Vec<u32>>, name: &str| -> Result<Vec<f32>> {
        state
            .get(name)
            .map(|w| w.iter().map(|x| f32::from_bits(*x)).collect())
            .ok_or_else(|| Fault::Unbound {
                what: format!("phase {at}: host op names {name}, which the fire has no buffer for"),
            })
    };
    let put = |state: &mut HashMap<String, Vec<u32>>, name: &str, v: Vec<f32>| {
        state.insert(name.to_string(), v.iter().map(|x| x.to_bits()).collect());
    };
    match &host.op {
        HostOp::Embed {
            ids,
            table,
            y,
            rows,
            width,
            limit,
        } => {
            let (rows, width, limit) = (*rows as usize, *width as usize, *limit as usize);
            let id_words = state.get(ids).cloned().unwrap_or_default();
            let table = take(state, table)?;
            let mut out = take(state, y)?;
            out.resize(rows * width, 0.0);
            for r in 0..rows {
                let id = id_words.get(r).map_or(0, |w| *w as i32);
                let id = if id < 0 || id as usize >= limit {
                    0
                } else {
                    id as usize
                };
                let src = table.get(id * width..(id + 1) * width).unwrap_or(&[]);
                let dst = &mut out[r * width..(r + 1) * width];
                dst[..src.len()].copy_from_slice(src);
            }
            put(state, y, out);
        }
        HostOp::SplitRows {
            x,
            left,
            right,
            rows,
            width,
            cut,
        } => {
            let (rows, width, cut) = (*rows as usize, *width as usize, *cut as usize);
            let src = take(state, x)?;
            let mut l = take(state, left)?;
            let mut r = take(state, right)?;
            let rw = width - cut;
            for i in 0..rows {
                let row = src.get(i * width..(i + 1) * width).unwrap_or(&[]);
                if row.len() < width {
                    continue;
                }
                if let Some(d) = l.get_mut(i * cut..(i + 1) * cut) {
                    d.copy_from_slice(&row[..cut]);
                }
                if let Some(d) = r.get_mut(i * rw..(i + 1) * rw) {
                    d.copy_from_slice(&row[cut..]);
                }
            }
            put(state, left, l);
            put(state, right, r);
        }
        HostOp::SplitQGate {
            packed,
            q,
            gate,
            rows,
            heads,
            head_dim,
        } => {
            let (rows, heads, hd) = (*rows as usize, *heads as usize, *head_dim as usize);
            let src = take(state, packed)?;
            let mut qv = take(state, q)?;
            let mut gv = take(state, gate)?;
            for r in 0..rows {
                for h in 0..heads {
                    let from = r * heads * 2 * hd + h * 2 * hd;
                    let to = r * heads * hd + h * hd;
                    if let (Some(s), Some(d)) = (src.get(from..from + hd), qv.get_mut(to..to + hd))
                    {
                        d.copy_from_slice(s);
                    }
                    if let (Some(s), Some(d)) =
                        (src.get(from + hd..from + 2 * hd), gv.get_mut(to..to + hd))
                    {
                        d.copy_from_slice(s);
                    }
                }
            }
            put(state, q, qv);
            put(state, gate, gv);
        }
        HostOp::TopkSoftmax {
            logits,
            routes,
            weights,
            rows,
            width,
            experts,
            top_k,
        } => {
            let (rows, width, experts, top_k) = (
                *rows as usize,
                *width as usize,
                *experts as usize,
                *top_k as usize,
            );
            let src = take(state, logits)?;
            let mut r = vec![0u32; rows * top_k];
            let mut w = vec![0f32; rows * top_k];
            for t in 0..rows {
                let row = src
                    .get(t * width..t * width + experts.min(width))
                    .unwrap_or(&[]);
                let mut picked: Vec<usize> = Vec::with_capacity(top_k);
                for _ in 0..top_k {
                    let mut best: Option<(usize, f32)> = None;
                    for (i, v) in row.iter().enumerate() {
                        if v.is_nan() || picked.contains(&i) {
                            continue;
                        }
                        if best.is_none_or(|(_, b)| *v > b) {
                            best = Some((i, *v));
                        }
                    }
                    match best {
                        Some((i, _)) => picked.push(i),
                        None => break,
                    }
                }
                let chosen: Vec<f32> = picked.iter().map(|i| row[*i]).collect();
                let m = chosen.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = chosen.iter().map(|v| (v - m).exp()).sum();
                for s in 0..top_k {
                    let at = t * top_k + s;
                    match picked.get(s).zip(chosen.get(s)) {
                        Some((i, v)) => {
                            r[at] = *i as u32;
                            w[at] = (v - m).exp() / z;
                        }
                        None => {
                            r[at] = (-1i32) as u32;
                            w[at] = 0.0;
                        }
                    }
                }
            }
            state.insert(routes.to_string(), r);
            put(state, weights, w);
        }
        HostOp::SplitQkv {
            packed,
            q,
            k,
            v,
            rows,
            q_width,
            kv_width,
        } => {
            let (rows, qw, kw) = (*rows as usize, *q_width as usize, *kv_width as usize);
            let width = qw + 2 * kw;
            let src = take(state, packed)?;
            let mut qv = take(state, q)?;
            let mut kv = take(state, k)?;
            let mut vv = take(state, v)?;
            for r in 0..rows {
                let Some(row) = src.get(r * width..(r + 1) * width) else {
                    break;
                };
                if let Some(d) = qv.get_mut(r * qw..(r + 1) * qw) {
                    d.copy_from_slice(&row[..qw]);
                }
                if let Some(d) = kv.get_mut(r * kw..(r + 1) * kw) {
                    d.copy_from_slice(&row[qw..qw + kw]);
                }
                if let Some(d) = vv.get_mut(r * kw..(r + 1) * kw) {
                    d.copy_from_slice(&row[qw + kw..]);
                }
            }
            put(state, q, qv);
            put(state, k, kv);
            put(state, v, vv);
        }
        HostOp::EmbedWeighted {
            ids,
            weights,
            table,
            y,
            rows,
            taps,
            width,
            limit,
        } => {
            let (rows, taps, width, limit) = (
                *rows as usize,
                *taps as usize,
                *width as usize,
                *limit as usize,
            );
            let id_words = state.get(ids).cloned().unwrap_or_default();
            let w = take(state, weights)?;
            let t = take(state, table)?;
            let mut out = vec![0f32; rows * width];
            for r in 0..rows {
                for tap in 0..taps {
                    let id = id_words.get(r * taps + tap).map_or(0, |v| *v as i32);
                    let id = if id < 0 || id as usize >= limit {
                        0
                    } else {
                        id as usize
                    };
                    let g = w.get(r * taps + tap).copied().unwrap_or(0.0);
                    let src = t.get(id * width..(id + 1) * width).unwrap_or(&[]);
                    for (o, v) in out[r * width..(r + 1) * width].iter_mut().zip(src) {
                        *o += g * v;
                    }
                }
            }
            put(state, y, out);
        }
        HostOp::ArgmaxMerge {
            parts,
            y,
            rows,
            cg,
            block,
            column,
            y_width,
        } => {
            let (rows, cg, block, column, y_width) = (
                *rows as usize,
                *cg as usize,
                *block as usize,
                *column as usize,
                *y_width as usize,
            );
            let src = take(state, parts)?;
            let mut out = state.get(y).cloned().unwrap_or_default();
            out.resize(rows * y_width, 0);
            for r in 0..rows {
                let pairs = src.get(r * 2 * cg..(r + 1) * 2 * cg).unwrap_or(&[]);
                let mut best: Option<(usize, f32)> = None;
                for c in 0..cg {
                    let (Some(&v), Some(&i)) = (pairs.get(2 * c), pairs.get(2 * c + 1)) else {
                        continue;
                    };
                    let local = i.to_bits() as i32;
                    if local < 0 || v.is_nan() {
                        continue;
                    }
                    if best.is_none_or(|(_, b)| v > b) {
                        best = Some((c * block + local as usize, v));
                    }
                }
                out[r * y_width + column] = best.map_or(0, |(i, _)| i as u32);
            }
            state.insert(y.to_string(), out);
        }
        HostOp::Argmax {
            x,
            y,
            rows,
            width,
            column,
            y_width,
        } => {
            let (rows, width, column, y_width) = (
                *rows as usize,
                *width as usize,
                *column as usize,
                *y_width as usize,
            );
            let src = take(state, x)?;
            let mut out = state.get(y).cloned().unwrap_or_default();
            out.resize(rows * y_width, 0);
            for r in 0..rows {
                let row = src.get(r * width..(r + 1) * width).unwrap_or(&[]);
                let mut best: Option<(usize, f32)> = None;
                for (i, v) in row.iter().enumerate() {
                    if v.is_nan() {
                        continue;
                    }
                    if best.is_none_or(|(_, b)| *v > b) {
                        best = Some((i, *v));
                    }
                }
                out[r * y_width + column] = best.map_or(0, |(i, _)| i as u32);
            }
            state.insert(y.to_string(), out);
        }
        HostOp::Matmul { act, w, y, m, k, n } => {
            let (m, k, n) = (*m as usize, *k as usize, *n as usize);
            let a = take(state, act)?;
            let wt = take(state, w)?;
            let mut out = take(state, y)?;
            for i in 0..m {
                let Some(row) = a.get(i * k..(i + 1) * k) else {
                    break;
                };
                for j in 0..n {
                    let Some(wr) = wt.get(j * k..(j + 1) * k) else {
                        break;
                    };
                    let dot: f32 = row.iter().zip(wr).map(|(x, y)| x * y).sum();
                    if let Some(slot) = out.get_mut(i * n + j) {
                        *slot = dot;
                    }
                }
            }
            put(state, y, out);
        }
        HostOp::MatmulGrouped {
            x,
            w,
            routes,
            y,
            rows,
            groups,
            k,
            n,
            experts,
        } => {
            let (rows, groups, k, n, experts) = (
                *rows as usize,
                *groups as usize,
                *k as usize,
                *n as usize,
                *experts as usize,
            );
            let a = take(state, x)?;
            let wt = take(state, w)?;
            let route_words = state.get(routes).cloned().unwrap_or_default();
            let mut out = take(state, y)?;
            out.resize(rows * groups * n, 0.0);
            for r in 0..rows {
                for g in 0..groups {
                    let e = route_words.get(r * groups + g).map_or(-1, |v| *v as i32);
                    if e < 0 || e as usize >= experts {
                        continue;
                    }
                    let Some(row) = a.get((r * groups + g) * k..(r * groups + g + 1) * k) else {
                        break;
                    };
                    let block = e as usize * n * k;
                    for j in 0..n {
                        let Some(wr) = wt.get(block + j * k..block + (j + 1) * k) else {
                            break;
                        };
                        out[(r * groups + g) * n + j] =
                            row.iter().zip(wr).map(|(x, y)| x * y).sum();
                    }
                }
            }
            put(state, y, out);
        }
        HostOp::TopK {
            x,
            values,
            indices,
            rows,
            width,
            k,
        } => {
            let (rows, width, k) = (*rows as usize, *width as usize, *k as usize);
            let src = take(state, x)?;
            let mut vs = vec![0f32; rows * k];
            let mut is = vec![0u32; rows * k];
            for r in 0..rows {
                let row = src.get(r * width..(r + 1) * width).unwrap_or(&[]);
                let mut picked: Vec<usize> = Vec::with_capacity(k);
                for s in 0..k {
                    let mut best: Option<(usize, f32)> = None;
                    for (i, v) in row.iter().enumerate() {
                        if v.is_nan() || picked.contains(&i) {
                            continue;
                        }
                        if best.is_none_or(|(_, b)| *v > b) {
                            best = Some((i, *v));
                        }
                    }
                    let Some((i, v)) = best else {
                        break;
                    };
                    picked.push(i);
                    vs[r * k + s] = v;
                    is[r * k + s] = i as u32;
                }
            }
            put(state, values, vs);
            state.insert(indices.to_string(), is);
        }
        HostOp::SelectorWalk {
            cand,
            indptr,
            unary,
            hp,
            tokens,
            pred,
            succ,
            picks,
            rows,
            k,
            rank,
            vocab,
            first,
        } => {
            let (rows, k, rank, vocab, first) = (
                *rows as usize,
                *k as usize,
                *rank as usize,
                *vocab as usize,
                *first as usize,
            );
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let cands = ints(cand);
            let ip = ints(indptr);
            let toks = ints(tokens);
            let un = take(state, unary)?;
            let hv = match hp {
                Some(h) => Some(take(state, h)?),
                None => None,
            };
            let pr = take(state, pred)?;
            let sc = take(state, succ)?;
            let mut out = ints(picks);
            out.resize(rows, 0);
            let at = |v: &[i32], i: usize| v.get(i).copied().unwrap_or(0);
            for l in 0..ip.len().saturating_sub(1) {
                let (lo, hi) = (at(&ip, l), at(&ip, l + 1));
                if hi <= lo || lo < 0 || lo as usize >= rows {
                    continue;
                }
                let (lo, hi) = (lo as usize, (hi as usize).min(rows));
                if first > 0 {
                    out[lo] = at(&cands, lo * k);
                }
                let mut prev = at(&toks, lo);
                for (row, slot) in out.iter_mut().enumerate().take(hi).skip(lo + first) {
                    let mut scores = Vec::with_capacity(k);
                    for c in 0..k {
                        let id = at(&cands, row * k + c);
                        let mut s = un.get(row * k + c).copied().unwrap_or(0.0);
                        if prev >= 0 && (prev as usize) < vocab && id >= 0 && (id as usize) < vocab
                        {
                            let bilinear: f32 = (0..rank)
                                .map(|d| {
                                    let h = hv.as_ref().map_or(1.0, |h| {
                                        h.get(row * rank + d).copied().unwrap_or(0.0)
                                    });
                                    pr.get(prev as usize * rank + d).copied().unwrap_or(0.0)
                                        * h
                                        * sc.get(id as usize * rank + d).copied().unwrap_or(0.0)
                                })
                                .sum();
                            s += bilinear;
                        }
                        scores.push(s);
                    }
                    // The first strict maximum; a NaN never wins past slot 0.
                    let mut best = 0usize;
                    if !scores[0].is_nan() {
                        let clean = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
                        for (c, s) in scores.iter().enumerate() {
                            if clean(*s) > clean(scores[best]) {
                                best = c;
                            }
                        }
                    }
                    let pick = at(&cands, row * k + best);
                    *slot = pick;
                    prev = pick;
                }
            }
            state.insert(picks.to_string(), out.iter().map(|v| *v as u32).collect());
        }
        HostOp::PleNgramIds {
            ids,
            indptr,
            slots,
            state: slab,
            out,
            rows,
            eos,
            mults,
            primes,
            offsets,
            heads_per_ngram,
        } => {
            let (rows, hpn) = (*rows as usize, *heads_per_ngram as usize);
            let eos = *eos as i32;
            let span = mults.len() - 1;
            let heads = primes.len();
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let id_words = ints(ids);
            let slot_words = ints(slots);
            let mut cells = ints(slab);
            let slots_rows = cells.len().checked_div(span).unwrap_or(0);
            let lanes: Vec<(usize, usize)> = match indptr {
                Some(ip) => {
                    let ip = ints(ip);
                    ip.windows(2)
                        .map(|w| (w[0].max(0) as usize, (w[1].max(w[0]).max(0)) as usize))
                        .collect()
                }
                None => (0..rows).map(|t| (t, t + 1)).collect(),
            };
            let mut result = vec![0u32; rows * heads];
            let id_at = |t: usize| id_words.get(t).copied().unwrap_or(0);
            for (b, e) in lanes {
                let (b, e) = (b.min(rows), e.min(rows));
                let n = e - b;
                let slot = slot_words.get(b).copied().unwrap_or(-1);
                let past: Vec<i32> = if slot >= 0 && (slot as usize) < slots_rows {
                    cells[slot as usize * span..(slot as usize + 1) * span].to_vec()
                } else {
                    vec![0; span]
                };
                for j in 0..n {
                    let t = b + j;
                    let mut window = vec![id_at(t)];
                    let mut crossed = false;
                    for p in 1..=span {
                        let src = j as i64 - p as i64;
                        let mut w = if src >= 0 {
                            id_at(t - p)
                        } else {
                            let cell = past[(span as i64 + src) as usize];
                            if cell == 0 { eos } else { cell - 1 }
                        };
                        if crossed {
                            w = eos;
                        }
                        crossed |= w == eos;
                        window.push(w);
                    }
                    let mut mixed = 0u64;
                    for (p, w) in window.iter().enumerate() {
                        mixed ^= (i64::from(*w) as u64).wrapping_mul(mults[p]);
                        if p == 0 {
                            continue;
                        }
                        let lo = (p - 1) * hpn;
                        for i in 0..hpn {
                            let r = (mixed % primes[lo + i]).wrapping_add(offsets[lo + i]);
                            result[t * heads + lo + i] = (r & 0xFFFF_FFFF) as u32;
                        }
                    }
                }
                if n > 0 && slot >= 0 && (slot as usize) < slots_rows {
                    let next: Vec<i32> = (0..span)
                        .map(|s| {
                            let src = n as i64 - span as i64 + s as i64;
                            if src >= 0 {
                                id_at(b + src as usize).wrapping_add(1)
                            } else {
                                past[(src + span as i64) as usize]
                            }
                        })
                        .collect();
                    cells[slot as usize * span..(slot as usize + 1) * span].copy_from_slice(&next);
                }
            }
            state.insert(out.to_string(), result);
            state.insert(slab.to_string(), cells.iter().map(|v| *v as u32).collect());
        }
        HostOp::PageWrite {
            src,
            table,
            write_page,
            write_offset,
            rows,
            width,
            page_size,
            table_rows,
        } => {
            let (rows, width, ps, table_rows) = (
                *rows as usize,
                *width as usize,
                *page_size as i64,
                *table_rows as usize,
            );
            let a = take(state, src)?;
            let mut t = take(state, table)?;
            let pages = state.get(write_page).cloned().unwrap_or_default();
            let offs = state.get(write_offset).cloned().unwrap_or_default();
            for r in 0..rows {
                let page = i64::from(pages.get(r).map_or(-1, |v| *v as i32));
                let off = i64::from(offs.get(r).map_or(-1, |v| *v as i32));
                if page < 0 || off < 0 || off >= ps {
                    continue;
                }
                let cell = (page * ps + off) as usize;
                let Some(row) = a.get(r * width..(r + 1) * width) else {
                    break;
                };
                if cell < table_rows
                    && let Some(dst) = t.get_mut(cell * width..(cell + 1) * width)
                {
                    dst.copy_from_slice(row);
                }
            }
            put(state, table, t);
        }
        HostOp::Boundary {
            positions,
            request_of_token,
            row_valid,
            boundary_pos,
            boundary_req,
            boundary_rope,
            rows,
            ratio,
        } => {
            let (rows, ratio) = (*rows as usize, *ratio as i32);
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (p, rq, v) = (ints(positions), ints(request_of_token), ints(row_valid));
            let at = |t: &[i32], i: usize| t.get(i).copied().unwrap_or(0);
            let mut bpos = vec![0u32; rows];
            let mut breq = vec![0u32; rows];
            let mut brope = vec![0u32; rows];
            for r in 0..rows {
                let pos = at(&p, r);
                let valid = at(&v, r) != 0;
                let closes = (pos.wrapping_add(1)) % ratio == 0;
                let is_b = valid && closes;
                bpos[r] = if is_b { pos } else { -1 } as u32;
                breq[r] = at(&rq, r) as u32;
                brope[r] = if is_b { (pos / ratio) * ratio } else { 0 } as u32;
            }
            state.insert(boundary_pos.to_string(), bpos);
            state.insert(boundary_req.to_string(), breq);
            state.insert(boundary_rope.to_string(), brope);
        }
        HostOp::BlockMean {
            boundary_pos,
            boundary_req,
            keys,
            indices,
            indptr,
            entries,
            rows,
            head_dim,
            ratio,
            page_size,
        } => {
            let (rows, hd, ratio, ps) = (
                *rows as usize,
                *head_dim as usize,
                *ratio as i64,
                i64::from(*page_size),
            );
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (bpos, breq, idx, ip) = (
                ints(boundary_pos),
                ints(boundary_req),
                ints(indices),
                ints(indptr),
            );
            let table = take(state, keys)?;
            // The index cache plane is `[rows][hd]`, as the op checked.
            let key_width = hd;
            let mut out = vec![0f32; rows * hd];
            for r in 0..rows {
                let bp = i64::from(bpos.get(r).copied().unwrap_or(-1));
                if bp < 0 {
                    continue;
                }
                let rq = breq.get(r).copied().unwrap_or(0);
                let mut sum = vec![0f32; hd];
                for i in 0..ratio {
                    let pos = bp + i - (ratio - 1);
                    if pos < 0 {
                        continue;
                    }
                    if let Some(cell) = cell_of(&idx, &ip, rq, pos, ps)
                        && let Some(row) = table.get(cell * key_width..cell * key_width + hd)
                    {
                        for (s, v) in sum.iter_mut().zip(row) {
                            *s += v;
                        }
                    }
                }
                for (o, s) in out[r * hd..(r + 1) * hd].iter_mut().zip(&sum) {
                    *o = s / ratio as f32;
                }
            }
            put(state, entries, out);
        }
        HostOp::PoolWrite {
            entries,
            boundary_pos,
            boundary_req,
            keys,
            indices,
            indptr,
            rows,
            width,
            page_size,
        } => {
            let (rows, width, ps) = (*rows as usize, *width as usize, i64::from(*page_size));
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (bpos, breq, idx, ip) = (
                ints(boundary_pos),
                ints(boundary_req),
                ints(indices),
                ints(indptr),
            );
            let e = take(state, entries)?;
            let mut table = take(state, keys)?;
            for r in 0..rows {
                let bp = i64::from(bpos.get(r).copied().unwrap_or(-1));
                if bp < 0 {
                    continue;
                }
                let rq = breq.get(r).copied().unwrap_or(0);
                let Some(row) = e.get(r * width..(r + 1) * width) else {
                    break;
                };
                if let Some(cell) = cell_of(&idx, &ip, rq, bp, ps)
                    && let Some(dst) = table.get_mut(cell * width..(cell + 1) * width)
                {
                    dst.copy_from_slice(row);
                }
            }
            put(state, keys, table);
        }
        HostOp::IndexTopk {
            q,
            weights,
            keys,
            indices,
            indptr,
            positions,
            request_of_token,
            selection,
            rows,
            heads,
            head_dim,
            top_k,
            ratio,
            page_size,
            max_pages,
        } => {
            let (rows, h, d, k, ratio, ps) = (
                *rows as usize,
                *heads as usize,
                *head_dim as usize,
                *top_k as usize,
                i64::from((*ratio).max(1)),
                i64::from(*page_size),
            );
            let nk = i64::from(*max_pages) * ps / ratio;
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (idx, ip, pos, rq) = (
                ints(indices),
                ints(indptr),
                ints(positions),
                ints(request_of_token),
            );
            let qv = take(state, q)?;
            let wv = match weights {
                Some(w) => Some(take(state, w)?),
                None => None,
            };
            let table = take(state, keys)?;
            let mut sel = vec![-1i32; rows * k];
            for r in 0..rows {
                let p = i64::from(pos.get(r).copied().unwrap_or(-1));
                let req = rq.get(r).copied().unwrap_or(0);
                let nkeys = if p + 1 > 0 {
                    ((p + 1) / ratio).clamp(0, nk)
                } else {
                    0
                } as usize;
                let mut scores = Vec::with_capacity(nkeys);
                for j in 0..nkeys {
                    let kpos = (j as i64 + 1) * ratio - 1;
                    let key: &[f32] = cell_of(&idx, &ip, req, kpos, ps)
                        .and_then(|cell| table.get(cell * d..(cell + 1) * d))
                        .unwrap_or(&[]);
                    let mut s = 0f32;
                    for hh in 0..h {
                        let qrow = &qv[(r * h + hh) * d..(r * h + hh + 1) * d];
                        let dot: f32 = qrow.iter().zip(key).map(|(a, b)| a * b).sum();
                        let w = wv
                            .as_ref()
                            .map_or(1.0, |w| w.get(r * h + hh).copied().unwrap_or(0.0));
                        s += dot.max(0.0) * w;
                    }
                    scores.push(s);
                }
                let taken: Vec<usize> = if nkeys <= k {
                    (0..nkeys).collect()
                } else {
                    let mut lo = scores.iter().copied().fold(f32::INFINITY, f32::min);
                    let mut hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for _ in 0..40 {
                        let mid = (lo + hi) * 0.5;
                        let cnt = scores.iter().filter(|s| **s >= mid).count();
                        if cnt > k {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    (0..nkeys).filter(|j| scores[*j] >= hi).take(k).collect()
                };
                for (slot, j) in taken.iter().enumerate() {
                    sel[r * k + slot] = *j as i32;
                }
            }
            state.insert(
                selection.to_string(),
                sel.iter().map(|v| *v as u32).collect(),
            );
        }
        HostOp::GroupRoutes {
            routes,
            rows,
            groups,
        } => {
            let (rows, groups) = (*rows as usize, *groups as usize);
            let out: Vec<u32> = (0..rows * groups).map(|i| (i % groups) as u32).collect();
            state.insert(routes.to_string(), out);
        }
        HostOp::EmbedConcat {
            ids,
            table,
            y,
            rows,
            heads,
            width,
            limit,
        } => {
            let (rows, heads, width, limit) = (
                *rows as usize,
                *heads as usize,
                *width as usize,
                *limit as usize,
            );
            let id_words = state.get(ids).cloned().unwrap_or_default();
            let table = take(state, table)?;
            let mut out = take(state, y)?;
            out.resize(rows * heads * width, 0.0);
            for slice in 0..rows * heads {
                let id = id_words.get(slice).map_or(-1, |w| *w as i32);
                let dst = &mut out[slice * width..(slice + 1) * width];
                if id < 0 || id as usize >= limit {
                    dst.fill(0.0);
                    continue;
                }
                let src = table
                    .get(id as usize * width..(id as usize + 1) * width)
                    .unwrap_or(&[]);
                dst[..src.len()].copy_from_slice(src);
            }
            put(state, y, out);
        }
    }
    for name in &host.rounds {
        if let Some(words) = state.get_mut(name) {
            for w in words.iter_mut() {
                *w = crate::bench::round_bf16(f32::from_bits(*w)).to_bits();
            }
        }
    }
    Ok(())
}

/// A lane plan resolved against the fire's data: which lanes and which
/// block of their slots' state each PE runs, which bank rows it holds, and
/// how the pieces come back.
struct LaneLayout {
    pes: usize,
    /// Per PE: `[l0, l1)` and the block `[a0, a1) × [b0, b1)`.
    ranges: Vec<(usize, usize, usize, usize, usize, usize)>,
    /// Per PE: the bank rows (slots) it holds, in local order.
    slots_of: Vec<Vec<usize>>,
    /// Per PE: the slot table with its lanes' slots rewritten to local rows.
    local_slots: Vec<Vec<u32>>,
    /// Per PE: the rows its lanes own.
    rows_of: Vec<Vec<usize>>,
    lanes_per_pe: usize,
    /// Bank slot rows one PE holds (its lanes' pages, or its share of a
    /// strided pool).
    bank_rows: usize,
    /// One slot's state as `(a, b, w)`, split as the plan says.
    block: (usize, usize, usize),
    /// Whether the plan splits slots into blocks at all.
    has_block: bool,
    header: String,
    slots: String,
    banks: Vec<(String, usize, BankSplit)>,
    row_outputs: Vec<(String, usize, usize, usize)>,
    /// Planes held by block: the block picks columns of every row, or
    /// whole rows.
    cols: Vec<ColPlane>,
    /// How row outputs held by several PEs merge.
    reduce: Vec<(String, Reduce)>,
    /// Per PE: the first page of each of its lanes it holds (page groups).
    page_of: Vec<usize>,
    /// Paged banks: local slot rows are pages, `max_pages` per lane (per PE).
    pages: Option<Paged>,
}

/// The paged form's per-PE tables.
struct Paged {
    max_pages: usize,
    /// Per PE: the CSR `indptr` with its lanes' entries pointing into the
    /// local indices, and the local indices (`lanes_per_pe × max_pages`).
    indptr: Option<(String, Vec<Vec<u32>>)>,
    indices: Option<(String, Vec<Vec<u32>>)>,
    /// Per-row page tables rewritten to local pages, per PE.
    rewritten: Vec<(String, Vec<Vec<u32>>)>,
}

impl LaneLayout {
    fn plan(
        lane: &kernels_cerebras::program::LanePlan,
        state: &HashMap<String, Vec<u32>>,
        at: usize,
    ) -> Result<LaneLayout> {
        use kernels_cerebras::program::{LaneKind, PageSource};
        let pes = lane.pes.max(1) as usize;
        // A plan without banks names no slot table.
        let no_slots = Vec::new();
        let slots_words = if lane.slots.is_empty() {
            &no_slots
        } else {
            state.get(&lane.slots).ok_or_else(|| Fault::Unbound {
                what: format!("phase {at}: slot table {} is not in the fire", lane.slots),
            })?
        };
        // A strided (resident) plan runs every row on every PE, so its lanes
        // need no contiguous rows; the row count is the table's.
        let strided = lane.pages.as_ref().is_some_and(|p| p.strided);
        let all_rows: usize;
        // Lanes' rows: `(first row, count)` per lane.
        let (lanes, rows_of_lane): (usize, Vec<(usize, usize)>) = match &lane.kind {
            LaneKind::PerRow { lanes } => {
                all_rows = *lanes as usize;
                (
                    *lanes as usize,
                    (0..*lanes as usize).map(|l| (l, 1)).collect(),
                )
            }
            LaneKind::Ragged { indptr, lanes } => {
                let p = state.get(indptr).ok_or_else(|| Fault::Unbound {
                    what: format!("phase {at}: indptr {indptr} is not in the fire"),
                })?;
                let ip: Vec<i32> = p.iter().map(|w| *w as i32).collect();
                let rows = (0..*lanes as usize)
                    .map(|l| {
                        let a = ip.get(l).copied().unwrap_or(0).max(0) as usize;
                        let b = ip.get(l + 1).copied().unwrap_or(a as i32).max(a as i32) as usize;
                        (a, b - a)
                    })
                    .collect();
                all_rows = ip.get(*lanes as usize).copied().unwrap_or(0).max(0) as usize;
                (*lanes as usize, rows)
            }
            LaneKind::ByTable { lane_of_row, lanes } => {
                let t = state.get(lane_of_row).ok_or_else(|| Fault::Unbound {
                    what: format!("phase {at}: lane table {lane_of_row} is not in the fire"),
                })?;
                all_rows = t.len();
                let mut rows = vec![(usize::MAX, 0usize); *lanes as usize];
                for (r, w) in t.iter().enumerate() {
                    let l = *w as i32;
                    if l < 0 || l as usize >= rows.len() {
                        continue;
                    }
                    let entry = &mut rows[l as usize];
                    if entry.1 == 0 {
                        *entry = (r, 1);
                    } else if entry.0 + entry.1 == r || strided {
                        entry.1 += 1;
                    } else {
                        return Err(Fault::Unbound {
                            what: format!(
                                "phase {at}: lane {l}'s rows are not contiguous, which a lane plan needs"
                            ),
                        });
                    }
                }
                (
                    *lanes as usize,
                    rows.into_iter()
                        .map(|(a, n)| if n == 0 { (0, 0) } else { (a, n) })
                        .collect(),
                )
            }
        };
        // Each lane's pages (paged banks) or its one slot.
        let pages_of_lane: Vec<Vec<usize>> = match &lane.pages {
            None => rows_of_lane
                .iter()
                .map(|(first, n)| {
                    if *n == 0 {
                        return Vec::new();
                    }
                    let slot = slots_words.get(*first).map(|w| *w as i32).unwrap_or(-1);
                    if slot < 0 {
                        Vec::new()
                    } else {
                        vec![slot as usize]
                    }
                })
                .collect(),
            Some(pp) => match &pp.source {
                PageSource::Csr { indptr, indices } => {
                    let ip = state.get(indptr).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page indptr {indptr} is not in the fire"),
                    })?;
                    let ix = state.get(indices).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page indices {indices} is not in the fire"),
                    })?;
                    (0..lanes)
                        .map(|l| {
                            let a = ip.get(l).map(|w| *w as i32).unwrap_or(0).max(0) as usize;
                            let b = ip
                                .get(l + 1)
                                .map(|w| *w as i32)
                                .unwrap_or(a as i32)
                                .max(a as i32) as usize;
                            ix.get(a..b.min(ix.len()))
                                .unwrap_or(&[])
                                .iter()
                                .map(|w| *w as i32)
                                .filter(|p| *p >= 0)
                                .map(|p| p as usize)
                                .collect()
                        })
                        .collect()
                }
                PageSource::Rows { table } => {
                    let t = state.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page table {table} is not in the fire"),
                    })?;
                    rows_of_lane
                        .iter()
                        .map(|(first, n)| {
                            (*first..*first + *n)
                                .filter_map(|r| {
                                    t.get(r)
                                        .map(|w| *w as i32)
                                        .filter(|p| *p >= 0)
                                        .map(|p| p as usize)
                                })
                                .collect()
                        })
                        .collect()
                }
            },
        };
        // A row window: the lanes' rows clipped to it and counted from its start.
        let rows_of_lane: Vec<(usize, usize)> = match lane.window {
            Some((start, len)) => rows_of_lane
                .iter()
                .map(|(first, n)| {
                    let (start, end) = (start as usize, start as usize + len as usize);
                    let lo = (*first).max(start).min(end);
                    let hi = (first + n).max(start).min(end);
                    (lo - start, hi - lo)
                })
                .collect(),
            None => rows_of_lane,
        };
        let lanes_per_pe = lane.lanes_per_pe.max(1) as usize;
        let (a, b, w, a_groups, b_groups) = match lane.block {
            Some(blk) => (
                blk.a as usize,
                blk.b as usize,
                blk.w as usize,
                blk.a_groups.max(1) as usize,
                blk.b_groups.max(1) as usize,
            ),
            None => {
                let stride = lane.banks.first().map_or(1, |(_, s, _)| *s as usize);
                (1, 1, stride, 1, 1)
            }
        };
        let groups = a_groups * b_groups;
        let page_groups = lane
            .pages
            .as_ref()
            .map_or(1, |p| p.page_groups.max(1) as usize);
        let pages_per_pe = lane.pages.as_ref().map_or(1, |p| p.pages_per_pe() as usize);
        // The resident placement: every PE runs every lane, page `g` on PE
        // `g % pes` as local page `g / pes`.
        let strided_pages = lane.pages.as_ref().map_or(0, |p| p.strided_pages as usize);
        if strided && (groups != 1 || page_groups != 1) {
            return Err(Fault::Unbound {
                what: format!("phase {at}: a strided pool splits over neither heads, rows nor page groups"),
            });
        }
        let mut page_of = Vec::with_capacity(pes);
        let mut ranges = Vec::with_capacity(pes);
        let mut slots_of = Vec::with_capacity(pes);
        let mut local_slots = Vec::with_capacity(pes);
        let mut rows_of = Vec::with_capacity(pes);
        let mut local_ips: Vec<Vec<u32>> = Vec::with_capacity(pes);
        let mut local_ixs: Vec<Vec<u32>> = Vec::with_capacity(pes);
        for p in 0..pes {
            if strided {
                // Every lane, its rows in order; the PE's fixed pages, the
                // lane lists naming them locally or -1.
                ranges.push((0, lanes, 0, a, 0, b));
                page_of.push(0);
                let held: Vec<usize> = (0..strided_pages).map(|s| s * pes + p).collect();
                let rows: Vec<usize> = (0..all_rows).collect();
                let mut local_ip: Vec<u32> = Vec::with_capacity(lanes + 1);
                let mut local_ix: Vec<u32> = Vec::new();
                for lane_pages in pages_of_lane.iter().take(lanes) {
                    local_ip.push(local_ix.len() as u32);
                    for &page in lane_pages {
                        local_ix.push(if page % pes == p { (page / pes) as u32 } else { u32::MAX });
                    }
                }
                local_ip.push(local_ix.len() as u32);
                slots_of.push(held);
                local_slots.push(slots_words.iter().map(|_| u32::MAX).collect());
                rows_of.push(rows);
                local_ips.push(local_ip);
                local_ixs.push(local_ix);
                continue;
            }
            let pg0 = (p % page_groups) * pages_per_pe;
            page_of.push(pg0);
            let q = p / page_groups;
            let lane_group = q / groups;
            // Page groups vary fastest, then the b axis's groups (the a
            // axis's when the plan puts its partial axis first), then the
            // other, then the lane groups.
            let (ag, bg) = if lane.a_first {
                (q % a_groups, (q % groups) / a_groups)
            } else {
                ((q % groups) / b_groups, q % b_groups)
            };
            let l0 = (lane_group * lanes_per_pe).min(lanes);
            let l1 = ((lane_group + 1) * lanes_per_pe).min(lanes);
            let (a0, a1) = (ag * (a / a_groups), (ag + 1) * (a / a_groups));
            let (b0, b1) = (bg * (b / b_groups), (bg + 1) * (b / b_groups));
            ranges.push((l0, l1, a0, a1, b0, b1));
            let mut held: Vec<usize> = Vec::new();
            let mut local: Vec<u32> = slots_words.iter().map(|_| u32::MAX).collect();
            let mut rows = Vec::new();
            let mut local_ip: Vec<u32> = Vec::new();
            let mut local_ix: Vec<u32> = Vec::new();
            for l in l0..l1 {
                let (first_row, n_rows) = rows_of_lane[l];
                rows.extend(first_row..first_row + n_rows);
                let lane_pages: Vec<usize> = pages_of_lane[l]
                    .iter()
                    .copied()
                    .skip(pg0)
                    .take(pages_per_pe)
                    .collect();
                let lane_pages = &lane_pages;
                let mut firsts = Vec::with_capacity(lane_pages.len());
                for &page in lane_pages {
                    let at = match held.iter().position(|s| *s == page) {
                        Some(i) => i,
                        None => {
                            held.push(page);
                            held.len() - 1
                        }
                    };
                    firsts.push(at as u32);
                }
                if lane.pages.is_none() {
                    if let Some(&at) = firsts.first() {
                        for r in first_row..first_row + n_rows {
                            if let Some(entry) = local.get_mut(r) {
                                *entry = at;
                            }
                        }
                    }
                } else {
                    // Local CSR: this lane's pages at local_ix[local_ip[l]..].
                    while local_ip.len() <= l {
                        local_ip.push(local_ix.len() as u32);
                    }
                    local_ix.extend(firsts);
                    local_ip.push(local_ix.len() as u32);
                }
            }
            if lane.pages.is_some() {
                while local_ip.len() <= lanes {
                    local_ip.push(local_ix.len() as u32);
                }
            }
            if rows.windows(2).any(|w| w[1] != w[0] + 1) {
                return Err(Fault::Unbound {
                    what: format!(
                        "phase {at}: PE {p}'s rows {rows:?} are not one contiguous run, which a lane plan's row range needs"
                    ),
                });
            }
            slots_of.push(held);
            local_slots.push(local);
            rows_of.push(rows);
            local_ips.push(local_ip);
            local_ixs.push(local_ix);
        }
        let pages = match &lane.pages {
            None => None,
            Some(pp) => {
                let max_pages = pp.pages_per_pe() as usize;
                let (indptr, indices) = match &pp.source {
                    PageSource::Csr { indptr, indices } => {
                        let ixs: Vec<Vec<u32>> = local_ixs
                            .iter()
                            .map(|ix| {
                                let mut v = ix.clone();
                                v.resize(lanes_per_pe * max_pages, 0);
                                v
                            })
                            .collect();
                        (
                            Some((indptr.clone(), local_ips.clone())),
                            Some((indices.clone(), ixs)),
                        )
                    }
                    PageSource::Rows { .. } => (None, None),
                };
                let mut rewritten = Vec::new();
                for table in &pp.rewritten {
                    let global = state.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page table {table} is not in the fire"),
                    })?;
                    let per_pe: Vec<Vec<u32>> = (0..pes)
                        .map(|p| {
                            let mut local: Vec<u32> = global.iter().map(|_| u32::MAX).collect();
                            for &r in &rows_of[p] {
                                let page = global.get(r).map(|w| *w as i32).unwrap_or(-1);
                                if page >= 0
                                    && let Some(at) =
                                        slots_of[p].iter().position(|s| *s == page as usize)
                                {
                                    local[r] = at as u32;
                                }
                            }
                            local
                        })
                        .collect();
                    rewritten.push((table.clone(), per_pe));
                }
                Some(Paged {
                    max_pages,
                    indptr,
                    indices,
                    rewritten,
                })
            }
        };
        Ok(LaneLayout {
            pes,
            ranges,
            slots_of,
            local_slots,
            rows_of,
            lanes_per_pe,
            bank_rows: if strided {
                strided_pages
            } else {
                lanes_per_pe * lane.pages.as_ref().map_or(1, |p| p.pages_per_pe() as usize)
            },
            block: (a, b, w),
            has_block: lane.block.is_some(),
            header: lane.header.clone(),
            slots: lane.slots.clone(),
            banks: lane
                .banks
                .iter()
                .map(|(n, s, split)| (n.clone(), *s as usize, *split))
                .collect(),
            row_outputs: lane
                .row_outputs
                .iter()
                .map(|(n, w, sa, sb)| (n.clone(), *w as usize, *sa as usize, *sb as usize))
                .collect(),
            reduce: lane.reduce.clone(),
            page_of,
            cols: lane.cols.clone(),
            pages,
        })
    }

    /// Whether `name` merges as a sum of partials (rounded here after).
    fn sums(&self, name: &str) -> bool {
        self.reduce
            .iter()
            .any(|(n, r)| n == name && matches!(r, Reduce::Sum))
    }

    /// A bank's `(stride, split)`; without a block plan every bank is whole.
    fn bank_stride(&self, name: &str) -> Option<(usize, BankSplit)> {
        self.banks
            .iter()
            .find(|(n, ..)| n == name)
            .map(|(_, s, split)| {
                (
                    *s,
                    if self.has_block {
                        *split
                    } else {
                        BankSplit::Whole
                    },
                )
            })
    }

    /// Words one PE holds of one slot of a bank of `stride` under `split`.
    fn bank_slot_words(&self, p: usize, stride: usize, split: BankSplit) -> usize {
        let (_, _, _, a0, a1, b0, b1) = self.range(p);
        match split {
            BankSplit::Whole => stride,
            BankSplit::Block => (a1 - a0) * (b1 - b0) * self.block.2,
            BankSplit::ByB => (b1 - b0) * (stride / self.block.1.max(1)),
        }
    }

    fn owns(&self, name: &str) -> bool {
        name == self.header
            || (name == self.slots && self.pages.is_none())
            || self.bank_stride(name).is_some()
            || self.row_outputs.iter().any(|(n, ..)| n == name)
            || self.cols.iter().any(|c| c.name == name)
            || self.page_table(name).is_some()
    }

    /// The indices below `bound` (columns, or rows) a plane's segments pick
    /// for PE `p`'s block.
    fn block_picks(&self, p: usize, bound: usize, segments: &[Segment]) -> Vec<usize> {
        if self.block.0 == 1 && self.block.1 == 1 {
            return (0..bound).collect();
        }
        let (_, _, _, a0, a1, b0, b1) = self.range(p);
        picks(
            segments,
            (a0 as u32, a1 as u32),
            (b0 as u32, b1 as u32),
            bound as u32,
        )
        .into_iter()
        .map(|c| c as usize)
        .collect()
    }

    /// The `(rows, cols)` of a block plane PE `p` holds.
    fn plane_cells(&self, p: usize, plane: &ColPlane) -> (Vec<usize>, Vec<usize>) {
        let (rows, width) = (plane.rows as usize, plane.width as usize);
        if plane.row_block {
            let (_, _, _, a0, a1, _, _) = self.range(p);
            return (
                (a0.min(rows)..a1.min(rows)).collect(),
                self.block_picks(p, width, &plane.segments),
            );
        }
        if plane.by_rows {
            (
                self.block_picks(p, rows, &plane.segments),
                (0..width).collect(),
            )
        } else {
            (
                (0..rows).collect(),
                self.block_picks(p, width, &plane.segments),
            )
        }
    }

    /// A page plan table this plan lays out per PE.
    fn page_table(&self, name: &str) -> Option<&Vec<Vec<u32>>> {
        let pages = self.pages.as_ref()?;
        if let Some((n, t)) = &pages.indptr
            && n == name
        {
            return Some(t);
        }
        if let Some((n, t)) = &pages.indices
            && n == name
        {
            return Some(t);
        }
        pages
            .rewritten
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }

    /// Local slot rows one PE holds (pages when paged; the PE's share of
    /// the pool when strided).
    fn local_rows(&self) -> usize {
        self.bank_rows
    }

    fn range(&self, p: usize) -> (usize, usize, usize, usize, usize, usize, usize) {
        let (l0, l1, a0, a1, b0, b1) = self.ranges[p];
        (p, l0, l1, a0, a1, b0, b1)
    }

    /// The per-PE words of `name`, when the plan lays it out.
    fn lay_out(&self, name: &str, words: &[u32]) -> Option<Vec<u32>> {
        if name == self.header {
            let mut out = Vec::with_capacity(self.pes * 16);
            for (p, (l0, l1, a0, a1, b0, b1)) in self.ranges.iter().enumerate() {
                let rows = &self.rows_of[p];
                let (r0, r1) = match (rows.first(), rows.last()) {
                    (Some(a), Some(b)) => (*a as u32, *b as u32 + 1),
                    _ => (0, 0),
                };
                let mut row = vec![
                    *l0 as u32,
                    *l1 as u32,
                    self.slots_of[p].len() as u32,
                    *a0 as u32,
                    *a1 as u32,
                    *b0 as u32,
                    *b1 as u32,
                    r0,
                    r1,
                    self.page_of.get(p).copied().unwrap_or(0) as u32,
                    self.pages.as_ref().map_or(1, |g| g.max_pages) as u32,
                ];
                row.resize(16, 0);
                out.extend(row);
            }
            return Some(out);
        }
        if name == self.slots && self.pages.is_none() {
            return Some(self.local_slots.concat());
        }
        if let Some(t) = self.page_table(name) {
            return Some(t.concat());
        }
        if let Some((stride, split)) = self.bank_stride(name) {
            let (b, w) = (self.block.1, self.block.2);
            let per_slot = self.bank_slot_words(0, stride, split);
            let mut out = Vec::with_capacity(self.pes * self.lanes_per_pe * per_slot);
            for p in 0..self.pes {
                let (_, _, _, a0, a1, b0, b1) = self.range(p);
                for slot in &self.slots_of[p] {
                    match split {
                        BankSplit::Block => {
                            for i in a0..a1 {
                                for j in b0..b1 {
                                    let from = slot * stride + (i * b + j) * w;
                                    let mut piece: Vec<u32> =
                                        words.get(from..from + w).unwrap_or(&[]).to_vec();
                                    piece.resize(w, 0);
                                    out.extend(piece);
                                }
                            }
                        }
                        BankSplit::ByB => {
                            let row = stride / b.max(1);
                            let from = slot * stride + b0 * row;
                            let len = (b1 - b0) * row;
                            let mut piece: Vec<u32> =
                                words.get(from..from + len).unwrap_or(&[]).to_vec();
                            piece.resize(len, 0);
                            out.extend(piece);
                        }
                        BankSplit::Whole => {
                            let from = slot * stride;
                            let mut piece: Vec<u32> =
                                words.get(from..from + stride).unwrap_or(&[]).to_vec();
                            piece.resize(stride, 0);
                            out.extend(piece);
                        }
                    }
                }
                out.resize((p + 1) * self.local_rows() * per_slot, 0);
            }
            return Some(out);
        }
        if self.row_outputs.iter().any(|(n, ..)| n == name) {
            // Every PE holds the whole plane; only its block's cells come back.
            return Some(
                words
                    .iter()
                    .copied()
                    .cycle()
                    .take(words.len() * self.pes)
                    .collect(),
            );
        }
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            let width = plane.width as usize;
            // A plane the fabric sums goes whole to each group's first PE
            // and as zeros to the rest, so the sum is the first's cells plus
            // what the others compute.
            let group = self.fabric_group(name);
            let mut out = Vec::new();
            for p in 0..self.pes {
                let (rows, cols) = self.plane_cells(p, plane);
                let zero = group.is_some_and(|g| p % g != 0);
                for r in &rows {
                    for c in &cols {
                        let w = words.get(r * width + c).copied().unwrap_or(0);
                        out.push(if zero { 0 } else { w });
                    }
                }
            }
            return Some(out);
        }
        None
    }

    /// The PE group width of a plane the fabric sums.
    fn fabric_group(&self, name: &str) -> Option<usize> {
        self.reduce.iter().find_map(|(n, r)| match r {
            Reduce::FabricSum { group } if n == name => Some((*group).max(1) as usize),
            _ => None,
        })
    }

    /// The whole of `name` from what the PEs hold and its `current` words.
    /// `sent` is what the PEs were given (`lay_out`'s words): a bank pastes
    /// back only the words a PE changed, since PEs may hold the same page
    /// (rows of two PEs landing in one page) and each writes its own rows.
    fn gather_up(
        &self,
        name: &str,
        laid: &[u32],
        sent: &[u32],
        mut current: Vec<u32>,
        laid_of: &HashMap<String, Vec<u32>>,
    ) -> Vec<u32> {
        if name == self.header {
            return laid[..laid.len().min(self.pes * 16)].to_vec();
        }
        if name == self.slots || self.page_table(name).is_some() {
            return current;
        }
        if let Some((stride, split)) = self.bank_stride(name) {
            let (b, w) = (self.block.1, self.block.2);
            let per_slot = self.bank_slot_words(0, stride, split);
            let per_pe = self.local_rows() * per_slot;
            for p in 0..self.pes {
                let (_, _, _, a0, a1, b0, b1) = self.range(p);
                let mut from = p * per_pe;
                let mut paste = |to: usize, len: usize, from: &mut usize| {
                    if let (Some(src), true) =
                        (laid.get(*from..*from + len), to + len <= current.len())
                    {
                        paste_changed(
                            &mut current[to..to + len],
                            src,
                            sent.get(*from..*from + len),
                        );
                    }
                    *from += len;
                };
                for slot in &self.slots_of[p] {
                    match split {
                        BankSplit::Block => {
                            for i in a0..a1 {
                                for j in b0..b1 {
                                    paste(slot * stride + (i * b + j) * w, w, &mut from);
                                }
                            }
                        }
                        BankSplit::ByB => {
                            let row = stride / b.max(1);
                            paste(slot * stride + b0 * row, (b1 - b0) * row, &mut from);
                        }
                        BankSplit::Whole => paste(slot * stride, stride, &mut from),
                    }
                }
            }
            return current;
        }
        if let Some((_, reduce)) = self.reduce.iter().find(|(n, _)| n == name) {
            return self.merge_rows(name, reduce, laid, sent, current, laid_of);
        }
        if let Some((_, width, sa, sb)) = self.row_outputs.iter().find(|(n, ..)| n == name) {
            let per_pe = current.len();
            for p in 0..self.pes {
                let (_, _, _, a0, a1, b0, b1) = self.range(p);
                for r in &self.rows_of[p] {
                    let from = p * per_pe + r * width;
                    let to = r * width;
                    if to + width > current.len() {
                        continue;
                    }
                    if self.block.0 == 1 && self.block.1 == 1 {
                        if let Some(src) = laid.get(from..from + width) {
                            current[to..to + width].copy_from_slice(src);
                        }
                        continue;
                    }
                    for i in a0..a1 {
                        for j in b0..b1 {
                            let col = i * sa + j * sb;
                            if col < *width
                                && let Some(v) = laid.get(from + col)
                            {
                                current[to + col] = *v;
                            }
                        }
                    }
                }
            }
            return current;
        }
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            let width = plane.width as usize;
            let mut from = 0;
            for p in 0..self.pes {
                let (rows, cols) = self.plane_cells(p, plane);
                for r in &rows {
                    for c in &cols {
                        if let Some(v) = laid.get(from)
                            && sent.get(from) != Some(v)
                            && let Some(slot) = current.get_mut(r * width + c)
                        {
                            *slot = *v;
                        }
                        from += 1;
                    }
                }
            }
            return current;
        }
        laid.to_vec()
    }
}

impl LaneLayout {
    /// The width of a row output or a block plane.
    fn plane_width(&self, name: &str) -> Option<usize> {
        self.row_outputs
            .iter()
            .find(|(n, ..)| n == name)
            .map(|(_, w, ..)| *w)
            .or_else(|| {
                self.cols
                    .iter()
                    .find(|c| c.name == name)
                    .map(|c| c.width as usize)
            })
    }

    /// The columns of `name` PE `p` computes: a plane's block columns, or a
    /// row output's block cells (all when the plan has no block).
    fn cells_of(&self, p: usize, name: &str, width: usize) -> Vec<usize> {
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            return self.plane_cells(p, plane).1;
        }
        let Some((_, _, sa, sb)) = self.row_outputs.iter().find(|(n, ..)| n == name) else {
            return Vec::new();
        };
        if !self.has_block {
            return (0..width).collect();
        }
        let (_, _, _, a0, a1, b0, b1) = self.range(p);
        let mut cols = std::collections::BTreeSet::new();
        for i in a0..a1 {
            for j in b0..b1 {
                let col = i * sa + j * sb;
                if col < width {
                    cols.insert(col);
                }
            }
        }
        cols.into_iter().collect()
    }

    /// Where PE `p` keeps cell `(r, col)` of `name` in its copy: a row output
    /// is whole per PE; a block plane holds only its block's columns.
    fn local_at(&self, p: usize, name: &str, r: usize, col: usize) -> Option<usize> {
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            let (_, cols) = self.plane_cells(p, plane);
            let i = cols.binary_search(&col).ok()?;
            return Some(r * cols.len() + i);
        }
        Some(r * self.plane_width(name)? + col)
    }

    /// Merges an output held by several PEs (page groups): each PE's copy
    /// of a row is partial over its pages (and its heads).
    fn merge_rows(
        &self,
        name: &str,
        reduce: &Reduce,
        laid: &[u32],
        sent: &[u32],
        mut current: Vec<u32>,
        laid_of: &HashMap<String, Vec<u32>>,
    ) -> Vec<u32> {
        let Some(width) = self.plane_width(name) else {
            return current;
        };
        if let Reduce::FabricSum { group } | Reduce::FabricRoot { group } = reduce {
            // The fabric merged each group into its first PE: its cells are
            // the result.
            let group = (*group).max(1) as usize;
            let per_pe = laid.len() / self.pes.max(1);
            let rows = current.len().checked_div(width).unwrap_or(0);
            for p in (0..self.pes).step_by(group) {
                for r in &self.rows_of[p] {
                    if *r >= rows {
                        continue;
                    }
                    for col in self.cells_of(p, name, width) {
                        let Some(i) = self.local_at(p, name, *r, col) else {
                            continue;
                        };
                        if let (Some(now), Some(slot)) =
                            (laid.get(p * per_pe + i), current.get_mut(r * width + col))
                        {
                            *slot = *now;
                        }
                    }
                }
            }
            return current;
        }
        if matches!(reduce, Reduce::Sum) {
            // Each PE's copy holds its cells; what it changed adds up.
            let per_pe = laid.len() / self.pes.max(1);
            let rows = current.len().checked_div(width).unwrap_or(0);
            for p in 0..self.pes {
                for r in &self.rows_of[p] {
                    if *r >= rows {
                        continue;
                    }
                    for col in self.cells_of(p, name, width) {
                        let Some(i) = self.local_at(p, name, *r, col) else {
                            continue;
                        };
                        let (Some(now), Some(was)) =
                            (laid.get(p * per_pe + i), sent.get(p * per_pe + i))
                        else {
                            continue;
                        };
                        let delta = f32::from_bits(*now) - f32::from_bits(*was);
                        if let Some(slot) = current.get_mut(r * width + col) {
                            *slot = (f32::from_bits(*slot) + delta).to_bits();
                        }
                    }
                }
            }
            return current;
        }
        let rows = current.len().checked_div(width).unwrap_or(0);
        let per_pe = laid.len() / self.pes.max(1);
        let at = |p: usize, r: usize, col: usize| -> Option<f32> {
            let i = self.local_at(p, name, r, col)?;
            laid.get(p * per_pe + i).map(|w| f32::from_bits(*w))
        };
        for r in 0..rows {
            let holders: Vec<usize> = (0..self.pes)
                .filter(|p| self.rows_of[*p].contains(&r))
                .collect();
            if holders.is_empty() {
                continue;
            }
            match reduce {
                Reduce::Sum | Reduce::FabricSum { .. } | Reduce::FabricRoot { .. } => {}
                Reduce::LogSumExp => {
                    for h in 0..width {
                        let vals: Vec<f32> = holders.iter().filter_map(|p| at(*p, r, h)).collect();
                        if !vals.is_empty() {
                            current[r * width + h] = log_sum_exp2(&vals).to_bits();
                        }
                    }
                }
                Reduce::Weighted { lse } => {
                    let Some(heads) = self.plane_width(lse) else {
                        continue;
                    };
                    let lse_laid = laid_of.get(lse).map_or(&[][..], |v| v.as_slice());
                    let lse_per_pe = lse_laid.len() / self.pes.max(1);
                    let lse_at = |p: usize, h: usize| -> Option<f32> {
                        let i = self.local_at(p, lse, r, h)?;
                        lse_laid.get(p * lse_per_pe + i).map(|w| f32::from_bits(*w))
                    };
                    let d = width / heads.max(1);
                    for h in 0..heads {
                        let pairs: Vec<(usize, f32)> = holders
                            .iter()
                            .filter_map(|p| lse_at(*p, h).map(|l| (*p, l)))
                            .collect();
                        if pairs.is_empty() {
                            continue;
                        }
                        let ls: Vec<f32> = pairs.iter().map(|(_, l)| *l).collect();
                        let total = log_sum_exp2(&ls);
                        for i in 0..d {
                            let col = h * d + i;
                            let v = if total == f32::NEG_INFINITY {
                                0.0
                            } else {
                                pairs
                                    .iter()
                                    .map(|(p, l)| {
                                        (l - total).exp2() * at(*p, r, col).unwrap_or(0.0)
                                    })
                                    .sum()
                            };
                            if let Some(slot) = current.get_mut(r * width + col) {
                                *slot = v.to_bits();
                            }
                        }
                    }
                }
            }
        }
        current
    }
}

/// `log2 Σ 2^v`, stable; `-inf` when every `v` is.
fn log_sum_exp2(vals: &[f32]) -> f32 {
    let m = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return m;
    }
    m + vals.iter().map(|v| (v - m).exp2()).sum::<f32>().log2()
}

/// Copies into `dst` the words of `laid` that differ from `sent` (all of
/// them when nothing was sent).
fn paste_changed(dst: &mut [u32], laid: &[u32], sent: Option<&[u32]>) {
    for (i, v) in laid.iter().enumerate() {
        if sent.and_then(|s| s.get(i)).is_none_or(|s| s != v) {
            dst[i] = *v;
        }
    }
}

/// The simulator's fatal error, if its log reports one: the first `FATAL`
/// line (the faulting PE, instruction and message).
fn simulator_fatal(log: &Path) -> Option<String> {
    let text = std::fs::read_to_string(log).ok()?;
    text.lines()
        .find(|l| l.contains("FATAL"))
        .map(|l| l.trim().to_string())
}

/// How long one phase program may run (`PIE_CEREBRAS_PHASE_TIMEOUT` seconds,
/// default 1800).
fn phase_timeout() -> std::time::Duration {
    let secs = std::env::var("PIE_CEREBRAS_PHASE_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1800);
    std::time::Duration::from_secs(secs.max(1))
}

/// A three-way grid shard's `(row parts, column parts, PE -> (row part, column part))`.
type GridParts = (usize, usize, Box<dyn Fn(usize) -> (usize, usize)>);

/// The per-PE layout of a buffer for a rectangle of `pes`: PE `x`'s words
/// back to back, as `memcpy_h2d` over the rectangle takes them.
fn lay_out(
    words: &[u32],
    rows: usize,
    width: usize,
    shard: kernels_cerebras::program::Shard,
    pes: usize,
) -> Vec<u32> {
    use kernels_cerebras::program::Shard;
    match shard {
        Shard::Whole | Shard::Local(_) => words
            .iter()
            .copied()
            .cycle()
            .take(words.len() * pes)
            .collect(),
        Shard::Rows(_) => words.to_vec(),
        Shard::Cols(n) => {
            let n = n as usize;
            let per = width / n;
            let mut out = Vec::with_capacity(words.len());
            for x in 0..n {
                for r in 0..rows {
                    out.extend_from_slice(&words[r * width + x * per..r * width + (x + 1) * per]);
                }
            }
            out
        }
        Shard::RowsBy { parts, period } => {
            let (parts, period) = (parts as usize, period.max(1) as usize);
            let per = rows / parts.max(1);
            let mut out = Vec::with_capacity(per * width * pes);
            for x in 0..pes {
                let part = (x / period) % parts.max(1);
                let from = part * per * width;
                out.extend_from_slice(words.get(from..from + per * width).unwrap_or(&[]));
                out.resize((x + 1) * per * width, 0);
            }
            out
        }
        Shard::ColsBy { parts, period } | Shard::SumCols { parts, period } => {
            let (parts, period) = (parts.max(1) as usize, period.max(1) as usize);
            let by_period = matches!(shard, Shard::SumCols { .. });
            let pc = width / parts;
            let mut out = Vec::with_capacity(rows * pc * pes);
            for x in 0..pes {
                let part = if by_period {
                    x / period
                } else {
                    (x / period) % parts
                };
                for r in 0..rows {
                    let from = r * width + part * pc;
                    out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                }
                out.resize((x + 1) * rows * pc, 0);
            }
            out
        }
        Shard::RowsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::ColsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::SumGrid {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::Roots {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            // (row parts, column parts) of this buffer and the PE's parts.
            let (rparts, cparts, part_of): GridParts = match shard {
                Shard::RowsDepth { .. } => (rg, kg, Box::new(move |x| (x / (cg * kg), x % kg))),
                Shard::ColsDepth { .. } => (cg, kg, Box::new(move |x| ((x / kg) % cg, x % kg))),
                _ => (rg, cg, Box::new(move |x| (x / (cg * kg), (x / kg) % cg))),
            };
            let (pr, pc) = (rows / rparts, width / cparts);
            let mut out = Vec::with_capacity(pr * pc * pes);
            for x in 0..pes {
                let (rp, cp) = part_of(x);
                for r in rp * pr..(rp + 1) * pr {
                    let from = r * width + cp * pc;
                    out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                }
                out.resize((x + 1) * pr * pc, 0);
            }
            out
        }
        Shard::Tile {
            rows: rg,
            cols: cg,
            segments,
        } => {
            let (rg, cg, seg) = (
                rg.max(1) as usize,
                cg.max(1) as usize,
                segments.max(1) as usize,
            );
            let (pr, sw) = (rows / rg, width / seg);
            let pc = sw / cg;
            let mut out = Vec::with_capacity(pr * pc * seg * pes);
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                for r in rp * pr..(rp + 1) * pr {
                    for sg in 0..seg {
                        let from = r * width + sg * sw + cp * pc;
                        out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                    }
                }
                out.resize((x + 1) * pr * pc * seg, 0);
            }
            out
        }
        Shard::Grid { rows: rg, cols: cg } => {
            let (rg, cg) = (rg.max(1) as usize, cg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut out = Vec::with_capacity(pr * pc * pes);
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                for r in rp * pr..(rp + 1) * pr {
                    let from = r * width + cp * pc;
                    out.extend_from_slice(words.get(from..from + pc).unwrap_or(&[]));
                }
                out.resize((x + 1) * pr * pc, 0);
            }
            out
        }
    }
}

/// The inverse of [`lay_out`]: a buffer's words from what the PEs hold.
fn gather_up(
    laid: &[u32],
    rows: usize,
    width: usize,
    shard: kernels_cerebras::program::Shard,
    pes: usize,
) -> Vec<u32> {
    use kernels_cerebras::program::Shard;
    let n = rows * width;
    match shard {
        Shard::Whole | Shard::Local(_) => laid[..n.min(laid.len())].to_vec(),
        Shard::Rows(_) => laid.to_vec(),
        Shard::Cols(parts) => {
            let parts = parts as usize;
            let per = width / parts;
            let mut out = vec![0u32; n];
            for x in 0..parts {
                for r in 0..rows {
                    let from = (x * rows + r) * per;
                    out[r * width + x * per..r * width + (x + 1) * per]
                        .copy_from_slice(&laid[from..from + per]);
                }
            }
            out
        }
        Shard::RowsBy { parts, period } => {
            let (parts, period) = (parts as usize, period.max(1) as usize);
            let per = rows / parts.max(1);
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let part = (x / period) % parts.max(1);
                let (from, to) = (x * per * width, part * per * width);
                if let (Some(src), Some(dst)) = (
                    laid.get(from..from + per * width),
                    out.get_mut(to..to + per * width),
                ) {
                    dst.copy_from_slice(src);
                }
            }
            out
        }
        Shard::ColsBy { parts, period } => {
            let (parts, period) = (parts.max(1) as usize, period.max(1) as usize);
            let pc = width / parts;
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let part = (x / period) % parts;
                for r in 0..rows {
                    let from = x * rows * pc + r * pc;
                    let to = r * width + part * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
        Shard::SumCols { parts, period } => {
            let (parts, period) = (parts.max(1) as usize, period.max(1) as usize);
            let pc = width / parts;
            let mut sums = vec![0f32; n];
            for x in 0..pes {
                let part = x / period;
                for r in 0..rows {
                    for c in 0..pc {
                        if let (Some(v), Some(slot)) = (
                            laid.get(x * rows * pc + r * pc + c),
                            sums.get_mut(r * width + part * pc + c),
                        ) {
                            *slot += f32::from_bits(*v);
                        }
                    }
                }
            }
            sums.iter().map(|v| v.to_bits()).collect()
        }
        Shard::RowsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        }
        | Shard::ColsDepth {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            let (rparts, cparts, part_of): GridParts = match shard {
                Shard::RowsDepth { .. } => (rg, kg, Box::new(move |x| (x / (cg * kg), x % kg))),
                _ => (cg, kg, Box::new(move |x| ((x / kg) % cg, x % kg))),
            };
            let (pr, pc) = (rows / rparts, width / cparts);
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let (rp, cp) = part_of(x);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    let from = x * pr * pc + i * pc;
                    let to = r * width + cp * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
        Shard::SumGrid {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut sums = vec![0f32; n];
            for x in 0..pes {
                let (rp, cp) = (x / (cg * kg), (x / kg) % cg);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    for c in 0..pc {
                        if let (Some(v), Some(slot)) = (
                            laid.get(x * pr * pc + i * pc + c),
                            sums.get_mut(r * width + cp * pc + c),
                        ) {
                            *slot += f32::from_bits(*v);
                        }
                    }
                }
            }
            sums.iter().map(|v| v.to_bits()).collect()
        }
        Shard::Roots {
            rows: rg,
            cols: cg,
            depth: kg,
        } => {
            // The fabric summed each block into its row's first PE.
            let (rg, cg, kg) = (rg.max(1) as usize, cg.max(1) as usize, kg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut out = vec![0u32; n];
            for x in (0..pes).step_by(kg) {
                let (rp, cp) = (x / (cg * kg), (x / kg) % cg);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    let from = x * pr * pc + i * pc;
                    let to = r * width + cp * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
        Shard::Tile {
            rows: rg,
            cols: cg,
            segments,
        } => {
            let (rg, cg, seg) = (
                rg.max(1) as usize,
                cg.max(1) as usize,
                segments.max(1) as usize,
            );
            let (pr, sw) = (rows / rg, width / seg);
            let pc = sw / cg;
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                let mut from = x * pr * pc * seg;
                for r in rp * pr..(rp + 1) * pr {
                    for sg in 0..seg {
                        let to = r * width + sg * sw + cp * pc;
                        if let (Some(src), Some(dst)) =
                            (laid.get(from..from + pc), out.get_mut(to..to + pc))
                        {
                            dst.copy_from_slice(src);
                        }
                        from += pc;
                    }
                }
            }
            out
        }
        Shard::Grid { rows: rg, cols: cg } => {
            let (rg, cg) = (rg.max(1) as usize, cg.max(1) as usize);
            let (pr, pc) = (rows / rg, width / cg);
            let mut out = vec![0u32; n];
            for x in 0..pes {
                let (rp, cp) = (x / cg, x % cg);
                for (i, r) in (rp * pr..(rp + 1) * pr).enumerate() {
                    let from = x * pr * pc + i * pc;
                    let to = r * width + cp * pc;
                    if let (Some(src), Some(dst)) =
                        (laid.get(from..from + pc), out.get_mut(to..to + pc))
                    {
                        dst.copy_from_slice(src);
                    }
                }
            }
            out
        }
    }
}

fn target_name(target: Target) -> &'static str {
    match target {
        Target::Wse2 => "wse2",
        Target::Wse3 => "wse3",
    }
}

fn io_fault(dir: &Path, e: std::io::Error) -> Fault {
    Fault::Unbound {
        what: format!("{}: {e}", dir.display()),
    }
}

/// The `fabric-run` binary: `$PIE_CEREBRAS_FABRIC_RUN`, or the one next to
/// the current executable.
pub fn fabric_run() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("PIE_CEREBRAS_FABRIC_RUN") {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe().map_err(|e| Fault::Unbound {
        what: format!("the current executable: {e}"),
    })?;
    let mut dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    if dir
        .file_name()
        .is_some_and(|n| n == "deps" || n == "examples")
    {
        dir.pop();
    }
    let p = dir.join("fabric-run");
    if p.is_file() {
        Ok(p)
    } else {
        Err(Fault::NoDevice {
            detail: format!(
                "fabric-run not found at {} (build it with `cargo build -p engine-cerebras`)",
                p.display()
            ),
        })
    }
}

/// Unused here; kept so the rendered form has one home.
#[allow(dead_code)]
fn _rendered(_: &Rendered) {}
