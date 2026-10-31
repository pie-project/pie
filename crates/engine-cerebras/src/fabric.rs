//! The program a fire is traced into, and how it reaches the fabric.
//!
//! Kernel entries emit device work (`kernels_cerebras::phase`): phases of
//! kernel calls over buffers. This module owns where that data lives and
//! when it runs: every tensor a phase touches becomes a device array with
//! an exported symbol or an arena slot; adjacent phases fuse into one
//! program when they fit a PE together; a program renders as functions
//! (its single RPC entry, `run`, calls the phases in order and releases the
//! host command stream) or table-driven (`table_mode`, one interpreter over
//! an op table and an arena), a whole fire as one program over a rectangle
//! (`whole_fire`), its kv pools kept on the device between fires
//! (`keep_pools`); and the manifest says which symbol holds which tensor.

use std::collections::BTreeMap;
use std::fmt::Write;

use dtype::Dtype;
use kernels_cerebras::Tensor;
use kernels_cerebras::csl::{Arg, Call, Func};
use kernels_cerebras::error::{Error, refuse};
use kernels_cerebras::library::Kernel;
use kernels_cerebras::phase::{
    self, Buf, FabricOp, FabricReduce, Guarded, HostOp, Lanes, LanesAt, PageSource, Reduce, Shard,
    Sink,
};
use kernels_cerebras::plan::{LaneSplit, Plan};

use lanes::LanePlan;

mod arena;
pub mod lanes;
mod plan;
mod render;
mod table;

use render::reduce_mentions;

pub use arena::{
    ARENA_CHUNK, MOVE_ROW_WORDS, MOVE_ROWS, MOVE_SCRATCH, OP_BROADCAST, OP_BROADCAST_FROM,
    OP_GATHER, OP_HANDOFF, OP_NOP, OP_ON, OP_REDUCE, OP_ROOT, OPS_SLACK_WORDS, RowShape, Table,
    TableOp, Unified, WORK_SLACK_WORDS, keep_layout, op_words, parse_global_table,
};

pub use plan::Planner;

/// A phase as the engine runs it: an entry's phase (its function, the PEs
/// it runs on, how each buffer it names spreads over them; or a host
/// phase), its lanes planned.
#[derive(Debug, Clone)]
pub struct Phase {
    pub func: Func,
    pub pes: u32,
    pub shards: Vec<(String, Shard)>,
    pub lane: Option<LanePlan>,
    pub host: Option<HostPhase>,
    /// A sum the fabric takes after the function runs.
    pub reduce: Option<FabricReduce>,
}

/// A phase's function: its statements in order.
fn func_of(name: String, body: Vec<phase::Statement>) -> Func {
    let mut func = Func::new(name);
    for s in body {
        match s {
            phase::Statement::Call(c) => {
                func.body.call(c);
            }
            phase::Statement::Raw(b) => func.body.append(b),
        }
    }
    func
}

/// A phase the host runs instead of the fabric: pure data movement, or a
/// vocabulary-sized op no rectangle holds. Every buffer of a fire lives on
/// the host between phases, so it costs no transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPhase {
    pub op: HostOp,
    /// Buffers rounded to bf16 after the op (bf16 writes).
    pub rounds: Vec<String>,
}

impl From<phase::Phase> for Phase {
    /// An entry's phase as the engine runs it: a host op over the buffers'
    /// copies (its calls dropped, its bf16 writes rounded after it), or
    /// its function.
    fn from(p: phase::Phase) -> Self {
        if let Some(op) = p.host {
            return Phase {
                func: Func::new(p.name),
                pes: 1,
                shards: Vec::new(),
                lane: None,
                host: Some(HostPhase {
                    op,
                    rounds: p.rounds.into_iter().map(|b| b.name).collect(),
                }),
                reduce: None,
            };
        }
        Phase {
            func: func_of(p.name, p.body),
            pes: p.pes,
            shards: p.shards,
            lane: p.lane.as_ref().map(lanes::plan_of),
            host: None,
            reduce: p.reduce,
        }
    }
}

/// Where each bf16 write of a device phase rounds: a partial sum stays f32
/// on the PE (the host rounds the sum); a buffer the fabric merges or fills
/// rounds once the steps land; every other one after the phase's calls.
fn round_writes(p: &mut phase::Phase) {
    // How many elements of `buf` each PE holds in the phase.
    let local_len =
        |shards: &[(String, Shard)], buf: &Buf| match shards.iter().find(|(n, _)| *n == buf.name) {
            Some((_, Shard::Local(n))) => *n,
            Some((_, s)) => buf.len() / u64::from(s.pes()),
            None => buf.len(),
        };
    let reduced = |name: &str, fabric: bool| {
        p.lane.as_ref().is_some_and(|l| {
            l.lanes.reduce.iter().any(|(n, r)| {
                n == name
                    && if fabric {
                        matches!(r, Reduce::FabricSum { .. } | Reduce::FabricRoot { .. })
                    } else {
                        matches!(r, Reduce::Sum)
                    }
            })
        })
    };
    let mut body = Vec::new();
    let mut late = Vec::new();
    for b in &p.rounds {
        let summed = p.shards.iter().any(|(n, s)| {
            *n == b.name && matches!(s, Shard::SumCols { .. } | Shard::SumGrid { .. })
        }) || reduced(&b.name, false);
        if summed {
            continue;
        }
        let n = local_len(&p.shards, b);
        let merged = p.late_rounds.contains(&b.name)
            || p.shards
                .iter()
                .any(|(n, s)| *n == b.name && matches!(s, Shard::Roots { .. }))
            || reduced(&b.name, true);
        if merged {
            late.push((b.name.clone(), n));
        } else {
            body.push(phase::Statement::Call(Call {
                kernel: Kernel::RoundBf16,
                args: vec![Arg::Ptr(b.clone()), Arg::Int(n as i64)],
            }));
        }
    }
    p.body.extend(body);
    if let Some(r) = p.reduce.as_mut() {
        r.round.extend(late);
    }
}

/// The terms the engine plans by: what a PE holds, how far a phase
/// spreads, the placement in force, whether partials merge on the fabric
/// and how bf16 operands are stored. [`Planner`] chooses every split by
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fabric {
    /// The most data words one phase places on a PE altogether (its share
    /// of every buffer plus scratch). A PE links 10240 words of data
    /// beside the memcpy module alone; the rest of a PE's memory is code,
    /// stack and the task tables, and the kernels' code takes its share
    /// below that.
    pub pe_words: u64,
    /// The most elements of one buffer a PE holds before it is split (a
    /// small budget makes a small model take every split form).
    pub shard_words: u64,
    /// The most PEs one phase spreads over (a rectangle up to 16 rows
    /// tall); a weight past `max_pes × shard_words` (the vocabulary
    /// projection, by default) runs on the host. The default suits the
    /// simulator; a wafer takes hundreds of thousands.
    pub max_pes: u32,
    /// The resident placement: every phase of a fire on one row of this
    /// many PEs, activations whole on every PE, the weights by rows across
    /// the row, a matmul's column blocks gathered back to every PE, lanes
    /// padded to the row; `None` is the per-phase placement.
    pub resident_pes: Option<u32>,
    /// Whether partials merge on the fabric rather than on the host
    /// (`Plan::merge`).
    pub fabric_sum: bool,
    /// Whether bf16 weights live on the device packed two halves a word:
    /// half the memory, the kernels widening as they read.
    pub half_weights: bool,
    /// Whether bf16 kv pools live on the device packed two halves a word:
    /// half the memory a page, attention widening a key or value row as it
    /// reads it, `kv_append` packing as it lands.
    pub half_pool: bool,
    /// The most rows one attention phase takes, when capped.
    pub attn_rows: Option<u32>,
}

impl Default for Fabric {
    fn default() -> Self {
        Fabric {
            pe_words: 8704,
            shard_words: 4096,
            max_pes: 2048,
            resident_pes: None,
            fabric_sum: true,
            half_weights: false,
            half_pool: false,
            attn_rows: None,
        }
    }
}

/// The fabric's terms as this process sets them: each from its switch
/// ([`crate::switches`]: `[cerebras.switches]`, else the old variable),
/// else the default.
///
/// - `pe_words`: the data words one phase places on a PE.
/// - `shard_words`: the elements of one buffer a PE holds
///   before it is split (a small budget makes a small model take every
///   split form).
/// - `max_pes`: the PEs one phase spreads over (a wafer takes
///   hundreds of thousands).
/// - `resident_pes=N` (N > 1): the resident placement, every
///   phase of a fire on one row of N PEs.
/// - `fabric_sum=0`: partials add on the host, not the fabric.
/// - `half_weights=1`, `half_pool=1`: bf16
///   weights, kv pools, packed two halves a word on the device.
/// - `attn_rows`: the most rows one attention phase takes.
#[must_use]
pub fn terms() -> Fabric {
    let number = |name: &str| crate::switches::parse::<u64>(name).filter(|v| *v > 0);
    let on = crate::switches::on;
    let default = Fabric::default();
    Fabric {
        pe_words: number("pe_words").unwrap_or(default.pe_words),
        shard_words: number("shard_words").unwrap_or(default.shard_words),
        max_pes: crate::switches::parse::<u32>("max_pes")
            .filter(|v| *v > 0)
            .unwrap_or(default.max_pes),
        resident_pes: crate::switches::parse::<u32>("resident_pes").filter(|v| *v > 1),
        fabric_sum: crate::switches::get("fabric_sum").is_none_or(|v| v != "0" && v != "false"),
        half_weights: on("half_weights"),
        half_pool: on("half_pool"),
        attn_rows: crate::switches::parse::<u32>("attn_rows").filter(|v| *v > 0),
    }
}

/// A handle placed on its root buffer: rows `[offset, offset + rows)` of
/// `whole`.
#[derive(Debug, Clone)]
pub struct Placed {
    pub whole: Buf,
    pub offset: u32,
    pub rows: u32,
}

impl Placed {
    /// Whether the handle is its whole root.
    fn is_whole(&self) -> bool {
        self.offset == 0 && self.rows == self.whole.rows
    }
}

/// What places a fire's operand handles on the program's buffers: the
/// tracer's handle tables, or the bench's one buffer a handle.
pub trait Operands {
    /// The root buffer holding `t` now.
    fn read(&mut self, p: &mut Program, t: Tensor) -> Result<Placed, Error>;

    /// The root buffer `t` holds from here on.
    fn write(&mut self, p: &mut Program, t: Tensor) -> Result<Placed, Error>;

    /// `t` as a packed bf16 buffer (two halves a word), when the engine
    /// keeps it so (a half-precision weight under `half_weights`); `None`
    /// means read it as f32.
    fn read_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<Placed>, Error> {
        let _ = (p, t);
        Ok(None)
    }

    /// A kv pool `t` as a packed bf16 buffer under its plain name (read and
    /// written in place), when the engine keeps pools so (`half_pool`);
    /// `None` means f32.
    fn pool_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<Placed>, Error> {
        let _ = (p, t);
        Ok(None)
    }
}

/// What one phase emits into: a program, the operands it places, and the
/// planner the entry asks for its splits.
pub struct Emitting<'a> {
    program: &'a mut Program,
    operands: &'a mut dyn Operands,
    plan: &'a dyn Plan,
    /// Row windows the phase names, by handle: exported view buffers the
    /// host cuts from their roots before the phase and pastes back after.
    windows: Vec<(u32, Buf)>,
}

impl<'a> Emitting<'a> {
    pub fn new(
        program: &'a mut Program,
        operands: &'a mut dyn Operands,
        plan: &'a dyn Plan,
    ) -> Self {
        Emitting {
            program,
            operands,
            plan,
            windows: Vec::new(),
        }
    }

    /// The device array a placed handle is: its root, or the phase's row
    /// window of it (one a handle).
    fn window(&mut self, t: Tensor, at: Placed) -> Result<Buf, Error> {
        if at.is_whole() {
            return Ok(at.whole);
        }
        if let Some((_, buf)) = self.windows.iter().find(|w| w.0 == t.buf) {
            return Ok(buf.clone());
        }
        let name = self.program.unique(&format!("w{}", at.whole.name));
        let offset = u64::from(at.offset) * u64::from(at.whole.width);
        let buf = self
            .program
            .view(&name, &at.whole.name, offset, at.rows, at.whole.width)?;
        self.windows.push((t.buf, buf.clone()));
        Ok(buf)
    }
}

impl Sink for Emitting<'_> {
    fn read(&mut self, t: Tensor) -> Result<Buf, Error> {
        let at = self.operands.read(self.program, t)?;
        self.window(t, at)
    }

    fn write(&mut self, t: Tensor) -> Result<Buf, Error> {
        let at = self.operands.write(self.program, t)?;
        self.window(t, at)
    }

    fn read_half(&mut self, t: Tensor) -> Result<Option<Buf>, Error> {
        Ok(self
            .operands
            .read_half(self.program, t)?
            .filter(Placed::is_whole)
            .map(|at| at.whole))
    }

    fn pool_half(&mut self, t: Tensor) -> Result<Option<Buf>, Error> {
        let Some(at) = self
            .operands
            .pool_half(self.program, t)?
            .filter(Placed::is_whole)
        else {
            return Ok(None);
        };
        self.program.written(&at.whole.name);
        Ok(Some(at.whole))
    }

    fn unique(&mut self, hint: &str) -> String {
        self.program.unique(hint)
    }

    fn declare(
        &mut self,
        name: &str,
        dtype: Dtype,
        rows: u32,
        width: u32,
        input: bool,
    ) -> Result<Buf, Error> {
        self.program.declare(name, dtype, rows, width, input)
    }

    fn view(
        &mut self,
        name: &str,
        root: &str,
        offset: u64,
        rows: u32,
        width: u32,
    ) -> Result<Buf, Error> {
        self.program.view(name, root, offset, rows, width)
    }

    fn written(&mut self, name: &str) {
        self.program.written(name);
    }

    fn scratch(&mut self, hint: &str, len: u64, elem: &'static str) -> String {
        self.program.scratch_of(hint, len, elem)
    }

    /// Named by its contents, so equal tables share one global and one
    /// kept slot across every program of a model.
    fn table(&mut self, hint: &str, elem: &str, lits: &[String]) -> String {
        use std::hash::{Hash, Hasher};
        let body = lits.join(", ");
        let mut h = std::collections::hash_map::DefaultHasher::new();
        elem.hash(&mut h);
        body.hash(&mut h);
        let name = format!("{hint}_{:016x}", h.finish());
        self.program
            .global(format!("var {name} = [{}]{elem} {{ {body} }};", lits.len()));
        name
    }

    fn global(&mut self, line: String) {
        self.program.global(line);
    }

    fn lanes(&mut self, lanes: &Lanes, split: LaneSplit) -> Result<LanesAt, Error> {
        lanes::place(self.program, self.plan.row(), lanes, split)
    }

    fn phase(&mut self, mut phase: phase::Phase) {
        if phase.host.is_some() {
            self.program.push(Phase::from(phase));
            return;
        }
        round_writes(&mut phase);
        // The library units the phase's calls name join the program (the
        // prelude is in every program already).
        let calls = phase.body.iter().filter_map(|s| match s {
            phase::Statement::Call(c) => Some(c),
            phase::Statement::Raw(_) => None,
        });
        let steps = phase.reduce.iter().flat_map(|r| {
            r.steps
                .iter()
                .flat_map(|s| &s.calls)
                .chain(&r.finish)
                .map(|g| &g.call)
        });
        for call in calls.chain(steps) {
            for u in call.kernel.unit().with_deps() {
                if u.path != "prim/prelude" {
                    self.program.helper(u.name, u.text);
                }
            }
        }
        let mut phase = Phase::from(phase);
        // The resident placement runs every device phase on its row of PEs
        // (whole buffers replicated); a phase spread over PEs of its own
        // keeps them.
        if let Some(n) = self.plan.row()
            && phase.lane.is_none()
            && phase.pes == 1
        {
            phase.pes = n;
        }
        self.program.push(phase);
    }

    fn plan(&self) -> &dyn Plan {
        self.plan
    }
}

/// How the host touches an exported buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symbol {
    /// Read by the program before any phase writes it: the host uploads it.
    Input,
    /// Written by a phase and never read first: the host downloads it.
    Output,
    /// Read first, then written: uploaded before and downloaded after.
    InOut,
}

/// One exported buffer as the host sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub buf: u32,
    pub name: String,
    pub rows: u32,
    pub width: u32,
    /// The handle's storage dtype. Float handles are f32 arrays on the
    /// device (the host converts); integer handles keep their width.
    pub dtype: Dtype,
    /// The device element type.
    pub elem: &'static str,
    pub role: Symbol,
    pub shard: Shard,
    /// Two bf16 halves a u32 word (see [`Program::declare_packed`]).
    pub packed: bool,
    /// Kept on the device between runs of a resident program (see
    /// [`Program::keep`]).
    pub keep: bool,
}

impl Export {
    /// Array elements (words) one PE holds.
    pub fn local(&self) -> u64 {
        let per = if self.packed { 2 } else { 1 };
        match self.shard {
            Shard::Local(n) => n,
            shard => self.rows as u64 * self.width as u64 / per / u64::from(shard.pes()),
        }
    }
}

/// A buffer that is a row window of another: the host cuts `len` elements
/// at `offset` out of `root` before a phase that names it and pastes them
/// back after.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub name: String,
    pub root: String,
    pub offset: u64,
    pub len: u64,
}

/// What the host needs to run a rendered program.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Manifest {
    pub exports: Vec<Export>,
    /// The exported RPC that runs every phase.
    pub entry: String,
    /// The program rectangle, columns by rows.
    pub rect: (u32, u32),
    /// Exports that are windows of other buffers.
    pub views: Vec<View>,
    /// The lane plans the host lays the phase's buffers out by (a fused
    /// program may carry one per lane phase; a buffer belongs to the first
    /// that names it).
    pub lanes: Vec<LanePlan>,
    /// Set when the host runs this phase itself; `layout` and `pe` are then
    /// empty and nothing is compiled.
    pub host: Option<HostPhase>,
    /// Set when the program is table-driven: the device holds one arena
    /// and one op table instead of the exports, which the host packs into
    /// them (see [`Table`]).
    pub table: Option<Table>,
    /// Set instead of `table` when the program is a whole fire over
    /// several rows of PEs: row `y`'s table (its arena, its ops, its
    /// kernels), the rectangle `rect.1` rows high.
    pub tables: Vec<Table>,
}

/// Whether fused programs render table-driven (the `table` switch at 1):
/// one interpreter over the kernels the program uses, the buffers in an
/// arena at fixed offsets, the phases a table of kernel calls and
/// collectives the host uploads as data.
pub fn table_mode() -> bool {
    crate::switches::on("table")
}

/// Whether a fire renders as one table-driven program over every phase
/// (the `whole_fire` switch at 1, with `table_mode` and the resident
/// placement): the buffers keep one layout across phases, the arena is
/// allotted by liveness, and the host moves only inputs and outputs.
pub fn whole_fire() -> bool {
    table_mode() && crate::switches::on("whole_fire")
}

/// Whether kv pools stay on the device between fires
/// (the `keep_pools` switch at 1, with a whole-fire program of one unified
/// shape on a resident server): kept like the weights, their page moves
/// becoming table rows the host fills per fire.
pub fn keep_pools() -> bool {
    whole_fire() && crate::switches::on("keep_pools")
}

/// The rendered sources, ready for `cslc`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    pub layout: String,
    pub pe: String,
    pub manifest: Manifest,
}

/// The CSL element type a handle's dtype is stored as on the device.
pub fn elem_of(op: &'static str, dtype: Dtype) -> Result<&'static str, Error> {
    match dtype {
        Dtype::F32 | Dtype::Bf16 => Ok("f32"),
        Dtype::I32 => Ok("i32"),
        // Byte handles (masks, flags) are stored one per 32-bit word.
        Dtype::U32 | Dtype::U8 => Ok("u32"),
        other => Err(Error::DtypeUnsupported { op, dtype: other }),
    }
}

#[derive(Debug, Clone)]
struct Device {
    buf: Buf,
    dtype: Dtype,
    /// Read before any phase wrote it: the host uploads it.
    read: bool,
    written: bool,
    /// Order of first use, for stable rendering.
    seq: usize,
    /// A bf16 buffer kept two halves a u32 word (a weight the host packs
    /// on upload and never reads back).
    packed: bool,
    /// A buffer the device keeps between runs of a resident program (a
    /// weight): uploaded once, never read back.
    keep: bool,
}

/// A program under construction.
#[derive(Debug, Default)]
pub struct Program {
    /// The model-wide shape a whole-fire program follows (see [`Unified`]).
    pub unified: Option<Unified>,
    bufs: BTreeMap<String, Device>,
    scratch: Vec<(String, u64, &'static str)>,
    globals: Vec<String>,
    helpers: BTreeMap<String, String>,
    prologue: Vec<String>,
    phases: Vec<Phase>,
    views: Vec<View>,
    names: usize,
    scope: Option<String>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh identifier with `hint` in it.
    pub fn unique(&mut self, hint: &str) -> String {
        self.names += 1;
        format!("{hint}_{}", self.names)
    }

    /// Names what the next phase lowers; [`Self::take_scope`] hands it out.
    pub fn scope(&mut self, name: &str) {
        self.scope = Some(name.replace('.', "_"));
    }

    pub fn take_scope(&mut self) -> String {
        self.scope.take().unwrap_or_else(|| "op".to_string())
    }

    /// Declares (or finds) the exported buffer `name` holding `rows × width`
    /// elements of `dtype`. `input` marks it as read from the host before any
    /// phase writes it. A second declaration must agree on the shape.
    pub fn declare(
        &mut self,
        name: &str,
        dtype: Dtype,
        rows: u32,
        width: u32,
        input: bool,
    ) -> Result<Buf, Error> {
        self.declare_with(name, dtype, rows, width, input, false)
    }

    /// A bf16 buffer the device holds packed, two halves a u32 word: an
    /// input (a weight) the host packs on upload and never reads back; the
    /// kernels that take it unpack it (`k_matmul_packed`).
    pub fn declare_packed(
        &mut self,
        name: &str,
        dtype: Dtype,
        rows: u32,
        width: u32,
    ) -> Result<Buf, Error> {
        if dtype != Dtype::Bf16 || !width.is_multiple_of(2) {
            return Err(refuse(
                "declare",
                format!("buffer {name} ({rows}x{width} {dtype:?}) does not pack two bf16 a word"),
            ));
        }
        self.declare_with(name, dtype, rows, width, true, true)
    }

    fn declare_with(
        &mut self,
        name: &str,
        dtype: Dtype,
        rows: u32,
        width: u32,
        input: bool,
        packed: bool,
    ) -> Result<Buf, Error> {
        const OP: &str = "declare";
        if rows == 0 || width == 0 {
            return Err(refuse(
                OP,
                format!("buffer {name} is empty ({rows}x{width})"),
            ));
        }
        let elem = if packed { "u32" } else { elem_of(OP, dtype)? };
        let seq = self.bufs.len();
        let d = self.bufs.entry(name.to_string()).or_insert_with(|| Device {
            buf: Buf {
                name: name.to_string(),
                rows,
                width,
                elem,
            },
            dtype,
            read: false,
            written: false,
            packed,
            keep: false,
            seq,
        });
        if d.buf.rows != rows || d.buf.width != width || d.dtype != dtype {
            return Err(refuse(
                OP,
                format!(
                    "buffer {name} was {}x{} {:?} and is now declared as {rows}x{width} {dtype:?}",
                    d.buf.rows, d.buf.width, d.dtype
                ),
            ));
        }
        if input && !d.written {
            d.read = true;
        }
        Ok(d.buf.clone())
    }

    /// Marks `name` as written by a phase: the host downloads it after the run.
    pub fn written(&mut self, name: &str) {
        if let Some(d) = self.bufs.get_mut(name) {
            d.written = true;
        }
    }

    /// `name` stays on the device between runs of a resident program (a
    /// weight): the table-driven program puts it in its kept arena.
    pub fn keep(&mut self, name: &str) {
        if let Some(d) = self.bufs.get_mut(name) {
            d.keep = true;
        }
    }

    /// The device array holding handle `t` for reading (one buffer per handle).
    pub fn read(&mut self, op: &'static str, t: Tensor) -> Result<Buf, Error> {
        let _ = op;
        self.declare(&format!("b{}", t.buf), t.dtype, t.rows, t.width, true)
    }

    /// The device array handle `t` is written into (one buffer per handle).
    pub fn write(&mut self, op: &'static str, t: Tensor) -> Result<Buf, Error> {
        let _ = op;
        let b = self.declare(&format!("b{}", t.buf), t.dtype, t.rows, t.width, false)?;
        self.written(&b.name);
        Ok(b)
    }

    /// A statement run once when the program starts, before any phase.
    pub fn prologue(&mut self, line: impl Into<String>) {
        self.prologue.push(line.into());
    }

    /// A private f32 array of `len` elements for one phase's intermediates.
    pub fn scratch(&mut self, hint: &str, len: u64) -> String {
        self.scratch_of(hint, len, "f32")
    }

    /// A private array of `len` elements of `elem` (`f32`, `i32`, `u32`).
    pub fn scratch_of(&mut self, hint: &str, len: u64, elem: &'static str) -> String {
        let name = self.unique(&format!("s_{hint}"));
        self.scratch.push((name.clone(), len, elem));
        name
    }

    /// A global declaration line (a DSD, a constant table); `line` includes its `;`.
    pub fn global(&mut self, line: impl Into<String>) {
        let line = line.into();
        if !self.globals.contains(&line) {
            self.globals.push(line);
        }
    }

    /// A helper function, once per `name`; `source` is its full CSL text.
    pub fn helper(&mut self, name: &str, source: impl Into<String>) {
        self.helpers
            .entry(name.to_string())
            .or_insert_with(|| source.into());
    }

    /// Appends a phase; phases run in the order they were added.
    pub fn push(&mut self, phase: Phase) {
        self.phases.push(phase);
    }

    pub fn phases(&self) -> &[Phase] {
        &self.phases
    }

    /// Declares `name` as a window of `len` elements at `offset` of `root`
    /// (an exported buffer), `rows × width` of `root`'s dtype.
    pub fn view(
        &mut self,
        name: &str,
        root: &str,
        offset: u64,
        rows: u32,
        width: u32,
    ) -> Result<Buf, Error> {
        let dtype = self
            .bufs
            .get(root)
            .map(|d| d.dtype)
            .ok_or_else(|| refuse("view", format!("{root} is no exported buffer")))?;
        let buf = self.declare(name, dtype, rows, width, true)?;
        self.written(name);
        self.views.push(View {
            name: name.to_string(),
            root: root.to_string(),
            offset,
            len: u64::from(rows) * u64::from(width),
        });
        Ok(buf)
    }

    pub fn views(&self) -> &[View] {
        &self.views
    }

    /// The exports in first-use order.
    pub fn exports(&self) -> Vec<Export> {
        let mut v: Vec<&Device> = self.bufs.values().collect();
        v.sort_by_key(|d| d.seq);
        v.into_iter()
            .map(|d| Export {
                // `b{buf}`, or `b{buf}h` for the packed form of the same handle.
                buf: d
                    .buf
                    .name
                    .get(1..)
                    .and_then(|n| n.trim_end_matches('h').parse().ok())
                    .unwrap_or(0),
                name: d.buf.name.clone(),
                rows: d.buf.rows,
                width: d.buf.width,
                dtype: d.dtype,
                elem: d.buf.elem,
                role: match (d.read, d.written) {
                    (true, true) => Symbol::InOut,
                    (true, false) => Symbol::Input,
                    _ => Symbol::Output,
                },
                shard: Shard::Whole,
                packed: d.packed,
                keep: d.keep,
            })
            .collect()
    }
}

/// Whether `text` names the identifier `name` (not as part of a longer one).
fn mentions(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let mut from = 0;
    while let Some(at) = text[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let before = start.checked_sub(1).is_none_or(|i| !ident(bytes[i]));
        let after = end >= bytes.len() || !ident(bytes[end]);
        if before && after {
            return true;
        }
        from = end;
    }
    false
}

/// The name a `const NAME = ..` / `var NAME = ..` global declares.
fn global_name(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("const ")
        .or_else(|| line.strip_prefix("var "))?;
    let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
    Some(&rest[..end])
}

/// The name a `fn NAME(` line declares.
fn fn_name(line: &str) -> &str {
    let rest = line.strip_prefix("fn ").unwrap_or(line);
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    &rest[..end]
}

/// The most memcpy channels a program uses: one a row of its rectangle.
pub const MAX_CHANNELS: u32 = 16;

/// Words of code one non-comment line of a phase program links to, as
/// measured over compiled programs (6.3 to 7.5, the memcpy runtime's own
/// code folded into the first lines).
pub const CODE_WORDS_PER_LINE: u64 = 7;

/// Words of data and code a fused program may take of a PE's 12288: the
/// rest is the task tables, the stack and the memcpy runtime's state.
pub const FUSED_PE_WORDS: u64 = 11000;

/// Words the `collectives_2d` module adds to a program that reduces on the
/// fabric (its code, queues and state), as measured over programs that
/// linked and ones that did not.
pub const COLLECTIVES_WORDS: u64 = 2500;

fn trace_fusion(what: &str) {
    tracing::debug!("cerebras fusion: {what}");
}

/// Code words a line of a table-driven program costs, and the words it
/// costs regardless (the interpreter, the collectives module, the memcpy
/// runtime), as measured over linked programs (`.text` of 158 lines: 2822
/// words; 404 lines: 6048; 640 lines: 8658). A function program's
/// operands are literals and its lines cost `CODE_WORDS_PER_LINE`.
pub const TABLE_CODE_WORDS_PER_LINE: u64 = 13;
pub const TABLE_FIXED_WORDS: u64 = 750;

/// The rectangle `pes` PEs take: as many rows as divide `pes` up to
/// [`MAX_CHANNELS`] (each row streams its own memcpy channel), the rest
/// across.
#[must_use]
pub fn rect_of(pes: u32) -> (u32, u32) {
    let pes = pes.max(1);
    let h = (1..=MAX_CHANNELS.min(pes))
        .rev()
        .find(|d| pes.is_multiple_of(*d))
        .unwrap_or(1);
    (pes / h, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_cerebras::csl::Func;
    use kernels_cerebras::phase::FabricStep;

    /// A phase over `pes` PEs, its buffers spread as `shards`.
    fn on(
        func: Func,
        pes: u32,
        shards: Vec<(String, Shard)>,
        reduce: Option<FabricReduce>,
    ) -> Phase {
        Phase {
            func,
            pes,
            shards,
            lane: None,
            host: None,
            reduce,
        }
    }

    #[test]
    fn roles_follow_first_use() {
        let mut p = Program::new();
        let x = Tensor::new(1, 2, 3, Dtype::Bf16);
        let y = Tensor::new(2, 2, 3, Dtype::Bf16);
        let z = Tensor::new(3, 2, 3, Dtype::Bf16);
        p.read("t", x).unwrap();
        p.read("t", y).unwrap();
        p.write("t", y).unwrap();
        p.write("t", z).unwrap();
        p.read("t", z).unwrap();
        let roles: Vec<Symbol> = p.exports().iter().map(|e| e.role).collect();
        assert_eq!(roles, [Symbol::Input, Symbol::InOut, Symbol::Output]);
    }

    #[test]
    fn a_reshaped_handle_is_refused() {
        let mut p = Program::new();
        p.read("t", Tensor::new(1, 2, 3, Dtype::Bf16)).unwrap();
        assert!(p.read("t", Tensor::new(1, 3, 2, Dtype::Bf16)).is_err());
    }

    #[test]
    fn phases_render_with_only_the_buffers_they_name() {
        let mut p = Program::new();
        let x = Tensor::new(1, 2, 3, Dtype::F32);
        let y = Tensor::new(2, 2, 3, Dtype::F32);
        p.read("t", x).unwrap();
        p.write("t", y).unwrap();
        let mut f = Func::new("phase_0");
        f.body.line("b2[0] = b1[0];");
        p.push(on(f, 1, Vec::new(), None));
        let mut g = Func::new("phase_1");
        g.body.line("b2[1] = 1.0;");
        // Over two PEs: a phase on other PEs than the last does not fuse.
        p.push(on(g, 2, vec![("b2".into(), Shard::Rows(2))], None));
        let r = p.render_phases();
        assert_eq!(r.len(), 2);
        assert_eq!(
            r[0].manifest
                .exports
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            ["b1", "b2"]
        );
        assert_eq!(
            r[1].manifest
                .exports
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            ["b2"]
        );
        assert!(r[1].pe.contains("fn run() void {\n  phase_1();"));
        assert!(mentions("b12[0] + b1[1]", "b1") && !mentions("b12[0]", "b1"));
    }

    /// Two reducing phases on the same rectangle fuse into one program whose
    /// step machine runs each function, then its collective, in order.
    #[test]
    fn reducing_phases_fuse_into_one_step_machine() {
        let mut p = Program::new();
        let x = Tensor::new(1, 2, 4, Dtype::F32);
        let y = Tensor::new(2, 2, 4, Dtype::F32);
        p.read("t", x).unwrap();
        p.write("t", y).unwrap();
        let reduce = |partial: &str| FabricReduce {
            rect: (2, 1),
            steps: vec![FabricStep {
                calls: Vec::new(),
                op: FabricOp::Reduce {
                    send: partial.to_string(),
                    recv: "b2".to_string(),
                    count: 8,
                },
            }],
            finish: Vec::new(),
            round: Vec::new(),
        };
        let mut f = Func::new("phase_0");
        f.body.line("b2[0] = b1[0];");
        p.push(on(
            f,
            2,
            vec![(
                "b2".into(),
                Shard::Roots {
                    rows: 1,
                    cols: 1,
                    depth: 2,
                },
            )],
            Some(reduce("b1")),
        ));
        let mut g = Func::new("phase_1");
        g.body.line("b2[1] = 1.0;");
        p.push(on(
            g,
            2,
            vec![(
                "b2".into(),
                Shard::Roots {
                    rows: 1,
                    cols: 1,
                    depth: 2,
                },
            )],
            Some(reduce("b1")),
        ));
        let r = p.render_phases();
        assert_eq!(r.len(), 1, "one program");
        let pe = &r[0].pe;
        let at = |s: &str| pe.find(s).unwrap_or_else(|| panic!("{s} in {pe}"));
        assert!(at("0 => {") < at("phase_0();"));
        assert!(at("phase_0();") < at("mpi_x.reduce_fadds"));
        assert!(at("1 => {") < at("phase_1();"));
        assert!(pe.matches("mpi_x.reduce_fadds").count() == 2);
        assert!(at("2 => {") < at("sys_mod.unblock_cmd_stream();"));
        assert_eq!(r[0].manifest.rect, (2, 1));
    }

    /// Two tables unite into one shape: every kernel, every kept export at
    /// one slot, the larger work region, table and row.
    #[test]
    fn tables_unite_into_one_shape() {
        let table = |kernels: &[Kernel], keep: &[(&str, u64)], work: u64, rows: u32, width: u32| {
            let mut arena: Vec<(String, u64, u64)> = Vec::new();
            let mut at = 0;
            for (n, w) in keep {
                arena.push((n.to_string(), at, *w));
                at += w;
            }
            let keep_chunks = at.div_ceil(ARENA_CHUNK) as u32;
            let base = u64::from(keep_chunks) * ARENA_CHUNK;
            arena.push(("k_dummy".into(), base, 2));
            Table {
                kernels: kernels.to_vec(),
                ops_words: width,
                ops: Vec::new(),
                arena,
                arena_words: base + work,
                keep_chunks,
                rows,
                move_rows: 0,
                consts: Vec::new(),
                phases: 1,
            }
        };
        let a = table(
            &[Kernel::Matmul, Kernel::Rmsnorm],
            &[("b1", 4000), ("b2", 6000)],
            3000,
            20,
            10,
        );
        let b = table(
            &[Kernel::Rmsnorm, Kernel::Attend],
            &[("b2", 6000), ("b3", 100)],
            5000,
            35,
            12,
        );
        let u = Unified::union([&a, &b]);
        assert_eq!(u.rows[0].kernels.len(), 3);
        assert_eq!(
            u.keep.iter().map(|(n, ..)| n.as_str()).collect::<Vec<_>>(),
            ["b1", "b2", "b3"]
        );
        // b2 (6000) does not share b1's array (4000 + 6000 > 8192).
        assert_eq!(u.keep[1].1, ARENA_CHUNK);
        assert_eq!(u.keep_chunks, 2);
        assert_eq!(
            (u.rows[0].rows, u.rows[0].ops_words),
            (35, 12 + OPS_SLACK_WORDS)
        );
        assert_eq!(
            u.rows[0].arena_words,
            2 * ARENA_CHUNK + 5000 + WORK_SLACK_WORDS
        );
    }

    #[test]
    fn a_rendered_program_exports_its_buffers_and_entry() {
        let mut p = Program::new();
        let x = Tensor::new(7, 1, 4, Dtype::F32);
        p.read("t", x).unwrap();
        let mut f = Func::new("phase_0");
        f.body.line("b7[0] = 1.0;");
        p.push(on(f, 1, Vec::new(), None));
        let r = p.render();
        assert!(r.layout.contains("@export_name(\"b7\", [*]f32, true);"));
        assert_eq!(r.manifest.exports[0].elem, "f32");
        assert!(r.pe.contains("var b7: [4]f32;"));
        assert!(r.pe.contains("fn run() void {\n  phase_0();\n  sys_mod.unblock_cmd_stream();\n}"));
        assert_eq!(r.manifest.entry, "run");
        assert_eq!(r.manifest.exports[0].role, Symbol::Input);
    }

    #[test]
    fn a_global_array_literal_becomes_const_words() {
        let (name, words) =
            parse_global_table("var inv_freq_3 = [2]f32 { 1.0, 0.5 };").expect("parses");
        assert_eq!(name, "inv_freq_3");
        assert_eq!(words, vec![1.0f32.to_bits(), 0.5f32.to_bits()]);
        let (_, words) = parse_global_table("var p = [1]u64 { 4294967297 };").expect("parses");
        assert_eq!(words, vec![1, 1]);
        assert!(parse_global_table("const x = 3;").is_none());
        assert!(parse_global_table("var y = [2]i32 { 1 };").is_none());
    }

    #[test]
    fn classes_split_alike_under_one_partition() {
        let row = |phases: u32, kernels: &[Kernel]| Table {
            kernels: kernels.to_vec(),
            ops_words: 10,
            ops: Vec::new(),
            arena: vec![("k_dummy".into(), 0, 2)],
            arena_words: 100,
            keep_chunks: 0,
            rows: 1,
            move_rows: 0,
            consts: Vec::new(),
            phases,
        };
        let prefill = vec![
            row(10, &[Kernel::Matmul]),
            row(14, &[Kernel::Rmsnorm]),
            row(13, &[Kernel::Attend]),
        ];
        let decode = vec![
            row(20, &[Kernel::Matmul, Kernel::Delta]),
            row(17, &[Kernel::Attend]),
        ];
        let p = Unified::partition(&[prefill.clone(), decode.clone()]).expect("same phase count");
        assert_eq!(p.row_phases, Some(vec![10, 14, 13]));
        assert!(p.rows.is_empty());
        // Rows that do not match cannot unite; alike rows do, per row.
        assert!(Unified::union_rows(&[prefill.clone(), decode]).is_none());
        let decode = vec![
            row(10, &[Kernel::Matmul, Kernel::Delta]),
            row(14, &[Kernel::Rmsnorm]),
            row(13, &[Kernel::Rope]),
        ];
        let u = Unified::union_rows(&[prefill, decode]).expect("alike");
        assert_eq!(u.row_phases, Some(vec![10, 14, 13]));
        assert_eq!(u.rows[0].kernels, vec![Kernel::Matmul, Kernel::Delta]);
        assert_eq!(u.rows[2].kernels, vec![Kernel::Attend, Kernel::Rope]);
        // A fire of another length has no shared partition.
        assert!(Unified::partition(&[vec![row(5, &[])], vec![row(6, &[])]]).is_none());
    }
}
