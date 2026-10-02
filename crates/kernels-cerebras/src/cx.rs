//! The sink kernels emit into.
//!
//! A kernel entry asks for the device arrays behind its operand handles,
//! writes CSL statements into the phase under construction, and names which
//! handles it wrote. The engine's [`Env`] decides what a handle is; the
//! default [`Roots`] treats every handle as its own exported buffer.

use std::ops::{Deref, DerefMut};

use dtype::Dtype;

use crate::csl::{Block, Dsd1, Func};
use crate::error::{Error, refuse};
use crate::program::{
    Buf, FabricOp, FabricReduce, FabricStep, Guarded, HostOp, HostPhase, LanePlan, PagePlan,
    PageSource, Program, Reduce, Shard,
};
use crate::tensor::Tensor;

/// What a kernel entry emits into: the engine's sink for one node (or one
/// fire). An entry hands its body to [`Emit::emit`], which lends it a [`Cx`]
/// for the duration.
pub trait Emit {
    fn emit(&self, body: &mut dyn FnMut(&mut Cx<'_>) -> Result<(), Error>) -> Result<(), Error>;

    /// Names what is emitted next (the op a dispatch is lowering).
    fn scope(&self, name: &str) {
        let _ = name;
    }
}

pub type Ctx<'a> = dyn Emit + 'a;

/// A handle resolved to its root buffer: rows `[offset, offset + rows)` of
/// `whole`.
#[derive(Debug, Clone)]
pub struct View {
    pub whole: Buf,
    pub offset: u32,
    pub rows: u32,
}

/// What an engine supplies so kernels can name its handles.
pub trait Env {
    /// The root buffer holding `t` now.
    fn read(&mut self, p: &mut Program, t: Tensor) -> Result<View, Error>;

    /// The root buffer `t` holds from here on.
    fn write(&mut self, p: &mut Program, t: Tensor) -> Result<View, Error>;

    /// `t` as a packed bf16 buffer (two halves a word), when the engine
    /// keeps it so (a half-precision weight under `half_weights`, a kv pool
    /// under `half_pool`); `None` means read it as f32.
    fn read_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<View>, Error> {
        let _ = (p, t);
        Ok(None)
    }

    /// A kv pool `t` as a packed bf16 buffer under its plain name (read and
    /// written in place), when the engine keeps pools so; `None` means f32.
    fn pool_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<View>, Error> {
        let _ = (p, t);
        Ok(None)
    }
}

/// The default environment: every handle is one exported buffer.
#[derive(Debug, Default)]
pub struct Roots;

impl Env for Roots {
    fn read(&mut self, p: &mut Program, t: Tensor) -> Result<View, Error> {
        let whole = p.read("read", t)?;
        Ok(View {
            offset: 0,
            rows: whole.rows,
            whole,
        })
    }

    /// Every bf16 handle a kernel asks for packed is packed (the bench
    /// standing in for the engine's weights): a weight under its own
    /// packed name, a pool (asked through [`Cx::pool_half`]) under its
    /// plain name, read and written.
    fn read_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<View>, Error> {
        if !crate::linear::gemm::half_weights() || t.dtype != Dtype::Bf16 || !t.width.is_multiple_of(2) {
            return Ok(None);
        }
        let whole = p.declare_packed(&format!("b{}h", t.buf), t.dtype, t.rows, t.width)?;
        Ok(Some(View {
            offset: 0,
            rows: whole.rows,
            whole,
        }))
    }

    fn pool_half(&mut self, p: &mut Program, t: Tensor) -> Result<Option<View>, Error> {
        if !crate::linear::gemm::half_pool() || t.dtype != Dtype::Bf16 || !t.width.is_multiple_of(2) {
            return Ok(None);
        }
        let whole = p.declare_packed(&format!("b{}", t.buf), t.dtype, t.rows, t.width)?;
        Ok(Some(View {
            offset: 0,
            rows: whole.rows,
            whole,
        }))
    }

    fn write(&mut self, p: &mut Program, t: Tensor) -> Result<View, Error> {
        let whole = p.write("write", t)?;
        Ok(View {
            offset: 0,
            rows: whole.rows,
            whole,
        })
    }
}

/// A kernel's view: the phase under construction plus the engine's handle
/// resolution. Derefs to the phase body [`Block`], so a kernel emits with
/// `cx.line(..)`.
pub struct Cx<'a> {
    p: &'a mut Program,
    env: &'a mut dyn Env,
    phase: Func,
    rounds: Vec<Buf>,
    /// Row windows this phase names, by handle: exported view buffers the
    /// host cuts from their roots and pastes back.
    windows: Vec<(u32, Buf)>,
    /// How many PEs the phase runs on, and how its buffers spread over them.
    pes: u32,
    shards: Vec<(String, Shard)>,
    lane: Option<LanePlan>,
    host: Option<HostOp>,
    reduce: Option<FabricReduce>,
    /// Buffers whose bf16 rounding waits for the fabric steps (they are
    /// filled by a collective, not by the phase's function).
    late_rounds: Vec<String>,
}

impl<'a> Cx<'a> {
    /// Opens a phase named after `op`.
    pub fn new(p: &'a mut Program, env: &'a mut dyn Env, op: &str) -> Self {
        let name = p.unique(&format!("phase_{}", op.replace('.', "_")));
        Self {
            p,
            env,
            phase: Func::new(name),
            rounds: Vec::new(),
            windows: Vec::new(),
            pes: 1,
            shards: Vec::new(),
            lane: None,
            host: None,
            reduce: None,
            late_rounds: Vec::new(),
        }
    }

    /// Makes this phase a host phase running `op` (statements emitted into
    /// it are dropped).
    pub fn host(&mut self, op: HostOp) {
        self.host = Some(op);
    }

    /// Spreads this phase over lanes as `plan` says: the plan's header and
    /// banks get their per-PE shapes, and the phase runs on `plan.pes` PEs.
    pub fn lanes(&mut self, mut plan: LanePlan) -> Result<Buf, Error> {
        // The resident placement runs every phase on its row of PEs: a
        // lane plan takes the row (idle PEs past its own), and a lane whose
        // state would split over PEs (a merge) does not fit it.
        if let Some(pes) = crate::linear::gemm::resident_pes() {
            if plan.pes > pes {
                return Err(refuse(
                    "lanes",
                    format!("the lane plan wants {} PEs; the resident row has {pes}", plan.pes),
                ));
            }
            let whole_row = plan.reduce.iter().all(|(_, r)| {
                matches!(r, Reduce::FabricRoot { group } | Reduce::FabricSum { group } if *group == pes)
            });
            if !whole_row {
                return Err(refuse(
                    "lanes",
                    "the resident placement merges a lane's partials over the whole row or not at all",
                ));
            }
            plan.pes = pes;
        }
        self.pes = plan.pes.max(1);
        let header = self
            .p
            .declare(&plan.header, Dtype::I32, plan.pes, 16, true)?;
        self.shards
            .push((plan.header.clone(), Shard::Rows(plan.pes)));
        for (bank, stride, split) in &plan.banks {
            let buf = Buf {
                name: bank.clone(),
                rows: 0,
                width: 0,
                elem: "f32",
            };
            self.shards.retain(|(n, _)| *n != buf.name);
            self.shards.push((
                bank.clone(),
                Shard::Local(plan.page_slots() * plan.slot_words(*stride, *split)),
            ));
        }
        for plane in &plan.cols {
            self.shards.retain(|(n, _)| *n != plane.name);
            self.shards
                .push((plane.name.clone(), Shard::Local(plan.plane_words(plane))));
        }
        if let Some(PagePlan {
            source: PageSource::Csr { indices, .. },
            ..
        }) = &plan.pages
        {
            self.shards.retain(|(n, _)| *n != *indices);
            self.shards
                .push((indices.clone(), Shard::Local(plan.local_rows())));
        }
        self.lane = Some(plan);
        Ok(header)
    }

    /// Runs this phase over `pes` PEs; buffers spread as [`Cx::shard`] says.
    pub fn over(&mut self, pes: u32) {
        self.pes = pes.max(1);
    }

    /// Spreads `buf` over the phase's PEs as `shard`.
    pub fn shard(&mut self, buf: &Buf, shard: Shard) {
        self.shards.retain(|(n, _)| *n != buf.name);
        self.shards.push((buf.name.clone(), shard));
    }

    /// After the phase's statements, the fabric adds every PE's `partial`
    /// (`count` f32 words) along each row of the `rect` rectangle into the
    /// row's first PE's share of `into` (sharded [`Shard::Roots`]).
    pub fn fabric_sum(&mut self, partial: &str, into: &Buf, count: u64, rect: (u32, u32)) {
        self.fabric_sum_with(partial, into, count, rect, false);
    }

    /// [`Cx::fabric_sum`], then (`everywhere`) the sum broadcast back over
    /// the row so every PE holds it: what a whole activation needs under
    /// the resident placement, where the next phase reads it in place.
    pub fn fabric_sum_with(&mut self, partial: &str, into: &Buf, count: u64, rect: (u32, u32), everywhere: bool) {
        let mut steps = vec![FabricStep {
            calls: Vec::new(),
            op: FabricOp::Reduce {
                send: partial.to_string(),
                recv: into.name.clone(),
                count,
            },
        }];
        if everywhere {
            steps.push(FabricStep {
                calls: Vec::new(),
                op: FabricOp::Broadcast {
                    buf: into.name.clone(),
                    count,
                },
            });
        }
        self.fabric(rect, steps, Vec::new());
    }

    /// `buf` is filled by this phase's fabric steps, not its function: its
    /// bf16 rounding (when it is bf16) runs after the steps.
    pub fn round_after_steps(&mut self, buf: &Buf) {
        self.late_rounds.push(buf.name.clone());
    }

    /// After the phase's statements, the `steps` run on every PE of the
    /// `rect` rectangle, each ending in a collective along the PE's row;
    /// `finish` runs when the last lands.
    pub fn fabric(&mut self, rect: (u32, u32), steps: Vec<FabricStep>, finish: Vec<Guarded>) {
        self.reduce = Some(FabricReduce {
            rect,
            steps,
            finish,
            round: Vec::new(),
        });
    }

    /// How many elements of `buf` each PE holds in this phase.
    fn local_len(&self, buf: &Buf) -> u64 {
        match self.shards.iter().find(|(n, _)| *n == buf.name) {
            Some((_, Shard::Local(n))) => *n,
            Some((_, s)) => buf.len() / u64::from(s.pes()),
            None => buf.len(),
        }
    }

    /// The device array holding `t`, in f32.
    pub fn read(&mut self, t: Tensor) -> Result<Buf, Error> {
        let v = self.env.read(self.p, t)?;
        self.window(t, v, false)
    }

    /// `t` packed two bf16 a u32 word, when the engine keeps it so and it
    /// is a whole root (no window); else `None`, and the caller reads it
    /// as f32.
    pub fn read_half(&mut self, t: Tensor) -> Result<Option<Buf>, Error> {
        match self.env.read_half(self.p, t)? {
            Some(v) if v.offset == 0 && v.rows == v.whole.rows => Ok(Some(v.whole)),
            _ => Ok(None),
        }
    }

    /// A kv pool plane packed two bf16 a word, read and written in place,
    /// when the engine keeps pools so (see [`Env::pool_half`]).
    pub fn pool_half(&mut self, t: Tensor) -> Result<Option<Buf>, Error> {
        match self.env.pool_half(self.p, t)? {
            Some(v) if v.offset == 0 && v.rows == v.whole.rows => {
                self.p.written(&v.whole.name);
                Ok(Some(v.whole))
            }
            _ => Ok(None),
        }
    }

    /// A row window of a root as its own exported buffer: the host cuts it
    /// out of the root before the phase and pastes it back after.
    fn window(&mut self, t: Tensor, v: View, write: bool) -> Result<Buf, Error> {
        let _ = write;
        if v.offset == 0 && v.rows == v.whole.rows {
            return Ok(v.whole);
        }
        if let Some((_, buf)) = self.windows.iter().find(|w| w.0 == t.buf) {
            return Ok(buf.clone());
        }
        let name = self.p.unique(&format!("w{}", v.whole.name));
        let off = u64::from(v.offset) * u64::from(v.whole.width);
        let buf = self
            .p
            .view(&name, &v.whole.name, off, v.rows, v.whole.width)?;
        self.windows.push((t.buf, buf.clone()));
        Ok(buf)
    }

    /// The device array to write `t` into. Statements the entry emits after
    /// this compute in f32; if `t` stores bf16, the values are rounded to
    /// bf16 (nearest even) when the phase ends, so the write rounds once.
    pub fn write(&mut self, t: Tensor) -> Result<Buf, Error> {
        let v = self.env.write(self.p, t)?;
        let b = self.window(t, v, true)?;
        match t.dtype {
            Dtype::F32 | Dtype::I32 | Dtype::U32 | Dtype::U8 => {}
            Dtype::Bf16 => {
                if !self.rounds.contains(&b) {
                    self.rounds.push(b.clone());
                }
            }
            other => {
                return Err(Error::DtypeUnsupported {
                    op: "write",
                    dtype: other,
                });
            }
        }
        Ok(b)
    }

    /// Rows `[c0, c0 + n)` of `b` as their own exported window (the host
    /// cuts and pastes it); a bf16 rounding due on `b` moves to the window,
    /// so the phase names the window alone.
    pub fn window_of(&mut self, b: &Buf, c0: u32, n: u32) -> Result<Buf, Error> {
        if n >= b.rows && c0 == 0 {
            return Ok(b.clone());
        }
        let name = self.p.unique(&format!("w{}", b.name));
        let w = self.p.view(
            &name,
            &b.name,
            u64::from(c0) * u64::from(b.width),
            n,
            b.width,
        )?;
        if let Some(r) = self.rounds.iter_mut().find(|r| r.name == b.name) {
            *r = w.clone();
        }
        Ok(w)
    }

    /// `b` as a `rows × width` plane over the same words (a row-major
    /// reshape), or `b` itself when that is its shape.
    pub fn view_as(&mut self, b: &Buf, rows: u32, width: u32) -> Result<Buf, Error> {
        if b.rows == rows && b.width == width {
            return Ok(b.clone());
        }
        if u64::from(rows) * u64::from(width) != u64::from(b.rows) * u64::from(b.width) {
            return Err(refuse(
                "view_as",
                format!(
                    "{} is {}x{} and cannot be seen as {rows}x{width}",
                    b.name, b.rows, b.width
                ),
            ));
        }
        let name = self.p.unique(&format!("v{}", b.name));
        let v = self.p.view(&name, &b.name, 0, rows, width)?;
        if let Some(r) = self.rounds.iter_mut().find(|r| r.name == b.name) {
            *r = v.clone();
        }
        Ok(v)
    }

    /// A private f32 array of `len` elements.
    pub fn scratch(&mut self, hint: &str, len: u64) -> String {
        self.p.scratch(hint, len)
    }

    /// A private array of `len` elements of `elem`.
    pub fn scratch_of(&mut self, hint: &str, len: u64, elem: &'static str) -> String {
        self.p.scratch_of(hint, len, elem)
    }

    /// A global `mem1d_dsd` named after `hint`; phases copy and advance it.
    pub fn dsd(&mut self, hint: &str, d: &Dsd1) -> String {
        let name = self.p.unique(&format!("d_{hint}"));
        self.p.global(format!("const {name} = {};", d.expr()));
        name
    }

    /// A helper function shared by phases, declared once per `name`.
    pub fn helper(&mut self, name: &str, source: impl Into<String>) {
        self.p.helper(name, source)
    }

    /// The kernel library section `name`, declared once; returns the name.
    pub fn library(&mut self, name: &'static str) -> &'static str {
        let source = crate::library::section(name)
            .unwrap_or_else(|| panic!("no kernel library section {name}"));
        self.p.helper(name, source);
        name
    }

    /// A kernel call in this phase: rendered as a statement and kept
    /// structured for the table-driven program. The kernel's library
    /// section is registered by the caller (`library`) as before.
    pub fn call(&mut self, kernel: &str, args: Vec<crate::csl::Arg>) {
        self.phase.body.call(crate::csl::Call {
            kernel: kernel.to_string(),
            args,
        });
    }

    /// A fresh local identifier.
    pub fn unique(&mut self, hint: &str) -> String {
        self.p.unique(hint)
    }

    /// A constant table (`var NAME = [n]elem { lits };`) named by its
    /// contents, so equal tables share one global (declared once) and one
    /// kept slot across every program of a model.
    pub fn table(&mut self, hint: &str, elem: &str, lits: &[String]) -> String {
        use std::hash::{Hash, Hasher};
        let body = lits.join(", ");
        let mut h = std::collections::hash_map::DefaultHasher::new();
        elem.hash(&mut h);
        body.hash(&mut h);
        let name = format!("{hint}_{:016x}", h.finish());
        self.p.global(format!("var {name} = [{}]{elem} {{ {body} }};", lits.len()));
        name
    }

    pub fn program(&mut self) -> &mut Program {
        self.p
    }

    /// Closes the phase and appends it to the program.
    pub fn finish(mut self) {
        if let Some(op) = self.host.take() {
            let rounds = std::mem::take(&mut self.rounds)
                .into_iter()
                .map(|b| b.name)
                .collect();
            let phase = std::mem::replace(&mut self.phase, Func::new("closed"));
            self.p.host_phase(phase, HostPhase { op, rounds });
            return;
        }
        for b in std::mem::take(&mut self.rounds) {
            // A partial sum stays f32 on the PE; the host rounds the sum.
            let summed = self.shards.iter().any(|(n, s)| {
                *n == b.name && matches!(s, Shard::SumCols { .. } | Shard::SumGrid { .. })
            }) || self.lane.as_ref().is_some_and(|l| {
                l.reduce
                    .iter()
                    .any(|(n, r)| *n == b.name && matches!(r, Reduce::Sum))
            });
            if summed {
                continue;
            }
            // A buffer the fabric merges is rounded once the merge lands.
            let roots = self.late_rounds.contains(&b.name)
                || self
                    .shards
                    .iter()
                    .any(|(n, s)| *n == b.name && matches!(s, Shard::Roots { .. }))
                || self.lane.as_ref().is_some_and(|l| {
                    l.reduce.iter().any(|(n, r)| {
                        *n == b.name
                            && matches!(r, Reduce::FabricSum { .. } | Reduce::FabricRoot { .. })
                    })
                });
            if roots {
                let n = self.local_len(&b);
                if let Some(r) = self.reduce.as_mut() {
                    r.round.push((b.name.clone(), n));
                }
                continue;
            }
            let n = self.local_len(&b);
            self.phase.body.call(crate::csl::Call {
                kernel: "k_round_bf16".to_string(),
                args: vec![crate::csl::Arg::Ptr(b.clone()), crate::csl::Arg::Int(n as i64)],
            });
        }
        let phase = std::mem::replace(&mut self.phase, Func::new("closed"));
        let shards = std::mem::take(&mut self.shards);
        let lane = self.lane.take();
        let reduce = self.reduce.take();
        // The resident placement: every phase on the row of PEs (whole
        // buffers replicated; a phase on fewer PEs stays as it is only when
        // it already spans the row).
        let pes = match crate::linear::gemm::resident_pes() {
            Some(n) if lane.is_none() && self.pes == 1 => n,
            _ => self.pes,
        };
        self.p.phase_over(phase, pes, shards, lane, reduce);
    }
}

impl Deref for Cx<'_> {
    type Target = Block;
    fn deref(&self) -> &Block {
        &self.phase.body
    }
}

impl DerefMut for Cx<'_> {
    fn deref_mut(&mut self) -> &mut Block {
        &mut self.phase.body
    }
}

/// The fewest PEs (a divisor of `rows`) a row-wise phase spreads its rows
/// over so that its `per_row` words a row plus `whole` words (weights,
/// scratch) fit one PE's [`pe_words`](crate::linear::gemm::pe_words).
pub fn rows_split(op: &'static str, rows: u32, per_row: u64, whole: u64) -> Result<u32, Error> {
    use crate::linear::gemm::pe_words;
    let rows = rows.max(1);
    let budget = pe_words();
    (1..=rows)
        .filter(|d| rows.is_multiple_of(*d))
        .find(|d| per_row * u64::from(rows / d) + whole <= budget)
        .ok_or_else(|| {
            refuse(
                op,
                format!("one row's {per_row} words and the phase's {whole} other words exceed a PE's {budget}"),
            )
        })
}

/// The `(row groups, column groups)` a row-wise phase over `rows × width`
/// spreads over so that one PE's share fits: `per_row` words a row that
/// split with the columns, `per_col` words a column (a gain) that split
/// with the columns, `whole` words that do not split; columns split only
/// in multiples of `unit` (a head). The fewest PEs win, rows first.
pub fn tile_split(
    op: &'static str,
    rows: u32,
    width: u32,
    per_row: u64,
    per_col: u64,
    whole: u64,
    unit: u32,
) -> Result<(u32, u32), Error> {
    use crate::linear::gemm::{ARRAY_WORDS, pe_words, resident_pes};
    let budget = pe_words();
    let (rows, width, unit) = (rows.max(1), width.max(1), unit.max(1));
    // The resident placement runs a row-wise phase whole on every PE.
    if resident_pes().is_some() {
        let words = per_row * u64::from(rows) + per_col * u64::from(width) + whole;
        return if words <= budget && per_row * u64::from(rows) <= ARRAY_WORDS {
            Ok((1, 1))
        } else {
            Err(refuse(
                op,
                format!("the phase's {words} words exceed a resident PE's {budget}"),
            ))
        };
    }
    let units = width / unit;
    let fits = |rg: u32, cg: u32| {
        let words = per_row * u64::from(rows / rg) / u64::from(cg)
            + per_col * u64::from(width / cg)
            + whole;
        words <= budget && per_row * u64::from(rows / rg) / u64::from(cg) <= ARRAY_WORDS
    };
    let mut best: Option<(u32, u32)> = None;
    for cg in (1..=units).filter(|d| units.is_multiple_of(*d)) {
        for rg in (1..=rows).filter(|d| rows.is_multiple_of(*d)) {
            if fits(rg, cg) {
                if best.is_none_or(|(br, bc)| rg * cg < br * bc) {
                    best = Some((rg, cg));
                }
                break;
            }
        }
    }
    best.ok_or_else(|| {
        refuse(
            op,
            format!(
                "one row of {width} ({per_row} words with its planes) does not fit a PE's {budget} even one column of {unit} at a time"
            ),
        )
    })
}

/// Refuses unless `t` is one of `dtypes`.
pub fn expect(op: &'static str, t: Tensor, dtypes: &[Dtype]) -> Result<(), Error> {
    if dtypes.contains(&t.dtype) {
        Ok(())
    } else {
        Err(Error::DtypeUnsupported { op, dtype: t.dtype })
    }
}

/// Refuses unless `t` is `[rows, width]`.
pub fn shaped(op: &'static str, what: &str, t: Tensor, rows: u32, width: u32) -> Result<(), Error> {
    if t.rows == rows && t.width == width {
        Ok(())
    } else {
        Err(refuse(
            op,
            format!(
                "the {what} is {}x{}, and this op wants {rows}x{width}",
                t.rows, t.width
            ),
        ))
    }
}

/// A recording sink over one program: what a bench or a unit test emits
/// into. [`Emit::emit`] opens a phase per call.
pub struct Tracer {
    program: std::cell::RefCell<Program>,
    env: std::cell::RefCell<Box<dyn Env>>,
}

impl Tracer {
    pub fn new(env: impl Env + 'static) -> Self {
        Tracer {
            program: std::cell::RefCell::new(Program::new()),
            env: std::cell::RefCell::new(Box::new(env)),
        }
    }

    pub fn into_program(self) -> Program {
        self.program.into_inner()
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new(Roots)
    }
}

impl Emit for Tracer {
    fn emit(&self, body: &mut dyn FnMut(&mut Cx<'_>) -> Result<(), Error>) -> Result<(), Error> {
        let mut p = self.program.borrow_mut();
        let mut env = self.env.borrow_mut();
        let mut cx = Cx::new(&mut p, env.as_mut(), "op");
        body(&mut cx)?;
        cx.finish();
        Ok(())
    }
}
