//! The device work an entry emits: phases of kernel calls over buffers, each
//! saying the PEs it runs on and how its buffers spread over them, and the
//! [`Sink`] the engine supplies to place operands and take the phases.
//!
//! Kernels emit device work; the engine owns where data lives and when it
//! runs: which buffer is an exported symbol or an arena slot, which phases
//! fuse into one program, whether a program is a function or a table.

use dtype::Dtype;

use crate::csl::{Arg, Call};
use crate::error::Error;
use crate::library::Kernel;
use crate::plan::{LaneSplit, Plan};
use crate::tensor::Tensor;

/// What an entry emits into: the engine's program under construction. It
/// places operand handles on buffers, declares the buffers and private
/// arrays an entry asks for, and takes each phase when it closes.
pub trait Sink {
    /// The device array holding `t` now: its root buffer, or a row window
    /// of the root the host cuts out before the phase (one a handle within
    /// a phase).
    fn read(&mut self, t: Tensor) -> Result<Buf, Error>;

    /// The device array `t` holds from here on (a root, or a row window the
    /// host pastes back after the phase).
    fn write(&mut self, t: Tensor) -> Result<Buf, Error>;

    /// `t` as a packed bf16 buffer (two halves a word), when the engine
    /// keeps it so (a half-precision weight read whole); `None` means read
    /// it as f32.
    fn read_half(&mut self, t: Tensor) -> Result<Option<Buf>, Error>;

    /// A kv pool `t` as a packed bf16 buffer under its plain name, read and
    /// written in place, when the engine keeps pools so and `t` is the
    /// whole pool; `None` means f32.
    fn pool_half(&mut self, t: Tensor) -> Result<Option<Buf>, Error>;

    /// A fresh identifier with `hint` in it.
    fn unique(&mut self, hint: &str) -> String;

    /// Declares (or finds) the buffer `name` holding `rows × width` elements
    /// of `dtype`; `input` marks it read before any phase writes it. A
    /// second declaration must agree on the shape.
    fn declare(
        &mut self,
        name: &str,
        dtype: Dtype,
        rows: u32,
        width: u32,
        input: bool,
    ) -> Result<Buf, Error>;

    /// `name` as a window of `rows × width` elements at `offset` of the
    /// buffer `root`.
    fn view(
        &mut self,
        name: &str,
        root: &str,
        offset: u64,
        rows: u32,
        width: u32,
    ) -> Result<Buf, Error>;

    /// Marks buffer `name` as written by a phase.
    fn written(&mut self, name: &str);

    /// A private array of `len` elements of `elem` (`f32`, `i32`, `u32`).
    fn scratch(&mut self, hint: &str, len: u64, elem: &'static str) -> String;

    /// A constant table of `lits.len()` elements of `elem`, by its name
    /// (`hint` in it): equal contents share one name, declared once.
    fn table(&mut self, hint: &str, elem: &str, lits: &[String]) -> String;

    /// A raw global declaration line, once (the device probes of CSL's own
    /// semantics only).
    fn global(&mut self, line: String);

    /// Places a lane phase: the PEs it runs on, its header, and the share of
    /// a PE each buffer it holds by lane or block takes; or why the
    /// placement refuses it.
    fn lanes(&mut self, lanes: &Lanes, split: LaneSplit) -> Result<LanesAt, Error>;

    /// Appends a closed phase; phases run in the order they are appended.
    fn phase(&mut self, phase: Phase);

    /// The engine's planner: its terms, and the splits it chooses.
    fn plan(&self) -> &dyn Plan;
}

/// A device buffer a kernel computes with: an array named `name` of `elem`
/// (`f32` for float handles, `i32`/`u32` for integer ones) holding
/// `rows × width` elements row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buf {
    pub name: String,
    pub rows: u32,
    pub width: u32,
    pub elem: &'static str,
}

impl Buf {
    pub fn len(&self) -> u64 {
        self.rows as u64 * self.width as u64
    }

    /// A many-item pointer to the array, for the kernel library.
    pub fn ptr(&self) -> String {
        format!("@ptrcast([*]{}, &{})", self.elem, self.name)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// How a phase's rectangle of PEs holds a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Shard {
    /// Every PE holds the whole array.
    #[default]
    Whole,
    /// PE `x` of `n` holds rows `[x * rows / n, (x + 1) * rows / n)`.
    Rows(u32),
    /// PE `x` of `n` holds every row's columns `[x * width / n, (x + 1) * width / n)`.
    Cols(u32),
    /// Every PE holds `elements` of the buffer, laid out by the phase's
    /// lanes ([`Lanes`]) on the host.
    Local(u64),
    /// PE `x` holds row part `(x / period) % parts` of `parts`: rows split
    /// over one axis of a rectangle whose other axis is `period` wide.
    RowsBy { parts: u32, period: u32 },
    /// PE `x` of a `rows × cols` grid holds row part `x / cols` and column
    /// part `x % cols`, laid out `[rows / parts][width / cols]`.
    Grid { rows: u32, cols: u32 },
    /// PE `x` holds column part `(x / period) % parts` of `parts`.
    ColsBy { parts: u32, period: u32 },
    /// PE `x` holds column part `x / period` of `parts`, a partial sum: the
    /// host adds the `period` PEs' copies of each part (f32).
    SumCols { parts: u32, period: u32 },
    /// PE `x` of a `rows × cols` grid holds row part `x / cols` and, of
    /// each of the `segments` equal segments of a row, column part `x %
    /// cols`, laid out `[rows / parts][width / cols]` (the segments' parts
    /// back to back).
    Tile { rows: u32, cols: u32, segments: u32 },
    /// The activations of a `rows × cols × depth` matmul grid: PE `x`
    /// holds row part `x / (cols · depth)` and column part `x % depth`.
    RowsDepth { rows: u32, cols: u32, depth: u32 },
    /// The weight of that grid: PE `x` holds row part `(x / depth) % cols`
    /// and column part `x % depth`.
    ColsDepth { rows: u32, cols: u32, depth: u32 },
    /// The output of that grid: PE `x` holds row part `x / (cols · depth)`
    /// and column part `(x / depth) % cols`, a partial the host adds over
    /// the `depth` PEs sharing them (f32).
    SumGrid { rows: u32, cols: u32, depth: u32 },
    /// The output of that grid summed on the fabric: the `depth` PEs of a
    /// block (one row of the program's `depth × (rows · cols)` rectangle)
    /// reduce their partials into the row's first PE (`x % depth == 0`),
    /// which alone holds the block after the run.
    Roots { rows: u32, cols: u32, depth: u32 },
}

impl Shard {
    /// How many distinct shares the buffer splits into.
    pub fn pes(self) -> u32 {
        match self {
            Shard::Whole | Shard::Local(_) => 1,
            Shard::Rows(n) | Shard::Cols(n) => n,
            Shard::RowsBy { parts, .. }
            | Shard::ColsBy { parts, .. }
            | Shard::SumCols { parts, .. } => parts,
            Shard::Grid { rows, cols } | Shard::Tile { rows, cols, .. } => rows * cols,
            Shard::RowsDepth { rows, depth, .. } => rows * depth,
            Shard::ColsDepth { cols, depth, .. } => cols * depth,
            Shard::SumGrid { rows, cols, .. } | Shard::Roots { rows, cols, .. } => rows * cols,
        }
    }
}

/// How a phase's lanes are found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneKind {
    /// One row per lane: `lanes` rows.
    PerRow { lanes: u32 },
    /// A CSR `indptr` (an exported i32 buffer of `lanes + 1` entries) over
    /// the rows.
    Ragged { indptr: String, lanes: u32 },
    /// Each row names its lane in `lane_of_row` (an exported i32 buffer);
    /// a lane's rows must be contiguous.
    ByTable { lane_of_row: String, lanes: u32 },
}

/// Where a lane's pages are listed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageSource {
    /// Lane `l` holds pages `indices[indptr[l]..indptr[l + 1]]`.
    Csr { indptr: String, indices: String },
    /// Row `r` (its own lane) names page `table[r]`.
    Rows { table: String },
}

/// How a bank's slot lands on a PE under a block plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankSplit {
    /// The whole slot on every PE of the lane.
    Whole,
    /// The PE's block `[a0, a1) × [b0, b1) × w` of the slot as `[a][b][w]`.
    Block,
    /// Rows `[b0, b1)` of the slot as `[b][stride / b]` (a bank whose rows
    /// follow the plan's `b` axis and nothing else).
    ByB,
}

/// One slot's state `[a][b][w]`: `w` words contiguous, split (when the
/// engine splits it) by ranges of `a` and of `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotState {
    pub a: u32,
    pub b: u32,
    pub w: u32,
}

/// A plane every PE holds a block of: its `segments` pick, from the PE's
/// block `[a0, a1) × [b0, b1)` (all when the plan has no block), the
/// columns it holds of every row (laid out `[rows][cols]`, the columns in
/// ascending order) or, `by_rows`, the rows it holds whole (`[rows'][width]`).
/// The words a PE changes come back after the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColPlane {
    pub name: String,
    pub rows: u32,
    pub width: u32,
    pub by_rows: bool,
    pub segments: Vec<Segment>,
    /// The plane's rows are the block's `[a0, a1)` and its columns what the
    /// segments pick over `[b0, b1)` (a key/value plane split by key rows
    /// and by heads).
    pub row_block: bool,
}

/// One run of a plane's indices a block picks: for each `i` in `[a0, a1)`
/// and each `j` in `[b0, b1)` (just `0` when `b_stride` is 0), the `span`
/// indices from `base + (i / a_group) * a_stride + j * b_stride` (the `i`
/// term is dropped when `a_group` is 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub base: u32,
    pub a_group: u32,
    pub a_stride: u32,
    pub b_stride: u32,
    pub span: u32,
}

impl Segment {
    /// Columns (or rows) `[b0, b1)` of a plane: what a conv's channels pick.
    pub const fn along_b() -> Self {
        Segment {
            base: 0,
            a_group: 0,
            a_stride: 0,
            b_stride: 1,
            span: 1,
        }
    }
}

/// The paged form of a lane's state: lane `l` holds pages
/// `indices[indptr[l]..indptr[l + 1]]` of the banks (each page `page_size`
/// rows of a bank plane), at most `max_pages` a lane; a PE reads `indptr`,
/// `indices` and the `rewritten` per-row page tables (kv_append's
/// `write_page`) with local page numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paged {
    pub source: PageSource,
    pub page_size: u32,
    pub max_pages: u32,
    /// Per-row page tables rewritten to local pages for the PE's rows.
    pub rewritten: Vec<String>,
}

/// How a row output held by several PEs (page groups) is merged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reduce {
    /// The plane is a per-head log-sum-exp (base 2): merged as `log2 Σ 2^v`.
    LogSumExp,
    /// The plane is a per-head normalised sum (`[heads × d]` a row) with its
    /// log-sum-exp in `lse` (`[heads]` a row): merged weighted by `2^(lse_p
    /// - lse)`.
    Weighted { lse: String },
    /// Every PE adds a partial to the cells it holds: the host sums what
    /// each PE changed (f32), rounding to bf16 after when the buffer is.
    Sum,
    /// Every PE holds a partial of the cells it holds; the fabric adds the
    /// `group` consecutive PEs' partials (`p / group` equal) into the first
    /// of them (`p % group == 0`), which alone supplies the cells after the
    /// run; the others are given zeros, so a row no PE writes comes back
    /// as it went.
    FabricSum { group: u32 },
    /// Every PE holds a partial of the cells it holds; the phase's own
    /// fabric steps merge the `group` consecutive PEs' partials into the
    /// first of them, which alone supplies the cells after the run (the
    /// others' copies are uploaded as they are).
    FabricRoot { group: u32 },
}

/// A phase spread over PEs by lanes, as its entry states it: how the
/// lanes are found, the banks and planes a PE holds its share of, and how
/// row outputs held by several PEs merge. The engine splits it
/// ([`LaneSplit`]) and places it ([`Sink::lanes`]): PE `p` runs lanes
/// `[l0, l1)` (rows `[r0, r1)`) over block `[a0, a1) × [b0, b1)`, read from
/// its row of the header (`l0, l1, held slots, a0, a1, b0, b1, r0, r1, pg0,
/// pages, 0…`), holding only its lanes' slot blocks of `banks`, with
/// `slots` rewritten to those local rows. The host cuts and pastes all of
/// it around the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lanes {
    pub kind: LaneKind,
    /// The slot table (i32, one entry per row): lane `l`'s slot is entry
    /// `l` (per row) or entry `indptr[l]` (ragged).
    pub slots: String,
    /// Banks `(name, stride, split)`: `[slots_total, stride]` f32 planes; a
    /// PE holds its local slot rows, each the whole `stride`, its block of
    /// `[a][b][w]`, or rows `[b0, b1)` of `[b][stride / b]`.
    pub banks: Vec<(String, u32, BankSplit)>,
    /// A slot's state, when the engine may split it by blocks; `stride`
    /// then equals `a × b × w`.
    pub block: Option<SlotState>,
    /// The PE order: page groups vary fastest, then the b axis's groups,
    /// then the a axis's, then the lane groups; `a_first` swaps the two
    /// block axes, so a merge over page groups and the a axis (attention's
    /// key rows) spans consecutive PEs.
    pub a_first: bool,
    /// Outputs written row by row per lane `(name, width, a_stride,
    /// b_stride)`: after the run PE `p`'s copy supplies, for the rows of
    /// lanes `[l0, l1)`, the columns `i * a_stride + j * b_stride` of its
    /// block (all columns when the state is not split by blocks).
    pub row_outputs: Vec<(String, u32, u32, u32)>,
    /// Planes held by block (columns, or rows): activations and weights
    /// too wide for a PE.
    pub cols: Vec<ColPlane>,
    /// How row outputs held by several PEs merge (page groups > 1).
    pub reduce: Vec<(String, Reduce)>,
    /// The lanes' rows clipped to `[start, start + len)` and counted from
    /// `start`: the row-shaped buffers of this phase are that window.
    pub window: Option<(u32, u32)>,
    /// Paged banks instead of one slot per lane.
    pub pages: Option<Paged>,
}

/// A lane phase: what its entry stated, the split the engine chose, and
/// where the engine placed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanePhase {
    pub lanes: Lanes,
    pub split: LaneSplit,
    pub at: LanesAt,
}

/// Where the engine placed a lane phase ([`Sink::lanes`]): its header
/// buffer, the PEs it runs on, each buffer's share of a PE, and the words a
/// PE holds of each of the plan's `cols`, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanesAt {
    pub header: Buf,
    pub pes: u32,
    pub shards: Vec<(String, Shard)>,
    pub planes: Vec<u64>,
}

/// One phase: its name and the kernel calls it makes, the PEs it runs on,
/// and how each buffer it names is spread over them (unlisted buffers are
/// whole on every PE); or a host op, run by the host over its copies of the
/// buffers (the calls are then dropped).
#[derive(Debug, Clone)]
pub struct Phase {
    pub name: String,
    pub body: Vec<Statement>,
    pub pes: u32,
    pub shards: Vec<(String, Shard)>,
    pub lane: Option<LanePhase>,
    pub host: Option<HostOp>,
    /// A sum the fabric takes after the function runs.
    pub reduce: Option<FabricReduce>,
    /// The bf16 buffers the phase writes, computed in f32: each rounds to
    /// bf16 once, where the engine puts it (after the calls, after the
    /// fabric's merge, or on the host after a sum).
    pub rounds: Vec<Buf>,
    /// Of `rounds`, those the fabric steps fill (a collective, not the
    /// calls): they round after the steps.
    pub late_rounds: Vec<String>,
}

/// One statement of a phase: a kernel call, or raw CSL lines (only the
/// device probes of CSL's own semantics write them).
#[derive(Debug, Clone)]
pub enum Statement {
    Call(Call),
    Raw(crate::csl::Block),
}

/// A phase whose PEs combine their results over the fabric after the
/// function runs: the program's rectangle is `rect.0` PEs wide, each row
/// of it one group, and the `steps` run in order on every PE, each its
/// statements then one `collectives_2d` operation along the row (rooted at
/// the row's first PE), the next step starting when the operation lands;
/// after the last, every PE rounds `round` and runs `finish`, and the run
/// ends. `px` holds the PE's column in the statements.
#[derive(Debug, Clone, PartialEq)]
pub struct FabricReduce {
    pub rect: (u32, u32),
    pub steps: Vec<FabricStep>,
    pub finish: Vec<Guarded>,
    /// Buffers rounded to bf16 when the steps are done: `(name, words)`
    /// (the engine's, from the phase's `rounds`).
    pub round: Vec<(String, u64)>,
}

/// Kernel calls, then one collective along the rectangle row. A call
/// flagged `root` runs on the row's first PE only.
#[derive(Debug, Clone, PartialEq)]
pub struct FabricStep {
    pub calls: Vec<Guarded>,
    pub op: FabricOp,
}

/// A kernel call that every PE runs, or the row's root alone.
#[derive(Debug, Clone, PartialEq)]
pub struct Guarded {
    pub root: bool,
    pub call: Call,
}

impl Guarded {
    pub fn all(kernel: Kernel, args: Vec<Arg>) -> Self {
        Guarded {
            root: false,
            call: Call { kernel, args },
        }
    }

    pub fn root(kernel: Kernel, args: Vec<Arg>) -> Self {
        Guarded {
            root: true,
            call: Call { kernel, args },
        }
    }

    /// The statement: the call, under `if (px == 0)` for the root alone.
    pub fn render(&self) -> String {
        if self.root {
            format!("if (px == 0) {{ {} }}", self.call.render())
        } else {
            self.call.render()
        }
    }
}

/// One `collectives_2d` operation over the row, rooted at its first PE;
/// `count` is in 32-bit words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FabricOp {
    /// `recv = Σ send` (f32) on the root.
    Reduce {
        send: String,
        recv: String,
        count: u64,
    },
    /// `recv[p · count ..]` = PE `p`'s `send` on the root.
    Gather {
        send: String,
        recv: String,
        count: u64,
    },
    /// Every PE's `buf` = the root's.
    Broadcast { buf: String, count: u64 },
    /// Every PE's `buf` = PE `root`'s (a page moving between PEs).
    BroadcastFrom { root: u32, buf: String, count: u64 },
    /// Down the column: every row's `buf` = row `root`'s (a buffer one
    /// row of a whole fire produced and later rows read).
    Handoff { root: u32, buf: String, count: u64 },
}

/// An op the host runs instead of the fabric: pure data movement, or a
/// vocabulary-sized op no rectangle holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostOp {
    /// `y[r] = table[ids[r]]` for `rows` rows of `width`; an id outside
    /// `[0, limit)` reads row 0.
    Embed {
        ids: String,
        table: String,
        y: String,
        rows: u32,
        width: u32,
        limit: u32,
    },
    /// `left[r] = x[r][..cut]`, `right[r] = x[r][cut..]` for `rows` rows of
    /// `width`.
    SplitRows {
        x: String,
        left: String,
        right: String,
        rows: u32,
        width: u32,
        cut: u32,
    },
    /// Per row and head, `packed = [q_h ‖ gate_h]` cut into `q` and `gate`.
    SplitQGate {
        packed: String,
        q: String,
        gate: String,
        rows: u32,
        heads: u32,
        head_dim: u32,
    },
    /// `y[r][column] = argmax of row r of x` (`rows` rows of `width`; ties
    /// to the lowest index, NaN never picked, an all-NaN row 0), `y` i32.
    Argmax {
        x: String,
        y: String,
        rows: u32,
        width: u32,
        column: u32,
        y_width: u32,
    },
    /// `y[r][column]` = the index of the greatest of row `r`'s `parts`
    /// (value, local index) pairs the fabric found over `cg` column blocks
    /// of `block` lanes (`parts[r][2c]`, `parts[r][2c + 1]`; a block with no
    /// lane holds index -1), the lowest index on a tie, 0 when none.
    ArgmaxMerge {
        parts: String,
        y: String,
        rows: u32,
        cg: u32,
        block: u32,
        column: u32,
        y_width: u32,
    },
    /// `routes[t][s]`, `weights[t][s]`: the `top_k` largest of the first
    /// `experts` of row `t` of `logits` (`rows` rows of `width`) and the
    /// softmax over them; picks past the experts hold `-1` and 0.
    TopkSoftmax {
        logits: String,
        routes: String,
        weights: String,
        rows: u32,
        width: u32,
        experts: u32,
        top_k: u32,
    },
    /// `q[r] = packed[r][..q_width]`, `k[r]`, `v[r]` the two `kv_width`
    /// runs after.
    SplitQkv {
        packed: String,
        q: String,
        k: String,
        v: String,
        rows: u32,
        q_width: u32,
        kv_width: u32,
    },
    /// `y[r] = Σ_t weights[r][t] · table[ids[r][t]]` (`taps` taps, rows of
    /// `width`; an id outside `[0, limit)` reads row 0).
    EmbedWeighted {
        ids: String,
        weights: String,
        table: String,
        y: String,
        rows: u32,
        taps: u32,
        width: u32,
        limit: u32,
    },
    /// `y[i][j] = Σ_k act[i][k] · w[j][k]` for `m` rows, `k` deep, `n` wide.
    Matmul {
        act: String,
        w: String,
        y: String,
        m: u32,
        k: u32,
        n: u32,
    },
    /// Slice `g` of row `r` of `x` (`rows` rows of `groups` slices `k`
    /// wide) times expert `routes[r][g]` of `w` (`experts` blocks of `n × k`)
    /// into slice `g` of row `r` of `y` (`groups` slices `n` wide); a route
    /// outside the experts leaves its slice as it was.
    MatmulGrouped {
        x: String,
        w: String,
        routes: String,
        y: String,
        rows: u32,
        groups: u32,
        k: u32,
        n: u32,
        experts: u32,
    },
    /// `values[r][s]`, `indices[r][s]`: the `k` largest of row `r` of `x`
    /// (`rows` rows of `width`), largest first, ties to the lower column,
    /// NaN never picked; a slot nothing fills holds 0 at column 0.
    TopK {
        x: String,
        values: String,
        indices: String,
        rows: u32,
        width: u32,
        k: u32,
    },
    /// The drafter's greedy walk over each lane's candidate rows (`rows`
    /// rows of `k` ids, lanes by `indptr`): row `t` scores its candidates
    /// `unary[t][c] + Σ_d pred[prev][d] · hp[t][d] · succ[cand][d]` (the
    /// bilinear term only when both ids are in the `vocab`; `hp` 1 when
    /// absent), picks the first best, and the pick is the next row's `prev`
    /// (the lane's first `prev` is `tokens[begin]`). With `first` 1 the
    /// anchor row takes its first candidate unscored. Rows outside every
    /// lane keep `picks`.
    SelectorWalk {
        cand: String,
        indptr: String,
        unary: String,
        hp: Option<String>,
        tokens: String,
        pred: String,
        succ: String,
        picks: String,
        rows: u32,
        k: u32,
        rank: u32,
        vocab: u32,
        first: u32,
    },
    /// Per-layer-embedding n-gram hashing: each row's id and the ids before
    /// it in its lane (the lane's kept window first, `state` cells of `id +
    /// 1`, 0 reading as `eos`; past the first eos every older id reads as
    /// eos) hash to one table row per head: `mixed_o = XOR_{p < o} u64(w_p)
    /// · mults[p]`, head `i` of order `o` is `(mixed_o % primes[i] +
    /// offsets[i])` low 32 bits. Lanes are the CSR `indptr` over `rows`
    /// rows (none: one row per lane), a lane's slot that of its first row
    /// in `slots`; a lane's last `span` ids land as its kept cells.
    PleNgramIds {
        ids: String,
        indptr: Option<String>,
        slots: String,
        state: String,
        out: String,
        rows: u32,
        eos: u32,
        mults: Vec<u64>,
        primes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: u32,
    },
    /// `table[write_page[r] · page_size + write_offset[r]][..width] =
    /// src[r]` for `rows` rows; a negative page, an offset outside the page
    /// or a cell past `table_rows` drops the row.
    PageWrite {
        src: String,
        table: String,
        write_page: String,
        write_offset: String,
        rows: u32,
        width: u32,
        page_size: u32,
        table_rows: u32,
    },
    /// Marks the rows whose position closes a `ratio` block:
    /// `boundary_pos` the position (or -1), `boundary_rope` the block's
    /// first position (or 0), `boundary_req` each row's request.
    Boundary {
        positions: String,
        request_of_token: String,
        row_valid: String,
        boundary_pos: String,
        boundary_req: String,
        boundary_rope: String,
        rows: u32,
        ratio: u32,
    },
    /// Each boundary row's mean of the `ratio` cached keys (`head_dim`
    /// wide, paged by `indices`/`indptr` in pages of `page_size`) ending at
    /// its position (positions before 0 add nothing; the divisor stays
    /// `ratio`); 0 for a row closing no block.
    BlockMean {
        boundary_pos: String,
        boundary_req: String,
        keys: String,
        indices: String,
        indptr: String,
        entries: String,
        rows: u32,
        head_dim: u32,
        ratio: u32,
        page_size: u32,
    },
    /// Files `entries[r][..width]` at the pool cell of boundary row `r`
    /// (request `boundary_req[r]`, position `boundary_pos[r]`); a row
    /// closing no block writes nothing.
    PoolWrite {
        entries: String,
        boundary_pos: String,
        boundary_req: String,
        keys: String,
        indices: String,
        indptr: String,
        rows: u32,
        width: u32,
        page_size: u32,
    },
    /// Per query row: scores `Σ_h max(q_h · k_j, 0) · w_h` against the
    /// cached index keys `j < (pos + 1) / ratio` (key `j` at position
    /// `(j + 1) · ratio − 1`, at most `max_pages · page_size / ratio`),
    /// then the `top_k` keys: every key when there are no more than
    /// `top_k`, else (in key order) those scoring at or above a threshold
    /// bisected 40 times between the row's min and max; unfilled slots
    /// are -1.
    IndexTopk {
        q: String,
        weights: Option<String>,
        keys: String,
        indices: String,
        indptr: String,
        positions: String,
        request_of_token: String,
        selection: String,
        rows: u32,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        ratio: u32,
        page_size: u32,
        max_pages: u32,
    },
    /// `routes[r][g] = g` for `rows` rows of `groups`.
    GroupRoutes {
        routes: String,
        rows: u32,
        groups: u32,
    },
    /// `y[r] = table[ids[r][0]] ‖ … ‖ table[ids[r][heads - 1]]` for `rows`
    /// rows, each slice `width` wide; an id outside `[0, limit)` lands
    /// zeros.
    EmbedConcat {
        ids: String,
        table: String,
        y: String,
        rows: u32,
        heads: u32,
        width: u32,
        limit: u32,
    },
    /// Each of `rows` token rows' cell, `width` f32 columns: a row of one of
    /// the `clips` boxes of `tgrid` (i32 `t, h, w, off`) its `(t, h, w)` in
    /// the box, row-major; any other row `positions[row]` (i32) in column
    /// `axis`; zeros elsewhere. `base`: the row `tgrid` addresses `y`'s row
    /// 0 at.
    Cells {
        tgrid: String,
        positions: String,
        y: String,
        clips: u32,
        rows: u32,
        axis: u32,
        width: u32,
        base: u32,
    },
    /// `y[r] = csr[g + 1] - csr[g]` (f32) for `g = groups[lanes[r]]`, 0 when
    /// the lane or group is out of the tables (`lane_count`, `csr_len`).
    /// `y[r][at..at + width] = x[r]` for `rows` rows, `y` rows `stride`
    /// wide: one input of a side-by-side concatenation.
    ConcatCols {
        x: String,
        y: String,
        rows: u32,
        width: u32,
        stride: u32,
        at: u32,
    },
    GroupRows {
        csr: String,
        groups: String,
        lanes: String,
        y: String,
        csr_len: u32,
        lane_count: u32,
        rows: u32,
    },
    /// A patch move over clip tables (`clips` boxes in each): `fold` the
    /// pixel unshuffle by `p` (`x`'s `channels` voxels of `grid` into `y`'s
    /// rows of `o_grid`, `channels · |p|` wide), else the pixel shuffle
    /// (`x`'s `channels · |p|`-wide rows of `grid` into `y`'s `channels`
    /// voxels of `o_grid`); `y` has `rows` rows, a row no box holds zero.
    /// `bases`: the rows `grid` and `o_grid` address `x`'s and `y`'s row 0 at.
    Patch {
        x: String,
        grid: String,
        y: String,
        o_grid: String,
        clips: u32,
        rows: u32,
        channels: u32,
        p: [u32; 3],
        fold: bool,
        bases: [u32; 2],
    },
}

impl HostOp {
    /// The buffers the op names.
    pub fn names(&self) -> Vec<&str> {
        match self {
            HostOp::Embed { ids, table, y, .. } => vec![ids, table, y],
            HostOp::SplitRows { x, left, right, .. } => vec![x, left, right],
            HostOp::SplitQGate {
                packed, q, gate, ..
            } => vec![packed, q, gate],
            HostOp::Matmul { act, w, y, .. } => vec![act, w, y],
            HostOp::Argmax { x, y, .. } => vec![x, y],
            HostOp::ArgmaxMerge { parts, y, .. } => vec![parts, y],
            HostOp::SplitQkv {
                packed, q, k, v, ..
            } => vec![packed, q, k, v],
            HostOp::EmbedWeighted {
                ids,
                weights,
                table,
                y,
                ..
            } => vec![ids, weights, table, y],
            HostOp::TopkSoftmax {
                logits,
                routes,
                weights,
                ..
            } => vec![logits, routes, weights],
            HostOp::MatmulGrouped {
                x, w, routes, y, ..
            } => vec![x, w, routes, y],
            HostOp::GroupRoutes { routes, .. } => vec![routes],
            HostOp::PageWrite {
                src,
                table,
                write_page,
                write_offset,
                ..
            } => vec![src, table, write_page, write_offset],
            HostOp::Boundary {
                positions,
                request_of_token,
                row_valid,
                boundary_pos,
                boundary_req,
                boundary_rope,
                ..
            } => vec![
                positions,
                request_of_token,
                row_valid,
                boundary_pos,
                boundary_req,
                boundary_rope,
            ],
            HostOp::BlockMean {
                boundary_pos,
                boundary_req,
                keys,
                indices,
                indptr,
                entries,
                ..
            } => vec![boundary_pos, boundary_req, keys, indices, indptr, entries],
            HostOp::PoolWrite {
                entries,
                boundary_pos,
                boundary_req,
                keys,
                indices,
                indptr,
                ..
            } => vec![entries, boundary_pos, boundary_req, keys, indices, indptr],
            HostOp::IndexTopk {
                q,
                weights,
                keys,
                indices,
                indptr,
                positions,
                request_of_token,
                selection,
                ..
            } => {
                let mut names = vec![q];
                names.extend(weights.iter());
                names.extend([
                    keys,
                    indices,
                    indptr,
                    positions,
                    request_of_token,
                    selection,
                ]);
                names.into_iter().map(String::as_str).collect()
            }
            HostOp::PleNgramIds {
                ids,
                indptr,
                slots,
                state,
                out,
                ..
            } => {
                let mut names = vec![ids];
                names.extend(indptr.iter());
                names.extend([slots, state, out]);
                names.into_iter().map(String::as_str).collect()
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
                ..
            } => {
                let mut names = vec![cand, indptr, unary];
                names.extend(hp.iter());
                names.extend([tokens, pred, succ, picks]);
                names.into_iter().map(String::as_str).collect()
            }
            HostOp::TopK {
                x, values, indices, ..
            } => vec![x, values, indices],
            HostOp::EmbedConcat { ids, table, y, .. } => vec![ids, table, y],
            HostOp::Cells {
                tgrid,
                positions,
                y,
                ..
            } => vec![tgrid, positions, y],
            HostOp::GroupRows {
                csr,
                groups,
                lanes,
                y,
                ..
            } => vec![csr, groups, lanes, y],
            HostOp::ConcatCols { x, y, .. } => vec![x, y],
            HostOp::Patch {
                x, grid, y, o_grid, ..
            } => vec![x, grid, y, o_grid],
        }
    }
}
