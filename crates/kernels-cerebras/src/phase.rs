//! The device work an entry emits: phases of kernel calls over buffers, each
//! saying the PEs it runs on and how its buffers spread over them, and the
//! [`Sink`] the engine supplies to place operands and take the phases.
//!
//! Kernels emit device work; the engine owns where data lives and when it
//! runs: which buffer is an exported symbol or an arena slot, which phases
//! fuse into one program, whether a program is a function or a table.

use dtype::Dtype;

use crate::csl::{Arg, Call, Func};
use crate::cx::View;
use crate::error::Error;
use crate::library::Kernel;
use crate::tensor::Tensor;

/// What an entry emits into: the engine's program under construction. It
/// places operand handles on buffers, declares the buffers and private
/// arrays an entry asks for, and takes each phase when it closes.
pub trait Sink {
    /// The root buffer holding `t` now, and `t`'s rows in it.
    fn read(&mut self, t: Tensor) -> Result<View, Error>;

    /// The root buffer `t` holds from here on.
    fn write(&mut self, t: Tensor) -> Result<View, Error>;

    /// `t` as a packed bf16 buffer (two halves a word), when the engine
    /// keeps it so (a half-precision weight); `None` means read it as f32.
    fn read_half(&mut self, t: Tensor) -> Result<Option<View>, Error>;

    /// A kv pool `t` as a packed bf16 buffer under its plain name (read and
    /// written in place), when the engine keeps pools so; `None` means f32.
    fn pool_half(&mut self, t: Tensor) -> Result<Option<View>, Error>;

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

    /// A global declaration line (a DSD, a constant table), once.
    fn global(&mut self, line: String);

    /// A library unit's text under its name, once.
    fn helper(&mut self, name: &'static str, text: &'static str);

    /// Appends a closed phase; phases run in the order they are appended.
    fn phase(&mut self, phase: Phase);

    /// The engine's terms for the fabric the phases run on.
    fn fabric(&self) -> Fabric;
}

/// The engine's terms an entry plans its splits within: what a PE holds,
/// how far a phase spreads, the placement in force and how bf16 operands
/// are stored. The engine sets them, and chooses tiles and grids by them;
/// the planners still in the entries read them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fabric {
    /// The most data words one phase places on a PE altogether (its share
    /// of every buffer plus scratch). A probe (`pe_memory_probe`) links
    /// 10240 words of data beside the memcpy module alone; the rest of a
    /// PE's memory is code, stack and the task tables, and the kernels'
    /// code takes its share below that.
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
    /// Whether a depth-split matmul (and a lane plan's partials) add on
    /// the fabric rather than on the host.
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

/// What a row-wise op holds on a PE, for the engine to choose its tile:
/// `rows × width`, with `per_row` words a row that split with the columns,
/// `per_col` words a column (a gain) that split with the columns, `whole`
/// words that do not split; columns split only in multiples of `unit` (a
/// head).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    pub rows: u32,
    pub width: u32,
    pub per_row: u64,
    pub per_col: u64,
    pub whole: u64,
    pub unit: u32,
}

/// A product `y[m, n] = act[m, k] · w[n, k]ᵀ`, for the engine to choose the
/// grid of PEs it spreads over; `packed` when the weight is held two bf16
/// a word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Product {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub packed: bool,
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
    /// [`LanePlan`] on the host.
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

/// How one slot's state `[a][b][w]` splits within a lane: `a` into
/// `a_groups` and `b` into `b_groups` ranges, `w` words contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockPlan {
    pub a: u32,
    pub b: u32,
    pub w: u32,
    pub a_groups: u32,
    pub b_groups: u32,
}

impl BlockPlan {
    /// Words one PE holds of one slot.
    pub fn words(&self) -> u64 {
        u64::from(self.a / self.a_groups) * u64::from(self.b / self.b_groups) * u64::from(self.w)
    }
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

    /// The indices this segment picks for block `[a0, a1) × [b0, b1)`, below
    /// `bound`, ascending and deduplicated.
    pub fn picks(&self, (a0, a1): (u32, u32), (b0, b1): (u32, u32), bound: u32) -> Vec<u32> {
        let mut seen = std::collections::BTreeSet::new();
        for i in a0..a1 {
            let ia = i.checked_div(self.a_group).unwrap_or(0);
            let js = if self.b_stride > 0 { b0..b1 } else { 0..1 };
            for j in js {
                for t in 0..self.span {
                    let at = self.base + ia * self.a_stride + j * self.b_stride + t;
                    if at < bound {
                        seen.insert(at);
                    }
                }
            }
        }
        seen.into_iter().collect()
    }
}

/// The indices `segments` pick together, ascending and deduplicated.
pub fn picks(segments: &[Segment], a: (u32, u32), b: (u32, u32), bound: u32) -> Vec<u32> {
    let mut all: Vec<u32> = segments.iter().flat_map(|s| s.picks(a, b, bound)).collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// The paged form of a lane's state: lane `l` holds pages
/// `indices[indptr[l]..indptr[l + 1]]` of the banks (each page `page_size`
/// rows of a bank plane, `stride` words a page). A PE holds `lanes_per_pe ×
/// max_pages` local pages; it reads `indptr`, `indices` and the `rewritten`
/// per-row page tables (kv_append's `write_page`) with local page numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagePlan {
    pub source: PageSource,
    pub page_size: u32,
    pub max_pages: u32,
    /// Per-row page tables rewritten to local pages for the PE's rows.
    pub rewritten: Vec<String>,
    /// A lane's pages split over this many PEs (1: one PE holds them all):
    /// PE `p` holds pages `[pg0, pg0 + max_pages / page_groups)` of each of
    /// its lanes, with `pg0` (and the pages per PE) in its header words 9
    /// and 10; its row outputs are then partial and merged by `reduce`.
    pub page_groups: u32,
    /// The resident placement of a pool: every PE runs every lane, page
    /// `g` of the pool lives on PE `g % pes` as its local page `g / pes`
    /// (`strided_pages` local pages a PE, the pool's pages over the row),
    /// a lane's local page list keeping every page with the ones elsewhere
    /// as -1, so key positions stay whole and the row merges the lanes'
    /// partials; the pool then stays where it is between fires.
    pub strided: bool,
    pub strided_pages: u32,
}

impl PagePlan {
    /// Pages one PE holds of one lane.
    pub fn pages_per_pe(&self) -> u32 {
        self.max_pages.max(1).div_ceil(self.page_groups.max(1))
    }
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

/// A phase spread over PEs by lanes (and blocks of each lane's state): PE
/// `p` runs lanes `[l0, l1)` (rows `[r0, r1)`) over block `[a0, a1) × [b0,
/// b1)`, read from its row of `header` (a `[pes, 16]` i32 buffer: `l0, l1,
/// held slots, a0, a1, b0, b1, r0, r1, 0…`), holding only its lanes' slot
/// blocks of `banks`, with `slots` rewritten to those local rows. The host
/// cuts and pastes all of it around the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanePlan {
    pub pes: u32,
    pub kind: LaneKind,
    pub header: String,
    /// The slot table (i32, one entry per row): lane `l`'s slot is entry
    /// `l` (per row) or entry `indptr[l]` (ragged).
    pub slots: String,
    /// Banks `(name, stride, split)`: `[slots_total, stride]` f32 planes; a
    /// PE holds `lanes_per_pe` local slot rows, each the whole `stride`, its
    /// block of `[a][b][w]`, or rows `[b0, b1)` of `[b][stride / b]`.
    pub banks: Vec<(String, u32, BankSplit)>,
    /// How each slot's state splits within a lane, when it does; `stride`
    /// then equals `a × b × w`.
    pub block: Option<BlockPlan>,
    /// The most lanes one PE runs; the local bank rows it holds.
    pub lanes_per_pe: u32,
    /// The PE order: page groups vary fastest, then the b axis's groups,
    /// then the a axis's, then the lane groups; `a_first` swaps the two
    /// block axes, so a merge over page groups and the a axis (attention's
    /// key rows) spans consecutive PEs.
    pub a_first: bool,
    /// Outputs written row by row per lane `(name, width, a_stride,
    /// b_stride)`: after the run PE `p`'s copy supplies, for the rows of
    /// lanes `[l0, l1)`, the columns `i * a_stride + j * b_stride` of its
    /// block (all columns when the plan has no block).
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
    pub pages: Option<PagePlan>,
}

impl LanePlan {
    /// Words one PE holds of one slot of `stride` (its block when `blocked`).
    pub fn slot_words(&self, stride: u32, split: BankSplit) -> u64 {
        match (self.block, split) {
            (Some(b), BankSplit::Block) => b.words(),
            (Some(b), BankSplit::ByB) => {
                u64::from(b.b / b.b_groups.max(1)) * u64::from(stride / b.b.max(1))
            }
            _ => u64::from(stride),
        }
    }

    /// Local slot (or page) rows one PE holds.
    pub fn local_rows(&self) -> u64 {
        u64::from(self.lanes_per_pe)
            * u64::from(self.pages.as_ref().map_or(1, |p| p.pages_per_pe()))
    }

    /// Bank slots (pages) one PE holds: its lanes' pages, or its share of
    /// the pool under the resident placement.
    pub fn page_slots(&self) -> u64 {
        match &self.pages {
            Some(p) if p.strided => u64::from(p.strided_pages.max(1)),
            _ => self.local_rows(),
        }
    }

    /// The words of `plane` one PE holds (the most over the PEs).
    pub fn plane_words(&self, plane: &ColPlane) -> u64 {
        let (rows, width) = (u64::from(plane.rows), u64::from(plane.width));
        let Some(b) = self.block else {
            return rows * width;
        };
        if plane.row_block {
            let ai = b.a / b.a_groups.max(1);
            let bj = b.b / b.b_groups.max(1);
            let cols = picks(&plane.segments, (0, ai), (0, bj), plane.width).len() as u64;
            return u64::from(ai) * cols;
        }
        let bound = if plane.by_rows {
            plane.rows
        } else {
            plane.width
        };
        let (ai, bj) = (b.a / b.a_groups.max(1), b.b / b.b_groups.max(1));
        let picked = (0..b.a_groups.max(1))
            .flat_map(|ag| (0..b.b_groups.max(1)).map(move |bg| (ag, bg)))
            .map(|(ag, bg)| {
                picks(
                    &plane.segments,
                    (ag * ai, (ag + 1) * ai),
                    (bg * bj, (bg + 1) * bj),
                    bound,
                )
                .len() as u64
            })
            .max()
            .unwrap_or(0);
        if plane.by_rows {
            picked * width
        } else {
            rows * picked
        }
    }
}

/// One phase: its function, the PEs it runs on, and how each buffer it
/// names is spread over them (unlisted buffers are whole on every PE); or
/// a host phase, run by the host over its copies of the buffers.
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
    /// Buffers rounded to bf16 when the steps are done: `(name, words)`.
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

/// A phase the host runs instead of the fabric: pure data movement, or a
/// vocabulary-sized op no rectangle holds. Every buffer of a fire lives on
/// the host between phases, so it costs no transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPhase {
    pub op: HostOp,
    /// Buffers rounded to bf16 after the op (bf16 writes).
    pub rounds: Vec<String>,
}

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
        }
    }
}
