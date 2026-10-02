//! The program a fire is traced into: one PE rectangle's CSL source plus the
//! manifest the host needs to feed and read it.
//!
//! Every tensor a kernel touches becomes a device-resident f32 array with an
//! exported symbol. Every kernel entry appends one phase function; the
//! program's single RPC entry, `run`, calls the phases in order and then
//! releases the host command stream. Placement is a single PE today; the
//! [`Placement`] enum is where sharding across a rectangle will land.

use std::collections::BTreeMap;
use std::fmt::Write;

use dtype::Dtype;

use crate::csl::{Arg, Call, Func};
use crate::error::{Error, refuse};
use crate::tensor::Tensor;

mod table;

/// Where a buffer lives on the fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// The whole array on PE (0, 0).
    Single,
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

/// Whether fused programs render table-driven (`PIE_CEREBRAS_TABLE=1`):
/// one interpreter over the kernels the program uses, the buffers in an
/// arena at fixed offsets, the phases a table of kernel calls and
/// collectives the host uploads as data.
pub fn table_mode() -> bool {
    std::env::var("PIE_CEREBRAS_TABLE").is_ok_and(|v| v != "0")
}

/// Whether a fire renders as one table-driven program over every phase
/// (`PIE_CEREBRAS_WHOLE_FIRE=1`, with `table_mode` and the resident
/// placement): the buffers keep one layout across phases, the arena is
/// allotted by liveness, and the host moves only inputs and outputs.
pub fn whole_fire() -> bool {
    table_mode() && std::env::var("PIE_CEREBRAS_WHOLE_FIRE").is_ok_and(|v| v != "0")
}

/// Whether kv pools stay on the device between fires
/// (`PIE_CEREBRAS_KEEP_POOLS=1`, with a whole-fire program of one unified
/// shape on a resident server): kept like the weights, their page moves
/// becoming table rows the host fills per fire.
pub fn keep_pools() -> bool {
    whole_fire() && std::env::var("PIE_CEREBRAS_KEEP_POOLS").is_ok_and(|v| v != "0")
}

/// Rows a kept-pool program reserves at the table's start for the fire's
/// page moves (three rows a move between PEs), and the words of the
/// scratch page they travel through.
pub const MOVE_ROWS: u32 = 48;
pub const MOVE_SCRATCH: u64 = 1024;

/// Words one arena array holds (a word address of 16 bits): a larger
/// arena is several arrays, an offset naming its array by `offset /
/// ARENA_CHUNK`.
pub const ARENA_CHUNK: u64 = 8192;

/// Words an op row holds beside its length word: the opcode and its
/// operands.
#[must_use]
pub fn op_words(op: &TableOp) -> u32 {
    match op {
        TableOp::Call { args, on, root, .. } => args.len() as u32 + 1 + u32::from(on.is_some()) * 2 + u32::from(*root && on.is_none()),
        TableOp::Collective(_) => 4,
        TableOp::Nop => MOVE_ROW_WORDS,
    }
}

/// Words a page-move row holds (`OP_ON pe k_copy dst 0 src 0 words`).
pub const MOVE_ROW_WORDS: u32 = 8;

/// The table-driven form of a program: the kernels it dispatches (opcode =
/// index), the op rows (each its length word, then the opcode and the
/// operands; a zero length ends the table), and the arena every buffer
/// lives in.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub kernels: Vec<String>,
    /// Words the op table holds (the rows, padded with zeros to a unified
    /// capacity).
    pub ops_words: u32,
    pub ops: Vec<TableOp>,
    /// Arena slots `(name, offset, words)` per PE: the exports (as laid
    /// out for the PE) and the scratch arrays (zeros).
    pub arena: Vec<(String, u64, u64)>,
    pub arena_words: u64,
    /// The first `keep_chunks` arena arrays hold the kept exports alone:
    /// a resident server uploads them once and never reads them back.
    pub keep_chunks: u32,
    /// Rows the table holds (`ops.len()` padded with no-ops to a unified
    /// capacity).
    pub rows: u32,
    /// The first `move_rows` rows are no-ops the host fills with the
    /// fire's page moves (kept pools).
    pub move_rows: u32,
    /// Constant tables `(name, words)` the kernels index (rope frequencies,
    /// n-gram primes): kept arena slots the host fills once, in place of
    /// the CSL globals the op table cannot point at.
    pub consts: Vec<(String, Vec<u32>)>,
    /// Phases the table runs (a whole fire's row: its share of the fire).
    pub phases: u32,
}

/// A CSL global array literal `var NAME = [N]T { a, b, … };` of `f32`,
/// `i32`, `u32` or `u64`, as the words it holds (a `u64` low word first,
/// as the PE lays it out).
#[must_use]
pub fn parse_global_table(line: &str) -> Option<(String, Vec<u32>)> {
    let rest = line.trim().strip_prefix("var ")?;
    let (name, rest) = rest.split_once(" = [")?;
    let (count, rest) = rest.split_once(']')?;
    let count: usize = count.trim().parse().ok()?;
    let (elem, rest) = rest.split_once('{')?;
    let elem = elem.trim();
    let body = rest.trim().trim_end_matches(';').trim().strip_suffix('}')?;
    let lits: Vec<&str> = body.split(',').map(str::trim).filter(|l| !l.is_empty()).collect();
    if lits.len() != count {
        return None;
    }
    let mut words = Vec::with_capacity(count * 2);
    for l in lits {
        match elem {
            "f32" => words.push(l.parse::<f32>().ok()?.to_bits()),
            "i32" => words.push(l.parse::<i32>().ok()? as u32),
            "u32" => words.push(l.parse::<u32>().ok()?),
            "u64" => {
                let v = l.parse::<u64>().ok()?;
                words.push(v as u32);
                words.push((v >> 32) as u32);
            }
            _ => return None,
        }
    }
    Some((name.trim().to_string(), words))
}

impl Table {
    /// Arena arrays the PE declares: `arena_words` in chunks of
    /// [`ARENA_CHUNK`] (one array a chunk).
    #[must_use]
    pub fn chunks(&self) -> u64 {
        self.arena_words.div_ceil(ARENA_CHUNK).max(1)
    }

    /// Words arena array `c` holds: the chunk's prefix the arena's slots
    /// reach (a kept chunk padded to the boundary declares its kept words
    /// alone, so a model's few kept weights do not cost a whole chunk).
    #[must_use]
    pub fn chunk_words(&self, c: u64) -> u64 {
        let lo = c * ARENA_CHUNK;
        let hi = (lo + ARENA_CHUNK).min(self.arena_words);
        let reach = self
            .arena
            .iter()
            .filter(|(_, off, words)| *off < hi && off + words > lo)
            .map(|(_, off, words)| (off + words).min(hi))
            .max()
            .unwrap_or(lo);
        // A chunk below the arena's end without a slot still carries one
        // word (the arena stays addressable chunk by chunk).
        if c + 1 < self.chunks() && reach == lo {
            return 1;
        }
        reach.saturating_sub(lo).max(1)
    }

    /// Words the PE's arena arrays take together.
    #[must_use]
    pub fn data_words(&self) -> u64 {
        (0..self.chunks()).map(|c| self.chunk_words(c)).sum::<u64>() + u64::from(self.ops_words)
    }
}

/// What every class of a model agrees on so their whole-fire programs are
/// one program text (one compiled binary, one resident server whose kept
/// arena outlives the fire classes): the partition of the fire into rows
/// of PEs, the kept exports' slots (one layout over the whole rectangle,
/// each slot an exported symbol the host uploads once), and per row the
/// kernels dispatched, the arena's size, the table's capacity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Unified {
    /// Phases a row, when the classes' fires split alike (`None` leaves
    /// each fire to its own fused groups: a single row, or a partition
    /// still to be chosen).
    pub row_phases: Option<Vec<usize>>,
    /// Kept exports `(name, offset, words)`, in the arena's first arrays,
    /// the same on every row.
    pub keep: Vec<(String, u64, u64)>,
    pub keep_chunks: u32,
    /// Row by row (one for a single-row program); empty while only the
    /// partition is fixed.
    pub rows: Vec<RowShape>,
}

/// One row's shape under a unified whole fire.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RowShape {
    pub kernels: Vec<String>,
    pub arena_words: u64,
    pub rows: u32,
    pub ops_words: u32,
}

/// Slack a unified shape adds over the probes' largest work region and op
/// table, so a fire a little off the probes still renders under the shape
/// (a fire the shape refuses falls back to its own programs and loses
/// the kept pools).
pub const WORK_SLACK_WORDS: u64 = 256;
pub const OPS_SLACK_WORDS: u32 = 64;

/// Lays kept slots `(name, words)` from the arena's start, each within one
/// chunk: the slots and the chunks they take.
#[must_use]
pub fn keep_layout(mut kept: Vec<(String, u64)>) -> (Vec<(String, u64, u64)>, u32) {
    kept.sort();
    kept.dedup_by(|a, b| {
        if a.0 == b.0 {
            b.1 = b.1.max(a.1);
            true
        } else {
            false
        }
    });
    let mut at = 0u64;
    let mut placed = Vec::new();
    for (name, words) in kept {
        let chunk = at / ARENA_CHUNK;
        if words <= ARENA_CHUNK && (at + words - 1) / ARENA_CHUNK != chunk {
            at = (chunk + 1) * ARENA_CHUNK;
        }
        placed.push((name, at, words));
        at += words;
    }
    (placed, at.div_ceil(ARENA_CHUNK) as u32)
}

impl RowShape {
    /// The union of several tables of one row: every kernel, the largest
    /// work region, table and row.
    pub fn union<'a>(tables: impl IntoIterator<Item = &'a Table>) -> RowShape {
        let mut kernels: Vec<String> = Vec::new();
        let (mut work, mut rows, mut ops_words) = (0u64, 0u32, 0u32);
        for t in tables {
            for k in &t.kernels {
                if !kernels.contains(k) {
                    kernels.push(k.clone());
                }
            }
            work = work.max(t.arena_words.saturating_sub(u64::from(t.keep_chunks) * ARENA_CHUNK));
            rows = rows.max(t.rows);
            ops_words = ops_words.max(t.ops_words);
        }
        if keep_pools() && !kernels.iter().any(|k| k == "k_copy") {
            kernels.push("k_copy".to_string());
        }
        RowShape {
            kernels,
            // The work region's words, with slack: the first-fit liveness
            // layout is not monotone in a fire's rows (a six-row prefill
            // once needed one word more than the eight-row probe), so a
            // fire a little off the probes still fits the shape. The kept
            // chunks are added once the shared kept layout is known.
            arena_words: work + WORK_SLACK_WORDS,
            rows,
            ops_words: ops_words + OPS_SLACK_WORDS,
        }
    }
}

impl Unified {
    /// The union of several single-row programs' tables.
    pub fn union<'a>(tables: impl IntoIterator<Item = &'a Table>) -> Unified {
        let tables: Vec<&Table> = tables.into_iter().collect();
        let mut u = Unified {
            row_phases: None,
            keep: Vec::new(),
            keep_chunks: 0,
            rows: vec![RowShape::union(tables.iter().copied())],
        };
        u.unite_keep(tables.iter().copied());
        u
    }

    /// The partition every class is to follow: that of the program with
    /// the most rows, when every program runs the same number of phases
    /// (`None` otherwise). A program is its rows' tables.
    pub fn partition(programs: &[Vec<Table>]) -> Option<Unified> {
        let phases = |p: &Vec<Table>| p.iter().map(|t| t.phases as usize).sum::<usize>();
        let first = programs.first()?;
        if programs.iter().any(|p| phases(p) != phases(first)) {
            return None;
        }
        let widest = programs.iter().max_by_key(|p| p.len())?;
        Some(Unified {
            row_phases: Some(widest.iter().map(|t| t.phases as usize).collect()),
            keep: Vec::new(),
            keep_chunks: 0,
            rows: Vec::new(),
        })
    }

    /// The union of several programs that split alike (every program the
    /// same rows of the same phase counts): a shape a row, one kept layout.
    pub fn union_rows(programs: &[Vec<Table>]) -> Option<Unified> {
        let first = programs.first()?;
        let phases: Vec<usize> = first.iter().map(|t| t.phases as usize).collect();
        if programs
            .iter()
            .any(|p| p.iter().map(|t| t.phases as usize).collect::<Vec<_>>() != phases)
        {
            return None;
        }
        let rows = (0..first.len())
            .map(|g| RowShape::union(programs.iter().map(|p| &p[g])))
            .collect();
        let mut u = Unified {
            row_phases: (first.len() > 1).then_some(phases),
            keep: Vec::new(),
            keep_chunks: 0,
            rows,
        };
        u.unite_keep(programs.iter().flatten());
        Some(u)
    }

    /// Lays every table's kept slots out once and puts the rows' work
    /// regions after them.
    fn unite_keep<'a>(&mut self, tables: impl IntoIterator<Item = &'a Table>) {
        let mut kept: Vec<(String, u64)> = Vec::new();
        for t in tables {
            for (name, off, words) in &t.arena {
                if t.keep_chunks > 0 && *off < u64::from(t.keep_chunks) * ARENA_CHUNK {
                    kept.push((name.clone(), *words));
                }
            }
        }
        let (keep, keep_chunks) = keep_layout(kept);
        self.keep = keep;
        self.keep_chunks = keep_chunks;
        for r in &mut self.rows {
            r.arena_words += u64::from(keep_chunks) * ARENA_CHUNK;
        }
    }
}

/// One row of the op table.
#[derive(Debug, Clone, PartialEq)]
pub enum TableOp {
    /// Kernel `kernel` with `args` (pointers as arena offsets), on every PE,
    /// the row's root alone (`root`), or one PE alone (`on`).
    Call { root: bool, on: Option<u32>, kernel: u32, args: Vec<Arg> },
    /// A collective along the row; the interpreter resumes after it lands.
    Collective(FabricOp),
    /// A row the host may fill (a page move), else skipped.
    Nop,
}

/// Opcodes below the kernels: the collectives and the root-only call.
pub const OP_REDUCE: i32 = -1;
pub const OP_GATHER: i32 = -2;
pub const OP_BROADCAST: i32 = -3;
pub const OP_ROOT: i32 = -4;
pub const OP_NOP: i32 = -5;
pub const OP_ON: i32 = -6;
pub const OP_BROADCAST_FROM: i32 = -7;
pub const OP_HANDOFF: i32 = -8;

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
    pub fn all(kernel: &str, args: Vec<Arg>) -> Self {
        Guarded {
            root: false,
            call: Call {
                kernel: kernel.to_string(),
                args,
            },
        }
    }

    pub fn root(kernel: &str, args: Vec<Arg>) -> Self {
        Guarded {
            root: true,
            call: Call {
                kernel: kernel.to_string(),
                args,
            },
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
    Reduce { send: String, recv: String, count: u64 },
    /// `recv[p · count ..]` = PE `p`'s `send` on the root.
    Gather { send: String, recv: String, count: u64 },
    /// Every PE's `buf` = the root's.
    Broadcast { buf: String, count: u64 },
    /// Every PE's `buf` = PE `root`'s (a page moving between PEs).
    BroadcastFrom { root: u32, buf: String, count: u64 },
    /// Down the column: every row's `buf` = row `root`'s (a buffer one
    /// row of a whole fire produced and later rows read).
    Handoff { root: u32, buf: String, count: u64 },
}

impl FabricReduce {
    /// The CSL of a fused group's step machine: the phases' functions run
    /// in order, each phase's steps after its function, every step handing
    /// the next to its collective's callback; the last state rounds and
    /// finishes the last reduce and ends the run.
    fn render_chain(chain: &[(&str, Option<&FabricReduce>)], pe: &mut String) {
        // States: statements, then one collective; the trailing statements
        // (functions after the last collective, roundings, finishes) end
        // the run.
        let mut states: Vec<(Vec<String>, &FabricOp)> = Vec::new();
        let mut pending: Vec<String> = Vec::new();
        for (func, reduce) in chain {
            pending.push(format!("{func}();"));
            if let Some(r) = reduce {
                for step in &r.steps {
                    let mut lines = std::mem::take(&mut pending);
                    lines.extend(step.calls.iter().map(Guarded::render));
                    states.push((lines, &step.op));
                }
                // The finish lays the merged words out, then they round.
                pending.extend(r.finish.iter().map(Guarded::render));
                for (name, n) in &r.round {
                    pending.push(format!("k_round_bf16(@ptrcast([*]f32, &{name}), {n});"));
                }
            }
        }
        pe.push_str("var px: i32 = 0;\nvar st: u16 = 0;\nfn step_fn() void {\n  switch (st) {\n");
        for (i, (lines, op)) in states.iter().enumerate() {
            let _ = writeln!(pe, "    {i} => {{");
            for l in lines {
                let _ = writeln!(pe, "      {l}");
            }
            let _ = writeln!(pe, "      st = {};", i + 1);
            match op {
                FabricOp::Reduce { send, recv, count } => {
                    let _ = writeln!(
                        pe,
                        "      mpi_x.reduce_fadds(0, @ptrcast([*]f32, &{send}), @ptrcast([*]f32, &{recv}), {count}, step_id);"
                    );
                }
                FabricOp::Gather { send, recv, count } => {
                    let _ = writeln!(
                        pe,
                        "      mpi_x.gather(0, @ptrcast([*]u32, &{send}), @ptrcast([*]u32, &{recv}), {count}, step_id);"
                    );
                }
                FabricOp::Broadcast { buf, count } => {
                    let _ = writeln!(
                        pe,
                        "      mpi_x.broadcast(0, @ptrcast([*]u32, &{buf}), {count}, step_id);"
                    );
                }
                FabricOp::BroadcastFrom { root, buf, count } => {
                    let _ = writeln!(
                        pe,
                        "      mpi_x.broadcast({root}, @ptrcast([*]u32, &{buf}), {count}, step_id);"
                    );
                }
                // A handoff belongs to a whole fire's rows, never to a
                // function program's step machine.
                FabricOp::Handoff { .. } => {}
            }
            pe.push_str("    },\n");
        }
        let _ = writeln!(pe, "    {} => {{", states.len());
        for l in &pending {
            let _ = writeln!(pe, "      {l}");
        }
        pe.push_str("      sys_mod.unblock_cmd_stream();\n    },\n    else => {},\n  }\n}\n");
        pe.push_str("task step_task() void { step_fn(); }\n");
        pe.push_str("fn run() void {\n  mpi_x.init();\n  px = @as(i32, mpi_x.pe_id);\n  st = 0;\n  step_fn();\n}\n");
    }

    /// Every name the steps and the finish mention (for export selection).
    fn mentions(&self) -> String {
        let mut text = String::new();
        for step in &self.steps {
            for g in &step.calls {
                text.push_str(&g.render());
                text.push('\n');
            }
            match &step.op {
                FabricOp::Reduce { send, recv, .. } | FabricOp::Gather { send, recv, .. } => {
                    let _ = writeln!(text, "{send} {recv}");
                }
                FabricOp::Broadcast { buf, .. } | FabricOp::BroadcastFrom { buf, .. } | FabricOp::Handoff { buf, .. } => {
                    let _ = writeln!(text, "{buf}");
                }
            }
        }
        for g in &self.finish {
            text.push_str(&g.render());
            text.push('\n');
        }
        for (name, _) in &self.round {
            text.push_str(name);
            text.push('\n');
        }
        text
    }
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

    /// Names what the next phase lowers; [`take_scope`] hands it out.
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

    /// A private array under a name the caller chose (unique among its
    /// scratch); `elem` is `f32`, `i32` or `u32`.
    pub fn scratch_named(&mut self, name: &str, len: u64, elem: &'static str) {
        self.scratch.push((name.to_string(), len, elem));
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

    /// Appends a phase on one PE; phases run in the order they were added.
    pub fn phase(&mut self, f: Func) {
        self.phases.push(Phase {
            func: f,
            pes: 1,
            shards: Vec::new(),
            lane: None,
            host: None,
            reduce: None,
        });
    }

    /// Appends a phase over `pes` PEs with its buffers spread as `shards`,
    /// laid out by `lane` when the phase splits by lanes, its partials
    /// summed on the fabric by `reduce`.
    pub fn phase_over(
        &mut self,
        f: Func,
        pes: u32,
        shards: Vec<(String, Shard)>,
        lane: Option<LanePlan>,
        reduce: Option<FabricReduce>,
    ) {
        self.phases.push(Phase {
            func: f,
            pes,
            shards,
            lane,
            host: None,
            reduce,
        });
    }

    /// Records a phase the host runs.
    pub fn host_phase(&mut self, f: Func, host: HostPhase) {
        self.phases.push(Phase {
            func: f,
            pes: 1,
            shards: Vec::new(),
            lane: None,
            host: Some(host),
            reduce: None,
        });
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

    /// The helpers `body` needs, in declaration order: those it names, and
    /// those the chosen helpers name in turn (a kernel's own helpers).
    fn helpers_for(&self, body: &str) -> Vec<&str> {
        let wanted = |text: &str, name: &str, h: &str| {
            mentions(text, name)
                || h.lines().any(|l| l.starts_with("fn ") && mentions(text, fn_name(l)))
        };
        let all: Vec<(&str, &str)> = self.helpers.iter().map(|(n, h)| (n.as_str(), h.as_str())).collect();
        let mut chosen: Vec<bool> = all.iter().map(|(n, h)| wanted(body, n, h)).collect();
        loop {
            let mut grew = false;
            for i in 0..all.len() {
                if chosen[i] {
                    continue;
                }
                let (n, h) = all[i];
                let by_chosen = all
                    .iter()
                    .zip(&chosen)
                    .any(|((_, src), c)| *c && wanted(src, n, h));
                if by_chosen {
                    chosen[i] = true;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        all.iter()
            .zip(&chosen)
            .filter(|(_, c)| **c)
            .map(|((_, h), _)| *h)
            .collect()
    }

    /// The phases in run order, adjacent fabric phases fused into one
    /// program when they run on the same PEs, hold every shared buffer
    /// the same way, carry no lane plan, and fit a PE together: a fused
    /// group runs its functions in sequence, so a buffer one writes and
    /// the next reads stays on the PE instead of crossing the host twice
    /// (`PIE_CEREBRAS_FUSE=0` keeps every phase apart).
    fn fused_phases(&self) -> Vec<Vec<&Phase>> {
        self.fused_phases_with(0)
    }

    /// The fused groups when every group also carries `extra` data words
    /// (a whole fire's kept slots, the same on every row).
    pub(crate) fn fused_phases_with(&self, extra: u64) -> Vec<Vec<&Phase>> {
        let fuse = std::env::var("PIE_CEREBRAS_FUSE")
            .ok()
            .is_none_or(|v| v != "0");
        let exports = self.exports();
        let mut groups: Vec<Vec<&Phase>> = Vec::new();
        let words_of = |group: &[&Phase]| -> u64 {
            let mut body = String::new();
            for p in group {
                p.func.render(&mut body);
            }
            let shard_of = |name: &str| -> Shard {
                group
                    .iter()
                    .flat_map(|p| p.shards.iter())
                    .find(|(n, _)| n == name)
                    .map_or(Shard::Whole, |(_, s)| *s)
            };
            let buffers: u64 = exports
                .iter()
                .filter(|e| mentions(&body, &e.name))
                .map(|e| {
                    Export {
                        shard: shard_of(&e.name),
                        ..e.clone()
                    }
                    .local()
                })
                .sum();
            let scratch: u64 = self
                .scratch
                .iter()
                .filter(|(name, _, _)| mentions(&body, name))
                .map(|(_, len, _)| *len)
                .sum();
            buffers + scratch
        };
        // A view is its own array on the PE, cut from its root before the
        // program runs: a phase naming a view cannot share a program with
        // one naming the view's root or another view of it, or one would
        // read what the other wrote through a stale copy.
        let text_of = |p: &Phase| {
            let mut t = String::new();
            p.func.render(&mut t);
            t
        };
        let roots_of = |text: &str| -> Vec<(String, String)> {
            self.views
                .iter()
                .filter(|v| mentions(text, &v.name))
                .map(|v| (v.name.clone(), v.root.clone()))
                .collect()
        };
        let views_clash = |a: &str, b: &str| {
            let clash = |x: &str, y: &str| {
                roots_of(x).iter().any(|(view, root)| {
                    mentions(y, root)
                        || self
                            .views
                            .iter()
                            .any(|w| w.root == *root && w.name != *view && mentions(y, &w.name))
                })
            };
            clash(a, b) || clash(b, a)
        };
        // The code a fused group links beside its data: the kernels it
        // names, the prelude, its own functions and the export plumbing,
        // at `CODE_WORDS_PER_LINE` (the data budget alone assumes one
        // phase's worth of code).
        let code_of = |group: &[&Phase]| -> u64 {
            let mut body = String::new();
            for p in group {
                p.func.render(&mut body);
            }
            let lines = |t: &str| t.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//")).count() as u64;
            let helpers: u64 = self.helpers_for(&body).iter().map(|h| lines(h)).sum();
            let used = exports.iter().filter(|e| mentions(&body, &e.name)).count() as u64;
            // The step machine: about eight lines a collective, and the
            // collectives module's own code and buffers once.
            let steps: u64 = group
                .iter()
                .filter_map(|p| p.reduce.as_ref())
                .map(|r| r.steps.len() as u64 + r.finish.len() as u64 + r.round.len() as u64)
                .sum();
            // A table-driven group also holds its op table: a row (about
            // twelve words) per call and per step.
            let calls: u64 = group.iter().map(|p| p.func.body.calls().count() as u64).sum();
            let source = lines(crate::library::prelude())
                + helpers
                + lines(&body)
                + 2 * used
                + 2 * group.len() as u64
                + 8 * steps
                + 30;
            if table_mode() {
                // A table program's operands are runtime values and its
                // dispatch unpacks them: measured over linked programs,
                // about `TABLE_CODE_WORDS_PER_LINE` a line plus a fixed
                // part (collectives included), the op table beside.
                return TABLE_CODE_WORDS_PER_LINE * source + TABLE_FIXED_WORDS + 12 * (calls + steps);
            }
            let module = if steps > 0 { COLLECTIVES_WORDS } else { 0 };
            CODE_WORDS_PER_LINE * source + module
        };
        // Reducing phases bring their own rectangle (one group of partials
        // a row); every reduce of a program must agree on it, and PE `p`
        // sits at the same place under any rectangle of the same size.
        let rects_agree = |g: &[&Phase], p: &Phase| {
            let rects: Vec<(u32, u32)> = g
                .iter()
                .chain(std::iter::once(&p))
                .filter_map(|q| q.reduce.as_ref().map(|r| r.rect))
                .collect();
            rects.windows(2).all(|w| w[0] == w[1])
        };
        // Lane phases bring their own plans: a buffer one plan lays out
        // must be no other plan's.
        let owned = |l: &LanePlan| -> Vec<String> {
            let mut names: Vec<String> = vec![l.header.clone(), l.slots.clone()];
            names.extend(l.banks.iter().map(|(n, ..)| n.clone()));
            names.extend(l.row_outputs.iter().map(|(n, ..)| n.clone()));
            names.extend(l.cols.iter().map(|c| c.name.clone()));
            if let Some(p) = &l.pages {
                names.extend(p.rewritten.iter().cloned());
                match &p.source {
                    PageSource::Csr { indptr, indices } => {
                        names.push(indptr.clone());
                        names.push(indices.clone());
                    }
                    PageSource::Rows { table } => names.push(table.clone()),
                }
            }
            names.retain(|n| !n.is_empty());
            names
        };
        // Two strided plans (the resident placement) lay every shared
        // buffer out the same way (pages by `page % pes`, lane lists
        // whole), so they may share.
        let strided = |l: &LanePlan| l.pages.as_ref().is_some_and(|p| p.strided);
        let lanes_apart = |g: &[&Phase], p: &Phase| {
            let Some(mine) = &p.lane else {
                return true;
            };
            let names = owned(mine);
            g.iter().filter_map(|q| q.lane.as_ref()).all(|other| {
                (strided(mine) && strided(other) && mine.pes == other.pes)
                    || owned(other).iter().all(|n| !names.contains(n))
            })
        };
        for phase in &self.phases {
            let text = text_of(phase);
            let joins = fuse
                && phase.host.is_none()
                && groups.last().is_some_and(|g| {
                    let head = g[0];
                    let group_text: String = g.iter().map(|p| text_of(p)).collect();
                    head.host.is_none()
                        && head.pes == phase.pes
                        && !views_clash(&text, &group_text)
                        && rects_agree(g, phase)
                        && lanes_apart(g, phase)
                        && phase.shards.iter().all(|(n, s)| {
                            g.iter()
                                .flat_map(|p| p.shards.iter())
                                .all(|(m, t)| m != n || t == s)
                        })
                });
            if joins {
                let last = groups.last_mut().expect("a group");
                last.push(phase);
                let (words, code) = (words_of(last), code_of(last));
                if std::env::var_os("PIE_CEREBRAS_TRACE_FUSION").is_some() {
                    eprintln!(
                        "fusion: {} phases, {words} data words, {code} code words (budget {FUSED_PE_WORDS})",
                        last.len()
                    );
                }
                if words + code + extra > FUSED_PE_WORDS {
                    last.pop();
                    groups.push(vec![phase]);
                }
            } else {
                groups.push(vec![phase]);
            }
        }
        groups
    }

    /// Renders every phase as its own single-PE memcpy program: each one
    /// exports only the buffers its statements name, all as in-out (the host
    /// carries every buffer between phases). Globals, helpers and scratch
    /// arrays follow the phases that name them.
    pub fn render_phases(&self) -> Vec<Rendered> {
        self.render_phases_whole().0
    }

    /// `render_phases`, and whether the fire rendered as one whole-fire
    /// program (a fire that did not falls back to its fused groups; under
    /// kept pools that fallback would run on a pool of its own, so the
    /// caller refuses the fire instead).
    pub fn render_phases_whole(&self) -> (Vec<Rendered>, bool) {
        let exports = self.exports();
        if whole_fire()
            && let Some(rendered) = self.render_whole(&exports)
        {
            return (vec![rendered], true);
        }
        (self.render_groups(&exports), false)
    }

    fn render_groups(&self, exports: &[Export]) -> Vec<Rendered> {
        let exports = exports.to_vec();
        self.fused_phases()
            .into_iter()
            .map(|group| {
                let phase = group[0];
                let f = &phase.func;
                if let Some(host) = &phase.host {
                    let names = host.op.names();
                    let used: Vec<Export> = exports
                        .iter()
                        .filter(|e| names.contains(&e.name.as_str()))
                        .map(|e| Export {
                            role: Symbol::InOut,
                            ..e.clone()
                        })
                        .collect();
                    let views = self.views.iter().filter(|v| used.iter().any(|e| e.name == v.name)).cloned().collect();
                    return Rendered {
                        layout: String::new(),
                        pe: String::new(),
                        manifest: Manifest {
                            exports: used,
                            entry: "host".into(),
                            rect: (0, 0),
                            views,
                            lanes: Vec::new(),
                            host: Some(host.clone()),
                            table: None,
                            tables: Vec::new(),
                        },
                    };
                }
                // A fused group: every phase's function, run in order.
                let mut body = String::new();
                for p in &group {
                    p.func.render(&mut body);
                }
                // The reduces' epilogues name their partials and outputs:
                // they count as mentioned for export, scratch and helper
                // selection.
                let reduces: Vec<&FabricReduce> = group.iter().filter_map(|p| p.reduce.as_ref()).collect();
                let reduce = reduces.first().copied();
                for r in &reduces {
                    for l in r.mentions().lines() {
                        let _ = writeln!(body, "// {l}");
                    }
                }
                let shard_of = |name: &str| -> Shard {
                    group
                        .iter()
                        .flat_map(|p| p.shards.iter())
                        .find(|(n, _)| n == name)
                        .map_or(Shard::Whole, |(_, s)| *s)
                };
                // Every buffer travels both ways (the host holds the state)
                // but a packed one: a packed input (a weight) is never read
                // back, a packed in-out (a pool) comes back to be unpacked.
                let used: Vec<Export> = exports
                    .iter()
                    .filter(|e| mentions(&body, &e.name))
                    .map(|e| Export {
                        shard: shard_of(&e.name),
                        role: if e.packed { e.role } else { Symbol::InOut },
                        ..e.clone()
                    })
                    .collect();
                if table_mode()
                    && let Some(rendered) = self.render_table(&group, &used, &reduces, false)
                {
                    if let Some(t) = &rendered.manifest.table {
                        let lines = rendered.pe.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//")).count();
                        trace_fusion(&format!(
                            "table of {} phases: arena {} words, ops {} words, {lines} lines",
                            group.len(),
                            t.arena_words,
                            u64::from(t.ops_words)
                        ));
                        if std::env::var_os("PIE_CEREBRAS_TRACE_ARENA").is_some() {
                            for (n, o, w) in &t.arena {
                                eprintln!("fusion:   arena {o:>6} +{w:<6} {n}");
                            }
                        }
                    }
                    return rendered;
                }
                let mut pe = String::new();
                pe.push_str("param memcpy_params;\n");
                // The PE's index in the rectangle (`p` of the shards).
                pe.push_str("param pe_id: u16;\n");
                if reduce.is_some() {
                    pe.push_str("param c2d_params;\n");
                }
                pe.push_str("const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n");
                pe.push_str("const math = @import_module(\"<math>\");\n");
                if reduce.is_some() {
                    pe.push_str(
                        "const mpi_x = @import_module(\"<collectives_2d/pe>\", .{ .dim_params = c2d_params.x, .queues = [2]u16{2, 4}, .dest_dsr_ids = [1]u16{1}, .src0_dsr_ids = [1]u16{1}, .src1_dsr_ids = [1]u16{1} });\n",
                    );
                    pe.push_str("const step_id: local_task_id = @get_local_task_id(15);\n");
                }
                pe.push('\n');
                pe.push_str(crate::library::prelude());
                pe.push('\n');
                for e in &used {
                    let _ = writeln!(pe, "var {}: [{}]{};", e.name, e.local(), e.elem);
                    let _ = writeln!(pe, "var {0}_ptr: [*]{1} = &{0};", e.name, e.elem);
                }
                for (name, len, elem) in &self.scratch {
                    if mentions(&body, name) {
                        let _ = writeln!(pe, "var {name} = @zeros([{len}]{elem});");
                    }
                }
                pe.push('\n');
                for g in &self.globals {
                    if global_name(g).is_some_and(|n| mentions(&body, n)) {
                        pe.push_str(g);
                        pe.push('\n');
                    }
                }
                pe.push('\n');
                for h in self.helpers_for(&body) {
                    pe.push_str(h);
                    pe.push('\n');
                }
                pe.push_str(&body);
                pe.push('\n');
                match reduce {
                    // The functions run with their collectives between them
                    // (a step machine), the last callback ending the run.
                    Some(_) => {
                        let chain: Vec<(&str, Option<&FabricReduce>)> = group
                            .iter()
                            .map(|p| (p.func.name.as_str(), p.reduce.as_ref()))
                            .collect();
                        FabricReduce::render_chain(&chain, &mut pe);
                    }
                    None => {
                        pe.push_str("fn run() void {\n");
                        for p in &group {
                            let _ = writeln!(pe, "  {}();", p.func.name);
                        }
                        pe.push_str("  sys_mod.unblock_cmd_stream();\n}\n");
                    }
                }
                let _ = f;
                pe.push_str("comptime {\n");
                if reduce.is_some() {
                    pe.push_str("  @bind_local_task(step_task, step_id);\n");
                }
                for e in &used {
                    let _ = writeln!(pe, "  @export_symbol({0}_ptr, \"{0}\");", e.name);
                }
                pe.push_str("  @export_symbol(run);\n}\n");

                let pes = phase.pes.max(1);
                // PE `p` sits at `(p % w, p / w)` of a rectangle as tall as
                // fits: a row-major memcpy hands block `p` to that PE, and
                // the host copies stream one channel a row. A reducing phase
                // takes its own rectangle, one group of partials a row.
                let (w, h) = reduce.map_or_else(|| rect_of(pes), |r| r.rect);
                let mut layout = String::new();
                let _ = writeln!(
                    layout,
                    "const memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {w}, .height = {h} }});"
                );
                if reduce.is_some() {
                    layout.push_str("const c2d = @import_module(\"<collectives_2d/params>\");\n");
                }
                let _ = writeln!(layout, "\nlayout {{\n  @set_rectangle({w}, {h});");
                let _ = writeln!(
                    layout,
                    "  for (@range(i16, {h})) |y| {{\n    for (@range(i16, {w})) |x| {{"
                );
                if reduce.is_some() {
                    layout.push_str(
                        "      const params = c2d.get_params(@as(u16, x), @as(u16, y), .{ .x_colors = .{ @get_color(0), @get_color(1) }, .x_entrypoints = .{ @get_local_task_id(10), @get_local_task_id(11) }, .y_colors = .{ @get_color(4), @get_color(5) }, .y_entrypoints = .{ @get_local_task_id(12), @get_local_task_id(13) } });\n",
                    );
                    let _ = writeln!(layout, "      @set_tile_code(x, y, \"pe.csl\", .{{ .memcpy_params = memcpy.get_params(x), .pe_id = @as(u16, y * {w} + x), .c2d_params = params }});");
                } else {
                    let _ = writeln!(layout, "      @set_tile_code(x, y, \"pe.csl\", .{{ .memcpy_params = memcpy.get_params(x), .pe_id = @as(u16, y * {w} + x) }});");
                }
                layout.push_str("    }\n  }\n");
                for e in &used {
                    let _ = writeln!(layout, "  @export_name(\"{}\", [*]{}, true);", e.name, e.elem);
                }
                layout.push_str("  @export_name(\"run\", fn()void);\n}\n");
                let views = self.views.iter().filter(|v| used.iter().any(|e| e.name == v.name)).cloned().collect();
                Rendered {
                    layout,
                    pe,
                    manifest: Manifest {
                        exports: used,
                        entry: "run".into(),
                        rect: (w, h),
                        views,
                        lanes: group.iter().filter_map(|p| p.lane.clone()).collect(),
                        host: None,
                        table: None,
                        tables: Vec::new(),
                    },
                }
            })
            .collect()
    }

    /// The table-driven rendering of a fused group: `None` when a phase
    /// carries a statement that is not a kernel call, or an argument the
    /// table cannot carry (a pointer to a global), so the group renders as
    /// functions instead.
    /// A whole fire as one table-driven program: every phase on the same
    /// PEs (the resident placement), no host phase, no view, each buffer
    /// laid out one way throughout; `None` when the fire does not fit that
    /// shape, and the phases render as groups.
    /// Renders a single-PE memcpy program.
    pub fn render(&self) -> Rendered {
        let exports = self.exports();
        let entry = "run";

        let mut layout = String::new();
        layout.push_str("const memcpy = @import_module(\"<memcpy/get_params>\", .{ .width = 1, .height = 1 });\n\n");
        layout.push_str("layout {\n  @set_rectangle(1, 1);\n");
        layout.push_str(
            "  @set_tile_code(0, 0, \"pe.csl\", .{ .memcpy_params = memcpy.get_params(0), .pe_id = @as(u16, 0) });\n",
        );
        for e in &exports {
            let _ = writeln!(
                layout,
                "  @export_name(\"{}\", [*]{}, true);",
                e.name, e.elem
            );
        }
        let _ = writeln!(layout, "  @export_name(\"{entry}\", fn()void);");
        layout.push_str("}\n");

        let mut pe = String::new();
        pe.push_str("param memcpy_params;\nparam pe_id: u16;\n");
        pe.push_str("const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n");
        pe.push_str("const math = @import_module(\"<math>\");\n\n");
        pe.push_str(crate::library::prelude());
        pe.push('\n');
        for e in &exports {
            let n = e.rows as u64 * e.width as u64;
            match e.role {
                Symbol::Input | Symbol::InOut => {
                    let _ = writeln!(pe, "var {}: [{n}]{};", e.name, e.elem);
                }
                Symbol::Output => {
                    let _ = writeln!(pe, "var {} = @zeros([{n}]{});", e.name, e.elem);
                }
            }
            let _ = writeln!(pe, "var {0}_ptr: [*]{1} = &{0};", e.name, e.elem);
        }
        for (name, len, elem) in &self.scratch {
            let _ = writeln!(pe, "var {name} = @zeros([{len}]{elem});");
        }
        pe.push('\n');
        for g in &self.globals {
            pe.push_str(g);
            pe.push('\n');
        }
        pe.push('\n');
        for h in self.helpers.values() {
            pe.push_str(h);
            pe.push('\n');
        }
        for phase in &self.phases {
            phase.func.render(&mut pe);
            pe.push('\n');
        }
        let _ = writeln!(pe, "fn {entry}() void {{");
        for line in &self.prologue {
            let _ = writeln!(pe, "  {line}");
        }
        for phase in &self.phases {
            let _ = writeln!(pe, "  {}();", phase.func.name);
        }
        pe.push_str("  sys_mod.unblock_cmd_stream();\n}\n\n");
        pe.push_str("comptime {\n");
        for e in &exports {
            let _ = writeln!(pe, "  @export_symbol({0}_ptr, \"{0}\");", e.name);
        }
        let _ = writeln!(pe, "  @export_symbol({entry});");
        pe.push_str("}\n");

        Rendered {
            layout,
            pe,
            manifest: Manifest {
                exports,
                entry: entry.into(),
                rect: (1, 1),
                views: self.views.clone(),
                lanes: Vec::new(),
                host: None,
                table: None,
                tables: Vec::new(),
            },
        }
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
    if std::env::var_os("PIE_CEREBRAS_TRACE_FUSION").is_some() {
        eprintln!("fusion: {what}");
    }
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
        p.phase(f);
        let mut g = Func::new("phase_1");
        g.body.line("b2[1] = 1.0;");
        // Over two PEs: a phase on other PEs than the last does not fuse.
        p.phase_over(g, 2, vec![("b2".into(), Shard::Rows(2))], None, None);
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
        p.phase_over(f, 2, vec![("b2".into(), Shard::Roots { rows: 1, cols: 1, depth: 2 })], None, Some(reduce("b1")));
        let mut g = Func::new("phase_1");
        g.body.line("b2[1] = 1.0;");
        p.phase_over(g, 2, vec![("b2".into(), Shard::Roots { rows: 1, cols: 1, depth: 2 })], None, Some(reduce("b1")));
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
        let table = |kernels: &[&str], keep: &[(&str, u64)], work: u64, rows: u32, width: u32| {
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
                kernels: kernels.iter().map(|k| k.to_string()).collect(),
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
        let a = table(&["k_matmul", "k_rmsnorm"], &[("b1", 4000), ("b2", 6000)], 3000, 20, 10);
        let b = table(&["k_rmsnorm", "k_attend"], &[("b2", 6000), ("b3", 100)], 5000, 35, 12);
        let u = Unified::union([&a, &b]);
        assert_eq!(u.rows[0].kernels.len(), 3);
        assert_eq!(u.keep.iter().map(|(n, ..)| n.as_str()).collect::<Vec<_>>(), ["b1", "b2", "b3"]);
        // b2 (6000) does not share b1's array (4000 + 6000 > 8192).
        assert_eq!(u.keep[1].1, ARENA_CHUNK);
        assert_eq!(u.keep_chunks, 2);
        assert_eq!((u.rows[0].rows, u.rows[0].ops_words), (35, 12 + OPS_SLACK_WORDS));
        assert_eq!(u.rows[0].arena_words, 2 * ARENA_CHUNK + 5000 + WORK_SLACK_WORDS);
    }

    #[test]
    fn a_rendered_program_exports_its_buffers_and_entry() {
        let mut p = Program::new();
        let x = Tensor::new(7, 1, 4, Dtype::F32);
        p.read("t", x).unwrap();
        let mut f = Func::new("phase_0");
        f.body.line("b7[0] = 1.0;");
        p.phase(f);
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
        let (name, words) = parse_global_table("var inv_freq_3 = [2]f32 { 1.0, 0.5 };").expect("parses");
        assert_eq!(name, "inv_freq_3");
        assert_eq!(words, vec![1.0f32.to_bits(), 0.5f32.to_bits()]);
        let (_, words) = parse_global_table("var p = [1]u64 { 4294967297 };").expect("parses");
        assert_eq!(words, vec![1, 1]);
        assert!(parse_global_table("const x = 3;").is_none());
        assert!(parse_global_table("var y = [2]i32 { 1 };").is_none());
    }

    #[test]
    fn classes_split_alike_under_one_partition() {
        let row = |phases: u32, kernels: &[&str]| Table {
            kernels: kernels.iter().map(|k| k.to_string()).collect(),
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
        let prefill = vec![row(10, &["k_a"]), row(14, &["k_b"]), row(13, &["k_c"])];
        let decode = vec![row(20, &["k_a", "k_d"]), row(17, &["k_c"])];
        let p = Unified::partition(&[prefill.clone(), decode.clone()]).expect("same phase count");
        assert_eq!(p.row_phases, Some(vec![10, 14, 13]));
        assert!(p.rows.is_empty());
        // Rows that do not match cannot unite; alike rows do, per row.
        assert!(Unified::union_rows(&[prefill.clone(), decode]).is_none());
        let decode = vec![row(10, &["k_a", "k_d"]), row(14, &["k_b"]), row(13, &["k_e"])];
        let u = Unified::union_rows(&[prefill, decode]).expect("alike");
        assert_eq!(u.row_phases, Some(vec![10, 14, 13]));
        assert_eq!(u.rows[0].kernels, vec!["k_a", "k_d"]);
        assert_eq!(u.rows[2].kernels, vec!["k_c", "k_e"]);
        // A fire of another length has no shared partition.
        assert!(Unified::partition(&[vec![row(5, &[])], vec![row(6, &[])]]).is_none());
    }
}
