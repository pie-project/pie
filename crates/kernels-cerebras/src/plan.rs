//! What an entry states for the engine to split it by, and the [`Plan`]
//! the engine answers with.
//!
//! An entry knows the words its op holds on a PE as a function of a split
//! (rows, columns, lanes, pages, blocks of a lane's state); the engine owns
//! the terms of the fabric those words must fit and chooses the split. The
//! entry then writes its calls against the split it is handed.

use crate::error::Error;
use crate::phase::{BlockPlan, Reduce};

/// The engine's planner: the terms of the fabric, and the split it chooses
/// for each thing an op states.
pub trait Plan {
    /// The terms the engine plans by.
    fn fabric(&self) -> Fabric;

    /// Whether one PE holds `held`.
    fn holds(&self, held: Held) -> bool;

    /// The `(row groups, column groups)` a row-wise op of `footprint`
    /// spreads over.
    fn tile(&self, op: &'static str, footprint: Footprint) -> Result<(u32, u32), Error>;

    /// The `(row groups, column groups, depth groups)` a product spreads
    /// over; `None` multiplies it on the host.
    fn grid(&self, product: Product) -> Option<(u32, u32, u32)>;

    /// How many groups `spread` splits into; `None` when no split fits.
    fn spread(&self, spread: &Spread<'_>) -> Option<u32>;

    /// How blocks of a state split over PEs, and the rows of one phase;
    /// `None` when not even one row fits.
    fn blocks(&self, blocks: &Blocks<'_>) -> Option<BlockSplit>;

    /// How lanes over a recurrent bank spread over PEs.
    fn slots(&self, op: &'static str, slots: &Slots<'_>) -> Result<LaneSplit, Error>;

    /// How lanes over a routed bank spread over PEs; `None` when no split
    /// fits.
    fn routed(&self, routed: &Routed<'_>) -> Option<RoutedSplit>;

    /// How a paged pool's lanes spread over PEs, and the rows of one phase.
    fn pages(&self, op: &'static str, pages: &Pages<'_>) -> Result<PageSplit, Error>;
}

/// The engine's terms an entry plans its splits within: what a PE holds,
/// how far a phase spreads, the placement in force and how bf16 operands
/// are stored. The engine sets them and chooses every split by them.
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

/// What one PE holds: `words` in all, the largest array `array` of them
/// (an array has a bound of its own beside the PE's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Held {
    pub words: u64,
    pub array: u64,
}

impl Held {
    /// Arrays of these words side by side, and `beside` words more.
    pub fn arrays(words: &[u64], beside: u64) -> Self {
        Held {
            words: words.iter().sum::<u64>() + beside,
            array: words.iter().copied().max().unwrap_or(0),
        }
    }
}

/// An op over `units` that split into equal groups, one group a PE:
/// `held(groups)` is what a PE holds at that split. `collective` when the
/// groups combine their results over one row of PEs on the fabric (a sum,
/// a gather), which the engine then caps at a rectangle row and gives the
/// collectives' words.
pub struct Spread<'a> {
    pub units: u32,
    pub collective: bool,
    pub held: &'a dyn Fn(u32) -> Held,
}

/// A state `[a][b]` an op splits over PEs in blocks (attention's keys by
/// kv heads), over `rows` rows run a window at a time: `held(window,
/// a_groups, b_groups)` is what a PE holds, beside `code` words of the
/// kernel's code.
pub struct Blocks<'a> {
    pub rows: u32,
    pub a: u32,
    pub b: u32,
    pub code: u64,
    pub held: &'a dyn Fn(u32, u32, u32) -> u64,
}

/// A block split: the rows of one phase and the groups of each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSplit {
    pub window: u32,
    pub a_groups: u32,
    pub b_groups: u32,
}

/// Lanes over a recurrent bank: `elements` words of bank over `lanes`
/// lanes, a lane's slot state `[a][b][w]`, splitting along `b` and (when
/// `a_splits`) `a`; `beside(a_groups, b_groups)` is what a PE holds besides
/// its bank rows (scratch, its planes, whole buffers).
pub struct Slots<'a> {
    pub elements: u64,
    pub lanes: u32,
    pub a: u32,
    pub b: u32,
    pub w: u32,
    pub a_splits: bool,
    pub beside: &'a dyn Fn(u32, u32) -> u64,
}

/// A lane split: the PE count, the lanes one PE runs, and how a slot's
/// state splits within a lane (none: whole slots).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneSplit {
    pub pes: u32,
    pub lanes_per_pe: u32,
    pub block: Option<BlockPlan>,
}

/// Lanes over a routed bank (an adapter's or an expert's `[a][b]` block a
/// lane): `block(ag, bg)` words of one lane's block, `planes(ag, bg)` the
/// planes beside it, `partial(ag, bg)` the output plane a split of `b`
/// leaves partial; lanes per PE divide `lanes`.
pub struct Routed<'a> {
    pub lanes: u32,
    pub a: u32,
    pub b: u32,
    pub block: &'a dyn Fn(u32, u32) -> u64,
    pub planes: &'a dyn Fn(u32, u32) -> u64,
    pub partial: &'a dyn Fn(u32, u32) -> u64,
}

/// A routed bank's split: `lanes_per_pe` lanes a PE over block groups
/// `a_groups × b_groups`, on `pes` PEs; `sum` says how the partials over
/// the `b` groups add, when it splits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedSplit {
    pub pes: u32,
    pub lanes_per_pe: u32,
    pub a_groups: u32,
    pub b_groups: u32,
    pub sum: Option<Reduce>,
}

/// A paged pool an op spreads over PEs by lanes: `elements` words of pool
/// over `lanes` lanes of at most `max_pages` pages, a page `page_words`
/// (keys and values) of `page_rows` rows by `heads` column blocks;
/// `pool_pages` pages in all; the op runs over `rows` rows, a window of
/// them at a time (a phase each) when `windows`; `beside(window,
/// head_groups)` is what a PE holds besides pages for a window of rows.
pub struct Pages<'a> {
    pub elements: u64,
    pub lanes: u32,
    pub page_words: u64,
    pub max_pages: u32,
    pub heads: u32,
    pub page_rows: u32,
    pub pool_pages: u32,
    pub rows: u32,
    pub windows: bool,
    pub beside: &'a dyn Fn(u32, u32) -> u64,
}

/// A paged pool's split: the rows of one phase, and how the lanes spread
/// (none: one PE holds the pool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageSplit {
    pub window: u32,
    pub lanes: Option<PagedLanes>,
}

/// Paged lanes over PEs: `lanes_per_pe` lanes a PE, a lane's pages over
/// `page_groups` PEs, a page's heads over `head_groups` and its rows over
/// `row_groups`; under the resident placement the pool strided over the
/// row, `strided_pages` local pages a PE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedLanes {
    pub pes: u32,
    pub lanes_per_pe: u32,
    pub page_groups: u32,
    pub head_groups: u32,
    pub row_groups: u32,
    pub strided_pages: Option<u32>,
}
