//! What a fire stages beside its arena, and the one door the walk reads it
//! through.
//!
//! A `Program`'s slots answer for every value the TEXT names. Three kinds of
//! byte are not among them and never can be:
//!
//! * the RUNTIME PLANES — the tokens this fire runs, their absolute positions,
//!   the request CSR, the row-validity byte. A `Slot::Runtime` names one and
//!   says nothing else about it, because what it holds is the scheduler's
//!   answer and no text states a batch;
//! * the PAGE TRANSLATION — which physical page each request's KV row is read
//!   from and written to. The scheduler's answer again, about the pool;
//! * the POOLS themselves — the paged KV planes and the recurrent slabs, which
//!   outlive the fire — together with the geometry they were laid out with.
//!
//! # Why this is a trait and not a struct of regions
//!
//! Because the answer differs by build and the walk must not. On a device the
//! pools are `crate::device::Buffer`s and the planes are ranges of a staging
//! allocation; in `tests/the_walk_is_the_program.rs` they are numbers in a map.
//! Both are [`Slice`]s to the walk, which is the point: the executor is written
//! once and the device is behind the `dyn`.
//!
//! # It is NOT `crate::binding::FireTable`, and the difference is a measurement
//!
//! This crate already holds an enum of that name, filled by the legacy walk.
//! Three rows differ and each says something:
//!
//! * the legacy enum has no `QoIndptr` and no `RowValid`. Both are rows of
//!   `kernels::runtime::TIER1` that a `Slot::Runtime` can name directly, and a
//!   staging that could not answer them would make the walk refuse a plane the
//!   floor declares;
//! * it has no `RecurrentSlots`, because this driver allocates no recurrent
//!   slabs at all — `crate::hold`'s `slab` refuses always. The row is here
//!   because [`super::views::recurrent`] is what will fill it, and a view
//!   builder that had nowhere to ask would be the wrong place to find out;
//! * it has `RopeFrequencies`, and this one does not. A rotary ladder is a
//!   `Const` bank the model text names and the plan's parameter table carries,
//!   so it reaches a claim body through `BoundOp::tconst` like any other
//!   weight. That row was the routine era asking the driver for a table the
//!   catalog already had.

use super::marks::Slice;

/// Which of the fire's staged planes a slot wants.
///
/// MIRRORS `kernels::runtime::TIER1` rather than restating it: the first six
/// rows below are rows of that table, and the walk's own `Fire::runtime` maps a
/// `Slot::Runtime`'s string onto one so that a plane no table declares refuses
/// by name instead of arriving as a zero region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FireTable {
    /// The tokens this fire runs, one i32 per row.
    TokenIds,
    /// Each token's absolute position, one i32 per row.
    Positions,
    /// Which request owns each token, one i32 per row.
    RequestOfToken,
    /// The per-request query CSR. ITS ROWS ARE THE REQUEST COUNT, not the
    /// buffer's length — an appender reads `num_requests` off it.
    QoIndptr,
    /// One BYTE per row saying whether it carries a real token.
    RowValid,
    /// Which rows the fire samples, one per request.
    SamplingIndices,
    /// The KV page translation: physical page per logical page, per request.
    KvPageIndices,
    /// Its per-request CSR.
    KvPageIndptr,
    /// Per token: the physical page its KV row is written into.
    KvWritePage,
    /// Per token: the row within that page.
    KvWriteOffset,
    /// Per token: which recurrent-state slot its request occupies.
    ///
    /// A GDN stack's conv window and recurrent state are PER REQUEST and live
    /// in a slab that requests take turns in, exactly as KV pages do — so which
    /// slab row a token addresses is the fire's answer and not the model's.
    RecurrentSlots,
    /// The custom attention mask.
    AttentionMask,
    /// The per-row byte saying whether the mask applies.
    AttentionMaskEnabled,
    /// The decode split form's partials plane.
    ///
    /// `splits * rows * q_heads * (head_dim + 2)` floats, which is what
    /// `crate::binding::FireTable::AttnPartials` already documents: an
    /// unnormalised weighted-V accumulator per `(split, row, head)` and a
    /// `(max, sum_exp)` pair each, written by every workgroup of the split pass
    /// and read by the fold. A driver resource and not an operand — no traced
    /// value stands for it, so no plan mentions it and no arena holds it.
    AttnPartials,
}

/// A recurrent pool's three slabs, named as `kernels_vulkan::views` names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Slab {
    /// The recurrence's carry.
    State,
    /// The convolution window this fire reads.
    Conv,
    /// The one it writes.
    NewConv,
}

/// How the KV pool was laid out — the numbers a paged read divides by.
///
/// STATED ONCE, WHERE THE SLABS WERE ALLOCATED. A kernel that re-derived a
/// stride from a head count would be a second opinion about a fact the
/// allocator already settled, and the two would disagree the first time a tower
/// attended at two head widths — which gemma-4 does.
///
/// THREE FIELDS, WHERE `driver-wgpu`'s HAS FOUR, and the missing one is the
/// KV head count. That is not restraint: `kernels_vulkan::views::PagedKvView`
/// states no such field, so there is nowhere on this plane to put it, and
/// `kernels_vulkan::points::pool_heads` refuses by name because of it — *"the
/// paged pool's `(kv_heads, head_dim)`: no point states both and `PagedKvView`
/// carries neither"*. Adding a row here would state a number no claim body
/// could read. See [`super::views`] for what that costs at the fire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvGeometry {
    /// Token rows per page. ZERO IS NOT A PAGE SIZE and the claim bodies refuse
    /// it by name: a store with no pool behind it would plan a full write in
    /// which every token divides to page zero, offset zero — every layer
    /// landing on one row, with no refusal anywhere.
    pub page_size: i32,
    /// Elements between one token and the next within a head.
    pub seq_stride: u64,
    /// Elements between one KV head's pages and the next.
    pub head_stride: u64,
    /// KV heads in this pool's row.
    ///
    /// CARRIED RATHER THAN DERIVED. `seq_stride` and `head_stride` are both
    /// element counts and their ratio is not a head count -- one steps a token
    /// inside a head, the other steps a head -- so a driver that knows the
    /// number states it. `kernels_vulkan::attn` refuses a row whose width is
    /// not `kv_heads * head_dim`, which is the check this makes possible.
    pub kv_heads: i32,
    /// Elements in one KV head.
    pub head_dim: i32,
}

/// The decode split policy this fire runs under.
///
/// `kernels_vulkan::attn::decode_splits` is the function that decides it — a
/// fact about the FIRE (history depth against head and row count) and not about
/// the model — and `kernels_vulkan::views::SplitView` is what carries it to a
/// body. `splits <= 1` is the unsplit reading and the partials handle is then
/// never read, which is why a driver that stages no partials plane answers
/// [`Splits::UNSPLIT`].
///
/// NOT A METAL CONCEPT and shared with wgpu, which is the pattern this plane
/// keeps hitting: it takes the split from one sibling and the barriers from the
/// other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Splits {
    /// How many KV splits a decode runs; `<= 1` means unsplit.
    pub splits: i32,
}

impl Splits {
    /// The unsplit policy — one piece, no partials plane.
    pub const UNSPLIT: Self = Self { splits: 1 };
}

impl Default for Splits {
    fn default() -> Self {
        Self::UNSPLIT
    }
}

/// What answers for the bytes a `Program` does not name.
///
/// # FOUR OF THESE ROWS HAVE NO READER YET, AND THE READER IS NOT THE WALK'S
///
/// [`Pools::mask_stride`], [`Pools::splits`] and the two `FireTable` rows they
/// go with ([`FireTable::AttentionMask`], [`FireTable::AttnPartials`]) are
/// staged and not read, because what reads them on this plane is
/// `kernels_vulkan::attn`'s private `Fired::of` — through
/// `Staged::resident::<AttnMask>()` and `Staged::resident::<AttnSplit>()`,
/// whose blanket impl on `dyn Encode` refuses by name today.
///
/// They stay because they are what this driver STAGES rather than what it
/// currently binds: the legacy walk computes both
/// (`crate::binding::FireNumber::AttentionMaskStride` and
/// `kernels_vulkan::attn::decode_splits` behind `KvHistoryBucket`), and a
/// staging door that had to be invented on the day the `Staged` impl lands
/// would be a door invented under time pressure. [`super::views`] names the
/// file that opens it.
pub trait Pools {
    /// A layer's KV pages: its keys when `values` is clear, its values when
    /// set. `None` for a layer this driver holds no pool for.
    fn kv(&self, layer: u32, values: bool) -> Option<Slice>;

    /// A layer's recurrent slab. `None` REFUSES rather than binding nothing: a
    /// scan handed a null carry answers fluently and wrongly.
    fn slab(&self, layer: u32, which: Slab) -> Option<Slice>;

    /// The KV pool's own layout.
    fn kv_geometry(&self) -> KvGeometry;

    /// One of the fire's staged planes.
    fn table(&self, which: FireTable) -> Option<Slice>;

    /// The mask's row pitch, in keys. Zero means the fire staged no custom
    /// mask, which the enable plane's zeros say again — and the claim bodies
    /// read both.
    fn mask_stride(&self) -> u32 {
        0
    }

    /// How this fire's decodes are split. Defaulted to unsplit, which is what a
    /// driver that stages no partials plane must answer.
    fn splits(&self) -> Splits {
        Splits::UNSPLIT
    }
}
