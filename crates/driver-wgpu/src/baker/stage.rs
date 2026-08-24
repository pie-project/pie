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
//! pools are `wgpu::Buffer`s and the planes are ranges of a staging buffer; in
//! `tests/the_walk_is_the_program.rs` they are numbers in a map. Both are
//! [`Slice`]s to the walk, which is the point: the executor is written once and
//! the device is behind the `dyn`.

use super::marks::Slice;

/// Which of the fire's staged planes a slot wants.
///
/// MIRRORS `kernels::runtime::TIER1` rather than restating it: the first six
/// rows below are rows of that table, and the walk's own `Fire::runtime` maps a
/// `Slot::Runtime`'s string onto one so that a plane no table declares refuses
/// by name instead of arriving as a zero region.
///
/// It is `driver-metal`'s enum unchanged, including the two rows metal added
/// on its way from the legacy executor — the KV strides went into
/// [`KvGeometry`] because they are numbers the allocator settled rather than
/// planes a fire stages, and a rotary ladder is a `Const` bank the text names.
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
    /// NOT A ROW METAL HAS, and the difference is this plane's split decode.
    /// `kernels_wgpu::views::SplitView` carries a partials handle and a split
    /// count, and `attn/sdpa_paged.wgsl`'s split arm folds the first over the
    /// second. Metal fires unsplit and its driver answers `splits: 1`, so it
    /// stages no such plane; this one may.
    AttnPartials,
}

/// A recurrent pool's three slabs, named as `kernels_wgpu::views` names them.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvGeometry {
    /// Token rows per page. ZERO IS NOT A PAGE SIZE and the claim bodies refuse
    /// it by name: a store with no pool behind it would plan a full write in
    /// which every token divides to page zero, offset zero — every layer
    /// landing on one row, with no refusal anywhere. `kernels_wgpu::attn` makes
    /// that refusal explicitly and calls it load-bearing.
    pub page_size: i32,
    /// Elements between one token and the next within a head.
    pub seq_stride: u64,
    /// Elements between one KV head's pages and the next.
    pub head_stride: u64,
    /// The KV head count the pool was laid out with.
    ///
    /// A ROW METAL'S GEOMETRY DOES NOT HAVE, and it is here because
    /// `kernels_wgpu::views::AttnFireView` states it as a FIELD and argues for
    /// it at length: `attention.decode` declares no KV head count, cuda reads
    /// one off the pool's strides through a layout flag this plane's view does
    /// not carry, and rather than invent one the count is stated by the party
    /// that chose it — the allocator. `attention.prefill` declares its own and
    /// `kernels_wgpu::attn` refuses when the two disagree, which is the check
    /// this field exists to be checked against.
    pub kv_heads: i32,
}

/// The decode split policy this fire runs under.
///
/// NOT A METAL CONCEPT. `kernels_wgpu::views::SplitView` is one of the five
/// things this plane's attention bodies read off the pool view rather than off
/// their statement, and the split count is the driver's call: how many pieces
/// the KV range is cut into for a decode, and where the partials each piece
/// leaves are folded from. `splits <= 1` is the unsplit reading and the
/// partials handle is then never read, which is why an unsplit driver can
/// answer this with `Splits::UNSPLIT` and stage no plane.
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

    /// The mask's row pitch, in elements. Zero means the fire staged no custom
    /// mask, which the enable plane's zeros say again — and the claim bodies
    /// read both.
    fn mask_stride(&self) -> u32 {
        0
    }

    /// How this fire's decodes are split. Defaulted to unsplit, which is what
    /// a driver that stages no partials plane must answer.
    fn splits(&self) -> Splits {
        Splits::UNSPLIT
    }
}
