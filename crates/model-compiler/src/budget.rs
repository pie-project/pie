//! The two inputs beside the plan: what a fire is allowed to be
//! ([`Budget`]), and what the device charges for the shapes it can be
//! ([`DeviceProfile`]). Both are plain data and both are the caller's: the
//! shell measures a device once and hands the numbers in, so the compiler
//! itself never probes hardware.

/// The ceilings a fire is baked against. Every symbolic dim is sized here
/// and nowhere else: `Dim::Tokens`/`Dim::Lanes` become the maxima the
/// arena's rectangles are cut at, which is what makes an offset static.
/// Exceeding a budget is refused at compile time, not queued at run time. A
/// deployment that also serves patch rows states those in [`Budgets`]
/// instead of growing this struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    /// The most requests one fire may carry. `Dim::Lanes` is this number.
    pub max_lanes: u32,
    /// The most token rows one fire may carry, across every lane in it.
    /// `Dim::Tokens` is this number.
    pub max_tokens: u32,
    /// The shape lattice a fire's row count is rounded up to before launch —
    /// one immutable graph per entry. Ascending, each entry at most
    /// [`max_tokens`](Budget::max_tokens). Bucket membership doesn't change
    /// the arena (cut once at the ceiling); it exists because the fallback
    /// menu is bucket-dependent.
    pub buckets: Vec<u32>,
    /// How many adapter banks the device pool holds (LoRA and its kin). A
    /// capacity, honestly stated, so registering one past it is a refusal
    /// with a number in it.
    pub max_adapters: u32,
}

/// The ceilings a fire is baked against, one row axis at a time. [`Budget`]
/// is the token rectangle's, whole and unchanged; [`patches`](Budgets::patches)
/// is the second axis's. Kept as a container rather than two more fields on
/// `Budget` because every token-only caller already holds a bare `Budget`;
/// `Budgets::from(budget)` is the identity conversion for those.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budgets {
    /// The token rectangle's ceilings — `Dim::Tokens`, `Dim::Lanes` and their
    /// kin.
    pub tokens: Budget,
    /// The second row axis's seat, or `None` for a deployment that admits no
    /// patch rows. `None` is not `max_patches: 0`: a plan stating
    /// `Dim::Patches` against `None` is refused at the door rather than
    /// carving a rectangle at zero rows, so a text-only SKU's artifact is
    /// bit-identical whether or not this field exists.
    pub patches: Option<PatchLadder>,
}

impl Budgets {
    /// The token axis alone — what every pre-campaign deployment admits, and
    /// what [`compile`](crate::compile) passes.
    #[must_use]
    pub fn of(tokens: Budget) -> Budgets {
        Budgets {
            tokens,
            patches: None,
        }
    }

    /// The same, with a patch axis admitted — what a deployment serving a
    /// vision tower passes.
    #[must_use]
    pub fn with_patches(mut self, patches: PatchLadder) -> Budgets {
        self.patches = Some(patches);
        self
    }

    /// This deployment's ceilings on one row axis, or `None` for an axis it
    /// does not admit. Every pass that owes an answer per axis asks here
    /// rather than re-deriving the asymmetry.
    #[must_use]
    pub fn ladder(&self, axis: crate::RowAxis) -> Option<Ladder<'_>> {
        match axis {
            crate::RowAxis::Tokens => Some(self.tokens.ladder()),
            crate::RowAxis::Patches => self.patches.as_ref().map(PatchLadder::ladder),
        }
    }

    /// The ceiling `Dim::Patches` is sized at, `0` for a deployment with no
    /// patch axis (the value the bake refuses a patch-stating plan against).
    #[must_use]
    pub fn max_patches(&self) -> u32 {
        self.patches.as_ref().map_or(0, |ladder| ladder.max_patches)
    }

    /// Same, for `Dim::Images` (the patch rectangle's lane count).
    #[must_use]
    pub fn max_images(&self) -> u32 {
        self.patches.as_ref().map_or(0, |ladder| ladder.max_images)
    }
}

impl From<Budget> for Budgets {
    fn from(tokens: Budget) -> Budgets {
        Budgets::of(tokens)
    }
}

impl From<&Budget> for Budgets {
    fn from(tokens: &Budget) -> Budgets {
        Budgets::of(tokens.clone())
    }
}

/// What a fire is allowed to be on the patch axis: a ceiling and a ladder of
/// its own, sized independently of the token axis's numbers (patches-per-image
/// is fixed by the image resize policy, not by the decode gemv/gemm crossover
/// the token lattice doubles against).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchLadder {
    /// The most patch rows one fire may carry, across every image of every
    /// lane in it. `Dim::Patches` is this number.
    pub max_patches: u32,
    /// The shape lattice a fire's patch count is rounded up to — one
    /// immutable graph per entry, in the tower's own capture unit. Ascending,
    /// each entry at most [`max_patches`](PatchLadder::max_patches). A
    /// second vector rather than a combined one, so one ladder per axis
    /// means one exec per axis per fire, not their product.
    pub buckets: Vec<u32>,
    /// The most images one fire may carry, over every lane in it.
    /// `Dim::Images` is this number and `Dim::ImagesPlus(k)` is `this + k`.
    /// Not derived from `Budget::max_lanes` (a lane may submit any number of
    /// images) or from [`max_patches`](PatchLadder::max_patches) (an
    /// `images + 1` indptr sized off patches would carry an unrelated
    /// argument); the deployment states it directly.
    pub max_images: u32,
}

impl PatchLadder {
    /// A ceiling with no ladder — one implicit rung at the ceiling, which is
    /// what the token axis means by an empty `buckets` too.
    #[must_use]
    pub fn new(max_patches: u32, max_images: u32) -> PatchLadder {
        PatchLadder {
            max_patches,
            buckets: Vec::new(),
            max_images,
        }
    }

    /// This axis's ceilings and lattice, in the shape every pass reads them
    /// in — [`Ladder`], which is what a `PatchLadder` IS once the names
    /// `patches` and `images` are spent.
    #[must_use]
    pub fn ladder(&self) -> Ladder<'_> {
        Ladder {
            max_rows: self.max_patches,
            max_lanes: self.max_images,
            buckets: &self.buckets,
        }
    }
}

/// One row axis's ceilings and its lattice — the three numbers every axis
/// states, in axis-agnostic vocabulary, so a pass like the acceptance walk
/// doesn't need to know whether it's reading a `Budget` or a `PatchLadder`.
/// A borrowed view, not an owned struct, since the lattice `Vec` already has
/// a home on both faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ladder<'a> {
    /// The most rows one fire may carry on this axis —
    /// [`Budget::max_tokens`] on the token axis,
    /// [`PatchLadder::max_patches`] on the patch one.
    pub max_rows: u32,
    /// The most of this axis's own lanes one fire may carry —
    /// [`Budget::max_lanes`] (requests) on the token axis,
    /// [`PatchLadder::max_images`] (images) on the patch one.
    pub max_lanes: u32,
    /// The shape lattice a fire's row count on this axis is rounded up to —
    /// one immutable graph per entry. Ascending, each entry at most
    /// [`max_rows`](Ladder::max_rows).
    pub buckets: &'a [u32],
}

/// The smallest rung a patch ladder should state. The token lattice's floor
/// is about one-row-per-lane decode, which the patch axis has no analogue
/// of: a tower fire is always a prefill, so the smallest thing it can carry
/// is one whole image's worth of patches (post spatial-merge). A rung below
/// that would round up to a fire that cannot exist. A default for the
/// catalog's own resize policy; a different grid states its own rungs.
pub const PATCH_LATTICE_FLOOR: u32 = 64;

impl Budget {
    /// This axis's ceilings and lattice as a [`Ladder`]; `max_adapters` is
    /// not a row-space fact and does not travel.
    #[must_use]
    pub fn ladder(&self) -> Ladder<'_> {
        Ladder {
            max_rows: self.max_tokens,
            max_lanes: self.max_lanes,
            buckets: &self.buckets,
        }
    }

    /// The ceilings, with no bucket lattice and no adapter pool — what a test
    /// or a golden-path walk wants.
    #[must_use]
    pub fn new(max_lanes: u32, max_tokens: u32) -> Budget {
        Budget {
            max_lanes,
            max_tokens,
            buckets: Vec::new(),
            max_adapters: 0,
        }
    }
}

impl Default for Budget {
    /// Round numbers so a caller who hasn't measured anything still bakes
    /// something that runs.
    fn default() -> Budget {
        Budget::new(256, 8192)
    }
}

/// What the device charges, as a table the caller measured. Costs are input,
/// not knowledge: they cross the layout boundary as plain numbers rather
/// than living inside whichever shell took the measurement, so a host with
/// no device can still reproduce a layout.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceProfile {
    /// Streaming multiprocessors. `stream`'s `sm_hint` is a fraction of this,
    /// and a device with zero of them cannot run a graph.
    pub sms: u32,
    /// What one captured launch over an empty window costs — the price of
    /// always-launch, the default lowering.
    pub empty_launch_us: f32,
    /// The fixed half of one conditional evaluation point, paid whether the
    /// body is taken or not.
    pub cond_fixed_us: f32,
    /// The per-arm half of the same: `cond_fixed_us + cond_per_arm_us * K` for
    /// a SWITCH over `K` arms.
    pub cond_per_arm_us: f32,
    /// How fat a region's body must be before a conditional around it pays for
    /// itself. Layer granularity or coarser, in practice.
    pub fat_region_us: f32,

    /// How many streams the shell will open beside the main one. `0` is
    /// first-class, not a degradation: every region then gets the main
    /// stream and no event point. Kept small — a fork group is the arms of
    /// one merge, and a stream past the last arm buys nothing.
    pub side_streams: u32,


    /// How much device time both sides of an overlap must be worth before a
    /// side stream is used, so an event pair isn't paid for a saving smaller
    /// than itself. Set well above [`event_pair_us`](DeviceProfile::event_pair_us)
    /// since the quantity it gates is estimated, not measured.
    pub fork_floor_us: f32,

    /// What one non-empty launch of each op family costs — `stream`'s whole cost
    /// model, and the reason it is a TABLE rather than a measurement.
    pub family_us: FamilyCosts,

    /// Ops whose entries claim a workspace no second launch may be inside,
    /// by `Operands::name`; two regions that each contain one are ordered
    /// regardless of what the values and windows say. A device fact passed
    /// as data since the compiler has no dependency on a backend's
    /// allocator; empty is the honest default (no claim that no such op
    /// exists), not a widening.
    pub exclusive: Vec<String>,

    /// Ops whose kernel walks a segment list instead of a rectangle, by
    /// `Operands::name` — the fact that lets [`Fallback::Grouped`](crate::Fallback::Grouped)
    /// be chosen. A name here promises: given the region's rows as several
    /// intervals, the op computes what `r` separate launches would, in one
    /// launch, touching no row outside them. Empty is the default and the
    /// status quo (no mask is groupable).
    pub grouped: Vec<String>,
}

/// What one launch of each op family costs at fire scale, in microseconds.
/// An estimate, not a measurement: a compiler can't know a fire's actual
/// composition, only the character of its ops — attention and GEMM are fat
/// and divergent, norms and layout shuffles are not — which is enough to
/// place them on either side of [`fork_floor_us`](DeviceProfile::fork_floor_us).
/// Defaults are read off this tree's own measurements; a deployment that has
/// profiled its own kernels passes its own numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FamilyCosts {
    /// `Operation::Attention` — the schedule builders excluded (they are
    /// `Phase::Prepare` and `stream` never sees them).
    pub attention: f32,
    /// `Operation::Linear` — the projections and the MoE ladders.
    pub linear: f32,
    /// `Operation::Elementwise` — norms, residual adds, scales.
    pub elementwise: f32,
    /// `Operation::Layout` — embeds, splits, selects.
    pub layout: f32,
    /// `Operation::Collective` — never forked (a collective is a barrier),
    /// carried so the table is total.
    pub collective: f32,
    /// `Operation::CustomCuda` — the fused per-family entries.
    pub custom: f32,
}

impl FamilyCosts {
    /// What one node of this op costs.
    #[must_use]
    pub fn of(&self, op: &model_ir::Operation) -> f32 {
        match op {
            model_ir::Operation::Attention(_) => self.attention,
            model_ir::Operation::Linear(_) => self.linear,
            model_ir::Operation::Elementwise(_) => self.elementwise,
            model_ir::Operation::Layout(_) => self.layout,
            model_ir::Operation::Collective(_) => self.collective,
            model_ir::Operation::CustomCuda(_) => self.custom,
        }
    }
}

impl Default for FamilyCosts {
    fn default() -> FamilyCosts {
        FamilyCosts {
            attention: 60.0,
            linear: 40.0,
            elementwise: 4.0,
            layout: 4.0,
            collective: 50.0,
            custom: 20.0,
        }
    }
}

impl Default for DeviceProfile {
    /// A starting point, not a claim about any particular device; a shell
    /// that has measured its own hardware passes its own numbers. Used by
    /// tests and golden-path walks.
    fn default() -> DeviceProfile {
        DeviceProfile {
            sms: 132,
            empty_launch_us: 1.0,
            cond_fixed_us: 5.0,
            cond_per_arm_us: 0.6,
            fat_region_us: 250.0,
            side_streams: 2,
            fork_floor_us: 20.0,
            family_us: FamilyCosts::default(),
            exclusive: Vec::new(),
            grouped: Vec::new(),
        }
    }
}
