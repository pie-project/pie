//! The two inputs beside the plan: what a fire is allowed to be
//! ([`Budget`]), and what the device charges for the shapes it can be
//! ([`DeviceProfile`]). Both are PLAIN DATA and both are the caller's — the
//! shell measures a device once and hands the numbers in, because a compiler
//! that probed hardware would be a compiler that cannot be run on a laptop
//! against a plan for a machine somewhere else.

/// The ceilings a fire is baked against.
///
/// EVERY SYMBOLIC DIM IS SIZED HERE AND NOWHERE ELSE. `Dim::Tokens` becomes
/// [`max_tokens`](Budget::max_tokens), `Dim::Lanes` becomes
/// [`max_lanes`](Budget::max_lanes), and the arena's rectangles are cut at
/// those maxima (`arena::RowExpr`). That is what makes an offset static: a
/// value's column is as wide as the largest fire the deployment admits, and a
/// smaller fire uses its first rows and leaves the tail alone.
///
/// A BUDGET IS NOT AN ADMISSION CAP (decision #17). Exceeding one is a load
/// that is refused at compile time, not a request that is queued at run time.
///
/// ONE ROW AXIS, AND SINCE M1 THAT IS SAID OUT LOUD. This is the TOKEN
/// rectangle's ceilings; a deployment that also serves patch rows states
/// theirs beside it in [`Budgets`], which is the same doctrine over two row
/// spaces rather than a second place a dim is sized. Nothing here moved, and
/// an artifact baked through [`compile`](crate::compile) against this alone is
/// byte for byte the one this compiler produced before the second axis
/// existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    /// The most requests one fire may carry. `Dim::Lanes` is this number.
    pub max_lanes: u32,
    /// The most token rows one fire may carry, across every lane in it.
    /// `Dim::Tokens` is this number.
    pub max_tokens: u32,
    /// The shape lattice a fire's row count is rounded up to before it is
    /// launched — one immutable graph per entry (design §5). Ascending, each
    /// entry at most [`max_tokens`](Budget::max_tokens).
    ///
    /// v1 READS THE LENGTH AND NOTHING ELSE. The arena is cut once at the
    /// ceiling and every bucket addresses the same static offsets, so bucket
    /// membership does not change a byte of `CompiledModel` yet; it is here because
    /// P4's fallback menu is bucket-dependent (measured: copy 1.07x beats
    /// split 1.82x at M=64, and they converge at prefill scale) and that table
    /// cannot be keyed by something the compiler was never told.
    pub buckets: Vec<u32>,
    /// How many adapter banks the device pool holds — LoRA and its kin
    /// (design §8). A capacity, honestly stated, so that registering one more
    /// than there is room for is a refusal with a number in it.
    pub max_adapters: u32,
}

/// The ceilings a fire is baked against, ONE ROW AXIS AT A TIME.
///
/// **EVERY SYMBOLIC DIM IS STILL SIZED IN THE BUDGETS AND NOWHERE ELSE** —
/// there are simply two row spaces to size now (multimodal §5.1). [`Budget`]
/// above is the token rectangle's, whole and unchanged; [`patches`](Budgets::
/// patches) is the second axis's, and a third (the per-key attention-score
/// extent, attn-score §6.1) lands here as one more field rather than as a
/// parallel invention.
///
/// WHY A CONTAINER AND NOT TWO MORE FIELDS ON `Budget`. Because the token
/// axis's ceilings are exactly what every caller in the tree already holds and
/// exactly what a text-only deployment has to say, and growing the struct
/// would make every one of them state a second axis they do not serve.
/// [`compile`](crate::compile) is the one-axis door and stays the signature it
/// was; [`compile_axes`](crate::compile_axes) is the same pass told about a
/// second axis. `Budgets::from(budget)` is the conversion, and it is the
/// identity as far as any pre-campaign artifact is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budgets {
    /// The token rectangle's ceilings — `Dim::Tokens`, `Dim::Lanes` and their
    /// kin.
    pub tokens: Budget,
    /// **THE SECOND ROW AXIS'S SEAT** (multimodal §5.5), or `None` for a
    /// deployment that admits no patch rows.
    ///
    /// `None` IS NOT `max_patches: 0`, AND THE DIFFERENCE IS A REFUSAL. Zero
    /// patch rows would be a ceiling every rectangle is empty under — the
    /// same thing [`Budget::max_tokens`] is refused for — so a plan that
    /// states `Dim::Patches` against `None` is a load that does not happen,
    /// named at the door, rather than a tower carved at nothing. A plan that
    /// states NO patch row is exempt, which is what makes this field free for
    /// every pre-campaign SKU: the artifact it bakes is bit-identical whether
    /// the deployment declared a patch ladder or not.
    ///
    /// A SEAT AND NOT A VECTOR. §5.5's finding is that the ladder splits per
    /// axis and so does everything indexed by it: the patch ladder is its own
    /// ascending list, [`FallbackTable`](crate::FallbackTable) rows on the
    /// patch axis index THIS vector, and `LATTICE_FLOOR = 8`'s justification
    /// does not travel (see [`PATCH_LATTICE_FLOOR`]).
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

    /// **THIS DEPLOYMENT'S CEILINGS ON ONE ROW AXIS**, or `None` for an axis
    /// it does not admit at all.
    ///
    /// **THE ONE PLACE THIS CONTAINER'S SHAPE IS SPELLED.** The token
    /// rectangle is always stated — a fire has token rows by definition — and
    /// the patch one is an `Option`, which is `None` for every text-only
    /// deployment and is what refusals like [`Error::Unsized`](crate::Error)
    /// are decided off. Every pass that owes an answer per axis asks here
    /// rather than re-deriving the asymmetry, so a third rectangle lands as
    /// one more arm in this function and nowhere else.
    #[must_use]
    pub fn ladder(&self, axis: crate::RowAxis) -> Option<Ladder<'_>> {
        match axis {
            crate::RowAxis::Tokens => Some(self.tokens.ladder()),
            crate::RowAxis::Patches => self.patches.as_ref().map(PatchLadder::ladder),
        }
    }

    /// The ceiling `Dim::Patches` is sized at, and `0` for a deployment that
    /// declared no patch axis — which is the number that makes a patch
    /// rectangle empty, and therefore the number the bake refuses a
    /// patch-stating plan against rather than carving.
    #[must_use]
    pub fn max_patches(&self) -> u32 {
        self.patches.as_ref().map_or(0, |ladder| ladder.max_patches)
    }

    /// The ceiling `Dim::Images` is sized at, and `0` for a deployment that
    /// declared no patch axis — the patch rectangle's lane count, on the same
    /// terms [`max_patches`](Budgets::max_patches) is its row count.
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

/// What a fire is allowed to be on the PATCH axis: a ceiling and a ladder of
/// its own.
///
/// **ITS OWN, BECAUSE THE TOKEN AXIS'S NUMBERS ARE ABOUT TOKENS.** The token
/// lattice is doubling from 8 to `max_tokens` because that is where a decode
/// gemv arm stops beating a gemm; patches-per-image is fixed by the image
/// resize policy, so the patch ladder is a handful of rungs at multiples of
/// that number and a fire either has an image or does not. Sizing one off the
/// other would be a number carrying somebody else's argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchLadder {
    /// The most patch rows one fire may carry, across every image of every
    /// lane in it. `Dim::Patches` is this number.
    pub max_patches: u32,
    /// The shape lattice a fire's PATCH count is rounded up to — one
    /// immutable graph per entry, in the tower's own capture unit. Ascending,
    /// each entry at most [`max_patches`](PatchLadder::max_patches).
    ///
    /// **"6 + 6, NOT 6 × 6" IS A PROPERTY OF THIS BEING A SECOND VECTOR**
    /// (multimodal §5.3). One ladder per axis is one exec per axis per fire;
    /// a single ladder carrying both numbers would be the product.
    pub buckets: Vec<u32>,
    /// The most IMAGES one fire may carry, over every lane in it.
    /// `Dim::Images` is this number and `Dim::ImagesPlus(k)` is `this + k`.
    ///
    /// **THE PATCH AXIS'S `max_lanes`, AND NOT DERIVED FROM EITHER OF THE
    /// NUMBERS THAT LOOK LIKE IT.** It is not `Budget::max_lanes`: a lane may
    /// submit three images or none, so the two counts are two numbers in any
    /// mixed fire. And it is not [`max_patches`](PatchLadder::max_patches)
    /// either, even though an image contributes at least one patch row —
    /// reading it off that would reserve the `images + 1` indptr at the patch
    /// ceiling, which is a column sized by an argument that is not about it.
    /// A deployment states it, the way it states every other ceiling here,
    /// and the resize policy that fixes patches-per-image is what makes it a
    /// small number.
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

/// **ONE ROW AXIS'S CEILINGS AND ITS LATTICE** — the three numbers every axis
/// states, and the only three any pass in this crate reads off one.
///
/// **THE INTERNAL CURRENCY, AND THE FACES STAY FACES.** [`Budget`] is what
/// every caller in the tree holds and what a text-only deployment has to say;
/// [`PatchLadder`] is what a boot TOML parses into and what a `Budgets`
/// carries. Both are records with the axis's own vocabulary in their field
/// names — `max_tokens` and `max_lanes` over here, `max_patches` and
/// `max_images` over there — and that vocabulary is exactly what a pass
/// asking "does this ladder ascend" must not have to know. So the passes take
/// this: three numbers, no axis in any of their names, produced by
/// [`Budget::ladder`] and [`PatchLadder::ladder`] and handed to the one
/// acceptance walk (`accept_ladder`) instead of to two copies of it. P4's
/// fallback menu takes the two of the three it reads — the lattice and the
/// ceiling — because a whole ladder there was what made the patch axis
/// FABRICATE a `Budget` to carry them.
///
/// **A BORROWED VIEW AND NOT AN OWNED STRUCT**, because the lattice is a
/// `Vec` on both faces and a third owner of it would be a third copy per
/// bake. Nothing here outlives the budget it reads.
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

/// The smallest rung a PATCH ladder should state, and the argument is the
/// tower's rather than the trunk's.
///
/// **`LATTICE_FLOOR = 8` DOES NOT TRAVEL** (multimodal §5.5). The token
/// lattice floors at 8 because below it a decode's gemv arm beats the gemm
/// and the rungs stop buying anything — a claim about ONE-ROW-PER-LANE
/// decode, which the patch axis has no analogue of. A tower fire is a prefill
/// every time: the smallest thing it can carry is ONE IMAGE, and one image is
/// `(resize / patch)²` rows before the spatial merge divides it — 256 patches
/// at 224/14, 64 after a 2×2 merge. So the floor is the smallest whole image
/// a resize policy admits, and 64 is that number for the catalog's
/// patch-16 / merge-2 towers. A rung below it would round up to a fire that
/// cannot exist.
///
/// STATED HERE AND CHOSEN BY THE DEPLOYMENT. Like every other number in this
/// module it is a statute: a shell whose resize policy fixes a different
/// grid states its own rungs, and this is the argument it has to beat.
pub const PATCH_LATTICE_FLOOR: u32 = 64;

impl Budget {
    /// This axis's ceilings and lattice, in the shape every pass reads them
    /// in — [`Ladder`]. The token rectangle's `max_tokens` is its
    /// [`max_rows`](Ladder::max_rows) and its `max_lanes` is its
    /// [`max_lanes`](Ladder::max_lanes), which is the whole conversion:
    /// `max_adapters` is not a row-space fact and does not travel.
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
    /// A single-page decode-and-prefill deployment: 256 lanes, 8192 rows.
    /// Round numbers, chosen so that a caller who has not measured anything
    /// still bakes something that runs rather than something that divides by
    /// zero.
    fn default() -> Budget {
        Budget::new(256, 8192)
    }
}

/// What the device charges, as a table the caller measured.
///
/// COSTS ARE INPUT, NOT KNOWLEDGE (design §6, the `layout/` lineage row):
/// the rewrite kept its measurements inside the shell that took them, so a
/// second shell had no way to reach them and a host with no device had no way
/// to reproduce a layout. Here they cross the boundary as numbers.
///
/// v1 READS [`sms`](DeviceProfile::sms) — to refuse a profile that says the
/// device has none — and P6's four stream figures below. The three
/// microsecond figures in between are P3's whole decision procedure — a
/// region is worth a conditional exactly when its body outweighs the
/// evaluation point that guards it — and they are carried now so that P3 is a
/// pass to write rather than a pass plus a measurement campaign.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceProfile {
    /// Streaming multiprocessors. P6's `sm_hint` is a fraction of this, and a
    /// device with zero of them cannot run a graph.
    pub sms: u32,
    /// What one captured launch over an EMPTY window costs — the price of
    /// always-launch, and the reason it is the default lowering (decision #3).
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

    /// **P6'S CAP, AND ITS OFF SWITCH.** How many streams the shell will open
    /// BESIDE the main one. `0` is a first-class setting and not a
    /// degradation: a plan baked at zero side streams gets stream 0 for every
    /// region and not one event point, which is byte-for-byte the artifact
    /// this compiler produced before P6 existed. That is what makes the
    /// streams-off arm of a measurement a real arm rather than a claim.
    ///
    /// The cap is small on purpose. A fork group is the arms of one merge —
    /// three, on the fattest catalog plan — and a stream past the last arm
    /// costs a handle and buys nothing.
    pub side_streams: u32,

    /// What one fork/join event pair costs, in microseconds.
    ///
    /// INSIDE A RECORDED GRAPH THIS IS ALMOST FREE AND IT IS STILL NOT ZERO.
    /// A capture turns `cudaEventRecord` + `cudaStreamWaitEvent` into event
    /// nodes and a cross-stream edge, so a replay pays topology rather than a
    /// host call; what remains is the launch of two extra nodes and the
    /// scheduler's own cross-stream handoff. Two microseconds is the number
    /// this build assumes and the number a shell that has measured its own
    /// device should overwrite.
    pub event_pair_us: f32,

    /// **THE COST GATE.** How much device time BOTH sides of an overlap must
    /// be worth before P6 will fork one of them onto a side stream.
    ///
    /// The saving from running B beside A is at most `min(cost A, cost B)`;
    /// the price is one [`event_pair_us`](DeviceProfile::event_pair_us),
    /// paid on every fire whether or not the fire has rows for either window.
    /// So a twenty-microsecond kernel behind an event pair is a loss dressed
    /// as a schedule, and the floor is what refuses it. Twenty microseconds
    /// is an order of magnitude above the event pair, which is the margin a
    /// threshold wants when the quantity it gates is estimated rather than
    /// measured.
    pub fork_floor_us: f32,

    /// What one non-empty launch of each op family costs — P6's whole cost
    /// model, and the reason it is a TABLE rather than a measurement.
    pub family_us: FamilyCosts,

    /// **OPS WHOSE ENTRIES CLAIM A WORKSPACE NO SECOND LAUNCH MAY BE INSIDE**,
    /// by `Operands::name`. Two regions that each contain one are ORDERED by
    /// P6, whatever the values and the windows say.
    ///
    /// THIS IS A DEVICE FACT AND IT ARRIVES THE WAY EVERY OTHER DEVICE FACT
    /// ARRIVES — as data the caller passes, because the alternative is a
    /// backend-neutral compiler that knows a backend's allocator. What it
    /// describes on CUDA is `kernels_cuda::Ctx::scratch`: a slab keyed by a
    /// static NAME, process-global, grown but never shrunk, deliberately not
    /// per stream — an entry may not allocate per fire, because a capture
    /// forbids it. Two concurrent launches that both take the `attn.ssm_gdn_
    /// chunk_qk` plane get one pointer and stage into each other, and the
    /// failure is arithmetic rather than a fault.
    ///
    /// EMPTY IS THE HONEST DEFAULT, not a claim that no such op exists: a
    /// compiler with no device in the room cannot know which entries a
    /// backend's kernels reach a shared slab through. The CUDA shell passes
    /// `engine_cuda::EXCLUSIVE`; a golden-path walk that never launches
    /// anything needs nothing here.
    pub exclusive: Vec<String>,

    /// **OPS WHOSE KERNEL WALKS A SEGMENT LIST INSTEAD OF A RECTANGLE**, by
    /// `Operands::name` — the fact that turns [`Fallback::Grouped`](crate::Fallback::Grouped) from a
    /// typed seam into a menu entry `layout::menu` may actually choose.
    ///
    /// A DEVICE FACT, ARRIVING AS DATA, FOR THE REASON
    /// [`exclusive`](DeviceProfile::exclusive) ABOVE ARRIVES THAT WAY.
    /// `layout::menu`'s standing note says it declines `Grouped` because the
    /// entry "needs a kernel that takes a pointer/offset list — a fact about
    /// the backend's kernel table, which this crate has no dependency on and
    /// no business inventing". This is that fact, spelled the way every other
    /// one is: the compiler still does not know what a segment list IS, it
    /// only knows which op names the caller says can take one.
    ///
    /// What a name here promises is exactly one thing, and a shell that
    /// cannot keep it must not state it: given the region's rows as SEVERAL
    /// intervals, the op computes what `r` separate launches over those
    /// intervals compute, in ONE launch, and touches no row outside them. The
    /// last clause is the whole of why this is a per-op word rather than a
    /// per-backend one — a kernel that writes densely over the extent it was
    /// handed clobbers the rows in the gaps, which is what rules out
    /// `Attention::PrefillLse`'s split-kv fold on CUDA.
    ///
    /// EMPTY IS THE DEFAULT AND THEREFORE THE STATUS QUO: no mask is
    /// groupable, `menu` writes the entries it always wrote, and an artifact
    /// baked at this default is byte-for-byte the one this compiler produced
    /// before the field existed.
    pub grouped: Vec<String>,

}

/// What one launch of each op family costs at fire scale, in microseconds.
///
/// **AN ESTIMATE, STATED AS ONE.** A compiler cannot know a fire's
/// composition — which windows have rows is the batch the runtime happened to
/// assemble — so it cannot know what a region will actually cost. What it can
/// know is the CHARACTER of the ops in it, which is the same thing tart's
/// green-context finding turns on: attention and GEMM are the fat, divergent
/// kernels; norms and layout shuffles are not. These figures put those two
/// groups on opposite sides of
/// [`fork_floor_us`](DeviceProfile::fork_floor_us) and are not asked to do
/// anything finer.
///
/// The defaults are read off this tree's own measurements: a qwen35-d0.8b
/// decode fire is 3.30 ms over 423 captured nodes (build log 10), ~7.8 µs a
/// node averaged over a body that is mostly norms and residual adds, with the
/// projections and the attention arms carrying the rest. A deployment that
/// has profiled its own kernels passes its own numbers, exactly as it passes
/// its own [`sms`](DeviceProfile::sms).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FamilyCosts {
    /// `Operation::Attention` — the schedule builders excluded (they are
    /// `Phase::Prepare` and P6 never sees them).
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
    /// The figures palo design §4 cites, which are the ones `.wiki/tart`
    /// measured: ~1 us for a captured launch over an empty window, and
    /// `5 + 0.6*K` us per conditional evaluation point, against a ~250 us
    /// body threshold.
    ///
    /// A DEFAULT IS A STARTING POINT AND NOT A CLAIM ABOUT ANY PARTICULAR
    /// DEVICE. A shell that has measured its own hardware passes its own
    /// numbers; this is what a test and a golden-path walk use.
    fn default() -> DeviceProfile {
        DeviceProfile {
            sms: 132,
            empty_launch_us: 1.0,
            cond_fixed_us: 5.0,
            cond_per_arm_us: 0.6,
            fat_region_us: 250.0,
            side_streams: 2,
            event_pair_us: 2.0,
            fork_floor_us: 20.0,
            family_us: FamilyCosts::default(),
            exclusive: Vec::new(),
            grouped: Vec::new(),
        }
    }
}
