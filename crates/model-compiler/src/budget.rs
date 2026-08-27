//! The two inputs beside the plan: what a fire is allowed to be
//! ([`Budgets`]), and what the device charges for the shapes it can be
//! ([`DeviceProfile`]). Both are PLAIN DATA and both are the caller's — the
//! shell measures a device once and hands the numbers in, because a compiler
//! that probed hardware would be a compiler that cannot be run on a laptop
//! against a plan for a machine somewhere else.

/// The ceilings a fire is baked against.
///
/// EVERY SYMBOLIC DIM IS SIZED HERE AND NOWHERE ELSE. `Dim::Tokens` becomes
/// [`max_tokens`](Budgets::max_tokens), `Dim::Lanes` becomes
/// [`max_lanes`](Budgets::max_lanes), and the arena's rectangles are cut at
/// those maxima (`arena::RowExpr`). That is what makes an offset static: a
/// value's column is as wide as the largest fire the deployment admits, and a
/// smaller fire uses its first rows and leaves the tail alone.
///
/// A BUDGET IS NOT AN ADMISSION CAP (decision #17). Exceeding one is a load
/// that is refused at compile time, not a request that is queued at run time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budgets {
    /// The most requests one fire may carry. `Dim::Lanes` is this number.
    pub max_lanes: u32,
    /// The most token rows one fire may carry, across every lane in it.
    /// `Dim::Tokens` is this number.
    pub max_tokens: u32,
    /// The shape lattice a fire's row count is rounded up to before it is
    /// launched — one immutable graph per entry (design §5). Ascending, each
    /// entry at most [`max_tokens`](Budgets::max_tokens).
    ///
    /// v1 READS THE LENGTH AND NOTHING ELSE. The arena is cut once at the
    /// ceiling and every bucket addresses the same static offsets, so bucket
    /// membership does not change a byte of `Baked` yet; it is here because
    /// P4's fallback menu is bucket-dependent (measured: copy 1.07x beats
    /// split 1.82x at M=64, and they converge at prefill scale) and that table
    /// cannot be keyed by something the compiler was never told.
    pub buckets: Vec<u32>,
    /// How many adapter banks the device pool holds — LoRA and its kin
    /// (design §8). A capacity, honestly stated, so that registering one more
    /// than there is room for is a refusal with a number in it.
    pub max_adapters: u32,
}

impl Budgets {
    /// The ceilings, with no bucket lattice and no adapter pool — what a test
    /// or a golden-path walk wants.
    #[must_use]
    pub fn new(max_lanes: u32, max_tokens: u32) -> Budgets {
        Budgets {
            max_lanes,
            max_tokens,
            buckets: Vec::new(),
            max_adapters: 0,
        }
    }
}

impl Default for Budgets {
    /// A single-page decode-and-prefill deployment: 256 lanes, 8192 rows.
    /// Round numbers, chosen so that a caller who has not measured anything
    /// still bakes something that runs rather than something that divides by
    /// zero.
    fn default() -> Budgets {
        Budgets::new(256, 8192)
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
    /// `driver_cuda::EXCLUSIVE`; a golden-path walk that never launches
    /// anything needs nothing here.
    pub exclusive: Vec<String>,
}

/// What one launch of each op family costs at fire scale, in microseconds.
///
/// **AN ESTIMATE, STATED AS ONE.** A compiler cannot know a fire's
/// composition — which windows have rows is the batch the engine happened to
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
        }
    }
}
