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
/// v1 READS ONLY [`sms`](DeviceProfile::sms), and reads it to refuse a
/// profile that says the device has none. The three microsecond figures are
/// P3's whole decision procedure — a region is worth a conditional exactly
/// when its body outweighs the evaluation point that guards it — and they are
/// carried now so that P3 is a pass to write rather than a pass plus a
/// measurement campaign.
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
        }
    }
}
