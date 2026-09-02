//! Per-device tuning constants — every crossover this plane decides.
//!
//! A default-constructed [`DeviceTuning`] reproduces the reference Apple7
//! machine's measurements exactly; an unrecognised device gets those numbers
//! rather than an extrapolation. [`Overrides`] carries a boot document's
//! `[metal.tuning]` answers (no env vars read here); each field is an
//! `Option` so an explicit zero is distinguishable from "not set".
//!
//! Reached through one process-wide cell ([`describe`], [`override_with`],
//! [`current`]) rather than a threaded parameter, since the value is
//! constant for the process.

use std::sync::OnceLock;

/// What this plane knows about the GPU it is running on. Queried once by
/// the shell that binds the device; every field has a value on every path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `MTLGPUFamilyApple<N>`. Probe newest-first: families are cumulative,
    /// so an oldest-first probe would misclassify every newer GPU as the
    /// oldest recognized family. 0 when nothing answered (selects defaults).
    pub apple_family: u32,

    /// GPU cores. Recorded but unused here — the crossovers below are set
    /// by per-core matrix throughput, which the family names and the count
    /// does not. 0 when absent.
    pub gpu_core_count: u32,
}

impl DeviceInfo {
    /// The family a `MTLDevice`'s name implies, for a shell with a name and
    /// no `supportsFamily:` probe. Weaker than the probe: it can't report
    /// core count, and answers 0 (selecting the measured defaults) for any
    /// silicon newer than this table.
    #[must_use]
    pub fn of_name(name: &str) -> Self {
        let family = if name.contains("M1") {
            7
        } else if name.contains("M2") {
            8
        } else if name.contains("M3") || name.contains("M4") {
            9
        } else {
            0
        };
        Self {
            apple_family: family,
            gpu_core_count: 0,
        }
    }
}

/// The tuned constants, defaulted to the reference Apple7 measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceTuning {
    /// The batch at which the tiled GEMM overtakes the batched GEMV, for a
    /// checkpoint whose GEMM reaches the FP16 matrix path (dense only — see
    /// [`qmm_min_batch_moe`](Self::qmm_min_batch_moe) and
    /// [`qmm_min_batch_emulated`](Self::qmm_min_batch_emulated)).
    ///
    /// Default: 5, from sweeping ms/fire across four checkpoints; the value
    /// moves whenever the GEMM's own cost curve does.
    ///
    /// Read against the fire's own row count, not a load-time slot
    /// capacity, so the kernel picked can vary with neighboring
    /// compositions — an accepted small numerical drift, not a bug.
    pub qmm_min_batch: u32,

    /// The same crossover for a checkpoint whose FFN is routed: only the
    /// attention/head projections are dense here, since the routed FFN
    /// takes [`moe_tile_mid_per`](Self::moe_tile_mid_per)'s decision instead.
    ///
    /// Default: 8; Apple8 keeps 12.
    pub qmm_min_batch_moe: u32,

    /// The same crossover for a checkpoint whose quantization doesn't reach
    /// the FP16 matrix path (anything but 4-bit at group 64).
    ///
    /// Default: 12, one number for dense and routed since neither has been
    /// measured apart.
    pub qmm_min_batch_emulated: u32,

    /// The threadgroup count at which the unsplit GEMM's BN=32 tile
    /// overtakes BN=16.
    ///
    /// Default: 160, bracketed between 144 and 192 threadgroups. A machine
    /// with more cores saturates later and wants this higher; Apple9
    /// (fewer cores) reads lower, at (64, 96].
    pub qmm_bn_crossover_tg: u32,

    /// Rows an expert's run must hold before the mixture's GEMM takes the
    /// 32-row tile, and then the 64-row one.
    ///
    /// Default: 32, and 64 never — no device has been measured wide enough
    /// to want it.
    pub moe_tile_mid_per: u32,

    /// The 64-row rung's threshold — see
    /// [`moe_tile_mid_per`](Self::moe_tile_mid_per). Out of reach by
    /// default, by design and not omission.
    pub moe_tile_wide_per: u32,

    /// Whether a g64/b4 projection stages its input to FP16 and feeds
    /// native FP16 simdgroup MMA instead of BF16.
    ///
    /// Default: on — roughly 40% faster on the GEMM, since Apple7 and Apple8
    /// emulate bfloat16 matmul. Apple9 has a native BF16 path, so this may be
    /// actively wrong there rather than merely unmeasured.
    pub fp16_qmm: bool,

    /// Rows a request must contribute before its attention takes the tiled
    /// kernel instead of the per-row one.
    ///
    /// Default: 32 (also the tile's height). A machine whose shuffles are
    /// cheap relative to its FMAs wants this higher, since the tiled
    /// kernel's advantage is fewer reductions.
    pub sdpa_tile_min_rows_per_request: u32,

    /// Whether a tiled prefill attention runs on the simdgroup matrix unit
    /// rather than the scalar path.
    ///
    /// Default: on — the scalar path hand-walks Q·Kᵀ and P·V at a fraction
    /// of the matrix unit's throughput. A switch rather than a permanent
    /// choice: the matrix path depends on `simdgroup_matrix<T,8,8>`'s
    /// register layout, which could differ on another machine.
    pub sdpa_mma: bool,

    /// Lanes that share one value row of the gated-delta scan.
    ///
    /// Default: 32 — the scan is latency-bound on cross-lane reductions, and
    /// a full simdgroup per row is the shortest q/k read a lane can do;
    /// narrowing the row hurt, not helped.
    pub gdn_scan_lanes: u32,

    /// Value rows one lane group of that scan walks, sharing the q/k it
    /// read for all of them.
    ///
    /// Default: 4, swept jointly with [`gdn_scan_lanes`](Self::gdn_scan_lanes)
    /// (more lanes always helps at fixed register cost; too few rows don't
    /// amortize the read, too many spend the occupancy that hides latency).
    /// `0` is a real value: no fold is stamped, so `gated_delta_chunked`
    /// runs unfolded.
    pub gdn_scan_rows: u32,

    /// Rows an expert's run must hold before the mixture sorts and batches,
    /// rather than running the routed projections as matvecs.
    ///
    /// Default: 2 (`2 * experts` pairs). A 4-bit mixture is bandwidth-bound,
    /// so batching's win is reading each expert's slice once instead of
    /// once per pair; `0` is a real value ("batch at any width"), used to
    /// force the sorted arm. Measured on the decode rungs specifically,
    /// since prefill's larger pair count is past every threshold either
    /// arm names.
    pub moe_batch_min_per_expert: u32,

    /// The widest row group the vector point folds into one weight fetch.
    ///
    /// Default: 2. `1` disables the fold and restores `quant_qmv.metal`'s
    /// one-row point at every width — the control every measurement here
    /// was taken against. The fold saves load instructions and address/scale
    /// arithmetic (not bandwidth: the vector point is arithmetic-bound), and
    /// a wider fold (4) costs more in occupancy than it saves; `8` is
    /// stamped but unselected pending a machine that wants it.
    ///
    /// Also a numerics knob: folding reassociates the dot product, changing
    /// which 32 partial sums land where. See
    /// [`qmv_rows_packs`](DeviceTuning::qmv_rows_packs).
    pub qmv_rows_max: u32,

    /// Weight packs one thread of the multi-row vector point reads per k
    /// step.
    ///
    /// Default: 1 (the one-row point's own width). This is also the vector
    /// arm's numerical policy, not just a fetch schedule: pack width sets
    /// the dot-product's block size, so `qmv_rows_packs = 2` reproduces the
    /// one-row point's arithmetic bit-for-bit while every other value
    /// reassociates it — enough to move a routed expert's pick on some
    /// inputs. The default stays 1 (fast by default); `2` trades ~11% of an
    /// a4b decode for row-count-invariant arithmetic.
    pub qmv_rows_packs: u32,
}

impl Default for DeviceTuning {
    fn default() -> Self {
        Self {
            qmm_min_batch: 5,
            qmm_min_batch_moe: 8,
            qmm_min_batch_emulated: 12,
            qmm_bn_crossover_tg: 160,
            moe_tile_mid_per: 32,
            moe_tile_wide_per: 1 << 24,
            fp16_qmm: true,
            sdpa_tile_min_rows_per_request: 32,
            sdpa_mma: true,
            gdn_scan_lanes: 32,
            gdn_scan_rows: 4,
            moe_batch_min_per_expert: 2,
            qmv_rows_max: 2,
            qmv_rows_packs: 1,
        }
    }
}

impl DeviceTuning {
    /// The table, read at one device. A family with no entry inherits the
    /// defaults — see the module header for why.
    #[must_use]
    pub fn of(info: DeviceInfo) -> Self {
        let mut t = Self::default();
        match info.apple_family {
            // Apple9: dense crossover named rather than inherited, since the
            // widths measured here (gemma-4-E4B @ +3.7%) say nothing about
            // the default's. Tile crossover moves down: fewer cores (20 vs
            // 32) fill sooner, so the bracket is (64, 96].
            9 => {
                t.qmm_min_batch = 8;
                t.qmm_bn_crossover_tg = 96;
            }
            // Apple8: dense crossover is 8 from this machine's own sweep
            // (won on all four dense checkpoints at 8, lost at 7, and at 6
            // the GEMV still won). Routed crossover stays 12.
            8 => {
                t.qmm_min_batch = 8;
                t.qmm_min_batch_moe = 12;
            }
            _ => {}
        }
        t
    }

    /// The same table with a boot document's answers laid over it.
    #[must_use]
    pub fn with(mut self, over: &Overrides) -> Self {
        macro_rules! lay {
            ($($field:ident),+ $(,)?) => {
                $(if let Some(v) = over.$field { self.$field = v; })+
            };
        }
        lay!(
            qmm_min_batch,
            qmm_min_batch_moe,
            qmm_min_batch_emulated,
            qmm_bn_crossover_tg,
            moe_tile_mid_per,
            moe_tile_wide_per,
            fp16_qmm,
            sdpa_tile_min_rows_per_request,
            sdpa_mma,
            gdn_scan_lanes,
            gdn_scan_rows,
            moe_batch_min_per_expert,
            qmv_rows_max,
            qmv_rows_packs,
        );
        self
    }

    /// The GEMV/GEMM crossover for one projection, given whether the
    /// checkpoint's FFN is routed and whether this weight's format reaches
    /// the FP16 matrix path.
    #[must_use]
    pub const fn qmm_min_batch(&self, routed: bool, fp16_gemm: bool) -> u32 {
        if !fp16_gemm {
            return self.qmm_min_batch_emulated;
        }
        if routed {
            self.qmm_min_batch_moe
        } else {
            self.qmm_min_batch
        }
    }

    /// Whether a bank of this format reaches the FP16 matrix path — the
    /// staged-input GEMM is stamped at 4 bits, group 64, and nowhere else.
    #[must_use]
    pub const fn fp16_gemm_format(&self, bits: u32, group: u32) -> bool {
        self.fp16_qmm && bits == 4 && group == 64
    }
}

/// One boot document's answers: `None` is "keep what the device's table
/// said", and every value — zero included — is a value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overrides {
    pub qmm_min_batch: Option<u32>,
    pub qmm_min_batch_moe: Option<u32>,
    pub qmm_min_batch_emulated: Option<u32>,
    pub qmm_bn_crossover_tg: Option<u32>,
    pub moe_tile_mid_per: Option<u32>,
    pub moe_tile_wide_per: Option<u32>,
    pub fp16_qmm: Option<bool>,
    pub sdpa_tile_min_rows_per_request: Option<u32>,
    pub sdpa_mma: Option<bool>,
    pub gdn_scan_lanes: Option<u32>,
    pub gdn_scan_rows: Option<u32>,
    pub moe_batch_min_per_expert: Option<u32>,
    pub qmv_rows_max: Option<u32>,
    pub qmv_rows_packs: Option<u32>,
}

static DEVICE: OnceLock<DeviceInfo> = OnceLock::new();
static OVERRIDES: OnceLock<Overrides> = OnceLock::new();
static RESOLVED: OnceLock<DeviceTuning> = OnceLock::new();

/// Say what device this process is running on — called once, by the shell
/// that binds it.
///
/// The device and the boot document's overrides arrive from different
/// places in no fixed order, so each is seated independently and folded
/// together only at the first [`current`] call, whose answer then freezes
/// for the process — a table that moved under an in-flight fire would let
/// two dispatches of one step disagree about which kernel they used.
/// Never calling this is supported and lands on the default measurements.
pub fn describe(info: DeviceInfo) -> bool {
    DEVICE.set(info).is_ok()
}

/// Lay a boot document's answers over whatever the device's table says. See
/// [`describe`] for the ordering.
pub fn override_with(over: Overrides) -> bool {
    OVERRIDES.set(over).is_ok()
}

/// The tuning every selection here reads. `Copy`, because a call site that
/// held a borrow would be holding one for the life of the process.
#[must_use]
pub fn current() -> DeviceTuning {
    *RESOLVED.get_or_init(|| {
        let info = DEVICE.get().copied().unwrap_or_default();
        let over = OVERRIDES.get().copied().unwrap_or_default();
        DeviceTuning::of(info).with(&over)
    })
}

