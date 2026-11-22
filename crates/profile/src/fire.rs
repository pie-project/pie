//! What a device profile states for choosing a load's top rung of fires:
//! the device's rates (`[rates]`), a handful of measured constants and two
//! short curves a fire's time is estimated from (`engine::load::
//! FireEstimate`, over `engine::roofline`). They live in the backend's
//! device file (or the user's layer above it); a higher layer replaces a
//! lower one's key by key. `pie tune --rates` (or a load on a device whose
//! profile lacks them) measures them through the backend's own paths, at the
//! shapes `engine::rates` states; a vendor's peak, where one is cited, is a
//! bound in a comment, not a rate.
//!
//! ```toml
//! [rates]
//! launch_ns = 900          # what a dependent dispatch adds to its work
//! # bf16 matmul TFLOP/s by rows, at each width n = k
//! gemm_tflops_by_rows = [
//!   [1024, [[256, 40.0], [8192, 150.0]]],
//!   [4096, [[256, 90.0], [8192, 210.0]]],
//! ]
//! # prefill attention TFLOP/s by keys (one sequence attending itself, d 128)
//! attn_tflops_by_keys = [[512, 60.0], [2048, 200.0], [8192, 400.0]]
//! hbm_gbps = 650.0         # a streaming copy, read + write
//! elementwise_gbps = 450.0 # the per-row ops (rmsnorm, swiglu) as served
//! scan_gelems = 2700.0     # gated-delta scan state elements, G a second
//! few_row_tflops = 4.2     # bf16 matmul TFLOP/s at 4-256 rows, as served
//! workgroups_per_core = 3.3 # workgroups a core holds for a kernel's full rate
//! # each dense kernel's TFLOP/s by rows at each of a model's dense shapes:
//! # [kind, key, n, k, stopped at rows, why, method, [[kernel, code]],
//! #  [[rows, TFLOP/s], ...], [[rows, spread], ...]]
//! kernel_tflops_by_shape_keyed = [[1, 4, 1024, 1024, 0, 0, 1,
//!   [["linear_gemm_gemv_v_bf16", 3735928559]], [[1, 0.3], [256, 2.0]], []]]
//! # a user's file only: the device its measurements were taken on, and
//! # each rate's key: [rate, method, [[kernel, code], ...]]
//! measured_on = "vulkan 10de:2c31 driver 0x91c40000"
//! measured_by = [["launch_ns", 1, [["linear_gemm_tiled_bf16", 4027432458]]]]
//! ```
//!
//! A user's file is the loads' own measured output, and each measurement in
//! it is keyed by what it depends on ([`MeasuredBy`], [`Rates::sift`]): the
//! device, the measuring method's version and the code of each kernel it
//! fired. A load reuses every one whose key matches and measures again only
//! those whose key does not; nothing else of the build (its version, its
//! commit, the day) keys one.
//!
//! Every key is a measurement of the device as serving runs it, but one:
//! `workgroups_per_core` is derived where the device states its occupancy
//! (CUDA, ROCm) and measured only where its API hides it (Vulkan, wgpu,
//! Metal). A key stated at or below zero, or not finite, is no rate: the
//! profile refuses it by name, a rates step that settles at it is refused
//! by name before anything is saved (`engine::rates::unmeasured`), and the
//! roofline divides by none of it (`engine::roofline::Constants::of`).

use std::fmt::Write as _;

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// What a dependent dispatch adds to its work, in nanoseconds.
    pub launch_ns: Option<u64>,
    /// Dense bf16 matmul TFLOP/s by rows at each width (columns and depth
    /// both): `(width, [(rows, TFLOP/s)])`, ascending in width, each curve
    /// strictly ascending in rows.
    pub gemm_tflops_by_rows: Option<Vec<(u32, Vec<(u32, f64)>)>>,
    /// Prefill attention TFLOP/s (the full rectangle's flops, four a query
    /// element a key) by keys, one sequence attending itself: `(keys,
    /// TFLOP/s)`, strictly ascending in keys.
    pub attn_tflops_by_keys: Option<Vec<(u32, f64)>>,
    /// Device memory GB/s, achieved by a streaming copy (read plus write).
    pub hbm_gbps: Option<f64>,
    /// GB/s of reads the last-level caches serve: every workgroup reading a
    /// small set the others read too, as a normed skinny product's
    /// workgroups read its rows (what such a fold's redundant reads cost).
    pub cached_gbps: Option<f64>,
    /// GB/s the per-row ops (a norm, the gated activation) achieve as a
    /// fire runs them.
    pub elementwise_gbps: Option<f64>,
    /// Gated-delta scan state elements a nanosecond (G a second), the conv
    /// before it folded in.
    pub scan_gelems: Option<f64>,
    /// Dense bf16 matmul TFLOP/s (two flops a multiply-add) at decode-shaped
    /// rows, 4 to 256 (below the first row count `gemm_tflops_by_rows`
    /// times), on the kernel serving picks at those rows (on the slang
    /// backends the vector kernels, which a rung takes where the estimate
    /// weighs them faster): the rate its
    /// compute runs at once its weight reads no longer bound it, its flops
    /// counted over the rows its kernel's tile pads to. A matmul of that few
    /// rows costs `launch_ns + max(bytes / hbm_gbps, padded flops /
    /// few_row_tflops)`. Measured, on every backend.
    pub few_row_tflops: Option<f64>,
    /// The workgroups (CUDA's blocks, Metal's threadgroups) each compute
    /// unit (an SM, a CU, an execution core) must hold at once for a
    /// kernel to run at its full rate: a device fact every wave-fill choice
    /// reads, not a kernel's knob. A dispatch of `W` workgroups over `cores`
    /// units of `S = workgroups_per_core · cores` runs at `min(1, W / S)` of
    /// its rate, so a kernel of `work` at rate `r` costs `work / r · max(1,
    /// S / W)`. Derived where the device states its occupancy (CUDA, ROCm:
    /// the resident blocks a multiprocessor holds); measured where its API
    /// hides it (Vulkan, wgpu, Metal), once, at the first load: one
    /// workgroup a head of a kernel at its compute-bound shape timed at `t`
    /// against `t_full`, the same work at the rate its full curve states,
    /// `S = W · t / t_full`, over the cores. A real number of workgroups a
    /// unit.
    pub workgroups_per_core: Option<f64>,
    /// How much slower the device ran than its own measurements said, as
    /// its rates step measured: the largest of the worst ratio of a frame's
    /// time to the same step's frame before it, the clock's swing from fire
    /// to fire, and the drop of its settled clock under sustained load (the
    /// warm-up's cube rotated again once every rate is timed). The margin a
    /// serving step's estimate is held to a stated bound on one GPU
    /// submission at (`engine_gpu::submit`). At least 1. Measured, and read,
    /// only where the platform states such a bound (Android); not one of
    /// [`KEYS`], which the estimate weighs a fire by.
    pub submit_worst_ratio: Option<f64>,
    /// The workgroups the device holds at once for a kernel's full rate,
    /// the whole device's, as its rates step measured them: one workgroup
    /// of a compute-bound kernel runs at `1 / S` of the kernel's full rate.
    /// What a guest step's grid strides over, and where the device states
    /// no count of its compute units (wgpu, a Vulkan driver that reports
    /// none) what sizes a fire's work to it. Not one of [`KEYS`], which the
    /// estimate weighs a fire by.
    pub resident_workgroups: Option<f64>,
    /// Prefill attention TFLOP/s (the rectangle's flops, four a query
    /// element a key, as [`Rates::attn_tflops_by_keys`] counts them) of a
    /// step's rows against a longer prefix: `(rows, [(keys, TFLOP/s)])`, the
    /// rows a prompt step fires at its rung attending `keys` keys of which
    /// all but their own are held before them, ascending in rows, each curve
    /// strictly ascending in keys. A square self-attending prompt skips half
    /// its rectangle causally, a prefixed step almost none of it, so the
    /// square curve reads such a step's attention as up to twice as fast as
    /// it runs. Measured, and read, only where the platform states a bound
    /// on one submission (`engine_gpu::submit`); not one of [`KEYS`].
    pub attn_tflops_by_rows_keys: Option<Vec<(u32, Vec<(u32, f64)>)>>,
    /// Each decode product's time on each kernel that can serve it, measured
    /// in place (a captured burst over the model's own layer weights, as a
    /// step meets them), keyed by the product's shape: what a backend picks
    /// between candidate kernels by, where a threshold would otherwise stand.
    /// Entries accumulate load to load, a load measuring only the shapes the
    /// table lacks. Not a roofline rate: no load requires it.
    pub product_ns: Option<Vec<ProductTime>>,
    /// Each dense matmul kernel's bf16 TFLOP/s by rows at each dense shape
    /// of the models loaded on the device ([`KernelCurve`]), its flops at the
    /// rows it lands over its time (every weight pass it makes in it): what
    /// the slang backends choose a rung's dense kernel by, each kernel
    /// weighed at the model's own planes. Entries accumulate load to load, a
    /// load measuring only the shapes and kernels the table lacks to its rows.
    /// Measured; not one of [`KEYS`] (a load measures it once its model's
    /// shapes are known).
    pub kernel_tflops_by_shape: Option<Vec<KernelCurve>>,
    /// The dense kernels each load's settle took in context over the
    /// curves' picks ([`DENSE_SETTLED`]), one entry per key (a model's
    /// planes, curves and rungs: [`DenseSettled::of`]), many a device: what
    /// a later load of the same curves, model and kernels takes again
    /// without walking a fire. A new model's settle joins the others, never
    /// replaces them. Measured; empty where none is stored.
    pub dense_settled: Vec<DenseSettled>,
    /// The device a user's file's measurements were taken on
    /// ([`MEASURED_ON`]): its vendor, device id and driver, as the backend
    /// states them ([`Now::on`]). A file measured on another device (or
    /// driver) has every measurement measured again ([`Rates::sift`]).
    pub measured_on: Option<String>,
    /// Each measured rate's key ([`MEASURED_BY`]): the rate's name and what
    /// it was measured by. A rate a user's file states without one is
    /// measured again ([`Rates::sift`]).
    pub measured_by: Option<Vec<(String, MeasuredBy)>>,
}

/// What one stored measurement was measured by, beside the device it was
/// measured on ([`Rates::measured_on`]): the version of its measuring method
/// (bumped only when the method changes, stated beside the measuring code)
/// and each kernel it fired with its code's fingerprint in the build that
/// measured it ([`Now::code`]). A measurement whose method or any of whose
/// kernels' code differs from this build's is measured again; nothing else
/// of the build (its version, its commit, the day) keys it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MeasuredBy {
    pub method: u32,
    pub kernels: Vec<(String, u32)>,
}

impl MeasuredBy {
    /// Why a measurement by `self` is not valid under `now`, `None` where
    /// it is: its method's version moved, or a kernel's code changed or left
    /// the build.
    #[must_use]
    pub fn stale(&self, what: &str, now: &dyn Now) -> Option<String> {
        match now.method(what) {
            None => return Some(format!("`{what}` is measured by no method of this build")),
            Some(method) if method != self.method => {
                return Some(format!(
                    "its measuring method is version {method} now, {} when stored",
                    self.method
                ));
            }
            Some(_) => {}
        }
        for (kernel, code) in &self.kernels {
            match now.code(kernel) {
                None => return Some(format!("kernel `{kernel}` is not in this build")),
                Some(now) if now != *code => {
                    return Some(format!(
                        "kernel `{kernel}`'s code changed ({code:#010x} stored, {now:#010x} now)"
                    ));
                }
                Some(_) => {}
            }
        }
        None
    }

    /// The key as the load log states it: `method v<n>, <kernel> <code>, ...`.
    #[must_use]
    pub fn said(&self) -> String {
        let kernels: Vec<String> = self
            .kernels
            .iter()
            .map(|(k, c)| format!("{k} {c:#010x}"))
            .collect();
        format!("method v{}, kernels [{}]", self.method, kernels.join(", "))
    }
}

/// What a stored measurement's key is checked against: this build on this
/// device. A backend that stores measurements states it.
pub trait Now {
    /// The device as its measurements depend on it: vendor, device id and
    /// driver version.
    fn on(&self) -> String;
    /// The version of the method that measures `what` (a `[rates]` key), or
    /// `None` where this build measures no such thing.
    fn method(&self, what: &str) -> Option<u32>;
    /// The fingerprint of `kernel`'s compiled code in this build, `None`
    /// where the build carries no such kernel.
    fn code(&self, kernel: &str) -> Option<u32>;
}

/// The fingerprint of `bytes` a stored key holds: FNV-1a folded to 32 bits,
/// the same on every build and toolchain (std's hasher is not).
#[must_use]
pub fn fingerprint(bytes: &[u8]) -> u32 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash ^ (hash >> 32)) as u32
}

/// A settle's outcome, keyed: `method`, the version of the settle's
/// measuring method; `of`, a fingerprint of what it weighed (the curves, the
/// model's planes, the load's rungs, lanes and kernel choice); `code`, a
/// fingerprint of the code of every kernel the load's served step fires
/// (what each weighed step ran); and each row range it decided, `[bottom,
/// top, replaced, vector, wide, tier]`: 1 where it replaced the curves'
/// kernel with the one named (the tier `kernels_slang::Tier` as a number), 0
/// where it kept theirs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseSettled {
    pub method: u32,
    pub of: u32,
    pub code: u32,
    pub ranges: Vec<[u32; 6]>,
}

/// One dense kernel's measured curve at one shape
/// ([`Rates::kernel_tflops_by_shape`]): the kernel (`kind`
/// [`KernelCurve::VECTOR`], `key` the rows a workgroup carries, or
/// [`KernelCurve::TILE`], `key` its row tile at the shape), the plane's `n x
/// k`, its TFLOP/s by rows ascending, and where it stopped short of the rows
/// it was asked to reach, if it did (`stop`: the rows not timed, and why,
/// [`KernelCurve::BOUND`]: that fire's estimate passed the device's bound on
/// one submission; [`KernelCurve::BUFFER`]: an operand passed the device's
/// largest buffer; [`KernelCurve::BEATEN`]: a vector kernel ran slower than
/// twice a tiled kernel at the same rows, beaten at every rows past them).
/// A stopped curve is final: no load times it again.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelCurve {
    pub kind: u32,
    pub key: u32,
    pub n: u32,
    pub k: u32,
    pub curve: Vec<(u32, f64)>,
    pub stop: Option<(u32, u32)>,
    /// Each point's spread over its repeats, `(max - min) / median`, by
    /// rows as `curve`: what a choice between two kernels may call a
    /// difference (none stated: no spread measured).
    pub spread: Vec<(u32, f64)>,
    /// What the curve was measured by: the curve's method and its kernel's
    /// code ([`MeasuredBy`]).
    pub by: MeasuredBy,
}

impl KernelCurve {
    pub const VECTOR: u32 = 1;
    pub const TILE: u32 = 2;
    pub const BOUND: u32 = 1;
    pub const BUFFER: u32 = 2;
    pub const BEATEN: u32 = 3;
    /// Every sample at the rows was dropped (a fire timed below its
    /// launches): no measurement there, and nothing past it.
    pub const UNTIMED: u32 = 4;

    /// What the curve measures: its kernel and shape.
    #[must_use]
    pub const fn id(&self) -> (u32, u32, u32, u32) {
        (self.kind, self.key, self.n, self.k)
    }

    /// The curve as a load's log names it: `<kind>/<key> at <n>x<k>`.
    #[must_use]
    pub fn name(&self) -> String {
        format!("{}/{} at {}x{}", self.kind, self.key, self.n, self.k)
    }

    /// Whether a load of up to `rows` needs nothing more of it: it reaches
    /// them, or stopped for a reason it records.
    #[must_use]
    pub fn serves(&self, rows: u32) -> bool {
        self.stop.is_some() || self.curve.last().is_some_and(|&(at, _)| at >= rows)
    }
}

/// One product's measured time on one kernel ([`Rates::product_ns`]): the
/// product's kind, bank rows (`2n` for a packed `[gate | up]` bank), K and
/// rows, the kernel (the backend's code), and the median ns of a launch with
/// the spread of its repeats (max − min, ns), which bounds what a choice
/// between two kernels may call a difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductTime {
    pub kind: u32,
    pub rows: u32,
    pub k: u32,
    pub m: u32,
    pub kernel: u32,
    pub ns: f64,
    pub spread: f64,
}

impl ProductTime {
    /// What the entry measures: everything but its time.
    #[must_use]
    pub const fn key(&self) -> (u32, u32, u32, u32, u32) {
        (self.kind, self.rows, self.k, self.m, self.kernel)
    }
}

/// Every roofline key `[rates]` holds, in the order a profile writes them
/// ([`Rates::product_ns`] is not one: no load requires it).
pub const KEYS: [&str; 9] = [
    "launch_ns",
    "gemm_tflops_by_rows",
    "attn_tflops_by_keys",
    "hbm_gbps",
    "cached_gbps",
    "elementwise_gbps",
    "scan_gelems",
    "few_row_tflops",
    "workgroups_per_core",
];

/// The `[rates]` key of the dense kernels' curves at a model's planes
/// (`Rates::kernel_tflops_by_shape`), each curve keyed by what measured it
/// ([`KernelCurve::by`]). Curves an older pie wrote under
/// `kernel_tflops_by_shape` were timed with a timestamp pair around every
/// dispatch, which on a tiler ran its few-row points ~10x long, and those
/// under `kernel_tflops_by_shape_spanned` carry no key: neither key is read
/// any more, so a user file stating one is measured again.
pub const KERNEL_CURVES: &str = "kernel_tflops_by_shape_keyed";

/// The key an older pie wrote its dense kernels' curves under, timed per
/// dispatch ([`KERNEL_CURVES`]).
pub const KERNEL_CURVES_PER_DISPATCH: &str = "kernel_tflops_by_shape";

/// The `[rates]` key of [`Rates::dense_settled`]: a list of `[method, of,
/// code, [[bottom, top, replaced, vector, wide, tier], ...]]`, one per key. The settle an older
/// pie stored under `dense_settled`, keyed by its version, is no longer read.
pub const DENSE_SETTLED: &str = "dense_settled_keyed";

/// The `[rates]` key of [`Rates::measured_on`].
pub const MEASURED_ON: &str = "measured_on";

/// The `[rates]` key of [`Rates::measured_by`]: `[[rate, method, [[kernel,
/// code], ...]], ...]`.
pub const MEASURED_BY: &str = "measured_by";

/// The rates a stored key covers ([`Rates::measured_by`]): every `[rates]`
/// key of one value a load measures. The curves and the settle carry their
/// own keys.
pub const KEYED: [&str; 13] = [
    "launch_ns",
    "gemm_tflops_by_rows",
    "attn_tflops_by_keys",
    "hbm_gbps",
    "cached_gbps",
    "elementwise_gbps",
    "scan_gelems",
    "few_row_tflops",
    "workgroups_per_core",
    SUBMIT_WORST_RATIO,
    RESIDENT_WORKGROUPS,
    ATTN_BY_ROWS_KEYS,
    "product_ns",
];

/// The `[rates]` key of [`Rates::submit_worst_ratio`]: what a load on a
/// platform that states a bound on one submission measures beside the
/// estimate's [`KEYS`].
pub const SUBMIT_WORST_RATIO: &str = "submit_worst_ratio";

/// The `[rates]` key of [`Rates::attn_tflops_by_rows_keys`]: measured beside
/// [`SUBMIT_WORST_RATIO`] where the platform states a submit bound.
pub const ATTN_BY_ROWS_KEYS: &str = "attn_tflops_by_rows_keys";

/// The `[rates]` key of [`Rates::resident_workgroups`]: measured beside the
/// estimate's [`KEYS`] on the slang backends.
pub const RESIDENT_WORKGROUPS: &str = "resident_workgroups";

impl Rates {
    /// `self` laid over `below`: each key `self` states replaces `below`'s.
    #[must_use]
    pub fn over(self, below: Rates) -> Rates {
        Rates {
            launch_ns: self.launch_ns.or(below.launch_ns),
            gemm_tflops_by_rows: self.gemm_tflops_by_rows.or(below.gemm_tflops_by_rows),
            attn_tflops_by_keys: self.attn_tflops_by_keys.or(below.attn_tflops_by_keys),
            hbm_gbps: self.hbm_gbps.or(below.hbm_gbps),
            cached_gbps: self.cached_gbps.or(below.cached_gbps),
            elementwise_gbps: self.elementwise_gbps.or(below.elementwise_gbps),
            scan_gelems: self.scan_gelems.or(below.scan_gelems),
            few_row_tflops: self.few_row_tflops.or(below.few_row_tflops),
            workgroups_per_core: self.workgroups_per_core.or(below.workgroups_per_core),
            submit_worst_ratio: self.submit_worst_ratio.or(below.submit_worst_ratio),
            resident_workgroups: self.resident_workgroups.or(below.resident_workgroups),
            attn_tflops_by_rows_keys: self
                .attn_tflops_by_rows_keys
                .or(below.attn_tflops_by_rows_keys),
            // Entry by entry: a load's shapes join those already measured,
            // its own replacing a shape both state.
            product_ns: match (self.product_ns, below.product_ns) {
                (Some(mine), Some(theirs)) => {
                    let mut all: Vec<ProductTime> = theirs
                        .into_iter()
                        .filter(|t| mine.iter().all(|m| m.key() != t.key()))
                        .collect();
                    all.extend(mine);
                    all.sort_by_key(ProductTime::key);
                    Some(all)
                }
                (mine, theirs) => mine.or(theirs),
            },
            // Entry by entry, as `product_ns`: a load's curves join those
            // already measured, its own replacing one both state.
            kernel_tflops_by_shape: match (
                self.kernel_tflops_by_shape,
                below.kernel_tflops_by_shape,
            ) {
                (Some(mine), Some(theirs)) => {
                    let mut all: Vec<KernelCurve> = theirs
                        .into_iter()
                        .filter(|t| mine.iter().all(|m| m.id() != t.id()))
                        .collect();
                    all.extend(mine);
                    all.sort_by_key(KernelCurve::id);
                    Some(all)
                }
                (mine, theirs) => mine.or(theirs),
            },
            // Entry by entry, by its key: a load's settle joins the others
            // (another model's), its own replacing one of the same key.
            dense_settled: {
                let mut all: Vec<DenseSettled> = below
                    .dense_settled
                    .into_iter()
                    .filter(|t| self.dense_settled.iter().all(|m| m.of != t.of))
                    .collect();
                all.extend(self.dense_settled);
                all.sort_by_key(|d| d.of);
                all
            },
            measured_on: self.measured_on.or(below.measured_on),
            // Entry by entry: a rate's key goes with its value.
            measured_by: match (self.measured_by, below.measured_by) {
                (Some(mine), Some(theirs)) => {
                    let mut all: Vec<(String, MeasuredBy)> = theirs
                        .into_iter()
                        .filter(|(t, _)| mine.iter().all(|(m, _)| m != t))
                        .collect();
                    all.extend(mine);
                    all.sort_by(|a, b| a.0.cmp(&b.0));
                    Some(all)
                }
                (mine, theirs) => mine.or(theirs),
            },
        }
    }

    /// Whether `key` (one of [`KEYED`]) is stated.
    #[must_use]
    pub fn states(&self, key: &str) -> bool {
        !self.missing_any(key)
    }

    fn missing_any(&self, key: &str) -> bool {
        match key {
            "launch_ns" => self.launch_ns.is_none(),
            "gemm_tflops_by_rows" => self.gemm_tflops_by_rows.is_none(),
            "attn_tflops_by_keys" => self.attn_tflops_by_keys.is_none(),
            "hbm_gbps" => self.hbm_gbps.is_none(),
            "cached_gbps" => self.cached_gbps.is_none(),
            "elementwise_gbps" => self.elementwise_gbps.is_none(),
            "scan_gelems" => self.scan_gelems.is_none(),
            "few_row_tflops" => self.few_row_tflops.is_none(),
            "workgroups_per_core" => self.workgroups_per_core.is_none(),
            SUBMIT_WORST_RATIO => self.submit_worst_ratio.is_none(),
            RESIDENT_WORKGROUPS => self.resident_workgroups.is_none(),
            ATTN_BY_ROWS_KEYS => self.attn_tflops_by_rows_keys.is_none(),
            "product_ns" => self.product_ns.is_none(),
            KERNEL_CURVES => self.kernel_tflops_by_shape.is_none(),
            DENSE_SETTLED => self.dense_settled.is_empty(),
            _ => true,
        }
    }

    /// `key` (one of [`KEYED`]) unstated, its stored key with it.
    pub fn unstate(&mut self, key: &str) {
        match key {
            "launch_ns" => self.launch_ns = None,
            "gemm_tflops_by_rows" => self.gemm_tflops_by_rows = None,
            "attn_tflops_by_keys" => self.attn_tflops_by_keys = None,
            "hbm_gbps" => self.hbm_gbps = None,
            "cached_gbps" => self.cached_gbps = None,
            "elementwise_gbps" => self.elementwise_gbps = None,
            "scan_gelems" => self.scan_gelems = None,
            "few_row_tflops" => self.few_row_tflops = None,
            "workgroups_per_core" => self.workgroups_per_core = None,
            SUBMIT_WORST_RATIO => self.submit_worst_ratio = None,
            RESIDENT_WORKGROUPS => self.resident_workgroups = None,
            ATTN_BY_ROWS_KEYS => self.attn_tflops_by_rows_keys = None,
            "product_ns" => self.product_ns = None,
            _ => {}
        }
        if let Some(by) = &mut self.measured_by {
            by.retain(|(rate, _)| rate != key);
        }
    }

    /// The key `key` was stored under, where the file states one.
    #[must_use]
    pub fn key_of(&self, key: &str) -> Option<&MeasuredBy> {
        self.measured_by
            .iter()
            .flatten()
            .find(|(rate, _)| rate == key)
            .map(|(_, by)| by)
    }

    /// A user's file's measurements (`self`, its `[rates]` alone) checked
    /// against `now`, this build on this device ([`Rates::sift_as`], said as
    /// `stored`).
    pub fn sift(&mut self, now: &dyn Now) -> Vec<String> {
        self.sift_as(now, "stored")
    }

    /// Stored measurements (`self`, a file's `[rates]` alone: the user's,
    /// `what` "stored", or a committed measured profile's, "committed")
    /// checked against `now`, this build on this device: each whose key
    /// matches is kept, to be taken with no measurement; each whose key
    /// does not (or that states none) is dropped, by itself. Measured on
    /// another device (or driver), every one is dropped. Each drop is said,
    /// a line each, with why; what is kept is said once the layers are
    /// joined ([`Rates::whence`]), by the layer that decides it.
    pub fn sift_as(&mut self, now: &dyn Now, what: &str) -> Vec<String> {
        let mut said = Vec::new();
        let on = now.on();
        let stored = KEYED.iter().any(|key| self.states(key))
            || self.kernel_tflops_by_shape.is_some()
            || !self.dense_settled.is_empty();
        if !stored {
            said.push(format!("{what} measurements: none for `{on}`"));
            *self = Rates::default();
            return said;
        }
        if self.measured_on.as_deref() != Some(on.as_str()) {
            said.push(format!(
                "{what} measurements: taken on `{}`, this device is `{on}`: none is taken",
                self.measured_on.as_deref().unwrap_or("no device stated")
            ));
            *self = Rates::default();
            return said;
        }
        for key in KEYED {
            if !self.states(key) {
                continue;
            }
            let why = match self.key_of(key) {
                None => Some("stored without a key".to_string()),
                Some(by) => by.stale(key, now),
            };
            if let Some(why) = why {
                said.push(format!("{what} `{key}`: not taken, {why}"));
                self.unstate(key);
            }
        }
        // A key whose rate is not stated keys nothing.
        let stated: Vec<&str> = KEYED.into_iter().filter(|key| self.states(key)).collect();
        self.measured_by = self
            .measured_by
            .take()
            .map(|by| {
                by.into_iter()
                    .filter(|(rate, _)| stated.contains(&rate.as_str()))
                    .collect::<Vec<_>>()
            })
            .filter(|by| !by.is_empty());
        if let Some(curves) = self.kernel_tflops_by_shape.take() {
            let mut kept = Vec::with_capacity(curves.len());
            for curve in curves {
                match curve.by.stale(KERNEL_CURVES, now) {
                    None => kept.push(curve),
                    Some(why) => said.push(format!(
                        "{what} curve {}: not taken, {why}",
                        curve.name()
                    )),
                }
            }
            self.kernel_tflops_by_shape = (!kept.is_empty()).then_some(kept);
        }
        for settled in std::mem::take(&mut self.dense_settled) {
            match now.method(DENSE_SETTLED) {
                Some(method) if method == settled.method => {
                    self.dense_settled.push(settled);
                }
                now => said.push(format!(
                    "{what} dense settle {:#010x}: not taken, its method is version {} now, {} \
                     when stored",
                    settled.of,
                    now.map_or_else(|| "none".to_string(), |m| m.to_string()),
                    settled.method
                )),
            }
        }
        said
    }

    /// Each entry `self` (the rates a load takes, the user's stored over a
    /// committed profile's, each sifted: [`Rates::sift_as`]) states by its
    /// key, said once by the layer that decides it: `stored` (`user`, the
    /// user's sifted `[rates]`, from `user_from`) where the user's file
    /// states it, else `committed` (from `committed_from`). The dense
    /// settle's key is checked by the settle itself, which says which it
    /// takes.
    #[must_use]
    pub fn whence(&self, user: &Rates, user_from: &str, committed_from: &str) -> Vec<String> {
        let mut said = Vec::new();
        for key in KEYED {
            if !self.states(key) {
                continue;
            }
            let Some(by) = self.key_of(key) else {
                continue;
            };
            let from = if user.states(key) {
                format!("stored ({user_from})")
            } else {
                format!("committed ({committed_from})")
            };
            said.push(format!("`{key}`: {from}, its key matches: {}", by.said()));
        }
        for curve in self.kernel_tflops_by_shape.iter().flatten() {
            let mine = user
                .kernel_tflops_by_shape
                .iter()
                .flatten()
                .any(|c| c.id() == curve.id());
            let from = if mine {
                format!("stored ({user_from})")
            } else {
                format!("committed ({committed_from})")
            };
            said.push(format!(
                "curve {}: {from}, its key matches: {}",
                curve.name(),
                curve.by.said()
            ));
        }
        said
    }

    /// The first entry of `self` a committed measured profile could not
    /// take ([`crate::Strict::Measured`]): no device stated, or a rate (one
    /// of [`KEYED`]) stored without a key. `None` where every entry is keyed.
    #[must_use]
    pub fn unkeyed(&self) -> Option<String> {
        if self.measured_on.is_none() {
            return Some(format!("`rates.{MEASURED_ON}` names no device"));
        }
        KEYED
            .iter()
            .find(|key| self.states(key) && self.key_of(key).is_none())
            .map(|key| format!("`rates.{key}` is stored without a key"))
    }

    /// `self` in the order a committed measured profile writes it: each
    /// rate's key by its rate, each curve by its kernel and shape, so one
    /// device's export writes the same text whatever order its loads stored
    /// them in.
    #[must_use]
    pub fn ordered(mut self) -> Rates {
        if let Some(by) = &mut self.measured_by {
            by.sort_by(|a, b| a.0.cmp(&b.0));
        }
        if let Some(curves) = &mut self.kernel_tflops_by_shape {
            curves.sort_by_key(KernelCurve::id);
        }
        if let Some(product) = &mut self.product_ns {
            product.sort_by_key(ProductTime::key);
        }
        self.dense_settled.sort_by_key(|d| d.of);
        self
    }

    fn is_empty(&self) -> bool {
        *self == Rates::default()
    }

    /// Only `keys` of `self` (as [`Rates::missing`] names them), the rest
    /// unstated.
    #[must_use]
    pub fn only(&self, keys: &[&str]) -> Rates {
        let has = |key: &str| keys.contains(&key);
        Rates {
            launch_ns: self.launch_ns.filter(|_| has("launch_ns")),
            gemm_tflops_by_rows: self
                .gemm_tflops_by_rows
                .clone()
                .filter(|_| has("gemm_tflops_by_rows")),
            attn_tflops_by_keys: self
                .attn_tflops_by_keys
                .clone()
                .filter(|_| has("attn_tflops_by_keys")),
            hbm_gbps: self.hbm_gbps.filter(|_| has("hbm_gbps")),
            cached_gbps: self.cached_gbps.filter(|_| has("cached_gbps")),
            elementwise_gbps: self.elementwise_gbps.filter(|_| has("elementwise_gbps")),
            scan_gelems: self.scan_gelems.filter(|_| has("scan_gelems")),
            few_row_tflops: self.few_row_tflops.filter(|_| has("few_row_tflops")),
            workgroups_per_core: self
                .workgroups_per_core
                .filter(|_| has("workgroups_per_core")),
            submit_worst_ratio: self.submit_worst_ratio.filter(|_| has(SUBMIT_WORST_RATIO)),
            resident_workgroups: self
                .resident_workgroups
                .filter(|_| has(RESIDENT_WORKGROUPS)),
            attn_tflops_by_rows_keys: self
                .attn_tflops_by_rows_keys
                .clone()
                .filter(|_| has(ATTN_BY_ROWS_KEYS)),
            product_ns: self.product_ns.clone().filter(|_| has("product_ns")),
            kernel_tflops_by_shape: self
                .kernel_tflops_by_shape
                .clone()
                .filter(|_| has(KERNEL_CURVES)),
            dense_settled: if has(DENSE_SETTLED) {
                self.dense_settled.clone()
            } else {
                Vec::new()
            },
            measured_on: self.measured_on.clone(),
            measured_by: self
                .measured_by
                .as_ref()
                .map(|by| by.iter().filter(|(rate, _)| has(rate)).cloned().collect()),
        }
    }

    /// The keys of `keys` (a backend's estimate's: those its roofline weighs
    /// a fire by, of [`KEYS`]) this states none of: what a load measures
    /// before it is ready (`engine::rates::rates_at_load`).
    #[must_use]
    pub fn missing(&self, keys: &[&'static str]) -> Vec<&'static str> {
        keys.iter()
            .copied()
            .filter(|key| self.missing_any(key))
            .collect()
    }

    /// Whether every key of `keys` is stated.
    #[must_use]
    pub fn complete(&self, keys: &[&'static str]) -> bool {
        self.missing(keys).is_empty()
    }

    /// `[rates]` as profile TOML.
    #[must_use]
    pub fn toml(&self) -> String {
        let mut out = String::from("[rates]\n");
        if let Some(v) = self.launch_ns {
            let _ = writeln!(out, "launch_ns = {v}");
        }
        let keyed = |out: &mut String, key: &str, v: &Option<Vec<(u32, Vec<(u32, f64)>)>>| {
            if let Some(keyed) = v {
                let curves: Vec<String> = keyed
                    .iter()
                    .map(|(w, c)| {
                        let points: Vec<String> =
                            c.iter().map(|(x, r)| format!("[{x}, {r:?}]")).collect();
                        format!("  [{w}, [{}]],\n", points.join(", "))
                    })
                    .collect();
                let _ = writeln!(out, "{key} = [\n{}]", curves.concat());
            }
        };
        keyed(&mut out, "gemm_tflops_by_rows", &self.gemm_tflops_by_rows);
        let mut curve = |key: &str, v: &Option<Vec<(u32, f64)>>| {
            if let Some(points) = v {
                let points: Vec<String> = points
                    .iter()
                    .map(|(x, r)| format!("[{x}, {r:?}]"))
                    .collect();
                let _ = writeln!(out, "{key} = [{}]", points.join(", "));
            }
        };
        curve("attn_tflops_by_keys", &self.attn_tflops_by_keys);
        let mut float = |key: &str, v: Option<f64>| {
            if let Some(v) = v {
                let _ = writeln!(out, "{key} = {v:?}");
            }
        };
        float("hbm_gbps", self.hbm_gbps);
        float("cached_gbps", self.cached_gbps);
        float("elementwise_gbps", self.elementwise_gbps);
        float("scan_gelems", self.scan_gelems);
        float("few_row_tflops", self.few_row_tflops);
        float("workgroups_per_core", self.workgroups_per_core);
        float(SUBMIT_WORST_RATIO, self.submit_worst_ratio);
        float(RESIDENT_WORKGROUPS, self.resident_workgroups);
        if let Some(times) = &self.product_ns {
            let rows: Vec<String> = times
                .iter()
                .map(|t| {
                    format!(
                        "  [{}, {}, {}, {}, {}, {:?}, {:?}],\n",
                        t.kind, t.rows, t.k, t.m, t.kernel, t.ns, t.spread
                    )
                })
                .collect();
            let _ = writeln!(out, "product_ns = [\n{}]", rows.concat());
        }
        keyed(&mut out, ATTN_BY_ROWS_KEYS, &self.attn_tflops_by_rows_keys);
        if let Some(curves) = &self.kernel_tflops_by_shape {
            let rows: Vec<String> = curves
                .iter()
                .map(|c| {
                    let points: Vec<String> = c
                        .curve
                        .iter()
                        .map(|(x, r)| format!("[{x}, {r:?}]"))
                        .collect();
                    let (at, why) = c.stop.unwrap_or((0, 0));
                    let spread: Vec<String> = c
                        .spread
                        .iter()
                        .map(|(x, f)| format!("[{x}, {f:?}]"))
                        .collect();
                    format!(
                        "  [{}, {}, {}, {}, {at}, {why}, {}, [{}], [{}]],\n",
                        c.kind,
                        c.key,
                        c.n,
                        c.k,
                        by_toml(&c.by),
                        points.join(", "),
                        spread.join(", ")
                    )
                })
                .collect();
            let _ = writeln!(out, "{KERNEL_CURVES} = [\n{}]", rows.concat());
        }
        if !self.dense_settled.is_empty() {
            let mut settles: Vec<&DenseSettled> = self.dense_settled.iter().collect();
            settles.sort_by_key(|d| d.of);
            let rows: Vec<String> = settles
                .iter()
                .map(|settled| {
                    let ranges: Vec<String> = settled
                        .ranges
                        .iter()
                        .map(|r| {
                            format!(
                                "[{}, {}, {}, {}, {}, {}]",
                                r[0], r[1], r[2], r[3], r[4], r[5]
                            )
                        })
                        .collect();
                    format!(
                        "  [{}, {}, {}, [{}]],\n",
                        settled.method,
                        settled.of,
                        settled.code,
                        ranges.join(", ")
                    )
                })
                .collect();
            let _ = writeln!(out, "{DENSE_SETTLED} = [\n{}]", rows.concat());
        }
        if let Some(on) = &self.measured_on {
            let _ = writeln!(out, "{MEASURED_ON} = {}", toml_string(on));
        }
        if let Some(by) = &self.measured_by {
            let rows: Vec<String> = by
                .iter()
                .map(|(rate, by)| format!("  [{}, {}],\n", toml_string(rate), by_toml(by)))
                .collect();
            let _ = writeln!(out, "{MEASURED_BY} = [\n{}]", rows.concat());
        }
        out
    }
}

/// `by` as a stored key writes it: `method, [[kernel, code], ...]`.
fn by_toml(by: &MeasuredBy) -> String {
    let kernels: Vec<String> = by
        .kernels
        .iter()
        .map(|(k, c)| format!("[{}, {c}]", toml_string(k)))
        .collect();
    format!("{}, [{}]", by.method, kernels.join(", "))
}

/// `text` as a TOML basic string.
fn toml_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `curve` (`(x, y)`, ascending in `x`) at `x`: linear in `log x` between
/// its points, clamped at its ends. `None` for an empty curve.
#[must_use]
pub fn interpolate(curve: &[(u32, f64)], x: f64) -> Option<f64> {
    let (&(x0, y0), &(x1, y1)) = (curve.first()?, curve.last()?);
    if x <= f64::from(x0) {
        return Some(y0);
    }
    if x >= f64::from(x1) {
        return Some(y1);
    }
    let at = curve.partition_point(|&(p, _)| f64::from(p) <= x);
    let ((lo, a), (hi, b)) = (curve[at - 1], curve[at]);
    let t = (x.ln() - f64::from(lo).ln()) / (f64::from(hi).ln() - f64::from(lo).ln());
    Some(a + (b - a) * t)
}

/// A matmul's rate at `m` rows of an `n x k` plane, read off `by_rows` (rows
/// curves at each square width, ascending in width, in any unit of rate):
/// each width's curve at the timed rows at or below `m` (its first below
/// that), in `log width` at `sqrt(n k)` between the widths. Past the widest
/// width its last span's slope in log width carries on (a wider plane fills
/// more tiles), never below the widest's rate; below the narrowest, its rate.
/// Infinite for no curves.
#[must_use]
pub fn gemm_rate(by_rows: &[(u32, Vec<(u32, f64)>)], m: u32, n: u32, k: u32) -> f64 {
    let at = |curve: &[(u32, f64)]| -> f64 {
        curve
            .iter()
            .rev()
            .find(|&&(rows, _)| rows <= m)
            .or(curve.first())
            .map_or(f64::INFINITY, |&(_, r)| r)
    };
    let width: Vec<(u32, f64)> = by_rows.iter().map(|(w, c)| (*w, at(c))).collect();
    let square = (f64::from(n) * f64::from(k)).sqrt();
    if let [.., (w0, r0), (w1, r1)] = width[..]
        && square > f64::from(w1)
    {
        let slope = (r1 - r0) / (f64::from(w1).ln() - f64::from(w0).ln());
        return (r1 + slope * (square.ln() - f64::from(w1).ln())).max(r1);
    }
    interpolate(&width, square).unwrap_or(f64::INFINITY)
}

/// `(tokens, ns)` of a fire at each rung width, ascending in tokens: what a
/// check of the estimate measures. Never a profile's.
pub type FireTable = Vec<(u32, u64)>;

/// What a profile states beside its decisions: its rates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tuned {
    pub rates: Rates,
    /// The rates were measured on the device by this load, not read from
    /// a layer (`engine::rates::rates_at_load`): no layer stated them all.
    pub measured_this_load: bool,
    /// Where this load saved what it measured (the user's profile of the
    /// device), for later loads to read; `None` where it kept them for
    /// itself alone (no profiles directory: a browser's masked adapter).
    pub saved_to: Option<String>,
    /// A frame of this load's rates step stopped at the device's bound on
    /// one submission (`max_submit_ms`) before it settled
    /// (`engine::rates::timed_frames`).
    pub frame_cut: bool,
    /// A checked-in layer this device takes that states a `[rates]` key
    /// this pie does not read (`<file>: <key>`): the layer was refused, and
    /// a load refuses by it rather than run without the file.
    pub refused: Option<String>,
    /// The dense settle a committed measured profile states
    /// ([`crate::Source::Committed`]), sifted by its method: the settle
    /// takes it where the user's stored one's key does not match and its
    /// own does, never copied into the user's file.
    pub committed_settle: Vec<DenseSettled>,
    /// The `[rates]` keys, curves and settle this load took no measurement
    /// of under `[engine] measure = false`, each served by the engine's
    /// estimate and named where it is (`engine::rates::Measuring::Skip`).
    pub skipped: Vec<String>,
}

impl Tuned {
    /// `upper` laid over `self`.
    pub fn lay(&mut self, upper: &Tuned) {
        self.rates = upper.rates.clone().over(std::mem::take(&mut self.rates));
        self.measured_this_load |= upper.measured_this_load;
        self.saved_to = upper.saved_to.clone().or(self.saved_to.take());
        self.frame_cut |= upper.frame_cut;
        self.refused = upper.refused.clone().or(self.refused.take());
        if !upper.committed_settle.is_empty() {
            self.committed_settle = upper.committed_settle.clone();
        }
        self.skipped.extend(upper.skipped.iter().cloned());
    }

    /// `[rates]` as profile TOML, nothing where nothing is stated or the
    /// rates were measured this load.
    #[must_use]
    pub fn toml(&self) -> String {
        if self.rates.is_empty() || self.measured_this_load {
            return String::new();
        }
        format!("\n{}", self.rates.toml())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_upper_layer_replaces_key_by_key() {
        let mut low = Tuned {
            rates: Rates {
                gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 1.0)])]),
                hbm_gbps: Some(2.0),
                cached_gbps: None,
                ..Rates::default()
            },
            measured_this_load: false,
            saved_to: None,
            frame_cut: false,
            refused: None,
            ..Tuned::default()
        };
        let up = Tuned {
            rates: Rates {
                hbm_gbps: Some(3.0),
                cached_gbps: None,
                launch_ns: Some(7),
                ..Rates::default()
            },
            measured_this_load: false,
            saved_to: None,
            frame_cut: false,
            refused: None,
            ..Tuned::default()
        };
        low.lay(&up);
        assert_eq!(
            low.rates,
            Rates {
                gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 1.0)])]),
                hbm_gbps: Some(3.0),
                cached_gbps: None,
                launch_ns: Some(7),
                ..Rates::default()
            }
        );
        assert!(!low.rates.complete(&KEYS));
    }

    #[test]
    fn the_block_is_a_few_lines() {
        let rates = Rates {
            launch_ns: Some(904),
            gemm_tflops_by_rows: Some(vec![
                (1024, vec![(256, 40.0), (8192, 150.0)]),
                (4096, vec![(256, 90.5), (8192, 210.0)]),
            ]),
            attn_tflops_by_keys: Some(vec![(512, 60.0), (8192, 400.0)]),
            hbm_gbps: Some(654.4),
            cached_gbps: Some(28800.0),
            elementwise_gbps: Some(450.0),
            scan_gelems: Some(2684.3),
            few_row_tflops: Some(4.2),
            workgroups_per_core: Some(3.3),
            submit_worst_ratio: None,
            resident_workgroups: None,
            attn_tflops_by_rows_keys: None,
            product_ns: None,
            kernel_tflops_by_shape: Some(vec![KernelCurve {
                kind: KernelCurve::VECTOR,
                key: 4,
                n: 1024,
                k: 2048,
                curve: vec![(1, 0.5)],
                stop: Some((512, KernelCurve::BOUND)),
                spread: Vec::new(),
                by: MeasuredBy {
                    method: 1,
                    kernels: vec![("linear_gemm_gemv_v_bf16".into(), 7)],
                },
            }]),
            dense_settled: vec![DenseSettled {
                method: 1,
                of: 77,
                code: 9,
                ranges: vec![[3, 4, 1, 4, 0, 1]],
            }],
            measured_on: Some("vk 10de:2c31 driver 1".into()),
            measured_by: Some(vec![(
                "launch_ns".into(),
                MeasuredBy {
                    method: 2,
                    kernels: vec![("linear_gemm_tiled_bf16".into(), 5)],
                },
            )]),
        };
        assert!(rates.complete(&KEYS));
        assert_eq!(
            rates.only(&KEYS[..6]).missing(&KEYS),
            ["scan_gelems", "few_row_tflops", "workgroups_per_core",]
        );
        assert_eq!(
            rates.toml(),
            "[rates]\nlaunch_ns = 904\ngemm_tflops_by_rows = [\n  [1024, [[256, 40.0], [8192, \
             150.0]]],\n  [4096, [[256, 90.5], [8192, 210.0]]],\n]\n\
             attn_tflops_by_keys = [[512, 60.0], [8192, 400.0]]\nhbm_gbps = 654.4\n\
             cached_gbps = 28800.0\nelementwise_gbps = 450.0\nscan_gelems = 2684.3\nfew_row_tflops = 4.2\n\
             workgroups_per_core = 3.3\nkernel_tflops_by_shape_keyed = [\n  [1, 4, 1024, 2048, 512, 1, 1, \
             [[\"linear_gemm_gemv_v_bf16\", 7]], [[1, 0.5]], []],\n]\n\
             dense_settled_keyed = [\n  [1, 77, 9, [[3, 4, 1, 4, 0, 1]]],\n]\n\
             measured_on = \"vk 10de:2c31 driver 1\"\n\
             measured_by = [\n  [\"launch_ns\", 2, [[\"linear_gemm_tiled_bf16\", 5]]],\n]\n"
        );
        // It reads back.
        let registry = crate::Registry::new("test", &[], &[]);
        let read = registry
            .parse(
                &format!("api = 1\n{}", rates.toml()),
                "user",
                crate::registry::Strict::User,
            )
            .unwrap();
        assert_eq!(read.tuned.rates, rates);
        // A checked-in file states no keys: its rates are decisions.
        let refused = registry
            .parse(
                &format!("api = 1\n{}", rates.toml()),
                "checked in",
                crate::registry::Strict::CheckedIn,
            )
            .unwrap_err();
        assert!(refused.to_string().contains("measured_on"), "{refused}");
    }

    /// This build on a device, as a test states it: each kernel's code and
    /// each method's version a table.
    struct Build {
        on: &'static str,
        methods: Vec<(&'static str, u32)>,
        code: Vec<(&'static str, u32)>,
    }

    impl Now for Build {
        fn on(&self) -> String {
            self.on.to_string()
        }
        fn method(&self, what: &str) -> Option<u32> {
            self.methods
                .iter()
                .find(|(m, _)| *m == what)
                .map(|&(_, v)| v)
        }
        fn code(&self, kernel: &str) -> Option<u32> {
            self.code
                .iter()
                .find(|(k, _)| *k == kernel)
                .map(|&(_, c)| c)
        }
    }

    fn build() -> Build {
        Build {
            on: "vulkan 10de:2c31 driver 0x91c40000",
            methods: vec![
                ("launch_ns", 1),
                ("hbm_gbps", 1),
                ("gemm_tflops_by_rows", 1),
                (KERNEL_CURVES, 1),
                (DENSE_SETTLED, 1),
            ],
            code: vec![("tiled", 11), ("coopmat", 22), ("gemv", 33)],
        }
    }

    fn by(kernels: &[&str], now: &Build) -> MeasuredBy {
        MeasuredBy {
            method: 1,
            kernels: kernels
                .iter()
                .map(|k| ((*k).to_string(), now.code(k).unwrap()))
                .collect(),
        }
    }

    /// What a load measured and stored under `now`.
    fn stored(now: &Build) -> Rates {
        let curve = |key, kernel: &str| KernelCurve {
            kind: KernelCurve::TILE,
            key,
            n: 1024,
            k: 1024,
            curve: vec![(1, 0.5), (64, 9.0)],
            stop: None,
            spread: vec![(1, 0.01), (64, 0.02)],
            by: by(&[kernel], now),
        };
        Rates {
            launch_ns: Some(900),
            hbm_gbps: Some(700.0),
            gemm_tflops_by_rows: Some(vec![(1024, vec![(256, 20.0)])]),
            kernel_tflops_by_shape: Some(vec![curve(32, "tiled"), curve(64, "coopmat")]),
            dense_settled: vec![DenseSettled {
                method: 1,
                of: 5,
                code: 6,
                ranges: vec![[1, 1, 0, 0, 0, 0]],
            }],
            measured_on: Some(now.on()),
            measured_by: Some(vec![
                ("gemm_tflops_by_rows".into(), by(&["tiled", "coopmat"], now)),
                ("hbm_gbps".into(), by(&["tiled"], now)),
                ("launch_ns".into(), by(&["tiled"], now)),
            ]),
            ..Rates::default()
        }
    }

    /// Each model's settle is its own entry, by its key: a second model's
    /// joins the first's (a save lays the new over the stored), one of the
    /// same key replaces it, and the list writes and reads back whole.
    #[test]
    fn each_models_settle_is_kept_beside_the_others() {
        let settle = |of: u32, replaced: u32| DenseSettled {
            method: 1,
            of,
            code: 6,
            ranges: vec![[1, 1, replaced, 0, 0, 0]],
        };
        let first = Rates {
            dense_settled: vec![settle(7, 0)],
            ..Rates::default()
        };
        let second = Rates {
            dense_settled: vec![settle(3, 0)],
            ..Rates::default()
        };
        let both = second.over(first.clone());
        assert_eq!(both.dense_settled, vec![settle(3, 0), settle(7, 0)]);
        let again = Rates {
            dense_settled: vec![settle(7, 1)],
            ..Rates::default()
        }
        .over(both.clone());
        assert_eq!(again.dense_settled, vec![settle(3, 0), settle(7, 1)]);
        let registry = crate::Registry::new("test", &[], &[]);
        let read = registry
            .load_user(&format!("api = 1\n{}", both.toml()), "two models")
            .expect("both settles read back");
        assert_eq!(read.rates().dense_settled, both.dense_settled);
        let twice = "api = 1\n[rates]\ndense_settled_keyed = [[1, 7, 6, []], [1, 7, 6, []]]\n";
        assert!(registry.load_user(twice, "twice").is_err(), "a key stated twice is refused");
    }

    /// The same file read by another build of the same kernels (a version
    /// bump, a new commit, another day) keeps every measurement: nothing
    /// is measured again.
    #[test]
    fn a_build_bump_keeps_every_measurement() {
        let now = build();
        let mut rates = stored(&now);
        let said = rates.sift(&now);
        assert_eq!(rates, stored(&now), "{said:#?}");
        assert!(said.is_empty(), "nothing is dropped: {said:#?}");
        // Each entry taken is said by the layer deciding it, with its key.
        let whence = rates.whence(&rates, "user.toml", "no committed profile");
        assert!(
            whence
                .iter()
                .any(|l| l.starts_with("`launch_ns`: stored (user.toml), its key matches")),
            "{whence:#?}"
        );
        assert_eq!(
            whence.iter().filter(|l| l.starts_with("curve ")).count(),
            2,
            "{whence:#?}"
        );
    }

    /// One kernel's code changed: the measurements it fired are dropped,
    /// to be measured again, by name; every other stands.
    #[test]
    fn a_changed_kernel_drops_only_its_measurements() {
        let mut now = build();
        let before = stored(&now);
        now.code = vec![("tiled", 11), ("coopmat", 23), ("gemv", 33)];
        let mut rates = before.clone();
        let said = rates.sift(&now);
        assert_eq!(rates.gemm_tflops_by_rows, None, "it fired coopmat");
        assert_eq!(rates.launch_ns, before.launch_ns);
        assert_eq!(rates.hbm_gbps, before.hbm_gbps);
        let curves = rates.kernel_tflops_by_shape.clone().unwrap();
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].key, 32, "the tiled curve stands");
        assert_eq!(
            rates.dense_settled, before.dense_settled,
            "the settle checks its own"
        );
        assert_eq!(
            rates.measured_by.as_ref().map(|by| by.len()),
            Some(2),
            "the dropped rate's key goes with it"
        );
        assert!(
            said.iter()
                .any(|l| l.contains("gemm_tflops_by_rows`: not taken, kernel `coopmat`")),
            "{said:#?}"
        );
        assert_eq!(
            rates.missing(&["gemm_tflops_by_rows", "launch_ns"]),
            ["gemm_tflops_by_rows"]
        );
        // A method's version bumped drops what it measured alone.
        let mut now = build();
        now.methods[0].1 = 2;
        let mut rates = stored(&now);
        let said = rates.sift(&now);
        assert_eq!(rates.launch_ns, None);
        assert_eq!(rates.hbm_gbps, Some(700.0));
        assert!(
            said.iter()
                .any(|l| l.contains("version 2 now, 1 when stored")),
            "{said:#?}"
        );
    }

    /// Another device (or driver) measures every one again; a rate stored
    /// without a key too.
    #[test]
    fn another_device_drops_every_measurement() {
        let now = build();
        let mut rates = stored(&now);
        let other = Build {
            on: "vulkan 10de:2c32 driver 0x91c40000",
            ..build()
        };
        let said = rates.sift(&other);
        assert_eq!(rates, Rates::default());
        assert!(said[0].contains("none is taken"), "{said:#?}");
        let mut unkeyed = Rates {
            measured_by: None,
            ..stored(&now)
        };
        let said = unkeyed.sift(&now);
        assert_eq!(unkeyed.launch_ns, None);
        assert!(
            said.iter().any(|l| l.contains("stored without a key")),
            "{said:#?}"
        );
    }

    /// A user file an older pie wrote, stating rates this pie no longer
    /// reads, is read for the rest (those left unread, named); a checked-in
    /// file stating one is refused by it.
    #[test]
    fn a_user_files_obsolete_rates_are_left_unread_and_the_rest_read() {
        let text = "api = 1\n[rates]\nlaunch_ns = 904\nhbm_gbps = 650.0\n\
                    vector_tflops_by_rows = [[1, [[1, 0.5]]]]\nkernel_stops = []\n";
        let registry = crate::Registry::new("test", &[], &[]);
        let file = registry
            .parse(text, "user", crate::registry::Strict::User)
            .unwrap();
        assert_eq!(file.tuned.rates.launch_ns, Some(904));
        assert_eq!(file.tuned.rates.hbm_gbps, Some(650.0));
        assert_eq!(
            file.ignored_rates,
            ["vector_tflops_by_rows", "kernel_stops"]
        );
        let read = registry.load_user(text, "user").unwrap().rates();
        assert_eq!(read.launch_ns, Some(904));
        let refused = registry
            .parse(text, "checked in", crate::registry::Strict::CheckedIn)
            .unwrap_err();
        assert_eq!(
            refused.stale_rates,
            ["vector_tflops_by_rows", "kernel_stops"]
        );
        // A save drops them.
        let (kept, dropped) = registry.prune(text);
        assert!(!kept.contains("vector_tflops_by_rows"), "{kept}");
        assert_eq!(dropped.len(), 2, "{dropped:?}");
    }

    /// Curves an older pie timed per dispatch (`kernel_tflops_by_shape`)
    /// are left unread in a user file, as any key this pie no longer reads:
    /// the rates read hold no curves, so the next load measures them again
    /// by frame spans and saves them under [`KERNEL_CURVES`], the old key
    /// pruned.
    #[test]
    fn curves_timed_per_dispatch_are_measured_again() {
        let text = "api = 1\n[rates]\nlaunch_ns = 904\n\
                    kernel_tflops_by_shape = [[1, 4, 1024, 2048, 0, 0, [[1, 0.006], [256, 0.37]]]]\n";
        let registry = crate::Registry::new("test", &[], &[]);
        let file = registry
            .parse(text, "user", crate::registry::Strict::User)
            .unwrap();
        assert_eq!(file.ignored_rates, [KERNEL_CURVES_PER_DISPATCH]);
        let read = registry.load_user(text, "user").unwrap().rates();
        assert_eq!(read.kernel_tflops_by_shape, None);
        assert!(read.missing(&[KERNEL_CURVES]).contains(&KERNEL_CURVES));
        let (kept, _) = registry.prune(text);
        assert!(!kept.contains(KERNEL_CURVES_PER_DISPATCH), "{kept}");
    }

    /// The margin a load measures under a stated submit bound is saved with
    /// the rates and read back: a boot on saved rates holds its steps at it.
    #[test]
    fn the_submit_rates_are_saved_and_read_back_with_the_rates() {
        let measured = Rates {
            launch_ns: Some(904),
            submit_worst_ratio: Some(1.83),
            resident_workgroups: None,
            attn_tflops_by_rows_keys: Some(vec![(512, vec![(1024, 0.35), (4096, 0.3)])]),
            ..Rates::default()
        };
        let text = format!("api = 1\n{}", measured.toml());
        assert!(text.contains("submit_worst_ratio = 1.83\n"), "{text}");
        let registry = crate::Registry::new("test", &[], &[]);
        let read = registry.load_user(&text, "test").unwrap().rates();
        assert_eq!(read.submit_worst_ratio, Some(1.83));
        assert_eq!(
            read.attn_tflops_by_rows_keys,
            measured.attn_tflops_by_rows_keys
        );
        assert!(read.missing(&[SUBMIT_WORST_RATIO]).is_empty());
        assert_eq!(
            Rates::default().missing(&[SUBMIT_WORST_RATIO]),
            [SUBMIT_WORST_RATIO]
        );
        assert_eq!(
            measured.only(&[SUBMIT_WORST_RATIO]).submit_worst_ratio,
            Some(1.83)
        );
    }

    #[test]
    fn a_curve_is_read_in_log_x_and_clamped() {
        let c = [(256, 100.0), (1024, 200.0)];
        assert_eq!(interpolate(&c, 1.0), Some(100.0));
        assert_eq!(interpolate(&c, 4096.0), Some(200.0));
        assert!((interpolate(&c, 512.0).unwrap() - 150.0).abs() < 1e-9);
        assert_eq!(interpolate(&[], 1.0), None);
    }
}
