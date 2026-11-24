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
//! kernel_curves = [[1, 4, 1024, 1024, 0, 0, 1,
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

use toml::Spanned;
use toml::de::DeValue;

use crate::curves::{DenseSettled, KernelCurve};
use crate::registry::Walk;
use form::Slot;

mod form;
mod read;
mod stored;

/// A curve: `(x, rate)` pairs, strictly ascending in `x`.
pub type Curve = Vec<(u32, f64)>;

/// A curve at each of several widths: `(width, curve)`, ascending in width.
pub type WidthCurves = Vec<(u32, Curve)>;

pub use stored::{MeasuredBy, Now, fingerprint};

/// A device's rates, each where a layer states it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// What a dependent dispatch adds to its work, in nanoseconds.
    pub launch_ns: Option<u64>,
    /// Dense bf16 matmul TFLOP/s by rows at each width (columns and depth
    /// both): `(width, [(rows, TFLOP/s)])`, ascending in width, each curve
    /// strictly ascending in rows.
    pub gemm_tflops_by_rows: Option<WidthCurves>,
    /// Prefill attention TFLOP/s (the full rectangle's flops, four a query
    /// element a key) by keys, one sequence attending itself: `(keys,
    /// TFLOP/s)`, strictly ascending in keys.
    pub attn_tflops_by_keys: Option<Curve>,
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
    /// How many programs a load reads from its disk cache at once behind
    /// serving (XLA): the width past which the load rate stops rising,
    /// measured where none is stated.
    pub load_width: Option<u32>,
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
    pub attn_tflops_by_rows_keys: Option<WidthCurves>,
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
    pub kernel_curves: Option<Vec<KernelCurve>>,
    /// The dense kernels each load's settle took in context over the
    /// curves' picks ([`DENSE_SETTLED`]), one entry per key (a model's
    /// planes, curves and rungs: [`DenseSettled::of`]), many a device: what
    /// a later load of the same curves, model and kernels takes again
    /// without walking a fire. A new model's settle joins the others, never
    /// replaces them. Measured; empty where none is stored.
    pub dense_settled: Vec<DenseSettled>,
    /// How this machine lands a checkpoint's weights on the device
    /// ([`LANDING`], [`Landing`]): its link's and its reader's rates, as a
    /// load first measures them. Measured; not one of [`KEYS`].
    pub landing: Option<Landing>,
    /// The device a user's file's measurements were taken on
    /// ([`MEASURED_ON`]): its vendor, device id and driver, as the backend
    /// states them ([`Now::on`]). A file measured on another device (or
    /// driver) has every measurement measured again ([`Rates::sift`]).
    pub measured_on: Option<String>,
    /// Each measured rate's key ([`MEASURED_BY`]): the rate's name and what
    /// it was measured by. A rate a user's file states without one is
    /// measured again ([`Rates::sift`]).
    pub measured_by: Option<Vec<(String, MeasuredBy)>>,
    /// The figures a stored or committed file states of this device under
    /// a key this build does not match (a kernel's code or the measuring
    /// method changed; [`Rates::sift_as`]): not taken as measured, but a
    /// fair estimate, which a load serves by while it times them again
    /// behind serving (`engine::rates::resolve_at_load`). Read, never
    /// written.
    pub stale: Option<Box<Rates>>,
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
/// (`Rates::kernel_curves`), each curve keyed by what measured it
/// ([`KernelCurve::by`]).
pub const KERNEL_CURVES: &str = "kernel_curves";

/// The `[rates]` key of [`Rates::dense_settled`]: a list of `[method, of,
/// code, [[bottom, top, replaced, vector, wide, tier], ...], [[n, k], ...]]`,
/// one per key.
pub const DENSE_SETTLED: &str = "dense_settled_keyed";

/// The `[rates]` key of [`Rates::measured_on`].
pub const MEASURED_ON: &str = "measured_on";

/// The `[rates]` key of [`Rates::measured_by`]: `[[rate, method, [[kernel,
/// code], ...]], ...]`.
pub const MEASURED_BY: &str = "measured_by";

/// The rates a stored key covers ([`Rates::measured_by`]): every `[rates]`
/// key of one value a load measures. The curves and the settle carry their
/// own keys.
pub const KEYED: [&str; 15] = [
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
    LANDING,
    LOAD_WIDTH,
];

/// The `[rates]` key of [`Rates::landing`]: `[copy_latency_ns, copy_gbps,
/// copy_full_bytes, fill_gbps, read_spawn_ns, read_thread_gbps,
/// read_whole_gbps]`.
pub const LANDING: &str = "landing";

/// How this machine lands weights ([`Rates::landing`]): a copy to the
/// device's fixed cost, its rate, and the smallest copy that reaches it; the
/// rate a pinned slot is filled at; one reading thread's spawn and join, one
/// thread's read rate, and the most the threads read at together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landing {
    pub copy_latency_ns: f64,
    pub copy_gbps: f64,
    pub copy_full_bytes: f64,
    pub fill_gbps: f64,
    pub read_spawn_ns: f64,
    pub read_thread_gbps: f64,
    pub read_whole_gbps: f64,
}

impl Landing {
    /// The values in [`LANDING`]'s order.
    #[must_use]
    pub fn values(&self) -> [f64; 7] {
        [
            self.copy_latency_ns,
            self.copy_gbps,
            self.copy_full_bytes,
            self.fill_gbps,
            self.read_spawn_ns,
            self.read_thread_gbps,
            self.read_whole_gbps,
        ]
    }

    /// The landing of `values` in [`LANDING`]'s order.
    #[must_use]
    pub fn of(values: [f64; 7]) -> Landing {
        let [
            copy_latency_ns,
            copy_gbps,
            copy_full_bytes,
            fill_gbps,
            read_spawn_ns,
            read_thread_gbps,
            read_whole_gbps,
        ] = values;
        Landing {
            copy_latency_ns,
            copy_gbps,
            copy_full_bytes,
            fill_gbps,
            read_spawn_ns,
            read_thread_gbps,
            read_whole_gbps,
        }
    }
}

/// The `[rates]` key of [`Rates::submit_worst_ratio`]: what a load on a
/// platform that states a bound on one submission measures beside the
/// estimate's [`KEYS`].
pub const SUBMIT_WORST_RATIO: &str = "submit_worst_ratio";

/// The `[rates]` key of [`Rates::load_width`]: what an XLA load measures as
/// it loads its cached programs behind serving.
pub const LOAD_WIDTH: &str = "load_width";

/// The `[rates]` key of [`Rates::attn_tflops_by_rows_keys`]: measured beside
/// [`SUBMIT_WORST_RATIO`] where the platform states a submit bound.
pub const ATTN_BY_ROWS_KEYS: &str = "attn_tflops_by_rows_keys";

/// The `[rates]` key of [`Rates::resident_workgroups`]: measured beside the
/// estimate's [`KEYS`] on the slang backends.
pub const RESIDENT_WORKGROUPS: &str = "resident_workgroups";

/// The keys a measurement is stated under, each beside the field that holds
/// it, in the order a profile writes them: the one place a key is named for
/// writing, reading, laying, unstating and picking ([`form::Slot`], a
/// value's form beside it in [`form`]). Adding a field to [`Rates`] without
/// a line here does not compile.
macro_rules! rate_keys {
    ($($field:ident: $key:expr,)*) => {
        /// Every `[rates]` key a measurement is stated under, in the order a
        /// profile writes them; [`MEASURED_ON`] and [`MEASURED_BY`], what
        /// keys those measurements, follow them.
        pub const RATES: &[&str] = &[$($key),*];

        impl Rates {
            /// `self` laid over `below`: each key `self` states replaces
            /// `below`'s; a list of entries (the products' times, the curves,
            /// the settles, the stored keys) joins `below`'s, its own
            /// replacing an entry both state.
            #[must_use]
            pub fn over(self, below: Rates) -> Rates {
                Rates {
                    $($field: Slot::lay(self.$field, below.$field),)*
                    measured_on: self.measured_on.or(below.measured_on),
                    measured_by: match (self.measured_by, below.measured_by) {
                        (Some(mine), Some(theirs)) => {
                            Some(form::joined(mine, theirs, |(rate, _)| rate.clone()))
                        }
                        (mine, theirs) => mine.or(theirs),
                    },
                    stale: match (self.stale, below.stale) {
                        (Some(mine), Some(theirs)) => Some(Box::new(mine.over(*theirs))),
                        (mine, theirs) => mine.or(theirs),
                    },
                }
            }

            /// Whether `key` (one of [`RATES`]) is stated.
            #[must_use]
            pub fn states(&self, key: &str) -> bool {
                $(if key == $key {
                    return Slot::stated(&self.$field);
                })*
                false
            }

            /// `key` unstated, its stored key with it.
            pub fn unstate(&mut self, key: &str) {
                $(if key == $key {
                    self.$field = Default::default();
                })*
                if key == MEASURED_ON {
                    self.measured_on = None;
                }
                if key == MEASURED_BY {
                    self.measured_by = None;
                }
                if let Some(by) = &mut self.measured_by {
                    by.retain(|(rate, _)| rate != key);
                }
            }

            /// Only `keys` of `self` (as [`Rates::missing`] names them), the
            /// rest unstated.
            #[must_use]
            pub fn only(&self, keys: &[&str]) -> Rates {
                let has = |key: &str| keys.contains(&key);
                Rates {
                    $($field: if has($key) { self.$field.clone() } else { Default::default() },)*
                    measured_on: self.measured_on.clone(),
                    measured_by: self
                        .measured_by
                        .as_ref()
                        .map(|by| by.iter().filter(|(rate, _)| has(rate)).cloned().collect()),
                    stale: None,
                }
            }

            /// `[rates]` as profile TOML.
            #[must_use]
            pub fn toml(&self) -> String {
                let mut out = String::from("[rates]\n");
                $(Slot::write(&self.$field, $key, &mut out);)*
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

            /// `[rates] key = v` read into its field; `false` where no
            /// measurement is stated under `key`.
            pub(crate) fn read_key(
                &mut self,
                walk: &mut Walk<'_>,
                key: &str,
                v: &Spanned<DeValue<'_>>,
            ) -> bool {
                $(if key == $key {
                    self.$field = Slot::read(walk, $key, v);
                    return true;
                })*
                false
            }
        }
    };
}

rate_keys! {
    launch_ns: "launch_ns",
    load_width: LOAD_WIDTH,
    gemm_tflops_by_rows: "gemm_tflops_by_rows",
    attn_tflops_by_keys: "attn_tflops_by_keys",
    hbm_gbps: "hbm_gbps",
    cached_gbps: "cached_gbps",
    elementwise_gbps: "elementwise_gbps",
    scan_gelems: "scan_gelems",
    few_row_tflops: "few_row_tflops",
    workgroups_per_core: "workgroups_per_core",
    submit_worst_ratio: SUBMIT_WORST_RATIO,
    resident_workgroups: RESIDENT_WORKGROUPS,
    product_ns: "product_ns",
    landing: LANDING,
    attn_tflops_by_rows_keys: ATTN_BY_ROWS_KEYS,
    kernel_curves: KERNEL_CURVES,
    dense_settled: DENSE_SETTLED,
}

impl Rates {
    /// `self` in the order a committed measured profile writes it: each
    /// rate's key by its rate, each curve by its kernel and shape, so one
    /// device's export writes the same text whatever order its loads stored
    /// them in.
    #[must_use]
    pub fn ordered(mut self) -> Rates {
        if let Some(by) = &mut self.measured_by {
            by.sort_by(|a, b| a.0.cmp(&b.0));
        }
        if let Some(curves) = &mut self.kernel_curves {
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

    /// The keys of `keys` (a backend's estimate's: those its roofline weighs
    /// a fire by, of [`KEYS`]) this states none of: what a load measures
    /// before it is ready (`engine::rates::rates_at_load`).
    #[must_use]
    pub fn missing(&self, keys: &[&'static str]) -> Vec<&'static str> {
        keys.iter()
            .copied()
            .filter(|key| !self.states(key))
            .collect()
    }

    /// Whether every key of `keys` is stated.
    #[must_use]
    pub fn complete(&self, keys: &[&'static str]) -> bool {
        self.missing(keys).is_empty()
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
    /// What this load timed, each by name (`rates (launch_ns, ...)`,
    /// `curves (n)`, `resident_workgroups`, `bound probe`, `settle`): what
    /// the load's last line says it measured ("measured 0" where none).
    pub timed: Vec<String>,
    /// The `[rates]` keys this load serves by a figure stored under a
    /// stale key ([`Rates::stale`]), each timed again behind serving and
    /// stored for the next load.
    pub behind: Vec<String>,
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
        self.timed.extend(upper.timed.iter().cloned());
        self.behind.extend(upper.behind.iter().cloned());
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
            stale: None,
            landing: None,
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
            load_width: None,
            resident_workgroups: None,
            attn_tflops_by_rows_keys: None,
            product_ns: None,
            kernel_curves: Some(vec![KernelCurve {
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
                planes: vec![(1024, 2048)],
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
             workgroups_per_core = 3.3\nkernel_curves = [\n  [1, 4, 1024, 2048, 512, 1, 1, \
             [[\"linear_gemm_gemv_v_bf16\", 7]], [[1, 0.5]], []],\n]\n\
             dense_settled_keyed = [\n  [1, 77, 9, [[3, 4, 1, 4, 0, 1]], [[1024, 2048]]],\n]\n\
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
            kernel_curves: Some(vec![curve(32, "tiled"), curve(64, "coopmat")]),
            dense_settled: vec![DenseSettled {
                method: 1,
                of: 5,
                code: 6,
                ranges: vec![[1, 1, 0, 0, 0, 0]],
                planes: vec![(1024, 1024)],
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
            planes: vec![(1024, 1024)],
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
        let twice =
            "api = 1\n[rates]\ndense_settled_keyed = [[1, 7, 6, [], []], [1, 7, 6, [], []]]\n";
        // A key stated twice: the second entry is skipped, by name (the
        // first and the rest read); a committed file carrying one fails the
        // build's check of committed profiles (`skipped_rates`).
        let file = registry
            .parse(twice, "twice.toml", crate::Strict::User)
            .expect("the rest reads");
        assert_eq!(file.tuned.rates.dense_settled.len(), 1, "the first is read");
        assert!(
            file.skipped_rates
                .iter()
                .any(|s| s.contains("stated twice")),
            "{:?}",
            file.skipped_rates
        );
    }

    /// A re-export of a home that took its rates from a committed profile
    /// and measured one model's curves keeps every committed entry still
    /// keyed for this build (carried) beside the home's (measured), each
    /// said; a committed rate whose key no longer matches and that the home
    /// did not measure again refuses the export by name.
    #[test]
    fn a_re_export_keeps_the_committed_entries_and_says_each() {
        let now = build();
        let committed = stored(&now);
        let measured = Rates {
            kernel_curves: committed.kernel_curves.clone().map(|curves| {
                curves
                    .into_iter()
                    .take(1)
                    .map(|mut c| {
                        c.n = 777;
                        c
                    })
                    .collect()
            }),
            measured_on: Some(now.on()),
            ..Rates::default()
        };
        let (merged, report) =
            crate::merge_committed(measured, Some(committed.clone()), Some(&now)).unwrap();
        assert_eq!(
            merged.launch_ns, committed.launch_ns,
            "the committed rate is carried"
        );
        assert_eq!(merged.hbm_gbps, committed.hbm_gbps);
        assert_eq!(
            merged.kernel_curves.as_ref().map(Vec::len),
            Some(committed.kernel_curves.as_ref().map_or(0, Vec::len) + 1),
            "the committed curves and the home's"
        );
        assert_eq!(merged.dense_settled, committed.dense_settled);
        assert!(
            report
                .iter()
                .any(|l| l == "`launch_ns`: carried from the committed profile"),
            "{report:#?}"
        );
        assert!(
            report.iter().any(|l| l.contains("1 measured")),
            "{report:#?}"
        );
        // A kernel whose code moved: its committed rates no longer match.
        let mut moved = build();
        moved.code = vec![("tiled", 11), ("coopmat", 23), ("gemv", 33)];
        let refused =
            crate::merge_committed(Rates::default(), Some(committed), Some(&moved)).unwrap_err();
        assert!(refused.contains("gemm_tflops_by_rows"), "{refused}");
    }

    /// An export never writes a stale entry. A committed rate or curve
    /// whose key no longer matches this build is written only as the home
    /// measured it again under this build (its figure, never the committed
    /// one); where the home holds no current measurement, the export
    /// refuses, naming each stale rate and curve. A home entry stored under
    /// a stale key is dropped too.
    #[test]
    fn an_export_writes_a_stale_committed_key_only_as_the_home_re_timed_it() {
        let exported = build();
        let committed = stored(&exported);
        let mut moved = build();
        moved.code = vec![("tiled", 11), ("coopmat", 23), ("gemv", 33)];
        // The home re-timed every rate and curve under the moved build.
        let mut home = stored(&moved);
        home.launch_ns = committed.launch_ns.map(|ns| ns + 1);
        home.gemm_tflops_by_rows = Some(vec![(4096, vec![(256, 7.5)])]);
        let (merged, report) =
            crate::merge_committed(home.clone(), Some(committed.clone()), Some(&moved)).unwrap();
        assert_eq!(
            merged.gemm_tflops_by_rows, home.gemm_tflops_by_rows,
            "re-timed, not copied"
        );
        assert_ne!(merged.gemm_tflops_by_rows, committed.gemm_tflops_by_rows);
        assert!(
            report
                .iter()
                .any(|l| l == "`gemm_tflops_by_rows`: measured"),
            "{report:#?}"
        );
        // The home holds nothing current: refused by name, never copied.
        let refused =
            crate::merge_committed(Rates::default(), Some(committed.clone()), Some(&moved))
                .unwrap_err();
        assert!(refused.contains("`gemm_tflops_by_rows`"), "{refused}");
        assert!(
            refused.contains("curve "),
            "the stale curves are named: {refused}"
        );
        // A home entry stored under the old build is dropped, not written:
        // the committed one (current under the old build) stands for it.
        let (merged, report) =
            crate::merge_committed(stored(&moved), Some(committed), Some(&exported)).unwrap();
        assert_eq!(
            merged.gemm_tflops_by_rows,
            stored(&exported).gemm_tflops_by_rows
        );
        assert!(
            report.iter().any(|l| l.starts_with("dropped: stored")),
            "{report:#?}"
        );
    }

    /// An export says each model's settle with the curves a boot of it
    /// reads, and leaves out one whose planes carry no curve.
    #[test]
    fn an_export_carries_a_models_settle_only_with_its_curves() {
        let curve = |n: u32, k: u32| KernelCurve {
            kind: KernelCurve::VECTOR,
            key: 1,
            n,
            k,
            curve: vec![(1, 0.5)],
            stop: None,
            spread: Vec::new(),
            by: MeasuredBy::default(),
        };
        let settle = |of: u32, planes: Vec<(u32, u32)>| DenseSettled {
            method: 1,
            of,
            code: 2,
            ranges: vec![[1, 1, 0, 0, 0, 0]],
            planes,
        };
        let mut rates = Rates {
            kernel_curves: Some(vec![curve(1024, 2048), curve(3072, 1024)]),
            dense_settled: vec![
                settle(7, vec![(1024, 2048), (3072, 1024)]),
                settle(9, vec![(1024, 2048), (5120, 1024)]),
            ],
            ..Rates::default()
        };
        let report = crate::settles_carried(&mut rates);
        assert_eq!(rates.dense_settled.len(), 1);
        assert_eq!(rates.dense_settled[0].of, 7);
        assert!(
            report[0].contains("carried") && report[0].contains("3072x1024: 1 curve"),
            "{report:?}"
        );
        assert!(
            report[1].starts_with("WARN") && report[1].contains("5120x1024"),
            "{report:?}"
        );
    }

    /// A committed (or user) file with one settle entry in an older form
    /// beside valid rates: the rates are read, the other entries too, and
    /// that entry alone is skipped, named, to be settled again.
    #[test]
    fn an_old_form_list_entry_is_skipped_alone() {
        let registry = crate::Registry::new("test", &[], &[]);
        let text = "api = 1\n[rates]\nlaunch_ns = 900\n\
                    dense_settled_keyed = [\n  [1, 77, 9, [[3, 4, 1, 4, 0, 1]]],\n  \
                    [1, 78, 9, [[1, 1, 0, 0, 0, 0]], [[1024, 2048]]],\n]\n";
        for strict in [crate::Strict::User, crate::Strict::Measured] {
            let file = registry.parse(text, "home.toml", strict);
            let file = match (strict, file) {
                (_, Ok(file)) => file,
                (crate::Strict::Measured, Err(e)) if e.to_string().contains("keys every entry") => {
                    // A committed file keys each rate: read past that here.
                    continue;
                }
                (_, Err(e)) => panic!("{strict:?}: {e}"),
            };
            assert_eq!(file.tuned.rates.launch_ns, Some(900), "{strict:?}");
            assert_eq!(file.tuned.rates.dense_settled.len(), 1, "{strict:?}");
            assert_eq!(file.tuned.rates.dense_settled[0].of, 78);
            assert!(
                file.skipped_rates
                    .iter()
                    .any(|s| s.starts_with("dense_settled_keyed[0]")),
                "{strict:?}: {:?}",
                file.skipped_rates
            );
        }
    }

    /// A user file one entry of which another pie wrote in another form
    /// (the settle as one entry, not a list): that entry alone is skipped,
    /// named, and the rest of the file read.
    #[test]
    fn a_user_files_unreadable_entry_is_skipped_alone() {
        let registry = crate::Registry::new("test", &[], &[]);
        let text = "api = 1\n[rates]\nlaunch_ns = 900\nhbm_gbps = 700.0\n\
                    dense_settled_keyed = [1, 77, 9, [[3, 4, 1, 4, 0, 1]]]\n";
        let file = registry
            .parse(text, "old.toml", crate::Strict::User)
            .expect("the rest of the file reads");
        assert_eq!(file.tuned.rates.launch_ns, Some(900));
        assert_eq!(file.tuned.rates.hbm_gbps, Some(700.0));
        assert!(file.tuned.rates.dense_settled.is_empty());
        assert!(!file.skipped_rates.is_empty(), "{:?}", file.skipped_rates);
        assert!(
            file.skipped_rates
                .iter()
                .all(|s| s.starts_with(DENSE_SETTLED)),
            "{:?}",
            file.skipped_rates
        );
        // A committed profile is the build's: refused whole, by name.
        assert!(
            registry
                .parse(text, "c.measured.toml", crate::Strict::Measured)
                .is_err()
        );
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
        let curves = rates.kernel_curves.clone().unwrap();
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
            said.iter().any(
                |l| l.contains("gemm_tflops_by_rows`: not taken as measured, kernel `coopmat`")
            ),
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

    /// The margin a load measures under a stated submit bound is saved with
    /// the rates and read back: a boot on saved rates holds its steps at it.
    #[test]
    fn the_submit_rates_are_saved_and_read_back_with_the_rates() {
        let measured = Rates {
            launch_ns: Some(904),
            submit_worst_ratio: Some(1.83),
            load_width: None,
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
}
