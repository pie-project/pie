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

use crate::curves::{DenseSettled, FireTerm, KernelCurve};
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
    /// The terms of a fire's time as `pie tune` measured them, each at the
    /// shape serving launches it at ([`FireTerm`], [`FIRE_TERMS`]): what the
    /// estimate reads a fire's attention, matmuls, routed experts, per-row
    /// pass and own fixed cost off, keyed entry by entry by what measured
    /// it. Entries accumulate tune to tune, a shape's own replacing it.
    /// Measured at `pie tune`, never at a load; not one of [`KEYS`].
    pub fire_terms: Option<Vec<FireTerm>>,
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

/// The `[rates]` key of [`Rates::fire_terms`]: a list of `[kind, [shape],
/// method, [[kernel, code], ...], [[rows, ns], ...]]`, one per kind and
/// shape.
pub const FIRE_TERMS: &str = "fire_terms";

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
    fire_terms: FIRE_TERMS,
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
        if let Some(terms) = &mut self.fire_terms {
            terms.sort_by_key(FireTerm::id);
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
    use crate::registry::Strict;

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
                (FIRE_TERMS, 1),
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
            fire_terms: Some(vec![
                FireTerm {
                    kind: FireTerm::ATTENTION,
                    shape: vec![16, 8, 128, 0, 0, 64, 16384],
                    curve: vec![(2048, 1_951_376.0), (4096, 3_376_128.0)],
                    by: by(&["tiled"], now),
                },
                FireTerm {
                    kind: FireTerm::GEMM,
                    shape: vec![6144, 4096, 1],
                    curve: vec![(256, 131_752.0), (4096, 1_347_296.0)],
                    by: by(&["coopmat"], now),
                },
                FireTerm {
                    kind: FireTerm::FIRE_FIXED,
                    shape: vec![],
                    curve: vec![(1, 530_000.0)],
                    by: by(&["gemv"], now),
                },
            ]),
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

    /// Every form a `[rates]` value takes reads back as written in a user's
    /// file; a checked-in file states no measurements and refuses them.
    #[test]
    fn every_form_reads_back_as_written() {
        let rates = Rates {
            attn_tflops_by_keys: Some(vec![(512, 60.0), (8192, 400.0)]),
            cached_gbps: Some(28800.0),
            elementwise_gbps: Some(450.0),
            scan_gelems: Some(2684.3),
            few_row_tflops: Some(4.2),
            workgroups_per_core: Some(3.3),
            submit_worst_ratio: Some(1.83),
            load_width: Some(4),
            resident_workgroups: Some(10.5),
            attn_tflops_by_rows_keys: Some(vec![(512, vec![(1024, 0.35), (4096, 0.3)])]),
            product_ns: Some(vec![ProductTime {
                kind: 3,
                rows: 6144,
                k: 1024,
                m: 4,
                kernel: 1,
                ns: 6100.0,
                spread: 40.0,
            }]),
            landing: Some(Landing::of([
                7411.0, 24.25, 2097152.0, 56.25, 217133.0, 1.81, 53.5,
            ])),
            ..stored(&build())
        };
        assert!(rates.complete(&KEYS));
        let registry = crate::Registry::new("test", &[], &[]);
        let text = format!("api = 1\n{}", rates.toml());
        let read = registry.parse(&text, "user", Strict::User).unwrap();
        assert_eq!(read.tuned.rates, rates, "{text}");
        assert!(read.skipped_rates.is_empty(), "{:?}", read.skipped_rates);
        assert!(
            registry
                .parse(&text, "checked in", Strict::CheckedIn)
                .is_err()
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
        let terms = rates.fire_terms.clone().unwrap();
        assert_eq!(
            terms.iter().map(|t| t.kind).collect::<Vec<_>>(),
            [FireTerm::ATTENTION, FireTerm::FIRE_FIXED],
            "the gemm term it fired goes, the others stand"
        );
        assert_eq!(
            rates
                .stale
                .as_ref()
                .and_then(|s| s.fire_terms.clone())
                .map(|t| t[0].kind),
            Some(FireTerm::GEMM),
            "kept as an estimate"
        );
        assert!(
            said.iter()
                .any(|l| l.contains("fire term gemm [6144, 4096, 1]: not taken")),
            "{said:#?}"
        );
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
    }

    /// An export never writes a stale entry: a committed rate whose key no
    /// longer matches is written only as the home re-timed it, and refuses
    /// the export, by name, where the home holds nothing current.
    #[test]
    fn an_export_writes_a_stale_committed_key_only_as_the_home_re_timed_it() {
        let committed = stored(&build());
        let mut moved = build();
        moved.code = vec![("tiled", 11), ("coopmat", 23), ("gemv", 33)];
        let mut home = stored(&moved);
        home.gemm_tflops_by_rows = Some(vec![(4096, vec![(256, 7.5)])]);
        let (merged, _) =
            crate::merge_committed(home.clone(), Some(committed.clone()), Some(&moved)).unwrap();
        assert_eq!(merged.gemm_tflops_by_rows, home.gemm_tflops_by_rows);
        let refused =
            crate::merge_committed(Rates::default(), Some(committed), Some(&moved)).unwrap_err();
        assert!(refused.contains("`gemm_tflops_by_rows`"), "{refused}");
    }

    /// A user's file one entry of which does not read (an older form, a key
    /// stated twice, a landing of the wrong length) skips that entry alone,
    /// by name, and reads the rest; a checked-in file is refused.
    #[test]
    fn an_unreadable_entry_is_skipped_alone() {
        let registry = crate::Registry::new("test", &[], &[]);
        let text = "api = 1\n[rates]\nlaunch_ns = 900\nlanding = [1.0, 2.0]\n\
                    dense_settled_keyed = [\n  [1, 77, 9, [[3, 4, 1, 4, 0, 1]]],\n  \
                    [1, 78, 9, [[1, 1, 0, 0, 0, 0]], [[1024, 2048]]],\n  \
                    [1, 78, 9, [[1, 1, 0, 0, 0, 0]], [[1024, 2048]]],\n]\n";
        let file = registry.parse(text, "home.toml", Strict::User).unwrap();
        assert_eq!(file.tuned.rates.launch_ns, Some(900));
        assert_eq!(file.tuned.rates.landing, None);
        assert_eq!(file.tuned.rates.dense_settled.len(), 1);
        assert_eq!(file.tuned.rates.dense_settled[0].of, 78);
        for said in ["dense_settled_keyed[0]", "stated twice", "rates.landing"] {
            assert!(
                file.skipped_rates.iter().any(|s| s.contains(said)),
                "{said}: {:?}",
                file.skipped_rates
            );
        }
        assert!(registry.parse(text, "c.toml", Strict::CheckedIn).is_err());
    }

    /// A fire term of a kind this pie does not read, or whose shape lacks a
    /// field, is skipped alone, by name, to be measured again at `pie
    /// tune`; the list's others are read.
    #[test]
    fn a_fire_term_in_another_form_is_skipped_alone() {
        let text = "api = 1\n[rates]\nfire_terms = [\n  [1, [16, 8, 128, 0, 0, 64, 4096], 1, [[\"tiled\", 11]], [[2048, 431856.0]]],\n  [9, [1], 1, [], [[1, 2.0]]],\n  [2, [6144, 4096], 1, [], [[256, 1.0]]],\n]\n";
        let registry = crate::Registry::new("test", &[], &[]);
        let read = registry.parse(text, "user", Strict::User).unwrap();
        let terms = read.tuned.rates.fire_terms.clone().unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].shape, [16, 8, 128, 0, 0, 64, 4096]);
        assert_eq!(read.skipped_rates.len(), 2, "{:?}", read.skipped_rates);
        assert!(
            read.skipped_rates
                .iter()
                .all(|s| s.starts_with("fire_terms["))
        );
    }
}
