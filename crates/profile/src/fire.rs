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
//! # each tiled kernel's TFLOP/s by rows below its full fill, keyed by its row tile
//! tile_tflops_by_rows = [[32, [[8, 1.0], [256, 9.0]]], [256, [[8, 0.2], [256, 12.0]]]]
//! # each vector kernel's TFLOP/s by rows, keyed by the rows a workgroup carries
//! vector_tflops_by_rows = [[1, [[1, 0.3], [256, 2.0]]], [16, [[1, 0.1], [256, 6.0]]]]
//! ```
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
    /// Each tiled matmul kernel's bf16 TFLOP/s by rows below the matmul
    /// curve's, its flops counted at the rows it lands (not its tile's):
    /// `(row tile, [(rows, TFLOP/s)])`, ascending in row tile, keyed by the
    /// rows one tile of the kernel computes (the slang backends' tiles).
    /// What a matmul of few rows costs on each tiled kernel, so a choice
    /// between them and the vector kernels weighs measured costs at partial
    /// fill. Measured.
    pub tile_tflops_by_rows: Option<Vec<(u32, Vec<(u32, f64)>)>>,
    /// Each vector matmul kernel's bf16 TFLOP/s by rows, its flops at the
    /// rows it lands over its time (its weight passes in it): `(rows a
    /// workgroup carries, [(rows, TFLOP/s)])`, ascending in the rows a
    /// workgroup carries (the slang backends' vector kernels). Measured.
    pub vector_tflops_by_rows: Option<Vec<(u32, Vec<(u32, f64)>)>>,
    /// Each decode product's time on each kernel that can serve it, measured
    /// in place (a captured burst over the model's own layer weights, as a
    /// step meets them), keyed by the product's shape: what a backend picks
    /// between candidate kernels by, where a threshold would otherwise stand.
    /// Entries accumulate load to load, a load measuring only the shapes the
    /// table lacks. Not a roofline rate: no load requires it.
    pub product_ns: Option<Vec<ProductTime>>,
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
pub const KEYS: [&str; 11] = [
    "launch_ns",
    "gemm_tflops_by_rows",
    "attn_tflops_by_keys",
    "hbm_gbps",
    "cached_gbps",
    "elementwise_gbps",
    "scan_gelems",
    "few_row_tflops",
    "workgroups_per_core",
    "tile_tflops_by_rows",
    "vector_tflops_by_rows",
];

/// The `[rates]` key of [`Rates::submit_worst_ratio`]: what a load on a
/// platform that states a bound on one submission measures beside the
/// estimate's [`KEYS`].
pub const SUBMIT_WORST_RATIO: &str = "submit_worst_ratio";

/// The `[rates]` key of [`Rates::attn_tflops_by_rows_keys`]: measured beside
/// [`SUBMIT_WORST_RATIO`] where the platform states a submit bound.
pub const ATTN_BY_ROWS_KEYS: &str = "attn_tflops_by_rows_keys";

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
            attn_tflops_by_rows_keys: self
                .attn_tflops_by_rows_keys
                .or(below.attn_tflops_by_rows_keys),
            tile_tflops_by_rows: self.tile_tflops_by_rows.or(below.tile_tflops_by_rows),
            vector_tflops_by_rows: self.vector_tflops_by_rows.or(below.vector_tflops_by_rows),
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
        }
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
            submit_worst_ratio: self
                .submit_worst_ratio
                .filter(|_| has(SUBMIT_WORST_RATIO)),
            attn_tflops_by_rows_keys: self
                .attn_tflops_by_rows_keys
                .clone()
                .filter(|_| has(ATTN_BY_ROWS_KEYS)),
            tile_tflops_by_rows: self
                .tile_tflops_by_rows
                .clone()
                .filter(|_| has("tile_tflops_by_rows")),
            vector_tflops_by_rows: self
                .vector_tflops_by_rows
                .clone()
                .filter(|_| has("vector_tflops_by_rows")),
            product_ns: self.product_ns.clone().filter(|_| has("product_ns")),
        }
    }

    /// The keys of `keys` (a backend's estimate's: those its roofline weighs
    /// a fire by, of [`KEYS`]) this states none of: what a load measures
    /// before it is ready (`engine::rates::rates_at_load`).
    #[must_use]
    pub fn missing(&self, keys: &[&'static str]) -> Vec<&'static str> {
        let unstated = |key: &str| match key {
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
            ATTN_BY_ROWS_KEYS => self.attn_tflops_by_rows_keys.is_none(),
            "tile_tflops_by_rows" => self.tile_tflops_by_rows.is_none(),
            "vector_tflops_by_rows" => self.vector_tflops_by_rows.is_none(),
            "product_ns" => self.product_ns.is_none(),
            _ => true,
        };
        keys.iter().copied().filter(|key| unstated(key)).collect()
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
        keyed(&mut out, "tile_tflops_by_rows", &self.tile_tflops_by_rows);
        keyed(
            &mut out,
            "vector_tflops_by_rows",
            &self.vector_tflops_by_rows,
        );
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
        out
    }
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
}

impl Tuned {
    /// `upper` laid over `self`.
    pub fn lay(&mut self, upper: &Tuned) {
        self.rates = upper.rates.clone().over(std::mem::take(&mut self.rates));
        self.measured_this_load |= upper.measured_this_load;
        self.saved_to = upper.saved_to.clone().or(self.saved_to.take());
        self.frame_cut |= upper.frame_cut;
        self.refused = upper.refused.clone().or(self.refused.take());
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
            attn_tflops_by_rows_keys: None,
            tile_tflops_by_rows: Some(vec![(32, vec![(8, 1.0), (256, 9.5)])]),
            vector_tflops_by_rows: Some(vec![(4, vec![(1, 0.5)])]),
            product_ns: None,
        };
        assert!(rates.complete(&KEYS));
        assert_eq!(
            rates.only(&KEYS[..6]).missing(&KEYS),
            [
                "scan_gelems",
                "few_row_tflops",
                "workgroups_per_core",
                "tile_tflops_by_rows",
                "vector_tflops_by_rows"
            ]
        );
        assert_eq!(
            rates.toml(),
            "[rates]\nlaunch_ns = 904\ngemm_tflops_by_rows = [\n  [1024, [[256, 40.0], [8192, \
             150.0]]],\n  [4096, [[256, 90.5], [8192, 210.0]]],\n]\n\
             attn_tflops_by_keys = [[512, 60.0], [8192, 400.0]]\nhbm_gbps = 654.4\n\
             cached_gbps = 28800.0\nelementwise_gbps = 450.0\nscan_gelems = 2684.3\nfew_row_tflops = 4.2\n\
             workgroups_per_core = 3.3\ntile_tflops_by_rows = [\n  [32, [[8, 1.0], [256, 9.5]]],\n]\nvector_tflops_by_rows = [\n  [4, [[1, 0.5]]],\n]\n"
        );
    }

    /// The margin a load measures under a stated submit bound is saved with
    /// the rates and read back: a boot on saved rates holds its steps at it.
    #[test]
    fn the_submit_rates_are_saved_and_read_back_with_the_rates() {
        let measured = Rates {
            launch_ns: Some(904),
            submit_worst_ratio: Some(1.83),
            attn_tflops_by_rows_keys: Some(vec![(512, vec![(1024, 0.35), (4096, 0.3)])]),
            ..Rates::default()
        };
        let text = format!("api = 1\n{}", measured.toml());
        assert!(text.contains("submit_worst_ratio = 1.83\n"), "{text}");
        let registry = crate::Registry::new("test", &[], &[]);
        let read = registry.load_user(&text, "test").unwrap().rates();
        assert_eq!(read.submit_worst_ratio, Some(1.83));
        assert_eq!(read.attn_tflops_by_rows_keys, measured.attn_tflops_by_rows_keys);
        assert!(read.missing(&[SUBMIT_WORST_RATIO]).is_empty());
        assert_eq!(
            Rates::default().missing(&[SUBMIT_WORST_RATIO]),
            [SUBMIT_WORST_RATIO]
        );
        assert_eq!(measured.only(&[SUBMIT_WORST_RATIO]).submit_worst_ratio, Some(1.83));
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
