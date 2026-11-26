//! The dense kernels' measured curves at a model's planes, the settle a
//! load took over their picks, and reading a rate off a curve.

use crate::rates::MeasuredBy;

/// A settle's outcome, keyed: `method`, the version of the settle's
/// measuring method; `of`, a fingerprint of what it weighed (the curves, the
/// model's planes, the load's rungs, lanes and kernel choice); `code`, a
/// fingerprint of the code of every kernel the load's served step fires
/// (what each weighed step ran); and each row range it decided, `[bottom,
/// top, replaced, vector, wide, tier]`: 1 where it replaced the curves'
/// kernel with the one named (the tier `kernels_slang::Tier` as a number), 2
/// where it kept theirs and names it (every plane took that one), 0 where
/// it kept theirs unnamed (a row from before the kept kernel was named, or
/// planes that took different kernels): a load serves the curves' pick
/// there, never a guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseSettled {
    pub method: u32,
    pub of: u32,
    pub code: u32,
    pub ranges: Vec<[u32; 6]>,
    /// The model's dense planes the settle weighed, `(n, k)`, sorted: what a
    /// boot of that model reads the curves at, so an export can say which
    /// curves the settle's model needs and whether each is carried.
    pub planes: Vec<(u32, u32)>,
}

/// One term of a fire's time, measured at `pie tune` as serving launches it
/// ([`Rates::fire_terms`]): its kind, the shape it was timed at, its time a
/// launch in nanoseconds by rows (ascending), and what measured it. What the
/// estimate reads a fire's terms off in place of its roofline's, an
/// unprofiled shape falling back to the roofline, named.
///
/// The kinds and their shapes:
/// - [`FireTerm::ATTENTION`]: `[q_heads, kv_heads, head_dim, window (0:
///   none), sinks (0 or 1), lanes (the capture's carve), keys]`, the rows a
///   rung's prefill attends at `keys` keys, launched through serving's
///   masked prefill and padded to the carve's lanes;
/// - [`FireTerm::GEMM`]: `[n, k, dtype]`, a dense plane's matmul;
/// - [`FireTerm::EXPERTS`]: `[experts, top_k, n, k, dtype]`, one routed
///   select (gate/up or down) with its routing, sort and combine;
/// - [`FireTerm::ROW_OPS`]: `[hidden, inter]`, a layer's per-row pass (norm,
///   rope, residual, gated activation, kv write);
/// - [`FireTerm::FIRE_FIXED`]: `[]`, a fire's own cost past its kernels
///   (host preparation, replay, sync), by rows: the rung's body fired end
///   to end less the same body's device time;
/// - [`FireTerm::SCAN`]: `[k_heads, v_heads, k_dim, v_dim, conv_width]`, one
///   gated-delta layer's recurrent part (its conv, chunked scan, gates and
///   norm together; its projections are [`FireTerm::GEMM`] terms).
///
/// Every kind's `rows` are the rungs the engine arms (the estimate
/// interpolates between them in log rows). An attention entry is one key
/// count (its shape's last field); a rung's attention at other keys is read
/// linearly between the entries at the same heads and lanes. `lanes` is the
/// carve the engine's captured plan pads the rung to. A term is one layer's
/// (or one plane's); the estimate multiplies by the layers that hold it.
/// `dtype` codes: [`FireTerm::BF16`] and on.
#[derive(Debug, Clone, PartialEq)]
pub struct FireTerm {
    pub kind: u32,
    pub shape: Vec<u32>,
    pub curve: Vec<(u32, f64)>,
    pub by: MeasuredBy,
}

impl FireTerm {
    pub const ATTENTION: u32 = 1;
    pub const GEMM: u32 = 2;
    pub const EXPERTS: u32 = 3;
    pub const ROW_OPS: u32 = 4;
    pub const FIRE_FIXED: u32 = 5;
    pub const SCAN: u32 = 6;

    /// A plane's stored dtype, as a gemm or experts shape states it.
    pub const BF16: u32 = 1;
    pub const FP16: u32 = 2;
    pub const FP8_E4M3: u32 = 3;
    pub const MXFP4: u32 = 4;
    pub const NVFP4: u32 = 5;
    /// Affine 4-bit codes in groups (`u4g64`).
    pub const INT4_AFFINE: u32 = 6;
    pub const FP32: u32 = 7;

    /// The fields a kind's shape holds, the kind named; none for a kind
    /// this pie does not read.
    #[must_use]
    pub fn shape_of(kind: u32) -> Option<(&'static str, usize)> {
        Some(match kind {
            Self::ATTENTION => ("attention", 7),
            Self::GEMM => ("gemm", 3),
            Self::EXPERTS => ("experts", 5),
            Self::ROW_OPS => ("row ops", 2),
            Self::FIRE_FIXED => ("fire fixed", 0),
            Self::SCAN => ("scan", 5),
            _ => return None,
        })
    }

    /// What one entry is joined, sorted and replaced by: its kind and shape.
    #[must_use]
    pub fn id(&self) -> (u32, Vec<u32>) {
        (self.kind, self.shape.clone())
    }

    #[must_use]
    pub fn name(&self) -> String {
        let kind = Self::shape_of(self.kind).map_or("?", |(name, _)| name);
        let shape: Vec<String> = self.shape.iter().map(u32::to_string).collect();
        format!("{kind} [{}]", shape.join(", "))
    }
}

/// One dense kernel's measured curve at one shape
/// ([`Rates::kernel_curves`]): the kernel (`kind`
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
    /// A tensor-op (Metal 4 MPP) tile, `key` its row tile: Metal's.
    pub const MPP: u32 = 3;
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
