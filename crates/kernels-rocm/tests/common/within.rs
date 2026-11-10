//! The contract a generated row kernel (and a hand body in the row skeleton)
//! is held to, and its tolerances: kernels-cuda's (`elemwise/within.rs`),
//! the same policy on HIP's skeleton. And the harness the synth tests share,
//! in kernels-cuda's harness's shape: raw device planes by address, a seeded
//! draw of bf16 rows, a graph captured and replayed.
//!
//! - **The reference** is the kernels-ir interpreter (`interp::eval`): the
//!   region's own math, f32 ops as the region states them, a reduction's sum
//!   in f64 and rounded to f32 once.
//! - **Exact where the region says.** A value rounds to its dtype where the
//!   region rounds it. `+ - * /`, `fma` and `sqrt` are IEEE on the device
//!   (the JIT compiles `-ffp-contract=off`, and HIP's division and square
//!   root are correctly rounded), so a chain of them lands the reference's
//!   bits: [`EXACT`].
//! - **A reduction's order is the kernel's.** The device sums in f32 in its
//!   schedule's order (a wavefront's butterfly, then the wavefronts'), and
//!   `rsqrtf` is within an f32 ulp or two of `1 / sqrt`. The normed value's
//!   relative error is of order `sqrt(n)` f32 ulps plus 2, far under half a
//!   bf16 ulp, so it moves a bf16 result only where the exact value lies
//!   that close to a rounding edge, by one ulp: [`NORM`]. Each rounding
//!   point between a norm and a result can carry that ulp on, one more
//!   each: [`NORM_SCALED`].
//! - **A turn's angle is the skeleton's**: HIP's skeleton turns through
//!   ocml's precise `sinf` and `cosf` (kernels-cuda's through `__sincosf`),
//!   and its frequency `powf` is within 2 f32 ulps, so at a position `p` the
//!   angle is off by up to `p · 2^-22`. A rotated element is then off by up
//!   to two bf16 ulps of the row's scale, and a norm's ulp before the turn
//!   carries on: [`TURN`], kernels-cuda's bound.
//! - **Run to run** a kernel lands the same bits over the same input
//!   ([`deterministic`]). Nothing is asked across kernels: another kernel's
//!   order is its own schedule's.
#![allow(dead_code)]
#![allow(unsafe_code)]

use dtype::Dtype;
use kernels_rocm::Ctx;
use kernels_rocm::hip;
use kernels_rocm::tensor::Tensor;

/// A chain of IEEE ops and declared roundings: the reference's bits.
pub const EXACT: f32 = 0.0;
/// A norm (its sum of squares and `rsqrt`) and one rounding after it.
pub const NORM: f32 = 1.0;
/// A norm, its row rounded, then a scale of it (a modulation) rounded
/// again: the norm's ulp and the second rounding's.
pub const NORM_SCALED: f32 = 2.0;
/// A norm's row turned at a position up to 32767: the angle's error and
/// the norm's ulp before it.
pub const TURN: f32 = 4.0;

/// How far apart `got` (bf16) and `want` are, in bf16 ulps at the scale of
/// the larger of them, or of a half where both are smaller: a row of order
/// one counts a value near zero in a half's ulps, not its own far finer
/// ones.
pub fn ulps(got: u16, want: f32) -> f32 {
    let got = from_bf16(got);
    let scale = got.abs().max(want.abs()).max(0.5);
    let spacing = 2f32.powi(scale.log2().floor() as i32 - 7);
    (got - want).abs() / spacing
}

/// The elements of `got` apart from `want`, and the most ulps one is.
pub fn apart(got: &[u16], want: &[f32]) -> (usize, f32) {
    assert_eq!(got.len(), want.len(), "the landed row and its reference");
    got.iter()
        .zip(want)
        .filter(|(g, w)| from_bf16(**g) != **w)
        .fold((0, 0.0), |(n, most), (g, w)| {
            (n + 1, most.max(ulps(*g, *w)))
        })
}

/// `got` within `tolerance` ulps of `want`, naming `case` where it is not.
pub fn assert_within(case: &str, got: &[u16], want: &[f32], tolerance: f32) {
    let (n, most) = apart(got, want);
    assert!(
        most <= tolerance,
        "{case}: {n} of {} elements apart from the reference, {most} ulps at most (within {tolerance})",
        got.len()
    );
}

/// Two fires of one kernel over one input land the same bits.
pub fn deterministic(case: &str, first: &[u16], second: &[u16]) {
    assert!(
        first == second,
        "{case}: two fires of the kernel land different bits"
    );
}

/// `row` as bf16 values, the reference's arguments.
pub fn widened(row: &[u16]) -> Vec<f32> {
    row.iter().map(|&v| from_bf16(v)).collect()
}

/// `region`'s results by the interpreter over `rows` rows, `args(r)` row
/// `r`'s arguments: each result's rows end to end, as its bf16 plane holds
/// them.
pub fn reference(
    region: &kernels_ir::Region,
    rows: usize,
    args: impl Fn(usize) -> Vec<kernels_ir::interp::Arg>,
) -> Vec<Vec<f32>> {
    let mut out = vec![Vec::new(); region.results.len()];
    for r in 0..rows {
        let results =
            kernels_ir::interp::eval(region, &args(r)).expect("the interpreter runs the region");
        for (k, row) in results.into_iter().enumerate() {
            out[k].extend(row.into_iter().map(kernels_ir::skeleton::round_bf16));
        }
    }
    out
}

/// Row `r` of a plane of rows `width` wide, widened.
pub fn row_of(plane: &[u16], r: usize, width: usize) -> kernels_ir::interp::Arg {
    kernels_ir::interp::Arg::Row(widened(&plane[r * width..(r + 1) * width]))
}

/// f32 → bf16 bits, round to nearest even.
#[must_use]
pub fn to_bf16(x: f32) -> u16 {
    crate::common::bf16(x)
}

#[must_use]
pub fn from_bf16(v: u16) -> f32 {
    crate::common::from_bf16(v)
}

/// kernels-cuda's seeded draw: bf16-exact values in [-1, 1).
pub struct Lcg(u64);

impl Lcg {
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 40) as u32;
        let raw = (bits as f32 / 8_388_608.0) - 1.0;
        from_bf16(to_bf16(raw))
    }

    /// `count` draws: their bf16 bits and their values.
    pub fn row(&mut self, count: usize) -> (Vec<u16>, Vec<f32>) {
        let mut raw = Vec::with_capacity(count);
        let mut exact = Vec::with_capacity(count);
        for _ in 0..count {
            let value = self.unit();
            raw.push(to_bf16(value));
            exact.push(value);
        }
        (raw, exact)
    }
}

/// A `[rows][width]` tensor of `dtype` at `at`.
#[must_use]
pub fn ten(at: u64, rows: usize, width: usize, dtype: Dtype) -> Tensor {
    Tensor::new(at, rows as u32, width as u32, dtype)
}

/// Whether there is a HIP device: a test returns early where there is
/// none, passing vacuously.
#[must_use]
pub fn present() -> bool {
    let here = hip::device_count().is_ok_and(|n| n > 0);
    if !here {
        eprintln!("no HIP device; skipping");
    }
    here
}

/// A device's planes by address, as kernels-cuda's harness holds them.
pub struct Gpu(crate::common::Gpu);

impl Gpu {
    /// The device (a test checked [`present`] first).
    #[must_use]
    pub fn open() -> Self {
        Self(crate::common::Gpu::open().expect("a HIP device"))
    }

    #[must_use]
    pub fn ctx(&self) -> Ctx {
        self.0.ctx()
    }

    pub fn sync(&self) {
        self.0.sync();
    }

    /// `values` in a fresh plane, its address.
    pub fn up<T: Copy>(&mut self, values: &[T]) -> u64 {
        self.0.up(values)
    }

    /// `bytes` zero bytes, their address.
    pub fn zeros(&mut self, bytes: usize) -> u64 {
        self.0.up(&vec![0u8; bytes.max(1)])
    }

    /// `count` elements at `at`, after the stream's work.
    pub fn down<T: Copy + Default>(&self, at: u64, count: usize) -> Vec<T> {
        self.0.down(at, count)
    }
}

/// hipStreamCaptureModeThreadLocal: another thread's calls do not break
/// this capture.
const THREAD_LOCAL: core::ffi::c_int = 1;

/// The graph `body` fires on `gpu`'s stream, captured and instantiated.
pub fn captured(gpu: &Gpu, body: impl FnOnce()) -> hip::rt::GraphExec {
    let ok = |code: hip::sys::HipError, call: &str| {
        assert_eq!(code, hip::sys::HIP_SUCCESS, "`{call}` answered {code}");
    };
    let stream = gpu.0.stream.raw();
    // SAFETY: the stream is `gpu`'s, live while it is; `graph` and `exec`
    // are local out-pointers, the graph freed once its exec is made.
    unsafe {
        ok(
            hip::rt::hipStreamBeginCapture(stream, THREAD_LOCAL),
            "hipStreamBeginCapture",
        );
        body();
        let mut graph: hip::rt::Graph = core::ptr::null_mut();
        ok(
            hip::rt::hipStreamEndCapture(stream, &raw mut graph),
            "hipStreamEndCapture",
        );
        let mut exec: hip::rt::GraphExec = core::ptr::null_mut();
        ok(
            hip::rt::hipGraphInstantiateWithFlags(&raw mut exec, graph, 0),
            "hipGraphInstantiateWithFlags",
        );
        hip::rt::hipGraphDestroy(graph);
        exec
    }
}

/// Launches a [`captured`] graph on `gpu`'s stream.
pub fn replay(gpu: &Gpu, exec: hip::rt::GraphExec) {
    // SAFETY: `exec` is a graph [`captured`] made and [`forget`] has not yet
    // freed; the stream is `gpu`'s.
    let code = unsafe { hip::rt::hipGraphLaunch(exec, gpu.0.stream.raw()) };
    assert_eq!(
        code,
        hip::sys::HIP_SUCCESS,
        "`hipGraphLaunch` answered {code}"
    );
}

/// Destroys a [`captured`] graph.
pub fn forget(exec: hip::rt::GraphExec) {
    // SAFETY: `exec` is a graph [`captured`] made, forgotten once, after its
    // last replay.
    unsafe {
        hip::rt::hipGraphExecDestroy(exec);
    }
}
