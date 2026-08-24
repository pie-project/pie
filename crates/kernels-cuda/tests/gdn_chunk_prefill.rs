//! The three GDN claim bodies against the composites they replace.
//!
//! `ssm.gated_delta_chunked` is the long half of this file and came first.
//! `ssm.gdn_prep` and `ssm.gated_delta` are at the bottom, and they are here
//! rather than in a file of their own because they share every piece of
//! machinery below — the slabs, the pool view, the deviation report — and
//! because the three points are one seam: the prep writes the fused decay
//! row, and both recurrences read it.
//!
//! # The step form's tests are BIT-EXACT, and that is the claim
//!
//! The chunked comparisons below carry tolerances because their references
//! are genuinely different scans. The step form's do not: the body fires the
//! same three kernels the composite fires, in the same order, over the same
//! planes. What changed is only WHERE the packed rows are cut — a kernel
//! that is told the packing rather than an executor that offsets a pointer —
//! so the answer must be identical to the bit at every token count. It is
//! the token count that is the point: the executor's cut was right at ONE
//! row and silently wrong at two, and these run at four.
//!
//! MUTATION-CHECKED. Rewriting [`Step::halves`] to build the reference the
//! way `Rect::column` described it — one pointer offset, row stride `v_h`
//! for bytes whose stride is `2 * v_h` — fails both: the prep test names
//! every column from row 1 on, and the step test reports exactly
//! `2048/8192` result elements bit-identical, which is row 0 of four and
//! nothing else. That is W1's measured symptom (row 0 correct, the rest
//! garbage) reproduced in two seconds instead of two model loads.
//!
//! The point is claimed by a BODY (`src/ssm.rs`), not by a `canon` retag, so
//! there is no routine whose row a table test could check. What can be
//! checked is the arithmetic, and against two references rather than one:
//!
//!  * **the legacy composite** — `repeat_interleave_heads_fp32` on the key
//!    planes and then `chunk_gated_delta_prefill_batched_cached`, which is
//!    what `model-legacy/src/qwen_3_5/forward/mod.rs:584-620` staged by hand.
//!    Its state lives in SHARED MEMORY as fp32 for the whole window, so it is
//!    close but not equal: a bf16 tolerance is the honest bar.
//!  * **the per-token form** — the same repeat, then
//!    `chunk_gated_delta_prefill_batched`, whose state round-trips bf16
//!    through HBM once per token. That is the rounding discipline the DECODE
//!    step is pinned to (`kernels/ssm/gated_delta_net.cuh:1370-1381`) and the
//!    one the body's fla arm reproduces in registers
//!    (`gated_delta_net.cuh:1414-1420` calls the two bit-identical), so this
//!    reference is held to a tight bar.
//!
//! Both are fed a prologue computed on the HOST — the l2-norm, the q scale,
//! the value widen, the `[g_log | beta]` cut — so the body's own staging
//! kernels are under test too, not just its scan.
//!
//! # The handoff is the part that rots
//!
//! Every case runs TWICE against the same slab, with a second window
//! continuing from the state the first left. A body that wrote the tail at
//! the wrong offset, in the wrong element, or in the wrong `(k, v)` order
//! passes a single-window comparison and fails here.

#![cfg(feature = "_cuda")]
#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels::points::Ssm;
use kernels::raises::Struct;
use kernels::routine::{Bind, Cache, Const, In, Out};
use kernels::Fire;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::jit::{Ctx, Launch};
use kernels_cuda::views::{RecurrentState, RecurrentView};

/// The device scratch is a process-global named-slab arena
/// (`jit::device::slabs`) sized for one fire at a time, which is what the
/// driver's stream serialization guarantees and the test harness's thread
/// pool does not. One guard per test restores the contract.
static FIRE: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ── the device, or a skip ────────────────────────────────────────────────

/// Run `f`, answering `None` if it panics, without printing a crash report.
///
/// `cudarc` is `fallback-dynamic-loading` and this crate has no `DT_NEEDED`
/// on `libcudart`: the first call `dlopen`s it and PANICS when no candidate
/// name resolves, so on a box with no CUDA at all the skip is unreachable
/// without catching. `driver-cuda/tests/common/mod.rs` says the same.
fn quietly<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Option<R> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    out.ok()
}

/// Bind device 0, or answer `false` and say why.
fn device_or_skip(what: &str) -> bool {
    let Some(count) = quietly(|| {
        let mut n: i32 = 0;
        let code = unsafe { rt::cudaGetDeviceCount(&raw mut n) };
        (code == rt::cudaError::cudaSuccess).then_some(n)
    }) else {
        eprintln!("skipping {what}: no CUDA runtime library on this machine");
        return false;
    };
    match count {
        Some(n) if n > 0 => {}
        _ => {
            eprintln!("skipping {what}: no CUDA device this build can drive");
            return false;
        }
    }
    // `cudaSetDevice` only records a thread-local ordinal; the driver-API
    // calls the JIT makes need the primary context up, which is what the
    // null free forces (`baker-smoke/src/dev.rs:88-94`).
    assert_eq!(
        unsafe { rt::cudaSetDevice(0) },
        rt::cudaError::cudaSuccess,
        "a device is present but cudaSetDevice(0) failed"
    );
    assert_eq!(
        unsafe { rt::cudaFree(core::ptr::null_mut()) },
        rt::cudaError::cudaSuccess,
        "a device is present but the primary context would not come up"
    );
    true
}

// ── device memory, freed when the run ends ───────────────────────────────

struct Slab {
    ptr: *mut c_void,
}

impl Slab {
    fn of(bytes: &[u8]) -> Slab {
        let mut ptr: *mut c_void = core::ptr::null_mut();
        assert_eq!(
            unsafe { rt::cudaMalloc(&raw mut ptr, bytes.len().max(1)) },
            rt::cudaError::cudaSuccess,
            "cudaMalloc({})",
            bytes.len()
        );
        let slab = Slab { ptr };
        slab.upload(bytes);
        slab
    }

    fn zeroed(bytes: usize) -> Slab {
        Slab::of(&vec![0u8; bytes])
    }

    fn upload(&self, src: &[u8]) {
        if src.is_empty() {
            return;
        }
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    self.ptr,
                    src.as_ptr().cast(),
                    src.len(),
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            },
            rt::cudaError::cudaSuccess,
            "host to device"
        );
    }

    fn download(&self, dst: &mut [u8]) {
        assert_eq!(
            unsafe { rt::cudaDeviceSynchronize() },
            rt::cudaError::cudaSuccess,
            "device synchronize"
        );
        assert_eq!(
            unsafe {
                rt::cudaMemcpy(
                    dst.as_mut_ptr().cast(),
                    self.ptr,
                    dst.len(),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            },
            rt::cudaError::cudaSuccess,
            "device to host"
        );
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { rt::cudaFree(self.ptr) };
        }
    }
}

// ── elements ─────────────────────────────────────────────────────────────

fn wide(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// `__float2bfloat16`: round to nearest, ties to even.
fn narrow(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// xorshift64*, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // [-1, 1)
        ((self.0 >> 40) as f32) / 8_388_608.0 - 1.0
    }

    /// A bf16 draw in `[-scale, scale)`, already rounded to the element the
    /// device will read.
    fn bf16(&mut self, scale: f32) -> u16 {
        narrow(self.next() * scale)
    }
}

// ── the case ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Case {
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
}

impl Case {
    const fn conv_dim(self) -> i32 {
        2 * self.k_h * self.k_d + self.v_h * self.v_d
    }

    const fn slot_stride(self) -> usize {
        (self.v_h * self.k_d * self.v_d) as usize
    }
}

/// One prefill window: the packed post-conv row, the fused decay row, and
/// the CSR that cuts them into requests.
struct Window {
    n: i32,
    r: i32,
    qkv: Vec<u16>,
    gates: Vec<f32>,
    indptr: Vec<i32>,
}

impl Window {
    fn draw(case: Case, lens: &[i32], rng: &mut Rng) -> Window {
        let n: i32 = lens.iter().sum();
        let mut indptr = vec![0i32];
        for len in lens {
            indptr.push(indptr[indptr.len() - 1] + len);
        }
        let qkv = (0..n as usize * case.conv_dim() as usize)
            .map(|_| rng.bf16(1.0))
            .collect();
        // `g_log` is `-exp(a_log) * softplus(..)` and therefore NEVER
        // positive: `exp(g_log)` is the decay and a decay above one is not a
        // number this recurrence ever sees. `beta` is a sigmoid, so it is
        // strictly inside the unit interval.
        let mut gates = vec![0.0f32; n as usize * 2 * case.v_h as usize];
        for t in 0..n as usize {
            for h in 0..case.v_h as usize {
                let row = t * 2 * case.v_h as usize;
                gates[row + h] = -0.25 * (rng.next() + 1.0);
                gates[row + case.v_h as usize + h] = 0.5 + 0.45 * rng.next();
            }
        }
        Window {
            n,
            r: lens.len() as i32,
            qkv,
            gates,
            indptr,
        }
    }

    /// The five f32 planes the scans take, computed on the host.
    ///
    /// A transcription of `qwen_gdn_qk_norm` and `qwen_gdn_v_gates`
    /// (`kernels/ssm/gated_delta_net_prep.cuh`) and of nothing else: the same
    /// `1e-6` inside the reciprocal square root, the same `1/sqrt(k_dim)` on
    /// the query alone, the same `[g_log | beta]` halves.
    fn prologue(&self, case: Case) -> Planes {
        let (k_h, v_h, k_d, v_d) = (case.k_h as usize, case.v_h as usize, case.k_d as usize, case.v_d as usize);
        let conv_dim = case.conv_dim() as usize;
        let k_dim = k_h * k_d;
        let n = self.n as usize;
        let q_scale = (k_d as f32).sqrt().recip();

        let mut q = vec![0.0f32; n * k_dim];
        let mut k = vec![0.0f32; n * k_dim];
        for t in 0..n {
            for h in 0..k_h {
                let qb = t * conv_dim + h * k_d;
                let kb = t * conv_dim + k_dim + h * k_d;
                let mut qs = 0.0f32;
                let mut ks = 0.0f32;
                for i in 0..k_d {
                    qs += wide(self.qkv[qb + i]) * wide(self.qkv[qb + i]);
                    ks += wide(self.qkv[kb + i]) * wide(self.qkv[kb + i]);
                }
                let qi = (qs + 1e-6).sqrt().recip() * q_scale;
                let ki = (ks + 1e-6).sqrt().recip();
                for i in 0..k_d {
                    q[(t * k_h + h) * k_d + i] = wide(self.qkv[qb + i]) * qi;
                    k[(t * k_h + h) * k_d + i] = wide(self.qkv[kb + i]) * ki;
                }
            }
        }

        let mut v = vec![0.0f32; n * v_h * v_d];
        let mut g_log = vec![0.0f32; n * v_h];
        let mut beta = vec![0.0f32; n * v_h];
        for t in 0..n {
            for h in 0..v_h {
                let vb = t * conv_dim + 2 * k_dim + h * v_d;
                for i in 0..v_d {
                    v[(t * v_h + h) * v_d + i] = wide(self.qkv[vb + i]);
                }
                g_log[t * v_h + h] = self.gates[t * 2 * v_h + h];
                beta[t * v_h + h] = self.gates[t * 2 * v_h + v_h + h];
            }
        }
        Planes { q, k, v, g_log, beta }
    }
}

struct Planes {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g_log: Vec<f32>,
    beta: Vec<f32>,
}

/// A slab and the view that addresses it, as the pool would hand it over.
struct Pool {
    _slab: Slab,
    _slots: Slab,
    view: RecurrentView,
}

impl Pool {
    fn new(case: Case, slots: i32, init: &[u16]) -> Pool {
        let slab = Slab::of(&bytes_of_u16(init));
        let ids: Vec<i32> = (0..slots).collect();
        let slot_ids = Slab::of(&bytes_of_i32(&ids));
        let view = RecurrentView {
            slab: slab.ptr,
            slot_ids: slot_ids.ptr.cast(),
            slot_stride_elems: case.slot_stride() as i64,
            slots: slot_ids.ptr.cast(),
            // `state` ALIASES `slab` on cuda; the swap plane is the shader
            // planes' spelling (`baker-smoke/src/smoke.rs:736-739`).
            state: slab.ptr,
            conv_state: core::ptr::null_mut(),
            new_conv_state: core::ptr::null_mut(),
            conv_slab: core::ptr::null_mut(),
            conv_stride: 0,
        };
        Pool {
            _slab: slab,
            _slots: slot_ids,
            view,
        }
    }

    fn raised(&self) -> In<Struct<RecurrentState>> {
        In {
            ptr: core::ptr::from_ref(&self.view),
            rows: 0,
            width: 0,
        }
    }

    fn cached(&self) -> Cache<Struct<RecurrentState>> {
        Cache {
            ptr: core::ptr::from_ref(&self.view),
        }
    }

    fn read(&self, elems: usize) -> Vec<u16> {
        let mut bytes = vec![0u8; elems * 2];
        self._slab.download(&mut bytes);
        as_u16(&bytes)
    }
}

// ── deviations ───────────────────────────────────────────────────────────

/// How far apart two answers are: the largest single gap, and the gap in the
/// 2-norm against the reference's own size.
///
/// A PER-ELEMENT relative bar is the wrong instrument here and reports noise
/// as failure: these arrays are recurrences whose smallest elements are a
/// couple of bf16 ulps from zero, and one flipped rounding decision there is
/// a 100% "relative error" on a number that carries no information. The
/// normalised 2-norm is the measure the whole array agrees on.
struct Gap {
    abs: f32,
    rms: f32,
    scale: f32,
}

impl core::fmt::Display for Gap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "abs {:.3e} rms {:.3e} (|ref| {:.3e})",
            self.abs, self.rms, self.scale
        )
    }
}

fn deviation(mine: &[f32], theirs: &[f32]) -> Gap {
    assert_eq!(mine.len(), theirs.len());
    let mut abs = 0.0f32;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in mine.iter().zip(theirs) {
        assert!(a.is_finite() && b.is_finite(), "a scan produced {a} vs {b}");
        abs = abs.max((a - b).abs());
        num += f64::from(a - b) * f64::from(a - b);
        den += f64::from(*b) * f64::from(*b);
    }
    let n = mine.len() as f64;
    Gap {
        abs,
        #[allow(clippy::cast_possible_truncation)]
        rms: if den > 0.0 { (num / den).sqrt() as f32 } else { 0.0 },
        #[allow(clippy::cast_possible_truncation)]
        scale: (den / n).sqrt() as f32,
    }
}

fn widened(v: &[u16]) -> Vec<f32> {
    v.iter().map(|b| wide(*b)).collect()
}

fn exact(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}

// ── the two references ───────────────────────────────────────────────────

/// The key planes broadcast up to the value heads, for the scans that index
/// q and k by `V_h`. At `k_h == v_h` this is the identity the legacy fired
/// anyway; past it, it is the whole reason the composite existed.
fn repeat(ctx: &Ctx<'_>, case: Case, n: i32, src: *const f32, dst: *mut f32) {
    kernels_cuda::ssm::repeat_interleave_heads_fp32(
        ctx,
        In {
            ptr: src,
            rows: n,
            width: case.k_h * case.k_d,
        },
        Out {
            ptr: dst,
            rows: n,
            width: case.v_h * case.k_d,
        },
        Const::new(case.k_h),
        Const::new(case.v_h),
        Const::new(case.k_d),
    )
    .expect("the repeat the composite stages");
}

/// `repeat_interleave` + `chunk_gated_delta_prefill_batched_cached`, which is
/// the composite `model-legacy` staged for its `TokensLE(cached_max)` arm.
fn composite_cached(
    ctx: &Ctx<'_>,
    case: Case,
    w: &Window,
    p: &Planes,
    pool: &Pool,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    indptr: *const i32,
    out: *mut f32,
) {
    let _ = p;
    kernels_cuda::ssm::chunk_gated_delta_prefill_batched_cached_state_bf16(
        ctx,
        In { ptr: q, rows: w.n, width: case.v_h * case.k_d },
        In { ptr: k, rows: w.n, width: case.v_h * case.k_d },
        In { ptr: v, rows: w.n, width: case.v_h * case.v_d },
        In { ptr: g, rows: w.n, width: case.v_h },
        In { ptr: beta, rows: w.n, width: case.v_h },
        Out { ptr: out, rows: w.n, width: case.v_h * case.v_d },
        Const::new(case.v_h),
        Const::new(case.k_d),
        Const::new(case.v_d),
        pool.raised(),
        In { ptr: indptr, rows: w.r, width: w.r + 1 },
        Const::new(true),
    )
    .expect("the cached arm of the legacy composite");
}

/// `repeat_interleave` + `chunk_gated_delta_prefill_batched`, the per-token
/// form: one bf16 round-trip through HBM per token, which is the decode
/// step's discipline.
fn composite_per_token(
    ctx: &Ctx<'_>,
    case: Case,
    w: &Window,
    pool: &Pool,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    indptr: *const u32,
    out: *mut f32,
) {
    const BLOCK: u32 = 128;

    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::state_bf16, false>",
        )
        .apply(
            Launch::grid(
                [w.r.unsigned_abs(), case.v_h.unsigned_abs(), 1],
                [BLOCK, 1, 1],
            )
            .smem(2 * case.k_d.unsigned_abs() * 4),
        ),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            g.arg(),
            beta.arg(),
            pool.view.slab.arg(),
            pool.view.slot_ids.arg(),
            indptr.arg(),
            pool.view.slot_stride_elems.arg(),
            out.arg(),
            case.v_h.arg(),
            case.k_d.arg(),
            case.v_d.arg(),
        ],
    )
    .expect("the per-token arm of the legacy composite");
}

// ── the run ──────────────────────────────────────────────────────────────

struct Answer {
    out: Vec<f32>,
    state: Vec<u16>,
}

/// Fire the point, and the two references, over one window against three
/// slabs that started life identical.
fn windows(case: Case, lens: &[&[i32]], seed: u64) -> Vec<[Answer; 3]> {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let mut rng = Rng(seed);

    let slots = lens[0].len() as i32;
    let state_elems = slots as usize * case.slot_stride();
    let state0: Vec<u16> = (0..state_elems).map(|_| rng.bf16(0.5)).collect();

    let mine = Pool::new(case, slots, &state0);
    let cached = Pool::new(case, slots, &state0);
    let per_token = Pool::new(case, slots, &state0);

    let mut answers = Vec::new();
    for lens in lens {
        assert_eq!(lens.len(), slots as usize, "one slot per request, per window");
        let w = Window::draw(case, lens, &mut rng);
        let p = w.prologue(case);

        let d_qkv = Slab::of(&bytes_of_u16(&w.qkv));
        let d_gates = Slab::of(&bytes_of_f32(&w.gates));
        let d_indptr = Slab::of(&bytes_of_i32(&w.indptr));
        // `z` is the OUT-NORM's gate; the point declares it and spends
        // nothing, so it is drawn and handed over unread.
        let d_z = Slab::of(&bytes_of_u16(
            &(0..w.n as usize * (case.v_h * case.v_d) as usize)
                .map(|_| rng.bf16(1.0))
                .collect::<Vec<_>>(),
        ));

        let out_elems = w.n as usize * (case.v_h * case.v_d) as usize;
        let d_out_mine = Slab::zeroed(out_elems * 4);
        let d_out_cached = Slab::zeroed(out_elems * 4);
        let d_out_per_token = Slab::zeroed(out_elems * 4);

        // The point, through the trait: what a dispatch would call.
        ctx.gated_delta_chunked::<bf16>(
            In {
                ptr: d_qkv.ptr.cast::<bf16>(),
                rows: w.n,
                width: case.conv_dim(),
            },
            In {
                ptr: d_indptr.ptr.cast::<i32>(),
                rows: w.r,
                width: w.r + 1,
            },
            In {
                ptr: d_z.ptr.cast::<bf16>(),
                rows: w.n,
                width: case.v_h * case.v_d,
            },
            In {
                ptr: d_gates.ptr.cast::<f32>(),
                rows: w.n,
                width: 2 * case.v_h,
            },
            mine.cached(),
            case.k_h.unsigned_abs(),
            case.v_h.unsigned_abs(),
            case.k_d.unsigned_abs(),
            case.v_d.unsigned_abs(),
            Out {
                ptr: d_out_mine.ptr.cast::<f32>(),
                rows: w.n,
                width: case.v_h * case.v_d,
            },
        )
        .expect("ssm.gated_delta_chunked");

        // The references, from the host prologue.
        let d_q = Slab::of(&bytes_of_f32(&p.q));
        let d_k = Slab::of(&bytes_of_f32(&p.k));
        let d_v = Slab::of(&bytes_of_f32(&p.v));
        let d_g = Slab::of(&bytes_of_f32(&p.g_log));
        let d_beta = Slab::of(&bytes_of_f32(&p.beta));
        let wide_elems = w.n as usize * (case.v_h * case.k_d) as usize;
        let d_qr = Slab::zeroed(wide_elems * 4);
        let d_kr = Slab::zeroed(wide_elems * 4);
        repeat(&ctx, case, w.n, d_q.ptr.cast(), d_qr.ptr.cast());
        repeat(&ctx, case, w.n, d_k.ptr.cast(), d_kr.ptr.cast());

        composite_cached(
            &ctx,
            case,
            &w,
            &p,
            &cached,
            d_qr.ptr.cast(),
            d_kr.ptr.cast(),
            d_v.ptr.cast(),
            d_g.ptr.cast(),
            d_beta.ptr.cast(),
            d_indptr.ptr.cast(),
            d_out_cached.ptr.cast(),
        );
        composite_per_token(
            &ctx,
            case,
            &w,
            &per_token,
            d_qr.ptr.cast(),
            d_kr.ptr.cast(),
            d_v.ptr.cast(),
            d_g.ptr.cast(),
            d_beta.ptr.cast(),
            d_indptr.ptr.cast(),
            d_out_per_token.ptr.cast(),
        );

        let read = |slab: &Slab| {
            let mut bytes = vec![0u8; out_elems * 4];
            slab.download(&mut bytes);
            as_f32(&bytes)
        };
        answers.push([
            Answer {
                out: read(&d_out_mine),
                state: mine.read(state_elems),
            },
            Answer {
                out: read(&d_out_cached),
                state: cached.read(state_elems),
            },
            Answer {
                out: read(&d_out_per_token),
                state: per_token.read(state_elems),
            },
        ]);
    }
    answers
}

/// Which scan the body's branch lands this shape on, and therefore which
/// reference it is supposed to AGREE with.
///
/// The two references are two rounding disciplines, not one right answer and
/// one approximation:
///
/// * `Fla` and `Repeat` round the state to bf16 twice per token, which is
///   what the per-token kernel does through HBM.
/// * `WarpTiled` carries fp32 state in registers for the whole window and
///   rounds once at the end, which is what the cached kernel does in shared
///   memory.
///
/// So the bars swap with the arm. An arm asserted against the wrong
/// reference would be asserting that two different recurrences agree, and
/// the only way to pass it would be to loosen the bar until it proved
/// nothing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// `k_dim <= 128` and `128 | v_dim`.
    Fla,
    /// Past fla's `v_dim`, inside `k_dim <= 256`.
    WarpTiled,
    /// Past both: the key planes broadcast, then the per-token form.
    Repeat,
}

fn check(name: &str, case: Case, arm: Arm, lens: &[&[i32]], seed: u64) {
    let runs = windows(case, lens, seed);
    let mut failures: Vec<String> = Vec::new();
    for (i, [mine, cached, per_token]) in runs.iter().enumerate() {
        let out_c = deviation(&mine.out, &cached.out);
        let out_p = deviation(&mine.out, &per_token.out);
        let st_c = deviation(&widened(&mine.state), &widened(&cached.state));
        let st_p = deviation(&widened(&mine.state), &widened(&per_token.state));
        let elems = mine.state.len();
        eprintln!("{name} window {i}:");
        eprintln!(
            "    out   vs per-token  {out_p}\n    state vs per-token  {st_p}, {}/{elems} bit-identical",
            exact(&mine.state, &per_token.state)
        );
        eprintln!(
            "    out   vs cached     {out_c}\n    state vs cached     {st_c}, {}/{elems} bit-identical",
            exact(&mine.state, &cached.state)
        );

        let mut want = |ok: bool, what: String| {
            if !ok {
                failures.push(format!("{name} window {i}: {what}"));
            }
        };
        // THE MATCHING REFERENCE IS THE STRICT ONE. Same recurrence, same
        // rounding, same order of summation over `k` -- the only gap left is
        // the last bits of the prologue, the host's `1.0/sqrt(..)` against
        // the device's `rsqrtf`, and what that buys is a handful of flipped
        // roundings rather than a trajectory. `1e-4` in the normalised
        // 2-norm is two orders below bf16's own resolution.
        //
        // THE OTHER ONE IS THE BF16 BAR. It is a genuinely different
        // trajectory: `2e-2` is what a few dozen tokens of divergent
        // eight-mantissa-bit rounding costs, and it is the bar the gate asks
        // for.
        let (strict_out, strict_state, strict_bits, loose_out, loose_state) = match arm {
            Arm::Fla | Arm::Repeat => (
                (&out_p, "per-token"),
                (&st_p, "per-token"),
                exact(&mine.state, &per_token.state),
                (&out_c, "cached"),
                (&st_c, "cached"),
            ),
            Arm::WarpTiled => (
                (&out_c, "cached"),
                (&st_c, "cached"),
                exact(&mine.state, &cached.state),
                (&out_p, "per-token"),
                (&st_p, "per-token"),
            ),
        };
        want(
            strict_out.0.rms < 1e-4,
            format!("out vs {} {}", strict_out.1, strict_out.0),
        );
        want(
            strict_state.0.rms < 1e-4,
            format!("state vs {} {}", strict_state.1, strict_state.0),
        );
        want(
            strict_bits * 1000 >= elems * 995,
            format!(
                "only {strict_bits}/{elems} state elements bit-identical to the {} composite",
                strict_state.1
            ),
        );
        want(
            loose_out.0.rms < 2e-2,
            format!("out vs {} {}", loose_out.1, loose_out.0),
        );
        want(
            loose_state.0.rms < 2e-2,
            format!("state vs {} {}", loose_state.1, loose_state.0),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ── the cases ────────────────────────────────────────────────────────────

/// qwen3.5-0.8B's mixer: `k_heads == v_heads`, so the repeat is an identity
/// and what is under test is the prologue, the fla scan and the handoff.
#[test]
fn the_0_8b_shape_matches_the_composite() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.gated_delta_chunked at the 0.8B shape") {
        return;
    }
    check(
        "0.8b 16/16/128/128",
        Case {
            k_h: 16,
            v_h: 16,
            k_d: 128,
            v_d: 128,
        },
        Arm::Fla,
        &[&[21, 15], &[9, 12]],
        0x9E37_79B9_7F4A_7C15,
    );
}

/// qwen3.5-3B and -A3B: `v_heads == 2 * k_heads`, the GQA the composite
/// needed `repeat_interleave` for and the body's fla arm does not.
#[test]
fn the_gqa_shape_matches_the_composite() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.gated_delta_chunked at the GQA shape") {
        return;
    }
    check(
        "3b 16/32/128/128",
        Case {
            k_h: 16,
            v_h: 32,
            k_d: 128,
            v_d: 128,
        },
        Arm::Fla,
        &[&[19, 17], &[13, 8]],
        0xD1B5_4A32_D192_ED03,
    );
}

/// The second arm, reached by a `v_dim` that `BV = 128` does not divide.
/// Nothing in this tree ships this shape; the branch is here because the
/// kernel's bound is, and an arm nothing exercises is an arm that rots.
#[test]
fn a_value_width_fla_cannot_tile_takes_the_warp_tiled_arm() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.gated_delta_chunked past fla's value tile") {
        return;
    }
    check(
        "warp-tiled 8/16/64/96",
        Case {
            k_h: 8,
            v_h: 16,
            k_d: 64,
            v_d: 96,
        },
        Arm::WarpTiled,
        &[&[11, 7], &[5, 9]],
        0xB5AD_4ECE_DA10_1373,
    );
}

/// The third arm, reached by a `k_dim` past `MAX_K_PER_LANE * 32`. This is
/// the only arm that stages the `repeat_interleave` the legacy composite
/// always staged -- the two GQA-native scans above do not need it -- so it
/// is also the only one whose reference shares a kernel with the body.
#[test]
fn a_key_width_past_both_bounds_takes_the_repeat_arm() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.gated_delta_chunked past both GQA-native bounds") {
        return;
    }
    check(
        "repeat 4/8/320/64",
        Case {
            k_h: 4,
            v_h: 8,
            k_d: 320,
            v_d: 64,
        },
        Arm::Repeat,
        &[&[13, 6], &[4, 10]],
        0x2545_F491_4F6C_DD1D,
    );
}

// ── the step form, and the packed rows it no longer cuts ─────────────────

/// One decode fire: `R` requests, one token each.
///
/// The step form's row count IS its request count — that is the fact
/// (`qo_one`) the lane carrying the statement was selected on — so there is
/// no CSR here and `n` does double duty, exactly as the claim body reads it.
struct Step {
    n: i32,
    qkv: Vec<u16>,
    /// `[b | a]`, packed: one row of `2 * v_h` per token, which is what
    /// `gemm.matmul` writes and what `ssm.gdn_prep` declares.
    ba: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias: Vec<u16>,
}

impl Step {
    fn draw(case: Case, n: i32, rng: &mut Rng) -> Step {
        let qkv = (0..n as usize * case.conv_dim() as usize)
            .map(|_| rng.bf16(1.0))
            .collect();
        let ba = (0..n as usize * 2 * case.v_h as usize)
            .map(|_| rng.bf16(1.0))
            .collect();
        // `A_log` is exponentiated and negated into the decay, so a large
        // positive one is a decay of zero and measures nothing; the
        // checkpoint's are small.
        let a_log = (0..case.v_h as usize).map(|_| 0.5 * rng.next()).collect();
        let dt_bias = (0..case.v_h as usize).map(|_| rng.bf16(0.5)).collect();
        Step {
            n,
            qkv,
            ba,
            a_log,
            dt_bias,
        }
    }

    /// The two COMPACT `[n, v_h]` halves of the packed projection.
    ///
    /// This is the reference the executor used to build with pointer
    /// arithmetic, and building it here by COPY is the whole difference:
    /// `ba.column(v_h, v_h)` answered a rectangle whose row stride is
    /// `v_h` for bytes whose stride is `2 * v_h`, which is this array only
    /// when `n == 1`.
    fn halves(&self, case: Case) -> (Vec<u16>, Vec<u16>) {
        let v_h = case.v_h as usize;
        let mut b = vec![0u16; self.n as usize * v_h];
        let mut a = vec![0u16; self.n as usize * v_h];
        for t in 0..self.n as usize {
            for h in 0..v_h {
                b[t * v_h + h] = self.ba[t * 2 * v_h + h];
                a[t * v_h + h] = self.ba[t * 2 * v_h + v_h + h];
            }
        }
        (b, a)
    }
}

/// The five compact f32 planes `qwen_gdn_post_conv_prep_bf16` writes, on the
/// device, from honestly-cut halves. The composite both step tests measure
/// against.
struct Composite {
    q_norm: Slab,
    k_norm: Slab,
    v: Slab,
    g_log: Slab,
    beta: Slab,
}

fn composite_prep(ctx: &Ctx<'_>, case: Case, step: &Step, qkv: &Slab) -> Composite {
    let (b, a) = step.halves(case);
    let d_b = Slab::of(&bytes_of_u16(&b));
    let d_a = Slab::of(&bytes_of_u16(&a));
    let d_a_log = Slab::of(&bytes_of_f32(&step.a_log));
    let d_dt = Slab::of(&bytes_of_u16(&step.dt_bias));
    let key = step.n as usize * (case.k_h * case.k_d) as usize;
    let val = step.n as usize * (case.v_h * case.v_d) as usize;
    let decay = step.n as usize * case.v_h as usize;
    let out = Composite {
        q_norm: Slab::zeroed(key * 4),
        k_norm: Slab::zeroed(key * 4),
        v: Slab::zeroed(val * 4),
        g_log: Slab::zeroed(decay * 4),
        beta: Slab::zeroed(decay * 4),
    };
    kernels_cuda::driver_internal::qwen_gdn_post_conv_prep_bf16(
        ctx,
        In {
            ptr: qkv.ptr.cast::<bf16>(),
            rows: step.n,
            width: case.conv_dim(),
        },
        In {
            ptr: d_a.ptr.cast::<bf16>(),
            rows: step.n,
            width: case.v_h,
        },
        In {
            ptr: d_b.ptr.cast::<bf16>(),
            rows: step.n,
            width: case.v_h,
        },
        Const::new(d_a_log.ptr.cast::<f32>().cast_const()),
        Const::new(d_dt.ptr.cast::<bf16>().cast_const()),
        Out {
            ptr: out.q_norm.ptr.cast::<f32>(),
            rows: step.n,
            width: case.k_h * case.k_d,
        },
        Out {
            ptr: out.k_norm.ptr.cast::<f32>(),
            rows: step.n,
            width: case.k_h * case.k_d,
        },
        Out {
            ptr: out.v.ptr.cast::<f32>(),
            rows: step.n,
            width: case.v_h * case.v_d,
        },
        Out {
            ptr: out.g_log.ptr.cast::<f32>(),
            rows: step.n,
            width: case.v_h,
        },
        Out {
            ptr: out.beta.ptr.cast::<f32>(),
            rows: step.n,
            width: case.v_h,
        },
        Const::new(case.k_h),
        Const::new(case.v_h),
        Const::new(case.k_d),
        Const::new(case.v_d),
        Const::new(case.conv_dim()),
    )
    .expect("the composite prep");
    out
}

fn read_f32(slab: &Slab, elems: usize) -> Vec<f32> {
    let mut raw = vec![0u8; elems * 4];
    slab.download(&mut raw);
    as_f32(&raw)
}

/// `ssm.gdn_prep` writes, packed, exactly what the composite computes
/// compact — at four rows, where the executor's old pointer cut did not.
#[test]
fn the_prep_point_writes_the_fused_row_the_composite_computes() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.gdn_prep") {
        return;
    }
    let case = Case {
        k_h: 16,
        v_h: 16,
        k_d: 128,
        v_d: 128,
    };
    const N: i32 = 4;

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let mut rng = Rng(0x5DEE_CE66_D1EC_1234);
    let step = Step::draw(case, N, &mut rng);

    let d_qkv = Slab::of(&bytes_of_u16(&step.qkv));
    let reference = composite_prep(&ctx, case, &step, &d_qkv);

    let d_ba = Slab::of(&bytes_of_u16(&step.ba));
    let d_a_log = Slab::of(&bytes_of_f32(&step.a_log));
    let d_dt = Slab::of(&bytes_of_u16(&step.dt_bias));
    let fused = (N * 2 * case.v_h) as usize;
    let d_gates = Slab::zeroed(fused * 4);
    ctx.gdn_prep::<bf16>(
        In {
            ptr: d_ba.ptr.cast::<bf16>(),
            rows: N,
            width: 2 * case.v_h,
        },
        Const::new(d_dt.ptr.cast::<bf16>().cast_const()),
        Const::new(d_a_log.ptr.cast::<f32>().cast_const()),
        Out {
            ptr: d_gates.ptr.cast::<f32>(),
            rows: N,
            width: 2 * case.v_h,
        },
    )
    .expect("ssm.gdn_prep");

    let gates = read_f32(&d_gates, fused);
    let decay = (N * case.v_h) as usize;
    let g_log = read_f32(&reference.g_log, decay);
    let beta = read_f32(&reference.beta, decay);

    // THE PACKING IS THE ASSERTION. `[g_log | beta]` per token, and the
    // reference's compact planes read row by row — a body that wrote the
    // two halves as blocks, or a kernel that strided by `v_h` instead of
    // `2 * v_h`, agrees with this only at `N == 1`.
    let v_h = case.v_h as usize;
    let mut wrong = Vec::new();
    for t in 0..N as usize {
        for h in 0..v_h {
            let (mine_g, mine_b) = (gates[t * 2 * v_h + h], gates[t * 2 * v_h + v_h + h]);
            let (want_g, want_b) = (g_log[t * v_h + h], beta[t * v_h + h]);
            if mine_g.to_bits() != want_g.to_bits() {
                wrong.push(format!("g_log[{t},{h}] {mine_g:e} vs {want_g:e}"));
            }
            if mine_b.to_bits() != want_b.to_bits() {
                wrong.push(format!("beta[{t},{h}] {mine_b:e} vs {want_b:e}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the prep point and the composite disagree at {} of {} columns:\n{}",
        wrong.len(),
        2 * decay,
        wrong.iter().take(8).cloned().collect::<Vec<_>>().join("\n"),
    );
    // A row of zeroes would pass every comparison above.
    assert!(
        gates.iter().any(|x| *x != 0.0),
        "the fused decay row is all zero; nothing was written",
    );
}

/// `ssm.gated_delta` answers what the composite answers — the same three
/// kernels, at four rows, to the bit.
///
/// The old executor path IS the composite fed a pointer cut: it handed
/// `qwen_gdn_post_conv_prep_bf16` two rectangles taken `v_h` elements into
/// the packed `[b | a]` row and called them `[n, v_h]`, then handed the step
/// scan two more taken out of `gates` the same way. This runs the composite
/// on the halves COPIED out honestly, which is what those cuts describe at
/// one row and misdescribe at four.
#[test]
fn the_step_point_matches_the_composite_at_many_rows() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.gated_delta") {
        return;
    }
    let case = Case {
        k_h: 16,
        v_h: 16,
        k_d: 128,
        v_d: 128,
    };
    const N: i32 = 4;

    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let mut rng = Rng(0x14057B7E_F767_814F);
    let step = Step::draw(case, N, &mut rng);

    // ONE SLOT PER REQUEST, and both pools start from the same bytes: a
    // recurrence that read the right planes and wrote the wrong slot passes
    // an output comparison and fails the state one.
    let state_elems = N as usize * case.slot_stride();
    let state0: Vec<u16> = (0..state_elems).map(|_| rng.bf16(0.5)).collect();
    let mine = Pool::new(case, N, &state0);
    let theirs = Pool::new(case, N, &state0);

    let d_qkv = Slab::of(&bytes_of_u16(&step.qkv));
    let d_ba = Slab::of(&bytes_of_u16(&step.ba));
    let d_a_log = Slab::of(&bytes_of_f32(&step.a_log));
    let d_dt = Slab::of(&bytes_of_u16(&step.dt_bias));
    // `z` is the OUT-NORM's gate; the point declares it and spends nothing.
    let d_z = Slab::of(&bytes_of_u16(
        &(0..(N * case.v_h * case.v_d) as usize)
            .map(|_| rng.bf16(1.0))
            .collect::<Vec<_>>(),
    ));

    let fused = (N * 2 * case.v_h) as usize;
    let d_gates = Slab::zeroed(fused * 4);
    ctx.gdn_prep::<bf16>(
        In {
            ptr: d_ba.ptr.cast::<bf16>(),
            rows: N,
            width: 2 * case.v_h,
        },
        Const::new(d_dt.ptr.cast::<bf16>().cast_const()),
        Const::new(d_a_log.ptr.cast::<f32>().cast_const()),
        Out {
            ptr: d_gates.ptr.cast::<f32>(),
            rows: N,
            width: 2 * case.v_h,
        },
    )
    .expect("ssm.gdn_prep");

    let out_elems = (N * case.v_h * case.v_d) as usize;
    let d_out_mine = Slab::zeroed(out_elems * 4);
    ctx.gated_delta::<bf16>(
        In {
            ptr: d_qkv.ptr.cast::<bf16>(),
            rows: N,
            width: case.conv_dim(),
        },
        In {
            ptr: d_z.ptr.cast::<bf16>(),
            rows: N,
            width: case.v_h * case.v_d,
        },
        In {
            ptr: d_gates.ptr.cast::<f32>(),
            rows: N,
            width: 2 * case.v_h,
        },
        mine.cached(),
        case.k_h.unsigned_abs(),
        case.v_h.unsigned_abs(),
        case.k_d.unsigned_abs(),
        case.v_d.unsigned_abs(),
        Out {
            ptr: d_out_mine.ptr.cast::<f32>(),
            rows: N,
            width: case.v_h * case.v_d,
        },
    )
    .expect("ssm.gated_delta");

    let reference = composite_prep(&ctx, case, &step, &d_qkv);
    let d_out_theirs = Slab::zeroed(out_elems * 4);
    kernels_cuda::ssm::recurrent_gated_delta_step_batched_gqa_state_bf16(
        &ctx,
        In {
            ptr: reference.q_norm.ptr.cast::<f32>().cast_const(),
            rows: N,
            width: case.k_h * case.k_d,
        },
        In {
            ptr: reference.k_norm.ptr.cast::<f32>().cast_const(),
            rows: N,
            width: case.k_h * case.k_d,
        },
        In {
            ptr: reference.v.ptr.cast::<f32>().cast_const(),
            rows: N,
            width: case.v_h * case.v_d,
        },
        In {
            ptr: reference.g_log.ptr.cast::<f32>().cast_const(),
            rows: N,
            width: case.v_h,
        },
        In {
            ptr: reference.beta.ptr.cast::<f32>().cast_const(),
            rows: N,
            width: case.v_h,
        },
        Out {
            ptr: d_out_theirs.ptr.cast::<f32>(),
            rows: N,
            width: case.v_h * case.v_d,
        },
        Const::new(case.k_h),
        Const::new(case.v_h),
        Const::new(case.k_d),
        Const::new(case.v_d),
        Const::new(N),
        theirs.raised(),
    )
    .expect("the composite step");

    let out_mine = read_f32(&d_out_mine, out_elems);
    let out_theirs = read_f32(&d_out_theirs, out_elems);
    let st_mine = mine.read(state_elems);
    let st_theirs = theirs.read(state_elems);

    eprintln!(
        "step 16/16/128/128 x{N}: out {}, state {} ({}/{state_elems} bit-identical)",
        deviation(&out_mine, &out_theirs),
        deviation(&widened(&st_mine), &widened(&st_theirs)),
        exact(&st_mine, &st_theirs),
    );

    let same_out = out_mine
        .iter()
        .zip(&out_theirs)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    assert_eq!(
        same_out, out_elems,
        "only {same_out}/{out_elems} result elements are bit-identical to the \
         composite; the step body fires the same kernels and must agree exactly",
    );
    assert_eq!(
        exact(&st_mine, &st_theirs),
        state_elems,
        "the recurrent tails differ; the state a batched step leaves behind is \
         what the next fire reads",
    );
    // Both pools starting from the same bytes means an unfired recurrence
    // would pass both equalities above.
    assert_ne!(
        st_mine, state0,
        "the state slab is unchanged; nothing ran",
    );
}
