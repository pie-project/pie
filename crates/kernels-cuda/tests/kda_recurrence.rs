//! `ssm.kda_step` and `ssm.kda_chunked` against what they replace.
//!
//! Both points are claimed by a BODY (`src/ssm.rs`), not by a `canon` retag,
//! so there is no routine whose row a table test could check. What can be
//! checked is the arithmetic, and each point has a different thing to prove.
//!
//! # The step: against the sequence the legacy fired
//!
//! `model-legacy/src/kimi_k3/forward/mod.rs:199-283` staged the KDA mixer by
//! hand. The new text keeps the projections and folds three of them into one
//! packed bank, one packed conv; what is left over is what the body now owns,
//! and it is exactly five launches:
//!
//! ```text
//!   l2norm_scale_bf16_to_fp32(q)   ─┐
//!   l2norm_scale_bf16_to_fp32(k)    ├─ the body's `kda_qkv_prep`, one launch
//!   bf16_to_fp32(v)                ─┘
//!   kda_gate_beta(f, b, a_log, dt_bias)
//!   kda_recurrent_step_batched(...)
//! ```
//!
//! [`the_step_is_the_legacy_sequence`] fires that sequence AS THE LEGACY DID
//! — the three routines above, on compact planes the host cuts out of the
//! packed row with a copy and no arithmetic — and compares it to the point.
//! The bar is BIT-IDENTITY on both the result and the state slab, because
//! the body's one launch is a transcription of those three and not an
//! improvement on them. A tolerance here would be a way not to notice the
//! difference between the two.
//!
//! # The chunked form: against a step loop, because nothing else has run it
//!
//! `kda_prefill_batched` has never been fired by anything.
//! `.wiki/driver/new-horizon.md:7652` lists it among the seven symbols in the
//! tree with "nothing at all — not a test, not a comment, only the row and
//! its wrapper", and the legacy KDA leg ran the STEP kernel even on prefill,
//! one launch per token. So `ssm.kda_chunked` claiming it is a claim on an
//! unrun kernel, and [`the_window_is_the_step_loop`] is what makes it a
//! measurement instead: the window against a per-request loop of
//! `ssm.kda_step` fires, one token at a time, over the same drawn operands.
//!
//! W2'S ROUNDING-TRAJECTORY LAW is what that loop is really testing. A
//! prefill tail must leave the state the decode step would have left, or the
//! first decoded token continues a trajectory the window never walked. The
//! gated-delta family had to CHOOSE an arm to satisfy it; KDA does not, and
//! the reason is visible in `kernels/ssm/kda.cuh` — both kernels carry fp32
//! state through the slab and round nowhere — but "visible in the source" is
//! how the two bugs W7 found got shipped. The bar is again bit-identity.
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
use kernels::routine::{Cache, Const, In, Out};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
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
/// without catching. `gdn_chunk_prefill.rs` says the same.
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
    // null free forces.
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

    fn f32s(&self, elems: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; elems * 4];
        self.download(&mut bytes);
        as_f32(&bytes)
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
    h: i32,
    d: i32,
    /// The l2 norm's epsilon, which the statement states and the body
    /// spends — `norm_eps_micro: 10` on the shipped kimi facts.
    eps: f32,
}

impl Case {
    /// One plane of the packed row: `heads * head_dim`.
    const fn w(self) -> i32 {
        self.h * self.d
    }

    /// `[heads, head_dim, head_dim]` f32, per slot.
    const fn slot_stride(self) -> usize {
        (self.h * self.d * self.d) as usize
    }
}

/// One fire's operands: the packed post-convolution row, the forget and beta
/// projections, and the CSR that cuts them into requests.
struct Draw {
    n: i32,
    r: i32,
    mixed: Vec<u16>,
    f: Vec<u16>,
    b: Vec<u16>,
    indptr: Vec<i32>,
}

impl Draw {
    fn of(case: Case, lens: &[i32], rng: &mut Rng) -> Draw {
        let n: i32 = lens.iter().sum();
        let mut indptr = vec![0i32];
        for len in lens {
            indptr.push(indptr[indptr.len() - 1] + len);
        }
        let w = case.w() as usize;
        Draw {
            n,
            r: lens.len() as i32,
            mixed: (0..n as usize * 3 * w).map(|_| rng.bf16(1.0)).collect(),
            // The forget projection rides a softplus and then a negated
            // `exp(a_log)`; drawn wide it saturates the decay to zero and
            // every recurrence in the comparison agrees trivially. Half a
            // unit keeps `exp(g)` inside `(0.4, 1)` at the `a_log` below.
            f: (0..n as usize * w).map(|_| rng.bf16(0.5)).collect(),
            b: (0..n as usize * case.h as usize)
                .map(|_| rng.bf16(2.0))
                .collect(),
            indptr,
        }
    }

    /// The three COMPACT bf16 planes the legacy's separate convs left
    /// behind, cut out of the packed row here. A copy and no arithmetic:
    /// every element is the same `u16` on both sides, so nothing the
    /// comparison measures can come from this function.
    fn planes(&self, case: Case) -> [Vec<u16>; 3] {
        let w = case.w() as usize;
        let cut = |p: usize| -> Vec<u16> {
            (0..self.n as usize)
                .flat_map(|t| self.mixed[t * 3 * w + p * w..t * 3 * w + (p + 1) * w].to_vec())
                .collect()
        };
        [cut(0), cut(1), cut(2)]
    }
}

/// The decay weights, which are F32 on both slots because
/// `ssm/kda.cuh`'s `kda_gate_beta` takes both as `const float*`.
struct Decay {
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
}

impl Decay {
    fn of(case: Case, rng: &mut Rng) -> Decay {
        Decay {
            // `-exp(a_log[h]) * softplus(..)` is the gate, so `a_log`
            // decides how fast the state forgets. Around zero, `exp` is
            // one and the decay stays in a range where a trajectory
            // difference would still be visible many tokens later.
            a_log: (0..case.h as usize).map(|_| rng.next() * 0.5).collect(),
            dt_bias: (0..(case.h * case.d) as usize)
                .map(|_| rng.next() * 0.5)
                .collect(),
        }
    }
}

/// A slab and the view that addresses it, as the pool would hand it over.
///
/// F32, not bf16: `kda_recurrent_step_batched` takes `float* state_base`
/// where every gated-delta scan takes `state_bf16`.
struct Pool {
    slab: Slab,
    _slots: Slab,
    slot_ids: *const i32,
    stride: i64,
}

impl Pool {
    fn new(case: Case, slots: i32, init: &[f32]) -> Pool {
        let slab = Slab::of(&bytes_of_f32(init));
        let ids: Vec<i32> = (0..slots).collect();
        let slot_ids = Slab::of(&bytes_of_i32(&ids));
        Pool {
            slab,
            slot_ids: slot_ids.ptr.cast(),
            _slots: slot_ids,
            stride: case.slot_stride() as i64,
        }
    }

    /// The view a statement's cache row resolves to. `first` is the slot
    /// index the fire's row zero addresses — the whole pool for a batched
    /// fire, and one request for a step-loop reference, which is what
    /// `slot_ids[r]` reads.
    fn view(&self, first: i32) -> RecurrentView {
        RecurrentView {
            slab: self.slab.ptr,
            slot_ids: unsafe { self.slot_ids.add(first as usize) },
            slot_stride_elems: self.stride,
            slots: self.slot_ids,
            // `state` ALIASES `slab` on cuda; the swap plane is the shader
            // planes' spelling.
            state: self.slab.ptr,
            conv_state: core::ptr::null_mut(),
            new_conv_state: core::ptr::null_mut(),
            conv_slab: core::ptr::null_mut(),
            conv_stride: 0,
        }
    }

    fn read(&self, elems: usize) -> Vec<f32> {
        self.slab.f32s(elems)
    }
}

fn cached(view: &RecurrentView) -> Cache<Struct<RecurrentState>> {
    Cache {
        ptr: core::ptr::from_ref(view),
    }
}

fn raised(view: &RecurrentView) -> In<Struct<RecurrentState>> {
    In {
        ptr: core::ptr::from_ref(view),
        rows: 0,
        width: 0,
    }
}

// ── deviations ───────────────────────────────────────────────────────────

/// How far apart two answers are, how many of them are the SAME BITS, and
/// HOW BIG THE REFERENCE IS.
///
/// The last one is the guard against a vacuous pass. Two arrays of zeros are
/// bit-identical, and every way this comparison could go wrong by accident —
/// a fire that refused into a `?` nobody read, a scratch plane that stayed
/// zeroed, an out slab nothing wrote — produces exactly that. So the scale
/// is measured and asserted, not printed for interest.
struct Gap {
    abs: f32,
    same: usize,
    of: usize,
    scale: f32,
}

impl core::fmt::Display for Gap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}/{} bit-identical, worst |Δ| {:.3e} (rms |ref| {:.3e})",
            self.same, self.of, self.abs, self.scale
        )
    }
}

fn deviation(mine: &[f32], theirs: &[f32]) -> Gap {
    assert_eq!(mine.len(), theirs.len());
    let mut abs = 0.0f32;
    let mut same = 0usize;
    let mut den = 0.0f64;
    for (a, b) in mine.iter().zip(theirs) {
        assert!(a.is_finite() && b.is_finite(), "a scan produced {a} vs {b}");
        abs = abs.max((a - b).abs());
        den += f64::from(*b) * f64::from(*b);
        if a.to_bits() == b.to_bits() {
            same += 1;
        }
    }
    Gap {
        abs,
        same,
        of: mine.len(),
        #[allow(clippy::cast_possible_truncation)]
        scale: (den / mine.len() as f64).sqrt() as f32,
    }
}

// ── the point, and the two references ────────────────────────────────────

/// Everything one window's fire needs on the device, uploaded once and
/// shared by the point and its reference.
struct Uploaded {
    mixed: Slab,
    f: Slab,
    b: Slab,
    indptr: Slab,
    a_log: Slab,
    dt_bias: Slab,
    planes: [Slab; 3],
}

impl Uploaded {
    fn of(draw: &Draw, decay: &Decay, case: Case) -> Uploaded {
        let [q, k, v] = draw.planes(case);
        Uploaded {
            mixed: Slab::of(&bytes_of_u16(&draw.mixed)),
            f: Slab::of(&bytes_of_u16(&draw.f)),
            b: Slab::of(&bytes_of_u16(&draw.b)),
            indptr: Slab::of(&bytes_of_i32(&draw.indptr)),
            a_log: Slab::of(&bytes_of_f32(&decay.a_log)),
            dt_bias: Slab::of(&bytes_of_f32(&decay.dt_bias)),
            planes: [
                Slab::of(&bytes_of_u16(&q)),
                Slab::of(&bytes_of_u16(&k)),
                Slab::of(&bytes_of_u16(&v)),
            ],
        }
    }

    fn a_log(&self) -> Const<kernels_cuda::jit::abi::Tensor<f32>> {
        Const::new(self.a_log.ptr.cast::<f32>().cast_const())
    }

    fn dt_bias(&self) -> Const<kernels_cuda::jit::abi::Tensor<f32>> {
        Const::new(self.dt_bias.ptr.cast::<f32>().cast_const())
    }
}

/// THE LEGACY SEQUENCE, launch for launch: two l2 norms, one widen, the
/// gate/beta cook and the step. Every one of the five is the routine the
/// legacy text named, at the arguments its generated builder passed.
fn legacy_step(ctx: &Ctx<'_>, case: Case, draw: &Draw, up: &Uploaded, view: &RecurrentView, out: *mut f32) {
    let (n, w, h, d) = (draw.n, case.w(), case.h, case.d);
    let staged: Vec<Slab> = (0..3).map(|_| Slab::zeroed((n * w) as usize * 4)).collect();
    let gate = Slab::zeroed((n * w) as usize * 4);
    let beta = Slab::zeroed((n * h) as usize * 4);

    for (src, dst) in up.planes.iter().zip(&staged).take(2) {
        kernels_cuda::ssm::l2norm_scale_bf16_to_fp32(
            ctx,
            In {
                ptr: src.ptr.cast_const(),
                rows: n,
                width: w,
            },
            Out {
                ptr: dst.ptr.cast::<f32>(),
                rows: n,
                width: w,
            },
            Const::new(case.eps),
        )
        .expect("the legacy q/k l2 norm");
    }
    kernels_cuda::ssm::bf16_to_fp32(
        ctx,
        In {
            ptr: up.planes[2].ptr.cast_const(),
            rows: n,
            width: w,
        },
        Out {
            ptr: staged[2].ptr.cast::<f32>(),
            rows: n,
            width: w,
        },
    )
    .expect("the legacy value widen");
    kernels_cuda::ssm::kda_gate_beta::<bf16>(
        ctx,
        In {
            ptr: up.f.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: w,
        },
        In {
            ptr: up.b.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: h,
        },
        up.a_log(),
        up.dt_bias(),
        Out {
            ptr: gate.ptr.cast::<f32>(),
            rows: n,
            width: w,
        },
        Out {
            ptr: beta.ptr.cast::<f32>(),
            rows: n,
            width: h,
        },
        Const::new(d),
    )
    .expect("the legacy gate/beta cook");
    let plane = |s: &Slab| In {
        ptr: s.ptr.cast::<f32>().cast_const(),
        rows: n,
        width: w,
    };
    kernels_cuda::ssm::kda_recurrent_step_batched(
        ctx,
        plane(&staged[0]),
        plane(&staged[1]),
        plane(&staged[2]),
        In {
            ptr: gate.ptr.cast::<f32>().cast_const(),
            rows: n,
            width: w,
        },
        In {
            ptr: beta.ptr.cast::<f32>().cast_const(),
            rows: n,
            width: h,
        },
        Out {
            ptr: out,
            rows: n,
            width: w,
        },
        Const::new(h),
        Const::new(d),
        raised(view),
    )
    .expect("the legacy recurrent step");
}

/// The point, at the whole fire: one token per request.
fn point_step(ctx: &Ctx<'_>, case: Case, draw: &Draw, up: &Uploaded, view: &RecurrentView, out: *mut f32) {
    let (n, w, h, d) = (draw.n, case.w(), case.h, case.d);
    ctx.kda_step::<bf16>(
        In {
            ptr: up.mixed.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: 3 * w,
        },
        In {
            ptr: up.f.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: w,
        },
        In {
            ptr: up.b.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: h,
        },
        up.dt_bias(),
        up.a_log(),
        cached(view),
        h.unsigned_abs(),
        d.unsigned_abs(),
        case.eps,
        Out {
            ptr: out,
            rows: n,
            width: w,
        },
    )
    .expect("ssm.kda_step");
}

/// The point, at a window: the CSR cuts the rows into requests.
fn point_chunked(ctx: &Ctx<'_>, case: Case, draw: &Draw, up: &Uploaded, view: &RecurrentView, out: *mut f32) {
    let (n, w, h, d) = (draw.n, case.w(), case.h, case.d);
    ctx.kda_chunked::<bf16>(
        In {
            ptr: up.mixed.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: 3 * w,
        },
        In {
            ptr: up.indptr.ptr.cast::<i32>().cast_const(),
            rows: draw.r,
            width: draw.r + 1,
        },
        In {
            ptr: up.f.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: w,
        },
        In {
            ptr: up.b.ptr.cast::<bf16>().cast_const(),
            rows: n,
            width: h,
        },
        up.dt_bias(),
        up.a_log(),
        cached(view),
        h.unsigned_abs(),
        d.unsigned_abs(),
        case.eps,
        Out {
            ptr: out,
            rows: n,
            width: w,
        },
    )
    .expect("ssm.kda_chunked");
}

/// THE STEP LOOP: `ssm.kda_step` once per token, per request, in the order
/// the recurrence serializes them.
///
/// A one-row fire addresses ONE slot, so the view is rebuilt per request
/// with `slot_ids` advanced — `kda_recurrent_step_batched` reads
/// `slot_ids[r]` and this loop's `r` is always zero. That is the same
/// arithmetic the batched fire does, spelled from the other side.
fn step_loop(ctx: &Ctx<'_>, case: Case, draw: &Draw, up: &Uploaded, pool: &Pool, out: *mut f32) {
    let (w, h, d) = (case.w(), case.h, case.d);
    for r in 0..draw.r as usize {
        let view = pool.view(r as i32);
        for t in draw.indptr[r]..draw.indptr[r + 1] {
            let t = t as usize;
            let row = |ptr: *mut c_void, width: i32| In {
                ptr: unsafe { ptr.cast::<bf16>().add(t * width as usize) }.cast_const(),
                rows: 1,
                width,
            };
            ctx.kda_step::<bf16>(
                row(up.mixed.ptr, 3 * w),
                row(up.f.ptr, w),
                row(up.b.ptr, h),
                up.dt_bias(),
                up.a_log(),
                cached(&view),
                h.unsigned_abs(),
                d.unsigned_abs(),
                case.eps,
                Out {
                    ptr: unsafe { out.add(t * w as usize) },
                    rows: 1,
                    width: w,
                },
            )
            .expect("ssm.kda_step, one token of the loop");
        }
    }
}

// ── the runs ─────────────────────────────────────────────────────────────

struct Answer {
    out: Vec<f32>,
    state: Vec<f32>,
}

/// One side of a comparison: everything a fire needs, and the slab it
/// writes its result into.
type Fires<'a> = &'a dyn Fn(&Ctx<'_>, Case, &Draw, &Uploaded, &Pool, *mut f32);

/// Fire `mine` and `theirs` over the same drawn windows, against two slabs
/// that started life identical, and answer both sides of every window.
fn windows(
    case: Case,
    lens: &[&[i32]],
    seed: u64,
    mine: Fires<'_>,
    theirs: Fires<'_>,
) -> Vec<[Answer; 2]> {
    let stream: *mut c_void = core::ptr::null_mut();
    let ctx = unsafe { Ctx::on(stream) };
    let mut rng = Rng(seed);

    let slots = lens[0].len() as i32;
    let state_elems = slots as usize * case.slot_stride();
    // A drawn state rather than a zeroed one: a zero slab makes the first
    // token's decay unobservable, and the handoff below is the whole point.
    let state0: Vec<f32> = (0..state_elems).map(|_| rng.next() * 0.5).collect();
    let decay = Decay::of(case, &mut rng);

    let a = Pool::new(case, slots, &state0);
    let b = Pool::new(case, slots, &state0);

    let mut answers = Vec::new();
    for lens in lens {
        assert_eq!(lens.len(), slots as usize, "one slot per request, per window");
        let draw = Draw::of(case, lens, &mut rng);
        let up = Uploaded::of(&draw, &decay, case);
        let out_elems = (draw.n * case.w()) as usize;
        let out_a = Slab::zeroed(out_elems * 4);
        let out_b = Slab::zeroed(out_elems * 4);

        mine(&ctx, case, &draw, &up, &a, out_a.ptr.cast::<f32>());
        theirs(&ctx, case, &draw, &up, &b, out_b.ptr.cast::<f32>());

        answers.push([
            Answer {
                out: out_a.f32s(out_elems),
                state: a.read(state_elems),
            },
            Answer {
                out: out_b.f32s(out_elems),
                state: b.read(state_elems),
            },
        ]);
    }
    answers
}

/// Both sides bit-identical, on the result AND on the state the next window
/// continues from.
fn check(
    name: &str,
    case: Case,
    lens: &[&[i32]],
    seed: u64,
    against: &str,
    mine: Fires<'_>,
    theirs: Fires<'_>,
) {
    let runs = windows(case, lens, seed, mine, theirs);
    let mut failures: Vec<String> = Vec::new();
    for (i, [a, b]) in runs.iter().enumerate() {
        let out = deviation(&a.out, &b.out);
        let state = deviation(&a.state, &b.state);
        eprintln!("{name} window {i}, against the {against}:");
        eprintln!("    out    {out}");
        eprintln!("    state  {state}");
        // See [`Gap`]: an all-zero comparison is bit-identical and proves
        // nothing, and every accidental way to get one lands here.
        assert!(
            out.scale > 1e-3 && state.scale > 1e-3,
            "{name} window {i}: the reference is too close to zero to compare against \
             (out {out}, state {state})"
        );
        if out.same != out.of {
            failures.push(format!("{name} window {i}: out {out}"));
        }
        if state.same != state.of {
            failures.push(format!("{name} window {i}: state {state}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ── the cases ────────────────────────────────────────────────────────────

/// K3's own mixer: 16 heads of 128, so the packed row is 6144 wide and the
/// l2 norm the body fires walks 2048 columns per block, which is the shape
/// `l2norm_scale<bf16, 128>` was fired at.
const K3: Case = Case {
    h: 16,
    d: 128,
    eps: 1e-5,
};

/// A narrow twin, so a failure has a small enough state to read and so the
/// prefill kernel's `min(32, D)` warp clamp is exercised on the other side
/// of its bound.
const NARROW: Case = Case {
    h: 4,
    d: 24,
    eps: 1e-5,
};

/// The step body IS the five launches the legacy fired, and the bar is the
/// bits.
#[test]
fn the_step_is_the_legacy_sequence() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.kda_step against the legacy sequence") {
        return;
    }
    for (name, case, seed) in [
        ("k3 16/128", K3, 0x9E37_79B9_7F4A_7C15),
        ("narrow 4/24", NARROW, 0xD1B5_4A32_D192_ED03),
    ] {
        check(
            name,
            case,
            // One token per request, twice, so the second fire continues
            // from the state the first left.
            &[&[1, 1, 1], &[1, 1, 1]],
            seed,
            "legacy five",
            &|ctx, case, draw, up, pool, out| point_step(ctx, case, draw, up, &pool.view(0), out),
            &|ctx, case, draw, up, pool, out| legacy_step(ctx, case, draw, up, &pool.view(0), out),
        );
    }
}

/// The window leaves the state the step loop leaves — W2's
/// rounding-trajectory law, measured on the kernel nothing had ever run.
#[test]
fn the_window_is_the_step_loop() {
    let _fire = FIRE.lock().unwrap_or_else(|e| e.into_inner());
    if !device_or_skip("ssm.kda_chunked against a step loop") {
        return;
    }
    for (name, case, lens, seed) in [
        (
            "k3 16/128",
            K3,
            [[9i32, 5].as_slice(), [4, 7].as_slice()],
            0x2545_F491_4F6C_DD1D,
        ),
        (
            "narrow 4/24",
            NARROW,
            [[11i32, 3].as_slice(), [6, 8].as_slice()],
            0xB5AD_4ECE_DA10_1373,
        ),
    ] {
        check(
            name,
            case,
            &lens,
            seed,
            "step loop",
            &|ctx, case, draw, up, pool, out| {
                point_chunked(ctx, case, draw, up, &pool.view(0), out);
            },
            &step_loop,
        );
    }
}
