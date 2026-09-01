//! **THE TILED AFFINE POINT, ON A REAL DEVICE** (§J4 hybrid, phases A and
//! B):
//! `linear::tiled::matmul` over the repacked post-affine W4A16 planes, held
//! against the same host fold the interim decoded arm's golden already uses.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --release --test tiled_matmul -- --nocapture
//! ```
//!
//! The gates:
//!
//! ```text
//! (a) the repack is a RELABELLING: a host un-repack of the device's own
//!     output recovers the dense plane and both factor planes bit for bit,
//!     and the band padding is zeros
//! (b) the tiled projection matches the host fold at 32 and 512 tokens, over
//!     a k with several groups to a contraction step and an n whose last
//!     tile is not whole
//! (c) the tiled point and `quant::matmul`'s fused GEMV answer the same
//!     numbers on identical planes
//! (d) the refusal ladder: a streamed seat, a k that is not a whole step, an
//!     un-padded plane, a group finer than an mma k tile
//! (e) a REPORT-ONLY timing line at 512x2048x10240 — tiled vs the interim
//!     decoded arm vs the fused GEMV. Phase A's bar was tiled <= decoded;
//!     phase B's is cuBLAS on an already-decoded weight, and it reaches it
//! (f) both config tuples against the fold on every aspect, plus the row
//!     count that picks between them
//! (g) a REPORT-ONLY sweep: both projection directions by three prefill row
//!     counts by four tuples, which is the table phase B's pick stands on
//! ```
//!
//! **THE ORACLE IS THE DECODED FOLD, AND THAT IS EXACT HERE.** The fused
//! GEMV never materialises a weight element; the tiled point does, in a
//! register, as `__hfma2(code, s, b)` — one rounding, on operands that are
//! all exactly bf16 (the lop3 lands `128 + code` and 128..143 fit bf16's
//! seven mantissa bits, so the subtraction back to `code` loses nothing).
//! `to_bf16(code * s + b)` computed in f32 is the SAME NUMBER, so
//! [`fold_decoded`] below is not a tolerance-matched approximation of what
//! the kernel does — it is what the kernel does, with only the mma's f32
//! accumulation order left over.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, TOLERANCE, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::cudarc::cublas::sys as blas;
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::gemm;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::{self, OffsetKind};
use kernels_cuda::linear::tiled;
use kernels_cuda::tensor::Tensor;

/// The mma band the repack pads a weight's rows up to.
const BAND: u32 = 16;

/// The k tiles the repacked plane groups into one lane's superword — the
/// four 16-wide mma k tiles of one 64-wide contraction step.
const QUAD: usize = 4;

/// One stored form this point serves. Four bits, a bf16 factor pair, the
/// post-offset fold — the row of `linear::quant`'s matrix the tiled point is
/// stamped for, and the only one.
#[derive(Clone, Copy)]
struct Spec {
    group: usize,
    rows: u32,
    n: u32,
    k: u32,
}

impl Spec {
    const fn new(rows: u32, n: u32, k: u32, group: usize) -> Self {
        Self {
            group,
            rows,
            n,
            k,
        }
    }

    fn groups(self) -> usize {
        self.k as usize / self.group
    }

    fn factor_row(self) -> u32 {
        (self.k / self.group as u32) * 2
    }

    fn codes_row(self) -> u32 {
        self.k / 2
    }

    fn n_pad(self) -> u32 {
        self.n.div_ceil(BAND) * BAND
    }
}

/// A cuBLAS handle, which `common::Gpu` does not carry — the timing line
/// reaches `quant::matmul_via_dense`, and that reaches `linear::gemm`.
struct Blas(blas::cublasHandle_t);

impl Blas {
    fn open() -> Self {
        let mut handle: blas::cublasHandle_t = core::ptr::null_mut();
        let status = unsafe { blas::cublasCreate_v2(&raw mut handle) };
        assert_eq!(
            status,
            blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "`cublasCreate_v2` refused"
        );
        Self(handle)
    }

    fn ctx(&self, gpu: &Gpu) -> Ctx {
        // SAFETY: the handle outlives every fire of the test that holds it.
        unsafe { gpu.ctx().with_cublas(self.0.cast()) }
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            blas::cublasDestroy_v2(self.0);
        }
    }
}

/// A bf16 factor and the number it decodes to, drawn bits-first so nothing
/// in this file rounds — `dense_affine_matmul.rs`'s generator, narrowed to
/// the one dtype this point reads.
fn factor(rng: &mut Lcg, exp2: i32) -> (u16, f32) {
    let draw = rng.unit();
    let negative = draw < 0.0;
    let step = 128.0f32;
    let mantissa = (draw.abs() * step) as u32 & 0x7f;
    let bits = (u16::from(negative) << 15) | (((exp2 + 127) as u16) << 7) | mantissa as u16;
    let magnitude = (1.0 + mantissa as f32 / step) * 2.0f32.powi(exp2);
    (bits, if negative { -magnitude } else { magnitude })
}

/// The dense planes a checkpoint seats, and the numbers a host fold reads
/// back out of them.
struct Planes {
    codes: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
    scale_of: Vec<f32>,
    bias_of: Vec<f32>,
}

fn planes(spec: Spec, rng: &mut Lcg) -> Planes {
    let groups = spec.groups();
    let factors = spec.n as usize * groups;
    let mut codes = Vec::with_capacity(spec.n as usize * spec.k as usize / 2);
    let mut byte = 0u8;
    for at in 0..(spec.n as usize * spec.k as usize) {
        let nibble = ((rng.unit().abs() * 255.0) as u32 & 0xF) as u8;
        if at % 2 == 0 {
            byte = nibble;
        } else {
            codes.push(byte | (nibble << 4));
        }
    }

    let mut scales = Vec::with_capacity(factors * 2);
    let mut biases = Vec::with_capacity(factors * 2);
    let mut scale_of = Vec::with_capacity(factors);
    let mut bias_of = Vec::with_capacity(factors);
    for _ in 0..factors {
        // A scale near 2^-5 and a bias near 1: the orders an affine weight's
        // factors sit at, and the pair `Spec::mlx` draws next door.
        let (bits, value) = factor(rng, -5);
        scales.extend_from_slice(&bits.to_le_bytes());
        scale_of.push(value);
        let (bits, value) = factor(rng, 0);
        biases.extend_from_slice(&bits.to_le_bytes());
        bias_of.push(value);
    }
    Planes {
        codes,
        scales,
        biases,
        scale_of,
        bias_of,
    }
}

/// The code stored at flat position `at` of the dense plane.
fn code_at(codes: &[u8], at: usize) -> u8 {
    let byte = codes[at / 2];
    if at % 2 == 0 { byte & 0xF } else { byte >> 4 }
}

/// **THE ORACLE.** Every weight element materialised and rounded to bf16
/// exactly where the kernel's `__hfma2` rounds it, then folded in f32 —
/// `dense_affine_matmul.rs`'s `fold_decoded`, at the post-offset arm.
fn fold_decoded(spec: Spec, seeded: &Planes, x: &[f32]) -> Vec<f32> {
    let groups = spec.groups();
    let mut y = vec![0.0f32; spec.rows as usize * spec.n as usize];
    for t in 0..spec.rows as usize {
        let xt = &x[t * spec.k as usize..][..spec.k as usize];
        for r in 0..spec.n as usize {
            let mut acc = 0.0f32;
            for g in 0..groups {
                let fx = r * groups + g;
                let s = seeded.scale_of[fx];
                let b = seeded.bias_of[fx];
                for j in 0..spec.group {
                    let at = r * spec.k as usize + g * spec.group + j;
                    let w = f32::from(code_at(&seeded.codes, at)) * s + b;
                    acc += from_bf16(to_bf16(w)) * xt[g * spec.group + j];
                }
            }
            y[t * spec.n as usize + r] = acc;
        }
    }
    y
}

/// **THE HOST UN-REPACK** — the inverse of `repack_affine_tiled`, written
/// from the layout the kernel's banner states and not from its code, so that
/// the two have to agree about what the layout IS.
///
/// Word `lane` of tile `(band, k tile)` holds, at nibble `s + 4h`, the code
/// at `k = 16*kt + 2*(lane%4) + 8*(s&1) + h` and
/// `n = 16*band + lane/4 + 8*(s>=2)`.
///
/// **AND PHASE B MOVED WHERE THAT WORD SITS.** Four k tiles are grouped as
/// one lane's `uint4`, so the word order is `[band][k quad][lane][4]` and
/// not phase A's `[band][k tile][lane]`. That is the whole layout change the
/// superword bought, and [`QUAD`] below is the only place this file knows
/// about it — a reader written against the old order recovers the right
/// bytes in the wrong places, which is exactly what this gate catches.
fn unrepack(repacked: &[u32], n: u32, k: u32) -> Vec<u8> {
    let bands = n.div_ceil(BAND) as usize;
    let k_tiles = (k / BAND) as usize;
    let quads = k_tiles / QUAD;
    let row_bytes = k as usize / 2;
    let mut out = vec![0u8; n as usize * row_bytes];
    for band in 0..bands {
        for kt in 0..k_tiles {
            for lane in 0..32usize {
                let word =
                    repacked[((band * quads + kt / QUAD) * 32 + lane) * QUAD + kt % QUAD];
                let col_of = lane / 4;
                let k_base = kt * 16 + 2 * (lane % 4);
                for s in 0..4usize {
                    let col = band * 16 + col_of + if s >= 2 { 8 } else { 0 };
                    for h in 0..2usize {
                        let kk = k_base + if s % 2 == 1 { 8 } else { 0 } + h;
                        let code = ((word >> (4 * (s + 4 * h))) & 0xF) as u8;
                        if col < n as usize {
                            let at = col * k as usize + kk;
                            out[at / 2] |= code << (4 * (at % 2));
                        } else {
                            assert_eq!(code, 0, "the band padding is not a zero code");
                        }
                    }
                }
            }
        }
    }
    out
}

/// The device planes for one spec, repacked, plus the activations.
struct Seated {
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    dense_codes: Tensor,
    dense_scales: Tensor,
    dense_biases: Tensor,
    act: Tensor,
    x: Vec<f32>,
}

fn seat(gpu: &mut Gpu, ctx: &Ctx, spec: Spec, seeded: &Planes, rng: &mut Lcg) -> Seated {
    let (x_raw, x) = rng.row(spec.rows as usize * spec.k as usize);
    let n_pad = spec.n_pad();

    let dense_codes = Tensor::new(gpu.up(&seeded.codes), spec.n, spec.codes_row(), Dtype::U8);
    let dense_scales = Tensor::new(gpu.up(&seeded.scales), spec.n, spec.factor_row(), Dtype::U8);
    let dense_biases = Tensor::new(gpu.up(&seeded.biases), spec.n, spec.factor_row(), Dtype::U8);

    let mut codes = Tensor::new(
        gpu.zeros(n_pad as usize * spec.codes_row() as usize),
        n_pad,
        spec.codes_row(),
        Dtype::U8,
    );
    let mut scales = Tensor::new(
        gpu.zeros(n_pad as usize * spec.factor_row() as usize),
        n_pad,
        spec.factor_row(),
        Dtype::U8,
    );
    let mut biases = Tensor::new(
        gpu.zeros(n_pad as usize * spec.factor_row() as usize),
        n_pad,
        spec.factor_row(),
        Dtype::U8,
    );
    tiled::repack(
        ctx,
        dense_codes,
        dense_scales,
        dense_biases,
        &mut codes,
        &mut scales,
        &mut biases,
    )
    .expect("the relabelling fires");

    Seated {
        codes,
        scales,
        biases,
        dense_codes,
        dense_scales,
        dense_biases,
        act: Tensor::new(gpu.up(&x_raw), spec.rows, spec.k, Dtype::Bf16),
        x,
    }
}

/// One tiled fire, and the fold it is held against.
fn fired(gpu: &mut Gpu, ctx: &Ctx, spec: Spec, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::seeded(seed);
    let seeded = planes(spec, &mut rng);
    let seated = seat(gpu, ctx, spec, &seeded, &mut rng);
    let y_at = gpu.zeros(spec.rows as usize * spec.n as usize * 2);
    let mut y = Tensor::new(y_at, spec.rows, spec.n, Dtype::Bf16);
    tiled::matmul(
        ctx,
        seated.act,
        seated.codes,
        seated.scales,
        seated.biases,
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the tiled projection fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, spec.rows as usize * spec.n as usize);
    (
        got.into_iter().map(from_bf16).collect(),
        fold_decoded(spec, &seeded, &seated.x),
    )
}

fn held(spec: Spec, seed: u64) {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let (got, want) = fired(&mut gpu, &ctx, spec, seed);
    for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "y[{at}] answered {g} where the decoded fold says {w} \
             (rows {}, n {}, k {}, group {})",
            spec.rows,
            spec.n,
            spec.k,
            spec.group
        );
    }
}

// ─── (a) the repack is a relabelling ───────────────────────────────────────

#[test]
fn the_repack_is_a_relabelling_and_nothing_else() {
    let spec = Spec::new(1, 100, 512, 64);
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut rng = Lcg::seeded(0x7117);
    let seeded = planes(spec, &mut rng);
    let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);
    gpu.sync();

    let n_pad = spec.n_pad();
    let words = n_pad as usize * spec.codes_row() as usize / 4;
    let repacked: Vec<u32> = gpu.down(seated.codes.ptr, words);
    assert_eq!(
        unrepack(&repacked, spec.n, spec.k),
        seeded.codes,
        "the un-repacked code plane is not the plane that went in"
    );

    // The factor planes: `[n][group]` became `[band][group][16]`, and the
    // tail of a band is a zero factor.
    let groups = spec.groups();
    let got_s: Vec<u16> = gpu.down(seated.scales.ptr, n_pad as usize * groups);
    let got_b: Vec<u16> = gpu.down(seated.biases.ptr, n_pad as usize * groups);
    let want_s: Vec<u16> = seeded
        .scales
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect();
    let want_b: Vec<u16> = seeded
        .biases
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect();
    for band in 0..(n_pad / BAND) as usize {
        for g in 0..groups {
            for j in 0..BAND as usize {
                let at = (band * groups + g) * BAND as usize + j;
                let row = band * BAND as usize + j;
                let (s, b) = if row < spec.n as usize {
                    (want_s[row * groups + g], want_b[row * groups + g])
                } else {
                    (0, 0)
                };
                assert_eq!(got_s[at], s, "the scale at band {band} group {g} row {j} moved");
                assert_eq!(got_b[at], b, "the bias at band {band} group {g} row {j} moved");
            }
        }
    }
}

// ─── (b) the projection matches the host fold ──────────────────────────────

/// A decode-shaped fire: one whole tile of tokens is more than a decode
/// brings, so this is the shape where the M edge and the N edge both run.
#[test]
fn the_tiled_projection_matches_the_host_fold() {
    // n = 100 is six whole bands and a seventh that is four columns of
    // weight and twelve of padding; it is also not a whole 64-wide block.
    held(Spec::new(32, 100, 512, 64), 0x5eed);
}

/// Several groups to a contraction step, and a k long enough that the group
/// turns over inside the mainloop rather than only between steps.
#[test]
fn a_group_finer_than_the_contraction_step_folds() {
    held(Spec::new(32, 128, 512, 32), 0x5eed + 1);
    held(Spec::new(32, 128, 512, 128), 0x5eed + 2);
}

/// A prefill's rows, at the projection width §J4a measured: the shape the
/// whole hybrid exists for.
#[test]
fn a_prefill_of_five_hundred_and_twelve_rows_matches_the_host_fold() {
    held(Spec::new(512, 100, 2048, 64), 0x5eed + 3);
}

/// A row count that is not a whole tile, from both sides of one.
#[test]
fn a_ragged_row_count_matches_the_host_fold() {
    held(Spec::new(1, 64, 512, 64), 0x5eed + 4);
    held(Spec::new(65, 64, 512, 64), 0x5eed + 5);
}

// ─── (c) the tiled point and the fused GEMV agree ──────────────────────────

/// **TWO READINGS OF ONE STORED ROW.** The fused GEMV folds `s·c + b` in f32
/// inside the dot and materialises no weight; the tiled point materialises
/// every element as a bf16 register on the way into the B fragment. So they
/// answer the same numbers and not the same bits, and the ruler is the ROW's
/// scale for the reason `dense_affine_matmul.rs` states at length: a dot
/// product's cancellation is not bounded by its own result.
#[test]
fn the_tiled_point_answers_what_the_fused_gemv_answers() {
    let spec = Spec::new(32, 100, 512, 64);
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut rng = Lcg::seeded(0xf00d);
    let seeded = planes(spec, &mut rng);
    let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);

    let count = spec.rows as usize * spec.n as usize;
    let tiled_at = gpu.zeros(count * 2);
    let mut y = Tensor::new(tiled_at, spec.rows, spec.n, Dtype::Bf16);
    tiled::matmul(
        &ctx,
        seated.act,
        seated.codes,
        seated.scales,
        seated.biases,
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the tiled projection fires");

    let fused_at = gpu.zeros(count * 2);
    let mut y = Tensor::new(fused_at, spec.rows, spec.n, Dtype::Bf16);
    quant::matmul(
        &ctx,
        seated.act,
        seated.dense_codes,
        seated.dense_scales,
        OffsetKind::Post,
        Some(seated.dense_biases),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the fused projection fires");
    gpu.sync();

    let got: Vec<f32> = gpu
        .down::<u16>(tiled_at, count)
        .into_iter()
        .map(from_bf16)
        .collect();
    let fused: Vec<f32> = gpu
        .down::<u16>(fused_at, count)
        .into_iter()
        .map(from_bf16)
        .collect();
    for t in 0..spec.rows as usize {
        let at0 = t * spec.n as usize;
        let row = &fused[at0..at0 + spec.n as usize];
        let scale = row.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (r, f) in row.iter().enumerate() {
            let d = got[at0 + r];
            assert!(
                (d - f).abs() <= TOLERANCE * scale,
                "the two readings parted at row {t} column {r}: tiled {d}, fused {f}, \
                 over a row whose scale is {scale}"
            );
        }
    }
}

// ─── (d) the refusal ladder ────────────────────────────────────────────────

#[test]
fn the_shapes_this_point_is_not_stamped_for_are_refused() {
    let spec = Spec::new(4, 64, 512, 64);
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut rng = Lcg::seeded(0xbad);
    let seeded = planes(spec, &mut rng);
    let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);
    let mut y = Tensor::new(
        gpu.zeros(spec.rows as usize * spec.n as usize * 2),
        spec.rows,
        spec.n,
        Dtype::Bf16,
    );

    // A moving plane has not been repacked.
    let streamed = GroupSeat {
        cell: gpu.zeros(16),
        hits: 0,
    };
    assert!(
        tiled::matmul(
            &ctx,
            seated.act,
            seated.codes,
            seated.scales,
            seated.biases,
            &mut y,
            streamed,
        )
        .is_err(),
        "a streamed seat took the tiled arm"
    );

    // A contraction that is not a whole 64-wide step.
    let short = Tensor::new(seated.act.ptr, spec.rows, 32, Dtype::Bf16);
    let mut narrow = Tensor::new(y.ptr, spec.rows, spec.n, Dtype::Bf16);
    assert!(
        tiled::matmul(
            &ctx,
            short,
            Tensor::new(seated.codes.ptr, spec.n_pad(), 16, Dtype::U8),
            seated.scales,
            seated.biases,
            &mut narrow,
            GroupSeat::RESIDENT,
        )
        .is_err(),
        "a 32-wide contraction fired"
    );

    // An unpadded code plane: the last band would be read past.
    assert!(
        tiled::matmul(
            &ctx,
            seated.act,
            Tensor::new(seated.codes.ptr, spec.n_pad() - BAND, spec.codes_row(), Dtype::U8),
            seated.scales,
            seated.biases,
            &mut y,
            GroupSeat::RESIDENT,
        )
        .is_err(),
        "a short code plane fired"
    );

    // A group of eight codes: finer than the 16-wide mma k tile a lane's
    // whole B fragment covers.
    let fine = Tensor::new(seated.scales.ptr, spec.n_pad(), spec.k / 8 * 2, Dtype::U8);
    assert!(
        tiled::matmul(
            &ctx,
            seated.act,
            seated.codes,
            fine,
            fine,
            &mut y,
            GroupSeat::RESIDENT,
        )
        .is_err(),
        "an eight-code group fired"
    );
}

// ─── (e) the timing line, report-only ──────────────────────────────────────

/// **WHERE THE POINT LANDED ON §J4a's OWN SHAPE**, printed and not asserted.
/// §J4a measured the fused GEMV at 11.6 ms here and §J4a-1 the decoded arm
/// at 0.209 ms (1.8x pure cuBLAS at 0.118); phase A's bar was
/// tiled <= decoded and it answered 0.161. Phase B's bar was the cuBLAS
/// floor itself, and this line is where it is held. The numbers move with
/// the box, so this gate reports and the wiki records — an assertion here
/// would be a benchmark pinned to one machine, which is not what a golden
/// is for.
///
/// The four arms share ONE seeded weight, so what is timed is four readings
/// of a single stored row and not four different problems. `cublas (bf16
/// dense)` is the floor: the decoded weight already in the slab, projected
/// with nothing to unpack.
#[test]
fn the_prefill_shape_is_timed_against_the_standing_arms() {
    const ITERS: u32 = 20;
    let spec = Spec::new(512, 10240, 2048, 64);
    let mut gpu = Gpu::open();
    let blas = Blas::open();
    let ctx = blas.ctx(&gpu);
    let mut rng = Lcg::seeded(0x71_3d);
    let seeded = planes(spec, &mut rng);
    let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);
    let count = spec.rows as usize * spec.n as usize;
    let y_at = gpu.zeros(count * 2);
    // The bf16 rectangle the cuBLAS floor projects: the same weight, already
    // decoded, so that arm times the multiply alone.
    let dense_at = gpu.zeros(spec.n as usize * spec.k as usize * 2);
    let dense = Tensor::new(dense_at, spec.n, spec.k, Dtype::Bf16);

    println!(
        "\ntiled affine, tokens {} k {} n {} group {}",
        spec.rows, spec.k, spec.n, spec.group
    );
    let timed = |name: &str, fire: &mut dyn FnMut(&Ctx, &mut Tensor)| {
        let mut y = Tensor::new(y_at, spec.rows, spec.n, Dtype::Bf16);
        for _ in 0..3 {
            fire(&ctx, &mut y);
        }
        gpu.sync();
        let at = std::time::Instant::now();
        for _ in 0..ITERS {
            fire(&ctx, &mut y);
        }
        gpu.sync();
        let each = at.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);
        println!("  {name:<26} {each:8.3} ms");
    };

    timed(
        "tiled (this point)",
        &mut |ctx, y| {
            tiled::matmul(
                ctx,
                seated.act,
                seated.codes,
                seated.scales,
                seated.biases,
                y,
                GroupSeat::RESIDENT,
            )
            .expect("the tiled projection fires");
        },
    );
    timed(
        "via_dense (interim arm)",
        &mut |ctx, y| {
            quant::matmul_via_dense(
                ctx,
                seated.act,
                seated.dense_codes,
                seated.dense_scales,
                OffsetKind::Post,
                Some(seated.dense_biases),
                Dtype::Bf16,
                y,
                GroupSeat::RESIDENT,
            )
            .expect("the decoded projection fires");
        },
    );
    timed("cublas (bf16 dense)", &mut |ctx, y| {
        gemm::act_x_wt(ctx, "linear.matmul", seated.act, dense, y)
            .expect("the dense projection fires");
    });
    timed("fused GEMV (decode arm)", &mut |ctx, y| {
        quant::matmul(
            ctx,
            seated.act,
            seated.dense_codes,
            seated.dense_scales,
            OffsetKind::Post,
            Some(seated.dense_biases),
            Dtype::Bf16,
            y,
            GroupSeat::RESIDENT,
        )
        .expect("the fused projection fires");
    });
}

// ─── (f) both tuples, on every aspect ──────────────────────────────────────

/// **THE SECOND TUPLE ANSWERS WHAT THE FIRST ONE DOES.** `tuple_for` sends a
/// projection to [`tiled::LONG`] or [`tiled::SHORT`] by its row count, and
/// the two walk different tiles with different stage depths; a config axis
/// that is not held against the fold is a silently wrong answer at one
/// shape. So every tuple is fired on every shape here, not only the ones its
/// own rule would send it.
#[test]
fn every_tuple_matches_the_host_fold_on_every_aspect() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    // n < k, n > k, and an n whose last tile is neither whole nor a whole
    // band — with row counts on both sides of both tuples' row tiles.
    for (at, spec) in [
        Spec::new(100, 320, 128, 64),
        Spec::new(100, 128, 320, 64),
        Spec::new(65, 100, 512, 64),
        Spec::new(1, 100, 512, 64),
    ]
    .into_iter()
    .enumerate()
    {
        let mut rng = Lcg::seeded(0xc0_11 + at as u64);
        let seeded = planes(spec, &mut rng);
        let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);
        let want = fold_decoded(spec, &seeded, &seated.x);
        for tuple in [tiled::LONG, tiled::SHORT] {
            let y_at = gpu.zeros(spec.rows as usize * spec.n as usize * 2);
            let mut y = Tensor::new(y_at, spec.rows, spec.n, Dtype::Bf16);
            tiled::matmul_with(
                &ctx,
                seated.act,
                seated.codes,
                seated.scales,
                seated.biases,
                &mut y,
                GroupSeat::RESIDENT,
                tuple,
            )
            .expect("the tiled projection fires");
            gpu.sync();
            let got: Vec<f32> = gpu
                .down::<u16>(y_at, spec.rows as usize * spec.n as usize)
                .into_iter()
                .map(from_bf16)
                .collect();
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    close(*g, *w),
                    "y[{i}] answered {g} where the decoded fold says {w} \
                     (rows {}, n {}, k {}, tuple {tuple:?})",
                    spec.rows,
                    spec.n,
                    spec.k
                );
            }
        }
    }
}

/// The pick is a pure function of the row count and it is the whole
/// the whole selection, so it is checked here rather than read off a timing
/// line.
/// **AND IT IS NOT AN ASPECT TEST**: both projection directions cross at the
/// same row count, which is the finding that made it one number.
#[test]
fn the_row_count_and_not_the_aspect_picks_the_tuple() {
    assert_eq!(tiled::tuple_for(tiled::LONG_PREFILL_ROWS), tiled::LONG);
    assert_eq!(tiled::tuple_for(tiled::LONG_PREFILL_ROWS - 1), tiled::SHORT);
    assert_eq!(tiled::tuple_for(2048), tiled::LONG);
    assert_eq!(tiled::tuple_for(1), tiled::SHORT);
}

// ─── (g) phase B's sweep, report-only ──────────────────────────────────────

/// **THE TABLE PHASE B'S PICK STANDS ON**, printed and not asserted.
///
/// Two projection directions — `n >= k` (up/gate, 2048×10240) and `k > n`
/// (down, 10240×2048, which §J4a measured at 20.5 ms on the fused GEMV) —
/// by three prefill row counts, by four config tuples, against the interim
/// decoded arm and the cuBLAS floor. BOTH shipping tuples are fired on BOTH
/// directions at every row count, because a pick nobody measured the other
/// side of is not a pick; phase A's own tuple and its four-stage twin stay
/// in the table because the four-stage twin is the item on phase B's list
/// that lost, and a loss that is not printed is a loss that gets retried.
///
/// The numbers move with the box, so this gate reports and the wiki records:
/// an assertion here would be a benchmark pinned to one machine, which is
/// not what a golden is for.
#[test]
fn both_projection_directions_are_swept_over_both_tuples() {
    const ITERS: u32 = 50;
    const ROWS: u32 = 2048;
    let arms: [(&str, tiled::Tuple); 4] = [
        ("A: 64x64 s2", tiled::Tuple { m: 64, n: 64, threads: 128, stages: 2 }),
        ("64x64 s4", tiled::Tuple { m: 64, n: 64, threads: 128, stages: 4 }),
        ("LONG", tiled::LONG),
        ("SHORT", tiled::SHORT),
    ];

    let mut gpu = Gpu::open();
    let blas = Blas::open();
    let ctx = blas.ctx(&gpu);
    println!("\ntiled affine sweep, group 64, {ITERS} iterations a point");
    for (label, n, k) in [("n >= k  (up/gate)", 10240u32, 2048u32), ("k >  n  (down)", 2048, 10240)]
    {
        let widest = Spec::new(ROWS, n, k, 64);
        let mut rng = Lcg::seeded(0x71_3d);
        let seeded = planes(widest, &mut rng);
        let seated = seat(&mut gpu, &ctx, widest, &seeded, &mut rng);
        let y_at = gpu.zeros(ROWS as usize * n as usize * 2);
        let dense = Tensor::new(
            gpu.zeros(n as usize * k as usize * 2),
            n,
            k,
            Dtype::Bf16,
        );
        println!("\n  {label}: n {n} k {k}");
        let mut head = format!("    {:>6}", "tokens");
        for (name, _) in arms {
            head.push_str(&format!("  {name:>10}"));
        }
        head.push_str(&format!("  {:>10}  {:>10}", "via_dense", "cublas"));
        println!("{head}");
        for tokens in [128u32, 512, 2048] {
            let act = Tensor::new(seated.act.ptr, tokens, k, Dtype::Bf16);
            let timed = |fire: &mut dyn FnMut(&Ctx, &mut Tensor)| {
                let mut y = Tensor::new(y_at, tokens, n, Dtype::Bf16);
                for _ in 0..3 {
                    fire(&ctx, &mut y);
                }
                gpu.sync();
                let at = std::time::Instant::now();
                for _ in 0..ITERS {
                    fire(&ctx, &mut y);
                }
                gpu.sync();
                at.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS)
            };
            let mut line = format!("    {tokens:>6}");
            for (_, tuple) in arms {
                let each = timed(&mut |ctx, y| {
                    tiled::matmul_with(
                        ctx,
                        act,
                        seated.codes,
                        seated.scales,
                        seated.biases,
                        y,
                        GroupSeat::RESIDENT,
                        tuple,
                    )
                    .expect("the tiled projection fires");
                });
                line.push_str(&format!("  {each:10.3}"));
            }
            let each = timed(&mut |ctx, y| {
                quant::matmul_via_dense(
                    ctx,
                    act,
                    seated.dense_codes,
                    seated.dense_scales,
                    OffsetKind::Post,
                    Some(seated.dense_biases),
                    Dtype::Bf16,
                    y,
                    GroupSeat::RESIDENT,
                )
                .expect("the decoded projection fires");
            });
            line.push_str(&format!("  {each:10.3}"));
            let each = timed(&mut |ctx, y| {
                gemm::act_x_wt(ctx, "linear.matmul", act, dense, y)
                    .expect("the dense projection fires");
            });
            line.push_str(&format!("  {each:10.3}"));
            println!("{line}");
        }
    }
}

// ─── (h) the decode point reads the same layout ────────────────────────────

/// **THE OTHER READING OF THE REPACKED PLANE.** `tiled::matmul_gemv` gathers
/// through the same fragment map the mma mainloop does and closes the dot
/// across four lanes instead of a tensor core, so it is held against the
/// same oracle: every weight element materialised and rounded to bf16 once,
/// then folded in f32.
///
/// Both carves on every shape, for the reason the tuple sweep above gives:
/// `carve_for` sends a projection to one of them by column count, and a
/// carve nobody held against the fold at the other shape is a silently wrong
/// answer at one aspect.
#[test]
fn the_decode_point_matches_the_host_fold() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    for (at, spec) in [
        // One token, which is the shape this point exists for.
        Spec::new(1, 128, 512, 64),
        // A band that is part padding, and a k with several groups a step.
        Spec::new(1, 100, 512, 32),
        Spec::new(3, 100, 512, 128),
        // Every row bucket boundary, and one that is not a bucket.
        Spec::new(2, 128, 320, 64),
        Spec::new(5, 128, 320, 64),
        Spec::new(16, 100, 512, 64),
        // k > n: the direction the deepest split is the carve for.
        Spec::new(4, 128, 1024, 64),
    ]
    .into_iter()
    .enumerate()
    {
        let mut rng = Lcg::seeded(0xdec0_de00 + at as u64);
        let seeded = planes(spec, &mut rng);
        let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);
        let want = fold_decoded(spec, &seeded, &seated.x);
        for carve in [
            tiled::Carve { bands: 1, split: 8 },
            tiled::Carve { bands: 1, split: 32 },
            tiled::Carve { bands: 4, split: 2 },
        ] {
            let count = spec.rows as usize * spec.n as usize;
            let y_at = gpu.zeros(count * 2);
            let mut y = Tensor::new(y_at, spec.rows, spec.n, Dtype::Bf16);
            tiled::gemv_with(
                &ctx,
                "linear.matmul",
                seated.act,
                seated.codes,
                seated.scales,
                seated.biases,
                &mut y,
                GroupSeat::RESIDENT,
                carve,
            )
            .expect("the tiled decode point fires");
            gpu.sync();
            let got: Vec<f32> = gpu
                .down::<u16>(y_at, count)
                .into_iter()
                .map(from_bf16)
                .collect();
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    close(*g, *w),
                    "y[{i}] answered {g} where the decoded fold says {w} \
                     (rows {}, n {}, k {}, group {}, carve {carve:?})",
                    spec.rows,
                    spec.n,
                    spec.k,
                    spec.group
                );
            }
        }
    }
}

/// The decode point and the fused GEMV on identical planes, at the ruler
/// `the_tiled_point_answers_what_the_fused_gemv_answers` argues for: the row's
/// own scale, because a dot product's cancellation is not bounded by its
/// own result.
#[test]
fn the_decode_point_answers_what_the_fused_gemv_answers() {
    let spec = Spec::new(4, 100, 512, 64);
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut rng = Lcg::seeded(0xdecaf);
    let seeded = planes(spec, &mut rng);
    let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);

    let count = spec.rows as usize * spec.n as usize;
    let tiled_at = gpu.zeros(count * 2);
    let mut y = Tensor::new(tiled_at, spec.rows, spec.n, Dtype::Bf16);
    tiled::matmul_gemv(
        &ctx,
        seated.act,
        seated.codes,
        seated.scales,
        seated.biases,
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the tiled decode point fires");

    let fused_at = gpu.zeros(count * 2);
    let mut y = Tensor::new(fused_at, spec.rows, spec.n, Dtype::Bf16);
    quant::matmul(
        &ctx,
        seated.act,
        seated.dense_codes,
        seated.dense_scales,
        OffsetKind::Post,
        Some(seated.dense_biases),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the fused projection fires");
    gpu.sync();

    let got: Vec<f32> = gpu
        .down::<u16>(tiled_at, count)
        .into_iter()
        .map(from_bf16)
        .collect();
    let fused: Vec<f32> = gpu
        .down::<u16>(fused_at, count)
        .into_iter()
        .map(from_bf16)
        .collect();
    for t in 0..spec.rows as usize {
        let at0 = t * spec.n as usize;
        let row = &fused[at0..at0 + spec.n as usize];
        let scale = row.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (r, f) in row.iter().enumerate() {
            let d = got[at0 + r];
            assert!(
                (d - f).abs() <= TOLERANCE * scale,
                "the two readings parted at row {t} column {r}: decode {d}, fused {f}, \
                 over a row whose scale is {scale}"
            );
        }
    }
}

/// The decode point refuses what it is not stamped for: a streamed seat and
/// a row count past the mma tile it holds in registers.
#[test]
fn the_decode_point_refuses_a_prefill_and_a_streamed_seat() {
    let spec = Spec::new(32, 64, 512, 64);
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut rng = Lcg::seeded(0xbad2);
    let seeded = planes(spec, &mut rng);
    let seated = seat(&mut gpu, &ctx, spec, &seeded, &mut rng);
    let mut y = Tensor::new(
        gpu.zeros(spec.rows as usize * spec.n as usize * 2),
        spec.rows,
        spec.n,
        Dtype::Bf16,
    );
    assert!(
        tiled::matmul_gemv(
            &ctx,
            seated.act,
            seated.codes,
            seated.scales,
            seated.biases,
            &mut y,
            GroupSeat::RESIDENT,
        )
        .is_err(),
        "a 32-row fire took the decode point"
    );
    let streamed = GroupSeat {
        cell: gpu.zeros(16),
        hits: 0,
    };
    let act = Tensor::new(seated.act.ptr, 1, spec.k, Dtype::Bf16);
    let mut one = Tensor::new(y.ptr, 1, spec.n, Dtype::Bf16);
    assert!(
        tiled::matmul_gemv(
            &ctx,
            act,
            seated.codes,
            seated.scales,
            seated.biases,
            &mut one,
            streamed,
        )
        .is_err(),
        "a streamed seat took the decode point"
    );
}

/// The carve is a pure function of the shape and the row count, and it is
/// the whole selection, so it is checked here rather than read off a timing
/// line.
#[test]
fn the_band_count_and_the_row_count_pick_the_carve() {
    // One band a block, in both directions and at every row count: nothing
    // is shared between the warps of a decode block, so a wider block is
    // only fewer blocks.
    for n in [512u32, 2048, 10240, 151936] {
        for rows in [1u32, 8, 16] {
            assert_eq!(tiled::carve_for(n, rows).bands, 1, "n {n} rows {rows}");
        }
    }
    // The wide direction has bands to spare and takes the shallow split; the
    // tall one has 128 of them and takes the deepest its rows allow.
    assert_eq!(tiled::carve_for(10240, 1).split, 16);
    assert_eq!(tiled::carve_for(10240, 16).split, 16);
    assert_eq!(tiled::carve_for(2048, 1).split, tiled::THIN_SPLIT);
    assert_eq!(tiled::carve_for(2048, tiled::THIN_ROWS).split, tiled::THIN_SPLIT);
    assert_eq!(tiled::carve_for(2048, tiled::THIN_ROWS + 1).split, tiled::WIDE_SPLIT);
    // A vocabulary-sized head has bands enough that the floor binds.
    assert_eq!(tiled::carve_for(151_936, 1).split, tiled::MIN_SPLIT);
}

// ─── (i) the decode sweep, report-only ─────────────────────────────────────

/// **THE TABLE THE DISPATCH FLIP STANDS ON**, printed and not asserted.
///
/// Both projection directions by four decode row counts by both carves,
/// against the arm the flip would replace — `quant::matmul`'s fused GEMV,
/// which is what a decode step fires today and what §J2a's 8.55 tok/s pin
/// was measured on. The flip is only safe if this point is not materially
/// slower there; the numbers move with the box, so this gate reports and the
/// wiki records.
#[test]
fn the_decode_shapes_are_swept_against_the_fused_gemv() {
    const ITERS: u32 = 200;
    let carves: [(&str, tiled::Carve); 6] = [
        ("1x8", tiled::Carve { bands: 1, split: 8 }),
        ("1x16", tiled::Carve { bands: 1, split: 16 }),
        ("1x32", tiled::Carve { bands: 1, split: 32 }),
        ("2x8", tiled::Carve { bands: 2, split: 8 }),
        ("2x16", tiled::Carve { bands: 2, split: 16 }),
        ("4x8", tiled::Carve { bands: 4, split: 8 }),
    ];

    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    println!("\ntiled decode sweep, group 64, {ITERS} iterations a point (ms)");
    for (label, n, k) in [
        ("n >= k  (up/gate)", 10240u32, 2048u32),
        ("k >  n  (down)", 2048, 10240),
    ] {
        let widest = Spec::new(16, n, k, 64);
        let mut rng = Lcg::seeded(0x71_3d);
        let seeded = planes(widest, &mut rng);
        let seated = seat(&mut gpu, &ctx, widest, &seeded, &mut rng);
        let y_at = gpu.zeros(16 * n as usize * 2);
        println!("\n  {label}: n {n} k {k}");
        let mut head = format!("    {:>6}", "tokens");
        for (name, _) in carves {
            head.push_str(&format!("  {name:>10}"));
        }
        head.push_str(&format!("  {:>10}", "fused GEMV"));
        println!("{head}");
        for tokens in [1u32, 2, 4, 8, 16] {
            let act = Tensor::new(seated.act.ptr, tokens, k, Dtype::Bf16);
            let timed = |fire: &mut dyn FnMut(&Ctx, &mut Tensor)| {
                let mut y = Tensor::new(y_at, tokens, n, Dtype::Bf16);
                for _ in 0..5 {
                    fire(&ctx, &mut y);
                }
                gpu.sync();
                let at = std::time::Instant::now();
                for _ in 0..ITERS {
                    fire(&ctx, &mut y);
                }
                gpu.sync();
                at.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS)
            };
            let mut line = format!("    {tokens:>6}");
            for (_, carve) in carves {
                let each = timed(&mut |ctx, y| {
                    tiled::gemv_with(
                        ctx,
                        "linear.matmul",
                        act,
                        seated.codes,
                        seated.scales,
                        seated.biases,
                        y,
                        GroupSeat::RESIDENT,
                        carve,
                    )
                    .expect("the tiled decode point fires");
                });
                line.push_str(&format!("  {each:10.4}"));
            }
            let each = timed(&mut |ctx, y| {
                quant::matmul(
                    ctx,
                    act,
                    seated.dense_codes,
                    seated.dense_scales,
                    OffsetKind::Post,
                    Some(seated.dense_biases),
                    Dtype::Bf16,
                    y,
                    GroupSeat::RESIDENT,
                )
                .expect("the fused projection fires");
            });
            line.push_str(&format!("  {each:10.4}"));
            println!("{line}");
        }
    }
}
