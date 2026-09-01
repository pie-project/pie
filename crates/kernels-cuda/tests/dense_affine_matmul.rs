//! **THE DENSE AFFINE FAMILY, ON A REAL DEVICE** (qwen4 stored-form wave,
//! generalized by QNF P1): `linear::quant::matmul` / `lm_head` over codes and
//! a per-group factor plane, held against a host fold of the same planes —
//! at both code widths, all four offset arms, both factor dtypes, and three
//! groupings.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test dense_affine_matmul -- --nocapture
//! ```
//!
//! The gates:
//!
//! ```text
//! (a) eight-bit codes: y = act · (s·c + b)^T matches the host fold, on a
//!     rectangle whose N is not a whole row tile (the clamp path runs)
//! (b) four-bit codes: the same, through the nibble decode
//! (c) `lm_head` answers what `matmul` answers (one launch, two names)
//! (d) the integer pre-offset (GPTQ/AWQ), the real one (HQQ) and the
//!     constant one (excess-binary, no plane at all) each match their fold
//! (e) an f16 factor plane folds what a bf16 one folds, on its own numbers
//! (f) thirty-two and a hundred and twenty-eight wide groupings both fold —
//!     the group is the factor plane's statement, not the kernel's constant
//! (g) the arms do not answer each other: one seed's planes under two arms
//!     are two different answers, so an axis dropped on the floor fails here
//! (h) the refusal ladder: a factor row that groups nothing whole, a group
//!     that is not a whole code word, a missing offset plane, an offset
//!     plane an arm cannot use, and a zero row of the wrong width
//! ```
//!
//! And at the foot of the file, its own block: the INTERIM prefill arm
//! (`matmul_via_dense` / `lm_head_via_dense`), which decodes the weight once
//! into scratch and hands the rectangle to the dense cuBLAS point.
//!
//! The reference folds in the kernel's own order — codes to a group in
//! storage order, one `(part, xsum)` pair per group — so what the comparison
//! allows is float reassociation across the warp reduce, nothing more.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, TOLERANCE, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::cudarc::cublas::sys as blas;
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::{self, OffsetKind};
use kernels_cuda::tensor::Tensor;

const ROWS: u32 = 5;
const N: u32 = 100;
const K: u32 = 128;

/// One stored form this point serves: the code width, the codes under one
/// factor, the offset arm, and the dtype the factor planes hold.
#[derive(Clone, Copy)]
struct Spec {
    bits: u32,
    group: usize,
    arm: OffsetKind,
    factor: Dtype,
    /// The activation rows the fire projects. The decode-shaped gates keep
    /// [`ROWS`]; the prefill arm's gates are the reason this is a field —
    /// its whole claim is about what happens when this number grows.
    rows: u32,
    /// The projection's rectangle: `n` columns landed over a `k`-wide row.
    /// Also a field for the prefill arm alone, which reuses ONE scratch
    /// slab across fires and must be shown two shapes to prove it.
    n: u32,
    k: u32,
}

impl Spec {
    /// The MLX affine triplet at `bits` — what the qwen4 serve path binds,
    /// and the arm every other spelling below is a departure from.
    const fn mlx(bits: u32) -> Self {
        Self {
            bits,
            group: 64,
            arm: OffsetKind::Post,
            factor: Dtype::Bf16,
            rows: ROWS,
            n: N,
            k: K,
        }
    }

    fn with(self, arm: OffsetKind, factor: Dtype) -> Self {
        Self {
            arm,
            factor,
            ..self
        }
    }

    fn grouped(self, group: usize) -> Self {
        Self { group, ..self }
    }

    fn rows(self, rows: u32) -> Self {
        Self { rows, ..self }
    }

    fn shape(self, n: u32, k: u32) -> Self {
        Self { n, k, ..self }
    }

    fn groups(self) -> usize {
        self.k as usize / self.group
    }

    /// The byte row a plane of one factor per group takes.
    fn factor_row(self) -> u32 {
        (self.k / self.group as u32) * 2
    }

    fn codes_row(self) -> u32 {
        if self.bits == 8 { self.k } else { self.k / 2 }
    }
}

/// **A cuBLAS HANDLE, WHICH `common::Gpu` DOES NOT CARRY.** Every golden in
/// that harness fires a jit kernel, which wants a stream and nothing else.
/// The prefill arm below reaches `linear::gemm`, and `Ctx::cublas` refuses a
/// context that holds no handle — so this file mints one and hangs it on the
/// same `Ctx::on(..).with_cublas(..)` an engine `Run` builds.
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

    /// `gpu`'s context with this handle on it.
    fn ctx(&self, gpu: &Gpu) -> Ctx {
        // SAFETY: the handle outlives every fire of the test that holds it,
        // and `linear::dense` binds it to this context's stream per call.
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

/// **A FACTOR AND THE NUMBER IT DECODES TO**, built bits-first so nothing in
/// this file rounds. `to_bf16` is the device's own rounding transcribed, but
/// there is no f16 twin in `common` and a second rounder would be a second
/// thing to be wrong; drawing the STORED bits and decoding them is exact for
/// both dtypes, which is all a reference needs.
///
/// `exp2` is the binary exponent, kept well inside the normal range so the
/// decode is one multiply and no subnormal case exists.
fn factor(rng: &mut Lcg, dtype: Dtype, exp2: i32) -> (u16, f32) {
    let draw = rng.unit();
    let negative = draw < 0.0;
    let (mantissa_bits, exp_bias) = match dtype {
        Dtype::F16 => (10u32, 15i32),
        Dtype::Bf16 => (7, 127),
        other => panic!("a factor plane holds bf16 or f16, not {other:?}"),
    };
    let step = (1u32 << mantissa_bits) as f32;
    let mantissa = (draw.abs() * step) as u32 & ((1 << mantissa_bits) - 1);
    let bits = (u16::from(negative) << 15)
        | (((exp2 + exp_bias) as u16) << mantissa_bits)
        | mantissa as u16;
    let magnitude = (1.0 + mantissa as f32 / step) * 2.0f32.powi(exp2);
    (bits, if negative { -magnitude } else { magnitude })
}

/// The planes a form seats, and the numbers a host fold reads back out of
/// them. `offset` is empty under `PreConst`, which reads no plane.
struct Planes {
    codes: Vec<u8>,
    scales: Vec<u8>,
    offset: Vec<u8>,
    scale_of: Vec<f32>,
    offset_of: Vec<f32>,
}

/// A seeded form, and the numbers the device will read out of it.
///
/// **EVERY ARM DRAWS THE SAME NUMBER OF TIMES**, including the one that
/// stores nothing, so two arms over one seed see the same codes, the same
/// scales and the same activations — which is what makes comparing two arms
/// a statement about the fold rather than about the generator.
fn planes(spec: Spec, rng: &mut Lcg) -> Planes {
    let groups = spec.groups();
    let factors = spec.n as usize * groups;
    let mut codes = Vec::new();
    let mut byte = 0u8;
    for at in 0..(spec.n as usize * spec.k as usize) {
        let code = (rng.unit().abs() * 255.0) as u32;
        match spec.bits {
            8 => codes.push(code as u8),
            4 => {
                let nibble = (code & 0xF) as u8;
                if at % 2 == 0 {
                    byte = nibble;
                } else {
                    codes.push(byte | (nibble << 4));
                }
            }
            _ => unreachable!(),
        }
    }

    let mut scales = Vec::with_capacity(factors * 2);
    let mut scale_of = Vec::with_capacity(factors);
    let mut offset = Vec::new();
    let mut offset_of = Vec::with_capacity(factors);
    for _ in 0..factors {
        // A scale near 2^-5, the order an affine weight's factors sit at.
        let (bits, value) = factor(rng, spec.factor, -5);
        scales.extend_from_slice(&bits.to_le_bytes());
        scale_of.push(value);

        // The value-domain offset and HQQ's real code-domain zero are the
        // same rectangle of the same reals, told apart only by the arm —
        // which is the reason the entry takes the arm and does not guess it.
        // The exponents differ because the two are not the same size: a bias
        // sits beside a value, a pre-scale zero beside a CODE.
        let exp2 = if spec.arm == OffsetKind::Post { 0 } else { 3 };
        let (bits, value) = factor(rng, spec.factor, exp2);
        match spec.arm {
            OffsetKind::Post | OffsetKind::PreReal => {
                offset.extend_from_slice(&bits.to_le_bytes());
                offset_of.push(value);
            }
            // One byte per group, holding an unsigned code-domain zero of
            // the codes' own width.
            OffsetKind::PreInt => {
                let z = (value.abs() as u32) & ((1 << spec.bits) - 1);
                offset.push(z as u8);
                offset_of.push(z as f32);
            }
            // The zero the format fixes: nothing stored, nothing read, and
            // the draw above spent anyway to keep the streams aligned.
            OffsetKind::PreConst => offset_of.push((1u32 << (spec.bits - 1)) as f32),
        }
    }
    Planes {
        codes,
        scales,
        offset,
        scale_of,
        offset_of,
    }
}

/// The code stored at flat position `at` of the weight, at either width.
fn code_at(spec: Spec, planes: &Planes, at: usize) -> f32 {
    match spec.bits {
        8 => f32::from(planes.codes[at]),
        4 => {
            let byte = planes.codes[at / 2];
            f32::from(if at % 2 == 0 { byte & 0xF } else { byte >> 4 })
        }
        _ => unreachable!(),
    }
}

/// The host fold, in the kernel's own per-group order.
fn fold(spec: Spec, planes: &Planes, x: &[f32]) -> Vec<f32> {
    let groups = spec.groups();
    let mut y = vec![0.0f32; spec.rows as usize * spec.n as usize];
    for t in 0..spec.rows as usize {
        let xt = &x[t * spec.k as usize..][..spec.k as usize];
        for r in 0..spec.n as usize {
            let mut acc = 0.0f32;
            for g in 0..groups {
                let mut part = 0.0f32;
                let mut xsum = 0.0f32;
                for j in 0..spec.group {
                    let at = r * spec.k as usize + g * spec.group + j;
                    let code = code_at(spec, planes, at);
                    let xv = xt[g * spec.group + j];
                    part += code * xv;
                    xsum += xv;
                }
                let fx = r * groups + g;
                let s = planes.scale_of[fx];
                let off = planes.offset_of[fx];
                acc += match spec.arm {
                    OffsetKind::Post => part * s + xsum * off,
                    OffsetKind::PreInt | OffsetKind::PreReal | OffsetKind::PreConst => {
                        s * (part - off * xsum)
                    }
                };
            }
            y[t * spec.n as usize + r] = acc;
        }
    }
    y
}

/// **THE PREFILL ARM'S OWN ORACLE.** The same planes, decoded ELEMENT BY
/// ELEMENT and rounded to bf16 exactly where `dequant_affine` rounds, then
/// folded in f32. It is a second reference and not a duplicate of [`fold`]:
/// the fused point never materialises a weight element, so the number it
/// answers and the number a decoded tile answers are two different (and
/// both correct) foldings of one stored row.
fn fold_decoded(spec: Spec, planes: &Planes, x: &[f32]) -> Vec<f32> {
    let groups = spec.groups();
    let mut y = vec![0.0f32; spec.rows as usize * spec.n as usize];
    for t in 0..spec.rows as usize {
        let xt = &x[t * spec.k as usize..][..spec.k as usize];
        for r in 0..spec.n as usize {
            let mut acc = 0.0f32;
            for g in 0..groups {
                let fx = r * groups + g;
                let s = planes.scale_of[fx];
                let off = planes.offset_of[fx];
                for j in 0..spec.group {
                    let at = r * spec.k as usize + g * spec.group + j;
                    let code = code_at(spec, planes, at);
                    let w = match spec.arm {
                        OffsetKind::Post => code * s + off,
                        OffsetKind::PreInt | OffsetKind::PreReal | OffsetKind::PreConst => {
                            s * (code - off)
                        }
                    };
                    acc += from_bf16(to_bf16(w)) * xt[g * spec.group + j];
                }
            }
            y[t * spec.n as usize + r] = acc;
        }
    }
    y
}

/// Which of the four entries over these planes fires: the fused point or the
/// prefill arm, under the matmul's op name or the head's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Fused,
    FusedHead,
    Dense,
    DenseHead,
}

impl Arm {
    /// Does this arm materialise a bf16 weight tile? — which is the only
    /// thing that decides which host fold is its oracle.
    fn decodes(self) -> bool {
        matches!(self, Self::Dense | Self::DenseHead)
    }
}

/// One fire on a device the CALLER owns, so a test can put two fires of two
/// shapes on ONE stream — which is what it takes to observe the prefill
/// arm's scratch slab being reused rather than freshly allocated.
fn fire_on(gpu: &mut Gpu, ctx: &Ctx, spec: Spec, arm: Arm) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::seeded(0x5eed + u64::from(spec.bits) + spec.group as u64);
    let seeded = planes(spec, &mut rng);
    let (x_raw, x) = rng.row(spec.rows as usize * spec.k as usize);

    let act = Tensor::new(gpu.up(&x_raw), spec.rows, spec.k, Dtype::Bf16);
    let codes = Tensor::new(gpu.up(&seeded.codes), spec.n, spec.codes_row(), Dtype::U8);
    let scales = Tensor::new(gpu.up(&seeded.scales), spec.n, spec.factor_row(), Dtype::U8);
    let biases = (spec.arm != OffsetKind::PreConst).then(|| {
        let width = (seeded.offset.len() / spec.n as usize) as u32;
        Tensor::new(gpu.up(&seeded.offset), spec.n, width, Dtype::U8)
    });
    let y_at = gpu.zeros(spec.rows as usize * spec.n as usize * 2);
    let mut y = Tensor::new(y_at, spec.rows, spec.n, Dtype::Bf16);

    let entry = match arm {
        Arm::Fused => quant::matmul,
        Arm::FusedHead => quant::lm_head,
        Arm::Dense => quant::matmul_via_dense,
        Arm::DenseHead => quant::lm_head_via_dense,
    };
    entry(
        ctx,
        act,
        codes,
        scales,
        spec.arm,
        biases,
        spec.factor,
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the affine projection fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, spec.rows as usize * spec.n as usize);
    let want = if arm.decodes() {
        fold_decoded(spec, &seeded, &x)
    } else {
        fold(spec, &seeded, &x)
    };
    (got.into_iter().map(from_bf16).collect(), want)
}

fn fired(spec: Spec, head: bool) -> (Vec<f32>, Vec<f32>) {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    fire_on(
        &mut gpu,
        &ctx,
        spec,
        if head { Arm::FusedHead } else { Arm::Fused },
    )
}

fn held(spec: Spec, head: bool) {
    let (got, want) = fired(spec, head);
    for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "y[{at}] answered {g} where the host fold says {w} ({:?}, {} bits, group {})",
            spec.arm,
            spec.bits,
            spec.group
        );
    }
}

#[test]
fn the_eight_bit_projection_matches_the_host_fold() {
    held(Spec::mlx(8), false);
}

#[test]
fn the_four_bit_projection_matches_the_host_fold() {
    held(Spec::mlx(4), false);
}

#[test]
fn the_head_answers_what_the_matmul_answers() {
    held(Spec::mlx(8), true);
}

#[test]
fn the_integer_pre_offset_matches_the_host_fold() {
    held(Spec::mlx(4).with(OffsetKind::PreInt, Dtype::F16), false);
}

#[test]
fn the_real_pre_offset_matches_the_host_fold() {
    held(Spec::mlx(4).with(OffsetKind::PreReal, Dtype::F16), false);
}

#[test]
fn the_constant_pre_offset_matches_the_host_fold() {
    held(Spec::mlx(4).with(OffsetKind::PreConst, Dtype::F16), false);
    held(Spec::mlx(8).with(OffsetKind::PreConst, Dtype::Bf16), false);
}

#[test]
fn an_f16_factor_plane_matches_the_host_fold() {
    held(Spec::mlx(8).with(OffsetKind::Post, Dtype::F16), false);
}

#[test]
fn a_grouping_of_thirty_two_and_of_a_hundred_and_twenty_eight_both_fold() {
    held(Spec::mlx(4).grouped(32), false);
    held(Spec::mlx(4).grouped(128), false);
    held(Spec::mlx(8).grouped(32), false);
}

/// **THE ARMS ARE NOT EACH OTHER.** Every gate above compares a fire against
/// a fold that agrees with it, so a kernel that ignored `kOffset` and always
/// took one arm would still have to disagree SOMEWHERE — this is where. One
/// seed's planes under `PreInt` and under `PreConst` differ only in the zero
/// subtracted, and the two answers must part.
#[test]
fn the_offset_arm_changes_the_answer() {
    let spec = Spec::mlx(4).with(OffsetKind::PreInt, Dtype::F16);
    let (pre_int, _) = fired(spec, false);
    let (pre_const, _) = fired(spec.with(OffsetKind::PreConst, Dtype::F16), false);
    assert!(
        pre_int
            .iter()
            .zip(pre_const.iter())
            .any(|(a, b)| !close(*a, *b)),
        "the integer and the constant pre-offset answered the same numbers"
    );
}

#[test]
fn a_foreign_grouping_and_a_missing_plane_are_refused() {
    let mut gpu = Gpu::open();
    let act = Tensor::new(gpu.zeros(2 * K as usize), 1, K, Dtype::Bf16);
    let codes = Tensor::new(gpu.zeros(K as usize), 1, K, Dtype::U8);
    let mut y = Tensor::new(gpu.zeros(2), 1, 1, Dtype::Bf16);

    // Forty-two factors do not group a 128-wide row into whole groups — what
    // the fixed sixty-four became once the group came off the plane.
    let ragged = Tensor::new(gpu.zeros(84), 1, 84, Dtype::U8);
    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        ragged,
        OffsetKind::Post,
        Some(ragged),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "a fractional grouping fired");

    // Sixty-four factors over a 128-wide row is a group of two codes, and
    // the kernel reads eight-bit codes four to a 32-bit word.
    let fine = Tensor::new(gpu.zeros(128), 1, 128, Dtype::U8);
    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        fine,
        OffsetKind::Post,
        Some(fine),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "a two-code group fired");

    let scales = Tensor::new(gpu.zeros(4), 1, 4, Dtype::U8);
    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        scales,
        OffsetKind::Post,
        None,
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "a two-plane post-offset projection fired");
}

/// The arm the caller declares and the plane it bound must be the same
/// statement — a `PreConst` fire would read none of an offset plane, and a
/// `PreInt` zero row is one byte per group and not one factor per group.
#[test]
fn the_offset_arm_and_its_plane_must_agree() {
    let mut gpu = Gpu::open();
    let act = Tensor::new(gpu.zeros(2 * K as usize), 1, K, Dtype::Bf16);
    let codes = Tensor::new(gpu.zeros(K as usize), 1, K, Dtype::U8);
    let mut y = Tensor::new(gpu.zeros(2), 1, 1, Dtype::Bf16);
    // Two groups of sixty-four: four bytes of factors, two bytes of zeros.
    let scales = Tensor::new(gpu.zeros(4), 1, 4, Dtype::U8);
    let zeros = Tensor::new(gpu.zeros(2), 1, 2, Dtype::U8);

    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        scales,
        OffsetKind::PreConst,
        Some(zeros),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "a constant pre-offset read a plane");

    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        scales,
        OffsetKind::PreInt,
        None,
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "an integer pre-offset fired with no zeros");

    // The factor rectangle, where the arm states one byte per group.
    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        scales,
        OffsetKind::PreInt,
        Some(scales),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "a four-byte zero row fired over two groups");

    // And the mirror: a post-offset arm handed the `PreInt` container.
    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        scales,
        OffsetKind::Post,
        Some(zeros),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "a two-byte bias row fired over two groups");
}

/// The bits-first generator, held against the rounding `common` already
/// carries: a bf16 factor built here IS what `to_bf16` would have written,
/// and decodes to the number the fold reads. Nothing device-side runs — this
/// gates the REFERENCE, which every other assertion above trusts.
#[test]
fn the_generated_factor_bits_are_the_number_the_fold_reads() {
    let mut rng = Lcg::seeded(7);
    for _ in 0..64 {
        let (bits, value) = factor(&mut rng, Dtype::Bf16, -5);
        assert_eq!(bits, to_bf16(value), "a bf16 factor is not its own bits");
        assert_eq!(
            from_bf16(bits),
            value,
            "a bf16 factor does not decode to the number the fold reads"
        );
    }
}

// ─── the prefill arm (INTERIM): decode once, project dense ──────────────────
//
// `quant::matmul` carves one block column per activation row and re-reads the
// whole weight inside each of them. That is parity with cuBLAS bf16 at one
// token and 98–189× slower than it over 128–2048 rows, so above a row count
// the caller decodes the weight ONCE into scratch and fires the dense point:
// `quant::matmul_via_dense`, same signature, same planes, same declared arm.
//
// The gates below:
//
// ```text
// (i)   the decoded arm answers the fused point's numbers at a prefill shape,
//       at both code widths and on the two arms a serving checkpoint brings
// (ii)  the decoded head answers the decoded matmul (one pair, two names)
// (iii) a streamed seat is refused — the slab has no fixed rectangle to hold
// (iv)  the scratch slab grows for a bigger projection and is not misread by
//       a smaller one that follows it on the same stream
// ```

/// A prefill's rows — above `engine_cuda`'s `PREFILL_ROWS` gate, and small
/// enough that the host fold stays a test and not a benchmark.
const PREFILL: u32 = 32;

/// The prefill twin of [`Spec::mlx`]: the same stored form, at a row count
/// that would send the dispatch down the decoded arm.
fn prefill(bits: u32, arm: OffsetKind) -> Spec {
    Spec::mlx(bits).with(arm, Dtype::Bf16).rows(PREFILL)
}

/// **THE TWO ARMS ANSWER THE SAME NUMBERS AND NOT THE SAME BITS.** The fused
/// point folds `s·c + b` in f32 inside the dot and materialises no weight;
/// the decoded arm rounds every element to bf16 once, on the way into the
/// tile. So each arm is held against ITS OWN fold — `fold` for the fused
/// point, `fold_decoded` for the decoded one, which is the honest oracle for
/// a rounding the kernel really does — and only then are the two ANSWERS
/// held against each other.
///
/// **AND THE CROSS-ARM RULER IS THE ROW'S AND NOT THE ELEMENT'S**, which is
/// a measurement and not a preference. `common::close` scales by the element
/// it is checking, and a dot product's cancellation is not bounded by its own
/// result. Measured over the four forms below: the arms disagree by up to
/// **14% of an element** — the worst is an eight-bit `Post` column that lands
/// at 1.46 out of a row whose scale is 109 — and by at most **0.95% of the
/// row** that column sits in. One bf16 rounding per weight element perturbs
/// the row's SCALE, so the row's scale is the ruler, held at the harness's
/// own tolerance. (The per-element gate above keeps `close`, because
/// `fold_decoded` rounds where the kernel rounds and the decoded arm agrees
/// with it to 0.39%.)
#[test]
fn the_decoded_arm_answers_what_the_fused_point_answers() {
    for spec in [
        prefill(8, OffsetKind::Post),
        prefill(4, OffsetKind::Post),
        prefill(8, OffsetKind::PreInt),
        prefill(4, OffsetKind::PreInt),
    ] {
        let mut gpu = Gpu::open();
        let blas = Blas::open();
        let ctx = blas.ctx(&gpu);
        let (fused, fused_want) = fire_on(&mut gpu, &ctx, spec, Arm::Fused);
        let (dense, dense_want) = fire_on(&mut gpu, &ctx, spec, Arm::Dense);
        for (at, (g, w)) in dense.iter().zip(dense_want.iter()).enumerate() {
            assert!(
                close(*g, *w),
                "the decoded arm's y[{at}] answered {g} where the decoded fold says \
                 {w} ({:?}, {} bits)",
                spec.arm,
                spec.bits
            );
        }
        for (at, (g, w)) in fused.iter().zip(fused_want.iter()).enumerate() {
            assert!(close(*g, *w), "the fused point's y[{at}] moved: {g} vs {w}");
        }
        for t in 0..spec.rows as usize {
            let at0 = t * spec.n as usize;
            let row = &fused[at0..at0 + spec.n as usize];
            let scale = row.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            for (r, f) in row.iter().enumerate() {
                let d = dense[at0 + r];
                assert!(
                    (d - f).abs() <= TOLERANCE * scale,
                    "the two arms parted at row {t} column {r}: decoded {d}, fused \
                     {f}, over a row whose scale is {scale} ({:?}, {} bits)",
                    spec.arm,
                    spec.bits
                );
            }
        }
    }
}

/// One pair of launches, two op names — `linear::gemm`'s pairing, kept.
#[test]
fn the_decoded_head_answers_what_the_decoded_matmul_answers() {
    let spec = prefill(8, OffsetKind::Post);
    let mut gpu = Gpu::open();
    let blas = Blas::open();
    let ctx = blas.ctx(&gpu);
    let (matmul, _) = fire_on(&mut gpu, &ctx, spec, Arm::Dense);
    let (head, _) = fire_on(&mut gpu, &ctx, spec, Arm::DenseHead);
    assert_eq!(
        matmul, head,
        "the decoded head and the decoded matmul are one launch under two names"
    );
}

/// **A MOVING PLANE HAS NO RECTANGLE TO DECODE.** The decoded arm reads the
/// weight once into a slab sized `n·k`; a streamed seat's planes live
/// wherever the tier staged them this fire, and the address the kernel would
/// read is the cell's and not the launch's. The refusal is typed rather than
/// a silent fall back to the fused point — and the same call under a
/// RESIDENT seat fires, so what is refused is the seat and not the planes.
#[test]
fn a_streamed_seat_has_no_decoded_tile() {
    let spec = prefill(8, OffsetKind::Post);
    let mut gpu = Gpu::open();
    let blas = Blas::open();
    let ctx = blas.ctx(&gpu);
    let mut rng = Lcg::seeded(0x5eed);
    let seeded = planes(spec, &mut rng);
    let (x_raw, _) = rng.row(spec.rows as usize * spec.k as usize);

    let act = Tensor::new(gpu.up(&x_raw), spec.rows, spec.k, Dtype::Bf16);
    let codes = Tensor::new(gpu.up(&seeded.codes), spec.n, spec.codes_row(), Dtype::U8);
    let scales = Tensor::new(gpu.up(&seeded.scales), spec.n, spec.factor_row(), Dtype::U8);
    let width = (seeded.offset.len() / spec.n as usize) as u32;
    let biases = Tensor::new(gpu.up(&seeded.offset), spec.n, width, Dtype::U8);
    let mut y = Tensor::new(
        gpu.zeros(spec.rows as usize * spec.n as usize * 2),
        spec.rows,
        spec.n,
        Dtype::Bf16,
    );

    // A `MoeGroupBases` cell, which is what a streamed seat points at. It is
    // never dereferenced: the refusal happens before any launch.
    let streamed = GroupSeat {
        cell: gpu.zeros(16),
        hits: 0,
    };
    let refused = quant::matmul_via_dense(
        &ctx,
        act,
        codes,
        scales,
        OffsetKind::Post,
        Some(biases),
        Dtype::Bf16,
        &mut y,
        streamed,
    );
    assert!(refused.is_err(), "a streamed seat took the decoded arm");

    quant::matmul_via_dense(
        &ctx,
        act,
        codes,
        scales,
        OffsetKind::Post,
        Some(biases),
        Dtype::Bf16,
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the same planes fire under a resident seat");
    gpu.sync();
}

/// **ONE SLAB, THREE FIRES, TWO SHAPES.** `Ctx::scratch` grows a named slab
/// and never shrinks it, so a bigger projection after a smaller one moves
/// the block, and a smaller one after a bigger one reads a slab with a tail
/// of somebody else's decoded weight beyond its own `n·k`. Both are the
/// misread this gate would catch: every fire is held against its own fold,
/// and all three share ONE `Gpu` — one stream, therefore one slab key.
#[test]
fn the_decoded_tile_grows_and_is_never_misread() {
    let small = prefill(4, OffsetKind::Post);
    let large = small.shape(256, 256);
    let mut gpu = Gpu::open();
    let blas = Blas::open();
    let ctx = blas.ctx(&gpu);

    for (turn, spec) in [small, large, small].into_iter().enumerate() {
        let (got, want) = fire_on(&mut gpu, &ctx, spec, Arm::Dense);
        for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                close(*g, *w),
                "fire {turn} over a [{}, {}] weight answered {g} at y[{at}] where the \
                 decoded fold says {w}",
                spec.n,
                spec.k
            );
        }
    }
}
