//! **THE NUMERIC FLOOR UNDER THE AFFINE LADDER.** `device_floor` asks
//! whether the shaders compile and whether one entry's arguments land where
//! its `[[buffer(n)]]` declarations say. This file asks the next question of
//! the one family whose answer is arithmetic: given a bank whose every code,
//! scale and zero point is known, does each point `linear::quant` can select
//! compute `y = act x w^T`?
//!
//! # Why it is separate from the checkpoint test
//!
//! `four_bit_first_light` is the gold assertion — token for token against
//! `mlx_lm` over a real MLX checkpoint — and it needs a 400 MiB artifact and
//! a tokenizer that are not this repository's to ship. It also answers only
//! the arms a fire happens to select: the row rungs a prompt's length reaches
//! and, since the tuning table freezes at the first `current()`, only the
//! arms one process's `[metal.tuning]` admits.
//!
//! This file needs nothing but a GPU. It binds every live affine point by
//! name, at every row tile it is stamped at, over one synthetic bank, and
//! compares against a host reference — so a point the ladder does not
//! currently select is still measured, and a point the ladder stops selecting
//! does not go quietly unmeasured.
//!
//! # The points, and what each one is
//!
//! | point | what it is |
//! |---|---|
//! | `affine_qmv_fast` | the vector floor, and the REFERENCE ARM: the only one that guards every read it makes, so it is what a bisect trusts |
//! | `affine_qmm_t` | the plain stamped tile, minted by `PIE_STAMP_qmm_t` |
//! | `cast_qmm_input` + `affine_qmm_t_fp16_precast` | the pre-cast PAIR — the staging dispatch and the GEMM that reads its halves at buffer 12 |
//!
//! **THE SPLIT PAIR IS NOT ON THAT LIST ANY MORE.** `affine_qmm_t_splitk` and
//! `qmm_splitk_reduce` are still in `quant_qmm_t.metal` and nothing composes
//! a name for either: the arm was deleted for having different bits from the
//! rest of the family (which this file measured), and its selection helpers
//! went with it. A point no `linear::quant` entry can name is a point this
//! floor cannot bind.
//!
//! Each is fired with the ARGUMENT LIST `linear::quant` builds for it, in the
//! same order and at the same indices, because that list is half of what is
//! under test: a shader that computes correctly off arguments the driver
//! seats one position over computes something else.
//!
//! # The tolerance, and why it is not tight
//!
//! Every one of these tiles accumulates in the operand's own precision — the
//! plain and split points in `bfloat16`, the pre-cast one in `half` — so the
//! residual against an `f32` host reference is a real property of the kernel
//! and not slack. What makes the gate meaningful anyway is the SIZE of the
//! failures it is looking for: reading the wrong nibble, indexing a scale by
//! the wrong group, or seating an argument one buffer over does not perturb
//! an answer, it replaces it. The measured residual is printed at every
//! point and the gate sits an order of magnitude above it.
//!
//! # Gating
//!
//! Apple at compile time, and SKIPS at run time when the machine publishes no
//! device — `device_floor`'s idiom, for `device_floor`'s reason.
//!
//! ```text
//! cargo test -p engine-metal --release --test affine_floor -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Arg, Ctx, Fire, Grid, Tensor};
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME** — `device_floor`'s reason, restated because this
/// file has its own tests and its own mutex: each binds a device and reserves
/// buffers, and two of them compiling shaders at once meets the Metal
/// compiler's own concurrency and learns nothing.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The two sources the points live in — `linear::quant`'s own two constants,
/// which are private to it because nothing but it selects a point.
const QMM_FILE: &str = "linear/quant_qmm_t.metal";
const QMV_FILE: &str = "linear/quant_qmv.metal";
const QMV_ROWS_FILE: &str = "linear/quant_qmv_rows.metal";

/// The threadgroup the vector point launches. **This is the SHADER's and not
/// the selection's**: two simdgroups, fixed in `quant_qmv.metal` where the
/// accumulators are declared. `quant`'s grid helpers below supply the lane
/// counts, which are the part that IS a selection.
///
/// The TILE's threadgroup is no longer a constant and so is not one here:
/// `quant::qmm_group` answers 64 lanes at the 8 rung and 128 above it,
/// because `BlockMMA`'s warp tile is `BM / (8 * WM)` rows and an 8-row block
/// admits ONE simdgroup down M. Asking that function is half of what these
/// cases test — a rung fired at the other rung's threadgroup reads half of
/// every staged tile out of whatever threadgroup memory last held, which is a
/// wrong answer and one this file is the floor under.
const QMV_GROUP: [u32; 3] = [32, 2, 1];

/// The format every point here is stamped for. Group 64 at 4 bits is the one
/// the pre-cast loader exists for and the one `mlx_lm.convert` writes at this
/// tree's default settings, so it is the format the ladder actually meets.
const GROUP: u32 = 64;
const BITS: u32 = 4;

/// The contraction. Two whole groups, four `BK = 32` steps, and — divided in
/// two — still a whole group per split, which is what lets the split point
/// run at the same shape as the others.
const K: u32 = 128;

/// The output width: divisible by all three column tiles, so `bn` is a free
/// choice rather than a constraint the shape imposes.
const N: u32 = 128;

/// The tallest launch any case below asks for, and so the rows every
/// rectangle is reserved at.
const ROWS: u32 = 64;

/// How far a point may sit from the host reference, as a fraction of the
/// reference's own rms.
///
/// **MEASURED AT EXACTLY ZERO, AT EVERY POINT, ON AN M1 MAX**, and the
/// synthetic bank above is why: every activation is a signed power of two and
/// every weight a multiple of `1/64` with at most five significant bits, so
/// every product is exact and all 8192 elements of the reference are
/// themselves `bfloat16` numbers. There is nothing here for the result's own
/// dtype to round.
///
/// The gate is not zero even so. What these tiles accumulate in is the
/// OPERAND's precision — `bfloat16` for the plain and split points, `half`
/// for the pre-cast one — and a partial sum is not held to the five bits its
/// terms are, so a device whose matrix unit accumulates narrowly where this
/// one does not would round where this one did not: at 128 terms that is
/// about 3% of the answer's rms, which is arithmetic and not a defect. Five
/// percent is above that and two orders below what this file is looking for.
/// A nibble read at the wrong end, a scale indexed by the wrong group or an
/// argument seated one buffer over does not perturb an answer, it replaces
/// it — the one such failure this file has actually produced measured 45%.
const TOLERANCE: f32 = 0.05;

/// One code, spread over all sixteen so no arm can be right by reading a
/// constant nibble, and varying with the group so a scale indexed by the
/// wrong group lands on the wrong values rather than on similar ones.
fn code(n: u32, k: u32) -> u32 {
    (n * 7 + k * 3 + (k / GROUP) * 5) % 16
}

/// One group's scale — a small multiple of `1/64`, exact in `bfloat16`, so
/// the reference's weights are the device's weights and the only difference
/// between the two answers is the accumulation.
fn scale(n: u32, g: u32) -> f32 {
    (1 + (n + g) % 3) as f32 / 64.0
}

/// One group's zero point. `-8 * scale` centres the sixteen codes on zero,
/// which is what a real affine bank does and what makes a sum over `K` a walk
/// rather than a ramp.
fn zero_point(n: u32, g: u32) -> f32 {
    -8.0 * scale(n, g)
}

/// One activation element: a signed power of two, so every product is exact
/// and the residual measured below is the ACCUMULATION's alone.
fn act(m: u32, k: u32) -> f32 {
    let sign = if (m + k) % 2 == 0 { 1.0 } else { -1.0 };
    sign * 0.5f32.powi(((m * 5 + k * 3) % 4) as i32)
}

/// **THE SECOND BANK, AND IT EXISTS TO BE ROUNDED.**
///
/// Everything above this line is chosen so that no arm has anything to round:
/// signed powers of two against five-bit codes at a scale of `1/64` makes
/// every product exact and every reference element a `bfloat16` number, which
/// is what lets [`agrees`] read a residual as the kernel's own accumulation.
/// It also makes every arm on this ladder land the SAME BITS whatever order
/// it walks k in — which is the one thing [`the_fingerprint_matrix`] must not
/// be able to conclude by accident.
///
/// So the fingerprint runs against a bank with the exactness taken out:
///
///   * the SCALE is an ordinary `bfloat16` fraction rather than a multiple of
///     `1/64`, so `code * scale` needs eleven mantissa bits — exact in
///     `half`, rounded in `bfloat16`, which is the ONE difference between the
///     pre-cast weight loader and the plain one;
///   * the ACTIVATION is an ordinary `bfloat16` number rather than a power of
///     two, so a sum of products is a sum that rounds and the ORDER of the
///     sum reaches the answer.
///
/// A bank that cannot tell two orders apart is not a control; it is a blind
/// spot, and this file had one.
fn scale_rough(n: u32, g: u32) -> f32 {
    // Ordinary bf16 fractions near 1/64, none of them a dyadic rational with
    // fewer than eight significant bits.
    of_bf16(&bf16(0.0121 + 0.0037 * ((n * 5 + g * 3) % 13) as f32))[0]
}

/// The zero point of the rough bank. Still `-8 * scale` — a power of two
/// times a `bfloat16` value is exact, so this is a REAL affine bank's zero
/// point and not a second source of rounding.
fn zero_point_rough(n: u32, g: u32) -> f32 {
    -8.0 * scale_rough(n, g)
}

/// One activation element of the rough bank: an ordinary `bfloat16` number in
/// `[-1, 1)`, from a hash rather than from a ramp so that no row is another
/// row scaled.
fn act_rough(m: u32, k: u32) -> f32 {
    let mut h = (m.wrapping_mul(0x9e37_79b9) ^ k.wrapping_mul(0x85eb_ca6b)) as u64;
    h ^= h >> 15;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 13;
    of_bf16(&bf16(((h & 0xffff) as f32 / 32768.0) - 1.0))[0]
}

/// Which bank a [`Floor`] was staged with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bank {
    /// The exact bank the residual gates are measured against.
    Exact,
    /// The rough bank the fingerprint is taken over.
    Rough,
}

impl Bank {
    fn scale(self, n: u32, g: u32) -> f32 {
        match self {
            Bank::Exact => scale(n, g),
            Bank::Rough => scale_rough(n, g),
        }
    }
    fn zero_point(self, n: u32, g: u32) -> f32 {
        match self {
            Bank::Exact => zero_point(n, g),
            Bank::Rough => zero_point_rough(n, g),
        }
    }
    fn act(self, m: u32, k: u32) -> f32 {
        match self {
            Bank::Exact => act(m, k),
            Bank::Rough => act_rough(m, k),
        }
    }
    /// The weight the bank means: `code * scale + bias`, MLX's affine form
    /// and the one `dequantize<U, N, 4>` computes.
    fn weight(self, n: u32, k: u32) -> f32 {
        code(n, k) as f32 * self.scale(n, k / GROUP) + self.zero_point(n, k / GROUP)
    }
}

/// `y = act x w^T` in f32 — the answer every point below is measured against.
fn reference(rows: u32) -> Vec<f32> {
    reference_of(Bank::Exact, K, rows)
}

fn reference_of(bank: Bank, depth: u32, rows: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((rows * N) as usize);
    for m in 0..rows {
        for n in 0..N {
            out.push(
                (0..depth)
                    .map(|k| bank.act(m, k) * bank.weight(n, k))
                    .sum::<f32>(),
            );
        }
    }
    out
}

/// f32 → the two bytes of its bf16 truncation, little-endian. Every value
/// this file stages is exactly representable, so the truncation is the round.
fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

fn of_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect()
}

/// The device, the pipelines, and every rectangle the points read and write —
/// bound once, because a handle table that is never rewound holds its rows for
/// the life of the test.
struct Floor {
    device: Context,
    pipelines: Pipelines,
    handles: Handles,
    /// Kept alive for their handles, and written back through where a case
    /// needs to change what a fire reads. `_codes`/`_scales`/`_biases` are
    /// never touched from the host again.
    _codes: Buffer,
    _scales: Buffer,
    _biases: Buffer,
    activation: Buffer,
    staged: Buffer,
    y: Buffer,

    /// **THE CONTRACTION IS THE FLOOR'S AND NOT THE FILE'S.** The residual
    /// gates run at [`K`], where the exact bank makes every reference element
    /// a `bfloat16` number; the fingerprint runs at [`DEEP_K`], because two
    /// k-orders that part once every few thousand terms part invisibly over
    /// 128 of them. See [`the_fingerprint_matrix`].
    k: u32,

    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    x: Tensor,
    half_x: Tensor,
    out: Tensor,
}

impl Floor {
    /// Reserve and stage the exact bank, or print a skip and answer `None`.
    fn reserve(what: &str) -> Option<Floor> {
        Floor::staged(what, Bank::Exact, K)
    }

    /// The same reservation over whichever bank the caller is asking about —
    /// see [`Bank`] for why there are two.
    fn staged(what: &str, bank: Bank, k: u32) -> Option<Floor> {
        if !device::present() {
            println!("SKIP {what}: this machine publishes no Metal device");
            return None;
        }
        let device = Context::bind().expect("the device binds");
        let pipelines = Pipelines::new();

        // The codes: one 4-bit value per element, eight to a u32 word, low
        // nibble first — `dequantize`'s own order, read here as the bytes it
        // reads.
        let mut codes = Vec::with_capacity((N * k / 8) as usize * 4);
        for n in 0..N {
            for word in 0..k / 8 {
                let packed = (0..8).fold(0u32, |acc, i| acc | (code(n, word * 8 + i) << (4 * i)));
                codes.extend_from_slice(&packed.to_le_bytes());
            }
        }
        // One scale and one zero point per group, `[N, K / GROUP]`.
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for n in 0..N {
            for g in 0..k / GROUP {
                scales.extend_from_slice(&bf16(bank.scale(n, g)));
                biases.extend_from_slice(&bf16(bank.zero_point(n, g)));
            }
        }
        // The activation, at the tallest rectangle any case asks for.
        let mut act_bytes = Vec::with_capacity((ROWS * k) as usize * 2);
        for m in 0..ROWS {
            for kk in 0..k {
                act_bytes.extend_from_slice(&bf16(bank.act(m, kk)));
            }
        }

        let hold = |bytes: &[u8]| {
            let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a reservation");
            buffer.write(0, bytes).expect("the staging lands");
            buffer
        };
        let codes_buf = hold(&codes);
        let scales_buf = hold(&scales);
        let biases_buf = hold(&biases);
        let act_buf = hold(&act_bytes);

        // What the fires write: the staged halves and the result.
        let staged = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(k) * 2)
            .expect("the staging plane reserves");
        let y = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(N) * 2)
            .expect("the result reserves");

        let handles = Handles::new();
        let bind = |buffer: &Buffer| {
            handles
                .bind(buffer, 0, buffer.bytes())
                .expect("the handle table seats a rectangle")
        };
        let floor = Floor {
            codes: Tensor::new(bind(&codes_buf), N, k, Dtype::MlxU4),
            scales: Tensor::new(bind(&scales_buf), N, k / GROUP, Dtype::Bf16),
            biases: Tensor::new(bind(&biases_buf), N, k / GROUP, Dtype::Bf16),
            x: Tensor::new(bind(&act_buf), ROWS, k, Dtype::Bf16),
            half_x: Tensor::new(bind(&staged), ROWS, k, Dtype::F16),
            out: Tensor::new(bind(&y), ROWS, N, Dtype::Bf16),
            _codes: codes_buf,
            _scales: scales_buf,
            _biases: biases_buf,
            activation: act_buf,
            staged,
            y,
            device,
            pipelines,
            handles,
            k,
        };
        Some(floor)
    }

    /// Overwrite activation rows `from..to` with a number no real row holds.
    ///
    /// **THIS IS WHAT MAKES THE PAD CASE A TEST.** `mb_block` pads a launch up
    /// to its rung on the ground that the added rows land in slots the fire
    /// does not read; a pad over rows that happened to hold valid data would
    /// pass whether or not that were true. 1024 is exact in `bfloat16`, three
    /// orders above anything the reference produces, and cannot overflow the
    /// product of a weight bounded by `3/8`.
    fn poison(&mut self, from: u32, to: u32) {
        let row: Vec<u8> = (0..self.k).flat_map(|_| bf16(1024.0)).collect();
        for m in from..to {
            self.activation
                .write(u64::from(m) * u64::from(self.k) * 2, &row)
                .expect("the poison lands");
        }
    }

    /// Encode one chain onto a fresh command buffer, commit it, and read the
    /// top `rows x N` of the result back.
    ///
    /// The result and the two working planes are wiped first: a point that
    /// wrote nothing at all would otherwise pass off the previous case's
    /// answer, which is the one failure a floor test must not be able to
    /// miss.
    fn fired(
        &mut self,
        rows: u32,
        chain: impl FnOnce(&Ctx<'_>) -> Result<(), kernels_metal::Error>,
    ) -> Vec<f32> {
        for buffer in [&mut self.y, &mut self.staged] {
            let blank = vec![0u8; buffer.bytes() as usize];
            buffer.write(0, &blank).expect("the plane wipes");
        }
        let frame = self.device.frame().expect("a command buffer opens");
        {
            let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
            chain(&sink).expect("the chain encodes");
        }
        frame.commit().expect("the fire completes");

        let mut bytes = vec![0u8; (rows * N) as usize * 2];
        self.y.read(0, &mut bytes).expect("the result reads back");
        of_bf16(&bytes)
    }
}

/// Print the residual and gate on it. See [`TOLERANCE`].
fn agrees(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: {} rows came back", got.len());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{what}: the answer holds a NaN or an infinity"
    );
    let rms = |xs: &[f32]| (xs.iter().map(|v| v * v).sum::<f32>() / xs.len() as f32).sqrt();
    let residual: Vec<f32> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    let (scale, error) = (rms(want), rms(&residual));
    let worst = residual
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    println!(
        "{what}: residual {:.5} rms against {:.5}, worst element {:.5} — {:.2}%",
        error,
        scale,
        worst,
        100.0 * error / scale
    );
    assert!(
        scale > 1e-3,
        "{what}: the reference itself is flat, so nothing here is being asked"
    );
    assert!(
        error <= TOLERANCE * scale,
        "{what}: {:.2}% of the answer's own magnitude, which is not accumulation — \
         got {:?} where the host says {:?}",
        100.0 * error / scale,
        &got[..got.len().min(8)],
        &want[..want.len().min(8)],
    );
}

/// **THE REFERENCE ARM.** `affine_qmv_fast` guards every read it makes, takes
/// no tile that has to divide anything, and is what the ladder falls to when
/// no other rung will hold — so it is the arm a bisect trusts, and the one
/// that has to be checked against something other than another arm.
#[test]
fn the_vector_point_computes_the_affine_product_the_host_computes() {
    let _serial = serialized();
    let Some(mut floor) = Floor::reserve("the affine vector point") else {
        return;
    };
    // Four rows: the vector point takes any count, and more than one says the
    // per-row stride is the row's and not a constant.
    let rows = 4u32;
    let point = quant::qmv_point("floor.qmv", "fast", GROUP as i32, BITS as i32)
        .expect("the vector point is stamped at this format");
    let (k, n) = (K as i32, N as i32);
    let (codes, scales, biases, x, out) = (
        floor.codes,
        floor.scales,
        floor.biases,
        floor.x,
        floor.out,
    );
    let got = floor.fired(rows, |ctx| {
        ctx.fire(
            Fire::at(QMV_FILE, point.entry).apply(Grid::of(
                quant::qmv_grid("floor.qmv", rows as i32, n)?,
                QMV_GROUP,
            )),
            &[
                codes.arg(),
                scales.arg(),
                biases.arg(),
                x.arg(),
                out.arg_mut(),
                k.arg(),
                n.arg(),
            ],
        )
    });
    agrees("qmv", &got, &reference(rows));
}

/// **THE FOLDED VECTOR POINT LANDS THE ONE-ROW POINT'S BITS, EXACTLY.**
///
/// This is the one arm of the ladder whose two speeds ARE two speeds of one
/// kernel, and it is worth a bit-for-bit gate where the tile-versus-vector
/// parting is not. `quant_qmv_rows.metal`'s `qdot_staged` is
/// `quant_qmv.metal`'s `qdot` with its pack handed in rather than read in:
/// same term order, same `float` accumulator, same `scale * SUM + bias * sum`
/// factoring, same `simd_sum` fold over the same thirty-two lanes. Nothing
/// about the fold touches the arithmetic — it changes who FETCHES, not who
/// adds.
///
/// So `act_x_wt` may move a width between the two without moving a token,
/// which is what makes `qmv_rows_max` a pure performance knob rather than a
/// second numerical policy beside `qmm_min_batch`. An edit that made the
/// folded point cheaper by reassociating its sum would take that away, and
/// this is where it would be caught.
///
/// Run over [`Bank::Rough`] at [`DEEP_K`] for `the_fingerprint_matrix`'s
/// reason: the exact bank rounds nowhere, so it would report every arm
/// identical and prove nothing.
#[test]
fn the_folded_vector_point_lands_the_one_row_bits() {
    let _serial = serialized();
    let Some(mut floor) = Floor::staged("the folded vector point", Bank::Rough, DEEP_K) else {
        return;
    };
    let (k, n) = (floor.k as i32, N as i32);
    let (codes, scales, biases, x, out) = (
        floor.codes,
        floor.scales,
        floor.biases,
        floor.x,
        floor.out,
    );

    let one_row = |rows: u32| {
        let point = quant::qmv_point("floor.qmv", "fast", GROUP as i32, BITS as i32)
            .expect("the one-row point is stamped");
        move |ctx: &Ctx<'_>| {
            ctx.fire(
                Fire::at(QMV_FILE, point.entry).apply(Grid::of(
                    quant::qmv_grid("floor.qmv", rows as i32, n)?,
                    QMV_GROUP,
                )),
                &[
                    codes.arg(),
                    scales.arg(),
                    biases.arg(),
                    x.arg(),
                    out.arg_mut(),
                    k.arg(),
                    n.arg(),
                ],
            )
        }
    };
    let folded = |rows: u32, r: i32, p: i32| {
        let point = quant::qmv_rows_point("floor.qmv_rows", GROUP as i32, BITS as i32, r, p)
            .expect("the folded point is stamped");
        move |ctx: &Ctx<'_>| {
            ctx.fire(
                Fire::at(QMV_ROWS_FILE, point.entry).stamp(point.stamp).apply(
                    Grid::of(
                        quant::qmv_rows_grid("floor.qmv_rows", rows as i32, r, n)?,
                        QMV_GROUP,
                    ),
                ),
                &[
                    codes.arg(),
                    scales.arg(),
                    biases.arg(),
                    x.arg(),
                    out.arg_mut(),
                    k.arg(),
                    n.arg(),
                    (rows as i32).arg(),
                ],
            )
        }
    };

    // Every fold this file can mint, at a batch it divides — including the
    // rungs the M1 Max's table declines to select, because the gate is on
    // the KERNEL and not on which rung a machine happens to want.
    for &(rows, r, p) in &[
        (2u32, 2i32, 1i32),
        (2, 2, 2),
        (4, 2, 1),
        (4, 4, 1),
        (4, 4, 2),
        (8, 2, 1),
        (8, 4, 1),
        (8, 8, 1),
        (8, 8, 2),
    ] {
        let want = floor.fired(rows, one_row(rows));
        let got = floor.fired(rows, folded(rows, r, p));
        assert_eq!(
            fingerprint(&got),
            fingerprint(&want),
            "the fold at r={r} p={p} over {rows} rows parts from the one-row point: {:?}",
            first_parting(&got, &want),
        );
    }
    eprintln!(
        "the folded vector point: nine (rows, fold, pack) triples, all bit-identical to \
         `affine_qmv_fast` over {DEEP_K} terms of the rough bank"
    );
}

/// **THE PLAIN STAMPED TILE, AT EVERY RUNG IT IS STAMPED AT.** `bm` is the
/// row block and `bn` the column tile; the ladder reaches only the pair a
/// given prompt and a given tuning table select, so all TWELVE are fired
/// here — four row rungs against three column tiles.
///
/// The 8 rung joins them without a line of its own in the source: the stamp
/// macro is unchanged and `affine_qmm_t_aligned` reads its `WM` from
/// `qmm_wm<BM>()`, so the same `PIE_STAMP_qmm_t` that mints the 16 mints the
/// 8 at one simdgroup down M.
#[test]
fn the_plain_stamped_tile_computes_it_at_every_row_and_column_rung() {
    let _serial = serialized();
    let Some(mut floor) = Floor::reserve("the plain stamped tile") else {
        return;
    };
    let (k, n) = (K as i32, N as i32);
    for bm in [8i32, 16, 32, 64] {
        for bn in [16i32, 32, 64] {
            let rows = bm.unsigned_abs();
            let point = quant::qmm_point(
                "floor.qmm_t",
                "",
                "PIE_STAMP_qmm_t",
                GROUP as i32,
                BITS as i32,
                bm,
                bn,
            )
            .expect("an axis point");
            let (codes, scales, biases, x, out) = (
                floor.codes,
                floor.scales,
                floor.biases,
                floor.x,
                floor.out,
            );
            let got = floor.fired(rows, |ctx| {
                ctx.fire(
                    Fire::at(QMM_FILE, point.entry).stamp(point.stamp).apply(
                        Grid::of(
                            quant::qmm_grid("floor.qmm_t", n, bn, rows as i32, bm, 1)?,
                            quant::qmm_group(bm),
                        ),
                    ),
                    &[
                        codes.arg(),
                        scales.arg(),
                        biases.arg(),
                        x.arg(),
                        out.arg_mut(),
                        k.arg(),
                        n.arg(),
                    ],
                )
            });
            agrees(&format!("qmm bm={bm} bn={bn}"), &got, &reference(rows));
        }
    }
}

/// **THE PRE-CAST PAIR.** Two dispatches and one plane: `cast_qmm_input`
/// writes `rows x K` halves at buffer 12 and the GEMM reads them there,
/// leaving the bf16 activation seat at buffer 3 NULL. The two are fired into
/// one command buffer exactly as the ladder fires them, because the claim is
/// about the pair — a staging pass that wrote the wrong rectangle and a GEMM
/// that read the right one would each look correct alone.
#[test]
fn the_precast_pair_stages_the_activation_and_computes_it_in_half() {
    let _serial = serialized();
    let Some(mut floor) = Floor::reserve("the pre-cast pair") else {
        return;
    };
    let (k, n) = (K as i32, N as i32);
    for bm in [8i32, 16, 32, 64] {
        for bn in [16i32, 32, 64] {
            let rows = bm.unsigned_abs();
            let entry = quant::precast_point("floor.precast", "", bm, bn)
                .expect("the pre-cast point is stamped at this tile");
            let (codes, scales, biases, x, half_x, out) = (
                floor.codes,
                floor.scales,
                floor.biases,
                floor.x,
                floor.half_x,
                floor.out,
            );
            let got = floor.fired(rows, |ctx| {
                let count = rows as i32 * k;
                let mut cast = vec![ctx.absent()?; 3];
                cast.push(x.arg());
                for _ in 4..12 {
                    cast.push(ctx.absent()?);
                }
                cast.push(half_x.arg_mut());
                cast.push(count.arg());
                ctx.fire(
                    Fire::at(QMM_FILE, quant::PRECAST_STAGE)
                        .apply(quant::precast_stage("floor.precast", rows as i32, k)?),
                    &cast,
                )?;
                let mut gemm = vec![
                    codes.arg(),
                    scales.arg(),
                    biases.arg(),
                    ctx.absent()?,
                    out.arg_mut(),
                    k.arg(),
                    n.arg(),
                ];
                for _ in 7..12 {
                    gemm.push(ctx.absent()?);
                }
                gemm.push(half_x.arg());
                ctx.fire(
                    Fire::at(QMM_FILE, entry).apply(Grid::of(
                        quant::qmm_grid("floor.precast", n, bn, rows as i32, bm, 1)?,
                        quant::qmm_group(bm),
                    )),
                    &gemm,
                )
            });
            agrees(&format!("precast bm={bm} bn={bn}"), &got, &reference(rows));
        }
    }
}

#[test]
fn a_padded_launch_computes_the_same_rows_over_a_poisoned_tail() {
    let _serial = serialized();
    let Some(mut floor) = Floor::reserve("the inert pad") else {
        return;
    };
    let (k, n) = (K as i32, N as i32);
    // Two cases, WIDEST FIRST, because the poison stays where it is written:
    // the narrow case reads rows 0..5, which the wide case's tail does not
    // reach, and the other order would poison rows the wide case reads.
    //
    //   * twenty rows at a rung of sixteen, padded to thirty-two — the case
    //     that was here before the 8 rung existed;
    //   * FIVE rows at a rung of EIGHT, padded to eight, which is what the 8
    //     rung was stamped for. Five rows used to pad to sixteen and so
    //     needed a slot sixteen rows deep to launch at all; three poisoned
    //     rows now sit where eleven did.
    for (meant, rung, pad) in [(20u32, 16i32, 32i32), (5, 8, 8)] {
        // `mb_block`' own answer, asked rather than asserted from a table.
        let (bm, padded) = quant::mb_block(meant as i32, ROWS as i32)
            .expect("this floor's slot holds a padded launch");
        assert_eq!(
            (bm, padded),
            (rung, pad),
            "the rung and the pad this case is written for"
        );
        floor.poison(meant, padded.unsigned_abs());

        let point = quant::qmm_point(
            "floor.pad",
            "",
            "PIE_STAMP_qmm_t",
            GROUP as i32,
            BITS as i32,
            bm,
            32,
        )
        .expect("an axis point");
        let (codes, scales, biases, x, out) = (
            floor.codes,
            floor.scales,
            floor.biases,
            floor.x,
            floor.out,
        );
        let got = floor.fired(padded.unsigned_abs(), |ctx| {
            ctx.fire(
                Fire::at(QMM_FILE, point.entry).stamp(point.stamp).apply(Grid::of(
                    quant::qmm_grid("floor.pad", n, 32, padded, bm, 1)?,
                    quant::qmm_group(bm),
                )),
                &[
                    codes.arg(),
                    scales.arg(),
                    biases.arg(),
                    x.arg(),
                    out.arg_mut(),
                    k.arg(),
                    n.arg(),
                ],
            )
        });
        let read = (meant * N) as usize;
        agrees(
            &format!("pad bm={bm}: the rows the fire reads"),
            &got[..read],
            &reference(meant),
        );
        // And the pad really launched: a run that quietly covered only the first
        // twenty rows would satisfy everything above it.
        assert!(
            got[read..].iter().any(|v| v.abs() > 100.0),
            "the rows the pad added hold {:?}, which is not the product of a poisoned row — \
             this case did not launch the pad it is about",
            &got[read..read + 8.min(got.len() - read)],
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The cross-arm fingerprint
// ─────────────────────────────────────────────────────────────────────────────

/// **THE CONTRACTION THE FINGERPRINT IS TAKEN OVER.** [`K`] is 128 — two
/// groups and four `BK` steps — and two k-orders that part once in a few
/// thousand terms part nowhere at all over 128 of them: the first run of this
/// matrix at [`K`] reported every arm identical, which was the shape talking
/// and not the kernels. 1024 is sixteen groups and thirty-two `BK` steps,
/// still a whole number of 64-code groups per split at [`SPLIT`] = 2, and the
/// order of a real projection's contraction (Qwen3.5-0.8B's hidden is 1024).
const DEEP_K: u32 = 1024;

/// The bf16 bit pattern of every element, folded — an answer's IDENTITY
/// rather than its magnitude.
fn fingerprint(row: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in row {
        for byte in ((v.to_bits() >> 16) as u16).to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn first_parting(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
        .map(|at| (at, a[at], b[at]))
}

/// **THE TILE FAMILY IS ONE FINGERPRINT, AND STAYS ONE.**
///
/// This is a KERNEL-PROPERTY regression and not a policy gate. Every
/// instantiation of `quant_qmm_t.metal`'s template — twelve plain rungs and
/// twelve pre-cast ones — walks k in the same ascending `BK = 32` order into
/// an `f32` accumulator, and `BM`, `BN`, `WM` and `WN` decide only who holds
/// which element. So they land the same bits, and the ladder above them is
/// free to move between rungs: `mb_block` walks DOWN the row rungs on a
/// number the composition owns, the pre-cast rung declines whenever the
/// staging plane will not hold the rectangle, and neither is visible in the
/// answer. An edit that quietly changed one rung's k-order — a reordered
/// loop, a different accumulator width, a fused epilogue — would break that
/// and shows up here, cheaply, rather than as a mystery at a checkpoint.
///
/// **THE VECTOR POINT IS PRINTED AND NOT ASSERTED.** It never materializes a
/// weight at all (`scale * Σ code x + bias * Σ x` against the tile's
/// `bf16(code * scale + bias)`) and folds thirty-two lanes with `simd_sum`,
/// so it is a different computation and its fingerprint is expected to part.
/// The ladder takes it below `qmm_min_batch` anyway, on the owner's ruling —
/// see `linear::quant::act_x_wt`'s header — and a gate that pinned the
/// disagreement would fail the day somebody made them agree, which is the
/// wrong direction to guard. It is here so the parting stays VISIBLE and its
/// size stays readable.
///
/// # Why the second bank
///
/// The bank the residual gates use is chosen so that nothing rounds — signed
/// powers of two against five-bit codes at a scale of `1/64`. Run over that,
/// this matrix reported every arm identical, INCLUDING the split pair, and
/// that was the shape talking. [`Bank::Rough`] and [`DEEP_K`] are what make
/// the question askable; see both.
#[test]
fn every_tile_the_ladder_may_pick_lands_the_same_bits() {
    let _serial = serialized();
    let Some(mut floor) = Floor::staged("the fingerprint matrix", Bank::Rough, DEEP_K) else {
        return;
    };
    let (k, n) = (DEEP_K as i32, N as i32);
    // **ONE AND TWO ARE ON THIS LIST EVEN THOUGH THE LADDER SENDS THOSE
    // WIDTHS TO THE VECTOR POINT.** The tile rungs have to agree with each
    // other at every width `mb_block` can pad TO, and it pads a one-row fire
    // to eight — so the claim is about the launch and not about which arm a
    // fire of that width takes. A rung no row count divides is fired at the
    // row count `mb_block` would pad it to and compared over the rows the
    // fire brought, which is the launch the driver actually makes.
    for rows in [1u32, 2, 8, 64] {
        let mut arms: Vec<(String, Vec<f32>)> = Vec::new();
        let padded = |bm: i32| -> Option<u32> {
            let padded = rows.div_ceil(bm.unsigned_abs()) * bm.unsigned_abs();
            (padded <= ROWS).then_some(padded)
        };
        let head = (rows * N) as usize;

        // qmv
        {
            let point = quant::qmv_point("fp.qmv", "fast", GROUP as i32, BITS as i32).unwrap();
            let (codes, scales, biases, x, out) =
                (floor.codes, floor.scales, floor.biases, floor.x, floor.out);
            let got = floor.fired(rows, |ctx| {
                ctx.fire(
                    Fire::at(QMV_FILE, point.entry).apply(Grid::of(
                        quant::qmv_grid("fp.qmv", rows as i32, n)?,
                        QMV_GROUP,
                    )),
                    &[
                        codes.arg(),
                        scales.arg(),
                        biases.arg(),
                        x.arg(),
                        out.arg_mut(),
                        k.arg(),
                        n.arg(),
                    ],
                )
            });
            arms.push(("qmv".to_string(), got));
        }

        for bm in [8i32, 16, 32, 64] {
            let Some(launch) = padded(bm) else { continue };
            for bn in [16i32, 32, 64] {
                let point = quant::qmm_point(
                    "fp.qmm",
                    "",
                    "PIE_STAMP_qmm_t",
                    GROUP as i32,
                    BITS as i32,
                    bm,
                    bn,
                )
                .unwrap();
                let (codes, scales, biases, x, out) =
                    (floor.codes, floor.scales, floor.biases, floor.x, floor.out);
                let mut got = floor.fired(launch, |ctx| {
                    ctx.fire(
                        Fire::at(QMM_FILE, point.entry).stamp(point.stamp).apply(
                            Grid::of(
                                quant::qmm_grid("fp.qmm", n, bn, launch as i32, bm, 1)?,
                                quant::qmm_group(bm),
                            ),
                        ),
                        &[
                            codes.arg(),
                            scales.arg(),
                            biases.arg(),
                            x.arg(),
                            out.arg_mut(),
                            k.arg(),
                            n.arg(),
                        ],
                    )
                });
                got.truncate(head);
                arms.push((format!("qmm bm={bm} bn={bn}"), got));
            }

            // precast, same rungs
            for bn in [16i32, 32, 64] {
                let entry = quant::precast_point("fp.precast", "", bm, bn).unwrap();
                let (codes, scales, biases, x, half_x, out) = (
                    floor.codes,
                    floor.scales,
                    floor.biases,
                    floor.x,
                    floor.half_x,
                    floor.out,
                );
                let mut got = floor.fired(launch, |ctx| {
                    let count = launch as i32 * k;
                    let mut cast = vec![ctx.absent()?; 3];
                    cast.push(x.arg());
                    for _ in 4..12 {
                        cast.push(ctx.absent()?);
                    }
                    cast.push(half_x.arg_mut());
                    cast.push(count.arg());
                    ctx.fire(
                        Fire::at(QMM_FILE, quant::PRECAST_STAGE)
                            .apply(quant::precast_stage("fp.precast", launch as i32, k)?),
                        &cast,
                    )?;
                    let mut gemm = vec![
                        codes.arg(),
                        scales.arg(),
                        biases.arg(),
                        ctx.absent()?,
                        out.arg_mut(),
                        k.arg(),
                        n.arg(),
                    ];
                    for _ in 7..12 {
                        gemm.push(ctx.absent()?);
                    }
                    gemm.push(half_x.arg());
                    ctx.fire(
                        Fire::at(QMM_FILE, entry).apply(Grid::of(
                            quant::qmm_grid("fp.precast", n, bn, launch as i32, bm, 1)?,
                            quant::qmm_group(bm),
                        )),
                        &gemm,
                    )
                });
                got.truncate(head);
                arms.push((format!("precast bm={bm} bn={bn}"), got));
            }

        }

        println!("\n=== rows = {rows}, K = {DEEP_K} ===");
        // The reference is the FIRST TILE, because the tile family is what
        // the claim is about — not the vector point, which is only here to be
        // seen next to it.
        let (base_name, base) = arms
            .iter()
            .find(|(name, _)| name.starts_with("qmm"))
            .cloned()
            .expect("the matrix fires at least one tile");
        for (name, got) in &arms {
            let note = match first_parting(&base, got) {
                None => "IDENTICAL".to_string(),
                Some((at, a, b)) => {
                    let worst = base
                        .iter()
                        .zip(got)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let apart = base
                        .iter()
                        .zip(got)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    format!(
                        "parts at {at} ({a} against {b}), {apart} of {} elements, \
                         worst |delta| {worst}",
                        base.len()
                    )
                }
            };
            println!("  {name:24} {:016x}  vs {base_name}: {note}", fingerprint(got));
        }
        for (name, got) in &arms {
            if !name.starts_with("qmm") && !name.starts_with("precast") {
                continue;
            }
            assert!(
                first_parting(&base, got).is_none(),
                "at {rows} rows, `{name}` is not the bits `{base_name}` lands. Every \
                 instantiation of this template is supposed to be the same arithmetic \
                 at a different launch shape — the row block, the column tile and \
                 whether the operands were staged to `half` decide who holds which \
                 element and never the order k is walked in. A parting here is a \
                 k-order that moved, and `mb_block`'s rung walk and the pre-cast \
                 rung's decline both rest on it."
            );
        }
    }
}
