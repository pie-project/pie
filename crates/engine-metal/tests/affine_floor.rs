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
//! | `affine_qmm_t_splitk` + `qmm_splitk_reduce` | the split PAIR — partitioned contraction and the fold that is not optional |
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

/// The threadgroups the two families launch. **These are the SHADER's and not
/// the selection's**: `WM * WN * SIMD_SIZE` for the tile, two simdgroups for
/// the vector point, both fixed in `quant_qmm_t.metal` and `quant_qmv.metal`
/// where the accumulators are declared. `quant`'s grid helpers below supply
/// the lane counts, which are the part that IS a selection.
const QMM_GROUP: [u32; 3] = [32, 2, 2];
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

/// The weight the bank means: `code * scale + bias`, MLX's affine form and
/// the one `dequantize<U, N, 4>` computes.
fn weight(n: u32, k: u32) -> f32 {
    code(n, k) as f32 * scale(n, k / GROUP) + zero_point(n, k / GROUP)
}

/// One activation element: a signed power of two, so every product is exact
/// and the residual measured below is the ACCUMULATION's alone.
fn act(m: u32, k: u32) -> f32 {
    let sign = if (m + k) % 2 == 0 { 1.0 } else { -1.0 };
    sign * 0.5f32.powi(((m * 5 + k * 3) % 4) as i32)
}

/// `y = act x w^T` in f32 — the answer every point below is measured against.
fn reference(rows: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((rows * N) as usize);
    for m in 0..rows {
        for n in 0..N {
            out.push((0..K).map(|k| act(m, k) * weight(n, k)).sum::<f32>());
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
    partials: Buffer,

    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    x: Tensor,
    half_x: Tensor,
    out: Tensor,
    plane: Tensor,
}

impl Floor {
    /// Reserve and stage the synthetic bank, or print a skip and answer
    /// `None`.
    fn reserve(what: &str) -> Option<Floor> {
        if !device::present() {
            println!("SKIP {what}: this machine publishes no Metal device");
            return None;
        }
        let device = Context::bind().expect("the device binds");
        let pipelines = Pipelines::new();

        // The codes: one 4-bit value per element, eight to a u32 word, low
        // nibble first — `dequantize`'s own order, read here as the bytes it
        // reads.
        let mut codes = Vec::with_capacity((N * K / 8) as usize * 4);
        for n in 0..N {
            for word in 0..K / 8 {
                let packed = (0..8).fold(0u32, |acc, i| acc | (code(n, word * 8 + i) << (4 * i)));
                codes.extend_from_slice(&packed.to_le_bytes());
            }
        }
        // One scale and one zero point per group, `[N, K / GROUP]`.
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for n in 0..N {
            for g in 0..K / GROUP {
                scales.extend_from_slice(&bf16(scale(n, g)));
                biases.extend_from_slice(&bf16(zero_point(n, g)));
            }
        }
        // The activation, at the tallest rectangle any case asks for.
        let mut act_bytes = Vec::with_capacity((ROWS * K) as usize * 2);
        for m in 0..ROWS {
            for k in 0..K {
                act_bytes.extend_from_slice(&bf16(act(m, k)));
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

        // What the fires write: the staged halves, the result, and the
        // split's partials. `SPLIT` planes of `ROWS x N` f32 is the widest
        // any case below asks for.
        let staged = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(K) * 2)
            .expect("the staging plane reserves");
        let y = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(N) * 2)
            .expect("the result reserves");
        let partials =
            Buffer::zeroed(&device, u64::from(SPLIT) * u64::from(ROWS) * u64::from(N) * 4)
                .expect("the partials reserve");

        let handles = Handles::new();
        let bind = |buffer: &Buffer| {
            handles
                .bind(buffer, 0, buffer.bytes())
                .expect("the handle table seats a rectangle")
        };
        let floor = Floor {
            codes: Tensor::new(bind(&codes_buf), N, K, Dtype::MlxU4),
            scales: Tensor::new(bind(&scales_buf), N, K / GROUP, Dtype::Bf16),
            biases: Tensor::new(bind(&biases_buf), N, K / GROUP, Dtype::Bf16),
            x: Tensor::new(bind(&act_buf), ROWS, K, Dtype::Bf16),
            half_x: Tensor::new(bind(&staged), ROWS, K, Dtype::F16),
            out: Tensor::new(bind(&y), ROWS, N, Dtype::Bf16),
            plane: Tensor::new(bind(&partials), SPLIT * ROWS, N, Dtype::F32),
            _codes: codes_buf,
            _scales: scales_buf,
            _biases: biases_buf,
            activation: act_buf,
            staged,
            y,
            partials,
            device,
            pipelines,
            handles,
        };
        Some(floor)
    }

    /// Overwrite activation rows `from..to` with a number no real row holds.
    ///
    /// **THIS IS WHAT MAKES THE PAD CASE A TEST.** `mb_rows` pads a launch up
    /// to its rung on the ground that the added rows land in slots the fire
    /// does not read; a pad over rows that happened to hold valid data would
    /// pass whether or not that were true. 1024 is exact in `bfloat16`, three
    /// orders above anything the reference produces, and cannot overflow the
    /// product of a weight bounded by `3/8`.
    fn poison(&mut self, from: u32, to: u32) {
        let row: Vec<u8> = (0..K).flat_map(|_| bf16(1024.0)).collect();
        for m in from..to {
            self.activation
                .write(u64::from(m) * u64::from(K) * 2, &row)
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
        for buffer in [&mut self.y, &mut self.staged, &mut self.partials] {
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

/// How deep the split case partitions its contraction. Two, because `K` is
/// two whole groups and each partition must be a whole one.
const SPLIT: u32 = 2;

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

/// **THE PLAIN STAMPED TILE, AT EVERY RUNG IT IS STAMPED AT.** `bm` is the
/// row block and `bn` the column tile; the ladder reaches only the pair a
/// given prompt and a given tuning table select, so all nine are fired here.
#[test]
fn the_plain_stamped_tile_computes_it_at_every_row_and_column_rung() {
    let _serial = serialized();
    let Some(mut floor) = Floor::reserve("the plain stamped tile") else {
        return;
    };
    let (k, n) = (K as i32, N as i32);
    for bm in [16i32, 32, 64] {
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
                            QMM_GROUP,
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
    for bm in [16i32, 32, 64] {
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
                        QMM_GROUP,
                    )),
                    &gemm,
                )
            });
            agrees(&format!("precast bm={bm} bn={bn}"), &got, &reference(rows));
        }
    }
}

/// **THE SPLIT PAIR, AND THE FOLD THAT IS NOT OPTIONAL.** The split GEMM
/// lands `SPLIT` planes of partials at buffer 8 and writes NOTHING at buffer
/// 4; `qmm_splitk_reduce` is what turns them into the result. A split
/// dispatched without its reduce is not a slow answer but a wrong one, and
/// this is where that is stated in numbers.
///
/// The `f32` partials arm only: `engine_metal::scratch` reserves the plane in
/// `f32`, so `_bfloat16` is the point the dtype of the reservation would have
/// to change to reach.
#[test]
fn the_split_pair_partitions_the_contraction_and_folds_it_back() {
    let _serial = serialized();
    let Some(mut floor) = Floor::reserve("the split pair") else {
        return;
    };
    let (k, n) = (K as i32, N as i32);
    let split = SPLIT as i32;
    for bm in [16i32, 32, 64] {
        let rows = bm.unsigned_abs();
        let entry = quant::splitk_point(
            "floor.splitk",
            "f32_bfloat16",
            GROUP as i32,
            BITS as i32,
            bm,
        )
        .expect("the split point is stamped at this tile");
        let fold = quant::splitk_reduce_point("f32_bfloat16");
        let (codes, scales, biases, x, out, plane) = (
            floor.codes,
            floor.scales,
            floor.biases,
            floor.x,
            floor.out,
            floor.plane,
        );
        let got = floor.fired(rows, |ctx| {
            let stride = rows as i32 * n;
            ctx.fire(
                Fire::at(QMM_FILE, entry).apply(quant::splitk_grid(
                    "floor.splitk",
                    n,
                    rows as i32,
                    bm,
                    split,
                )?),
                &[
                    codes.arg(),
                    scales.arg(),
                    biases.arg(),
                    x.arg(),
                    ctx.absent()?,
                    k.arg(),
                    n.arg(),
                    ctx.absent()?,
                    plane.arg_mut(),
                    (k / split).arg(),
                    stride.arg(),
                ],
            )?;
            ctx.fire(
                Fire::at(QMM_FILE, fold)
                    .apply(quant::splitk_reduce_grid("floor.splitk", n, rows as i32)?),
                &[
                    ctx.absent()?,
                    ctx.absent()?,
                    ctx.absent()?,
                    ctx.absent()?,
                    out.arg_mut(),
                    ctx.absent()?,
                    n.arg(),
                    ctx.absent()?,
                    plane.arg(),
                    ctx.absent()?,
                    stride.arg(),
                    split.arg(),
                ],
            )
        });
        agrees(&format!("splitk bm={bm} split={split}"), &got, &reference(rows));
    }
}

/// **THE PAD IS INERT, AND HERE IS WHERE THAT IS A MEASUREMENT.**
///
/// `quant::mb_rows` pads a fire's row count up to its rung on the stated
/// ground that a GEMM row's output depends only on its own input row — so a
/// launch over more rows than the caller means computes, in the rows the
/// caller reads, what an unpadded launch would have. Everything above that
/// claim rests on it: it is what makes the tiled points reachable at a batch
/// that is not already an exact multiple of a rung, which for a decode is
/// almost never.
///
/// Stated here with the rows the pad adds POISONED, because that is the case
/// the claim is about — see [`Floor::poison`]. `four_bit_first_light`'s
/// ragged prompt is the same claim at a checkpoint, where what the tail holds
/// is whatever the activation slot happened to carry; this is the claim with
/// nothing else in it.
#[test]
fn a_padded_launch_computes_the_same_rows_over_a_poisoned_tail() {
    let _serial = serialized();
    let Some(mut floor) = Floor::reserve("the inert pad") else {
        return;
    };
    let (k, n) = (K as i32, N as i32);
    // Twenty rows of meaning at a rung of sixteen, padded to thirty-two —
    // `mb_rows`' own answer, asked rather than asserted from a table.
    let meant = 20u32;
    let bm = quant::bm_rung(meant as i32);
    let padded = quant::mb_rows(meant as i32, ROWS as i32, 8);
    assert_eq!(
        (bm, padded),
        (16, 32),
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
                QMM_GROUP,
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
    agrees("pad: the rows the fire reads", &got[..read], &reference(meant));
    // And the pad really launched: a run that quietly covered only the first
    // twenty rows would satisfy everything above it.
    assert!(
        got[read..].iter().any(|v| v.abs() > 100.0),
        "the rows the pad added hold {:?}, which is not the product of a poisoned row — \
         this case did not launch the pad it is about",
        &got[read..read + 8.min(got.len() - read)],
    );
}
