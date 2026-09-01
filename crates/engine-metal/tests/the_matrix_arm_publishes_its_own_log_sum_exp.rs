//! **THE MATRIX KERNEL'S LOG-SUM-EXP ARM, ON A REAL APPLE GPU.**
//!
//! `attn/sdpa_paged_mma.metal` grew a second entry point,
//! `sdpa_paged_mma_lse`, and `attn::arbiter::should_mma` stopped declining
//! fires that want a log-sum-exp plane. The host half of that wave is settled
//! and green with no device (`cargo test -p kernels-metal --lib attn::arbiter`
//! — the admission at 64, the refusal at 128/256, the `_lse` entry named with
//! the plane on buffer 18 as a WRITE binding, the tuning hatch, the fragment
//! map). **What no test in this tree has ever done is fire the thing.**
//!
//! This file is the device half, and it is the gate the verify queue calls
//! the one that matters: `.wiki/alto/metal-verify-queue.md`, "Session H — the
//! matrix kernel's log-sum-exp arm", entries 2 through 5. Its four claims,
//! in the order they can fail:
//!
//! 1. [`the_two_arms_agree_on_o_and_on_lse`] — scalar versus matrix on ONE
//!    fixture, both planes. **Wrong register layout is wrong answers, not
//!    slow ones**: a fold taken over the wrong lanes lands an `lse` that is a
//!    quarter of the row's mass, i.e. off by about `log2(4) = 2`, which is
//!    far outside any half-precision band and is what this gate is shaped to
//!    catch.
//! 2. [`a_live_row_that_kept_no_key_publishes_a_true_negative_infinity`] —
//!    EXACT, not close. Both consumers branch on `isfinite`
//!    (`merge_lse.metal` drops the whole side, `attn_sink.metal` rescales by
//!    1.0), so a `NEG_INF` of `-3.0e38` leaking out where a true `-inf`
//!    belongs is a wrong answer no tolerance sees. `NEG_INF` is the running
//!    max's SEED; it must never be its published value.
//! 3. [`each_arms_lse_describes_that_arms_own_o`] — the property the owner's
//!    numerics ruling actually requires. Bit-identity to the scalar arm is
//!    NOT required (fast ladders default). What is required is that the
//!    matrix arm's `lse` describes the matrix arm's OWN `o`. A plane that is
//!    self-consistent but drifts from the scalar arm passes; a plane that is
//!    closer to the scalar arm than to its own output fails, and that is the
//!    right way round.
//! 4. [`the_rows_past_n_rows_are_left_exactly_as_they_were_poisoned`] — the
//!    matrix arm addresses `lse` by `my_row`, which is a fragment-row offset
//!    and NOT a loop counter, so an off-by-a-fragment lands in the next
//!    tile's rows rather than out of bounds, where no bounds check would ever
//!    see it.
//!
//! # Why both arms are hand-encoded rather than arbitrated
//!
//! `attn::arbiter::arbitrate` picks the arm from `DeviceTuning`, and
//! `kernels_metal::tuning::current()` memoizes in a `OnceLock` — one answer
//! per process. A file that flipped the knob to get both arms would be
//! measuring whichever load ran first. The two entry points take the SAME
//! nineteen seats in the same order (buffers 0..=18, verified below against
//! both sources), so the honest shape is one argument list, two `Fire`s, and
//! nothing else different but the entry name, the file and the threadgroup
//! width. That also makes the diff a statement about the KERNELS rather than
//! about the arbiter.
//!
//! # Gating
//!
//! Apple-only at compile time; SKIPS at run time when the box publishes no
//! device, saying so. An `#[ignore]`d test on the one box that could run it
//! is a test nobody runs.
//!
//! ```text
//! cargo test -p engine-metal --release --test the_matrix_arm_publishes_its_own_log_sum_exp -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{Arg, Ctx, Encode, Fire, Grid, Tensor};
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME.** Each test here binds a device, reserves a pool
/// and compiles two attention points; two of them doing it at once is a way
/// to meet the Metal compiler's own concurrency and learn nothing.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device_or_skip(what: &str) -> Option<Context> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    Some(Context::bind().expect("the device binds"))
}

// ─────────────────────────────────────────────────────────────────────────────
// The fixture
// ─────────────────────────────────────────────────────────────────────────────

/// **THE ONLY WIDTH `sdpa_paged_mma_lse` IS STAMPED AT.** `MMA_LSE_HEAD_DIM`
/// is 64 and the source instantiates `(bfloat16, bfloat, 64, 16)` and nothing
/// else, which is also gpt-oss's head width and the whole reason the arm
/// exists.
const D: usize = 64;

const Q_HEADS: usize = 4;
const GQA: usize = 2;
const KV_HEADS: usize = Q_HEADS / GQA;

/// The live rows. Four, and every one of them is carrying a different part of
/// the claim — see [`POSITIONS`].
const ROWS: usize = 4;

/// **THE TILE BOTH ARMS CUT ROWS AT.** `SDPA_TILE` in the host, `QT` in both
/// shaders. It is 32 against [`ROWS`] = 4 on purpose: twenty-eight rows of
/// the tile are past `n_rows`, which is what claim 4 reads.
const TILE: usize = 32;

const PAGE_SIZE: usize = 3;
const PAGES: usize = 7;
const SCALE: f32 = 0.125;
const WINDOW: i32 = 4;

/// **THE PAGE TABLE IS REVERSED AND TWO REQUESTS SHARE THE POOL.** Physical
/// page `n` is nowhere near logical page `n`, so a walk that forgot the
/// indirection reads a plausible vector from the wrong place; and request 1's
/// pages are interleaved among request 0's, so a walk that forgot
/// `kv_page_indptr` reads the other request's history.
const INDICES: [u32; PAGES] = [6, 4, 2, 0, 5, 3, 1];
const INDPTR: [u32; 3] = [0, 4, 7];

/// Row 0: request 0, deep enough that the window clamps. Row 1: request 0,
/// masked with a hole. Row 2: request 1, position below the window so the
/// clamp does NOT engage. Row 3: request 1, masked to nothing — claim 2.
const POSITIONS: [i32; ROWS] = [6, 9, 2, 5];

/// **NON-DECREASING, AND IT IS A CONTRACT AND NOT A CONVENIENCE.** Both
/// bodies walk the tile in runs of equal `req_of_token` (`sub` / `sub_hi`),
/// and the matrix body's membership test is `mine = live && my_req == r` — a
/// request appearing in two runs would be counted twice.
const REQUESTS: [i32; ROWS] = [0, 0, 1, 1];

const MASK_STRIDE: u32 = 12;

/// Which rows read the mask at all. Rows 1 and 3; rows 0 and 2 keep whatever
/// the window leaves them.
const MASK_ON: [u8; ROWS] = [0, 1, 0, 1];

/// **f32 → the two bytes of its bf16 truncation, little-endian.**
fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

fn from_bf16(b: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
}

/// **VALUES WITH PERIOD 15, WHICH IS COPRIME WITH EVERY WIDTH HERE.** The
/// deleted `device_attention` fixture learned this the hard way: at period 16
/// every page, head and slot held the same vector and the page walk was
/// unobservable — the test passed with the indirection removed.
fn spread(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed) % 15) as f32 / 8.0 - 1.0)
        .collect()
}

fn as_bf16_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| bf16(*v)).collect()
}

fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: every `T` used here (`i32`, `u32`, `f32`) has no padding and no
    // invalid bit patterns, and the slice's lifetime is the borrow's.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// **THE MASK, AND THE ROW THAT KEEPS NOTHING.**
///
/// The keep test both bodies run is `kp <= q_pos && kp >= my_start`, then —
/// when the row's `attention_mask_enabled` byte is set — `kp < stride &&
/// mask[row * stride + kp] != 0`. So a row keeps NOTHING exactly when every
/// position in `[my_start, q_pos]` is zeroed, which is what row 3 is.
fn mask_bytes() -> Vec<u8> {
    let mut mask = vec![1u8; ROWS * MASK_STRIDE as usize];
    let start = |row: usize| {
        let p = POSITIONS[row];
        if WINDOW > 0 && p >= WINDOW { p - WINDOW + 1 } else { 0 }
    };
    // Row 1: one hole inside its own kept window, so it keeps SOME keys.
    let hole = 7usize;
    assert!(
        (start(1)..=POSITIONS[1]).contains(&(hole as i32)),
        "row 1's hole has to be inside the window it keeps, or it masks nothing"
    );
    mask[MASK_STRIDE as usize + hole] = 0;
    // Row 3: every position inside its own kept window zeroed — the
    // fully-masked live row claim 2 is written for.
    for kp in start(3)..=POSITIONS[3] {
        mask[3 * MASK_STRIDE as usize + kp as usize] = 0;
    }
    mask
}

/// **THE POISON THE TAIL IS WRITTEN WITH.** A value no correct fire could
/// produce: `o` is a convex combination of `v` rows, all of which live in
/// `[-1, 1]`, and `lse` is a log of a positive denominator whose magnitude
/// here is single digits.
const POISON: f32 = -7.5;

/// What one arm answered: `o` in the packed layout both entries write
/// (`(row * n_q_heads + q_head) * D`), `lse` at `row * n_q_heads + q_head`.
/// Both are sized for the WHOLE tile, not for `n_rows`, so the tail is
/// readable.
struct Answered {
    o: Vec<f32>,
    lse: Vec<f32>,
}

/// **BOTH ARMS OVER ONE FIXTURE, ONE ARGUMENT LIST, ONE POOL.**
///
/// The nineteen seats are transcribed from the two sources and they match
/// seat for seat — `sdpa_paged.metal:590` and `sdpa_paged_mma.metal:308`.
/// Neither `_lse` entry declares `q_row_pitch` or `o_row_pitch` (the strided
/// entry at `sdpa_paged.metal:648` does, and is not this one), so both write
/// the packed rectangle and the read-back below is the same for both.
fn fire_both_arms(device: &Context, pipelines: &Pipelines) -> [Answered; 2] {
    let handles = Handles::new();

    let pool_elems = PAGES * PAGE_SIZE * KV_HEADS * D;
    let q_seen = spread(ROWS * Q_HEADS * D, 1);
    let k_seen = spread(pool_elems, 5);
    let v_seen = spread(pool_elems, 11);

    let mut queries = Buffer::zeroed(device, (ROWS * Q_HEADS * D * 2) as u64).expect("q reserves");
    queries.write(0, &as_bf16_bytes(&q_seen)).expect("q lands");
    let mut k_pages = Buffer::zeroed(device, (pool_elems * 2) as u64).expect("k reserves");
    k_pages.write(0, &as_bf16_bytes(&k_seen)).expect("k lands");
    let mut v_pages = Buffer::zeroed(device, (pool_elems * 2) as u64).expect("v reserves");
    v_pages.write(0, &as_bf16_bytes(&v_seen)).expect("v lands");

    let mut positions = Buffer::zeroed(device, (ROWS * 4) as u64).expect("positions reserve");
    positions.write(0, as_bytes(&POSITIONS)).expect("positions");
    let mut requests = Buffer::zeroed(device, (ROWS * 4) as u64).expect("requests reserve");
    requests.write(0, as_bytes(&REQUESTS)).expect("requests");
    let mut indices = Buffer::zeroed(device, (PAGES * 4) as u64).expect("indices reserve");
    indices.write(0, as_bytes(&INDICES)).expect("indices");
    let mut indptr = Buffer::zeroed(device, (INDPTR.len() * 4) as u64).expect("indptr reserves");
    indptr.write(0, as_bytes(&INDPTR)).expect("indptr");

    let mask = mask_bytes();
    let mut mask_buf = Buffer::zeroed(device, mask.len() as u64).expect("mask reserves");
    mask_buf.write(0, &mask).expect("mask lands");
    let mut enabled = Buffer::zeroed(device, ROWS as u64).expect("enabled reserves");
    enabled.write(0, &MASK_ON).expect("enabled lands");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a whole reservation binds");
    let t_q = Tensor::new(bind(&queries), ROWS as u32, (Q_HEADS * D) as u32, Dtype::Bf16);
    let t_k = Tensor::new(bind(&k_pages), PAGES as u32, 1, Dtype::Bf16);
    let t_v = Tensor::new(bind(&v_pages), PAGES as u32, 1, Dtype::Bf16);
    let t_pos = Tensor::new(bind(&positions), ROWS as u32, 1, Dtype::I32);
    let t_req = Tensor::new(bind(&requests), ROWS as u32, 1, Dtype::I32);
    let t_idx = Tensor::new(bind(&indices), PAGES as u32, 1, Dtype::U32);
    let t_ptr = Tensor::new(bind(&indptr), INDPTR.len() as u32, 1, Dtype::U32);
    let t_mask = Tensor::new(bind(&mask_buf), ROWS as u32, MASK_STRIDE, Dtype::U8);
    let t_enabled = Tensor::new(bind(&enabled), ROWS as u32, 1, Dtype::U8);

    // **THE TILE, NOT `n_rows`.** Sized for all thirty-two rows so claim 4
    // can read the twenty-eight neither arm should touch.
    let o_elems = TILE * Q_HEADS * D;
    let lse_elems = TILE * Q_HEADS;

    let arms: [(&'static str, &'static str, u32); 2] = [
        (
            "attn/sdpa_paged.metal",
            "sdpa_paged_tiled_lse_bfloat16_d_64",
            1024,
        ),
        (
            "attn/sdpa_paged_mma.metal",
            "sdpa_paged_mma_lse_bfloat16_d_64",
            128,
        ),
    ];

    arms.map(|(file, entry, threads)| {
        // Poisoned before every fire, and freshly for each arm: the tail
        // claim is about what the fire did NOT write, so a buffer that
        // arrived zeroed would prove nothing a `calloc` did not already.
        let mut out = Buffer::zeroed(device, (o_elems * 2) as u64).expect("out reserves");
        out.write(0, &as_bf16_bytes(&vec![POISON; o_elems]))
            .expect("the out poison lands");
        let mut lse = Buffer::zeroed(device, (lse_elems * 4) as u64).expect("lse reserves");
        lse.write(0, as_bytes(&vec![POISON; lse_elems]))
            .expect("the lse poison lands");

        let t_out = Tensor::new(bind(&out), TILE as u32, (Q_HEADS * D) as u32, Dtype::Bf16);
        let t_lse = Tensor::new(bind(&lse), TILE as u32, Q_HEADS as u32, Dtype::F32);

        let frame = device.frame().expect("a command buffer opens");
        {
            let sink = Sink::new(device, &frame, pipelines, &handles);
            let ctx: &Ctx<'_> = &sink;
            let grid = [
                (Q_HEADS as u32) * threads,
                (ROWS as u32).div_ceil(TILE as u32).max(1),
                1,
            ];
            ctx.fire(
                Fire::at(file, entry).apply(Grid::of(grid, [threads, 1, 1])),
                &[
                    t_q.arg(),                      // 0  queries
                    t_k.arg(),                      // 1  k_pages
                    t_v.arg(),                      // 2  v_pages
                    t_out.arg_mut(),                // 3  out
                    (GQA as i32).arg(),             // 4  gqa_factor
                    t_pos.arg(),                    // 5  position_ids
                    t_req.arg(),                    // 6  req_of_token
                    t_idx.arg(),                    // 7  kv_page_indices
                    t_ptr.arg(),                    // 8  kv_page_indptr
                    (PAGE_SIZE as i32).arg(),       // 9  page_size
                    (KV_HEADS as i32).arg(),        // 10 n_kv_heads
                    SCALE.arg(),                    // 11 scale
                    t_mask.arg(),                   // 12 attention_mask
                    MASK_STRIDE.arg(),              // 13 attention_mask_stride
                    t_enabled.arg(),                // 14 attention_mask_enabled
                    WINDOW.arg(),                   // 15 window
                    sink.absent().expect("the sink seat takes a nil buffer"), // 16 sinks
                    (ROWS as i32).arg(),            // 17 n_rows
                    t_lse.arg_mut(),                // 18 lse
                ],
            )
            .unwrap_or_else(|why| panic!("`{entry}` encodes: {why}"));
        }
        frame.commit().expect("the fire completes");

        let mut o_raw = vec![0u8; o_elems * 2];
        out.read(0, &mut o_raw).expect("out reads back");
        let mut lse_raw = vec![0u8; lse_elems * 4];
        lse.read(0, &mut lse_raw).expect("lse reads back");
        Answered {
            o: o_raw.chunks_exact(2).map(from_bf16).collect(),
            lse: lse_raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        }
    })
}

/// The row a fully-masked live row is, so every claim names the same one.
const EMPTY_ROW: usize = 3;

/// **THE BAND, AND WHY IT IS THIS WIDE.** The matrix arm stages K and V as
/// `half` in threadgroup memory and accumulates both the scores and the P·V
/// product in `simdgroup_matrix<half, 8, 8>` fragments, where the scalar arm
/// keeps `U = float` throughout. The owner's numerics ruling is that bit
/// invariance is not required and fast ladders are the default, so drift of
/// this class is expected; `2e-2` is the band the existing mma-versus-tiled
/// comparison used and it is reused here rather than re-derived.
const RTOL: f32 = 2e-2;

/// **AND WHY `lse`'s BAND IS ADDITIVE.** A base-2 log of a denominator is not
/// a relative quantity — halving the mass moves it by exactly 1 whatever its
/// magnitude — so a relative tolerance on it would be tight where the number
/// is small and meaningless where it is large. The number this catches is
/// `log2(4) = 2`, which is a hundred times the band either way.
const LSE_BAND: f32 = 2e-2;

#[test]
fn the_two_arms_agree_on_o_and_on_lse() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the two lse arms") else {
        return;
    };
    let pipelines = Pipelines::new();
    let [scalar, matrix] = fire_both_arms(&device, &pipelines);

    let mut worst_o = 0.0f32;
    let mut worst_lse = 0.0f32;
    for row in 0..ROWS {
        for head in 0..Q_HEADS {
            let at = row * Q_HEADS + head;
            if row != EMPTY_ROW {
                // The empty row's `lse` is `-inf` on both arms; its
                // difference is a NaN and its claim is
                // `a_live_row_that_kept_no_key_publishes_a_true_negative_infinity`.
                let d = (scalar.lse[at] - matrix.lse[at]).abs();
                worst_lse = worst_lse.max(d);
                assert!(
                    d <= LSE_BAND,
                    "row {row} head {head}: the scalar arm published lse {} and the matrix arm \
                     {}, {d} apart. A difference near 2 is a fold taken over the wrong lanes — \
                     the matrix arm reduced a quarter of the row's mass and called it the whole \
                     row. `[metal.tuning] sdpa_mma = false` is the way back.",
                    scalar.lse[at], matrix.lse[at]
                );
            }
            for d in 0..D {
                let i = at * D + d;
                let (a, b) = (scalar.o[i], matrix.o[i]);
                let rel = (a - b).abs() / a.abs().max(1.0 / 256.0);
                worst_o = worst_o.max(rel);
                assert!(
                    rel <= RTOL,
                    "row {row} head {head} lane {d}: the scalar arm answered {a} and the matrix \
                     arm {b} ({rel} relative). The matrix arm is allowed to drift — it stages \
                     K/V as half — but not this far; a wrong register layout is wrong answers \
                     and not slow ones."
                );
            }
        }
    }
    // **THE TOLERANCE OF THE TOLERANCE.** A band nothing approaches is a band
    // that would not have caught anything either; if both arms ever agreed
    // exactly, the two `Fire`s would be firing one kernel and the diff would
    // be a tautology.
    assert!(
        worst_o > 0.0,
        "the two arms agreed to the last bit on every one of {} lanes, which is not what a half \
         fragment and an f32 accumulator do — check that the two `Fire`s really name two entry \
         points",
        ROWS * Q_HEADS * D
    );
    println!(
        "scalar vs matrix over {ROWS} rows x {Q_HEADS} heads x {D}: worst o {worst_o:.5} relative \
         (band {RTOL}), worst lse {worst_lse:.5} additive (band {LSE_BAND})"
    );
}

#[test]
fn a_live_row_that_kept_no_key_publishes_a_true_negative_infinity() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the empty row") else {
        return;
    };
    let pipelines = Pipelines::new();
    let [scalar, matrix] = fire_both_arms(&device, &pipelines);

    for (name, arm) in [("scalar", &scalar), ("matrix", &matrix)] {
        for head in 0..Q_HEADS {
            let got = arm.lse[EMPTY_ROW * Q_HEADS + head];
            assert_eq!(
                got.to_bits(),
                f32::NEG_INFINITY.to_bits(),
                "the {name} arm published {got} for row {EMPTY_ROW} head {head}, which kept no \
                 key. It must be a true -inf, bit for bit: `merge_lse.metal` drops a side on \
                 `isfinite` and `attn_sink.metal` rescales by 1.0 on it, so the seed value \
                 NEG_INF (-3.0e38) leaking out as a published one is a wrong answer that no \
                 tolerance sees."
            );
        }
    }
    // And the row it is, is genuinely a row: a fixture whose mask happened to
    // clear every row would pass the loop above and prove nothing.
    for row in 0..ROWS {
        if row == EMPTY_ROW {
            continue;
        }
        for head in 0..Q_HEADS {
            let got = scalar.lse[row * Q_HEADS + head];
            assert!(
                got.is_finite(),
                "row {row} head {head} came back {got}; the fixture is supposed to leave every \
                 row but {EMPTY_ROW} with keys, so this test's subject is not the mask"
            );
        }
    }
    println!("row {EMPTY_ROW} is -inf on both arms, exactly; the other {} rows are finite", ROWS - 1);
}

#[test]
fn each_arms_lse_describes_that_arms_own_o() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the sink rescale") else {
        return;
    };
    let pipelines = Pipelines::new();
    let [scalar, matrix] = fire_both_arms(&device, &pipelines);

    // **THE RESCALE, WHICH IS A FUNCTION OF THE PAIR AND OF NOTHING ELSE.**
    // `attn_sink.metal` folds one more logit `s` into a row that already has
    // `o = O/Z` and `lse = (max*log2e + log2 Z)`. Writing `L = lse * ln2 =
    // ln Z + max`, the unnormalized numerator is `o * e^L` and the new
    // denominator is `e^L + e^s`, so the rescaled row is
    // `o * e^L / (e^L + e^s)` — `o` and `lse` together, which is exactly why
    // this is the self-consistency gate and a diff of `o` alone is not. An
    // `lse` off by 2 in base 2 is a denominator off by 4 and moves the weight
    // from 1/2 to 1/5.
    let rescale = |o: f32, lse: f32, s: f32| -> f32 {
        let l = lse * std::f32::consts::LN_2;
        if !l.is_finite() {
            // The empty row: the sink is the whole mass, and both consumers
            // agree the answer is the sink's own.
            return 0.0;
        }
        o * (1.0 / (1.0 + (s - l).exp()))
    };

    let mut worst = 0.0f32;
    for row in 0..ROWS {
        for head in 0..Q_HEADS {
            let at = row * Q_HEADS + head;
            // The sink logit is put AT the scalar arm's own `L`, which is
            // where the rescale is most sensitive to a wrong `lse`: the
            // weight sits at 1/2 and its derivative is largest.
            let s = scalar.lse[at] * std::f32::consts::LN_2;
            let s = if s.is_finite() { s } else { 0.0 };
            for d in 0..D {
                let i = at * D + d;
                let a = rescale(scalar.o[i], scalar.lse[at], s);
                let b = rescale(matrix.o[i], matrix.lse[at], s);
                let rel = (a - b).abs() / a.abs().max(1.0 / 256.0);
                worst = worst.max(rel);
                assert!(
                    rel <= RTOL,
                    "row {row} head {head} lane {d}: rescaled through each arm's OWN (o, lse) \
                     the two answered {a} and {b} ({rel} relative). The arms are allowed to \
                     drift from each other; what they may not do is publish an lse that does \
                     not describe the o beside it. A plane closer to the scalar arm than to its \
                     own output is the failure this is shaped to catch."
                );
            }
        }
    }
    println!("sink-rescaled through each arm's own pair: worst {worst:.5} relative (band {RTOL})");
}

#[test]
fn the_rows_past_n_rows_are_left_exactly_as_they_were_poisoned() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the tail") else {
        return;
    };
    let pipelines = Pipelines::new();
    let [scalar, matrix] = fire_both_arms(&device, &pipelines);

    let poison_bf16 = from_bf16(&bf16(POISON));
    for (name, arm) in [("scalar", &scalar), ("matrix", &matrix)] {
        for row in ROWS..TILE {
            for head in 0..Q_HEADS {
                let at = row * Q_HEADS + head;
                assert_eq!(
                    arm.lse[at].to_bits(),
                    POISON.to_bits(),
                    "the {name} arm wrote {} into lse row {row} head {head}, which is past \
                     n_rows = {ROWS}. Both epilogues return before the lse write for !live \
                     rows; the matrix arm addresses lse by `my_row`, a fragment-row offset and \
                     not a loop counter, so an off-by-a-fragment lands in the NEXT tile's rows \
                     rather than out of bounds where a bounds check would see it.",
                    arm.lse[at]
                );
                for d in 0..D {
                    assert_eq!(
                        arm.o[at * D + d].to_bits(),
                        poison_bf16.to_bits(),
                        "the {name} arm wrote into o row {row} head {head} lane {d}, past \
                         n_rows = {ROWS}"
                    );
                }
            }
        }
    }
    println!(
        "{} tail rows of the {TILE}-row tile are untouched on both arms, in o and in lse",
        TILE - ROWS
    );
}
