//! The Session I tower kernels, on a real Apple GPU: four gates that ask
//! whether the shipped `.metal` sources compute the arithmetic their CUDA
//! twins compute, and not merely something plausible.
//!
//! **WHAT THIS IS FOR.** `device_floor` asks whether a kernel LAUNCHES —
//! whether the source compiles, whether a `Fire`'s positional argument list
//! lands where the shader's `[[buffer(n)]]` declarations say, whether a
//! `Tensor`'s `u32` resolves to the bytes the shell carved. It answers that
//! with `layout.embed`, whose answer is a permutation: a mis-indexed argument
//! shows up as the wrong rows. None of the tower rows are permutations. Every
//! one of them is an arithmetic claim that a wrong kernel satisfies to three
//! digits and fails in the fourth, and there is exactly one machine where
//! that can be measured.
//!
//! The four gates are `.wiki/alto/metal-verify-queue.md`'s Session I
//! settlement, items (i) through (l), and each is stated against a CPU-side
//! f32 reference derived from the CUDA reference semantics rather than
//! against the Metal kernel's own spelling:
//!
//! ```text
//! (i)   elemwise/norm_standardize.metal  — the planes are the COLUMN's, and
//!       the difference is taken in f32 and rounded ONCE at the store
//! (j)   elemwise/norm_layernorm.metal    — the moments are TWO reductions,
//!       and a row whose mean dwarfs its spread is where that shows
//! (k)   layout/fold.metal                — the pool averages in f32; the
//!       merge is the identity copy, byte for byte
//! (l)   elemwise/rope_mrope.metal        — the blocked ladder restarts at
//!       `within = 0` over `Σsections`; the interleaved one does neither
//! ```
//!
//! **THE FIRES GO THROUGH THE SHIPPED RUST ENCODERS**, not through
//! hand-bound buffers, exactly as `device_floor`'s one dispatch goes through
//! `kernels_metal::layout::embed`. That is deliberate: half of what these
//! gates can catch is a host-side launch shape — a grid sized off the wrong
//! extent, a plane validated against the wrong axis, an argument list in the
//! order the shader does not declare — and a test that bound the buffers
//! itself would have checked the shader against the test's own idea of the
//! signature rather than against the shell's.
//!
//! # Gating
//!
//! As `device_floor`: `cfg`'d to Apple at compile time, and SKIPS at run time
//! when `device::present()` says no, saying so.
//!
//! ```text
//! cargo test -p engine-metal --release --test tower_kernels_on_device -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, for `device_floor`'s reason: `cargo test` runs a
/// file's tests on several threads, each of these binds a device and reserves
/// buffers, and two of them compiling shaders at once is a way to meet the
/// Metal compiler's own concurrency and learn nothing.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The device, or a printed skip and `None`.
fn device_or_skip(what: &str) -> Option<Context> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    Some(Context::bind().expect("the device binds"))
}

// ---------------------------------------------------------------------------
// (i) `standardize`: the planes are the column's, and it rounds once.
// ---------------------------------------------------------------------------

/// gemma's wide tower's own hidden, so the launch under test is one that
/// ships (`vision_tower.std_{bias,scale}` are `[1152]`).
const STD_WIDTH: u32 = 1152;

/// **THREE ROWS, WHICH IS THE FEWEST THAT CAN TELL THE TWO INDEXINGS APART.**
/// Two rows would leave a row-indexed kernel reading `bias[0]` and `bias[1]` —
/// a rectangle that still varies down the column and could be mistaken for
/// the right one under a loose eye. Three rows over IDENTICAL inputs must
/// answer three identical rows, and a kernel reading `bias[tid.y]` answers
/// three different ones.
const STD_ROWS: u32 = 3;

/// (i), first claim: **the planes are indexed by COLUMN and not by row.**
///
/// Every column here holds a different `(bias, scale)` pair and every row
/// holds the same numbers, so the only rectangle a correct kernel can answer
/// is one row repeated. `standardize.metal` reads `bias[tid.x]`/`scale[tid.x]`
/// where `tid.x` is the column and `tid.y` the row — `add_bias`' own grid one
/// file over — and a kernel that read `tid.y` instead would answer three
/// different rows off the first three entries of a `[1152]` plane, which is
/// the whole of the mistake this fixture is shaped to make visible.
///
/// The elementwise reference is the second claim of the CUDA twin's (a)/(b)
/// pair (`crates/kernels-cuda/tests/tower_standardize.rs`): every element is
/// `(x − bias[c]) · scale[c]` in f32, and the op is IN PLACE, so the readback
/// is of the input's own bytes.
#[test]
fn the_standardization_reads_its_two_planes_by_column_so_three_identical_rows_answer_identically() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the standardization's axis") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    // Multiples of 1/8, 1/32 and 1/4 — every one exactly a bf16, so what the
    // device reads is what the reference computes with, and the only rounding
    // anywhere is the kernel's single store.
    let width = STD_WIDTH as usize;
    let column: Vec<f32> = (0..width).map(|c| (c % 53) as f32 / 4.0 - 6.0).collect();
    let bias: Vec<f32> = (0..width).map(|c| ((c % 97) as f32 - 48.0) / 8.0).collect();
    let scale: Vec<f32> = (0..width).map(|c| 0.5 + (c % 61) as f32 / 32.0).collect();

    let x: Vec<f32> = (0..STD_ROWS).flat_map(|_| column.iter().copied()).collect();

    let x_store = staged(&device, &encode(&x));
    let bias_store = staged(&device, &encode(&bias));
    let scale_store = staged(&device, &encode(&scale));

    let x_h = bind_whole(&handles, &x_store, "the rectangle");
    let bias_h = bind_whole(&handles, &bias_store, "the bias plane");
    let scale_h = bind_whole(&handles, &scale_store, "the scale plane");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::elemwise::norm::standardize(
            &sink,
            Tensor::new(bias_h, 1, STD_WIDTH, Dtype::Bf16),
            Tensor::new(scale_h, 1, STD_WIDTH, Dtype::Bf16),
            Tensor::new(x_h, STD_ROWS, STD_WIDTH, Dtype::Bf16),
        )
        .expect("the standardization encodes");
    }
    frame.commit().expect("the fire completes");

    // In place: the answer is read back from the INPUT's address, and reading
    // anywhere else would not notice if it were not.
    let got = decode(&read_back(&x_store, x.len() * 2));

    for r in 1..STD_ROWS as usize {
        let first = &got[..width];
        let row = &got[r * width..(r + 1) * width];
        assert_eq!(
            first, row,
            "row {r} answered differently from row 0 over identical inputs — the two \
             planes are the COLUMN's, and a row answering differently means the kernel \
             read `bias[tid.y]`, i.e. applied column {r}'s pair to the whole of row {r}"
        );
    }

    for r in 0..STD_ROWS as usize {
        for c in 0..width {
            let at = r * width + c;
            let want = (column[c] - bias[c]) * scale[c];
            assert!(
                near(got[at], want, 1.0),
                "row {r} column {c} landed {} where the f32 arithmetic says {want} — \
                 `y = (x − bias[c]) · scale[c]` is one multiply and one subtract, and a \
                 miss here is either the wrong plane or the wrong axis",
                got[at]
            );
        }
    }
}

/// (i), second claim: **the difference is taken in f32 and rounded ONCE, at
/// the store**, which is the CUDA twin's claim (c) and the reason
/// `elementwise.standardize` is a kernel rather than `add_bias` with a negated
/// plane followed by a per-column multiply.
///
/// Two column bands, because the queue's own fixture and a fixture with teeth
/// are not the same fixture:
///
/// * the queue's — `x = 1024`, `bias` ONE bf16 quantum below it (the quantum
///   at `|x| = 1024` is 8, so `bias = 1016`), `scale = 64`. The exact f32
///   answer is `512.0` and the assertion is bit-exact equality with it;
/// * and one where a plausible wrong spelling answers a DIFFERENT number.
///   `x = 1000`, `bias = 992`, `scale = 255/128`: the fused answer is
///   `8 · 1.9921875 = 15.9375`, exactly a bf16, while a kernel that
///   distributed the multiply — `x·scale − bias·scale`, algebraically the
///   same line — rounds `1992.1875` to `1992` and `1976.25` to `1976` and
///   answers `16.0`. The `assert_ne!` says so in as many words.
///
/// **AND A NOTE THE QUEUE'S OWN WORDING DOES NOT CARRY.** A spelling that
/// merely stored `x − bias` to bf16 before scaling cannot be caught by a
/// cancelling row at all: by Sterbenz's lemma the difference of two bf16
/// numbers within a factor of two of each other is itself exactly a bf16, so
/// the composed spelling is EXACT there and answers the same `512.0`. The
/// CUDA twin's own comment concedes as much ("storing the difference to bf16
/// before scaling it is exact HERE by construction"). What the cancelling
/// band pins is therefore the VALUE and not the spelling; the second band is
/// what pins a spelling.
#[test]
fn a_cancelling_column_lands_the_exact_f32_answer_and_not_a_distributed_spellings() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the standardization's rounding") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    const WIDTH: u32 = 64;
    const CANCELLING: usize = 32;

    let width = WIDTH as usize;
    let mut x = Vec::with_capacity(width);
    let mut bias = Vec::with_capacity(width);
    let mut scale = Vec::with_capacity(width);
    for c in 0..width {
        if c < CANCELLING {
            x.push(1024.0);
            bias.push(1016.0);
            scale.push(64.0);
        } else {
            x.push(1000.0);
            bias.push(992.0);
            scale.push(255.0 / 128.0);
        }
    }

    let x_store = staged(&device, &encode(&x));
    let bias_store = staged(&device, &encode(&bias));
    let scale_store = staged(&device, &encode(&scale));

    let x_h = bind_whole(&handles, &x_store, "the rectangle");
    let bias_h = bind_whole(&handles, &bias_store, "the bias plane");
    let scale_h = bind_whole(&handles, &scale_store, "the scale plane");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::elemwise::norm::standardize(
            &sink,
            Tensor::new(bias_h, 1, WIDTH, Dtype::Bf16),
            Tensor::new(scale_h, 1, WIDTH, Dtype::Bf16),
            Tensor::new(x_h, 1, WIDTH, Dtype::Bf16),
        )
        .expect("the standardization encodes");
    }
    frame.commit().expect("the fire completes");

    let got = decode(&read_back(&x_store, width * 2));

    for (c, landed) in got.iter().copied().enumerate() {
        let want = (x[c] - bias[c]) * scale[c];
        assert_eq!(
            landed.to_bits(),
            want.to_bits(),
            "column {c} landed {landed} where the f32 arithmetic says exactly {want} — \
             both operands and the answer are exactly bf16, so anything but bit equality \
             means the difference or the product was taken somewhere other than f32"
        );
    }
    for (c, landed) in got.iter().copied().enumerate().skip(CANCELLING) {
        assert_ne!(
            landed, 16.0,
            "column {c} answered 16.0, which is what `x·scale − bias·scale` answers when \
             both products round to bf16 first — the line is `(x − bias) · scale` and it \
             is not distributive under rounding"
        );
    }
    println!(
        "standardize: cancelling band {} | distributive band {}",
        got[0], got[CANCELLING]
    );
}

// ---------------------------------------------------------------------------
// (j) `layernorm`: two reductions, and the row where one would not do.
// ---------------------------------------------------------------------------

/// qwen35's tower row is 768 and qwen36's 1152; the wider one, because the
/// cancellation this gate is about gets worse with the number of terms the
/// one-pass form sums.
const LN_WIDTH: u32 = 1152;

const LN_ROWS: u32 = 2;

/// LayerNorm's own epsilon, INSIDE the root beside the variance, which is
/// where `norm_layernorm.metal` and `layernorm.cuh` both put it.
const LN_EPS: f32 = 1.0e-5;

/// (j): **THE MOMENTS ARE TWO REDUCTIONS AND NOT `E[x²] − E[x]²`.**
///
/// The one-pass form halves the barriers and is subtly wrong in exactly one
/// regime: a row whose mean is large against its spread. `E[x²]` and `E[x]²`
/// are then two nearly equal large numbers whose difference is the whole
/// answer, and f32 keeps 24 bits of it — so the variance comes back with the
/// leading bits of the mean's square where its own bits should be, and the
/// failure reads as a slightly wrong norm and never as a NaN.
///
/// **THE FIXTURE IS AS TIGHT AS bf16 PERMITS, AND THAT IS WHY IT IS NOT
/// LITERALLY THE QUEUE'S "mean 1000, spread 1".** bf16 carries eight
/// significant bits, so the quantum at `|x| = 1024` is 8 and a spread of 1 is
/// not a number this element can hold — the tightest spread two distinct bf16
/// values near 1024 can have is one quantum, and that is what this row uses:
/// 1024 everywhere, ±8 on a quarter of the columns. Mean 1024 exactly (the
/// deltas cancel and every partial sum is a multiple of 8, so the mean is
/// exact in f32 whatever order it is summed in), variance 16 exactly, spread
/// 4 against a mean of 1024. `mean² / var` is 65536, which is 16 of f32's 24
/// bits, and the one-pass form comes back with 14.25 where the truth is 16 —
/// an 11% error in the variance, 5% in its reciprocal root, and about fifteen
/// bf16 quanta in the answer.
///
/// The claim is stated the way B5 states it and not as a tolerance: the fused
/// row lands NEARER the f32 two-pass reference than the one-pass value does.
/// That direction is the whole of it — a saving may move a number only toward
/// the truth.
#[test]
fn the_layernorm_takes_two_reductions_so_a_row_whose_mean_dwarfs_its_spread_survives() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the centred norm's moments") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let width = LN_WIDTH as usize;
    let rows = LN_ROWS as usize;

    // Row `r` puts its two deltas at a different phase of the period, so the
    // second row is not a copy of the first and the reduction is exercised
    // over two different arrangements of the same moments.
    let mut x = vec![0.0f32; rows * width];
    for r in 0..rows {
        for c in 0..width {
            let phase = c % 8;
            let delta = if phase == 2 * r {
                8.0
            } else if phase == 2 * r + 1 {
                -8.0
            } else {
                0.0
            };
            x[r * width + c] = 1024.0 + delta;
        }
    }
    // A trained `nn.LayerNorm` ships a weight near one and a bias near zero;
    // both planes are exactly bf16 so the epilogue's `fma` is the only place
    // the answer can round.
    let w: Vec<f32> = (0..width).map(|c| 1.0 + (c % 4) as f32 * 0.25).collect();
    let b: Vec<f32> = (0..width).map(|c| (c % 2) as f32 * 0.5).collect();

    let x_store = staged(&device, &encode(&x));
    let w_store = staged(&device, &encode(&w));
    let b_store = staged(&device, &encode(&b));
    let y_store = Buffer::zeroed(&device, (x.len() * 2) as u64).expect("the destination reserves");

    let x_h = bind_whole(&handles, &x_store, "the rectangle");
    let w_h = bind_whole(&handles, &w_store, "the weight plane");
    let b_h = bind_whole(&handles, &b_store, "the bias plane");
    let y_h = handles
        .bind(&y_store, 0, (x.len() * 2) as u64)
        .expect("the destination binds");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::elemwise::norm::layernorm(
            &sink,
            Tensor::new(x_h, LN_ROWS, LN_WIDTH, Dtype::Bf16),
            Tensor::new(w_h, 1, LN_WIDTH, Dtype::Bf16),
            Tensor::new(b_h, 1, LN_WIDTH, Dtype::Bf16),
            LN_EPS,
            Tensor::new(y_h, LN_ROWS, LN_WIDTH, Dtype::Bf16),
        )
        .expect("the centred norm encodes");
    }
    frame.commit().expect("the fire completes");

    let got = decode(&read_back(&y_store, x.len() * 2));

    let (mut worst_fused, mut worst_naive) = (0.0f32, 0.0f32);
    let (mut total_fused, mut total_naive) = (0.0f64, 0.0f64);
    let mut gap = 0.0f32;
    for r in 0..rows {
        let row = &x[r * width..(r + 1) * width];
        let (mean_two, var_two, inv_two) = two_pass(row, LN_EPS);
        let (mean_one, var_one, inv_one) = one_pass(row, LN_EPS);
        println!(
            "layernorm row {r}: two-pass mean {mean_two} var {var_two} | one-pass mean \
             {mean_one} var {var_one}"
        );
        // **THE FIXTURE MUST STILL CANCEL.** If the two forms ever agree, the
        // gate below is true for a reason that has nothing to do with the
        // kernel, and this is the assertion that says so out loud.
        assert!(
            (var_one - var_two).abs() > 0.05 * var_two,
            "row {r}: the one-pass variance is {var_one} against the two-pass {var_two}, \
             within 5% — this fixture has stopped cancelling and the gate below would pass \
             vacuously; pick a row whose mean dwarfs its spread harder"
        );
        for c in 0..width {
            let at = r * width + c;
            let want = (row[c] - mean_two) * inv_two * w[c] + b[c];
            let naive = (row[c] - mean_one) * inv_one * w[c] + b[c];
            let (df, dn) = ((got[at] - want).abs(), (naive - want).abs());
            worst_fused = worst_fused.max(df / quantum(want.abs().max(1.0)));
            worst_naive = worst_naive.max(dn / quantum(want.abs().max(1.0)));
            total_fused += f64::from(df);
            total_naive += f64::from(dn);
            gap = gap.max(dn);
        }
    }
    println!(
        "layernorm: fused {worst_fused:.3} q worst / {total_fused:.4} total; one-pass \
         {worst_naive:.3} q worst / {total_naive:.4} total (widest one-pass miss {gap:.5})"
    );

    assert!(
        worst_fused < worst_naive,
        "the fused row sits {worst_fused} bf16 quanta from the f32 two-pass reference and \
         the one-pass value sits {worst_naive} — if the kernel were nearer the one-pass \
         number than the reference, its moments are `E[x²] − E[x]²` and B5's whole \
         argument is inverted"
    );
    assert!(
        total_fused < total_naive,
        "total error {total_fused} fused against {total_naive} one-pass — the two \
         reductions are what buy that, and a kernel that folded them into one would not"
    );
    assert!(
        worst_fused <= 1.0,
        "the fused norm sits {worst_fused} quanta from the f32 LayerNorm, and a kernel \
         that keeps the centred row in f32 to a single rounding at the store may not sit \
         further than one"
    );
}

// ---------------------------------------------------------------------------
// (k) `pool_rows` against `merge_rows`, on the same rectangle.
// ---------------------------------------------------------------------------

/// Divisible by both folds' blocks — 9 for gemma's `3 × 3` soft-token pool
/// and 4 for qwen's `2 × 2` spatial merge — so the two ops answer over
/// literally the same input bytes and the gate is one fixture and not two.
const FOLD_ROWS: u32 = 36;

const FOLD_WIDTH: u32 = 8;

/// (k): **THE POOL AVERAGES IN f32 AND THE MERGE IS THE IDENTITY COPY.**
///
/// Both fire over the same rectangle in the same frame, because the two ops
/// share `layout/fold.metal` and the thing worth knowing is that they do not
/// share an answer.
///
/// **THE POOL'S FIXTURE IS BUILT SO NINE bf16 ADDITIONS CANNOT REACH THE
/// ANSWER.** Each block of nine rows holds one row of `256·f` and eight rows
/// of `1·f`. In f32 that sums to `264·f` and averages to `29.3333·f`, which
/// stores as `29.375·f`. A running sum in the ELEMENT never leaves 256: `256
/// + 1` is exactly halfway between the two bf16 neighbours 256 and 258, so
/// round-to-even answers 256, eight times over, and the average comes back
/// `28.5·f`. Both numbers are far outside each other's quantum, so the
/// assertion can be bit-exact in both directions — the f32 answer asserted,
/// the nine-times-rounded answer denied by name.
///
/// `f` is a power of two per column and per output row, so every input, every
/// partial sum and both candidate answers are exactly bf16 and nothing in
/// this fixture rounds except the kernel's own store.
///
/// **THE MERGE'S CLAIM IS THAT THERE IS NO ARITHMETIC.** Row-major
/// `[rows, width]` and row-major `[rows/4, 4·width]` put the same element at
/// the same offset, so the whole answer is the input's bytes in the input's
/// order. It is asserted byte for byte rather than value for value on
/// purpose: a merge that permuted rows within a block, or transposed the
/// concatenation, would still answer a rectangle of plausible tower
/// activations, and only the ORDER says which.
#[test]
fn the_pool_averages_in_f32_where_the_merge_of_the_same_rectangle_is_the_identity_copy() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the two folds") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    const POOL_SIDE: u32 = 3;
    const MERGE_SIDE: u32 = 2;
    let pool_block = (POOL_SIDE * POOL_SIDE) as usize;
    let merge_block = (MERGE_SIDE * MERGE_SIDE) as usize;
    let rows = FOLD_ROWS as usize;
    let width = FOLD_WIDTH as usize;
    let pooled_rows = rows / pool_block;
    let merged_rows = rows / merge_block;
    let merged_width = merge_block * width;

    // `f = 2^(c % 4) · 2^(r / 9)`: a power of two per column and per pooled
    // block, so the whole fixture is exact and the pool's two candidate
    // answers scale with it.
    let factor = |r: usize, c: usize| -> f32 {
        (1u32 << (c % 4)) as f32 * (1u32 << (r / pool_block)) as f32
    };
    let mut x = vec![0.0f32; rows * width];
    for r in 0..rows {
        for c in 0..width {
            let head = if r % pool_block == 0 { 256.0 } else { 1.0 };
            x[r * width + c] = head * factor(r, c);
        }
    }
    let x_bytes = encode(&x);

    let x_store = staged(&device, &x_bytes);
    let pooled_bytes = (pooled_rows * width * 2) as u64;
    let merged_bytes = (merged_rows * merged_width * 2) as u64;
    let pooled_store = Buffer::zeroed(&device, pooled_bytes).expect("the pool's answer reserves");
    let merged_store = Buffer::zeroed(&device, merged_bytes).expect("the merge's answer reserves");

    let x_h = bind_whole(&handles, &x_store, "the rectangle");
    let pooled_h = handles
        .bind(&pooled_store, 0, pooled_bytes)
        .expect("the pool's answer binds");
    let merged_h = handles
        .bind(&merged_store, 0, merged_bytes)
        .expect("the merge's answer binds");

    let source = Tensor::new(x_h, FOLD_ROWS, FOLD_WIDTH, Dtype::Bf16);
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::layout::pool_rows(
            &sink,
            source,
            POOL_SIDE,
            Tensor::new(pooled_h, pooled_rows as u32, FOLD_WIDTH, Dtype::Bf16),
        )
        .expect("the pool encodes");
        kernels_metal::layout::merge_rows(
            &sink,
            source,
            MERGE_SIDE,
            Tensor::new(
                merged_h,
                merged_rows as u32,
                merged_width as u32,
                Dtype::Bf16,
            ),
        )
        .expect("the merge encodes");
    }
    frame.commit().expect("both fires complete");

    let pooled = decode(&read_back(&pooled_store, pooled_rows * width * 2));
    for j in 0..pooled_rows {
        for c in 0..width {
            let f = factor(j * pool_block, c);
            // The f32 accumulation: one 256 and eight 1s, divided by the
            // BLOCK and not by the count of live rows.
            let want = f32_round(264.0 * f / 9.0);
            // What nine roundings in the element would have answered.
            let rounded_nine = f32_round(256.0 * f / 9.0);
            let landed = pooled[j * width + c];
            assert_eq!(
                landed.to_bits(),
                want.to_bits(),
                "pooled row {j} column {c} landed {landed} where the f32 mean of the nine \
                 source rows is {want} — the accumulator is f32 whatever the element is, \
                 and the divisor is the block"
            );
            assert_ne!(
                landed.to_bits(),
                rounded_nine.to_bits(),
                "pooled row {j} column {c} landed {rounded_nine}, which is what a running \
                 sum in bf16 answers when `256 + 1` ties to even nine times over — the \
                 pool would look plausible and be wrong by 3%"
            );
        }
    }

    let merged = read_back(&merged_store, merged_rows * merged_width * 2);
    assert_eq!(
        merged, x_bytes,
        "the merge answered different bytes from the ones it read — `[rows, width]` and \
         `[rows/4, 4·width]` are the same row-major bytes in the same order, so a merge \
         that permuted rows inside a block or concatenated them the other way round \
         would still answer a plausible rectangle and this is the only claim that sees it"
    );
    println!(
        "fold: pooled {pooled_rows} x {width} from {rows}; merged {merged_rows} x \
         {merged_width}, byte-identical"
    );
}

// ---------------------------------------------------------------------------
// (l) `rope_mrope_blocked` against `rope_mrope_interleaved` at `[0, k, k]`.
// ---------------------------------------------------------------------------

const ROPE_HEAD_DIM: u32 = 64;

const ROPE_Q_HEADS: u32 = 2;

const ROPE_KV_HEADS: u32 = 1;

const ROPE_ROWS: u32 = 2;

const ROPE_THETA: f32 = 10_000.0;

/// **BOTH TOWERS' SECTIONS**: `[0, head_dim/4, head_dim/4]`, which turns by
/// `(h, w)` and reads no `t` at all. At a 64-wide head that is `[0, 16, 16]`,
/// which tiles exactly the 32 frequency pairs the head holds — so the two
/// arms' GRIDS coincide (`kernels-metal`'s own host gate,
/// `the_two_forms_differ_in_the_point_and_not_the_geometry`, pins that) and
/// nothing but the arithmetic can separate them here.
const ROPE_SECTIONS: [u32; 3] = [0, ROPE_HEAD_DIM / 4, ROPE_HEAD_DIM / 4];

/// (l): **THE TWO ROTATIONS ARE THE PAIR THAT ANSWER PLAUSIBLE NUMBERS FOR
/// EACH OTHER'S CHECKPOINT**, fired over the same bytes at the same sections
/// with the same grid, so the only thing left to be different is the
/// arithmetic. Three things separate them and all three are asserted:
///
/// * **WHICH PAIR TAKES WHICH AXIS.** Blocked hands out contiguous blocks —
///   `[0, s0)` by `t`, `[s0, s0+s1)` by `h`, the rest by `w`. Interleaved
///   hands them out `t, h, w, t, h, w, …`. The row's position triple is
///   `(0, 1, 2)`, so under the INTERLEAVED arm every third pair turns by
///   `t = 0` and is left bit-for-bit alone, and under the BLOCKED arm
///   `s0 == 0` means there is no `t` block at all and pair 0 turns by `h`.
///   That is the sharpest statement of the difference this fixture can make:
///   one arm's `t` block is empty and the other's is a third of the head.
/// * **THE DENOMINATOR OF THE LADDER.** Blocked divides by `Σsections` (32
///   here), interleaved by `head_dim` (64) — so the same pair index is a
///   different frequency under the two arms even where they agree about the
///   axis.
/// * **AND EACH BLOCKED SECTION RESTARTS AT `within = 0`.** Pair 0 and pair
///   16 both get `inv_freq = 1`, so the `h` block starts at angle `pos_h` and
///   the `w` block at angle `pos_w` — the part nobody would guess, and the
///   part a kernel that kept the head's global pair index still looks smooth
///   and sensible under.
#[test]
fn the_blocked_rotation_restarts_each_sections_ladder_where_the_interleaved_one_skips_every_third_pair()
 {
    let _serial = serialized();
    let Some(device) = device_or_skip("the two multimodal rotations") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();

    let rows = ROPE_ROWS as usize;
    let head_dim = ROPE_HEAD_DIM as usize;
    let q_width = ROPE_Q_HEADS as usize * head_dim;
    let k_width = ROPE_KV_HEADS as usize * head_dim;

    // Multiples of 1/8 in [−1, 1]: exactly bf16, and bounded, so one absolute
    // tolerance covers every element of the answer.
    let fill = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i + salt) % 17) as f32 / 8.0 - 1.0)
            .collect()
    };
    let q0 = fill(rows * q_width, 0);
    let k0 = fill(rows * k_width, 5);

    // `(t, h, w)`: the `t` column is zero on every row, which is what makes
    // the interleaved arm's every-third-pair identity observable and what the
    // towers themselves state (`s0 == 0`).
    let positions: Vec<i32> = vec![0, 1, 2, 0, 3, 5];

    let q_blocked = staged(&device, &encode(&q0));
    let k_blocked = staged(&device, &encode(&k0));
    let q_inter = staged(&device, &encode(&q0));
    let k_inter = staged(&device, &encode(&k0));
    let pos_store = staged(&device, as_bytes_i32(&positions));

    let qb_h = bind_whole(&handles, &q_blocked, "the blocked query");
    let kb_h = bind_whole(&handles, &k_blocked, "the blocked key");
    let qi_h = bind_whole(&handles, &q_inter, "the interleaved query");
    let ki_h = bind_whole(&handles, &k_inter, "the interleaved key");
    let pos_h = bind_whole(&handles, &pos_store, "the position triples");

    let triples = Tensor::new(pos_h, ROPE_ROWS, 3, Dtype::I32);
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::elemwise::rope_mrope::blocked(
            &sink,
            Tensor::new(qb_h, ROPE_ROWS, q_width as u32, Dtype::Bf16),
            Tensor::new(kb_h, ROPE_ROWS, k_width as u32, Dtype::Bf16),
            triples,
            ROPE_SECTIONS,
            ROPE_HEAD_DIM,
            ROPE_HEAD_DIM,
            ROPE_THETA,
        )
        .expect("the tower's rotation encodes");
        kernels_metal::elemwise::rope_mrope::interleaved(
            &sink,
            Tensor::new(qi_h, ROPE_ROWS, q_width as u32, Dtype::Bf16),
            Tensor::new(ki_h, ROPE_ROWS, k_width as u32, Dtype::Bf16),
            triples,
            ROPE_SECTIONS,
            ROPE_HEAD_DIM,
            ROPE_HEAD_DIM,
            ROPE_THETA,
        )
        .expect("the trunk's rotation encodes");
    }
    frame.commit().expect("both fires complete");

    let got_qb = decode(&read_back(&q_blocked, q0.len() * 2));
    let got_kb = decode(&read_back(&k_blocked, k0.len() * 2));
    let got_qi = decode(&read_back(&q_inter, q0.len() * 2));
    let got_ki = decode(&read_back(&k_inter, k0.len() * 2));

    let want_qb = rotate_reference(&q0, ROPE_Q_HEADS, &positions, true);
    let want_kb = rotate_reference(&k0, ROPE_KV_HEADS, &positions, true);
    let want_qi = rotate_reference(&q0, ROPE_Q_HEADS, &positions, false);
    let want_ki = rotate_reference(&k0, ROPE_KV_HEADS, &positions, false);

    // The trig is `fast::cos`/`fast::sin` and the store is bf16, so the band
    // is two bf16 quanta at the inputs' own magnitude — `|x| ≤ 1` in, `|y| ≤
    // √2` out, whose quantum is 1/128.
    const BAND: f32 = 0.03;
    for (what, got, want) in [
        ("blocked q", &got_qb, &want_qb),
        ("blocked k", &got_kb, &want_kb),
        ("interleaved q", &got_qi, &want_qi),
        ("interleaved k", &got_ki, &want_ki),
    ] {
        for (at, expected) in want.iter().copied().enumerate() {
            assert!(
                (got[at] - expected).abs() <= BAND,
                "{what} element {at} landed {} where the f32 rotation says {expected} — \
                 the two arms share a pairing and a grid and differ only in which axis \
                 each pair turns by and at what frequency, so a miss here names the \
                 arithmetic and nothing else",
                got[at]
            );
        }
    }

    // **THE INTERLEAVED ARM'S `t` PAIRS ARE UNTOUCHED, BIT FOR BIT.** Pair
    // `i` with `i % 3 == 0` turns by `pos_t`, and every row's `t` is zero, so
    // `cos = 1` and `sin = 0` and both halves of the pair come back the bytes
    // they went in as. This is the assertion that says the interleaved arm is
    // interleaved.
    let half = head_dim / 2;
    for m in 0..rows {
        for h in 0..ROPE_Q_HEADS as usize {
            for i in (0..half).step_by(3) {
                let lo = (m * ROPE_Q_HEADS as usize + h) * head_dim + i;
                let hi = lo + half;
                assert_eq!(
                    (got_qi[lo].to_bits(), got_qi[hi].to_bits()),
                    (q0[lo].to_bits(), q0[hi].to_bits()),
                    "the interleaved arm moved pair {i} of head {h}, row {m}, whose axis is \
                     `t` and whose `t` is zero — an interleaved split hands every third \
                     pair to the time axis and this row has no time"
                );
            }
        }
    }

    // **AND THE BLOCKED ARM HAS NO `t` BLOCK AT ALL**, because `s0 == 0`.
    // Pair 0 is the first pair of the `h` block, so it turns — and it turns at
    // `inv_freq = 1`, i.e. by exactly `pos_h` radians, which is the restart
    // this whole entry exists to pin. Pair `s0 + s1 == 16` is the first pair
    // of the `w` block and restarts likewise, at exactly `pos_w`.
    for (i, angle_of) in [(0usize, 1usize), (ROPE_SECTIONS[1] as usize, 2usize)] {
        for m in 0..rows {
            let angle = positions[m * 3 + angle_of] as f32;
            let (sin, cos) = angle.sin_cos();
            for h in 0..ROPE_Q_HEADS as usize {
                let lo = (m * ROPE_Q_HEADS as usize + h) * head_dim + i;
                let hi = lo + half;
                let (x1, x2) = (q0[lo], q0[hi]);
                assert!(
                    (got_qb[lo] - (x1 * cos - x2 * sin)).abs() <= BAND
                        && (got_qb[hi] - (x1 * sin + x2 * cos)).abs() <= BAND,
                    "the blocked arm turned pair {i} of head {h}, row {m}, by something \
                     other than {angle} radians — each blocked section RESTARTS the \
                     frequency ladder at `within = 0`, so the first pair of the `h` block \
                     and the first pair of the `w` block both turn at `inv_freq = 1`"
                );
                assert_ne!(
                    got_qb[lo].to_bits(),
                    x1.to_bits(),
                    "the blocked arm left pair {i} of head {h}, row {m}, alone — `s0 == 0` \
                     means there is NO `t` block, so no pair of a blocked rotation at \
                     `[0, k, k]` is a pair that does not turn"
                );
            }
        }
    }

    // The two arms are two, on the same bytes: if any element agreed
    // everywhere, one of them is a second name for the other.
    let differ = got_qb
        .iter()
        .zip(got_qi.iter())
        .filter(|(a, b)| (**a - **b).abs() > BAND)
        .count();
    assert!(
        differ > 0,
        "the blocked and interleaved arms answered the same rectangle over the same \
         bytes — one is then a second name for the other, and a tower would be rotated \
         by the trunk's ladder without anything refusing"
    );
    println!(
        "rope_mrope: {differ} of {} q elements separate the two arms",
        got_qb.len()
    );
}

/// The f32 rotation both `.metal` arms are read against, transcribed from the
/// shaders' stated semantics rather than from their spelling: `(i, i +
/// head_dim/2)` pairs, `rotate_half`, and a frequency `θ^(-d)` whose `d` is
/// the only thing the two forms disagree about.
///
/// * BLOCKED: contiguous sections, `d = 2·within / Σsections`;
/// * INTERLEAVED: `t, h, w, t, h, w, …`, `d = 2·i / head_dim`.
fn rotate_reference(x: &[f32], heads: u32, positions: &[i32], blocked: bool) -> Vec<f32> {
    let head_dim = ROPE_HEAD_DIM as usize;
    let half = head_dim / 2;
    let heads = heads as usize;
    let rows = x.len() / (heads * head_dim);
    let [s0, s1, s2] = ROPE_SECTIONS.map(|s| s as usize);
    let total = (s0 + s1 + s2) as f32;

    let mut out = x.to_vec();
    for m in 0..rows {
        for h in 0..heads {
            for i in 0..half {
                let (axis, d) = if blocked {
                    let (axis, within) = if i < s0 {
                        (0, i)
                    } else if i < s0 + s1 {
                        (1, i - s0)
                    } else {
                        (2, i - s0 - s1)
                    };
                    (axis, 2.0 * within as f32 / total)
                } else {
                    let axis = match i % 3 {
                        1 if i < 3 * s1 => 1,
                        2 if i < 3 * s2 => 2,
                        _ => 0,
                    };
                    (axis, 2.0 * i as f32 / head_dim as f32)
                };
                let angle = positions[m * 3 + axis] as f32 * ROPE_THETA.powf(-d);
                let (sin, cos) = angle.sin_cos();
                let lo = (m * heads + h) * head_dim + i;
                let hi = lo + half;
                out[lo] = x[lo] * cos - x[hi] * sin;
                out[hi] = x[lo] * sin + x[hi] * cos;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The host half: bf16, the two moment forms, and the staging shorthand.
// ---------------------------------------------------------------------------

/// f32 → the two bytes of its bf16, little-endian, ROUND TO NEAREST EVEN.
///
/// `device_floor`'s own `bf16` truncates, which is exact for the values it
/// stages (small integers) and wrong for these: the pool's answer is
/// `264/9 = 29.3333`, whose bf16 neighbours are 29.25 and 29.375, and Metal's
/// `bfloat(float)` conversion rounds to nearest — so a truncating reference
/// would disagree with a correct kernel by one quantum on exactly the
/// elements this file asserts bit-exactly. Same signature, one more line.
fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let word = if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        // A NaN keeps a payload bit rather than becoming an infinity.
        ((bits >> 16) | 0x0040) as u16
    } else {
        let rounding = 0x7fff + ((bits >> 16) & 1);
        (bits.wrapping_add(rounding) >> 16) as u16
    };
    word.to_le_bytes()
}

/// The value a bf16 word reads back as.
fn f32_of(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

/// An f32 as the number it becomes after one bf16 store and one load —
/// the reference for every bit-exact assertion in this file.
fn f32_round(v: f32) -> f32 {
    f32_of(u16::from_le_bytes(bf16(v)))
}

/// A row of f32 as the bf16 bytes the shell would stage.
fn encode(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| bf16(*v)).collect()
}

/// bf16 bytes as the f32 they read back as.
fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32_of(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// The bf16 quantum at `v`: eight significant bits below the binade.
fn quantum(v: f32) -> f32 {
    if v == 0.0 {
        return f32::MIN_POSITIVE;
    }
    v.abs().log2().floor().exp2() / 128.0
}

/// Within `quanta` bf16 quanta of `want`, with a floor of one quantum at 1.0
/// so an answer near zero is not held to an impossible relative band.
fn near(got: f32, want: f32, quanta: f32) -> bool {
    (got - want).abs() <= quanta * quantum(want.abs().max(1.0))
}

/// **THE TWO-REDUCTION MOMENTS**, which is what `norm_layernorm.metal` and
/// `layernorm.cuh` both compute: the mean, then the centred squares against
/// that mean. Sequential f32, so the reference is a number and not a range.
fn two_pass(row: &[f32], eps: f32) -> (f32, f32, f32) {
    let n = row.len() as f32;
    let mut sum = 0.0f32;
    for v in row {
        sum += *v;
    }
    let mean = sum / n;
    let mut spread = 0.0f32;
    for v in row {
        let c = *v - mean;
        spread += c * c;
    }
    let var = spread / n;
    (mean, var, 1.0 / (var + eps).sqrt())
}

/// **THE ONE-REDUCTION MOMENTS**, `var = E[x²] − E[x]²` — the form that
/// halves the barriers and is the reason the kernel would be subtly wrong.
/// Never fired; computed on the host so the gate can say which of the two
/// numbers the device landed nearer.
fn one_pass(row: &[f32], eps: f32) -> (f32, f32, f32) {
    let n = row.len() as f32;
    let (mut sum, mut squares) = (0.0f32, 0.0f32);
    for v in row {
        sum += *v;
        squares += *v * *v;
    }
    let mean = sum / n;
    let var = squares / n - mean * mean;
    (mean, var, 1.0 / (var + eps).sqrt())
}

/// A reservation holding exactly these bytes.
fn staged(device: &Context, bytes: &[u8]) -> Buffer {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("the reservation lands");
    buffer.write(0, bytes).expect("the bytes land");
    buffer
}

/// A handle over the whole of a reservation.
fn bind_whole(handles: &Handles, buffer: &Buffer, what: &str) -> u32 {
    handles
        .bind(buffer, 0, buffer.bytes())
        .unwrap_or_else(|fault| panic!("{what} binds: {fault}"))
}

/// The bytes of a reservation, read back.
fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut got = vec![0u8; bytes];
    buffer.read(0, &mut got).expect("the answer reads back");
    got
}

/// An `i32` slice as the bytes the shell would stage — `device_floor`'s own
/// `bytemuck_i32`, under the name this file's one i32 stream wants.
fn as_bytes_i32(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` has no padding and no invalid bit patterns, and the
    // slice's lifetime is the borrow's.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
