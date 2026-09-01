//! The hyper-connection family, on a real Apple GPU: the expansion, the
//! widening RMS norm, the Sinkhorn-gated split, and the fold that puts the
//! sublayer's output back into the streams.
//!
//! **WHAT THIS FILE IS FOR.** `elementwise.hc_*` was one of the two fatal
//! families for the dsv4 SKUs — four entries, all of them typed refusals, so
//! nothing on this plane had ever run the arithmetic. The port is
//! organ-for-organ off `kernels-cuda/kernels/elemwise/hc.cuh`, and the one
//! number a faithful port can still get wrong is the SINKHORN LOOP COUNT: the
//! combiner is seeded with a row softmax and ONE column normalization, and
//! only then does the alternating loop run `sinkhorn_iters - 1` times. Twenty
//! stated iterations are nineteen loop passes. An off-by-one there produces a
//! matrix that is still nearly doubly stochastic and still runs, which is
//! exactly the kind of wrong that survives every check but a measurement.
//!
//! So [`the_sinkhorn_count_is_the_devices_too`] does not assert the count — it
//! SWEEPS the device across `sinkhorn` in {19, 20, 21} against a CPU fp32
//! reference pinned at 20 and reports each. One lands at fp32 epsilon and the
//! neighbours do not.
//!
//! # The six gates
//!
//! ```text
//! (a) hc_expand      — every stream is the same row, byte for byte
//! (b) hc_rmsnorm_f32 — the wide row, weightlessly normed and widened, vs a
//!                      CPU fp32 reference (this one is f32 OUT, so the band
//!                      is fp32 epsilon and not a bf16 quantum)
//! (c) hc_gates       — post_mix and comb_mix vs the CPU fp32 reference, the
//!                      collapsed layer input vs the same in the bf16 band,
//!                      and comb_mix's own rows and columns summing to one
//! (d) hc_fold        — the recombination vs the CPU fp32 reference
//! (e) the count sweep — 19 / 20 / 21 iterations, measured
//! (f) hc_project    — the mix row projected out of the dynamic hyper plane,
//!                     vs a host dot product, and vs the normed buffer it
//!                     used to stand in for
//! ```
//!
//! **THE MIX ROW IS A PROJECTION NOW, AND GATE (f) IS IT.** `hc_gates` reads
//! its operand at a `2M + M²` stride and always has — that is the MLX
//! reference's `rmsnorm(streams) @ hc_fn`. While no plane fired the dynamic
//! hyper plane, both shells were handed `normed` itself and read its leading
//! `2M + M²` floats. `elementwise.hc_project` is that GEMM, and gate (f)
//! measures it against a host dot product AND against the stand-in it
//! replaced. Gates (c)-(e) keep feeding `hc_gates` the normed plane directly,
//! because what they measure is the gate's own arithmetic and the operand's
//! provenance is gate (f)'s business.
//!
//! # Gating
//!
//! As `device_floor` and `mla_on_device`: `cfg`'d to Apple at compile time,
//! and SKIPS at run time when `device::present()` says no, saying so.
//!
//! ```text
//! cargo test -p engine-metal --test hc_on_device -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::hc::reference;
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, for `device_floor`'s reason: two tests compiling
/// shaders at once meets the Metal compiler's own concurrency and learns
/// nothing.
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

// ---------------------------------------------------------------------------
// The geometry: dsv4's own stream fan, a hidden width that makes the per-row
// points loop more than twice over their 256-thread group, and enough rows
// that a kernel addressing row 0 by accident would be caught.
// ---------------------------------------------------------------------------

const M: usize = 4;
const H: usize = 576;
const ROWS: usize = 5;
/// dsv4's stated count. The seed plus NINETEEN sweeps.
const SINKHORN: u32 = 20;
const GATE_EPS: f32 = 1e-6;
/// The reference's `2·sigmoid` for the post (depth) gate.
const ALPHA: f32 = 2.0;
const NORM_EPS: f32 = 1e-6;

/// The mix row's stride: `M` pre logits, `M` post logits, `M²` combiner
/// logits. `hc.cuh`'s `mix_hc`.
const MIX_HC: usize = M * 2 + M * M;

// ---------------------------------------------------------------------------
// The fixture. Every plane already through bf16 where the device stores bf16,
// so the reference and the device read the SAME numbers and the only
// difference either can show is arithmetic.
// ---------------------------------------------------------------------------

struct Fixture {
    /// `[ROWS, M·H]` bf16 — the hyper stream.
    streams: Vec<f32>,
    /// `[ROWS, H]` bf16 — one sublayer's output, the fold's `x`.
    x: Vec<f32>,
    /// `[ROWS, M·H]` f32 — the "normed" plane the gate reads its mix row out
    /// of, at `MIX_HC` stride.
    normed: Vec<f32>,
    /// `[3]` f32 — the per-plane scales (pre, post, comb).
    scale: Vec<f32>,
    /// `[MIX_HC]` f32 — the per-logit bases.
    base: Vec<f32>,
}

impl Fixture {
    fn new(seed: u64) -> Self {
        let mut rng = Lcg(seed);
        Self {
            streams: rng.bf16_plane(ROWS * M * H),
            x: rng.bf16_plane(ROWS * H),
            normed: (0..ROWS * M * H).map(|_| rng.next_f32() * 4.0).collect(),
            // Deliberately not all one: a shell that read `scale[0]` for every
            // plane would pass a test whose scales agreed.
            scale: vec![1.25, 0.75, 1.5],
            base: (0..MIX_HC).map(|_| rng.next_f32() * 2.0).collect(),
        }
    }

    /// Row `n`'s mix row, as BOTH shells address it: the leading `MIX_HC`
    /// floats at `n · MIX_HC`, which is the normed buffer's own start and not
    /// its row stride. See the module note.
    fn mix_row(&self, n: usize) -> &[f32] {
        &self.normed[n * MIX_HC..(n + 1) * MIX_HC]
    }

    fn pre(&self, n: usize) -> Vec<f32> {
        let row = self.mix_row(n);
        (0..M)
            .map(|i| reference::pre_gate(row[i], self.scale[0], self.base[i], GATE_EPS))
            .collect()
    }

    fn post(&self, n: usize) -> Vec<f32> {
        let row = self.mix_row(n);
        (0..M)
            .map(|i| reference::post_gate(row[M + i], self.scale[1], self.base[M + i], ALPHA))
            .collect()
    }

    /// The combiner at a chosen iteration count — the sweep's knob.
    fn comb_at(&self, n: usize, iters: u32) -> Vec<f32> {
        let row = self.mix_row(n);
        let logits: Vec<f32> = (0..M * M)
            .map(|k| row[2 * M + k] * self.scale[2] + self.base[2 * M + k])
            .collect();
        reference::sinkhorn(&logits, M, iters, GATE_EPS)
    }

    /// A twin whose combiner LOGITS are a deliberately slow-converging matrix,
    /// installed through the normed plane so `scale[2]` and the bases stay
    /// live: rows 1 and 3 peak in the same column, so the alternating
    /// normalization has a real transport problem and is still moving at
    /// twenty iterations. Gate (e) needs that; the other gates do not, and use
    /// the random plane.
    fn with_a_slow_combiner(mut self) -> Self {
        for n in 0..ROWS {
            for k in 0..M * M {
                let target = (((k + n) * 7) % 11) as f32 - 5.0;
                self.normed[n * MIX_HC + 2 * M + k] =
                    (target - self.base[2 * M + k]) / self.scale[2];
            }
        }
        self
    }

    fn stream_row(&self, n: usize) -> &[f32] {
        &self.streams[n * M * H..(n + 1) * M * H]
    }

    fn x_row(&self, n: usize) -> &[f32] {
        &self.x[n * H..(n + 1) * H]
    }
}

// ---------------------------------------------------------------------------
// Gate (a): the expansion.
// ---------------------------------------------------------------------------

/// Every stream of an expanded row is the SAME row, byte for byte — the
/// expansion is a tile and not an arithmetic step, so the claim is equality
/// and not a band.
#[test]
fn the_expansion_tiles_one_row_across_every_stream() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the hc expansion") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x11c0_0001);

    let src = staged(&device, &encode_bf16(&fx.x));
    let dst = Buffer::zeroed(&device, (ROWS * M * H * 2) as u64).expect("the wide row reserves");
    let src_h = bind_whole(&handles, &src, "the narrow row");
    let dst_h = bind_whole(&handles, &dst, "the wide row");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::elemwise::hc::expand(
            &sink,
            Tensor::new(src_h, ROWS as u32, H as u32, Dtype::Bf16),
            M as u32,
            Tensor::new(dst_h, ROWS as u32, (M * H) as u32, Dtype::Bf16),
        )
        .expect("the expansion encodes");
    }
    frame.commit().expect("the expansion completes");

    let got = decode_bf16(&read_back(&dst, ROWS * M * H * 2));
    for n in 0..ROWS {
        for s in 0..M {
            for h in 0..H {
                assert_eq!(
                    got[(n * M + s) * H + h],
                    fx.x_row(n)[h],
                    "row {n} stream {s} lane {h} is not the row it was tiled from"
                );
            }
        }
    }
    println!("(a) hc_expand: {ROWS} rows x {M} streams x {H} lanes, exact");
}

// ---------------------------------------------------------------------------
// Gate (b): the widening norm.
// ---------------------------------------------------------------------------

/// The wide row is normed WEIGHTLESSLY over its whole `M·H` extent — not per
/// stream — and lands in f32. The output dtype is the point: the band here is
/// fp32 epsilon, so a reduction that had drifted through bf16 would show.
#[test]
fn the_widening_norm_matches_the_host_in_fp32() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the hc widening norm") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x11c0_0002);

    let src = staged(&device, &encode_bf16(&fx.streams));
    let dst = Buffer::zeroed(&device, (ROWS * M * H * 4) as u64).expect("the normed row reserves");
    let src_h = bind_whole(&handles, &src, "the stream row");
    let dst_h = bind_whole(&handles, &dst, "the normed row");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::elemwise::hc::rmsnorm_f32(
            &sink,
            Tensor::new(src_h, ROWS as u32, (M * H) as u32, Dtype::Bf16),
            NORM_EPS,
            Tensor::new(dst_h, ROWS as u32, (M * H) as u32, Dtype::F32),
        )
        .expect("the norm encodes");
    }
    frame.commit().expect("the norm completes");

    let got = decode_f32(&read_back(&dst, ROWS * M * H * 4));
    let mut worst = 0.0f32;
    for n in 0..ROWS {
        let want = reference::rmsnorm(fx.stream_row(n), NORM_EPS);
        for (k, w) in want.iter().enumerate() {
            let rel = (got[n * M * H + k] - w).abs() / w.abs().max(1e-3);
            worst = worst.max(rel);
        }
    }
    println!("(b) hc_rmsnorm_f32 vs the host fp32 norm: worst relative {worst:.3e}");
    assert!(
        worst < 1e-5,
        "the widening norm drifted {worst:.3e} relative — a tree reduction differs from a \
         sequential one at fp32 rounding, not at this scale"
    );
}

// ---------------------------------------------------------------------------
// Gate (f): the mix PROJECTION — the organ that used to be missing.
// ---------------------------------------------------------------------------

/// **THE MIX ROW IS PROJECTED NOW, AND THIS IS THE GEMM THAT DOES IT.**
///
/// `elementwise.hc_project` is `rmsnorm(streams) · hc_fn^T` — the
/// `{attn,ffn}_hc.fn` plane the model text interned for as long as no op
/// could fire it, landing the `[N, 2M + M²]` row `hc_gates` splits. Two
/// claims, and the second is the one that matters:
///
/// 1. every one of the `2M + M²` columns is the host's own dot product over
///    the whole `M·H` contraction, in the fp32 band this family insists on;
/// 2. the projected row is NOT the leading `2M + M²` floats of `normed` — the
///    stand-in both shells read while the plane was interned — so a chain
///    that quietly kept passing the normed buffer through would fail here
///    rather than answer plausible numbers.
#[test]
fn the_mix_row_is_the_projection_and_not_the_normed_row() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the hc mix projection") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x11c0_0006);

    // The dynamic hyper plane, `[MIX_HC, M·H]` f32 — small numbers, because
    // the contraction is `M·H` long and the reference sums it the same way.
    let mut rng = Lcg(0x11c0_00f6);
    let hc_fn: Vec<f32> = (0..MIX_HC * M * H).map(|_| rng.next_f32() * 0.5).collect();

    let normed = staged(&device, &encode_f32(&fx.normed));
    let plane = staged(&device, &encode_f32(&hc_fn));
    let mixes = Buffer::zeroed(&device, (ROWS * MIX_HC * 4) as u64).expect("the mix row reserves");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        kernels_metal::elemwise::hc::project(
            &sink,
            Tensor::new(
                bind_whole(&handles, &normed, "the normed row"),
                ROWS as u32,
                (M * H) as u32,
                Dtype::F32,
            ),
            Tensor::new(
                bind_whole(&handles, &plane, "the dynamic plane"),
                MIX_HC as u32,
                (M * H) as u32,
                Dtype::F32,
            ),
            M as u32,
            Tensor::new(
                bind_whole(&handles, &mixes, "the mix row"),
                ROWS as u32,
                MIX_HC as u32,
                Dtype::F32,
            ),
        )
        .expect("the projection encodes");
    }
    frame.commit().expect("the projection completes");

    let got = decode_f32(&read_back(&mixes, ROWS * MIX_HC * 4));

    let mut worst = 0.0f32;
    let mut apart = 0usize;
    for n in 0..ROWS {
        let row = &fx.normed[n * M * H..(n + 1) * M * H];
        for o in 0..MIX_HC {
            let w = &hc_fn[o * M * H..(o + 1) * M * H];
            let want: f32 = row.iter().zip(w).map(|(a, b)| a * b).sum();
            let rel = (got[n * MIX_HC + o] - want).abs() / want.abs().max(1e-3);
            worst = worst.max(rel);
            // Claim 2: the column is not the stand-in it replaced.
            if (got[n * MIX_HC + o] - fx.normed[n * MIX_HC + o]).abs() > 1e-3 {
                apart += 1;
            }
        }
    }
    println!(
        "(f) hc_project vs the host dot product: {ROWS} rows x {MIX_HC} columns over a \
         {}-long contraction, worst relative {worst:.3e}",
        M * H
    );
    assert!(
        worst < 1e-4,
        "the projection drifted {worst:.3e} relative — a tree reduction differs from a \
         sequential one at fp32 rounding, not at this scale"
    );
    assert_eq!(
        apart,
        ROWS * MIX_HC,
        "{} of {} projected columns equal the normed buffer's own leading floats, so the \
         mix row is the stand-in and not the projection",
        ROWS * MIX_HC - apart,
        ROWS * MIX_HC
    );
}

// ---------------------------------------------------------------------------
// Gates (c) and (d): the split and the fold.
// ---------------------------------------------------------------------------

/// The whole gated step, measured plane by plane: the two gate matrices in
/// fp32, the collapse and the fold in the bf16 band, and — the property the
/// Sinkhorn projection exists for — the combiner's own rows and columns
/// summing to one.
#[test]
fn the_gates_and_the_fold_are_the_host_arithmetic() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the hc gates and fold") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let fx = Fixture::new(0x11c0_0003);

    let (post_mix, comb_mix, layer_in) = fire_gates(&device, &pipelines, &handles, &fx, SINKHORN);

    // ── the gate matrices, in fp32 ───────────────────────────────────────
    let mut post_worst = 0.0f32;
    let mut comb_worst = 0.0f32;
    for n in 0..ROWS {
        for (i, w) in fx.post(n).iter().enumerate() {
            post_worst = post_worst.max((post_mix[n * M + i] - w).abs());
        }
        for (k, w) in fx.comb_at(n, SINKHORN).iter().enumerate() {
            comb_worst = comb_worst.max((comb_mix[n * M * M + k] - w).abs());
        }
    }
    println!("(c) hc_gates: post_mix worst {post_worst:.3e}, comb_mix worst {comb_worst:.3e}");
    assert!(post_worst < 1e-6, "the post gate drifted {post_worst:.3e}");
    assert!(
        comb_worst < 1e-5,
        "the combiner drifted {comb_worst:.3e} — twenty fp32 sweeps of a 4x4 do not accumulate \
         that, so this is the ITERATION COUNT and not the arithmetic"
    );

    // ── the combiner is doubly stochastic, on the device's own numbers ───
    let mut row_worst = 0.0f32;
    let mut col_worst = 0.0f32;
    for n in 0..ROWS {
        let c = &comb_mix[n * M * M..(n + 1) * M * M];
        for i in 0..M {
            let s: f32 = (0..M).map(|j| c[i * M + j]).sum();
            row_worst = row_worst.max((s - 1.0).abs());
        }
        for j in 0..M {
            let s: f32 = (0..M).map(|i| c[i * M + j]).sum();
            col_worst = col_worst.max((s - 1.0).abs());
        }
    }
    println!("    the device's combiner: rows off 1 by {row_worst:.3e}, columns by {col_worst:.3e}");
    assert!(row_worst < 1e-3, "the rows are not stochastic: {row_worst:.3e}");
    // The column normalization is the LAST half-sweep, so the columns are the
    // tighter of the two — and if this ever loosened it would mean the loop
    // ended on the wrong half.
    assert!(col_worst < 1e-5, "the columns are not stochastic: {col_worst:.3e}");

    // ── the collapse, in the bf16 band ───────────────────────────────────
    let mut collapse_worst = 0.0f32;
    for n in 0..ROWS {
        let want = reference::collapse(&fx.pre(n), fx.stream_row(n), M, H);
        for (h, w) in want.iter().enumerate() {
            let d = (layer_in[n * H + h] - w).abs();
            collapse_worst = collapse_worst.max(d / quantum(w.abs().max(0.05)));
        }
    }
    println!("    the collapsed layer input: {collapse_worst:.2} bf16 quanta");
    assert!(collapse_worst <= 2.0, "the collapse drifted {collapse_worst:.2} quanta");

    // ── the fold ─────────────────────────────────────────────────────────
    let folded = fire_fold(&device, &pipelines, &handles, &fx, &post_mix, &comb_mix);
    let mut fold_worst = 0.0f32;
    for n in 0..ROWS {
        let want = reference::fold(
            fx.x_row(n),
            fx.stream_row(n),
            &fx.post(n),
            &fx.comb_at(n, SINKHORN),
            M,
            H,
        );
        for (k, w) in want.iter().enumerate() {
            let d = (folded[n * M * H + k] - w).abs();
            fold_worst = fold_worst.max(d / quantum(w.abs().max(0.05)));
        }
    }
    println!("(d) hc_fold vs the host reference: {fold_worst:.2} bf16 quanta");
    assert!(fold_worst <= 2.0, "the fold drifted {fold_worst:.2} quanta");
}

// ---------------------------------------------------------------------------
// Gate (e): the count sweep.
// ---------------------------------------------------------------------------

/// **THE OFF-BY-ONE, MEASURED ACROSS THE DEVICE BOUNDARY — ON A MATRIX WHERE
/// IT IS WORTH SOMETHING.**
///
/// The first shape of this gate swept the device across `sinkhorn` in
/// {19, 20, 21} against a reference pinned at 20, on the random mix plane the
/// other gates use, and asked which matched. All three matched, to `1.2e-7`,
/// and the sweep proved nothing. That was not a slack test — it was a fact
/// about the fixture: Sinkhorn-Knopp is a contraction whose rate is the
/// logits' own, and a mix row whose row and column maxima already agree is
/// finished in two sweeps.
///
/// So the sweep is fired on a combiner that is still MOVING at twenty: rows
/// competing for the same column, installed through the normed plane so
/// `scale[2]` and the bases stay live. There one sweep is worth `2e-4` and the
/// device's own count is decidable — which is the honest version of the
/// gotcha. The fast plane is then reported beside it, so the file says both
/// halves: the trap costs three orders of magnitude on one matrix and nothing
/// at all on another, and only a test that knows the difference is measuring
/// anything.
#[test]
fn the_sinkhorn_count_is_the_devices_too() {
    let _serial = serialized();
    let Some(device) = device_or_skip("the sinkhorn count") else {
        return;
    };
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    let slow = Fixture::new(0x11c0_0004).with_a_slow_combiner();

    println!(
        "(e) a still-moving combiner, device vs a host reference pinned at {SINKHORN} iterations:"
    );
    let mut verdicts = Vec::new();
    for iters in [SINKHORN - 1, SINKHORN, SINKHORN + 1] {
        let (_, comb_mix, _) = fire_gates(&device, &pipelines, &handles, &slow, iters);
        let mut worst = 0.0f32;
        for n in 0..ROWS {
            for (k, w) in slow.comb_at(n, SINKHORN).iter().enumerate() {
                worst = worst.max((comb_mix[n * M * M + k] - w).abs());
            }
        }
        println!("      sinkhorn = {iters:<3} worst |device - reference@{SINKHORN}| = {worst:.3e}");
        verdicts.push((iters, worst));
    }

    let matched = verdicts
        .iter()
        .copied()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("three candidates");
    assert_eq!(
        matched.0, SINKHORN,
        "the device's own loop bound disagrees with `sinkhorn - 1`: {verdicts:?}"
    );
    assert!(
        matched.1 < 1e-6,
        "even the matching count is off by {:.3e}",
        matched.1
    );
    let neighbour = verdicts
        .iter()
        .filter(|(iters, _)| *iters != SINKHORN)
        .map(|(_, worst)| *worst)
        .fold(f32::INFINITY, f32::min);
    assert!(
        neighbour > matched.1 * 100.0,
        "one sweep either way is not observable on this matrix either, so the sweep proves \
         nothing: {verdicts:?}"
    );

    // The contrast, on the plane the other gates use — reported, not asserted
    // away, because it is why this gate needed its own fixture.
    let fast = Fixture::new(0x11c0_0004);
    println!("    and on the fast-converging plane the other gates use:");
    for iters in [SINKHORN - 1, SINKHORN, SINKHORN + 1] {
        let (_, comb_mix, _) = fire_gates(&device, &pipelines, &handles, &fast, iters);
        let mut worst = 0.0f32;
        for n in 0..ROWS {
            for (k, w) in fast.comb_at(n, SINKHORN).iter().enumerate() {
                worst = worst.max((comb_mix[n * M * M + k] - w).abs());
            }
        }
        println!("      sinkhorn = {iters:<3} worst |device - reference@{SINKHORN}| = {worst:.3e}");
    }
}

// ---------------------------------------------------------------------------
// The fires.
// ---------------------------------------------------------------------------

/// `hc_gates` on the device, at a chosen iteration count. Returns
/// `(post_mix, comb_mix, layer_input)` already decoded.
fn fire_gates(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    fx: &Fixture,
    sinkhorn: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let normed = staged(device, &encode_f32(&fx.normed));
    let streams = staged(device, &encode_bf16(&fx.streams));
    let scale = staged(device, &encode_f32(&fx.scale));
    let base = staged(device, &encode_f32(&fx.base));
    let x = Buffer::zeroed(device, (ROWS * H * 2) as u64).expect("the layer input reserves");
    let post = Buffer::zeroed(device, (ROWS * M * 4) as u64).expect("post_mix reserves");
    let comb = Buffer::zeroed(device, (ROWS * M * M * 4) as u64).expect("comb_mix reserves");

    let normed_h = bind_whole(handles, &normed, "the mix plane");
    let streams_h = bind_whole(handles, &streams, "the stream row");
    let scale_h = bind_whole(handles, &scale, "the mix scales");
    let base_h = bind_whole(handles, &base, "the mix bases");
    let x_h = bind_whole(handles, &x, "the layer input");
    let post_h = bind_whole(handles, &post, "post_mix");
    let comb_h = bind_whole(handles, &comb, "comb_mix");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        kernels_metal::elemwise::hc::gates(
            &sink,
            Tensor::new(normed_h, ROWS as u32, (M * H) as u32, Dtype::F32),
            Tensor::new(streams_h, ROWS as u32, (M * H) as u32, Dtype::Bf16),
            Tensor::new(scale_h, 1, 3, Dtype::F32),
            Tensor::new(base_h, 1, MIX_HC as u32, Dtype::F32),
            M as u32,
            GATE_EPS,
            ALPHA,
            sinkhorn,
            Tensor::new(x_h, ROWS as u32, H as u32, Dtype::Bf16),
            Tensor::new(post_h, ROWS as u32, M as u32, Dtype::F32),
            Tensor::new(comb_h, ROWS as u32, (M * M) as u32, Dtype::F32),
        )
        .expect("the gates encode");
    }
    frame.commit().expect("the gates complete");

    (
        decode_f32(&read_back(&post, ROWS * M * 4)),
        decode_f32(&read_back(&comb, ROWS * M * M * 4)),
        decode_bf16(&read_back(&x, ROWS * H * 2)),
    )
}

/// `hc_fold` on the device, handed the gate matrices the device itself
/// produced — the chain as a layer runs it, not a reference's copy of it.
fn fire_fold(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    fx: &Fixture,
    post_mix: &[f32],
    comb_mix: &[f32],
) -> Vec<f32> {
    let x = staged(device, &encode_bf16(&fx.x));
    let streams = staged(device, &encode_bf16(&fx.streams));
    let post = staged(device, &encode_f32(post_mix));
    let comb = staged(device, &encode_f32(comb_mix));
    let y = Buffer::zeroed(device, (ROWS * M * H * 2) as u64).expect("the folded row reserves");

    let x_h = bind_whole(handles, &x, "the sublayer output");
    let streams_h = bind_whole(handles, &streams, "the stream row");
    let post_h = bind_whole(handles, &post, "post_mix");
    let comb_h = bind_whole(handles, &comb, "comb_mix");
    let y_h = bind_whole(handles, &y, "the folded row");

    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        kernels_metal::elemwise::hc::fold(
            &sink,
            Tensor::new(x_h, ROWS as u32, H as u32, Dtype::Bf16),
            Tensor::new(streams_h, ROWS as u32, (M * H) as u32, Dtype::Bf16),
            Tensor::new(post_h, ROWS as u32, M as u32, Dtype::F32),
            Tensor::new(comb_h, ROWS as u32, (M * M) as u32, Dtype::F32),
            Tensor::new(y_h, ROWS as u32, (M * H) as u32, Dtype::Bf16),
        )
        .expect("the fold encodes");
    }
    frame.commit().expect("the fold completes");
    decode_bf16(&read_back(&y, ROWS * M * H * 2))
}

// ---------------------------------------------------------------------------
// Host staging.
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        (x - 0.5) * 0.5
    }

    /// `n` values in `[-0.25, 0.25)`, **already through bf16** — the reference
    /// and the device must read the same numbers.
    fn bf16_plane(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32_of(bf16_bits(self.next_f32()))).collect()
    }
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn f32_of(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32_of(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The bf16 quantum at `v`: eight significant bits below the binade.
fn quantum(v: f32) -> f32 {
    if v == 0.0 {
        return f32::MIN_POSITIVE;
    }
    v.abs().log2().floor().exp2() / 128.0
}

fn staged(device: &Context, bytes: &[u8]) -> Buffer {
    let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("the reservation lands");
    buffer.write(0, bytes).expect("the bytes land");
    buffer
}

fn bind_whole(handles: &Handles, buffer: &Buffer, what: &str) -> u32 {
    handles
        .bind(buffer, 0, buffer.bytes())
        .unwrap_or_else(|fault| panic!("{what} binds: {fault}"))
}

fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut got = vec![0u8; bytes];
    buffer.read(0, &mut got).expect("the answer reads back");
    got
}
