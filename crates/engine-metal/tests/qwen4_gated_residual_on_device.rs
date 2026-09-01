//! qwen4's gated-residual family and its concatenating gather, on a real
//! Apple GPU.
//!
//! **WHAT THIS FILE IS FOR.** Six points were typed refusals on this plane
//! until this lane, and every one of them stood between a `qwen38-flash-*`
//! load and its first token:
//!
//!   * `elementwise.rmsnorm_grouped_plus_one` — the hyper-connection norm,
//!     spent four times a layer;
//!   * `elementwise.silu_scaled` — the shared expert's gate;
//!   * `elementwise.hc_mix` / `elementwise.hc_inject` — the stream collapse
//!     and the write-back;
//!   * `elementwise.ple_gate` — the PLE's key·query gate;
//!   * `layout.embed_concat` — sixteen hashed gathers per token, concatenated.
//!
//! and beside them one that refused by ENUM ARM rather than by name:
//! `elementwise.rmsnorm_gated` served `silu` and answered `Unsupported` for
//! `sigmoid`, which is the arm qwen4's GatedDeltaNet fires.
//!
//! # How each is measured, and why differently
//!
//! Everything here lands a NUMBER, so the bands are bf16 quanta against a host
//! fp32 reference (`kernels_metal::elemwise::hc::reference`, which states the
//! same arithmetic in Rust and is itself pinned to hand-computed values in
//! that crate's `--lib` suite). But a band alone would pass a port that is
//! plausibly wrong in the three ways this family invites, so each of those is
//! held apart on its own:
//!
//!   * **the grouped norm's bank is PER STREAM.** A port that read one plane
//!     for every stream — which is what the per-head norm beside it does —
//!     lands the right spread around `M − 1` wrong centres, with no NaN
//!     anywhere. So the streams are gained by DIFFERENT planes and the answer
//!     is held against the per-head reading as well as against its own.
//!   * **the injection's gate divides INSIDE the sigmoid.** `2σ(g/M)` and
//!     `2σ(g)/M` agree at `g = 0` and nowhere else, so the fixture's logits
//!     are far from zero.
//!   * **the PLE gate's damping carries the DOT'S SIGN**, and `sign(0)` is
//!     zero rather than the `1e-6` clamp floor. Both signs and the exact zero
//!     are fixtures.
//!
//! The concatenating gather lands a TABLE ROW, so it is measured exactly: a
//! gather off by one head is a different row of a twenty-million-row table and
//! there is no band it could be inside.
//!
//! As `device_floor`, `hc_on_device` and `ple_conv_on_device`: `cfg`'d to Apple
//! at compile time, and SKIPS at run time when `device::present()` says no.
//!
//! ```text
//! cargo test -p engine-metal --test qwen4_gated_residual_on_device -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::elemwise::hc::reference;
use kernels_metal::elemwise::{hc, norm};
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

/// **ONE DEVICE AT A TIME**, for `device_floor`'s reason.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device_or_skip(what: &str) -> Option<(Context, Pipelines, Handles)> {
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    let context = Context::bind().expect("the device binds");
    let pipelines = Pipelines::new();
    let handles = Handles::new();
    Some((context, pipelines, handles))
}

/// `Qwen3.8-Flash-Next`'s own residual geometry, shrunk on the width axis only
/// so the numbers stay readable: FOUR streams, a hidden width that is a whole
/// number of simdgroups so the reductions take their multi-simdgroup path, and
/// three rows so a kernel addressing row zero by accident is caught.
const M: usize = 4;

const H: usize = 128;

const ROWS: usize = 3;

const EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// (a) the hyper-connection norm
// ---------------------------------------------------------------------------

/// **THE BANK IS PER STREAM, AND THIS IS THE ASSERT THAT SAYS SO.**
///
/// Held against the reference in bf16 quanta, and then held APART from the
/// per-head reading of the same row — the same shader family, one weight
/// index different — because that is the port this could have been and the
/// numbers it lands are finite, spread, and wrong.
#[test]
fn the_grouped_norm_gains_each_stream_by_its_own_plane() {
    let _serial = serialized();
    let Some((device, pipelines, handles)) = device_or_skip("the grouped norm") else {
        return;
    };
    let mut rng = Lcg(0x5eed_1234);
    let x = rng.bf16_plane(ROWS * M * H);
    // Every stream's plane is a DIFFERENT one, and none of them is near zero:
    // a bank of small weights would make the `+ 1` do all the work and the
    // per-stream reading indistinguishable from the shared one.
    let weight: Vec<f32> = (0..M * H)
        .map(|k| f32_of(bf16_bits(0.5 + (k / H) as f32)))
        .collect();

    let got = fire_grouped_norm(&device, &pipelines, &handles, &x, &weight);

    let mut faults = 0usize;
    for row in 0..ROWS {
        let at = row * M * H;
        let want = reference::rmsnorm_grouped_plus_one(&x[at..at + M * H], &weight, H, EPS);
        for k in 0..M * H {
            let band = 2.0 * quantum(want[k]).max(1e-6);
            if (got[at + k] - want[k]).abs() > band {
                faults += 1;
            }
        }
    }
    assert_eq!(faults, 0, "the grouped norm left the reference's band");

    // And the reading it is NOT: one plane shared across every stream.
    let shared: Vec<f32> = weight[..H].to_vec();
    let at = 0;
    let wrong = {
        let mut out = vec![0.0; M * H];
        for s in 0..M {
            let want = reference::rmsnorm_grouped_plus_one(
                &x[at + s * H..at + (s + 1) * H],
                &shared,
                H,
                EPS,
            );
            out[s * H..(s + 1) * H].copy_from_slice(&want);
        }
        out
    };
    let apart = (0..M * H)
        .filter(|k| (got[*k] - wrong[*k]).abs() > 4.0 * quantum(wrong[*k]).max(1e-6))
        .count();
    assert!(
        apart > M * H / 4,
        "only {apart} of {} elements separate the per-stream bank from a shared \
         one, so this fixture cannot tell the two readings apart",
        M * H
    );
    println!(
        "grouped norm: {} rows x {M} streams of {H} within a bf16 quantum, and \
         {apart} elements away from the shared-plane reading",
        ROWS
    );
}

// ---------------------------------------------------------------------------
// (b) the shared gate
// ---------------------------------------------------------------------------

/// `silu(s · x)`, in place — and the scale is INSIDE the curve, so it is held
/// against `s · silu(x)` as well as against its own reference.
#[test]
fn the_scaled_silu_scales_inside_the_curve() {
    let _serial = serialized();
    let Some((device, pipelines, handles)) = device_or_skip("the scaled silu") else {
        return;
    };
    const SCALE: f32 = 2.0;
    let mut rng = Lcg(0xfeed_0007);
    let x = rng.bf16_plane(ROWS * H);
    let got = fire_silu_scaled(&device, &pipelines, &handles, &x, SCALE);
    let want = reference::silu_scaled(&x, SCALE);
    let outside: Vec<f32> = reference::silu_scaled(&x, 1.0).iter().map(|v| v * SCALE).collect();

    let mut faults = 0usize;
    let mut apart = 0usize;
    for k in 0..x.len() {
        if (got[k] - want[k]).abs() > 2.0 * quantum(want[k]).max(1e-6) {
            faults += 1;
        }
        if (want[k] - outside[k]).abs() > 4.0 * quantum(want[k]).max(1e-6) {
            apart += 1;
        }
    }
    assert_eq!(faults, 0, "the scaled silu left the reference's band");
    assert!(
        apart > x.len() / 4,
        "only {apart} elements separate `silu(s·x)` from `s·silu(x)`, so this \
         fixture cannot tell them apart"
    );
    println!("scaled silu: {} elements in band, {apart} away from the outside reading", x.len());
}

// ---------------------------------------------------------------------------
// (c) and (d) the collapse and the write-back
// ---------------------------------------------------------------------------

/// The stream collapse: a MEAN over the fan, gated per element.
#[test]
fn the_stream_mix_averages_the_gated_fan() {
    let _serial = serialized();
    let Some((device, pipelines, handles)) = device_or_skip("the stream mix") else {
        return;
    };
    let mut rng = Lcg(0x1111_2222);
    let gates = rng.bf16_plane(ROWS * M * H);
    let normed = rng.bf16_plane(ROWS * M * H);
    let got = fire_mix(&device, &pipelines, &handles, &gates, &normed);

    let mut faults = 0usize;
    for row in 0..ROWS {
        let wide = row * M * H;
        let want = reference::mix(&gates[wide..wide + M * H], &normed[wide..wide + M * H], M, H);
        for k in 0..H {
            let band = 2.0 * quantum(want[k]).max(1e-6);
            if (got[row * H + k] - want[k]).abs() > band {
                faults += 1;
            }
        }
    }
    assert_eq!(faults, 0, "the stream mix left the reference's band");
    println!("stream mix: {ROWS} rows of {H}, every one within a bf16 quantum");
}

/// The write-back: `+= 2σ(g/M)·o`, in place on the wide row, with the fan
/// divided INSIDE the sigmoid. The logits are far from zero, which is the only
/// place `2σ(g/M)` and `2σ(g)/M` agree.
#[test]
fn the_injection_divides_the_fan_inside_the_sigmoid() {
    let _serial = serialized();
    let Some((device, pipelines, handles)) = device_or_skip("the stream injection") else {
        return;
    };
    let mut rng = Lcg(0x3333_4444);
    let o = rng.bf16_plane(ROWS * H);
    // Logits at ±(2..6): far enough from zero that the divide's position shows.
    let gates: Vec<f32> = (0..ROWS * M)
        .map(|k| f32_of(bf16_bits(if k % 2 == 0 { 2.0 + k as f32 } else { -(2.0 + k as f32) })))
        .collect();
    let hyper = rng.bf16_plane(ROWS * M * H);
    let got = fire_inject(&device, &pipelines, &handles, &o, &gates, &hyper);

    let mut faults = 0usize;
    let mut apart = 0usize;
    for row in 0..ROWS {
        let wide = row * M * H;
        let want = reference::inject(
            &o[row * H..(row + 1) * H],
            &gates[row * M..(row + 1) * M],
            &hyper[wide..wide + M * H],
            M,
            H,
        );
        // The port this is not: the fan divided outside the sigmoid.
        let outside: Vec<f32> = {
            let mut out = hyper[wide..wide + M * H].to_vec();
            for s in 0..M {
                let g = 2.0 / (1.0 + (-gates[row * M + s]).exp()) / M as f32;
                for k in 0..H {
                    out[s * H + k] += g * o[row * H + k];
                }
            }
            out
        };
        for k in 0..M * H {
            let band = 2.0 * quantum(want[k]).max(1e-6);
            if (got[wide + k] - want[k]).abs() > band {
                faults += 1;
            }
            if (want[k] - outside[k]).abs() > 4.0 * quantum(want[k]).max(1e-6) {
                apart += 1;
            }
        }
    }
    assert_eq!(faults, 0, "the injection left the reference's band");
    assert!(
        apart > ROWS * M * H / 4,
        "only {apart} elements separate `2σ(g/M)` from `2σ(g)/M`, so this fixture \
         cannot tell where the fan divides"
    );
    println!("injection: {ROWS} rows x {M} streams in band, {apart} away from the outside divide");
}

// ---------------------------------------------------------------------------
// (e) the PLE gate
// ---------------------------------------------------------------------------

/// The key·query gate, with BOTH signs and an exact zero in the fixture — the
/// three cases the damping's `sign(0) = 0` rule separates.
#[test]
fn the_ple_gate_damps_by_the_signed_root_and_zero_is_a_half() {
    let _serial = serialized();
    let Some((device, pipelines, handles)) = device_or_skip("the PLE gate") else {
        return;
    };
    let mut rng = Lcg(0x7777_8888);
    let value = rng.bf16_plane(ROWS * H);
    let mut key = rng.bf16_plane(ROWS * M * H);
    let mut query = rng.bf16_plane(ROWS * M * H);
    // Stream 0 of row 0: an EXACT zero dot, which must gate at σ(0) = ½ and
    // not at σ(√1e-6). Stream 1 of row 0: the same key against the negated
    // query, so the two are the same curve reflected.
    for i in 0..H {
        key[i] = 0.0;
        query[i] = 0.0;
        query[H + i] = -key[H + i];
    }

    let got = fire_ple_gate(&device, &pipelines, &handles, &key, &query, &value);

    let mut faults = 0usize;
    for row in 0..ROWS {
        let wide = row * M * H;
        let want = reference::ple_gate(
            &key[wide..wide + M * H],
            &query[wide..wide + M * H],
            &value[row * H..(row + 1) * H],
            M,
            H,
        );
        for k in 0..M * H {
            let band = 2.0 * quantum(want[k]).max(1e-6);
            if (got[wide + k] - want[k]).abs() > band {
                faults += 1;
            }
        }
    }
    assert_eq!(faults, 0, "the PLE gate left the reference's band");

    // The zero-dot stream, named: half the value row, to a bf16 quantum. A
    // port that clamped before taking the sign would gate at σ(1e-3) here,
    // which is 0.00025 off — inside any whole-row band and outside this one.
    for k in 0..H {
        let want = 0.5 * value[k];
        assert!(
            (got[k] - want).abs() <= 2.0 * quantum(want).max(1e-6),
            "the zero-dot stream gated at {} of its value where a half was owed",
            if value[k] == 0.0 { f32::NAN } else { got[k] / value[k] },
        );
    }
    println!("PLE gate: {ROWS} rows x {M} streams in band, the zero dot at exactly a half");
}

// ---------------------------------------------------------------------------
// (f) the concatenating gather
// ---------------------------------------------------------------------------

/// **THE CONCATENATING GATHER, EXACTLY.** Sixteen ids per row, each landing
/// one table row of a `(4, 32)` affine bank, side by side.
///
/// Measured exactly and not in a band: the codes are built so that table row
/// `r` dequantizes to a value that names `r`, so a gather that lands the right
/// SHAPE off the wrong row is caught by the number rather than by the norm of
/// a difference. And the ids are a permutation with repeats, so a kernel that
/// read `ids[row]` instead of `ids[row·heads + h]` — the one transposition
/// this entry's fold could get wrong — lands sixteen copies of one row.
#[test]
fn the_concatenating_gather_lands_the_row_each_id_names() {
    let _serial = serialized();
    let Some((device, pipelines, handles)) = device_or_skip("the concatenating gather") else {
        return;
    };
    const HEADS: usize = 16;
    const WIDTH: usize = 32;
    const VOCAB: usize = 64;

    // **ROW `r` READS EXACTLY `r`, AND THE TWO HALVES OF IT COME FROM
    // DIFFERENT PLANES.** A four-bit code carries the LOW nibble and the bias
    // carries the high one — `(r % 16) + 16·(r / 16)` — so the value names its
    // row INJECTIVELY over the whole table, and a gather that lands the right
    // shape off the wrong row is caught by the number.
    //
    // The injectivity is the part worth stating. Coding `r % 16` against a
    // bias of `r` — the obvious spelling — makes rows `r` and `r + 8` read the
    // SAME value whenever `r % 16 >= 8`, which aliases twenty-four pairs of a
    // sixty-four-row table and lets a gather off by eight rows pass a test
    // that calls itself exact. Splitting the row's bits across the code and
    // the bias also keeps both planes load-bearing: dropping the code fails
    // every row not a multiple of sixteen, and dropping the bias fails every
    // row past the first sixteen.
    let mut codes = vec![0u32; VOCAB * WIDTH / 8];
    for r in 0..VOCAB {
        let nibble = (r % 16) as u32;
        let word = (0..8).fold(0u32, |acc, i| acc | (nibble << (4 * i)));
        for w in 0..WIDTH / 8 {
            codes[r * (WIDTH / 8) + w] = word;
        }
    }
    let groups = WIDTH / 32;
    let scales: Vec<f32> = vec![1.0; VOCAB * groups];
    let biases: Vec<f32> = (0..VOCAB)
        .flat_map(|r| vec![(r / 16 * 16) as f32; groups])
        .collect();
    let ids: Vec<i32> = (0..ROWS * HEADS).map(|k| ((k * 7 + 3) % VOCAB) as i32).collect();

    let got = fire_embed_concat(
        &device,
        &pipelines,
        &handles,
        &codes,
        &scales,
        &biases,
        &ids,
        VOCAB as u32,
        ROWS,
        HEADS,
        WIDTH,
    );

    for row in 0..ROWS {
        for h in 0..HEADS {
            let id = ids[row * HEADS + h] as usize;
            let want = id as f32;
            for k in 0..WIDTH {
                let at = (row * HEADS + h) * WIDTH + k;
                assert_eq!(
                    got[at], want,
                    "row {row} head {h} names table row {id}, whose every element is \
                     {want}, and the gather landed {}",
                    got[at]
                );
            }
        }
    }
    println!(
        "concatenating gather: {ROWS} rows x {HEADS} ids of {WIDTH}, every element the \
         row its id names"
    );
}

/// **AN ID THE TABLE CANNOT ANSWER GATHERS ZERO, AND READS NOTHING.**
///
/// The banked gather reads THREE planes off one id — the packed codes, the
/// group scales and the group biases — so an unguarded id past the table is
/// three reads off the end of a checkpoint-sized buffer, not one. This is the
/// guard `kernels-cuda`'s `embed_concat_mlxu4` carries (`if (id < 0 || id >=
/// vocab) { y[at] = 0; return; }`), and the entry both quantized embed points
/// on that plane fire, so ZERO is the whole plane's answer and not this
/// shader's local choice.
///
/// **WHY IT IS WORTH A GATE THOUGH NO LIVE STREAM PRODUCES SUCH AN ID.** The
/// PLE hasher emits `mixed % primes + offsets` and a tokenizer bounds its own
/// vocabulary, so today the ids are structurally in range. The gate is not
/// about today's ids: it is about the two planes answering a corrupted or
/// hostile stream the SAME way, which is a property no in-range test can see.
///
/// The four poisoned ids are the four shapes of wrong: negative, one past the
/// last row, far past it, and `i32::MIN` — whose negation overflows, so a
/// guard written as `abs(id) >= vocab` would let it through.
#[test]
fn an_id_the_table_cannot_answer_gathers_zero_and_reads_nothing() {
    let _serial = serialized();
    let Some((device, pipelines, handles)) = device_or_skip("the gather's bounds guard") else {
        return;
    };
    const HEADS: usize = 16;
    const WIDTH: usize = 32;
    const VOCAB: usize = 64;

    // The same table the exact gather above reads: row `r` dequantizes to `r`.
    let mut codes = vec![0u32; VOCAB * WIDTH / 8];
    for r in 0..VOCAB {
        let nibble = (r % 16) as u32;
        let word = (0..8).fold(0u32, |acc, i| acc | (nibble << (4 * i)));
        for w in 0..WIDTH / 8 {
            codes[r * (WIDTH / 8) + w] = word;
        }
    }
    let groups = WIDTH / 32;
    let scales: Vec<f32> = vec![1.0; VOCAB * groups];
    let biases: Vec<f32> = (0..VOCAB)
        .flat_map(|r| vec![(r / 16 * 16) as f32; groups])
        .collect();

    // In-range everywhere, then four heads of the first row poisoned — so the
    // same fire proves both halves: the guard writes zero where it fires, and
    // the rows beside it are untouched.
    let poison: [(usize, i32); 4] = [(0, -1), (1, VOCAB as i32), (2, 1 << 20), (3, i32::MIN)];
    let mut ids: Vec<i32> = (0..ROWS * HEADS).map(|k| ((k * 7 + 3) % VOCAB) as i32).collect();
    for (h, bad) in poison {
        ids[h] = bad;
    }

    let got = fire_embed_concat(
        &device,
        &pipelines,
        &handles,
        &codes,
        &scales,
        &biases,
        &ids,
        VOCAB as u32,
        ROWS,
        HEADS,
        WIDTH,
    );

    for row in 0..ROWS {
        for h in 0..HEADS {
            let id = ids[row * HEADS + h];
            let addressable = id >= 0 && (id as usize) < VOCAB;
            let want = if addressable { id as f32 } else { 0.0 };
            for k in 0..WIDTH {
                let at = (row * HEADS + h) * WIDTH + k;
                assert_eq!(
                    got[at], want,
                    "row {row} head {h} names table row {id}, which the {VOCAB}-row table \
                     {}, and the gather landed {}",
                    if addressable { "answers" } else { "cannot answer" },
                    got[at]
                );
            }
        }
    }
    // The landing arrived at -1.0 everywhere, so a zero is a WRITE and not an
    // untouched buffer — the claim the poison was staged for.
    assert!(
        got.iter().all(|v| *v != -1.0),
        "every element of the landing was written"
    );
    println!(
        "banked gather bounds: ids {:?} out of a {VOCAB}-row table each gathered zero",
        poison.map(|(_, bad)| bad)
    );
}

// ---------------------------------------------------------------------------
// The fires.
// ---------------------------------------------------------------------------

fn fire_grouped_norm(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    x: &[f32],
    weight: &[f32],
) -> Vec<f32> {
    let x_b = staged(device, &encode_bf16(x));
    let w_b = staged(device, &encode_bf16(weight));
    let y = Buffer::zeroed(device, (x.len() * 2) as u64).expect("the output reserves");
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        norm::rmsnorm_grouped_plus_one(
            &sink,
            bf16(handles, &x_b, ROWS as u32, (M * H) as u32),
            bf16(handles, &w_b, 1, (M * H) as u32),
            H as u32,
            EPS,
            bf16(handles, &y, ROWS as u32, (M * H) as u32),
        )
        .expect("the grouped norm encodes");
    }
    frame.commit().expect("the grouped norm completes");
    decode_bf16(&read_back(&y, x.len() * 2))
}

fn fire_silu_scaled(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    x: &[f32],
    scale: f32,
) -> Vec<f32> {
    let x_b = staged(device, &encode_bf16(x));
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        norm::silu_scaled(&sink, scale, bf16(handles, &x_b, ROWS as u32, H as u32))
            .expect("the scaled silu encodes");
    }
    frame.commit().expect("the scaled silu completes");
    decode_bf16(&read_back(&x_b, x.len() * 2))
}

fn fire_mix(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    gates: &[f32],
    normed: &[f32],
) -> Vec<f32> {
    let g_b = staged(device, &encode_bf16(gates));
    let n_b = staged(device, &encode_bf16(normed));
    let y = Buffer::zeroed(device, (ROWS * H * 2) as u64).expect("the output reserves");
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        hc::mix(
            &sink,
            bf16(handles, &g_b, ROWS as u32, (M * H) as u32),
            bf16(handles, &n_b, ROWS as u32, (M * H) as u32),
            M as u32,
            bf16(handles, &y, ROWS as u32, H as u32),
        )
        .expect("the mix encodes");
    }
    frame.commit().expect("the mix completes");
    decode_bf16(&read_back(&y, ROWS * H * 2))
}

fn fire_inject(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    o: &[f32],
    gates: &[f32],
    hyper: &[f32],
) -> Vec<f32> {
    let o_b = staged(device, &encode_bf16(o));
    let g_b = staged(device, &encode_bf16(gates));
    let h_b = staged(device, &encode_bf16(hyper));
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        hc::inject(
            &sink,
            bf16(handles, &o_b, ROWS as u32, H as u32),
            bf16(handles, &g_b, ROWS as u32, M as u32),
            M as u32,
            bf16(handles, &h_b, ROWS as u32, (M * H) as u32),
        )
        .expect("the injection encodes");
    }
    frame.commit().expect("the injection completes");
    decode_bf16(&read_back(&h_b, hyper.len() * 2))
}

fn fire_ple_gate(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    key: &[f32],
    query: &[f32],
    value: &[f32],
) -> Vec<f32> {
    let k_b = staged(device, &encode_bf16(key));
    let q_b = staged(device, &encode_bf16(query));
    let v_b = staged(device, &encode_bf16(value));
    let y = Buffer::zeroed(device, (key.len() * 2) as u64).expect("the output reserves");
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        hc::ple_gate(
            &sink,
            bf16(handles, &k_b, ROWS as u32, (M * H) as u32),
            bf16(handles, &q_b, ROWS as u32, (M * H) as u32),
            bf16(handles, &v_b, ROWS as u32, H as u32),
            M as u32,
            bf16(handles, &y, ROWS as u32, (M * H) as u32),
        )
        .expect("the gate encodes");
    }
    frame.commit().expect("the gate completes");
    decode_bf16(&read_back(&y, key.len() * 2))
}

#[allow(clippy::too_many_arguments)]
fn fire_embed_concat(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    codes: &[u32],
    scales: &[f32],
    biases: &[f32],
    ids: &[i32],
    vocab: u32,
    rows: usize,
    heads: usize,
    width: usize,
) -> Vec<f32> {
    let c_b = staged(device, &encode_u32(codes));
    let s_b = staged(device, &encode_bf16(scales));
    let b_b = staged(device, &encode_bf16(biases));
    let i_b = staged(device, &encode_i32(ids));
    let out = rows * heads * width;
    // **NOT ZEROED — POISONED.** The claim under test one test down is that
    // the gather WRITES zero for an id it cannot address, and a landing that
    // arrived zero would pass that claim without the kernel writing anything
    // at all. Every element starts at a value no table row can produce.
    let mut y = Buffer::zeroed(device, (out * 2) as u64).expect("the output reserves");
    y.write(0, &encode_bf16(&vec![-1.0f32; out])).expect("the poison lands");
    // The bank's own row count is the table that was staged; `vocab` is what
    // the OP states, and the two are the same everywhere but a test that means
    // to hand an id neither can answer.
    let rows_of = (scales.len() * 32 / width) as u32;
    let frame = device.frame().expect("a command buffer opens");
    {
        let sink = Sink::new(device, &frame, pipelines, handles);
        kernels_metal::layout::embed_concat_mb_4bit(
            &sink,
            Tensor::new(bind(handles, &i_b), rows as u32, heads as u32, Dtype::I32),
            Bank {
                codes: Tensor::new(bind(handles, &c_b), rows_of, (width / 8) as u32, Dtype::U32),
                scales: Tensor::new(bind(handles, &s_b), rows_of, (width / 32) as u32, Dtype::Bf16),
                biases: Some(Tensor::new(
                    bind(handles, &b_b),
                    rows_of,
                    (width / 32) as u32,
                    Dtype::Bf16,
                )),
                group: 32,
                bits: 4,
            },
            vocab,
            bf16(handles, &y, rows as u32, (heads * width) as u32),
        )
        .expect("the concatenating gather encodes");
    }
    frame.commit().expect("the concatenating gather completes");
    decode_bf16(&read_back(&y, out * 2))
}

// ---------------------------------------------------------------------------
// Host staging — `ple_conv_on_device`'s, kept local for its reason: a test
// fixture shared between files is a fixture neither file can change.
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let x = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        (x - 0.5) * 2.0
    }

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
    values.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()).collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32_of(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
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

fn bind(handles: &Handles, buffer: &Buffer) -> u32 {
    handles
        .bind(buffer, 0, buffer.bytes())
        .expect("the handle table has a row")
}

fn bf16(handles: &Handles, buffer: &Buffer, rows: u32, width: u32) -> Tensor {
    Tensor::new(bind(handles, buffer), rows, width, Dtype::Bf16)
}

fn read_back(buffer: &Buffer, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    buffer.read(0, &mut out).expect("the bytes come back");
    out
}
