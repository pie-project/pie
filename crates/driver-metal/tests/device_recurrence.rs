//! The six recurrent points, on the GPU, for the first time.
//!
//! `ssm/causal_conv1d.metal`, `ssm/gated_delta.metal` and `ssm/kda.metal` are
//! `ssm.causal_conv1d`, `ssm.causal_conv1d_chunked`, `ssm.gated_delta`,
//! `ssm.gated_delta_chunked`, `ssm.kda_step` and `ssm.kda_chunked` -- six of
//! the fifty-one points this plane claims, all six written last week, and
//! none of them had ever produced a number. `device_gdn` measured the FUSED
//! gated-delta core (`gdn_core`, `gdn_prep`); these three files are the
//! separate family the declaration floor actually names, and nothing had
//! reached them.
//!
//! # Two references, and the second one is the interesting one
//!
//! Every point here gets a CPU model, written from the shader body. That
//! catches an `exp` on the wrong side of a negation and a `beta` read out of
//! the wrong half of a packed row, and it is the whole of what a model can do.
//!
//! What it CANNOT do is settle the claim these three files are built around:
//! that the step and the window are the same arithmetic, written twice
//! because the token loop is the only difference. `gated_delta` and
//! `gated_delta_chunked` are four hundred lines that differ in where a
//! barrier sits; `kda_step` and `kda_chunked` likewise. So the second
//! reference for each pair is THE OTHER ONE: the window is fired over three
//! tokens of one request, the step is fired three times against a state that
//! carries between fires, and the two must land the same output rows and the
//! same recurrent slab.
//!
//! That comparison is BIT-EXACT and it is asserted as one. There is no
//! reduction-order argument available to soften it -- the two bodies stage
//! the same threadgroup arrays with the same lanes in the same order, and the
//! only thing between them is a `for`. If they ever disagree by a bit, one of
//! them has a different contract, which is precisely the failure
//! `sdpa_paged_mma` had.
//!
//! # The GQA repeat is in the fixture on purpose
//!
//! `gated_delta` indexes its q and k rows by `hv / (v_heads / k_heads)`. That
//! expression equals `hv` exactly when `Hv == Hk`, which is true of every
//! checkpoint anyone here has run and false of `Qwen3.6-27B`. `device_gdn`
//! records what the difference costs -- a v-head past the sixteenth reading
//! its q from inside the K and V regions of the same row, in bounds, finite
//! and wrong -- so this fixture is `Hk = 2, Hv = 4`, and replacing the
//! expression with `hv` is one of the mutations.
//!
//! # The window that is shorter than the convolution
//!
//! `causal_conv1d_chunked` persists the trailing `K` rows of
//! `[slab window | x window]`, and when the window is shorter than `K` some
//! of those rows come back off the SLAB rather than out of `x`. The CSR here
//! is `{0, 1, 4, 8}`: a one-token window against four taps, a three-token
//! window, and a four-token window that exactly fills the slab. The first is
//! the case the `src < 0` branch exists for.
//!
//! # The seat table is a permutation
//!
//! Every fixture routes its requests through `slots` in an order that is not
//! the identity, because `slot = row` is the same expression as
//! `slot = slots[row]` for every fixture where it is not. The conv pool has
//! FOUR seats and three requests sit in `{2, 0, 3}`, so seat 1 is one no fire
//! names -- and it is checked, because `new_conv_state` is seeded with
//! `conv_state` the way `Pool::carry_forward` leaves it and a kernel writing
//! outside its seat would move a slab the model says came back untouched.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::{Arg, Rig};

const FILE_CONV: &str = "ssm/causal_conv1d.metal";
const FILE_DELTA: &str = "ssm/gated_delta.metal";
const FILE_KDA: &str = "ssm/kda.metal";

/// The conv's channel count and tap count, and the seats its three requests
/// sit in.
const C: usize = 20;
const KC: usize = 4;
const SLOTS: usize = 4;
const CONV_ROWS: usize = 3;
const CONV_SEATS: [u32; CONV_ROWS] = [2, 0, 3];

/// The prefill CSR: a window shorter than the convolution, one longer, and
/// one exactly as long.
const CONV_INDPTR: [i32; CONV_ROWS + 1] = [0, 1, 4, 8];
const CONV_TOKENS: usize = 8;

/// Gated delta: two key heads feeding four value heads, so `rep = 2`.
const HK: usize = 2;
const HV: usize = 4;
const DK: usize = 24;
const DV: usize = 6;
const QKV_WIDTH: usize = 2 * HK * DK + HV * DV;

/// KDA: three heads of eight, so the l2 the kernel takes over the whole
/// `H * D` plane is not the same number as one taken per head.
const H: usize = 3;
const D: usize = 8;
const PLANE: usize = H * D;
const NORM_EPS: f32 = 1e-6;

/// The tokens a window carries, and the seat they all sit in.
const STEPS: usize = 3;
const SEAT: u32 = 2;

/// The threadgroup both scans launch with.
const SCAN: u32 = 128;

const POISON: f32 = -99.0;

/// Relative to the widest element of the slab being compared.
///
/// MEASURED, and the first number written here was not. `2^-14` was reasoned
/// from "a recurrence accumulates, so give it room", and what this device
/// actually delivers over these fixtures is `2.04e-7` on the gated-delta
/// rows and `6.9e-8` on KDA's -- one and a half f32 ulps, which is a
/// `metal::rsqrt` that is not `precise::rsqrt` and nothing else. The
/// reasoned bound was three hundred times the truth, and a bound three
/// hundred times the truth accepts a kernel that is three hundred times
/// wrong.
///
/// `2^-22` is what the measurement says, and [`plane::tolerance_holds`] is
/// what keeps it saying it: the widest comparison in this file takes 0.86 of
/// it and the narrowest 0.29, so a hand that widens this to quiet a failure
/// trips the floor instead of getting away with it.
const SLAB_BOUND: f32 = 1.0 / 4_194_304.0;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_decode_convolution_reads_its_taps_off_the_slab_and_shifts_them_down() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `ssm.causal_conv1d` was not fired");
        return;
    };
    let fx = Conv::of();
    let (y, state) = fx.step(&rig, plane::kernels_dir().as_path());
    let (want_y, want_state) = fx.step_model();

    exact(&y, &want_y, "ssm.causal_conv1d, the token it lands");
    copied(&state, &want_state, "ssm.causal_conv1d, the slab it shifts");

    // The live window, off by one. Row 0 of the slab is where the shift's
    // tail goes and is never a tap; reading rows `0 .. K-2` instead of
    // `1 .. K-1` is a convolution over the same weights and the wrong taps.
    fx.step_bites(
        &rig,
        "conv_state[slab + (k + 1) * chans + col] * float(weight[tap0 + k])",
        "conv_state[slab + k * chans + col] * float(weight[tap0 + k])",
    );

    // The shift, dropped. The slab comes back holding what it held, which is
    // a request whose context stops advancing and whose output stays
    // plausible for exactly as long as nobody looks.
    fx.step_bites(
        &rig,
        "new_conv_state[slab + k * chans + col] =\n        conv_state[slab + (k + 1) * chans + col];",
        "new_conv_state[slab + k * chans + col] = conv_state[slab + k * chans + col];",
    );

    // The seat, taken as the row. `CONV_SEATS` is a permutation, so this
    // reads and writes three slabs that belong to other requests.
    fx.step_bites(
        &rig,
        "const size_t slab = size_t(slots[r]) * taps * chans;",
        "const size_t slab = size_t(r) * taps * chans;",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_prefill_convolution_reaches_back_into_the_slab_for_a_short_window() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `ssm.causal_conv1d_chunked` was not fired");
        return;
    };
    let fx = Conv::of();
    let (y, state) = fx.chunk(&rig, plane::kernels_dir().as_path());
    let (want_y, want_state) = fx.chunk_model();

    exact(
        &y,
        &want_y,
        "ssm.causal_conv1d_chunked, the window it lands",
    );
    copied(
        &state,
        &want_state,
        "ssm.causal_conv1d_chunked, the slab it leaves",
    );

    // The causal window, shifted one token later: `t - (K-1) + k` is the tap
    // that ends AT `t`, and `t - K + k` is the one that ends before it.
    fx.chunk_bites(
        &rig,
        "const int src = t - (width - 1) + k;",
        "const int src = t - width + k;",
    );

    // The trailing state, taken one row late. The slab a follow-up step
    // resumes from is the last `K` rows of the window; taking `K` rows
    // ending one past the end reads a token that does not exist yet.
    fx.chunk_bites(
        &rig,
        "    const int src = span - width + s;",
        "    const int src = span - width + s + 1;",
    );

    // The `src < 0` reach-back, refused. Request 0's window is ONE token
    // against four taps, so three of its four trailing rows come off the
    // slab; a kernel that only ever read `x` would answer it out of memory
    // it has no reason to be reading.
    fx.chunk_bites(
        &rig,
        "        ? conv_state[slab + size_t(width + src) * chans + col]\n        : float(x[size_t(begin + src) * chans + col]);\n  }\n}",
        "        ? 0.0f\n        : float(x[size_t(begin + src) * chans + col]);\n  }\n}",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_gated_delta_step_normalises_its_key_head_and_advances_one_cell() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `ssm.gated_delta` was not fired");
        return;
    };
    let fx = Delta::of();
    let (y, state) = fx.walk(&rig, plane::kernels_dir().as_path());
    let (want_y, want_state) = fx.model();

    near(&y, &want_y, "ssm.gated_delta, the rows it lands");
    near(&state, &want_state, "ssm.gated_delta, the slab it advances");

    // THE GQA REPEAT. `hv` and `hv / rep` are the same expression at
    // `Hv == Hk`, which is every checkpoint anyone here has run.
    fx.bites(
        &rig,
        "const int hk = hv / (v_heads / k_heads);\n\n  const size_t keys",
        "const int hk = hv;\n\n  const size_t keys",
    );

    // The decay's exponential, dropped. `gates` carries `g_log` and the
    // kernel is what exponentiates it, so a body reading the log directly
    // decays by a number that is usually negative.
    fx.bites(
        &rig,
        "const float decay = metal::exp(gates[fused]);\n  const float beta = gates[fused + size_t(v_heads)];",
        "const float decay = gates[fused];\n  const float beta = gates[fused + size_t(v_heads)];",
    );

    // The `[g_log | beta]` cut, taken at the neighbour rather than at the
    // half. Both halves are `Hv` wide, so `fused + 1` is another head's
    // decay read as this head's beta.
    fx.bites(
        &rig,
        "const float beta = gates[fused + size_t(v_heads)];",
        "const float beta = gates[fused + 1];",
    );

    // The decay applied after the delta instead of before it -- which is to
    // say not applied to the memory the delta is measured against.
    fx.bites(
        &rig,
        "      const float s = cell[i] * decay;\n      cell[i] = s;\n      kv_mem += s * sk[i];",
        "      const float s = cell[i];\n      cell[i] = s * decay;\n      kv_mem += s * sk[i];",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_gated_delta_window_is_the_step_walked_token_by_token() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the two gated-delta roads were not compared");
        return;
    };
    let fx = Delta::of();
    let root = plane::kernels_dir();
    let (stepped, step_state) = fx.walk(&rig, root.as_path());
    let (windowed, window_state) = fx.window(&rig, root.as_path());

    assert_eq!(
        stepped, windowed,
        "`gated_delta_chunked` over three tokens of one request is three \
         `gated_delta` fires against a state that carries, and the two \
         answered different rows"
    );
    assert_eq!(
        step_state, window_state,
        "the two gated-delta roads left different recurrent slabs"
    );
    plane::measured(
        "ssm.gated_delta_chunked against ssm.gated_delta",
        "bit-exact over three tokens, four value heads and a whole slab",
    );

    // The window's token loop, taken as one token. Three fires of the step
    // is what this is being compared against, so a window that only ever ran
    // its first token would answer the first row and stop.
    let (want_y, _) = fx.model();
    fx.window_bites(
        &rig,
        &want_y,
        "  for (int t = begin; t < end; ++t) {\n    const size_t row = size_t(t) * pitch;",
        "  for (int t = begin; t < begin + 1; ++t) {\n    const size_t row = size_t(t) * pitch;",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_kda_step_takes_its_l2_over_the_whole_plane_and_decays_per_channel() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `ssm.kda_step` was not fired");
        return;
    };
    let fx = Kda::of();
    let (y, state) = fx.walk(&rig, plane::kernels_dir().as_path());
    let (want_y, want_state) = fx.model();

    near(&y, &want_y, "ssm.kda_step, the rows it lands");
    near(&state, &want_state, "ssm.kda_step, the slab it advances");

    // THE NORM IS OVER THE WHOLE `H * D` PLANE and `kda.metal`'s header says
    // so at length: every published KDA normalises per head instead, the two
    // differ, and one of them is wrong. This kernel reproduces the cuda body
    // deliberately, so the seam stays in one place -- and the mutation is
    // what pins which side of it this file is on.
    fx.bites(
        &rig,
        "  for (int i = tid; i < plane; i += WIDTH) {\n    const float qv",
        "  for (int i = tid; i < d; i += WIDTH) {\n    const float qv",
    );

    // The decay's sign. `exp(-alpha * softplus)` is a number in `(0, 1]` and
    // `exp(alpha * softplus)` is one in `[1, inf)`, so the recurrence stops
    // forgetting and starts growing.
    fx.bites(
        &rig,
        "sg[i] = metal::exp(-alpha * sp);",
        "sg[i] = metal::exp(alpha * sp);",
    );

    // The beta projection's row. `b` is `[N, H]` -- one number per head per
    // token -- so dropping the head term gives every head the first one's.
    fx.bites(
        &rig,
        "float(b[size_t(n) * size_t(heads) + size_t(h)])",
        "float(b[size_t(n) * size_t(heads)])",
    );

    // softplus's guard, taken at the wrong end. `z > 20 ? z : log(1+exp(z))`
    // is the spelling every kernel in this family shares, and past twenty the
    // two expressions agree to far inside f32 -- so moving the guard to zero
    // is a real change only where `z` is small, which is where softplus is
    // not the identity. Every seventh channel of this fixture is biased past
    // twenty so that the branch itself is reached; the rest are what this
    // mutation moves.
    fx.bites(
        &rig,
        "const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::exp(z));\n    sg[i]",
        "const float sp = (z > 0.0f) ? z : metal::log(1.0f + metal::exp(z));\n    sg[i]",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_kda_window_is_the_step_walked_token_by_token() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the two KDA roads were not compared");
        return;
    };
    let fx = Kda::of();
    let root = plane::kernels_dir();
    let (stepped, step_state) = fx.walk(&rig, root.as_path());
    let (windowed, window_state) = fx.window(&rig, root.as_path());

    assert_eq!(
        stepped, windowed,
        "`kda_chunked` over three tokens of one request is three `kda_step` \
         fires against a state that carries, and the two answered different \
         rows"
    );
    assert_eq!(
        step_state, window_state,
        "the two KDA roads left different recurrent slabs"
    );
    plane::measured(
        "ssm.kda_chunked against ssm.kda_step",
        "bit-exact over three tokens, three heads and a whole slab",
    );

    let (want_y, _) = fx.model();
    fx.window_bites(
        &rig,
        &want_y,
        "  for (int t = begin; t < end; ++t) {\n    const size_t row = size_t(t) * 3 * wide;",
        "  for (int t = begin; t < begin + 1; ++t) {\n    const size_t row = size_t(t) * 3 * wide;",
    );
}

// ── the convolution ─────────────────────────────────────────────────────────

/// The depthwise convolution's operands, at values a bf16 buffer holds
/// exactly.
struct Conv {
    x: Vec<f32>,
    weight: Vec<f32>,
    slab: Vec<f32>,
}

impl Conv {
    fn of() -> Self {
        Self {
            x: bf16_spread(CONV_TOKENS * C, 3, 1.0),
            weight: bf16_spread(C * KC, 7, 0.8),
            slab: bf16_spread(SLOTS * KC * C, 11, 1.0),
        }
    }

    /// `y[r, c] = silu(sum_k W[c, k] * tap)` over the decode's window, and
    /// the slab shifted down one row with the arriving column at the end.
    fn step_model(&self) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0; CONV_ROWS * C];
        let mut out = self.slab.clone();
        for r in 0..CONV_ROWS {
            let slot = CONV_SEATS[r] as usize;
            for c in 0..C {
                let fresh = self.x[r * C + c];
                let mut acc = 0.0f32;
                for k in 0..KC - 1 {
                    acc += self.slab[(slot * KC + k + 1) * C + c] * self.weight[c * KC + k];
                }
                acc += fresh * self.weight[c * KC + KC - 1];
                y[r * C + c] = plane::narrowed(acc / (1.0 + plane::exp32(-acc)));
                for k in 0..KC - 1 {
                    out[(slot * KC + k) * C + c] = self.slab[(slot * KC + k + 1) * C + c];
                }
                out[(slot * KC + KC - 1) * C + c] = fresh;
            }
        }
        (y, out)
    }

    /// The same arithmetic over a CSR window, with the taps before the window
    /// read off the slab and the trailing `K` rows persisted back.
    fn chunk_model(&self) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0; CONV_TOKENS * C];
        let mut out = self.slab.clone();
        for r in 0..CONV_ROWS {
            let (begin, end) = (CONV_INDPTR[r] as usize, CONV_INDPTR[r + 1] as usize);
            if end <= begin {
                continue;
            }
            let span = end - begin;
            let slot = CONV_SEATS[r] as usize;
            for c in 0..C {
                for t in 0..span {
                    let mut acc = 0.0f32;
                    for k in 0..KC {
                        let src = t as isize - (KC as isize - 1) + k as isize;
                        let tap = if src < 0 {
                            self.slab[(slot * KC + (KC as isize + src) as usize) * C + c]
                        } else {
                            self.x[(begin + src as usize) * C + c]
                        };
                        acc += tap * self.weight[c * KC + k];
                    }
                    y[(begin + t) * C + c] = plane::narrowed(acc / (1.0 + plane::exp32(-acc)));
                }
                for s in 0..KC {
                    let src = span as isize - KC as isize + s as isize;
                    out[(slot * KC + s) * C + c] = if src < 0 {
                        self.slab[(slot * KC + (KC as isize + src) as usize) * C + c]
                    } else {
                        self.x[(begin + src as usize) * C + c]
                    };
                }
            }
        }
        (y, out)
    }

    fn step(&self, rig: &Rig, root: &std::path::Path) -> (Vec<f32>, Vec<f32>) {
        let x = plane::alloc_bf16(&rig.context, &self.x[..CONV_ROWS * C], "x");
        let weight = plane::alloc_bf16(&rig.context, &self.weight, "conv weight");
        let state = plane::alloc_f32(&rig.context, &self.slab, "conv_state");
        // Seeded with what it reads, which is `Pool::carry_forward`'s
        // invariant: a slot the fire does not name keeps what it had.
        let fresh = plane::alloc_f32(&rig.context, &self.slab, "new_conv_state");
        let seats = plane::alloc_u32(&rig.context, &CONV_SEATS, "slots");
        let y = plane::alloc_bf16(&rig.context, &vec![POISON; CONV_ROWS * C], "y");
        plane::fire(
            rig,
            root,
            FILE_CONV,
            "causal_conv1d_bfloat16",
            [C as u32, CONV_ROWS as u32, 1],
            [C.min(256) as u32, 1, 1],
            &[
                Arg::Buf(&x),
                Arg::Buf(&weight),
                Arg::Buf(&state),
                Arg::Buf(&fresh),
                Arg::Buf(&seats),
                Arg::Buf(&y),
                Arg::I32(C as i32),
                Arg::I32(KC as i32),
            ],
        );
        (
            plane::read_bf16(&y, CONV_ROWS * C),
            plane::read_f32(&fresh, SLOTS * KC * C),
        )
    }

    fn chunk(&self, rig: &Rig, root: &std::path::Path) -> (Vec<f32>, Vec<f32>) {
        // One seat per TOKEN, which is what `bind::tables` stages: every
        // token of a request repeats its request's seat.
        let mut per_token = vec![0u32; CONV_TOKENS];
        for r in 0..CONV_ROWS {
            let window = CONV_INDPTR[r] as usize..CONV_INDPTR[r + 1] as usize;
            per_token[window].fill(CONV_SEATS[r]);
        }
        let x = plane::alloc_bf16(&rig.context, &self.x, "x");
        let indptr = plane::alloc_i32(&rig.context, &CONV_INDPTR, "indptr");
        let weight = plane::alloc_bf16(&rig.context, &self.weight, "conv weight");
        let state = plane::alloc_f32(&rig.context, &self.slab, "conv_state");
        let fresh = plane::alloc_f32(&rig.context, &self.slab, "new_conv_state");
        let seats = plane::alloc_u32(&rig.context, &per_token, "slots");
        let y = plane::alloc_bf16(&rig.context, &vec![POISON; CONV_TOKENS * C], "y");
        plane::fire(
            rig,
            root,
            FILE_CONV,
            "causal_conv1d_chunked_bfloat16",
            [C as u32, CONV_ROWS as u32, 1],
            [C.min(256) as u32, 1, 1],
            &[
                Arg::Buf(&x),
                Arg::Buf(&indptr),
                Arg::Buf(&weight),
                Arg::Buf(&state),
                Arg::Buf(&fresh),
                Arg::Buf(&seats),
                Arg::Buf(&y),
                Arg::I32(C as i32),
                Arg::I32(KC as i32),
            ],
        );
        (
            plane::read_bf16(&y, CONV_TOKENS * C),
            plane::read_f32(&fresh, SLOTS * KC * C),
        )
    }

    fn step_bites(&self, rig: &Rig, from: &str, to: &str) {
        let root = plane::mutant(FILE_CONV, from, to);
        let (y, state) = self.step(rig, root.path());
        let (want_y, want_state) = self.step_model();
        bites(
            &[&y, &state],
            &[&want_y, &want_state],
            "causal_conv1d",
            from,
            to,
        );
    }

    fn chunk_bites(&self, rig: &Rig, from: &str, to: &str) {
        let root = plane::mutant(FILE_CONV, from, to);
        let (y, state) = self.chunk(rig, root.path());
        let (want_y, want_state) = self.chunk_model();
        bites(
            &[&y, &state],
            &[&want_y, &want_state],
            "causal_conv1d_chunked",
            from,
            to,
        );
    }
}

// ── gated delta ─────────────────────────────────────────────────────────────

/// One request's three tokens, its gates, and the slab they advance.
struct Delta {
    qkv: Vec<f32>,
    gates: Vec<f32>,
    slab: Vec<f32>,
}

impl Delta {
    fn of() -> Self {
        // `g_log` is negative and `beta` is a logit, so the packed row is
        // built rather than drawn: a decay of `exp(g_log)` drawn from the
        // same spread as a beta would sometimes be greater than one.
        let mut gates = vec![0.0; STEPS * 2 * HV];
        for t in 0..STEPS {
            for h in 0..HV {
                gates[t * 2 * HV + h] = -0.05 * ((t * 3 + h * 5) % 7 + 1) as f32;
                gates[t * 2 * HV + HV + h] = 0.3 * ((t * 5 + h * 3) % 9) as f32 - 1.0;
            }
        }
        Self {
            qkv: bf16_spread(STEPS * QKV_WIDTH, 13, 1.0),
            gates,
            slab: bf16_spread(SLOTS * HV * DV * DK, 17, 0.5),
        }
    }

    /// `gated_delta.metal`'s body, in Rust, over the three tokens in order.
    fn model(&self) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0; STEPS * HV * DV];
        let mut state = self.slab.clone();
        let scale = 1.0 / (DK as f32).sqrt();
        let keys = HK * DK;
        for t in 0..STEPS {
            for hv in 0..HV {
                let hk = hv / (HV / HK);
                let row = t * QKV_WIDTH;
                let qbase = row + hk * DK;
                let kbase = qbase + keys;
                let vbase = row + 2 * keys + hv * DV;

                let (mut qsq, mut ksq) = (0.0f32, 0.0f32);
                for i in 0..DK {
                    qsq += self.qkv[qbase + i] * self.qkv[qbase + i];
                    ksq += self.qkv[kbase + i] * self.qkv[kbase + i];
                }
                let qinv = (qsq + 1e-6).sqrt().recip() * scale;
                let kinv = (ksq + 1e-6).sqrt().recip();
                let q: Vec<f32> = (0..DK).map(|i| self.qkv[qbase + i] * qinv).collect();
                let k: Vec<f32> = (0..DK).map(|i| self.qkv[kbase + i] * kinv).collect();

                let decay = plane::exp32(self.gates[t * 2 * HV + hv]);
                let beta = self.gates[t * 2 * HV + HV + hv];
                let base = ((SEAT as usize * HV + hv) * DV) * DK;
                for c in 0..DV {
                    let cell = &mut state[base + c * DK..base + c * DK + DK];
                    let mut mem = 0.0f32;
                    for (i, s) in cell.iter_mut().enumerate() {
                        *s *= decay;
                        mem += *s * k[i];
                    }
                    let delta = (self.qkv[vbase + c] - mem) * beta;
                    let mut acc = 0.0f32;
                    for (i, s) in cell.iter_mut().enumerate() {
                        *s += k[i] * delta;
                        acc += *s * q[i];
                    }
                    y[(t * HV + hv) * DV + c] = acc;
                }
            }
        }
        (y, state)
    }

    /// Three fires of the step against one slab, which is what a decode is.
    fn walk(&self, rig: &Rig, root: &std::path::Path) -> (Vec<f32>, Vec<f32>) {
        let state = plane::alloc_f32(&rig.context, &self.slab, "rstate");
        let mut y = Vec::with_capacity(STEPS * HV * DV);
        for t in 0..STEPS {
            let qkv = plane::alloc_bf16(
                &rig.context,
                &self.qkv[t * QKV_WIDTH..(t + 1) * QKV_WIDTH],
                "qkv",
            );
            let gates = plane::alloc_f32(
                &rig.context,
                &self.gates[t * 2 * HV..(t + 1) * 2 * HV],
                "gates",
            );
            let seats = plane::alloc_u32(&rig.context, &[SEAT], "slots");
            let row = plane::alloc_f32(&rig.context, &[POISON; HV * DV], "y");
            plane::fire(
                rig,
                root,
                FILE_DELTA,
                "gated_delta_bfloat16",
                [SCAN, HV as u32, 1],
                [SCAN, 1, 1],
                &[
                    Arg::Buf(&qkv),
                    Arg::Buf(&gates),
                    Arg::Buf(&state),
                    Arg::Buf(&seats),
                    Arg::Buf(&row),
                    Arg::I32(HK as i32),
                    Arg::I32(HV as i32),
                    Arg::I32(DK as i32),
                    Arg::I32(DV as i32),
                ],
            );
            y.extend(plane::read_f32(&row, HV * DV));
        }
        (y, plane::read_f32(&state, SLOTS * HV * DV * DK))
    }

    /// One fire of the window over all three.
    fn window(&self, rig: &Rig, root: &std::path::Path) -> (Vec<f32>, Vec<f32>) {
        let qkv = plane::alloc_bf16(&rig.context, &self.qkv, "qkv");
        let indptr = plane::alloc_i32(&rig.context, &[0, STEPS as i32], "indptr");
        let gates = plane::alloc_f32(&rig.context, &self.gates, "gates");
        let state = plane::alloc_f32(&rig.context, &self.slab, "rstate");
        let seats = plane::alloc_u32(&rig.context, &[SEAT; STEPS], "slots");
        let y = plane::alloc_f32(&rig.context, &vec![POISON; STEPS * HV * DV], "y");
        plane::fire(
            rig,
            root,
            FILE_DELTA,
            "gated_delta_chunked_bfloat16",
            [SCAN, HV as u32, 1],
            [SCAN, 1, 1],
            &[
                Arg::Buf(&qkv),
                Arg::Buf(&indptr),
                Arg::Buf(&gates),
                Arg::Buf(&state),
                Arg::Buf(&seats),
                Arg::Buf(&y),
                Arg::I32(HK as i32),
                Arg::I32(HV as i32),
                Arg::I32(DK as i32),
                Arg::I32(DV as i32),
            ],
        );
        (
            plane::read_f32(&y, STEPS * HV * DV),
            plane::read_f32(&state, SLOTS * HV * DV * DK),
        )
    }

    fn bites(&self, rig: &Rig, from: &str, to: &str) {
        let root = plane::mutant(FILE_DELTA, from, to);
        let (y, state) = self.walk(rig, root.path());
        let (want_y, want_state) = self.model();
        bites(
            &[&y, &state],
            &[&want_y, &want_state],
            "gated_delta",
            from,
            to,
        );
    }

    fn window_bites(&self, rig: &Rig, want_y: &[f32], from: &str, to: &str) {
        let root = plane::mutant(FILE_DELTA, from, to);
        let (y, _) = self.window(rig, root.path());
        bites(&[&y], &[want_y], "gated_delta_chunked", from, to);
    }
}

// ── kimi delta attention ────────────────────────────────────────────────────

/// One request's three tokens of `[q | k | v]`, its forget and beta
/// projections, and the slab they advance.
struct Kda {
    mixed: Vec<f32>,
    f: Vec<f32>,
    b: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    slab: Vec<f32>,
}

impl Kda {
    fn of() -> Self {
        Self {
            mixed: bf16_spread(STEPS * 3 * PLANE, 19, 1.0),
            f: bf16_spread(STEPS * PLANE, 23, 2.0),
            b: bf16_spread(STEPS * H, 29, 2.0),
            // Every seventh channel is biased past softplus's `z > 20`
            // guard, so the branch is taken rather than merely present.
            // The rest sit in `[-1, 1]`, where `log(1 + exp(z))` is the
            // expression that has to be evaluated.
            dt_bias: (0..PLANE)
                .map(|i| {
                    if i % 7 == 0 {
                        24.0
                    } else {
                        plane::narrowed((((i * 7 + 31 * 13) % 17) as f32 - 8.0) / 8.5)
                    }
                })
                .collect(),
            // `alpha = exp(a_log)` multiplies a softplus, so a positive
            // `a_log` here would decay the state to nothing in three tokens.
            a_log: (0..H).map(|h| -0.4 - 0.3 * h as f32).collect(),
            slab: bf16_spread(SLOTS * H * D * D, 37, 0.5),
        }
    }

    /// `kda.metal`'s body, in Rust, over the three tokens in order.
    fn model(&self) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0; STEPS * PLANE];
        let mut state = self.slab.clone();
        for t in 0..STEPS {
            let row = t * 3 * PLANE;
            // THE L2 IS OVER THE WHOLE PLANE, once per token, not per head.
            let mut qsq = 0.0f32;
            let mut ksq = 0.0f32;
            for i in 0..PLANE {
                qsq += self.mixed[row + i] * self.mixed[row + i];
                ksq += self.mixed[row + PLANE + i] * self.mixed[row + PLANE + i];
            }
            let qinv = (qsq + NORM_EPS).sqrt().recip();
            let kinv = (ksq + NORM_EPS).sqrt().recip();
            for h in 0..H {
                let head = h * D;
                let alpha = plane::exp32(self.a_log[h]);
                let q: Vec<f32> = (0..D).map(|i| self.mixed[row + head + i] * qinv).collect();
                let k: Vec<f32> = (0..D)
                    .map(|i| self.mixed[row + PLANE + head + i] * kinv)
                    .collect();
                let g: Vec<f32> = (0..D)
                    .map(|i| {
                        let z = self.f[t * PLANE + head + i] + self.dt_bias[head + i];
                        let sp = if z > 20.0 {
                            z
                        } else {
                            (1.0 + plane::exp32(z)).ln()
                        };
                        plane::exp32(-alpha * sp)
                    })
                    .collect();
                let beta = 1.0 / (1.0 + plane::exp32(-self.b[t * H + h]));
                let base = (SEAT as usize * H + h) * D * D;
                for vi in 0..D {
                    let cell = &mut state[base + vi * D..base + vi * D + D];
                    let mut mem = 0.0f32;
                    for (i, s) in cell.iter_mut().enumerate() {
                        *s *= g[i];
                        mem += *s * k[i];
                    }
                    let delta = (self.mixed[row + 2 * PLANE + head + vi] - mem) * beta;
                    let mut acc = 0.0f32;
                    for (i, s) in cell.iter_mut().enumerate() {
                        *s += k[i] * delta;
                        acc += *s * q[i];
                    }
                    y[t * PLANE + head + vi] = acc;
                }
            }
        }
        (y, state)
    }

    fn walk(&self, rig: &Rig, root: &std::path::Path) -> (Vec<f32>, Vec<f32>) {
        let dt_bias = plane::alloc_f32(&rig.context, &self.dt_bias, "dt_bias");
        let a_log = plane::alloc_f32(&rig.context, &self.a_log, "A_log");
        let state = plane::alloc_f32(&rig.context, &self.slab, "rstate");
        let mut y = Vec::with_capacity(STEPS * PLANE);
        for t in 0..STEPS {
            let mixed = plane::alloc_bf16(
                &rig.context,
                &self.mixed[t * 3 * PLANE..(t + 1) * 3 * PLANE],
                "mixed",
            );
            let f = plane::alloc_bf16(&rig.context, &self.f[t * PLANE..(t + 1) * PLANE], "f");
            let b = plane::alloc_bf16(&rig.context, &self.b[t * H..(t + 1) * H], "b");
            let seats = plane::alloc_u32(&rig.context, &[SEAT], "slots");
            let row = plane::alloc_f32(&rig.context, &[POISON; PLANE], "y");
            plane::fire(
                rig,
                root,
                FILE_KDA,
                "kda_step_bfloat16",
                [SCAN, H as u32, 1],
                [SCAN, 1, 1],
                &[
                    Arg::Buf(&mixed),
                    Arg::Buf(&f),
                    Arg::Buf(&b),
                    Arg::Buf(&dt_bias),
                    Arg::Buf(&a_log),
                    Arg::Buf(&state),
                    Arg::Buf(&seats),
                    Arg::Buf(&row),
                    Arg::I32(H as i32),
                    Arg::I32(D as i32),
                    Arg::F32(NORM_EPS),
                ],
            );
            y.extend(plane::read_f32(&row, PLANE));
        }
        (y, plane::read_f32(&state, SLOTS * H * D * D))
    }

    fn window(&self, rig: &Rig, root: &std::path::Path) -> (Vec<f32>, Vec<f32>) {
        let mixed = plane::alloc_bf16(&rig.context, &self.mixed, "mixed");
        let indptr = plane::alloc_i32(&rig.context, &[0, STEPS as i32], "indptr");
        let f = plane::alloc_bf16(&rig.context, &self.f, "f");
        let b = plane::alloc_bf16(&rig.context, &self.b, "b");
        let dt_bias = plane::alloc_f32(&rig.context, &self.dt_bias, "dt_bias");
        let a_log = plane::alloc_f32(&rig.context, &self.a_log, "A_log");
        let state = plane::alloc_f32(&rig.context, &self.slab, "rstate");
        let seats = plane::alloc_u32(&rig.context, &[SEAT; STEPS], "slots");
        let y = plane::alloc_f32(&rig.context, &vec![POISON; STEPS * PLANE], "y");
        plane::fire(
            rig,
            root,
            FILE_KDA,
            "kda_chunked_bfloat16",
            [SCAN, H as u32, 1],
            [SCAN, 1, 1],
            &[
                Arg::Buf(&mixed),
                Arg::Buf(&indptr),
                Arg::Buf(&f),
                Arg::Buf(&b),
                Arg::Buf(&dt_bias),
                Arg::Buf(&a_log),
                Arg::Buf(&state),
                Arg::Buf(&seats),
                Arg::Buf(&y),
                Arg::I32(H as i32),
                Arg::I32(D as i32),
                Arg::F32(NORM_EPS),
            ],
        );
        (
            plane::read_f32(&y, STEPS * PLANE),
            plane::read_f32(&state, SLOTS * H * D * D),
        )
    }

    fn bites(&self, rig: &Rig, from: &str, to: &str) {
        let root = plane::mutant(FILE_KDA, from, to);
        let (y, state) = self.walk(rig, root.path());
        let (want_y, want_state) = self.model();
        bites(&[&y, &state], &[&want_y, &want_state], "kda_step", from, to);
    }

    fn window_bites(&self, rig: &Rig, want_y: &[f32], from: &str, to: &str) {
        let root = plane::mutant(FILE_KDA, from, to);
        let (y, _) = self.window(rig, root.path());
        bites(&[&y], &[want_y], "kda_chunked", from, to);
    }
}

// ── the comparisons ─────────────────────────────────────────────────────────

/// A slab a kernel only ever COPIED into has one right answer, and it is
/// the bits it copied. Nothing here rounds -- the shifted rows come out of
/// a float slab and the arriving column out of a bf16 plane -- so a bound
/// of any width would be a bound on a memcpy.
fn copied(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got, want, "{what} is a copy, and it did not copy");
    plane::measured(what, &format!("bit-exact over {} elements", got.len()));
}

/// A bf16 result of a widen-compute-round has one right answer.
fn exact(got: &[f32], want: &[f32], what: &str) {
    let (widest, at, inexact) = plane::ulp_spread(got, want);
    assert!(
        widest <= 1,
        "{what}: element {at} is {widest} bf16 steps from the model -- {} \
         against {}",
        got[at],
        want[at]
    );
    plane::measured(
        what,
        &format!(
            "{widest} bf16 steps at worst, {inexact} of {} elements inexact",
            got.len()
        ),
    );
}

/// An f32 result of a recurrence, against the widest element of its own slab.
fn near(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = plane::worst(got, want, scale);
    assert!(
        worst <= SLAB_BOUND,
        "{what}: element {} is {worst} of the slab's widest element away \
         from the model, past the {SLAB_BOUND} this device was measured at",
        plane::worst_at(got, want, scale)
    );
    plane::tolerance_holds(worst, SLAB_BOUND, what);
    plane::measured(
        what,
        &format!("worst {worst} against the slab bound {SLAB_BOUND}"),
    );
}

/// A SABOTAGED shader must miss at least one of the slabs it writes.
fn bites(got: &[&[f32]], want: &[&[f32]], symbol: &str, from: &str, to: &str) {
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| {
            let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            plane::worst(g, w, scale)
        })
        .fold(0.0, f32::max);
    assert!(
        worst > SLAB_BOUND,
        "replacing `{from}` with `{to}` left every slab within {worst} of its \
         own widest element, so the comparison above would not have caught it"
    );
    plane::measured(
        symbol,
        &format!("`{from}` -> `{to}`: worst {worst} against the slab bound {SLAB_BOUND}"),
    );
}

/// A draw a bf16 buffer holds exactly, whose period shares no factor with any
/// stride here.
fn bf16_spread(n: usize, seed: usize, gain: f32) -> Vec<f32> {
    (0..n)
        .map(|i| plane::narrowed((((i * 7 + seed * 13) % 17) as f32 - 8.0) / 8.5 * gain))
        .collect()
}
