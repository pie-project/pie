//! The five RoPE points, on the GPU, for the first time.
//!
//! `rope/neox.metal` answers `rope.full`, `rope.partial`, `rope.partial_q`,
//! `rope.partial_last` and `rope.yarn` out of four entry points, and every
//! one of them rotates IN PLACE: the kernel reads a pair of channels, turns
//! it, and writes it back over itself. Nothing had ever run.
//!
//! # What the reference is, and what it deliberately is not
//!
//! A CPU model in this file, with one seam: the angle is computed in f32
//! exactly as the shader computes it -- `exp2(-d * base)`, then
//! `(scale * position) * inv_freq` -- and only the sine and cosine are taken
//! in f64. That is on purpose. `theta` is a float on the device and there is
//! no version of this kernel where it is not, so a model that computed the
//! angle in double would be measuring the shader against arithmetic the
//! shader is not allowed to do, and the difference at position 4001 is larger
//! than everything else this file measures put together.
//!
//! What is left over the seam is exactly one question: **is Metal's
//! `fast::cos` the cosine at the angles a served context reaches?** Every
//! entry point here spells its rotation with `fast::cos` and `fast::sin`, and
//! a fast trigonometric function is a range reduction somebody decided was
//! good enough. At position 4001 and `inv_freq = 1` the argument is four
//! thousand radians, which is six hundred turns; the tests below fire that
//! position beside 0, 1 and 37 and report what each one cost.
//!
//! # Position zero is the identity, and it is checked as one
//!
//! `theta = 0` gives `cos = 1`, `sin = 0`, so the first token of every prompt
//! must come back BIT FOR BIT unchanged. That is the one row in this file
//! with no tolerance at all, and it is worth having because it is the only
//! assertion here that a wrong `inv_freq` cannot satisfy by accident.
//!
//! # The bound is the pair's own scale
//!
//! A rotation of `(x1, x2)` produces `x1 cos - x2 sin`, which cancels when
//! the two channels are close and the angle is near a right angle. Bounding
//! that relative to the ANSWER would report an enormous error for an element
//! that is correct to every bit a bf16 holds, so the bound is `2^-8` of
//! `|x1| + |x2|` -- one bf16 step of the pair the rotation was taken over.
//!
//! # The four pairings are four different pairings, and that is the point
//!
//! `rope.full` pairs `(i, i + rotary/2)` and divides the exponent by
//! `rotary/2`. `rope.partial` pairs `(i, i + head_dim/2)` and divides by
//! `head_dim` -- the gemma4 reading, verified against mlx_lm at head_dim=512
//! and rotary=128, where "the channels that move are [0,63] and [256,319],
//! not [0,127]". `rope.partial_last` moves the head's TAIL, at
//! `head_dim - rotary`, and pairs either halves or neighbours depending on a
//! flag. `rope.yarn` pairs over the whole head and interpolates its
//! frequencies along a ramp. Four addresses and four exponents, and a
//! mutation below swaps each one for its neighbour's.
//!
//! # The channels outside the rotation must not move
//!
//! `rope.partial` at `rotary = 24` of a 64-wide head turns twelve pairs and
//! leaves forty channels alone; `rope.partial_last` turns the last twenty-
//! four and leaves the first forty. Those untouched channels are compared
//! bit-for-bit, not to a tolerance, because a kernel that rotated them would
//! not be slightly wrong.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::{Arg, Rig};

const FILE: &str = "rope/neox.metal";

const HEAD_DIM: usize = 64;
const HEADS: usize = 3;

/// Zero, so the identity is checked; one and thirty-seven, which are the
/// angles a short prompt sees; and four thousand and one, which is six
/// hundred turns of the fastest channel and is where a range reduction
/// either holds or does not.
const POSITIONS: [i32; 4] = [0, 1, 37, 4001];
const ROWS: usize = POSITIONS.len();

/// Not a divisor of `HEAD_DIM`, so `rotary / 2 = 12` pairs leave forty
/// channels untouched and no stride coincidence hides a mis-pairing.
const ROTARY: usize = 24;

const THETA: f32 = 10000.0;

/// One bf16 step of the pair a rotation was taken over.
const PAIR_BOUND: f32 = 1.0 / 256.0;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_full_rotation_pairs_a_channel_with_the_one_half_a_head_away() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `rope.full` was not fired");
        return;
    };
    let base = THETA.log2();
    let want = model(HEAD_DIM, |i, half| {
        let d = i as f32 / half as f32;
        (i, i + half, (-d * base).exp2(), 1.0)
    });

    let scalars = [Arg::F32(1.0), Arg::F32(base), Arg::I32(HEAD_DIM as i32)];
    let got = turn(
        &rig,
        plane::kernels_dir().as_path(),
        "neox_mb_bfloat16",
        HEAD_DIM,
        &scalars,
    );
    want.agrees(&got, "rope.full");
    want.identity_at_position_zero(&got);

    // The exponent's divisor. `rope_neox_decode` derives BOTH the pair
    // offset and the frequency divisor from `grid.x`, and the whole
    // `neox_prop_*` family exists because gemma4 wants a different one --
    // so taking this one's from `head_dim` is the confusion those two
    // kernels are two kernels to avoid.
    want.bites(
        &rig,
        "neox_mb_bfloat16",
        HEAD_DIM,
        &scalars,
        "const float d = float(i) / float(pair_half);",
        "const float d = float(i) / float(head_dim);",
    );

    // The pairing, taken as neighbours. NEOX rotates halves and the
    // interleaved form rotates adjacent channels; `rope.full` refuses the
    // interleaved flag outright, so a body that paired that way would be
    // answering a rotation nothing on this plane can ask for.
    want.bites(
        &rig,
        "neox_mb_bfloat16",
        HEAD_DIM,
        &scalars,
        "rope_rotate_pair<T, false>(x, i1, i1 + size_t(pair_half), theta, 1.0f);",
        "rope_rotate_pair<T, false>(x, i1, i1 + 1, theta, 1.0f);",
    );

    // The rotation's own sign. `(x1 cos - x2 sin, x1 sin + x2 cos)` is a
    // turn; flipping the second row is a reflection, and it preserves every
    // norm a sanity check would look at.
    want.bites(
        &rig,
        "neox_mb_bfloat16",
        HEAD_DIM,
        &scalars,
        "const float y2 = x1 * sintheta + x2 * costheta;",
        "const float y2 = x1 * sintheta - x2 * costheta;",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_proportional_rotation_pairs_across_the_whole_head() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `rope.partial` was not fired");
        return;
    };
    let base = THETA.log2();
    let want = model(ROTARY, |i, _| {
        let d = 2.0 * i as f32 / HEAD_DIM as f32;
        (i, i + HEAD_DIM / 2, (-d * base).exp2(), 1.0)
    });

    let scalars = [Arg::F32(1.0), Arg::F32(base), Arg::I32(HEAD_DIM as i32)];
    let got = turn(
        &rig,
        plane::kernels_dir().as_path(),
        "neox_prop_mb_bfloat16",
        ROTARY,
        &scalars,
    );
    want.agrees(&got, "rope.partial / rope.partial_q");
    want.identity_at_position_zero(&got);

    // The `2 *`, dropped: the exponent then divides by `head_dim` where the
    // gemma4 reading divides by half of it, which is the same family of
    // frequencies at the wrong spacing.
    want.bites(
        &rig,
        "neox_prop_mb_bfloat16",
        ROTARY,
        &scalars,
        "float d = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);",
        "float d = static_cast<float>(i) / static_cast<float>(head_dim);",
    );

    // The pair's other half, taken at the ROTATED width rather than the
    // head's. That is `neox_mb`'s pairing under `neox_prop_mb`'s name, and
    // it is the exact confusion this file's header records having been
    // measured against mlx_lm.
    want.bites(
        &rig,
        "neox_prop_mb_bfloat16",
        ROTARY,
        &scalars,
        "const int n_head = int(grid.y);\n  const int half_hd = head_dim / 2;",
        "const int n_head = int(grid.y);\n  const int half_hd = int(grid.x);",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_tail_rotation_turns_the_end_of_the_head_either_way_round() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `rope.partial_last` was not fired");
        return;
    };
    let base = THETA.log2();
    let offset = HEAD_DIM - ROTARY;
    let halves = model(ROTARY, |i, half| {
        let d = 2.0 * i as f32 / (2 * half) as f32;
        (offset + i, offset + i + half, (-d * base).exp2(), 1.0)
    });
    let neighbours = model(ROTARY, |i, half| {
        let d = 2.0 * i as f32 / (2 * half) as f32;
        (offset + 2 * i, offset + 2 * i + 1, (-d * base).exp2(), 1.0)
    });

    let root = plane::kernels_dir();
    let split = [Arg::F32(base), Arg::I32(HEAD_DIM as i32), Arg::I32(0)];
    let paired = [Arg::F32(base), Arg::I32(HEAD_DIM as i32), Arg::I32(1)];

    let got = turn(
        &rig,
        root.as_path(),
        "neox_last_mb_bfloat16",
        ROTARY,
        &split,
    );
    halves.agrees(&got, "rope.partial_last, halves");
    halves.identity_at_position_zero(&got);

    let got = turn(
        &rig,
        root.as_path(),
        "neox_last_mb_bfloat16",
        ROTARY,
        &paired,
    );
    neighbours.agrees(&got, "rope.partial_last, interleaved");

    // The tail's offset, dropped: the rotation lands on the head's FRONT,
    // which is a perfectly ordinary partial RoPE and the wrong one.
    halves.bites(
        &rig,
        "neox_last_mb_bfloat16",
        ROTARY,
        &split,
        "const int offset = head_dim - rotary;",
        "const int offset = 0;",
    );

    // The interleave flag, read backwards. Both fires above are needed for
    // this: a file that only ever passed one value would find the two
    // pairings agree with whichever model it wrote.
    halves.bites(
        &rig,
        "neox_last_mb_bfloat16",
        ROTARY,
        &split,
        "const int i1 = interleaved != 0 ? row_base + 2 * i : row_base + i;\n  const int i2 = interleaved != 0 ? i1 + 1 : i1 + rope_half;",
        "const int i1 = interleaved == 0 ? row_base + 2 * i : row_base + i;\n  const int i2 = interleaved == 0 ? i1 + 1 : i1 + rope_half;",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_yarn_rotation_interpolates_along_its_ramp_and_scales_by_mscale() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `rope.yarn` was not fired");
        return;
    };
    let base = THETA.log2();
    // Chosen so the ramp is flat at both ends and moving in between: at
    // `i <= 6` the frequency is untouched, at `i >= 20` it is divided by the
    // whole factor, and the eleven channels between are the interpolation
    // this kernel exists for. `kernels_metal::rope::ramp_bounds` derives
    // these from beta_fast and beta_slow; what is under test here is the
    // shader, so they arrive as the numbers a statement would carry.
    const LOW: f32 = 6.0;
    const HIGH: f32 = 20.0;
    const FACTOR: f32 = 32.0;
    const MSCALE: f32 = 1.3466;

    let want = model(HEAD_DIM, |i, half| {
        let d = 2.0 * i as f32 / HEAD_DIM as f32;
        let ramp = ((i as f32 - LOW) / (HIGH - LOW)).clamp(0.0, 1.0);
        let freq = (-d * base).exp2() * ((1.0 - ramp) + ramp / FACTOR);
        (i, i + half, freq, MSCALE)
    });

    let scalars = [
        Arg::F32(base),
        Arg::I32(HEAD_DIM as i32),
        Arg::F32(FACTOR),
        Arg::F32(LOW),
        Arg::F32(HIGH),
        Arg::F32(MSCALE),
        Arg::I32(0),
    ];
    let got = turn(
        &rig,
        plane::kernels_dir().as_path(),
        "neox_yarn_mb_bfloat16",
        HEAD_DIM,
        &scalars,
    );
    want.agrees(&got, "rope.yarn");

    // Position zero is NOT the identity here: `mscale` multiplies the
    // rotation rather than riding inside it, so the first token comes back
    // scaled. That is the attention-temperature correction and it is the
    // reason this test cannot reuse the identity check above.
    let at_zero = &got[..HEADS * HEAD_DIM];
    let flat = at_zero
        .iter()
        .zip(&want.x[..HEADS * HEAD_DIM])
        .all(|(g, x)| plane::bf16_ulps(*g, x * MSCALE) <= 1);
    assert!(
        flat,
        "at position zero YaRN is `mscale` times the identity, and it was not"
    );

    // The factor, inverted. `ramp / factor` stretches the wavelength and
    // `ramp * factor` shortens it; both are smooth interpolations between
    // two frequency series and only one of them is YaRN.
    want.bites(
        &rig,
        "neox_yarn_mb_bfloat16",
        HEAD_DIM,
        &scalars,
        "base_freq * ((1.0f - ramp) + ramp / factor)",
        "base_freq * ((1.0f - ramp) + ramp * factor)",
    );

    // `mscale`, dropped from the cosine only. Halving the correction on one
    // of the two terms is not a rotation at all any more, and it is the
    // shape of every "I moved the scale into the other line" edit.
    want.bites(
        &rig,
        "neox_yarn_mb_bfloat16",
        HEAD_DIM,
        &scalars,
        "const float costheta = fast::cos(theta) * mscale;",
        "const float costheta = fast::cos(theta);",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_yarn_ramp_survives_a_degenerate_window() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: YaRN's degenerate ramp was not fired");
        return;
    };
    let base = THETA.log2();
    // `ramp_bounds` clamps `high_dim` up to `low_dim`, so the two CAN arrive
    // equal -- a checkpoint whose beta_fast and beta_slow land in the same
    // channel. `high - low` is then zero and the shader's guard replaces the
    // denominator with `1e-3`, which makes the ramp a step at `low` rather
    // than a division by zero. Nothing else in this file reaches that
    // branch, and a NaN in a rotation is a NaN in every subsequent token.
    const AT: f32 = 12.0;
    const FACTOR: f32 = 8.0;

    let want = model(HEAD_DIM, |i, half| {
        let d = 2.0 * i as f32 / HEAD_DIM as f32;
        let ramp = ((i as f32 - AT) / 1e-3).clamp(0.0, 1.0);
        let freq = (-d * base).exp2() * ((1.0 - ramp) + ramp / FACTOR);
        (i, i + half, freq, 1.0)
    });

    let got = turn(
        &rig,
        plane::kernels_dir().as_path(),
        "neox_yarn_mb_bfloat16",
        HEAD_DIM,
        &[
            Arg::F32(base),
            Arg::I32(HEAD_DIM as i32),
            Arg::F32(FACTOR),
            Arg::F32(AT),
            Arg::F32(AT),
            Arg::F32(1.0),
            Arg::I32(0),
        ],
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "a ramp whose window is one channel wide produced a non-finite rotation"
    );
    want.agrees(&got, "rope.yarn, low_dim == high_dim");
}

/// The rectangle a rotation is fired over, the answer, and the scale each
/// element's bound is taken against.
struct Turned {
    /// The input, which is also what the untouched channels must still hold.
    x: Vec<f32>,
    want: Vec<f32>,
    /// `|x1| + |x2|` for a rotated channel, and `|x|` for one left alone.
    scale: Vec<f32>,
    /// Which channels the rotation moved, so the rest can be compared to the
    /// bit rather than to a bound.
    turned: Vec<bool>,
}

/// Build the model for one variant.
///
/// `pairs(i, half)` answers the two channel indices within a head, the f32
/// `inv_freq` for that pair, and the gain the rotation is multiplied by --
/// which is 1 everywhere except YaRN's `mscale`. `half` is `rotary / 2`,
/// which is the launch's `grid.x` and therefore what every body derives its
/// own geometry from.
fn model(rotary: usize, pairs: impl Fn(usize, usize) -> (usize, usize, f32, f32)) -> Turned {
    let n = ROWS * HEADS * HEAD_DIM;
    let x: Vec<f32> = (0..n)
        .map(|i| plane::narrowed(((i * 11) % 37) as f32 * 0.06 - 1.1))
        .collect();
    let mut want = x.clone();
    let mut scale: Vec<f32> = x.iter().map(|v| v.abs()).collect();
    let mut turned = vec![false; n];

    for (m, position) in POSITIONS.iter().enumerate() {
        for h in 0..HEADS {
            let head = (m * HEADS + h) * HEAD_DIM;
            for i in 0..rotary / 2 {
                let (a, b, inv_freq, gain) = pairs(i, rotary / 2);
                // The angle in f32, because the device has no other kind.
                let theta = (*position as f32) * inv_freq;
                let (c, s) = (
                    f64::from(theta).cos() * f64::from(gain),
                    f64::from(theta).sin() * f64::from(gain),
                );
                let (x1, x2) = (f64::from(x[head + a]), f64::from(x[head + b]));
                want[head + a] = (x1 * c - x2 * s) as f32;
                want[head + b] = (x1 * s + x2 * c) as f32;
                let magnitude = (x1.abs() + x2.abs()) as f32;
                scale[head + a] = magnitude;
                scale[head + b] = magnitude;
                turned[head + a] = true;
                turned[head + b] = true;
            }
        }
    }
    Turned {
        x,
        want,
        scale,
        turned,
    }
}

impl Turned {
    /// The widest disagreement over the ROTATED channels, as a fraction of
    /// each pair's own magnitude.
    fn against(&self, got: &[f32]) -> f32 {
        got.iter()
            .zip(&self.want)
            .zip(&self.scale)
            .zip(&self.turned)
            .filter(|(_, moved)| **moved)
            .map(|(((g, w), s), _)| (g - w).abs() / s.max(f32::MIN_POSITIVE))
            .fold(0.0, f32::max)
    }

    fn agrees(&self, got: &[f32], what: &str) {
        for (i, (g, x)) in got.iter().zip(&self.x).enumerate() {
            assert!(
                self.turned[i] || g == x,
                "{what} moved channel {} of head {}, which is outside the \
                 rotation it states",
                i % HEAD_DIM,
                i / HEAD_DIM
            );
        }
        let worst = self.against(got);
        assert!(
            worst <= PAIR_BOUND,
            "{what}: the widest rotated channel is {worst} of its pair's own \
             magnitude, past the {PAIR_BOUND} one bf16 step allows"
        );
        // Per position, because the whole question about `fast::cos` is
        // whether it holds at an angle a long context reaches.
        let per: Vec<String> = POSITIONS
            .iter()
            .enumerate()
            .map(|(m, p)| {
                let at = m * HEADS * HEAD_DIM;
                let row = Self {
                    x: self.x[at..at + HEADS * HEAD_DIM].to_vec(),
                    want: self.want[at..at + HEADS * HEAD_DIM].to_vec(),
                    scale: self.scale[at..at + HEADS * HEAD_DIM].to_vec(),
                    turned: self.turned[at..at + HEADS * HEAD_DIM].to_vec(),
                };
                format!("pos {p}: {}", row.against(&got[at..at + HEADS * HEAD_DIM]))
            })
            .collect();
        plane::tolerance_holds(worst, PAIR_BOUND, what);
        plane::measured(
            what,
            &format!(
                "worst {worst} against the pair bound {PAIR_BOUND}; {}",
                per.join(", ")
            ),
        );
    }

    /// `theta = 0` is `cos = 1, sin = 0`, so the first token of a prompt
    /// comes back untouched -- and untouched means the same bits.
    fn identity_at_position_zero(&self, got: &[f32]) {
        assert_eq!(
            POSITIONS[0], 0,
            "this check reads the first row, which has to be position zero"
        );
        let row = HEADS * HEAD_DIM;
        assert_eq!(
            got[..row],
            self.x[..row],
            "position zero rotates by nothing, so its row must come back bit \
             for bit"
        );
    }

    fn bites(
        &self,
        rig: &Rig,
        symbol: &'static str,
        rotary: usize,
        extra: &[Arg<'_>],
        from: &str,
        to: &str,
    ) {
        let root = plane::mutant(FILE, from, to);
        let got = turn(rig, root.path(), symbol, rotary, extra);
        let moved = self.against(&got);
        let outside = got
            .iter()
            .zip(&self.x)
            .enumerate()
            .any(|(i, (g, x))| !self.turned[i] && g != x);
        assert!(
            moved > PAIR_BOUND || outside,
            "replacing `{from}` with `{to}` left every rotated channel within \
             {moved} of the pair bound and moved nothing outside the \
             rotation, so the comparison above would not have caught it"
        );
        plane::measured(
            symbol,
            &format!(
                "`{from}` -> `{to}`: worst {moved} against the pair bound \
                 {PAIR_BOUND}, channels outside the rotation {}",
                if outside { "moved" } else { "still" }
            ),
        );
    }
}

/// One in-place dispatch, at the grid `kernels_metal::rope::rope_grid`
/// states: `rotary / 2` lanes, one head per y, one token per z.
fn turn(
    rig: &Rig,
    root: &std::path::Path,
    symbol: &'static str,
    rotary: usize,
    extra: &[Arg<'_>],
) -> Vec<f32> {
    let n = ROWS * HEADS * HEAD_DIM;
    let seed: Vec<f32> = (0..n)
        .map(|i| plane::narrowed(((i * 11) % 37) as f32 * 0.06 - 1.1))
        .collect();
    let x = plane::alloc_bf16(&rig.context, &seed, "q");
    let positions = plane::alloc_i32(&rig.context, &POSITIONS, "positions");
    let mut args = vec![Arg::Buf(&x), Arg::Buf(&positions)];
    args.extend_from_slice(extra);
    plane::fire(
        rig,
        root,
        FILE,
        symbol,
        [(rotary / 2) as u32, HEADS as u32, ROWS as u32],
        [(rotary / 2) as u32, 1, 1],
        &args,
    );
    plane::read_bf16(&x, n)
}
