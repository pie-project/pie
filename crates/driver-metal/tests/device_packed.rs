//! The five packed MLP activations, on the GPU, for the first time.
//!
//! `mlp/packed.metal` is `mlp.swiglu`, `mlp.swiglu_clamp`,
//! `mlp.swiglu_clamp_alpha`, `mlp.geglu_tanh_packed` and `mlp.situ` -- five
//! of the fifty-one points this plane claims, all five written last week on
//! a machine with no Metal compiler, and the first thing a real one said
//! about the file was that a variable in it was called `half`. That is fixed
//! and the file compiles. What it COMPUTES has never been compared to
//! anything, and that is what is below.
//!
//! # The reference is a CPU model in this file
//!
//! There is no CUDA on a Mac, so the reference for a Metal shader is either
//! a model written here or a vector carried from a machine that has one. For
//! these five the model is written here, and it is written the way the shader
//! is written -- same association, same order, `g / (1 + exp(-g))` and not
//! `g * sigmoid(g)` -- so that the only thing the two can disagree about is
//! the last bits of a transcendental.
//!
//! A model transcribed from the same source as the shader is a weak reference
//! against a transcription error and a strong one against everything else,
//! which is why the mutations below are what make this file evidence rather
//! than the model.
//!
//! # The comparison is a bf16 STEP, not a tolerance
//!
//! Every one of these kernels widens to float, computes, and rounds once at
//! the store. So the two answers can differ only where Metal's `exp` and
//! Rust's disagree by enough to move an eight-bit mantissa's rounding, and
//! the right bound for that is one representable bf16 value -- see
//! [`plane::bf16_ulps`]. It is the tightest bound available and it is not one
//! a later hand can quietly widen. Four of the five activations hold it
//! EXACTLY: zero elements of two hundred are inexact, at every fixture in
//! this file.
//!
//! # GELU DOES NOT, AND WHAT IT COST WAS MEASURED RATHER THAN ASSUMED
//!
//! `packed_geglu_tanh` failed the first time it was ever run: element 53 of
//! 200 came back **eighty-five bf16 steps** from the model, `-1.289e-6`
//! against `-1.933e-6`, at a gate of `-4.906` and an up half of `4.406`.
//!
//! Two things were wrong and only one of them was the device.
//!
//! The first was the model. It called Rust's `f32::tanh`, so the comparison
//! was one platform's libm against another's; [`plane::exp32`] and
//! [`plane::tanh32`] now take the double's answer and round it once, which
//! removes that side of the question entirely. It did not close the gap.
//!
//! What is left IS a device measurement, and it is this: `gelu =
//! 0.5 g (1 + tanh(inner))` CANCELS for a negative gate. At `inner = -8.128`
//! the true `1 + tanh` is `1.788e-7` -- **three ulps of one** in f32 -- so
//! the answer is carried entirely by the last two bits of `tanh`, and this
//! device's `precise::tanh` returns a value one ulp nearer `-1` than the
//! correctly-rounded one. Two ulps against three is the two-thirds ratio the
//! numbers above are.
//!
//! That is not a defect in `packed.metal` and it is not fixable inside it:
//! every backend evaluating the tanh form of GELU in f32 has it, cuda
//! included, and the alternative spelling (`1 + tanh` computed as
//! `2 / (1 + exp(-2 inner))`) is a different kernel with a different cost.
//! What matters is the SIZE of it. Twenty-six of the two hundred elements are
//! in the cancellation; the largest answer among them is `9.8e-4`, against a
//! plane whose activations reach `33.5`. It is four parts in ten million of
//! the tensor.
//!
//! So the comparison is not widened -- widening it would accept a third of an
//! answer at every element, including the ones where nothing cancels. The
//! model instead carries a per-element ALLOWANCE, and the allowance is not a
//! tolerance but an equality: `eps_f32 * |0.5 g u|` is exactly what a two-ulp
//! uncertainty about `tanh` does to `0.5 g (1 + tanh) u`. Where the
//! expression does not cancel it is a millionth of a bf16 step and changes
//! nothing; where it does, it is the whole difference between measuring the
//! kernel and measuring a subtraction. Nothing else in this file carries one,
//! and the four that do not are exact.
//!
//! # The fixture is shaped so the clamps clamp
//!
//! `limit` is 3 against gates that reach 6.1 and up halves that reach 6.38,
//! so `swiglu_clamp` and `gpt-oss`'s GLU both saturate on real elements
//! rather than passing every one through. `beta` is 2 against the same
//! gates, so SiTU's `tanh(g / beta)` is at 0.995 at the top of the range --
//! the saturation the activation exists for. The intermediate width is 40
//! and not 32 or 64, because a kernel that confused the packed row's stride
//! with the result's would still land inside a power of two.
//!
//! # `up_cap <= 0` MEANS NO CAP, and that is a contract, not a shortcut
//!
//! `packed_situ` takes its soft-cap as a number rather than as a second entry
//! point, so a statement with no cap says zero. That is one comparison and
//! one `if` in the shader, and it is fired both ways below: a kernel that
//! read the branch backwards would cap exactly the statements that asked for
//! no cap and pass every test written with one value.
//!
//! # The mutations
//!
//! Nine, and every one is a defect with a name rather than a perturbation:
//! the sign inside SiLU's exponent; the gate clamped symmetrically, which
//! `packed.metal`'s own header spends a paragraph forbidding; the up half
//! clamped one-sided instead; gpt-oss's `+ 1` dropped and its `alpha`
//! dropped; GELU's cubic term dropped; SiTU's `beta` taken out of the tanh's
//! argument and its cap branch inverted. Each must move the answer by more
//! than one bf16 step, and each is fired to prove it does.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::{Arg, Rig};

const FILE: &str = "mlp/packed.metal";

/// Neither a tile nor a power of two.
const I: usize = 40;
const ROWS: usize = 5;

/// What the two clamped activations are fired against, low enough that the
/// fixture's gates cross it.
const LIMIT: f32 = 3.0;

/// gpt-oss's own.
const ALPHA: f32 = 1.702;

/// SiTU's saturation width, and its soft cap.
const BETA: f32 = 2.0;
const UP_CAP: f32 = 1.5;

const POISON: f32 = -99.0;

/// `sqrt(2 / pi)`, the tanh approximation's outer coefficient.
const K: f32 = 0.797_884_6;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_packed_swiglu_is_the_silu_of_the_gate_times_the_up_half() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `mlp.swiglu` was not fired");
        return;
    };
    let packed = fixture();
    let want = model(&packed, |g, u| silu(g) * u);

    agrees(
        &rig,
        "packed_swiglu_bfloat16",
        &packed,
        &[],
        &want,
        "mlp.swiglu",
    );

    // The sign inside the exponent: `g / (1 + exp(g))` is a function that
    // looks like SiLU at zero and is its mirror everywhere else.
    bites(
        &rig,
        "packed_swiglu_bfloat16",
        &packed,
        &[],
        &want,
        "(g / (1.0f + metal::exp(-g))) * u",
        "(g / (1.0f + metal::exp(g))) * u",
    );

    // The two halves, swapped. `packed_row + i` is the gate and
    // `packed_row + intermediate + i` is the up half, and a file that had
    // them the other way round would still produce a plausible activation.
    bites(
        &rig,
        "packed_swiglu_bfloat16",
        &packed,
        &[],
        &want,
        "const float g = float(packed[packed_row + i]);\n  const float u = float(packed[packed_row + intermediate + i]);",
        "const float g = float(packed[packed_row + intermediate + i]);\n  const float u = float(packed[packed_row + i]);",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_clamped_swiglu_bounds_the_gate_above_and_the_up_half_both_ways() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `mlp.swiglu_clamp` was not fired");
        return;
    };
    let packed = fixture();
    let want = model(&packed, |g, u| {
        let g = g.min(LIMIT);
        let u = u.clamp(-LIMIT, LIMIT);
        silu(g) * u
    });
    let limit = [Arg::F32(LIMIT)];

    agrees(
        &rig,
        "packed_swiglu_clamp_bfloat16",
        &packed,
        &limit,
        &want,
        "mlp.swiglu_clamp",
    );

    // THE ASYMMETRY IS THE POINT. `packed.metal`'s header: "a gate clamped
    // from below saturates the branch the activation exists to switch off,
    // and the model still runs". Restoring the symmetry is the mutation.
    bites(
        &rig,
        "packed_swiglu_clamp_bfloat16",
        &packed,
        &limit,
        &want,
        "  g = min(g, limit);",
        "  g = clamp(g, -limit, limit);",
    );

    // And the other half of the same asymmetry, taken the other way.
    bites(
        &rig,
        "packed_swiglu_clamp_bfloat16",
        &packed,
        &limit,
        &want,
        "  u = clamp(u, -limit, limit);",
        "  u = min(u, limit);",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_gptoss_glu_scales_the_gate_by_alpha_and_offsets_the_up_half_by_one() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `mlp.swiglu_clamp_alpha` was not fired");
        return;
    };
    let packed = fixture();
    let want = model(&packed, |g, u| {
        let g = g.min(LIMIT);
        let u = u.clamp(-LIMIT, LIMIT);
        let sig = 1.0 / (1.0 + plane::exp32(-ALPHA * g));
        (g * sig) * (u + 1.0)
    });
    let scalars = [Arg::F32(LIMIT), Arg::F32(ALPHA)];

    agrees(
        &rig,
        "packed_gptoss_swiglu_bfloat16",
        &packed,
        &scalars,
        &want,
        "mlp.swiglu_clamp_alpha",
    );

    // The `+ 1`, dropped. gpt-oss's up half is an offset residual and not a
    // multiplicand, and without it the activation is zero wherever the up
    // half is -- which is a plausible-looking sparsity.
    bites(
        &rig,
        "packed_gptoss_swiglu_bfloat16",
        &packed,
        &scalars,
        &want,
        "(g * sig) * (u + 1.0f)",
        "(g * sig) * u",
    );

    // `alpha`, dropped. The gate is then an ordinary sigmoid and the
    // statement's scalar is bound to a kernel that ignores it.
    bites(
        &rig,
        "packed_gptoss_swiglu_bfloat16",
        &packed,
        &scalars,
        &want,
        "fast::exp(-alpha * g)",
        "fast::exp(-g)",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_packed_geglu_carries_the_cubic_term_of_the_tanh_approximation() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `mlp.geglu_tanh_packed` was not fired");
        return;
    };
    let packed = fixture();
    let want = model(&packed, |g, u| {
        let inner = K * (g + 0.044_715 * g * g * g);
        0.5 * g * (1.0 + plane::tanh32(inner)) * u
    })
    // `1 + tanh(inner)` is three ulps of one at the bottom of this
    // fixture's gate range, so two ulps of `tanh` are the whole answer
    // there. This is what those two ulps do to the product -- an equality,
    // not a tolerance, and a millionth of a bf16 step wherever the
    // expression does not cancel.
    .allowing(&packed, |g, u| f32::EPSILON * (0.5 * g * u).abs());

    agrees(
        &rig,
        "packed_geglu_tanh_bfloat16",
        &packed,
        &[],
        &want,
        "mlp.geglu_tanh_packed",
    );

    // The cubic, dropped. `0.5 g (1 + tanh(k g))` is a real function and a
    // smooth one; it is just not GELU, and nothing about a forward pass
    // through it looks wrong.
    bites(
        &rig,
        "packed_geglu_tanh_bfloat16",
        &packed,
        &[],
        &want,
        "k * (g + 0.044715f * g * g * g)",
        "k * g",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn situ_saturates_at_beta_and_caps_the_up_half_only_when_asked() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `mlp.situ` was not fired");
        return;
    };
    let packed = fixture();

    let capped = model(&packed, |g, u| {
        let s = BETA * plane::tanh32(g / BETA) / (1.0 + plane::exp32(-g));
        s * (UP_CAP * plane::tanh32(u / UP_CAP))
    });
    let with_cap = [Arg::F32(BETA), Arg::F32(UP_CAP)];
    agrees(
        &rig,
        "packed_situ_bfloat16",
        &packed,
        &with_cap,
        &capped,
        "mlp.situ, soft-capped",
    );

    // The other half of the contract: a statement with no soft cap states
    // zero, and gets the plain product.
    let plain = model(&packed, |g, u| {
        BETA * plane::tanh32(g / BETA) / (1.0 + plane::exp32(-g)) * u
    });
    let no_cap = [Arg::F32(BETA), Arg::F32(0.0)];
    agrees(
        &rig,
        "packed_situ_bfloat16",
        &packed,
        &no_cap,
        &plain,
        "mlp.situ, uncapped",
    );

    // `beta` taken out of the tanh's argument, so the gate saturates at
    // `beta` for an argument scale it was never meant to.
    bites(
        &rig,
        "packed_situ_bfloat16",
        &packed,
        &with_cap,
        &capped,
        "beta * precise::tanh(g / beta)",
        "beta * precise::tanh(g)",
    );

    // The cap's branch, inverted. This is the mutation the second fire above
    // exists for: it caps exactly the statements that asked for no cap, so a
    // file that only ever fired one value would pass it.
    bites(
        &rig,
        "packed_situ_bfloat16",
        &packed,
        &with_cap,
        &capped,
        "if (up_cap > 0.0f) {",
        "if (up_cap < 0.0f) {",
    );
}

/// `g / (1 + exp(-g))`, the way every packed form in the tree spells it.
fn silu(g: f32) -> f32 {
    g / (1.0 + plane::exp32(-g))
}

/// The packed `[gate | up]` rows the five are fired over.
///
/// 23 and 19 are prime and share no factor with 40 or 5, so the gate and the
/// up half differ at every column of every row -- which is what makes a
/// swapped pair visible rather than a no-op on the diagonal.
fn fixture() -> Vec<f32> {
    let mut packed = vec![0.0; ROWS * 2 * I];
    for r in 0..ROWS {
        for i in 0..I {
            let g = ((r * 3 + i * 7) % 23) as f32 * 0.55 - 6.0;
            let u = ((r * 5 + i * 11) % 19) as f32 * 0.66 - 5.5;
            packed[r * 2 * I + i] = plane::narrowed(g);
            packed[r * 2 * I + I + i] = plane::narrowed(u);
        }
    }
    packed
}

/// The model, and what an f32 cancellation inside it is allowed to cost.
///
/// `slack` is zero for four of the five activations, and it is not a
/// tolerance: it is the exact propagation of a two-ulp uncertainty about a
/// transcendental through the expression the shader evaluates. Where the
/// expression does not cancel it is a millionth of one bf16 step and changes
/// nothing; where it does, it is the whole difference between a comparison
/// that measures the kernel and one that measures two libms.
struct Want {
    y: Vec<f32>,
    slack: Vec<f32>,
}

/// The activation, over the same packed row the kernel reads.
fn model(packed: &[f32], f: impl Fn(f32, f32) -> f32) -> Want {
    let mut y = vec![0.0; ROWS * I];
    for r in 0..ROWS {
        for i in 0..I {
            let (g, u) = halves(packed, r * I + i);
            y[r * I + i] = f(g, u);
        }
    }
    Want {
        y,
        slack: vec![0.0; ROWS * I],
    }
}

impl Want {
    /// Allow each element the absolute error `s(g, u)` on top of one bf16
    /// step, because the expression amplifies an ulp there by that much.
    fn allowing(mut self, packed: &[f32], s: impl Fn(f32, f32) -> f32) -> Self {
        for (e, slack) in self.slack.iter_mut().enumerate() {
            let (g, u) = halves(packed, e);
            *slack = s(g, u);
        }
        self
    }

    /// Which elements this rectangle does not answer: too far in bf16 steps
    /// AND too far in absolute terms for the amplification to explain.
    fn misses(&self, got: &[f32]) -> Vec<usize> {
        (0..self.y.len())
            .filter(|e| {
                plane::bf16_ulps(got[*e], self.y[*e]) > 1
                    && (got[*e] - self.y[*e]).abs() > self.slack[*e]
            })
            .collect()
    }
}

/// The gate and the up half at output element `e`.
fn halves(packed: &[f32], e: usize) -> (f32, f32) {
    let (r, i) = (e / I, e % I);
    (packed[r * 2 * I + i], packed[r * 2 * I + I + i])
}

/// Fire the tree's own shader and require it to answer the model.
fn agrees(
    rig: &Rig,
    symbol: &'static str,
    packed: &[f32],
    extra: &[Arg<'_>],
    want: &Want,
    what: &str,
) {
    let got = activation(rig, plane::kernels_dir().as_path(), symbol, packed, extra);
    assert!(
        got[ROWS * I..].iter().all(|v| *v == POISON),
        "{what} wrote past the rectangle its point states"
    );
    let inside = &got[..ROWS * I];
    let missed = want.misses(inside);
    assert!(
        missed.is_empty(),
        "{what}: element {} is {} bf16 steps from the model and {} past what \
         its cancellation allows -- {} against {}, for a gate of {} and an up \
         half of {}",
        missed[0],
        plane::bf16_ulps(inside[missed[0]], want.y[missed[0]]),
        (inside[missed[0]] - want.y[missed[0]]).abs() - want.slack[missed[0]],
        inside[missed[0]],
        want.y[missed[0]],
        halves(packed, missed[0]).0,
        halves(packed, missed[0]).1,
    );
    let (widest, _, inexact) = plane::ulp_spread(inside, &want.y);
    let amplified = (0..want.y.len())
        .filter(|e| plane::bf16_ulps(inside[*e], want.y[*e]) > 1)
        .count();
    plane::measured(
        what,
        &format!(
            "{widest} bf16 steps at worst, {inexact} of {} elements inexact, \
             {amplified} explained by an amplified ulp",
            ROWS * I
        ),
    );
}

/// Fire a SABOTAGED shader and require the same comparison to fail.
fn bites(
    rig: &Rig,
    symbol: &'static str,
    packed: &[f32],
    extra: &[Arg<'_>],
    want: &Want,
    from: &str,
    to: &str,
) {
    let root = plane::mutant(FILE, from, to);
    let got = activation(rig, root.path(), symbol, packed, extra);
    let missed = want.misses(&got[..ROWS * I]);
    assert!(
        !missed.is_empty(),
        "replacing `{from}` with `{to}` left every element inside one bf16 \
         step, so the comparison above would not have caught it"
    );
    let (widest, _, _) = plane::ulp_spread(&got[..ROWS * I], &want.y);
    plane::measured(
        symbol,
        &format!(
            "`{from}` -> `{to}` misses {} of {} elements, {widest} bf16 steps \
             at worst",
            missed.len(),
            ROWS * I
        ),
    );
}

/// One dispatch, at the grid `kernels_metal::mlp::halves` states.
fn activation(
    rig: &Rig,
    root: &std::path::Path,
    symbol: &'static str,
    packed: &[f32],
    extra: &[Arg<'_>],
) -> Vec<f32> {
    let src = plane::alloc_bf16(&rig.context, packed, "packed");
    let out = plane::alloc_bf16(&rig.context, &vec![POISON; ROWS * I + I], "out");
    let mut args = vec![Arg::Buf(&src), Arg::Buf(&out), Arg::U32(I as u32)];
    args.extend_from_slice(extra);
    plane::fire(
        rig,
        root,
        FILE,
        symbol,
        [I as u32, ROWS as u32, 1],
        [I.min(256) as u32, 1, 1],
        &args,
    );
    plane::read_bf16(&out, ROWS * I + I)
}
