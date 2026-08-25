//! The two norm tails: the gated RMS fold, and the per-layer scalar.
//!
//! `norm/gated_rms.metal` answers `norm.rmsnorm_gated` and
//! `norm.rmsnorm_gated_by`; `norm/layer_scalar.metal` answers
//! `norm.mul_scalar` and `norm.scale`. All four are new, none had ever been
//! executed, and the two files fail in opposite ways -- which is why they
//! share a test file rather than each getting one.
//!
//! # One of these is bit-exact and the other cannot be
//!
//! `layer_scalar_mul` is a widen, a multiply and a round. There is no
//! transcendental in it and no reduction, so a Rust model reproduces it TO
//! THE BIT and the comparison below is `assert_eq!` rather than a bound.
//! That matters for one specific reason: the shader reads its stated scalar
//! as `float(static_cast<T>(scalar))` -- it rounds the statement's float to
//! bf16 BEFORE multiplying -- and the difference that makes is at most half
//! a bf16 step in the product, which any tolerance wide enough for a
//! transcendental would swallow whole. A bit-exact comparison is the only
//! one that can see it, and the mutation below proves it does.
//!
//! `gated_rms` has a `rsqrt`, an `exp` and a threadgroup-wide sum whose
//! association Metal chooses, so it gets the one-bf16-step bound this sweep
//! uses everywhere the answer lands in bf16.
//!
//! # `vd = 40` is the interesting width, and it is not the real one
//!
//! `rms_reduce.h` folds with `simd_sum` and then folds the per-simdgroup
//! partials, and the threadgroup is `[vd, 1, 1]` -- the value-head width
//! itself. A checkpoint's is 128, four whole simdgroups, and every lane of
//! every one of them is live. `head_width` accepts anything up to 1024, so
//! 40 is equally a fire, and 40 is 32 plus 8: one full simdgroup and one
//! with twenty-four lanes that do not exist. `simd_sum` over a partial
//! simdgroup, and `partials[]` entries that no simdgroup ever wrote, are
//! exactly the two things a header cannot promise and a device can. Both
//! widths are fired.
//!
//! # The two instantiations are two kernels, and that is checked
//!
//! `gated_rms` and `gated_rms_by` are one template at `SILU = true` and
//! `false`. This tree has already been bitten by a claim of that shape --
//! three paged attention bodies that answered the same softmax and did not
//! share a contract -- so the two are not merely each compared to their own
//! model: they are required to DISAGREE with each other. A build in which
//! both host names resolved to one instantiation would pass two model
//! comparisons and fail that one.
//!
//! # The mutations
//!
//! Six. Two of them are in `rms_reduce.h` rather than in the shader that
//! includes it, which is deliberate: `driver-metal` splices quoted includes
//! itself before handing Metal a translation unit, so a mutation that only
//! bit when it landed in the top-level file would be testing the splicer's
//! good behaviour rather than the kernel's. `acc / float(axis_size)` becoming
//! `acc` is the classic root-mean-square defect -- a root-SUM-square, which
//! is a smaller number by `sqrt(n)` and still a perfectly smooth
//! normalisation.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::{Arg, Rig};

const FILE_RMS: &str = "norm/gated_rms.metal";
const FILE_SCALAR: &str = "norm/layer_scalar.metal";

/// The real value-head width, four whole simdgroups.
const WIDE: usize = 128;
/// One full simdgroup and a partial one.
const RAGGED: usize = 40;

const HEADS: usize = 3;
const ROWS: usize = 2;
const EPS: f32 = 1e-6;

/// Wider than one threadgroup, and not a multiple of it.
const SCALARS: usize = 300;

/// Not representable in bf16: `1.3` rounds to `1.296875`, and the shader
/// rounds it before it multiplies.
const STATED: f32 = 1.3;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_gated_rms_normalises_over_the_value_head_and_gates_by_silu() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `norm.rmsnorm_gated` was not fired");
        return;
    };

    for vd in [WIDE, RAGGED] {
        let fold = Fold::of(vd);
        let got = fire_rms(
            &rig,
            plane::kernels_dir().as_path(),
            "gated_rms_f32_bfloat16",
            &fold,
        );
        let want = fold.model(true);
        agrees(&got, &want, &format!("norm.rmsnorm_gated at vd={vd}"));
    }

    let fold = Fold::of(WIDE);
    let want = fold.model(true);

    // The mean, dropped: a root-SUM-square rather than a root-mean-square.
    // This one lives in `rms_reduce.h` and is reached through the splice.
    bites(
        &rig,
        FILE_RMS,
        "gated_rms_f32_bfloat16",
        &fold,
        &want,
        "precise::rsqrt(acc / float(axis_size) + eps)",
        "precise::rsqrt(acc + eps)",
    );

    // The SiLU's own factor. `gate = sigmoid(z)` where the point states
    // `z * sigmoid(z)` is the `_by` arm's arithmetic under this arm's name.
    bites(
        &rig,
        FILE_RMS,
        "gated_rms_f32_bfloat16",
        &fold,
        &want,
        "const float gate = SILU ? zr * sig : sig;",
        "const float gate = SILU ? sig : zr * sig;",
    );

    // The norm weight, broadcast. `w` is one value per value CHANNEL and
    // `lid` is the channel, so reading element zero for every lane produces
    // a scaled normalisation that is smooth, finite and flat.
    bites(
        &rig,
        FILE_RMS,
        "gated_rms_f32_bfloat16",
        &fold,
        &want,
        "float(w[lid])",
        "float(w[0])",
    );

    // The head axis, dropped from the row address: every value head folds
    // head zero's channels.
    bites(
        &rig,
        FILE_RMS,
        "gated_rms_f32_bfloat16",
        &fold,
        &want,
        "size_t(tgpos.z * tpg.y + tgpos.y) * vd + lid",
        "size_t(tgpos.z * tpg.y) * vd + lid",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_ungated_arm_publishes_the_sigmoid_and_not_the_silu() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `norm.rmsnorm_gated_by` was not fired");
        return;
    };

    let fold = Fold::of(WIDE);
    let root = plane::kernels_dir();
    let by = fire_rms(&rig, root.as_path(), "gated_rms_by_f32_bfloat16", &fold);
    agrees(&by, &fold.model(false), "norm.rmsnorm_gated_by");

    // TWO NAMES, TWO KERNELS. `gated_rms` and `gated_rms_by` are one
    // template at two values of `SILU`, and a build that resolved both host
    // names to one instantiation would answer both model comparisons above
    // and this one is what would catch it.
    let silu = fire_rms(&rig, root.as_path(), "gated_rms_f32_bfloat16", &fold);
    assert_ne!(
        silu, by,
        "`gated_rms_f32_bfloat16` and `gated_rms_by_f32_bfloat16` answered \
         the same numbers, so one of the two instantiations is not being \
         reached"
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_per_layer_scalar_rounds_the_statement_before_it_multiplies() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `norm.mul_scalar` was not fired");
        return;
    };

    let x = row();
    // The kernel's own spelling: `float(static_cast<T>(scalar))`, so the
    // statement's 1.3 is 1.296875 by the time it reaches the multiply.
    let s = plane::narrowed(STATED);
    let want: Vec<f32> = x.iter().map(|v| plane::narrowed(v * s)).collect();

    let got = fire_stated(&rig, plane::kernels_dir().as_path(), &x, STATED);
    assert_eq!(
        got, want,
        "`layer_scalar_mul_stated_bfloat16` is a widen, a multiply and one \
         round, so it agrees to the bit or it is wrong"
    );
    plane::measured("norm.mul_scalar", "bit-exact over 300 elements");

    // The rounding, dropped. `1.3` against `1.296875` is a shift of 0.0015
    // relative -- under half a bf16 step, which is to say invisible to any
    // tolerance this sweep uses anywhere else, and visible here.
    let root = plane::mutant(
        FILE_SCALAR,
        "const float s = static_cast<float>(static_cast<T>(scalar));",
        "const float s = scalar;",
    );
    let bent = fire_stated(&rig, root.path(), &x, STATED);
    assert_ne!(
        bent, want,
        "a scalar multiplied at float width must not agree with one rounded \
         to the element first"
    );
    let moved = bent.iter().zip(&want).filter(|(a, b)| a != b).count();
    plane::measured(
        "norm.mul_scalar",
        &format!("dropping the round moves {moved} of {SCALARS} elements"),
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_resident_scalar_reads_its_one_element_from_the_buffer() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `norm.scale` was not fired");
        return;
    };

    let x = row();
    // A bf16 buffer of one element, so what the kernel reads and what the
    // model multiplies by are the same bits by construction.
    let s = plane::narrowed(0.6875);
    let want: Vec<f32> = x.iter().map(|v| plane::narrowed(v * s)).collect();

    let got = fire_resident(&rig, plane::kernels_dir().as_path(), &x, s);
    assert_eq!(
        got, want,
        "`layer_scalar_mul_bfloat16` broadcasts element zero of its scalar \
         plane over the whole row"
    );
    plane::measured("norm.scale", "bit-exact over 300 elements");

    // The broadcast, taken from the wrong end. The scalar plane is `[1]`, so
    // there is nothing to read at index one -- which is why the fixture
    // allocates two and puts a different number in the second: a kernel that
    // indexed the gid rather than zero would produce a row scaled by a
    // gradient, and no fixture with a one-element plane could tell.
    let root = plane::mutant(FILE_SCALAR, "scalar[0]", "scalar[1]");
    let bent = fire_resident(&rig, root.path(), &x, s);
    assert_ne!(
        bent, want,
        "a scale that reads element one of its plane must not agree with one \
         that reads element zero"
    );
}

/// One `gated_rms` fixture: the normed plane, its gate, and its weight.
struct Fold {
    vd: usize,
    x: Vec<f32>,
    z: Vec<f32>,
    w: Vec<f32>,
}

impl Fold {
    fn of(vd: usize) -> Self {
        let n = ROWS * HEADS * vd;
        // The gate spans `[-4, 4]`, so `sigmoid` is neither flat nor
        // saturated and the SiLU's factor changes sign with it.
        let z: Vec<f32> = (0..n)
            .map(|i| plane::narrowed(((i * 13) % 41) as f32 * 0.2 - 4.0))
            .collect();
        Self {
            vd,
            x: plane::spread_by(n, 5, 2.0),
            z,
            w: plane::spread_by(vd, 9, 0.5),
        }
    }

    /// `norm/gated_rms.metal`'s body, in Rust.
    fn model(&self, silu: bool) -> Vec<f32> {
        let mut out = vec![0.0; ROWS * HEADS * self.vd];
        for head in 0..ROWS * HEADS {
            let at = head * self.vd;
            let row = &self.x[at..at + self.vd];
            let mean = row.iter().map(|v| v * v).sum::<f32>() / self.vd as f32;
            let inv = 1.0 / (mean + EPS).sqrt();
            for c in 0..self.vd {
                let zr = self.z[at + c];
                let sig = 1.0 / (1.0 + plane::exp32(-zr));
                let gate = if silu { zr * sig } else { sig };
                out[at + c] = (row[c] * inv * self.w[c]) * gate;
            }
        }
        out
    }
}

/// The row `layer_scalar` is fired over.
fn row() -> Vec<f32> {
    plane::spread_by(SCALARS, 4, 3.0)
        .into_iter()
        .map(plane::narrowed)
        .collect()
}

fn agrees(got: &[f32], want: &[f32], what: &str) {
    let (widest, at, inexact) = plane::ulp_spread(got, want);
    assert!(
        widest <= 1,
        "{what}: element {at} is {widest} bf16 steps from the model -- {} \
         against {}",
        got[at],
        want[at],
    );
    plane::measured(
        what,
        &format!(
            "{widest} bf16 step at worst, {inexact} of {} elements inexact",
            got.len()
        ),
    );
}

fn bites(
    rig: &Rig,
    file: &'static str,
    symbol: &'static str,
    fold: &Fold,
    want: &[f32],
    from: &str,
    to: &str,
) {
    let root = plane::mutant(file, from, to);
    let got = fire_rms(rig, root.path(), symbol, fold);
    let (widest, _, _) = plane::ulp_spread(&got, want);
    assert!(
        widest > 1,
        "replacing `{from}` with `{to}` moved the answer by {widest} bf16 \
         steps, so the comparison above would not have caught it"
    );
    plane::measured(
        symbol,
        &format!("`{from}` -> `{to}` moves the answer {widest} bf16 steps"),
    );
}

/// One dispatch, at the grid `kernels_metal::norm::head_row_grid` states:
/// one threadgroup per (row, value head), `vd` lanes folding it.
fn fire_rms(rig: &Rig, root: &std::path::Path, symbol: &'static str, fold: &Fold) -> Vec<f32> {
    let n = ROWS * HEADS * fold.vd;
    let x = plane::alloc_f32(&rig.context, &fold.x, "normed");
    let z = plane::alloc_bf16(&rig.context, &fold.z, "gate");
    let w = plane::alloc_f32(&rig.context, &fold.w, "gate_norm_w");
    let out = plane::alloc_bf16(&rig.context, &vec![0.0; n], "out");
    plane::fire(
        rig,
        root,
        FILE_RMS,
        symbol,
        [fold.vd as u32, HEADS as u32, ROWS as u32],
        [fold.vd as u32, 1, 1],
        &[
            Arg::Buf(&x),
            Arg::Buf(&z),
            Arg::Buf(&w),
            Arg::Buf(&out),
            Arg::F32(EPS),
            Arg::I32(fold.vd as i32),
        ],
    );
    plane::read_bf16(&out, n)
}

/// `norm.mul_scalar`: the scalar arrives as a stated float.
fn fire_stated(rig: &Rig, root: &std::path::Path, x: &[f32], s: f32) -> Vec<f32> {
    let src = plane::alloc_bf16(&rig.context, x, "x");
    let out = plane::alloc_bf16(&rig.context, &vec![0.0; x.len()], "out");
    plane::fire(
        rig,
        root,
        FILE_SCALAR,
        "layer_scalar_mul_stated_bfloat16",
        [x.len() as u32, 1, 1],
        [256, 1, 1],
        &[Arg::Buf(&src), Arg::F32(s), Arg::Buf(&out)],
    );
    plane::read_bf16(&out, x.len())
}

/// `norm.scale`: the scalar is a resident `[1]` plane.
///
/// Allocated two wide with a different number in the second, so that a
/// kernel reading anything but element zero is visible.
fn fire_resident(rig: &Rig, root: &std::path::Path, x: &[f32], s: f32) -> Vec<f32> {
    let src = plane::alloc_bf16(&rig.context, x, "x");
    let scalar = plane::alloc_bf16(&rig.context, &[s, -s - 1.0], "layer_scalar");
    let out = plane::alloc_bf16(&rig.context, &vec![0.0; x.len()], "out");
    plane::fire(
        rig,
        root,
        FILE_SCALAR,
        "layer_scalar_mul_bfloat16",
        [x.len() as u32, 1, 1],
        [256, 1, 1],
        &[Arg::Buf(&src), Arg::Buf(&scalar), Arg::Buf(&out)],
    );
    plane::read_bf16(&out, x.len())
}
