//! THE GEMM, LAYOUT AND NORM WAVE, FIRED ON THE CARD.
//!
//! `tests/device_fire.rs` established the road — a device opens, every module
//! becomes a pipeline, one kernel dispatches and the numbers come back — for
//! `norm.rmsnorm`. This file walks that road for the six points that landed
//! with `kernels/gemm/dense.slang`, `kernels/layout/deinterleave.slang` and
//! `norm/layer_scalar.slang`'s stated arm:
//!
//! * `gemm.matmul`, `gemm.lm_head`, `gemm.attention_landing` — one arithmetic
//!   under three names, so ONE shader pair is fired and the three claim bodies
//!   are held against it in `tests/doors.rs` rather than dispatched three
//!   times here.
//! * `layout.split_rows`, `layout.select`
//! * `norm.mul_scalar`
//!
//! # What the numbers are held against
//!
//! BOTH references, and the difference between them is the point.
//!
//! The `f64` host model is computed in double so that neither device's `f32`
//! reassociation is baked into the thing both are being judged by. It travels:
//! no CUDA toolkit, no second vendor, no golden blob.
//!
//! The CUDA twin is the one that can catch the host model being this shader's
//! own mistake written twice. `scripts/`-adjacent tooling does not carry it,
//! so it is run out of tree and its output is diffed against
//! `PIE_VULKAN_GEMM_DUMP`; the measurement is recorded in the commit message
//! rather than asserted here, for the reason `device_fire.rs` gives at length:
//! a number quoted in prose that cannot be re-run is a number this tree has no
//! reason to believe, and an assertion that a bit-exact agreement HOLDS would
//! turn a lucky shape into a rule.
//!
//! What was run, on this L40S, against `nvcc -O2 -arch=sm_89` builds of
//! `kernels-cuda`'s own headers over the same inputs these generators produce:
//!
//! ```text
//! pie::layout::split_rows   2561 + 3432 / 5993 words  BIT-IDENTICAL
//! pie::layout::select              689 /  689 words  BIT-IDENTICAL
//! pie::norm::scalar_mul           5993 / 5993 words  BIT-IDENTICAL
//! cublasGemmEx                   11 of 12 shapes     BIT-IDENTICAL
//! ```
//!
//! The gemm reference is `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, ..,
//! lda = k, ldb = k, ldc = n, CUBLAS_COMPUTE_32F)` — the call
//! `kernels_cuda::gemm::act_x_wt_bf16_beta` makes, and the definition of the
//! `[N, K]` layout this shader indexes.
//!
//! ELEVEN OF TWELVE BIT-IDENTICAL WAS NOT ASSUMED AND IS NOT ASSERTED. Two
//! independently written kernels, two languages, two execution models and two
//! compilers agreeing to the BIT on eleven ragged shapes is a measurement, not
//! a property; the two accumulate in different orders and a shape will
//! eventually disagree.
//!
//! ONE ALREADY DOES, and it is the interesting one. At `(34, 96, 5376)` — the
//! longest contraction here — the two differ on 2441 of 3264 elements, and the
//! `f64` model says which is which:
//!
//! ```text
//! vulkan vs the f64 truth:  worst 2.879e-3   (budget 3.906e-3)
//! cuBLAS vs the f64 truth:  worst 7.918e-3
//! where they differ:        vulkan is closer on 2364, cuBLAS on 0
//! ```
//!
//! So the disagreement is cuBLAS's, on every element of it: `dense.slang`
//! keeps one `f32` running sum for the whole 5376, and a tensor-op tactic that
//! splits K and recombines does not. That is the reading `kernels-wgpu`'s own
//! GEMM commit recorded on its plane, arrived at here independently.
//!
//! # Why the shapes are ragged
//!
//! [`SHAPES`] straddles every seam the two arms have. M below, at and above
//! the 32-row tile, so the arm SELECTION is exercised in both directions and
//! the tile's row overhang is real. N = 47 and N = 1, so the last column tile
//! overhangs — the defect `quant/qmm_t.slang` records a GPU sweep catching, in
//! this tree, where the overhang wrote over the NEXT row rather than past the
//! buffer and was invisible at every tile-aligned shape. K = 8 below one
//! staging block, K = 258, 334 and 5376 so the K tail is never whole.
//!
//! Every one of those was a round number in some earlier suite, and a tail
//! that never exists is a tail nothing tests.

#![cfg(feature = "device")]
#![allow(clippy::print_stdout)]

use driver_vulkan::device::{Bound, Device, Pipelines};

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Half a bf16 ulp, relative to the matrix's own largest output.
///
/// bf16 keeps eight significand bits, so a correctly rounded result is within
/// `2^-9` of the exact value and within `2^-9 * max/|v|` of it when scaled by
/// the largest — `1/256` at the worst. Scaling by the matrix's own largest and
/// not by `max(|want|, 1.0)` is deliberate, for the reason `device_fire.rs`
/// records: a floor of one turns a relative claim into a flat absolute that is
/// meaningless at small magnitudes.
const TOLERANCE: f64 = 1.0 / 256.0;

/// What an untouched output word holds.
///
/// `0x4780` is bf16 for `65536.0`. Every shape below is bounded well under it,
/// so it is a value these kernels cannot produce. **Zero cannot be used**:
/// [`Device::empty`] hands back zeros, so a slot nothing wrote and a slot
/// written with a zero would be the same bytes, and a dispatch that never ran
/// would satisfy a check written against zero.
const SENTINEL: u8 = 0x47;

/// The twelve shapes, as `(m, n, k)`. See the header for what each straddles.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 64, 128),
    (1, 47, 258),
    (7, 32, 64),
    (31, 47, 334),
    (32, 32, 32),
    (33, 47, 258),
    (34, 96, 5376),
    (64, 128, 896),
    (128, 47, 8),
    (96, 512, 128),
    (33, 1, 96),
    (40, 1024, 64),
];

static GPU: OnceLock<Option<Mutex<Device>>> = OnceLock::new();

fn gpu() -> Option<MutexGuard<'static, Device>> {
    let held = GPU.get_or_init(|| match Device::open() {
        Ok(d) => Some(Mutex::new(d)),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    });
    held.as_ref()
        .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

macro_rules! gpu {
    () => {{
        if !kernels_vulkan::embedded() {
            eprintln!(
                "skipped: built without kernels-vulkan/native, so there are no \
                 modules to build a pipeline from"
            );
            return;
        }
        let Some(device) = gpu() else {
            return;
        };
        device
    }};
}

/// Round to nearest even — the narrowing `common/bf16.slang` performs.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// Widening is exact: bf16 IS the top half of an f32.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| to_bf16(*x).to_le_bytes()).collect()
}

fn bf16_read(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| from_bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// The activations, row-major. Every value is a multiple of `1/16` under 2.2,
/// so **every input is bf16-exact** and the comparison is about the kernel's
/// arithmetic rather than about who rounded the inputs.
///
/// `37` steps through the residues of `71` and `997` moves each row somewhere
/// unrelated, so neighbouring elements are far apart in value. A ramp would
/// make an off-by-one index a difference of one sixteenth, which no tolerance
/// worth having could see.
fn act_at(r: usize, i: usize) -> f32 {
    ((r * 997 + i * 37) % 71) as f32 / 16.0 - 35.0 / 16.0
}

/// The weights, `[N, K]` row-major with K contiguous — the layout the point
/// declares and the shader indexes.
///
/// A DIFFERENT stride from [`act_at`] on purpose: with the same formula, a
/// shader that transposed one operand could still land on a plausible number.
fn wgt_at(n: usize, k: usize) -> f32 {
    ((n * 131 + k * 17) % 53) as f32 / 32.0 - 26.0 / 32.0
}

/// An activation that uses the WHOLE bf16 significand, for the one test whose
/// subject is a ROUNDING.
///
/// [`act_at`] is a multiple of `1/16` under 2.2 — six significant bits and one
/// exponent — which is exactly right for the gemm and the cuts, where what is
/// being measured is an index or an accumulate and a coarse, widely-spaced
/// value makes an off-by-one loud. It is exactly WRONG for
/// [`mul_scalar_rounds_its_factor_through_bf16_and_it_is_measurable`], and
/// this was measured rather than reasoned about: at `act_at`'s values, the
/// rounded and unrounded factors give the SAME bf16 product on all 5993
/// elements, so the test passed while measuring nothing about the trip. Its
/// own mutation assertion is what caught that.
///
/// The reason is arithmetic. `sqrt(2048)` and `bf16(sqrt(2048))` differ by
/// about one part in 10^4, and bf16's own resolution is one part in 2^9 — so
/// the two products can only round apart when `x`'s significand puts the
/// result near a rounding boundary. Six bits of `x` never does; eight bits
/// across seven exponents does, on 141 of 5993.
fn fine_at(n: usize) -> f32 {
    let mantissa = 1.0 + (n % 128) as f32 / 128.0;
    let exponent = ((n / 128) % 7) as i32 - 3;
    from_bf16(to_bf16(mantissa * 2.0f32.powi(exponent)))
}

fn activations(m: usize, k: usize) -> Vec<f32> {
    (0..m)
        .flat_map(|r| (0..k).map(move |i| act_at(r, i)))
        .collect()
}

fn weights(n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|c| (0..k).map(move |i| wgt_at(c, i)))
        .collect()
}

/// `y[M, N] = act[M, K] @ w[N, K]^T`, in `f64`.
///
/// The shader's arithmetic and not a tidier equivalent: both operands widen
/// from the SAME bf16 the device was handed, and the accumulate is the plain
/// in-order sum. Returned alongside the matrix's largest magnitude, which is
/// what the tolerance scales by.
fn reference(act: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> (Vec<f64>, f64) {
    let mut y = vec![0.0f64; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += f64::from(act[r * k + i]) * f64::from(w[c * k + i]);
            }
            y[r * n + c] = acc;
        }
    }
    let scale = y.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    (y, scale)
}

/// The push block a fire packs: a run of four-byte scalars, end to end, in the
/// order the claim body passes them.
fn push_i32(words: &[i32]) -> Vec<u8> {
    words.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Groups from lanes, the way `driver_vulkan::serve` does it: a claim body
/// states INVOCATIONS and the driver divides by the module's own declared
/// local size, per axis.
///
/// Computed here rather than hard-coded, so that a shader whose `[numthreads]`
/// changed without its claim body would be caught by the sentinel rather than
/// by nothing.
fn groups(lanes: [u32; 3], local: [u32; 3]) -> [u32; 3] {
    [
        lanes[0].div_ceil(local[0].max(1)),
        lanes[1].div_ceil(local[1].max(1)),
        lanes[2].div_ceil(local[2].max(1)),
    ]
}

/// Where a dump goes, when one is asked for. See the header.
fn dump(name: &str, raw: &[u8]) {
    if let Some(dir) = std::env::var_os("PIE_VULKAN_GEMM_DUMP") {
        let path = std::path::Path::new(&dir).join(name);
        std::fs::write(path, raw).expect("the dump directory is writable");
    }
}

/// One dense projection, fired on the device, returned as `f32`.
///
/// The entrypoint is chosen the way `kernels_vulkan::gemm` chooses it — the
/// vector arm below the row tile, the staged tile at or above it — and the
/// grid is that module's own lanes divided by that module's own local size, so
/// what is exercised is the arithmetic the claim body would have asked for.
fn fire_gemm(
    device: &Device,
    cache: &mut Pipelines,
    m: usize,
    n: usize,
    k: usize,
    short_by: [u32; 3],
) -> Vec<f32> {
    let rows = i32::try_from(m).expect("m fits");
    let columns = i32::try_from(n).expect("n fits");
    let contraction = i32::try_from(k).expect("k fits");

    let (entrypoint, lanes) = if rows < kernels_vulkan::gemm::TILE_M {
        (
            "dense_gemv_t_bfloat16",
            kernels_vulkan::gemm::vector_lanes(rows, columns),
        )
    } else {
        (
            "dense_gemm_t_bfloat16_bm_32_bn_32",
            kernels_vulkan::gemm::tile_lanes(rows, columns).expect("a grid"),
        )
    };

    let (code, tier) = device.module_for(entrypoint).expect("a module");
    let push = push_i32(&[rows, columns, contraction]);
    let pipeline = cache
        .get(device, entrypoint, code, push.len() as u32, 0, tier)
        .expect("the pipeline builds");

    let declared = pipeline.declared();
    assert_eq!(
        declared.bindings, 3,
        "{entrypoint}: x, w and out_ are the three buffers `dense.slang` declares"
    );
    assert_eq!(
        declared.push_offsets,
        vec![0, 4, 8],
        "{entrypoint}: m, n, k -- the order `struct Push` declares and the \
         order the claim body passes them"
    );

    let xb = bf16_bytes(&activations(m, k));
    let wb = bf16_bytes(&weights(n, k));
    let out = device.empty((m * n * 2) as u64).expect("an output buffer");
    device
        .write(&out, &vec![SENTINEL; m * n * 2])
        .expect("the sentinel goes down before the dispatch");
    let bufs = [
        device.buffer(&xb).expect("x"),
        device.buffer(&wb).expect("w"),
        out,
    ];

    let mut grid = groups(lanes, declared.local);
    for (g, s) in grid.iter_mut().zip(short_by) {
        *g -= s;
    }
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(pipeline, &bound, &push, grid)
        .expect("the dispatch");

    let raw = device.read(&bufs[2]).expect("read back");
    // ONLY THE WHOLE GRID IS DUMPED. Both mutation tests re-fire a shape the
    // sweep already covered — `(1, 64, 128)` and `(96, 512, 128)` — and an
    // unconditional dump here overwrote those two files with a deliberately
    // undershot result. The out-of-tree CUDA diff then reported 4.7e3 and
    // 3.9e3 on exactly the two shapes the mutations name, which is the
    // mutation's own number and not a disagreement between the planes.
    if short_by == [0, 0, 0] {
        dump(&format!("vk-gemm-{m}-{n}-{k}.bin"), &raw);
    }
    let got = bf16_read(&raw);
    for b in bufs {
        device.free(b);
    }
    got
}

/// The worst relative error over a whole matrix, and where it was.
fn worst_of(got: &[f32], want: &[f64], scale: f64) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = (f64::from(*g) - *w).abs() / scale;
        if e > worst {
            worst = e;
            at = i;
        }
    }
    (worst, at)
}

/// THE DENSE PROJECTION IS THE PROJECTION THE REFERENCE COMPUTES, on twelve
/// ragged shapes across both arms.
///
/// This is the point without which no lane binds on this plane, so it is the
/// one held against the most shapes. Both arms are exercised by the shape list
/// itself: `m < 32` takes the vector arm and the rest take the staged tile,
/// and the count of each is printed so a change in the seam is visible.
#[test]
fn a_dense_projection_this_device_computes_is_the_one_the_reference_computes() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let mut worst_overall = 0.0f64;
    let mut exact = 0usize;
    let mut vector = 0usize;

    for &(m, n, k) in SHAPES {
        let act = activations(m, k);
        let w = weights(n, k);
        assert_eq!(
            bf16_read(&bf16_bytes(&act)),
            act,
            "activations are bf16-exact"
        );
        assert_eq!(bf16_read(&bf16_bytes(&w)), w, "weights are bf16-exact");

        let got = fire_gemm(&device, &mut cache, m, n, k, [0, 0, 0]);
        assert_eq!(got.len(), m * n);
        let (want, scale) = reference(&act, &w, m, n, k);
        assert!(scale > 0.0, "the reference for {m}x{n}x{k} is all zeros");

        let (worst, at) = worst_of(&got, &want, scale);
        assert!(
            worst <= TOLERANCE,
            "{m}x{n}x{k} at row {} column {}: the device says {}, the reference \
             says {}, a relative error of {worst:.3e} against a budget of \
             {TOLERANCE:.3e}",
            at / n,
            at % n,
            got[at],
            want[at],
        );
        if worst == 0.0 {
            exact += 1;
        }
        if m < 32 {
            vector += 1;
        }
        worst_overall = worst_overall.max(worst);
        println!("  {m:4}x{n:5}x{k:5}  worst {worst:.3e}  max |y| {scale:.4}");
    }

    println!(
        "dense gemm fired {} shapes on {} ({vector} on the vector arm, {} on \
         the staged tile): worst relative error {worst_overall:.3e} against a \
         budget of {TOLERANCE:.3e}, exact on {exact} of {}",
        SHAPES.len(),
        device.name(),
        SHAPES.len() - vector,
        SHAPES.len(),
    );

    cache.clear(&device);
}

/// THE MUTATION FOR THE TILE ARM, so the agreement above is known to be
/// falsifiable.
///
/// A grid one workgroup short on the COLUMN axis, which on Vulkan is the
/// silent defect: the columns nothing covered are never written, every call
/// returns success, and no validation layer objects. So the last column tile
/// keeps its [`SENTINEL`], the comparison that passed above fails here, and
/// both facts are asserted.
///
/// `(96, 512, 128)` is the shape, because 512 is sixteen whole column tiles —
/// dropping one removes a full 32 columns rather than a ragged remainder, so
/// what the check sees is unambiguous.
#[test]
fn a_gemm_grid_one_column_tile_short_is_a_failure_this_check_can_see() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let (m, n, k) = (96usize, 512usize, 128usize);

    let got = fire_gemm(&device, &mut cache, m, n, k, [1, 0, 0]);
    let (want, scale) = reference(&activations(m, k), &weights(n, k), m, n, k);

    // The columns that WERE covered still agree, so what follows is
    // attributable to the missing workgroup and not to a broken fire.
    for r in 0..m {
        for c in 0..n - 32 {
            let i = r * n + c;
            let e = (f64::from(got[i]) - want[i]).abs() / scale;
            assert!(
                e <= TOLERANCE,
                "column {c} was inside the short grid and should be untouched \
                 by the mutation, but row {r} is off by {e:.3e}"
            );
        }
    }

    let sentinel = from_bf16(u16::from_le_bytes([SENTINEL, SENTINEL]));
    for r in 0..m {
        for c in n - 32..n {
            assert_eq!(
                got[r * n + c],
                sentinel,
                "row {r} column {c} is past the short grid and should still \
                 hold the sentinel"
            );
        }
    }

    let (worst, _) = worst_of(&got, &want, scale);
    assert!(
        worst > TOLERANCE,
        "the check in `a_dense_projection_this_device_computes_is_the_one_the_\
         reference_computes` would have PASSED on a grid one column tile short \
         -- worst relative error {worst:.3e} against a budget of \
         {TOLERANCE:.3e} -- so it is not evidence of anything"
    );
    println!(
        "one column tile short: the last 32 columns kept the sentinel and the \
         same comparison fails at {worst:.3e}"
    );

    cache.clear(&device);
}

/// THE MUTATION FOR THE VECTOR ARM, which the tile mutation above cannot
/// reach: below `TILE_M` a different module runs, with a different grid and a
/// different reduction, and a check that only ever falsified the tile would
/// say nothing about it.
///
/// `(1, 64, 128)` is decode's own shape. The mutation is one workgroup short
/// on the COLUMN axis again — `vector_lanes` puts columns on `y` — which drops
/// the last eight of the sixty-four.
#[test]
fn a_gemv_grid_one_group_short_is_a_failure_this_check_can_see() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let (m, n, k) = (1usize, 64usize, 128usize);

    let got = fire_gemm(&device, &mut cache, m, n, k, [0, 1, 0]);
    let (want, scale) = reference(&activations(m, k), &weights(n, k), m, n, k);

    for c in 0..n - 8 {
        let e = (f64::from(got[c]) - want[c]).abs() / scale;
        assert!(
            e <= TOLERANCE,
            "column {c} was inside the short grid and should be untouched by \
             the mutation, but is off by {e:.3e}"
        );
    }
    let sentinel = from_bf16(u16::from_le_bytes([SENTINEL, SENTINEL]));
    for (c, v) in got.iter().enumerate().skip(n - 8) {
        assert_eq!(
            *v, sentinel,
            "column {c} is past the short grid and should still hold the sentinel"
        );
    }

    let (worst, _) = worst_of(&got, &want, scale);
    assert!(
        worst > TOLERANCE,
        "the vector arm's half of the comparison would have PASSED one group \
         short -- {worst:.3e} against {TOLERANCE:.3e}"
    );
    println!("gemv one group short: the same comparison fails at {worst:.3e}");

    cache.clear(&device);
}

/// BOTH ROW CUTS, FIRED, AGAINST A REFERENCE THAT IS A PURE COPY.
///
/// These two move bits and compute nothing, so the reference is exact and the
/// assertion is BIT-EQUALITY rather than a tolerance. A tolerance here would
/// be a check that could not see a one-word shift in a slowly-varying row, and
/// [`act_at`] is built so that neighbouring elements are about two apart
/// precisely so that it could — but equality is stronger and available, so it
/// is what is asserted.
///
/// Widths are odd on purpose. 461 cut at 197 leaves 264; 53 columns of a
/// 7-layer relay is a 371-wide source row. None is a multiple of the 256-wide
/// workgroup, so the guard is exercised, and none is even, which is where the
/// wgpu sibling has to refuse and this plane does not — its bf16 is one
/// `uint16_t` per slot, so there is no word pairing to straddle.
#[test]
fn the_two_row_cuts_move_the_words_the_reference_moves() {
    let device = gpu!();
    let mut cache = Pipelines::new();

    // `layout.split_rows`: 13 rows of 461, cut at 197.
    {
        let (rows, total, ld) = (13usize, 461usize, 197usize);
        let rd = total - ld;
        let entrypoint = "split_rows_bfloat16";
        let (code, tier) = device.module_for(entrypoint).expect("a module");
        let push = push_i32(&[ld as i32, rd as i32]);
        let pipeline = cache
            .get(&device, entrypoint, code, push.len() as u32, 0, tier)
            .expect("the pipeline builds");
        let declared = pipeline.declared();
        assert_eq!(declared.bindings, 3, "src, left and right");
        assert_eq!(declared.push_offsets, vec![0, 4], "left_dim then right_dim");

        let src: Vec<f32> = (0..rows)
            .flat_map(|r| (0..total).map(move |i| act_at(r, i)))
            .collect();
        let left = device.empty((rows * ld * 2) as u64).expect("left");
        let right = device.empty((rows * rd * 2) as u64).expect("right");
        device
            .write(&left, &vec![SENTINEL; rows * ld * 2])
            .expect("sentinel");
        device
            .write(&right, &vec![SENTINEL; rows * rd * 2])
            .expect("sentinel");
        let bufs = [device.buffer(&bf16_bytes(&src)).expect("src"), left, right];
        let grid = groups([total as u32, rows as u32, 1], declared.local);
        let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
        device
            .run(pipeline, &bound, &push, grid)
            .expect("the dispatch");

        let lraw = device.read(&bufs[1]).expect("read left");
        let rraw = device.read(&bufs[2]).expect("read right");
        dump("vk-split-left.bin", &lraw);
        dump("vk-split-right.bin", &rraw);
        let l = bf16_read(&lraw);
        let r = bf16_read(&rraw);
        for row in 0..rows {
            for i in 0..ld {
                assert_eq!(
                    l[row * ld + i],
                    act_at(row, i),
                    "the left half at row {row} element {i}"
                );
            }
            for i in 0..rd {
                assert_eq!(
                    r[row * rd + i],
                    act_at(row, ld + i),
                    "the right half at row {row} element {i}"
                );
            }
        }
        println!(
            "split_rows fired {rows}x{total} cut at {ld} on {}: {} words \
             bit-exact",
            device.name(),
            rows * total
        );
        for b in bufs {
            device.free(b);
        }
    }

    // `layout.select`: 13 rows of a 7-layer, 53-wide relay, taking layer 5.
    {
        let (rows, layers, width) = (13usize, 7usize, 53usize);
        let stride = layers * width;
        let layer = 5usize;
        let offset = layer * width;
        let entrypoint = "select_slice_bfloat16";
        let (code, tier) = device.module_for(entrypoint).expect("a module");
        let push = push_i32(&[stride as i32, offset as i32, width as i32]);
        let pipeline = cache
            .get(&device, entrypoint, code, push.len() as u32, 0, tier)
            .expect("the pipeline builds");
        let declared = pipeline.declared();
        assert_eq!(declared.bindings, 2, "table and out_");
        assert_eq!(
            declared.push_offsets,
            vec![0, 4, 8],
            "stride, offset then width"
        );

        let table: Vec<f32> = (0..rows)
            .flat_map(|r| (0..stride).map(move |i| act_at(r, i)))
            .collect();
        let out = device.empty((rows * width * 2) as u64).expect("out");
        device
            .write(&out, &vec![SENTINEL; rows * width * 2])
            .expect("sentinel");
        let bufs = [device.buffer(&bf16_bytes(&table)).expect("table"), out];
        let grid = groups([width as u32, rows as u32, 1], declared.local);
        let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
        device
            .run(pipeline, &bound, &push, grid)
            .expect("the dispatch");

        let raw = device.read(&bufs[1]).expect("read back");
        dump("vk-select.bin", &raw);
        let y = bf16_read(&raw);
        for row in 0..rows {
            for i in 0..width {
                assert_eq!(
                    y[row * width + i],
                    act_at(row, offset + i),
                    "the selected slice at row {row} element {i}"
                );
            }
        }
        println!(
            "select_slice fired {rows}x{width} at layer {layer} of {layers} on \
             {}: {} words bit-exact",
            device.name(),
            rows * width
        );
        for b in bufs {
            device.free(b);
        }
    }

    cache.clear(&device);
}

/// THE MUTATION FOR THE TWO CUTS: one word late, which is the whole class of
/// defect a copy kernel has.
///
/// Both are fired again with the push block naming a slice one element further
/// into the source row. Nothing else differs. The `select` case is the honest
/// one to mutate — moving `offset` by one is exactly the arithmetic slip that
/// `stride`-versus-`width` confusion produces — and the `split_rows` case
/// moves the cut instead, which is the same slip on the other kernel.
#[test]
fn a_cut_one_word_late_is_a_failure_these_checks_can_see() {
    let device = gpu!();
    let mut cache = Pipelines::new();

    // `select`, one word late.
    let (rows, layers, width) = (13usize, 7usize, 53usize);
    let stride = layers * width;
    let offset = 5 * width + 1;
    let entrypoint = "select_slice_bfloat16";
    let (code, tier) = device.module_for(entrypoint).expect("a module");
    let push = push_i32(&[stride as i32, offset as i32, width as i32]);
    let pipeline = cache
        .get(&device, entrypoint, code, push.len() as u32, 0, tier)
        .expect("the pipeline builds");
    let table: Vec<f32> = (0..rows)
        .flat_map(|r| (0..stride).map(move |i| act_at(r, i)))
        .collect();
    let out = device.empty((rows * width * 2) as u64).expect("out");
    let bufs = [device.buffer(&bf16_bytes(&table)).expect("table"), out];
    let grid = groups([width as u32, rows as u32, 1], pipeline.declared().local);
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(pipeline, &bound, &push, grid)
        .expect("the dispatch");
    let y = bf16_read(&device.read(&bufs[1]).expect("read back"));

    let differ = (0..rows * width)
        .filter(|i| y[*i] != act_at(i / width, 5 * width + i % width))
        .count();
    assert!(
        differ > rows * width / 2,
        "a slice read one word late should differ almost everywhere, and \
         differs in {differ} of {}; the bit-equality check above is not \
         evidence of anything if this is small",
        rows * width
    );
    println!(
        "select one word late: {differ} of {} words differ",
        rows * width
    );
    for b in bufs {
        device.free(b);
    }

    // `split_rows`, with the cut one word late.
    let (total, ld) = (461usize, 198usize);
    let rd = total - ld;
    let entrypoint = "split_rows_bfloat16";
    let (code, tier) = device.module_for(entrypoint).expect("a module");
    let push = push_i32(&[ld as i32, rd as i32]);
    let pipeline = cache
        .get(&device, entrypoint, code, push.len() as u32, 0, tier)
        .expect("the pipeline builds");
    let src: Vec<f32> = (0..rows)
        .flat_map(|r| (0..total).map(move |i| act_at(r, i)))
        .collect();
    let left = device.empty((rows * ld * 2) as u64).expect("left");
    let right = device.empty((rows * rd * 2) as u64).expect("right");
    let bufs = [device.buffer(&bf16_bytes(&src)).expect("src"), left, right];
    let grid = groups([total as u32, rows as u32, 1], pipeline.declared().local);
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(pipeline, &bound, &push, grid)
        .expect("the dispatch");
    let r = bf16_read(&device.read(&bufs[2]).expect("read right"));

    // The right half is what a late cut shifts: every element of it comes from
    // one further into the source row than the true cut at 197 would give.
    let want_rd = total - 197;
    let differ = (0..rows)
        .flat_map(|row| (0..rd).map(move |i| (row, i)))
        .filter(|(row, i)| r[row * rd + i] != act_at(*row, 197 + i))
        .count();
    assert!(
        differ > rows * want_rd / 2,
        "a cut one word late should shift almost the whole right half, and \
         shifts {differ} of {}",
        rows * rd
    );
    println!(
        "split_rows one word late: {differ} of {} words differ",
        rows * rd
    );
    for b in bufs {
        device.free(b);
    }

    cache.clear(&device);
}

/// `norm.mul_scalar` FIRED, AND THE ROUNDING TRIP MEASURED RATHER THAN
/// ARGUED ABOUT.
///
/// The kernel is `out = bf16(f32(x) * f32(bf16(s)))`, and the inner
/// `f32(bf16(s))` is the whole difference between this point and `norm.scale`
/// — whose factor comes off a `[1]` bank and is ALREADY rounded, so rounding
/// it again would be a second rounding of a value with none left to lose.
///
/// This asserts BOTH halves, which is what makes it a measurement:
///
/// 1. the device agrees BIT-EXACTLY with the rounded reference; and
/// 2. the UNROUNDED reference — the same kernel with the trip deleted —
///    differs on a substantial count.
///
/// The second is the mutation. It needs no edited shader and no short grid: it
/// is the comparison itself run against the wrong model, and if the count came
/// back zero then the first assertion would be passing for a reason that has
/// nothing to do with the rounding.
///
/// The factor is gemma-4's own `embed_normalizer`, `sqrt(2048)`, which is the
/// number this whole trip exists for.
#[test]
fn mul_scalar_rounds_its_factor_through_bf16_and_it_is_measurable() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let (rows, width) = (13usize, 461usize);
    let n = rows * width;
    let s = 2048.0f32.sqrt();
    let entrypoint = "layer_scalar_mul_stated_bfloat16";

    let (code, tier) = device.module_for(entrypoint).expect("a module");
    let push: Vec<u8> = s.to_le_bytes().to_vec();
    let pipeline = cache
        .get(&device, entrypoint, code, push.len() as u32, 0, tier)
        .expect("the pipeline builds");
    let declared = pipeline.declared();
    assert_eq!(
        declared.bindings, 2,
        "x and out_ -- the stated arm has no `scalar` buffer, so `out_` is \
         binding 1 where the read arm has it at 2"
    );
    assert_eq!(
        declared.push_offsets,
        vec![0],
        "one f32, which is the factor"
    );

    let x: Vec<f32> = (0..n).map(fine_at).collect();
    assert_eq!(bf16_read(&bf16_bytes(&x)), x, "the inputs are bf16-exact");
    let out = device.empty((n * 2) as u64).expect("out");
    device
        .write(&out, &vec![SENTINEL; n * 2])
        .expect("sentinel");
    let bufs = [device.buffer(&bf16_bytes(&x)).expect("x"), out];
    let grid = groups([n as u32, 1, 1], declared.local);
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(pipeline, &bound, &push, grid)
        .expect("the dispatch");

    let raw = device.read(&bufs[1]).expect("read back");
    dump("vk-mulscalar.bin", &raw);
    let got = bf16_read(&raw);

    let rounded = from_bf16(to_bf16(s));
    let want_rounded: Vec<f32> = x.iter().map(|v| from_bf16(to_bf16(v * rounded))).collect();
    let want_raw: Vec<f32> = x.iter().map(|v| from_bf16(to_bf16(v * s))).collect();

    assert_eq!(
        got, want_rounded,
        "the device does not agree with the reference that rounds the factor \
         through bf16 first"
    );

    let elsewhere = (0..n).filter(|i| want_rounded[*i] != want_raw[*i]).count();
    assert!(
        elsewhere > 0,
        "the rounded and unrounded references agree everywhere at this factor, \
         so the assertion above measures nothing about the trip; pick a factor \
         that is not already a bf16"
    );
    println!(
        "mul_scalar fired {rows}x{width} by sqrt(2048) on {}: {n} of {n} words \
         bit-exact against the rounded reference, and {elsewhere} of {n} land \
         on a different bf16 without the trip",
        device.name()
    );

    for b in bufs {
        device.free(b);
    }
    cache.clear(&device);
}
