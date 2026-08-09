//! The device half, on a real adapter.
//!
//! # Why these are not `#[ignore]`
//!
//! An ignored test is one nobody runs, and a suite that needs a GPU is exactly
//! the suite whose failures matter — `.wiki/new-driver/vulkan.md` §11 is the
//! record of 54 green tests that were green because nothing was checking. So
//! these are gated on the `native` FEATURE, which a build box does not turn on,
//! and on a machine that does they run in a plain
//! `cargo test -p driver-wgpu --features native`.
//!
//! A third case exists and is handled separately: the feature is on and no
//! adapter answers. Each test then prints why and returns. That is not the same
//! as ignoring it — an ignored test is skipped on the machine that HAS the
//! hardware too.
//!
//! # Why every size here is odd
//!
//! `.wiki/new-driver/vulkan.md` §12. Three pointwise tests there ran at n = 512
//! against a 256-wide workgroup and two GEMV tests at 16 rows against a kernel
//! covering 8 — exact multiples, so the last partial group never existed and
//! `div_ceil` and plain division were the same expression. **13 rows of 460**
//! throughout here: 460 is not a multiple of the 1024-element chunk
//! `norm/rms.wgsl` walks a row in, 230 words is not a multiple of its 256-lane
//! store loop, and 13 rows is not a multiple of anything.
//!
//! One thing that finding does NOT reach on this backend, and it is worth
//! writing down rather than leaving as an apparent gap. A bf16 tensor crosses as
//! `array<u32>` with two values per word, and the launch rules compute lanes in
//! ELEMENTS — so every bf16 elementwise dispatch launches twice the lanes it has
//! words, and an undershot division is still covered by the doubling. The tail
//! of the DIVISION is therefore unreachable for those rows at any size. What is
//! reachable is the tail of the ROW COUNT, which `Rule::Rms` puts on the grid
//! directly, and
//! `an_undershot_grid_leaves_the_last_row_holding_its_sentinel` is that finding
//! done on a device rather than argued about.
//!
//! # Why the reference is computed from the bf16 and not from the f32
//!
//! Every input here is generated as `f32`, rounded to bf16 for the device, and
//! then **read back into the reference as the widened bf16**. Comparing against
//! the original `f32` would fold the input's own rounding into the tolerance and
//! quietly widen it by a bf16 ulp per operand, which for a 460-long reduction is
//! most of the budget.
//!
//! And the tolerance scales by the ROW's own largest magnitude, not by
//! `max(|want|, 1.0)`. The Vulkan suite found that floor of one turned a 2%
//! claim into a flat absolute 0.02, which is 7% of an attention value and 16% of
//! a router weight.

#![allow(clippy::print_stdout)]

use driver_wgpu::binding::Bound;
use driver_wgpu::device::{Buffer, Ceiling, Device, Failed, Pipelines, Recorded};
use driver_wgpu::geometry::Dims;
use driver_wgpu::resources::{Frame, Request, Shape};
use driver_wgpu::serve::{Embedded, pick};
use kernels_wgpu::Capability;
use std::sync::{Mutex, MutexGuard};

/// Rows every dispatch here fires over. Not a multiple of anything.
const ROWS: u32 = 13;

/// Elements in one row. Not a multiple of `norm/rms.wgsl`'s 1024-element chunk,
/// and 230 words is not a multiple of its 256-lane store loop.
const WIDTH: u32 = 460;

/// What an untouched byte holds.
///
/// `0x4780` is bf16 for `65536.0`, so a word of two of them is a value nothing
/// below can produce: every input here is in `[-2, 2)` and the widest thing done
/// to a pair of them is an add.
///
/// **Zero cannot be used**: it is what a fresh `wgpu` buffer already holds, so a
/// slot that was never written and one written with a zero are the same bytes,
/// and a dispatch that ran nothing would pass a check written against zero.
///
/// Nor can `-1.0`, which was the first choice and was WRONG in a way worth
/// recording: [`spread`] draws from `[-2, 2)` and one of its 600 values rounded
/// to exactly `-1.0`, so a reference built by marking the untouched slots
/// counted 599 where 600 were written. A sentinel has to be outside the range of
/// the data, not merely unlikely in it.
const SENTINEL: u32 = 0x4780_4780;

/// One device at a time, for the whole suite.
///
/// **Not a style choice — measured.** With `cargo test`'s default parallelism
/// this file opens ten `wgpu::Device`s at once, each of which is a `VkDevice`
/// with three driver-owned helper threads behind it, and roughly one run in
/// three then wedges: nine test threads parked on futexes, one spinning, no
/// progress for as long as it is left alone. It reproduces on the NVIDIA
/// proprietary driver with the process's own stacks showing the block inside
/// the driver rather than inside anything here, and it does not reproduce at
/// `--test-threads=1`.
///
/// So the suite takes a lock and opens the device under it. That is closer to a
/// real deployment anyway — a server has ONE device — and it makes the run
/// independent of `--test-threads`, which is worth more than the four seconds it
/// costs.
///
/// It has a second benefit that would justify it on its own: [`Device`]'s error
/// sink is per-device and [`Device::drained`] takes whatever is in it, so two
/// tests sharing a process could otherwise hand each other's validation failures
/// around. Under this lock, `every_refusal_is_a_named_error_and_not_a_panic` is
/// the only thing running when it provokes one.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// A device nothing else is using, or a printed reason there is none.
///
/// `PIE_WGPU_FALLBACK=1` asks for the SOFTWARE adapter instead, which is how
/// this whole suite is run a second time against a completely different
/// implementation of the same WGSL on the same machine. `WGPU_POWER_PREF` picks
/// between two hardware adapters. Neither is a deployment knob; both exist
/// because "it agrees on the card it was written on" is the weakest form of
/// agreement there is.
///
/// The device is field ZERO of the pair so that it drops BEFORE the lock does —
/// tuple fields drop in declaration order — which is what keeps the "one at a
/// time" true across the teardown as well as the run.
fn adapter() -> Option<(Device, MutexGuard<'static, ()>)> {
    let held = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let opened = if std::env::var("PIE_WGPU_FALLBACK").is_ok() {
        Device::software()
    } else {
        Device::open()
    };
    match opened {
        Ok(device) => Some((device, held)),
        Err(why) => {
            println!("SKIP: {why}");
            None
        }
    }
}

/// Round an `f32` to a bf16 bit pattern, round to nearest even.
///
/// The host copy of `pie_f32_to_bf16` in `kernels/common/bf16.inc.wgsl`,
/// including its NaN branch. Written out rather than truncated because
/// truncating is a real accuracy loss over a long accumulation and because a
/// reference that rounds differently from the shader would spend the whole
/// tolerance on the rounding.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// Widen a bf16 bit pattern. Exact: bf16 IS the top half of an f32.
///
/// By SHIFT and not by cast, which is the same rule `serve::logits` states: `v
/// as f32` would turn `0x3f80` into 16256.0 where it means 1.0, and both are
/// finite floats no assertion downstream would object to.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// A run of `f32` as the packed bf16 words a shader reads, plus what the shader
/// will actually see.
///
/// Two answers from one call on purpose: the second is what every reference
/// below is computed from. See the module docs.
fn pack(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let rounded: Vec<u16> = values.iter().copied().map(to_bf16).collect();
    let seen: Vec<f32> = rounded.iter().copied().map(from_bf16).collect();
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in &rounded {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    // A whole number of words, since the shaders address `array<u32>`.
    if bytes.len() % 4 != 0 {
        bytes.extend_from_slice(&[0, 0]);
    }
    (bytes, seen)
}

/// Bytes back as the bf16 values they hold.
fn unpack(bytes: &[u8], n: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .take(n)
        .map(|c| from_bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// A spread of values that is not symmetric, not sorted and not near zero.
///
/// Seeded by a plain LCG so the same shape produces the same numbers on every
/// machine, which is what makes a disagreement between two adapters a finding
/// rather than a coincidence.
fn spread(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // In [-2, 2), which keeps every value and every partial sum well
            // clear of the denormal range -- a flush-to-zero adapter and a
            // conforming one must not be allowed to disagree about the answer
            // for a reason that is not the kernel's.
            (state >> 8) as f32 / 4_194_304.0 - 2.0
        })
        .collect()
}

/// Compare one row against its reference, on TWO claims at once.
///
/// `Err` rather than a panic so that the perturbation control below can assert
/// this FAILS — a check that has never failed has not been shown to check
/// anything.
///
/// # The first claim: no element is far out
///
/// Scaled by `max|want|` over the ROW, and the budget is two bf16 ulps of it. A
/// bf16 has an eight-bit significand, so the quantum at magnitude `M` is
/// `M/128`; two of them is what a rounded output can cost. NOT
/// `max(|want|, 1.0)`: the Vulkan suite found that floor of one turned a 2%
/// claim into a flat absolute 0.02, which is 7% of an attention value.
///
/// # The second claim, which is the one with teeth
///
/// That claim ALONE is far too weak, and it was measured rather than reasoned
/// about: a reference that divided by `axis - 1` instead of `axis` — a 0.1%
/// error in the norm, and exactly the kind of off-by-one a port introduces —
/// **passed it**. It has to, because 0.1% is well inside a bf16 half-ulp, so no
/// per-element absolute bound scaled by the row's magnitude can see it.
///
/// What can see it is the COUNT. Both sides round to bf16 the same way and their
/// pre-rounding values differ only by the reduction order and by WGSL's two-ulp
/// allowance on `inverseSqrt`, which is about `1e-6` relative — so an element
/// lands on a different bf16 value only if it sat within `1e-6` of a rounding
/// boundary, which is roughly one element in eight thousand. A systematic 0.1%
/// shift moves every element by an eighth of an ulp and flips something like one
/// in eight. So: at most one element in fifty may differ by more than the
/// rounding noise, and the `axis - 1` reference fails that by an order of
/// magnitude.
fn agrees(got: &[f32], want: &[f32], what: &str) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} values back and {} expected",
            got.len(),
            want.len()
        ));
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let budget = (scale / 128.0).max(f32::MIN_POSITIVE);
    // Below this two f32 values that were about to be rounded to the same bf16
    // are indistinguishable, so a difference under it is the reduction order and
    // not an answer.
    let noise = (scale * 1e-5).max(f32::MIN_POSITIVE);
    let mut moved = 0usize;
    for (at, (g, w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() || (g - w).abs() > budget {
            return Err(format!(
                "{what}: element {at} is {g} and should be {w}; the row's \
                 largest magnitude is {scale} and the budget is {budget}"
            ));
        }
        if (g - w).abs() > noise {
            moved += 1;
        }
    }
    if moved * 50 > got.len() {
        return Err(format!(
            "{what}: {moved} of {} elements landed on a different bf16 value \
             than the reference. Each one is inside the per-element budget, \
             which is why this count exists: two computations that agree \
             differ only where an element sat within a rounding boundary of \
             the other, and that is about one in eight thousand rather than \
             one in {}",
            got.len(),
            got.len() / moved.max(1)
        ));
    }
    Ok(())
}

/// `RmsParams` as `norm/rms.wgsl` declares it: five words at binding 3.
///
/// A STORAGE block and not the uniform one, which is what the row means by
/// `params: Buf` — a struct is a struct, and moving it into `@group(1)` would be
/// changing the kernel's ABI from the driver.
fn rms_params(eps: f32, axis: u32, w_stride: u32, plus_one: u32, gain: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&eps.to_le_bits_le());
    out.extend_from_slice(&axis.to_le_bytes());
    out.extend_from_slice(&w_stride.to_le_bytes());
    out.extend_from_slice(&plus_one.to_le_bytes());
    out.extend_from_slice(&gain.to_le_bits_le());
    out
}

/// `f32` little-endian, spelled as a method so `rms_params` reads as a run of
/// fields rather than a run of conversions.
trait LeBits {
    fn to_le_bits_le(self) -> [u8; 4];
}

impl LeBits for f32 {
    fn to_le_bits_le(self) -> [u8; 4] {
        self.to_bits().to_le_bytes()
    }
}

/// What `norm/rms.wgsl` computes for one row, from the bf16 the device was
/// given.
fn rms_reference(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let axis = x.len();
    let total: f32 = x.iter().map(|v| v * v).sum();
    let inv = (total / axis as f32 + eps).sqrt().recip();
    x.iter()
        .zip(w)
        .map(|(xi, wi)| from_bf16(to_bf16(wi * (xi * inv))))
        .collect()
}

/// 1. An adapter opens and says what it is.
///
/// Printed rather than asserted, because the useful content is the numbers
/// themselves: which adapter, over which native API, with which limits. The two
/// assertions are the ones that would make every other test in this file
/// meaningless if they failed.
#[test]
fn an_adapter_opens_and_reports_what_it_offers() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let info = device.info();
    let limits = device.limits();
    for (at, other) in device.adapters().iter().enumerate() {
        println!(
            "visible {at}    {} ({:?}, {:?})",
            other.name, other.backend, other.device_type
        );
    }
    println!("adapter      {}", info.name);
    println!("backend      {:?}", info.backend);
    println!("kind         {:?}", info.device_type);
    println!("driver       {} {}", info.driver, info.driver_info);
    println!("unified      {}", device.unified());
    println!("limits       {limits}");
    println!("tiers        {:?}", device.tiers());
    println!("features     {:?}", device.features());
    println!("unreachable  {:?}", device.unreachable());
    println!(
        "downlevel    {} rows need more than the guaranteed {} storage buffers",
        kernels_wgpu::over_downlevel_storage_limit().len(),
        kernels_wgpu::DOWNLEVEL_STORAGE_BUFFERS
    );

    assert!(
        !device.tiers().is_empty(),
        "an adapter always reports at least Baseline, which requires nothing"
    );
    assert_eq!(
        device.tiers().last(),
        Some(&Capability::Baseline),
        "the tier walk has to end somewhere every adapter can reach"
    );
    // The WebGPU floors. An adapter reporting less than the specification
    // guarantees is not one this driver's arithmetic holds for -- `binding.rs`
    // refuses an offset against 256 and `geometry.rs` refuses a grid against
    // 65535, both from the spec rather than from a card.
    assert!(limits.storage_offset >= 1 && limits.storage_offset <= 256);
    assert!(limits.uniform_offset >= 1 && limits.uniform_offset <= 256);
    assert!(limits.workgroups_per_dimension >= 65535);
    assert!(limits.storage_buffers >= kernels_wgpu::DOWNLEVEL_STORAGE_BUFFERS);
}

/// 2a. A real `rms_single_row_bfloat16` produces the right numbers.
///
/// The clearest row in the table to check: `out = w * x / rms(x)` has an
/// unambiguous closed form, its params ride a STORAGE buffer rather than the
/// uniform block — so this is the path `binding::ParamSlot::Storage` describes —
/// and its grid is one workgroup per row, so an undershoot drops a whole row
/// rather than a lane.
#[test]
fn a_norm_computes_what_its_closed_form_says() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let name = "rms_single_row_bfloat16";
    let (source, tier) = pick(&Embedded, name, Capability::Baseline)
        .expect("the tree holds the norm every text states");
    let pipeline = cache
        .get(&device, name, tier, &source)
        .expect("the norm builds");

    let eps = 1e-6f32;
    let n = (ROWS * WIDTH) as usize;
    let (x_bytes, x_seen) = pack(&spread(n, 7));
    let (w_bytes, w_seen) = pack(&spread(WIDTH as usize, 91));
    let x = device.buffer(&x_bytes).expect("x");
    let w = device.buffer(&w_bytes).expect("w");
    let out = device.buffer(&vec![0u8; x_bytes.len()]).expect("out");
    let params = device
        .buffer(&rms_params(eps, WIDTH, 1, 0, 1.0))
        .expect("params");

    let groups = driver_wgpu::device::groups_for(
        &device,
        pipeline,
        kernels_wgpu::sig(name).expect("the row").launch,
        Dims {
            rows: ROWS,
            width: WIDTH,
            axis: WIDTH,
            ..Dims::default()
        },
    )
    .expect("a grid");
    assert_eq!(groups, [ROWS, 1, 1], "one workgroup per row, on x");

    device
        .run(
            pipeline,
            &[
                Bound::whole(&x),
                Bound::whole(&w),
                Bound::whole(&out),
                Bound::whole(&params),
            ],
            &[],
            groups,
        )
        .expect("the norm ran");

    let got = unpack(&device.read(&out).expect("readback"), n);
    for row in 0..ROWS as usize {
        let span = row * WIDTH as usize..(row + 1) * WIDTH as usize;
        let want = rms_reference(&x_seen[span.clone()], &w_seen, eps);
        agrees(&got[span], &want, &format!("row {row}")).expect("the norm agrees");
    }
}

/// 2b. And so does `residual_add_bfloat16`.
///
/// The other closed form worth checking, and it is the one that says the bf16
/// PACKING is right: every invocation writes a whole word of two values, so a
/// half-index off by one would shift the entire tensor by one element and still
/// produce finite numbers everywhere.
#[test]
fn a_residual_add_is_the_sum_of_what_it_was_given() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let name = "residual_add_bfloat16";
    let (source, tier) = pick(&Embedded, name, Capability::Baseline).expect("the tree holds it");
    let pipeline = cache.get(&device, name, tier, &source).expect("it builds");

    let n = (ROWS * WIDTH) as usize;
    let (x_bytes, x_seen) = pack(&spread(n, 13));
    let (r_bytes, r_seen) = pack(&spread(n, 29));
    let x = device.buffer(&x_bytes).expect("x");
    let r = device.buffer(&r_bytes).expect("residual");
    // Born holding the sentinel, so a lane that never ran is distinguishable
    // from one that wrote a zero.
    let out = device
        .buffer(&SENTINEL.to_le_bytes().repeat(x_bytes.len() / 4))
        .expect("out");

    let groups = driver_wgpu::device::groups_for(
        &device,
        pipeline,
        kernels_wgpu::sig(name).expect("the row").launch,
        Dims {
            rows: ROWS,
            width: WIDTH,
            ..Dims::default()
        },
    )
    .expect("a grid");

    device
        .run(
            pipeline,
            &[Bound::whole(&x), Bound::whole(&r), Bound::whole(&out)],
            &[],
            groups,
        )
        .expect("the add ran");

    let got = unpack(&device.read(&out).expect("readback"), n);
    let want: Vec<f32> = x_seen
        .iter()
        .zip(&r_seen)
        .map(|(a, b)| from_bf16(to_bf16(a + b)))
        .collect();
    agrees(&got, &want, "the whole tensor").expect("the add agrees");
}

/// The perturbation control: the check above FAILS when its reference moves,
/// on both of its claims.
///
/// A test that has never failed has not been shown to test anything. Needs no
/// adapter, which is the point: it is a claim about the CHECK.
#[test]
fn a_perturbed_reference_is_refused_by_the_same_check() {
    let want = spread(WIDTH as usize, 3);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    agrees(&want, &want, "itself").expect("a row agrees with itself");

    // The first claim: one element far out.
    let mut moved = want.clone();
    moved[17] += scale / 40.0;
    agrees(&moved, &want, "perturbed").expect_err(
        "a value three bf16 ulps of the row's scale away from its reference \
         must be refused, or the per-element bound is not measuring anything",
    );

    // And the scaling is by the ROW and not by a floor of one. A row whose
    // values are all small must still refuse a small error -- which is exactly
    // what `max(|want|, 1.0)` would wave through.
    let small: Vec<f32> = want.iter().map(|v| v / 1000.0).collect();
    let mut nudged = small.clone();
    nudged[3] += scale / 40_000.0;
    agrees(&nudged, &small, "small and perturbed").expect_err(
        "a floor of 1.0 in the tolerance would accept this, which is the \
         defect the Vulkan suite found",
    );

    // The second claim: a SYSTEMATIC shift too small for any per-element bound
    // to see. Every value moved by a tenth of a percent -- an `axis - 1` in a
    // norm -- is inside the budget everywhere and must still be refused, on the
    // count.
    let drifted: Vec<f32> = want.iter().map(|v| v * 1.001).collect();
    let far = drifted
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        far < scale / 128.0,
        "this perturbation is supposed to be INSIDE the per-element budget, so \
         that it is the count that refuses it and not the bound"
    );
    agrees(&drifted, &want, "drifted").expect_err(
        "a systematic tenth of a percent is what an off-by-one in a reduction \
         width looks like, and no per-element bound scaled by a bf16 row can \
         see it",
    );

    // A NaN is refused whatever the budget: a kernel that produced one has not
    // produced a slightly wrong number.
    let mut bad = want.clone();
    bad[0] = f32::NAN;
    agrees(&bad, &want, "nan").expect_err("a NaN is not within any tolerance");
}

/// 3. `run` and `run_all` agree byte for byte over a chained plan.
///
/// **The most valuable test in the device half**, and its meaning here is not
/// its meaning next door. On Vulkan the two paths differ in whether the shell
/// wrote a barrier, so a disagreement is a missing barrier. `wgpu` writes the
/// barrier itself — before every dispatch that reuses a buffer in an exclusive
/// state, at every encoding granularity, which `device.rs`'s module docs cite
/// line by line — so the two paths are ordered identically and a disagreement
/// would be a defect in `wgpu`'s tracker or a hazard through memory it cannot
/// see.
///
/// That is the one claim in this half taken from reading somebody else's source
/// rather than from running something, so this is what runs it. Eight dispatches
/// chained head to tail, alternating a norm and an add, so every dispatch but
/// the first reads what the one before it wrote.
///
/// # Why the arena is a PAIR
///
/// Because one buffer would be refused. WebGPU will not let a dispatch bind one
/// allocation both readable and writable, so a stage reading region *k-1* and
/// writing region *k* of one arena is illegal — see `device.rs`'s own section on
/// it, which is the largest finding in this half. `Device::run_all` handles it
/// by shadowing the read side into a scratch buffer, and this test deliberately
/// does NOT lean on that: it reads from one arena and writes to the other, so
/// the run needs no copies and what is compared is the ORDERING and nothing
/// else. `an_arena_bound_both_ways_is_diagnosed_by_name_and_run_anyway` is where
/// the shadow is exercised.
#[test]
fn one_pass_and_one_submission_each_agree_over_a_chained_plan() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    for name in ["rms_single_row_bfloat16", "residual_add_bfloat16"] {
        let (source, tier) =
            pick(&Embedded, name, Capability::Baseline).expect("the tree holds it");
        cache.get(&device, name, tier, &source).expect("it builds");
    }
    let norm = cache
        .peek("rms_single_row_bfloat16", Capability::Baseline)
        .expect("built");
    let add = cache
        .peek("residual_add_bfloat16", Capability::Baseline)
        .expect("built");

    let n = (ROWS * WIDTH) as usize;
    let (x_bytes, _) = pack(&spread(n, 41));
    let (w_bytes, _) = pack(&spread(WIDTH as usize, 53));
    let (r_bytes, _) = pack(&spread(n, 67));
    let x = device.buffer(&x_bytes).expect("x");
    let w = device.buffer(&w_bytes).expect("w");
    let residual = device.buffer(&r_bytes).expect("residual");
    let params = device
        .buffer(&rms_params(1e-6, WIDTH, 1, 0, 1.0))
        .expect("params");

    let norm_grid = driver_wgpu::device::groups_for(
        &device,
        norm,
        kernels::LaunchRule::Rms,
        Dims {
            rows: ROWS,
            width: WIDTH,
            axis: WIDTH,
            ..Dims::default()
        },
    )
    .expect("a grid");
    let add_grid = driver_wgpu::device::groups_for(
        &device,
        add,
        kernels::LaunchRule::Elementwise,
        Dims {
            rows: ROWS,
            width: WIDTH,
            ..Dims::default()
        },
    )
    .expect("a grid");

    // Four regions in each half of the ping-pong, at offsets a storage binding
    // can start from. `min_storage_offset` is ASKED rather than assumed: 256 is
    // the specification's floor and an adapter may want more -- this one reports
    // 32, so a test that hard-coded 256 would be testing a coarser layout than
    // the driver will ever see.
    let align = device.min_storage_offset();
    let span = x_bytes.len() as u64;
    let stride = span.next_multiple_of(align);
    let stages = 8usize;
    let regions = stages.div_ceil(2) as u64;
    let fill = SENTINEL
        .to_le_bytes()
        .repeat((stride * regions) as usize / 4);

    let run = |a: &Buffer, b: &Buffer, all_at_once: bool| -> usize {
        let at = |arena: &'static Buffer, k: usize| {
            Bound::within(arena, stride * (k / 2) as u64, span, align).expect("a region")
        };
        // Leaked so the borrows live as long as the recorded run; a test binary
        // owns them until it exits either way.
        let a: &'static Buffer = Box::leak(Box::new(a.clone()));
        let b: &'static Buffer = Box::leak(Box::new(b.clone()));
        let x: &'static Buffer = Box::leak(Box::new(x.clone()));
        let w: &'static Buffer = Box::leak(Box::new(w.clone()));
        let r: &'static Buffer = Box::leak(Box::new(residual.clone()));
        let p: &'static Buffer = Box::leak(Box::new(params.clone()));
        // Stage k reads the arena it did NOT write and writes the other, so no
        // dispatch binds one allocation both ways.
        let bounds: Vec<[Bound<'static, Buffer>; 4]> = (0..stages)
            .map(|k| {
                let input = if k == 0 {
                    Bound::whole(x)
                } else if k % 2 == 1 {
                    at(a, k - 1)
                } else {
                    at(b, k - 1)
                };
                let out = if k % 2 == 0 { at(a, k) } else { at(b, k) };
                if k % 2 == 0 {
                    [input, Bound::whole(w), out, Bound::whole(p)]
                } else {
                    [input, Bound::whole(r), out, Bound::whole(p)]
                }
            })
            .collect();
        let recorded: Vec<Recorded<'_, '_>> = bounds
            .iter()
            .enumerate()
            .map(|(k, bound)| Recorded {
                pipeline: if k % 2 == 0 { norm } else { add },
                buffers: if k % 2 == 0 { &bound[..] } else { &bound[..3] },
                uniform: &[],
                groups: if k % 2 == 0 { norm_grid } else { add_grid },
            })
            .collect();
        if all_at_once {
            device.run_all(&recorded).expect("the whole plan ran")
        } else {
            for one in &recorded {
                device
                    .run(one.pipeline, one.buffers, one.uniform, one.groups)
                    .expect("a stage ran");
            }
            0
        }
    };

    let slow_a = device.buffer(&fill).expect("arena");
    let slow_b = device.buffer(&fill).expect("arena");
    run(&slow_a, &slow_b, false);

    let fast_a = device.buffer(&fill).expect("arena");
    let fast_b = device.buffer(&fill).expect("arena");
    let shadowed = run(&fast_a, &fast_b, true);
    assert_eq!(
        shadowed, 0,
        "this plan reads and writes different allocations, so nothing should \
         have needed shadowing -- if it did, the two paths are not comparable"
    );

    // The LAST stage is 7, which is odd, so it wrote arena b at region 3.
    let last = (stride * 3) as usize;
    for (which, one, all) in [("a", &slow_a, &fast_a), ("b", &slow_b, &fast_b)] {
        let one = device.read(one).expect("readback");
        let all = device.read(all).expect("readback");
        assert_ne!(
            &all[last..last + 4],
            &SENTINEL.to_le_bytes(),
            "arena {which}'s last region was never written, so this compares \
             two plans that both did nothing"
        );
        let differs = one.iter().zip(&all).position(|(p, q)| p != q);
        assert!(
            differs.is_none(),
            "the two paths disagree in arena {which} from byte {} on, over \
             {stages} chained dispatches",
            differs.unwrap_or(0)
        );
    }
}

/// The arena bound both ways is diagnosed by name, and run anyway.
///
/// **The largest divergence in the device half**, and it is a fact about WebGPU
/// rather than about `wgpu`: a dispatch is one usage scope, and a buffer in a
/// usage scope may carry any number of readable usages or exactly one writable
/// one, never both. Disjoint ranges do not help — a buffer has no subresources.
/// Every launch of every real plan does this, since the plan's input and its
/// output are two ranges of one arena.
///
/// So this checks both halves of the answer. [`Device::check`] names it, with
/// the two bindings and the two offsets, which is the diagnosis a caller wants;
/// [`Device::run_all`] shadows the read side into a scratch buffer and produces
/// the right numbers, which is what makes the driver usable at all. The numbers
/// are compared against the SAME closed form the un-aliased norm is checked
/// against, so "it ran" and "it was right" are one assertion.
#[test]
fn an_arena_bound_both_ways_is_diagnosed_by_name_and_run_anyway() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let name = "rms_single_row_bfloat16";
    let (source, tier) = pick(&Embedded, name, Capability::Baseline).expect("the tree holds it");
    let pipeline = cache.get(&device, name, tier, &source).expect("it builds");

    let eps = 1e-6f32;
    let n = (ROWS * WIDTH) as usize;
    let (x_bytes, x_seen) = pack(&spread(n, 23));
    let (w_bytes, w_seen) = pack(&spread(WIDTH as usize, 31));
    let w = device.buffer(&w_bytes).expect("w");
    let params = device
        .buffer(&rms_params(eps, WIDTH, 1, 0, 1.0))
        .expect("params");

    // ONE arena, input at 0 and output one region along -- which is the shape
    // `binding::Arena` produces and the shape WebGPU refuses.
    let align = device.min_storage_offset();
    let span = x_bytes.len() as u64;
    let stride = span.next_multiple_of(align);
    let mut fill = SENTINEL.to_le_bytes().repeat((stride * 2) as usize / 4);
    fill[..x_bytes.len()].copy_from_slice(&x_bytes);
    let arena = device.buffer(&fill).expect("arena");

    let input = Bound::within(&arena, 0, span, align).expect("the input range");
    let output = Bound::within(&arena, stride, span, align).expect("the output range");
    let one = Recorded {
        pipeline,
        buffers: &[input, Bound::whole(&w), output, Bound::whole(&params)],
        uniform: &[],
        groups: [ROWS, 1, 1],
    };

    // The diagnosis, by name and with both offsets.
    assert_eq!(
        device.check(&one),
        Err(Failed::Aliased {
            reader: 0,
            writer: 2,
            read_at: 0,
            write_at: stride,
        }),
        "the arena bound both ways has to be nameable, or a caller meets it as \
         a wgpu string about usage bits"
    );

    // And it runs, through the shadow, with the right numbers.
    let shadowed = device
        .run_all(&[one])
        .expect("the shadow made it dispatchable");
    assert_eq!(
        shadowed, 1,
        "one read operand shares the arena with the write, so exactly one \
         range should have been copied"
    );
    let got = unpack(&device.read_at(&arena, stride, span).expect("readback"), n);
    for row in 0..ROWS as usize {
        let at = row * WIDTH as usize..(row + 1) * WIDTH as usize;
        let want = rms_reference(&x_seen[at.clone()], &w_seen, eps);
        agrees(&got[at], &want, &format!("row {row}")).expect("the shadowed norm agrees");
    }
}

/// A grid one short leaves the last row holding its sentinel, and reports
/// success.
///
/// `.wiki/new-driver/vulkan.md` §12's finding, done on a device rather than
/// argued about: an undershot grid writes nothing, the gap reads back as
/// whatever the buffer was born with, and every call in the chain returns
/// success. 13 rows is what makes it visible — at 16 against a rule that
/// happened to divide, the same slip would be invisible.
///
/// This is also the only place the difference between `div_ceil` and plain
/// division is observable on this backend for these kernels; see the module
/// docs for why the bf16 packing hides it everywhere else.
#[test]
fn an_undershot_grid_leaves_the_last_row_holding_its_sentinel() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let name = "rms_single_row_bfloat16";
    let (source, tier) = pick(&Embedded, name, Capability::Baseline).expect("the tree holds it");
    let pipeline = cache.get(&device, name, tier, &source).expect("it builds");

    let n = (ROWS * WIDTH) as usize;
    let (x_bytes, _) = pack(&spread(n, 5));
    let (w_bytes, _) = pack(&spread(WIDTH as usize, 61));
    let x = device.buffer(&x_bytes).expect("x");
    let w = device.buffer(&w_bytes).expect("w");
    let params = device
        .buffer(&rms_params(1e-6, WIDTH, 1, 0, 1.0))
        .expect("params");
    let out = device
        .buffer(&SENTINEL.to_le_bytes().repeat(x_bytes.len() / 4))
        .expect("out");

    // ONE SHORT, on purpose.
    device
        .run(
            pipeline,
            &[
                Bound::whole(&x),
                Bound::whole(&w),
                Bound::whole(&out),
                Bound::whole(&params),
            ],
            &[],
            [ROWS - 1, 1, 1],
        )
        .expect("an undershot grid is legal and reports success");

    let got = device.read(&out).expect("readback");
    let word = |row: u32| {
        let at = (row * WIDTH) as usize * 2;
        u32::from_le_bytes([got[at], got[at + 1], got[at + 2], got[at + 3]])
    };
    for row in 0..ROWS - 1 {
        assert_ne!(word(row), SENTINEL, "row {row} should have been written");
    }
    assert_eq!(
        word(ROWS - 1),
        SENTINEL,
        "the last row was left untouched and nothing said so -- which is the \
         whole finding, and it is only visible because {ROWS} is not a round \
         number and because the buffer was born holding something no kernel \
         produces"
    );
}

/// 4. A KV append lands exactly where `Shape::slot` says, and nowhere else.
///
/// The sentinel is the whole design of this test. The cache is filled with
/// `-1.0` before the dispatch and every element of it is checked afterwards, so
/// "the appended rows are right" and "nothing else moved" are one assertion.
/// With zeros, an append that scattered into the wrong pages would be
/// indistinguishable from one that wrote zeros there — and a paged cache's
/// characteristic failure is exactly a write into somebody else's page.
///
/// The tables come from [`Frame::of`] rather than being filled by hand, because
/// `kv_write_page` is `kv_page_indices` indexed through the CSR and filling them
/// separately is filling them from six chances to be inconsistent.
#[test]
fn a_paged_append_lands_where_the_layout_says_and_leaves_the_rest_alone() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let name = "kv_append_paged_bfloat16";
    let (source, tier) =
        pick(&Embedded, name, Capability::Baseline).expect("the tree holds the paged append");
    let pipeline = cache.get(&device, name, tier, &source).expect("it builds");
    // The module declares 0, 1, 2, 3, 10 and 11 and nothing between: a hole is
    // legal on this backend and the layout simply does not have an entry for it.
    // On Vulkan the same module needs a descriptor at every number up to the
    // highest and the shell has to find something to put there.
    assert_eq!(
        pipeline.slots(),
        &[0, 1, 2, 3, 10, 11],
        "the layout is what the module declares and reads, holes and all"
    );

    // 13 pages of 16 rows, 3 heads of 10 channels: none of them a multiple of
    // another, and the head width even because a bf16 pair shares a word.
    let shape = Shape {
        layers: 1,
        kv_heads: 3,
        head_dim: 10,
        page_size: 16,
        pages: 13,
        bytes: 2,
    };
    // Two conversations, one of them spilling across its own pages, on pages
    // that are neither contiguous nor ascending -- which is the entire point of
    // a paged cache and the case a linear layout would pass anyway.
    let frame = Frame::of(
        shape,
        &[
            Request::of((0..19).collect(), vec![7, 2]),
            Request::of(vec![40], vec![11, 5, 3]),
        ],
    )
    .expect("a stageable fire");
    let rows = frame.rows() as u32;
    assert_eq!(rows, 20);

    let row_stride = (shape.kv_heads * shape.head_dim) as usize;
    let (k_bytes, k_seen) = pack(&spread(rows as usize * row_stride, 101));
    let (v_bytes, v_seen) = pack(&spread(rows as usize * row_stride, 103));
    let k_new = device.buffer(&k_bytes).expect("k_new");
    let v_new = device.buffer(&v_bytes).expect("v_new");
    let w_page = device.words(&frame.kv_write_page).expect("w_page");
    let w_off = device.words(&frame.kv_write_offset).expect("w_off");

    let elements = shape.elements() as usize;
    let fill = SENTINEL.to_le_bytes().repeat(elements / 2);
    let k_dst = device.buffer(&fill).expect("k cache");
    let v_dst = device.buffer(&fill).expect("v cache");

    // The uniform block, at the offsets `kernels_wgpu::uniform_layout` states
    // rather than packed end to end -- which for this row is the same thing and
    // is asked of the table anyway, because the row after it is not.
    let sig = kernels_wgpu::sig(name).expect("the row");
    let layout = kernels_wgpu::uniform_layout(sig);
    let mut uniform = vec![0u8; kernels_wgpu::uniform_size(sig) as usize];
    for (field, value) in layout
        .iter()
        .zip([shape.head_dim, shape.page_size, shape.kv_heads])
    {
        let at = field.offset as usize;
        uniform[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    assert_eq!(uniform.len(), 16, "three words, rounded to WGSL's 16");

    let groups = driver_wgpu::device::groups_for(
        &device,
        pipeline,
        sig.launch,
        Dims {
            rows,
            head_dim: shape.head_dim,
            kv_heads: shape.kv_heads,
            ..Dims::default()
        },
    )
    .expect("a grid");
    assert_eq!(groups, [1, shape.kv_heads, rows]);

    device
        .run(
            pipeline,
            &[
                Bound::whole(&k_new),
                Bound::whole(&v_new),
                Bound::whole(&k_dst),
                Bound::whole(&v_dst),
                Bound::whole(&w_page),
                Bound::whole(&w_off),
            ],
            &uniform,
            groups,
        )
        .expect("the append ran");

    // What SHOULD be where, computed from the layout formula rather than from
    // the shader's own arithmetic said twice.
    let mut want_k = vec![from_bf16(SENTINEL as u16); elements];
    let mut want_v = want_k.clone();
    for token in 0..rows {
        let page = frame.kv_write_page[token as usize];
        let off = frame.kv_write_offset[token as usize];
        for head in 0..shape.kv_heads {
            for at in 0..shape.head_dim {
                let dst = shape.slot(page, off, head, at) as usize;
                let src = token as usize * row_stride + (head * shape.head_dim + at) as usize;
                want_k[dst] = k_seen[src];
                want_v[dst] = v_seen[src];
            }
        }
    }

    for (side, buffer, want) in [("keys", &k_dst, &want_k), ("values", &v_dst, &want_v)] {
        let got = unpack(&device.read(buffer).expect("readback"), elements);
        let touched = want
            .iter()
            .filter(|v| **v != from_bf16(SENTINEL as u16))
            .count();
        assert_eq!(
            touched,
            rows as usize * row_stride,
            "the {side} reference does not cover every appended element"
        );
        assert!(
            touched < elements,
            "every slot of the {side} cache is expected to change, so an \
             append that scattered everywhere would pass"
        );
        // Element by element and not row by row, because the failure this is
        // written for is a write into ANOTHER page -- which is a correct value
        // at an address that belongs to somebody else.
        for (at, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g, w,
                "{side} element {at} is {g} and should be {w}; the append is \
                 exact, so this is an address and not a tolerance"
            );
        }
    }
}

/// 5. Every refusal this half defines is a named error and not a panic.
///
/// `wgpu`'s own default handler is `panic!("wgpu error: {err}")`, so the last
/// case here — a buffer bound at a slot its usage flags do not allow — is the
/// one that proves `Device::open`'s `on_uncaptured_error` is doing its job. A
/// driver whose refusals are panics has no refusals.
#[test]
fn every_refusal_is_a_named_error_and_not_a_panic() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let name = "rms_single_row_bfloat16";
    let (source, tier) = pick(&Embedded, name, Capability::Baseline).expect("the tree holds it");
    let pipeline = cache.get(&device, name, tier, &source).expect("it builds");

    let small = device.buffer(&[0u8; 64]).expect("a small buffer");
    let params = device
        .buffer(&rms_params(1e-6, WIDTH, 1, 0, 1.0))
        .expect("params");
    let four = [
        Bound::whole(&small),
        Bound::whole(&small),
        Bound::whole(&small),
        Bound::whole(&params),
    ];

    // Bindings: a set short of the layout. `wgpu` would also refuse this, at
    // `create_bind_group`, with a message about an entry count.
    assert_eq!(
        device.run(pipeline, &four[..3], &[], [1, 1, 1]),
        Err(Failed::Bindings {
            module: 4,
            bound: 3
        })
    );

    // Params: a uniform block offered where the module declares none. There is
    // nowhere to put it, and binding it into `@group(0)` would shift every
    // storage entry after it.
    assert_eq!(
        device.run(pipeline, &four, &[0u8; 16], [1, 1, 1]),
        Err(Failed::Params {
            needs: 0,
            given: 16
        })
    );

    // Short: the parameter struct bound with fewer bytes than it reads. The
    // quiet one -- WGSL bounds-checks, so the missing `axis_size` would read as
    // ZERO, and a zero axis is a plausible number that norms a row by nothing.
    let stub = device.buffer(&[0u8; 16]).expect("a short block");
    assert_eq!(
        device.run(
            pipeline,
            &[
                Bound::whole(&small),
                Bound::whole(&small),
                Bound::whole(&small),
                Bound::whole(&stub),
            ],
            &[],
            [1, 1, 1]
        ),
        Err(Failed::Short {
            binding: 3,
            needs: 20,
            given: 16
        })
    );

    // Empty: a grid of zero. Legal WebGPU, always a defect -- it runs nothing,
    // reports success, and leaves the output holding whatever it held.
    assert_eq!(
        device.run(pipeline, &four, &[], [1, 0, 1]),
        Err(Failed::Empty { groups: [1, 0, 1] })
    );

    // PastLimit, on two different ceilings.
    let limits = device.limits();
    assert_eq!(
        device.run(
            pipeline,
            &four,
            &[],
            [limits.workgroups_per_dimension + 1, 1, 1]
        ),
        Err(Failed::PastLimit {
            which: Ceiling::Workgroups,
            want: u64::from(limits.workgroups_per_dimension) + 1,
            limit: u64::from(limits.workgroups_per_dimension),
        })
    );
    assert_eq!(
        device.zeroed(limits.buffer_size + 1).err(),
        Some(Failed::PastLimit {
            which: Ceiling::BufferSize,
            want: limits.buffer_size + 1,
            limit: limits.buffer_size,
        })
    );

    // Geometry: a rule this backend has no shader for, refused by name through
    // the same `Failed`.
    assert!(matches!(
        driver_wgpu::device::groups_for(
            &device,
            pipeline,
            kernels::LaunchRule::RecurrentScan,
            Dims::default()
        ),
        Err(Failed::Geometry(_))
    ));

    // Wgpu: a real validation failure, captured rather than panicked. A uniform
    // buffer carries `UNIFORM | COPY_DST` and a storage binding wants
    // `STORAGE`, so this is refused inside `create_bind_group` -- which without
    // the error sink would take the process down.
    let wrong = device.uniform(&[0u8; 32]).expect("a uniform buffer");
    let refused = device.run(
        pipeline,
        &[
            Bound::whole(&wrong),
            Bound::whole(&small),
            Bound::whole(&small),
            Bound::whole(&params),
        ],
        &[],
        [1, 1, 1],
    );
    match refused {
        Err(Failed::Wgpu(why)) => {
            println!("captured, not panicked: {why}");
            assert!(
                why.to_lowercase().contains("usage"),
                "the message should name the usage flag that is missing: {why}"
            );
        }
        other => panic!("a usage violation should be a named refusal, not {other:?}"),
    }
    // The sink is drained by the refusal above, so an unrelated call after it is
    // not handed somebody else's message.
    device.forget_errors();
    device
        .wait()
        .expect("the device is still usable after a refusal");
}

/// A module `naga` will not take is a named refusal, not a panic.
///
/// The one path a shell cannot otherwise reach: every entry of the embedded tree
/// parses, so producing an unparseable module needs the `Modules` seam
/// `serve.rs` keeps for exactly this. It is a separate test rather than another
/// arm of the one above because building it needs the pipeline cache mutably
/// while the other holds a pipeline out of it.
#[test]
fn a_module_that_is_not_wgsl_is_a_named_refusal_and_not_a_panic() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    assert!(matches!(
        cache.get(
            &device,
            "not_a_kernel",
            Capability::Baseline,
            "this is not WGSL"
        ),
        Err(Failed::Module(_))
    ));
    // And one that parses but is not the one-compute-entry-point shape a
    // dispatch needs -- which `naga` accepts and `reflect` refuses by name.
    assert!(matches!(
        cache.get(
            &device,
            "no_entry_point",
            Capability::Baseline,
            "@group(0) @binding(0) var<storage, read> x: array<u32>;"
        ),
        Err(Failed::Module(_))
    ));
    assert_eq!(
        cache.built(),
        0,
        "nothing was cached for a module that failed"
    );
    device.forget_errors();
}

/// The rows an adapter at the WebGPU floor could not run are named, and this one
/// can run all of them.
///
/// [`Failed::Unreachable`] cannot be produced on a desktop adapter — it needs a
/// device that reports the guaranteed minimum of 8 storage buffers per stage —
/// so the refusal itself is checked by computing the same predicate against that
/// floor. What IS checked against the real adapter is that its list is empty,
/// which is the fact a deployment on this machine depends on.
#[test]
fn the_rows_a_floor_adapter_could_not_bind_are_named() {
    let over = kernels_wgpu::over_downlevel_storage_limit();
    assert!(
        over.iter().any(|sig| sig.name == "sdpa_paged_decode"),
        "the row this whole limit argument is about is no longer over the floor"
    );
    println!(
        "at the guaranteed {} storage buffers, {} rows are unreachable: {:?}",
        kernels_wgpu::DOWNLEVEL_STORAGE_BUFFERS,
        over.len(),
        over.iter().map(|s| s.name).collect::<Vec<_>>()
    );

    let Some((device, _held)) = adapter() else {
        return;
    };
    assert!(
        device.unreachable().is_empty(),
        "this adapter allows a compute stage {} storage buffers and cannot bind \
         {:?}, so the rest of this suite is testing a subset of the table",
        device.limits().storage_buffers,
        device.unreachable()
    );
}

/// The pipeline cache stops growing.
///
/// A server that rebuilt its pipelines would be correct and unusably slow, and
/// slower here than on either sibling:
/// `wgpu::Device::create_shader_module` runs `naga` — parse, validate, and a
/// whole backend translation to SPIR-V or MSL — every time it is called, where
/// `vkCreateShaderModule` copies a blob something else compiled at build time.
///
/// Asking for a tier the tree has no variant for must land on the same entry as
/// asking for baseline, or a deployment that picked a tier would double its
/// cache and the number would never settle.
#[test]
fn the_pipeline_cache_stops_growing() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    assert_eq!(cache.built(), 0);

    let names = [
        "rms_single_row_bfloat16",
        "residual_add_bfloat16",
        "kv_append_paged_bfloat16",
    ];
    for tier in [Capability::Baseline, Capability::Subgroup, Capability::Fp16] {
        for name in names {
            let (source, at) = pick(&Embedded, name, tier).expect("the tree holds it at some tier");
            cache.get(&device, name, at, &source).expect("it builds");
        }
    }
    let after = cache.built();
    assert!(
        after >= names.len(),
        "three distinct entrypoints built {after} pipelines"
    );

    // The second pass over the same three at the same tiers builds nothing.
    for tier in [Capability::Baseline, Capability::Subgroup, Capability::Fp16] {
        for name in names {
            let (source, at) = pick(&Embedded, name, tier).expect("the tree holds it");
            cache.get(&device, name, at, &source).expect("it is held");
            assert!(
                cache.peek(name, at).is_some(),
                "`{name}` at {at:?} was built and `peek` does not find it, so a \
                 fire's third pass would rebuild everything"
            );
        }
    }
    assert_eq!(
        cache.built(),
        after,
        "the cache grew on the second pass over the same work"
    );

    cache.clear();
    assert_eq!(cache.built(), 0);
    println!(
        "{after} pipelines for {} entrypoints over three tiers",
        names.len()
    );
}

/// Buffers outlive the device handle, which is why there is no `Drop` in
/// `shell.rs`.
///
/// `driver-vulkan`'s `Shell` has one, and its comment records a real defect: the
/// first shell to OWN a device destroyed it with buffers still on it and the
/// validation layer said `vkDestroyDevice(): VkBuffer 0x97 has not been
/// destroyed`. Without the layer it is a leak that grows one model's worth per
/// shell.
///
/// The concern does not transfer, and this is the evidence rather than the
/// argument: `wgpu::Buffer` holds a strong reference to the device that made it,
/// so dropping the last `Device` handle first destroys nothing. The buffers stay
/// valid — they still report their size — and the allocation returns when they
/// go. There is nothing to order, which is why `shell.rs` says so and writes no
/// `Drop`.
#[test]
fn a_buffer_outlives_the_device_handle_that_made_it() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let held = {
        let buffer = device
            .buffer(&[1u8, 2, 3, 4, 5, 6, 7, 8])
            .expect("a buffer");
        let read = device.read(&buffer).expect("readback");
        assert_eq!(&read[..8], &[1u8, 2, 3, 4, 5, 6, 7, 8]);
        // The device goes out of scope here and the buffer does not.
        drop(device);
        buffer
    };
    assert_eq!(held.size(), 8, "the allocation is still there");
    drop(held);
}
