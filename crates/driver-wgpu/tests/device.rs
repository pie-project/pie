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
use driver_wgpu::resources::{Frame, Pool, Request, Shape};
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

/// THE RUNNER SAYS WHETHER IT HAS A DEVICE, and this is the only test here
/// that is not gated on having one.
///
/// All but one of the tests in this file open through [`adapter`] and
/// `return` when it hands back `None`, printing a `SKIP:` line that
/// `cargo test` swallows unless someone passed `--nocapture`. So the step's
/// summary reads the same on a runner that measured all of them and on a
/// runner that measured none of them, and no one reading the log can tell
/// which run they are looking at.
///
/// No number is written in this paragraph on purpose. It said "sixteen" for a
/// while, and "eighteen" before that, and both were the file's test count at
/// some earlier hour. The scan below is what knows; prose that repeats it is
/// prose that will be wrong by the next test anyone adds -- as this was.
///
/// This one always runs and always says which. `PIE_WGPU_REQUIRE_DEVICE=1`
/// turns the absence into a failure, and CI is where that belongs: the
/// workflow installs `mesa-vulkan-drivers` immediately above this step for
/// the express purpose of GUARANTEEING an adapter, and nothing checked that
/// the install took. A green step is not evidence the guarantee held.
///
/// It asks [`adapter`] rather than [`Device::open`] because the claim it
/// prints is about THOSE TESTS, not about this process. A first draft
/// opened its own device to avoid taking the lock, and an injected failure
/// inside `adapter` had it print PRESENT while every one of them skipped -- the
/// exact lie it was written to catch, told by the catcher. Sharing the code
/// path is what makes the line evidence; the serialisation it costs is the
/// same serialisation every other test here already pays.
///
/// The reason for an absence is printed by `adapter` as its `SKIP:` line and
/// is deliberately not repeated on the `ABSENT` line: one place in this crate
/// knows why there is no device, and that place already says so.
///
/// The count is read off this file rather than kept as a number beside it,
/// for the reason every hand-kept count in this workspace has eventually
/// earned. The needle is split with `concat!` because a literal that looks
/// for itself finds itself: the same scan spelled in one piece returns one
/// too many.
#[test]
fn the_runner_states_whether_it_has_a_device() {
    const NEEDLE: &str = concat!("= adapter", "() else");
    let gated = include_str!("device.rs").matches(NEEDLE).count();
    let required = std::env::var_os("PIE_WGPU_REQUIRE_DEVICE").is_some_and(|v| v != "0");

    match adapter() {
        Some(_) => println!("WGPU DEVICE: PRESENT -- the {gated} gated test(s) here ran"),
        None => {
            println!("WGPU DEVICE: ABSENT -- the {gated} gated test(s) here did NOT run");
            assert!(
                !required,
                "`PIE_WGPU_REQUIRE_DEVICE` is set and `adapter()` opened no \
                 device; its `SKIP:` line above says why. A suite that skips \
                 in silence is what this test exists to prevent; on a Linux \
                 runner, `PIE_WGPU_FALLBACK=1` takes the software adapter, \
                 which is a real implementation of the same WGSL and not a \
                 way of passing."
            );
        }
    }

    // A control on the count: this file gates its tests the way the doc says
    // it does, so the number above is a measurement rather than a zero that
    // reads like one.
    assert!(
        gated >= 15,
        "only {gated} test(s) in this file are gated on `adapter()`, which is \
         fewer than it visibly has -- the scan has lost its needle and the \
         line printed above is not a count of anything."
    );
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

/// `RmsParams` as `norm/rms.wgsl` declares it: five words of `@group(1)`
/// UNIFORM, passed as [`Device::run`]'s `uniform` and not as a fourth buffer.
///
/// It used to be a storage block at `@group(0) @binding(3)`, and this doc used
/// to say that moving it into `@group(1)` would be changing the kernel's ABI
/// from the driver. The kernel moved it itself, and bindings 4 and 5 -- the
/// residual and its scale -- closed up to 3 and 4 behind it, because a
/// declared-and-unfilled slot is a bind group `driver-wgpu` cannot build.
/// `rms_single_row_bfloat16` therefore binds THREE buffers: x, w, out.
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
        "downlevel    the guaranteed floor is {} storage buffers per stage",
        8
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
    assert!(limits.storage_buffers >= 8);
}

/// The launch rule for a kernel whose family has RETIRED its rows.
///
/// `groups_for` takes a `Rule`, and these two kernels no longer have a row to
/// read one from. The rule is the same fact the ROUTINE now states as its
/// `lanes` — `norm/rms.wgsl` reduces one row per workgroup and
/// `norm/residual_add.wgsl` walks a flat run — and stating it here keeps two
/// real device checks alive rather than deleting them with the table.
///
/// Not derived from the routine because a body needs a `Ctx` to state
/// anything and that is the whole plan path; these tests are about what the
/// SHADER computes.
fn rule_of(name: &str) -> kernels::LaunchRule {
    match name {
        "rms_single_row_bfloat16" => kernels::LaunchRule::Rms,
        "residual_add_bfloat16" => kernels::LaunchRule::Elementwise,
        // `kv_append_paged`'s row said `PerHead`, which is `[1, kv_heads,
        // rows]` -- the shape this file's paged-append test asserts.
        "kv_append_paged_bfloat16" => kernels::LaunchRule::PerHead,
        // EXHAUSTIVE, and it has to be: this fell back to the row's `launch`
        // column, which no longer exists. A grid is the ROUTINE's to state
        // now, and a routine needs a `Ctx` to state anything -- which is the
        // whole plan path, and these two tests are about what the SHADER
        // computes rather than about planning. So the three rules they need
        // are written here, with the reason, and a fourth caller has to add
        // its own rather than get a wrong answer from a lookup that cannot
        // fail loudly.
        other => panic!(
            "`{other}` has no stated rule in this file. These tests bypass the \
             plan path deliberately, so a rule they need is stated here or not \
             at all -- add it above with why it is that shape."
        ),
    }
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
    let params = rms_params(eps, WIDTH, 1, 0, 1.0);

    let groups = driver_wgpu::device::groups_for(
        &device,
        pipeline,
        rule_of(name),
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
            &[Bound::whole(&x), Bound::whole(&w), Bound::whole(&out)],
            &params,
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
        rule_of(name),
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
    let params = rms_params(1e-6, WIDTH, 1, 0, 1.0);

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
        // Stage k reads the arena it did NOT write and writes the other, so no
        // dispatch binds one allocation both ways.
        let bounds: Vec<[Bound<'static, Buffer>; 3]> = (0..stages)
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
                    [input, Bound::whole(w), out]
                } else {
                    [input, Bound::whole(r), out]
                }
            })
            .collect();
        let recorded: Vec<Recorded<'_, '_>> = bounds
            .iter()
            .enumerate()
            .map(|(k, bound)| Recorded {
                pipeline: if k % 2 == 0 { norm } else { add },
                buffers: &bound[..],
                // The norm's five words are `@group(1)` uniform now; the add
                // takes none, and a body handed a block it does not declare is
                // a refusal, so only the even stages carry one.
                uniform: if k % 2 == 0 { &params } else { &[] },
                groups: if k % 2 == 0 { norm_grid } else { add_grid },
            })
            .collect();
        if all_at_once {
            device
                .run_all(&recorded)
                .expect("the whole plan ran")
                .shadowed
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

/// A `read` binding aliased with a `read_write` one is diagnosed by name, and
/// run anyway.
///
/// **A fact about WebGPU rather than about `wgpu`**: a dispatch is one usage
/// scope, and a buffer in a usage scope may carry any number of INCLUSIVE
/// usages or exactly one EXCLUSIVE one, never both. `STORAGE_READ_ONLY` is
/// inclusive and `STORAGE_READ_WRITE` is exclusive, so the pair is refused,
/// and disjoint ranges do not help — a buffer has no subresources.
///
/// # This used to use a real kernel, and cannot any more
///
/// Every launch of every real plan bound one arena both ways, and this test
/// took `rms_single_row_bfloat16` off the tree to show it. The shader tree no
/// longer declares a single `var<storage, read>`: two `read_write` bindings
/// of one buffer are the same bit, `is_power_of_two` holds, and the dispatch
/// is legal — which deleted 451 shadow copies from a 452-launch decode and
/// took it from 25.1 ms to 11.2 ms. `kernels-wgpu`'s
/// `no_shader_declares_a_read_only_storage_binding` is that decision, kept.
///
/// So the shader here is written out. The MACHINERY still has to work: a
/// `read` binding is legal WGSL, a future kernel may want one, and without
/// the shadow such a kernel would be refused by `wgpu` rather than run. This
/// is what keeps that true after its last real caller went away — the shape
/// of test that would otherwise have been deleted along with its subject.
///
/// Both halves: [`Device::check`] names it, with the two bindings and the two
/// offsets; [`Device::run_all`] shadows the read side and produces the right
/// numbers.
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
fn an_arena_bound_both_ways_is_diagnosed_by_name_and_run_anyway() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    // `src` is `read` DELIBERATELY. It is the only one left in either tree.
    let source = r"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64)
fn twice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&dst)) { dst[i] = src[i] * 2u; }
}
";
    let pipeline = cache
        .get(&device, "twice", Capability::Baseline, source)
        .expect("it builds");

    let align = device.min_storage_offset();
    let n = 64usize;
    let span = n as u64 * 4;
    let stride = span.next_multiple_of(align);
    // ONE arena, input at 0 and output one region along -- the shape
    // `binding::Arena` produces and the shape WebGPU refuses.
    let mut fill = SENTINEL.to_le_bytes().repeat((stride + span) as usize / 4);
    for (i, word) in fill.chunks_mut(4).take(n).enumerate() {
        word.copy_from_slice(&u32::try_from(i).expect("small").to_le_bytes());
    }
    let arena = device.buffer(&fill).expect("arena");

    let input = Bound::within(&arena, 0, span, align).expect("the input range");
    let output = Bound::within(&arena, stride, span, align).expect("the output range");
    let one = Recorded {
        pipeline,
        buffers: &[input, output],
        uniform: &[],
        groups: [1, 1, 1],
    };

    // The diagnosis, by name and with both offsets.
    assert_eq!(
        device.check(&one),
        Err(Failed::Aliased {
            reader: 0,
            writer: 1,
            read_at: 0,
            write_at: stride,
        }),
        "the arena bound both ways has to be nameable, or a caller meets it as \
         a wgpu string about usage bits"
    );

    // And it runs, through the shadow, with the right numbers.
    let ran = device
        .run_all(&[one])
        .expect("the shadow made it dispatchable");
    assert_eq!(
        ran.shadowed, 1,
        "one read operand shares the arena with the write, so exactly one \
         range should have been copied"
    );
    assert_eq!(
        ran.buffers, 1,
        "a shadow point ends a compute PASS, not an encoder; a fire is one \
         command buffer however much it copies"
    );
    let got = device.read_at(&arena, stride, span).expect("readback");
    let words: Vec<u32> = got
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let want: Vec<u32> = (0..n)
        .map(|i| u32::try_from(i * 2).expect("small"))
        .collect();
    assert_eq!(words, want, "the shadowed dispatch read the wrong bytes");
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
    let params = rms_params(1e-6, WIDTH, 1, 0, 1.0);
    let out = device
        .buffer(&SENTINEL.to_le_bytes().repeat(x_bytes.len() / 4))
        .expect("out");

    // ONE SHORT, on purpose.
    device
        .run(
            pipeline,
            &[Bound::whole(&x), Bound::whole(&w), Bound::whole(&out)],
            &params,
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

    // The uniform block, at the offsets the SHADER declares rather than packed
    // end to end. This asked `kernels_wgpu::uniform_layout` for the row's
    // offsets until the table emptied; the module's own `@group(1)` struct is
    // where the numbers were always coming from, and reading them here is
    // strictly closer to the thing being tested -- a mismatch between the two
    // is what `reflect`'s own layout check exists to catch, and this test
    // should not depend on the answer it is meant to be independent of.
    let declared = driver_wgpu::reflect::entrypoint(name, Capability::Baseline)
        .expect("the module declares this entrypoint");
    let end = declared
        .uniform_offsets
        .iter()
        .copied()
        .max()
        .map_or(0, |at| at as usize + 4);
    let mut uniform = vec![0u8; end.next_multiple_of(16)];
    for (at, value) in
        declared
            .uniform_offsets
            .iter()
            .zip([shape.head_dim, shape.page_size, shape.kv_heads])
    {
        let at = *at as usize;
        uniform[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    assert_eq!(uniform.len(), 16, "three words, rounded to WGSL's 16");

    let groups = driver_wgpu::device::groups_for(
        &device,
        pipeline,
        rule_of(name),
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
    let params = rms_params(1e-6, WIDTH, 1, 0, 1.0);
    let three = [
        Bound::whole(&small),
        Bound::whole(&small),
        Bound::whole(&small),
    ];

    // Bindings: a set short of the layout. `wgpu` would also refuse this, at
    // `create_bind_group`, with a message about an entry count.
    assert_eq!(
        device.run(pipeline, &three[..2], &params, [1, 1, 1]),
        Err(Failed::Bindings {
            module: 3,
            bound: 2
        })
    );

    // Params, in the direction this kernel can now be wrong in: the block it
    // declares, offered SHORT. WGSL bounds-checks a uniform the same way it
    // does a storage block, so the missing `axis_size` would read as ZERO, and
    // a zero axis is a plausible number that norms a row by nothing.
    assert_eq!(
        device.run(pipeline, &three, &[0u8; 16], [1, 1, 1]),
        Err(Failed::Params {
            needs: 20,
            given: 16
        })
    );

    // And the other direction, which needs a different kernel now that this
    // one declares a block: a uniform offered where the module declares none.
    // There is nowhere to put it, and binding it into `@group(0)` would shift
    // every storage entry after it.
    let (add_src, add_tier) =
        pick(&Embedded, "residual_add_bfloat16", Capability::Baseline).expect("the tree holds it");
    // Its OWN cache, because `Pipelines::get` takes `&mut self` and the norm's
    // pipeline is borrowed from `cache` for the whole of this test.
    let mut add_cache = Pipelines::new();
    let add = add_cache
        .get(&device, "residual_add_bfloat16", add_tier, &add_src)
        .expect("it builds");
    assert_eq!(
        device.run(add, &three, &[0u8; 16], [1, 1, 1]),
        Err(Failed::Params {
            needs: 0,
            given: 16
        })
    );

    // Short: a STORAGE block bound with fewer bytes than it reads, which is a
    // different check from `Params` -- that one guards `@group(1)`, this one
    // guards a fixed-size struct sitting among the storage bindings.
    //
    // It takes `sample/argmax.wgsl` to reach, because `norm/rms.wgsl` no
    // longer has one: `ArgmaxParams` is `{ vocab: u32, n_eos: u32, eos_ids:
    // array<u32, 8> }`, forty bytes at binding 2, and it stayed in `@group(0)`
    // exactly so the two scalars ride beside the eos list rather than being
    // split across two groups.
    let (arg_src, arg_tier) =
        pick(&Embedded, "argmax_logits_bfloat16", Capability::Baseline).expect("the tree holds it");
    // Its OWN cache, because `Pipelines::get` takes `&mut self` and the norm's
    // pipeline is borrowed from `cache` for the whole of this test.
    let mut argmax_cache = Pipelines::new();
    let argmax = argmax_cache
        .get(&device, "argmax_logits_bfloat16", arg_tier, &arg_src)
        .expect("it builds");
    let stub = device.buffer(&[0u8; 16]).expect("a short block");
    assert_eq!(
        device.run(
            argmax,
            &[
                Bound::whole(&small),
                Bound::whole(&small),
                Bound::whole(&stub),
                Bound::whole(&small),
            ],
            &[],
            [1, 1, 1]
        ),
        Err(Failed::Short {
            binding: 2,
            needs: 40,
            given: 16
        })
    );

    // Empty: a grid of zero. Legal WebGPU, always a defect -- it runs nothing,
    // reports success, and leaves the output holding whatever it held.
    assert_eq!(
        device.run(pipeline, &three, &params, [1, 0, 1]),
        Err(Failed::Empty { groups: [1, 0, 1] })
    );

    // PastLimit, on two different ceilings.
    let limits = device.limits();
    assert_eq!(
        device.run(
            pipeline,
            &three,
                &params,
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
        ],
        &params,
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

/// The ENTRYPOINTS an adapter at the WebGPU floor could not run are named, and
/// this one can run all of them.
///
/// [`Failed::Unreachable`] cannot be produced on a desktop adapter — it needs a
/// device that reports the guaranteed minimum of 8 storage buffers per stage —
/// so the refusal itself is checked by computing the same predicate against that
/// floor. What IS checked against the real adapter is that its list is empty,
/// which is the fact a deployment on this machine depends on.
#[test]
fn the_rows_a_floor_adapter_could_not_bind_are_named() {
    // Counted off the MODULES and not the table. `over_downlevel_storage_limit`
    // walked `KERNELS` and answered from `storage_count(sig)`; with the table
    // empty it answers "none", which would have turned this whole argument off
    // in silence and left the assertion below passing over an empty adapter
    // list for the wrong reason.
    //
    // `Declared::bindings` is one past the highest `@group(0)` binding the
    // module declares, which is the number a layout must cover -- a variant may
    // leave HOLES, and `wgpu` checks a bind group entry for entry -- so it is
    // the same quantity the row's `storage_count` was standing in for, read
    // from the shader that has to be bound rather than from a description of
    // it.
    let over: Vec<String> = kernels_wgpu::entrypoints()
        .into_iter()
        .filter(|name| {
            driver_wgpu::reflect::entrypoint(name, Capability::Baseline)
                .is_ok_and(|d| d.bindings > 8)
        })
        .collect();
    assert!(
        over.iter()
            .any(|name| name.starts_with("sdpa_paged_decode")),
        "the kernel this whole limit argument is about is no longer over the \
         floor. {} entrypoints are: {over:?}",
        over.len(),
    );
    println!(
        "at the guaranteed {} storage buffers, {} entrypoints are unreachable: {:?}",
        8,
        over.len(),
        over,
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

/// A pool that grows keeps every page it was holding, at the same number.
///
/// # Why this needed a test and did not have one
///
/// Because it is the sentence the whole elastic pool rests on, and it was
/// only a sentence. `Pool::resize`'s own doc says "the pages that survive
/// keep their contents, at the same page numbers", and nothing read it back:
/// the pool's tests are `pages.rs`'s, which are about the BOOK -- who holds
/// which page -- and never touch a buffer. Meanwhile two callers depend on
/// the claim. `Shell::admit` grows mid-conversation, so a growth that lost
/// the cache would answer a decode from zeros, fluently, with no fault
/// anywhere; and `Shell::copy_kv` now grows before copying, so a growth that
/// renumbered would copy the right bytes to the wrong page.
///
/// A pattern rather than a model, because what is under test is the transfer
/// and not the attention: distinct bytes per page, per layer, and per half of
/// the pair, so a resize that copied the key buffers over the value buffers,
/// or layer 1 over layer 0, is a different failure from one that lost the
/// tail.
///
/// The shrink is here too, and it is the same claim from the other side: the
/// pages BELOW the new size keep their contents. `Shell::resize_pool` refuses
/// a shrink that would strand a held page, so the tail being dropped is by
/// then nobody's, and this only asserts that the survivors survived.
#[test]
fn a_pool_that_resizes_keeps_the_pages_it_still_has() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let shape = Shape {
        layers: 2,
        kv_heads: 2,
        head_dim: 4,
        page_size: 3,
        pages: 4,
        bytes: 2,
    };
    let mut pool = Pool::open(&device, shape).expect("a four-page pool");

    // A byte per page, per layer, per half, none of them zero -- zero is what
    // a fresh buffer holds, so a resize that allocated and copied nothing
    // would pass a check written against it.
    let mark = |page: u32, layer: u16, values: bool| -> u8 {
        1 + page as u8 * 8 + layer as u8 * 2 + u8::from(values)
    };
    let page_bytes = (shape.page_size as u64 * shape.row() * shape.bytes as u64) as usize;
    for layer in 0..shape.layers {
        for values in [false, true] {
            let buffer = pool.cache(layer, values).expect("a layer");
            for page in 0..shape.pages {
                let at = shape.slot(page, 0, 0, 0) * shape.bytes as u64;
                device
                    .write(buffer, at, &vec![mark(page, layer, values); page_bytes])
                    .expect("a page of marks");
            }
        }
    }

    let holds = |pool: &Pool, page: u32, layer: u16, values: bool| -> Vec<u8> {
        let buffer = pool.cache(layer, values).expect("a layer");
        let at = shape.slot(page, 0, 0, 0) * shape.bytes as u64;
        device
            .read_at(buffer, at, page_bytes as u64)
            .expect("a page back")
    };

    pool.resize(&device, 7).expect("room to grow into");
    assert_eq!(pool.shape().pages, 7);
    for layer in 0..shape.layers {
        for values in [false, true] {
            for page in 0..4 {
                assert_eq!(
                    holds(&pool, page, layer, values),
                    vec![mark(page, layer, values); page_bytes],
                    "page {page} of layer {layer} (values={values}) did not \
                     survive the growth as itself"
                );
            }
            assert_eq!(
                holds(&pool, 6, layer, values),
                vec![0u8; page_bytes],
                "a page the pool grew into holds somebody's old bytes"
            );
        }
    }

    // ...and the same from the other side. Pages 0..2 keep their contents;
    // the tail is gone, which is the caller's business to have checked.
    pool.resize(&device, 2).expect("a shrink");
    assert_eq!(pool.shape().pages, 2);
    for layer in 0..shape.layers {
        for values in [false, true] {
            for page in 0..2 {
                assert_eq!(
                    holds(&pool, page, layer, values),
                    vec![mark(page, layer, values); page_bytes],
                    "the shrink moved page {page} of layer {layer}"
                );
            }
        }
    }

    // A pool of no pages is not a smaller pool.
    pool.resize(&device, 0).expect_err("a cache of zero pages");
    assert_eq!(pool.shape().pages, 2, "a refused resize changed the pool");
}

/// One device cannot be handed another device's buffer -- and what stops it
/// is a PANIC inside `wgpu-core`, not a refusal this driver can report.
///
/// # Why this test exists
///
/// `driver-vulkan` pointed the Vulkan validation layer at its own suite for
/// the first time and found two tests spending a lock as if it owned the
/// shell's memory: one read a pool's buffers through a second `VkDevice`, and
/// one passed that device to `copy_page`, putting two `VkDevice`s in one
/// `vkCmdCopyBuffer`. That is a `commonparent` violation -- undefined, not
/// untidy -- and the card tolerated it silently, so both tests had been green
/// throughout.
///
/// This suite has the same SHAPE: `adapter()` hands back a device and a lock
/// together, `tests/serving.rs` opens a `Shell` that owns another one, and
/// nothing but care keeps the two apart.
///
/// The layer was pointed here too, on the copy the sibling downloaded, and
/// CONFIRMED LOADED rather than assumed -- `VK_LOADER_DEBUG=layer` printing
/// "Insert instance layer" and "Inserted device layer" for
/// `VK_LAYER_KHRONOS_validation`. The first attempt did not have it loaded at
/// all (`VK_LAYER_PATH` is empty on this box and the package is not
/// installed), and reported clean, which is exactly the trap the sibling
/// documented. With it genuinely loaded: 14 device tests, 53 kernel proofs and
/// 13 serving proofs, no VUID.
///
/// # What actually stops the mistake, measured
///
/// Not this driver, and not the Vulkan layer. `wgpu-core` keeps its resources
/// in a per-device registry and IDs are per-device, so the other device's
/// buffer is simply not there:
///
/// ```text
/// panicked at wgpu-core-30.0.0/src/storage.rs:143:
/// Cannot get non-existent resource BufferId(0,1)
/// ```
///
/// Three things follow, and the middle one is the useful one:
///
/// * it is NOT undefined behaviour, so the sibling's defect cannot silently
///   pass here the way it did there;
/// * it is NOT a named refusal either. It is a panic in a dependency with a
///   message about an ID, raised before `Device::drained` ever sees anything,
///   so this crate's "the layer below reports rather than aborts" discipline
///   does not reach it. A caller cannot catch this and carry on;
/// * therefore the guarantee is real but it belongs to `wgpu`, and it is
///   pinned here so that a version which downgrades the panic to a silent
///   miss is a red test rather than a quiet return to the sibling's world.
///
/// # It wanted two DEVICES, asked for two ADAPTERS, and was red where it ran
///
/// This test used to carry `#[should_panic]` above a `Device::software()` and
/// a comment claiming the software adapter "is the one that ALWAYS exists
/// beside whatever `adapter()` opened, so this cannot be a test that only runs
/// on a two-GPU box". Three things were wrong, and each covered for the next.
///
/// `Device::software` asks wgpu for `PowerPreference::None` with
/// `force_fallback_adapter`. On Linux that finds `llvmpipe`, which is why CI
/// installs `mesa-vulkan-drivers`. **On macOS the Metal backend ships no
/// fallback adapter at all**, so `software()` returned `Unavailable` and this
/// test took its skip branch. Under `#[should_panic]` a skip is a FAILURE,
/// reported as `test did not panic as expected` -- a sentence that names the
/// wrong thing entirely, telling the reader the wgpu guarantee broke when what
/// happened is that the box has one adapter. So the test was RED on every Mac
/// and had never run on one.
///
/// And it never needed a second adapter. `wgpu-core`'s registry is per-DEVICE;
/// the adapter behind it is not what makes an ID foreign. A second
/// `Device::open()` on the very same GPU is a second registry, and binding
/// across the pair raises the identical `BufferId(0,1)` panic -- measured on a
/// single-adapter Mac, which is where the old spelling could not run at all.
/// The pair is now two `open()`s, so the proof runs EVERYWHERE rather than
/// only where a fallback driver happens to be installed.
///
/// The panic is caught here instead of declared in an attribute, which is a
/// strictly stronger pin: `#[should_panic]` accepts a panic from ANYWHERE in
/// the body, including the `expect` below, whereas this catches only the bind
/// and requires the message. The default hook is muted across the catch so the
/// backtrace of an EXPECTED panic does not read like a failure in the log.
#[test]
fn a_buffer_cannot_be_bound_by_the_device_that_did_not_make_it() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    // A SECOND device, not a second adapter: see the heading above.
    let Ok(other) = Device::open() else {
        println!("SKIP: a second device would not open, so there is no pair to mix");
        return;
    };
    let theirs = other
        .buffer(&[1u8, 2, 3, 4])
        .expect("a buffer on the other");

    let layout = device
        .raw()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mixed"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let bound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.raw().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mixed"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: theirs.raw().as_entire_binding(),
            }],
        })
    }));
    std::panic::set_hook(hook);

    let why = match bound {
        // Reached only if `wgpu-core` stopped panicking, in which case this
        // test is the thing that says so.
        Ok(_) => {
            let _ = device.drained();
            panic!(
                "binding another device's buffer was ACCEPTED. Whatever \
                 stopped this is no longer stopping it, and the sibling \
                 driver's defect can now be written here too."
            );
        }
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "a panic with no string payload".to_string()),
    };
    assert!(
        why.contains("non-existent resource"),
        "the bind panicked, but for a different reason than the registry \
         miss this pins: {why}"
    );
}

/// Are the two adapters this crate offers actually two?
///
/// `Device::software`'s doc makes the strongest claim in this crate: "a shader
/// that agrees on a discrete GPU and on `llvmpipe` has been checked by two
/// independent compilers and two independent schedulers, where one that agrees
/// only on the card it was written on has been checked by neither."
///
/// That claim is worth exactly as much as the two adapters being DIFFERENT,
/// and nothing checked it. On a machine with one adapter, `open()` and
/// `software()` return the same implementation and every cross-check between
/// them proves nothing twice.
///
/// # Why this was written
///
/// `.wiki/new-driver/wgpu.md` said, in a section of its own, "There is no GPU
/// on the machine", and discounted every timing in the file accordingly. The
/// default adapter here is an RTX 4090; `PIE_WGPU_FALLBACK=1` is what selects
/// llvmpipe. The knob was documented in three files and nobody had checked
/// which one its ABSENCE picks — because the answer is printed by a passing
/// test, and `cargo test` hides a passing test's stdout.
///
/// So the answer goes in an assertion message instead, where a failure has to
/// show it, and the test says plainly when the cross-check is vacuous rather
/// than letting a one-adapter machine look like a two-adapter one.
///
/// It does not FAIL on one adapter: a CI runner with only `llvmpipe` is a
/// legitimate machine and this suite runs there. `PIE_WGPU_REQUIRE_TWO_ADAPTERS`
/// turns it into a failure for anyone who means to be measuring both.
#[test]
fn the_two_adapters_this_crate_offers_are_two_implementations_or_say_so() {
    let Some((hardware, _held)) = adapter() else {
        return;
    };
    let fast = hardware.name().to_string();
    let api = format!("{:?}", hardware.backend());
    drop(hardware);

    let Ok(soft) = Device::software() else {
        println!("SKIP: this instance offers no software adapter; `{fast}` is all there is");
        return;
    };
    let slow = soft.name().to_string();

    println!("open() -> {fast} ({api});  software() -> {slow}");
    if fast == slow {
        assert!(
            std::env::var_os("PIE_WGPU_REQUIRE_TWO_ADAPTERS").is_none(),
            "`open()` and `software()` both answer `{fast}`, so every claim in \
             this crate about agreeing on TWO implementations is vacuous here, \
             and PIE_WGPU_REQUIRE_TWO_ADAPTERS says that is not acceptable"
        );
        println!(
            "NOTE: one adapter only (`{fast}`). Cross-adapter agreement is \
             UNMEASURED on this machine -- any claim of two implementations \
             needs a machine with two."
        );
        return;
    }

    // Both answer, and they are different. That is the premise every
    // cross-adapter claim in this crate rests on, and it is now written down
    // where a reader of a NUMBER can find which device produced it.
    assert!(
        !fast.is_empty() && !slow.is_empty(),
        "an adapter with no name cannot be attributed a measurement"
    );
}

/// The documented way to ask the second implementation actually asks it.
///
/// `Device::open`'s doc used to say `WGPU_POWER_PREF` was "how a machine with
/// two adapters is asked the same question twice … the way to ask has to be
/// reachable without editing anything". Measured on this machine — an RTX 4090
/// beside `llvmpipe` — unset, `low` and `high` all answer the 4090. A power
/// preference RANKS adapters; it never reaches a software one, which needs
/// `force_fallback_adapter`, the flag [`Device::software`] sets and no
/// preference implies.
///
/// So the stated mechanism could not do the stated thing, and the only paths
/// that could reach the second adapter were the three test files that spell
/// `PIE_WGPU_FALLBACK` themselves. No gate, no curated run and no server could
/// — which is to say the crate's strongest claim was available to unit tests
/// and to nothing that runs a model end to end.
///
/// `Device::open` reads the variable now, so every path gets it. This asserts
/// the behaviour rather than the wiring: with it set, `open()` answers what
/// `software()` answers.
#[test]
fn the_fallback_knob_reaches_the_second_adapter_through_open() {
    let Some((_probe, _held)) = adapter() else {
        return;
    };
    let Ok(soft) = Device::software() else {
        println!("SKIP: no software adapter on this instance");
        return;
    };
    let want = soft.name().to_string();
    drop(soft);

    // SAFETY: single-threaded here -- the suite's lock is held above, and this
    // is the one test that needs the process environment to differ.
    unsafe { std::env::set_var("PIE_WGPU_FALLBACK", "1") };
    let asked = Device::open().map(|d| d.name().to_string());
    unsafe { std::env::remove_var("PIE_WGPU_FALLBACK") };

    let got = asked.expect("a software adapter answered a moment ago");
    assert_eq!(
        got, want,
        "`PIE_WGPU_FALLBACK` did not reach the software adapter through \
         `Device::open`, so the gates, the curated suite and the server cannot \
         be run on the second implementation -- which is the cross-check this \
         crate's `software()` doc calls its strongest"
    );
}

/// Every entrypoint the table ships becomes a real pipeline on this adapter.
///
/// `kernels-wgpu`'s `every_module_parses_and_validates` proves all 481 survive
/// `naga`. That is the LANGUAGE, and it is not the same question as whether an
/// adapter will build one: `create_compute_pipeline` applies limits naga knows
/// nothing about, and the two adapters this machine offers differ by 16x on
/// the one that binds.
///
/// Nothing swept it. The gap showed up as a difference between adapters in the
/// curated suite -- `quest-attention` and `h2o-attention` come back
/// `channel is poisoned: pipeline: ...` on `llvmpipe` and as clean intrinsic
/// refusals on an RTX 4090 -- which is a twenty-minute round trip to learn
/// that SOMETHING did not build. This is the same question asked directly, in
/// seconds, naming the entrypoint.
///
/// Run it on both: `cargo test ... builds_a_pipeline_on_this_adapter` and again with
/// `PIE_WGPU_FALLBACK=1`. A row that builds on one and not the other is the
/// portability defect this backend exists to not have.
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
fn every_entrypoint_in_the_tree_builds_a_pipeline_on_this_adapter() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    println!("building every entrypoint on {}", device.name());

    let mut cache = Pipelines::new();
    let (mut built, mut refused) = (0usize, Vec::new());
    // Walks the SHADER TREE, not the table.
    //
    // It walked `entrypoints()` until Stage 3 deleted its first row, at which
    // point `argmax_logits_bfloat16` stopped being built here at all — the
    // shader was still in the tree and still fired by a routine, and the one
    // sweep that compiles every entrypoint on a real adapter had quietly
    // stopped covering it. Nothing failed, and the loss would have compounded
    // to a sweep that builds nothing.
    //
    // `kernels-wgpu::RETIRED` now keeps those names in `entrypoints()`, so
    // both readings agree again. This one stays on `source::declared()`
    // anyway, because it is the reading that does not depend on a
    // hand-written list being right — and `RETIRED` is hand-written.
    let all: std::collections::BTreeSet<String> = kernels_wgpu::source::declared()
        .into_iter()
        .map(|(_, v)| v.entrypoint)
        .collect();
    for name in all {
        let Some((source, tier)) = pick(&Embedded, &name, Capability::Baseline) else {
            refused.push(format!("{name}: the tree holds no source for it"));
            continue;
        };
        match cache.get(&device, &name, tier, &source) {
            Ok(_) => built += 1,
            Err(why) => refused.push(format!("{name} @{tier:?}: {why}")),
        }
    }

    println!("{built} built, {} refused", refused.len());
    assert!(
        refused.is_empty(),
        "{} of {} entrypoints do not become a pipeline on `{}`. A row that \
         builds on one adapter and not another is the portability this backend \
         exists for, failing:\n  {}",
        refused.len(),
        built + refused.len(),
        device.name(),
        refused.join("\n  ")
    );
    assert!(
        built >= 481,
        "only {built} entrypoints were built; the shader tree declares 481 \
         and a sweep that reads nothing agrees with everything"
    );
}

/// One launch over `symbol`, with one weight operand.
///
/// The smallest plan `serve::fire` will look at: one rectangle, a 256-byte
/// arena, and an operand that exists so the resolver in these tests is
/// something `fire` could reach. Shared by the three refusals below, which
/// differ only in what they point it at.
fn one_launch(symbol: &str) -> model_compiler::lower::Lowered {
    model_compiler::lower::Lowered {
        launches: vec![model_compiler::lower::Launch {
            kernel: 0,
            rows: 0..4,
            layers: 0..1,
            op: 11,
            args: 0..1,
            params: 0..0,
            peel: None,
            cond: model_compiler::lower::Launch::NO_COND,
        }],
        kernels: vec![symbol.to_owned()],
        rectangles: 1,
        arena_bytes: 256,
        value_offset: Vec::new(),
        value_owner: Vec::new(),
        epilogue_gather: usize::MAX,
        epilogue_norm: usize::MAX,
        args: vec![model_compiler::lower::Arg::Weight(
            "model.layers.0.mlp.down_proj.weight".to_owned(),
        )],
        // Zero is "no opinion", which is what a weight has: it is not measured
        // in rows of the batch, so there is nothing for a backend to override
        // the launch's own rectangle with.
        arg_rows: vec![0],
        structural: Vec::new(),
        residue: Vec::new(),
        params: Vec::new(),
        preps: Vec::new(),
        n_requests: 1,
        conds: Vec::new(),
        readout: None,
    }
}

/// A resolver that panics if anything asks it for anything.
struct NothingResolves;

impl driver_wgpu::binding::Resolve for NothingResolves {
    type Buffer = Buffer;
    fn weight(&self, name: &str) -> Option<&Buffer> {
        unreachable!("`fire` asked for the weight `{name}` of a plan it should have refused first")
    }
    fn named(&self, value: model_ir::trace::ValueId) -> Option<&Buffer> {
        unreachable!(
            "`fire` asked for the named value {value:?} of a plan it should have refused first"
        )
    }
}

/// `serve::fire` over [`one_launch`], against whatever module store is given.
fn fire_one<M: driver_wgpu::serve::Modules>(
    device: &Device,
    modules: &M,
    lowered: &model_compiler::lower::Lowered,
) -> Result<driver_wgpu::serve::Fired, driver_wgpu::serve::Unfired> {
    let arena = device.zeroed(256).expect("a 256-byte arena");
    let mut pipelines = Pipelines::default();
    // The bytes `fire` copies out with the work are the readout's, which this
    // fixture's plan does not state; the count is what it is checking.
    driver_wgpu::serve::fire(
        device,
        &mut pipelines,
        modules,
        lowered,
        driver_wgpu::serve::Fire {
            arena: driver_wgpu::binding::Arena {
                buffer: &arena,
                bytes: 256,
            },
            resolver: &NothingResolves,
            geometry: driver_wgpu::dispatch::Geometry {
                q_heads: 16,
                kv_heads: 8,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 0,
                experts_per_token: 0,
                ..Default::default()
            },
            tier: Capability::Baseline,
            one_at_a_time: false,
            // The whole plan: this fires a real text and reads its answer,
            // where a prefix is for finding where one goes wrong.
            prefix: None,
        },
    )
    .map(|(fired, _)| fired)
}

/// A launch naming a symbol no module has is refused BY THAT NAME, and refused
/// before anything is resolved.
///
/// `Unfired::NoModule` sat on the census of refusals no test names — see
/// `every_refusal_this_crate_builds_is_one_a_test_names` — in its "reachable
/// and untested" group, which is the group worth closing. Reachable it is: a
/// plan carries kernel names as strings, and a table that renames a row
/// without renaming its shader produces exactly this.
///
/// **The resolver here panics if it is asked for anything**, and the honest
/// account of that is worth more than the guard. It was written claiming to
/// prove that `fire` looks up every DISTINCT module before it resolves an
/// operand — and then the claim was checked, by pointing this same fixture at
/// a symbol the tree DOES have. It does not fire either: the planner refuses
/// first, with
///
/// ```text
/// launch 0 (`rms_single_row_bfloat16`): operand 0: the row names an operand
/// this statement does not state
/// ```
///
/// so the resolver is unreachable from this fixture whatever the ordering is,
/// and the `unreachable!` could never have failed. **The control is the
/// variant assertion below**, which catches an inversion as a wrong `Unfired`
/// rather than as a panic. The resolver stays because a guard that cannot fire
/// still costs nothing and still documents the expectation; it is described
/// here as what it is rather than as what it looked like.
#[test]
fn a_launch_naming_a_symbol_no_module_has_is_refused_before_anything_resolves() {
    let Some((device, _held)) = adapter() else {
        return;
    };

    const MISSING: &str = "rms_single_row_bfloat16_but_spelled_wrong";
    assert!(
        pick(&Embedded, MISSING, Capability::Baseline).is_none(),
        "this test is only about a missing module if the module is missing"
    );

    // A real operand, so that the resolver above is something `fire` WOULD
    // reach. With no args it could never be called and the `unreachable!`
    // would prove nothing -- which is what this test said it proved until the
    // claim was checked.
    let lowered = one_launch(MISSING);

    let refused =
        fire_one(&device, &Embedded, &lowered).expect_err("a symbol no module has cannot fire");

    match &refused {
        driver_wgpu::serve::Unfired::NoModule { at, symbol } => {
            assert_eq!(*at, 0, "the launch index, and there is one launch");
            assert_eq!(symbol, MISSING, "the refusal names the symbol it wanted");
        }
        other => panic!("expected `NoModule`, got `{other}`"),
    }
    assert!(
        refused.to_string().contains(MISSING),
        "and the MESSAGE names it too, which is the whole point of the variant \
         carrying the string: {refused}"
    );
}

/// A module the WGSL front end cannot read is refused by the module's name.
///
/// `Unfired::Unreadable` is what `reflect::declared` failing becomes, and it
/// is reachable without inventing anything exotic: [`Modules`] is a one-method
/// trait, so a store that hands back text `naga` will not parse is four lines.
///
/// It matters more than it looks. The embedded tree is generated — includes
/// spliced, `//#if` arms resolved, defines substituted — and an expansion that
/// produces text no front end accepts is a build-time mistake that arrives at
/// runtime. This is the refusal that has to name WHICH entrypoint, because the
/// generated source is not what anyone wrote and the symbol is the only handle
/// back to the file that produced it.
#[test]
fn a_module_the_front_end_cannot_read_is_refused_by_the_entrypoint_that_named_it() {
    let Some((device, _held)) = adapter() else {
        return;
    };

    struct NotWgsl;
    impl driver_wgpu::serve::Modules for NotWgsl {
        // THE FIRE PATH'S LOOKUP, which a body reaches through `Fire::at`.
        // Same prose here as at `source` below: what is under test is what a
        // front end does with text that is not WGSL, and both lookups have to
        // hand it the same non-shading language for the refusal to be the
        // one this names.
        fn at(&self, _file: &str, _entrypoint: &str, _tier: Capability) -> Option<String> {
            Some("this is prose, not a shading language".to_owned())
        }

        fn source(&self, _entrypoint: &str, _tier: Capability) -> Option<String> {
            Some("this is prose, not a shading language".to_owned())
        }
    }

    const SYMBOL: &str = "rms_single_row_bfloat16";
    let refused = fire_one(&device, &NotWgsl, &one_launch(SYMBOL))
        .expect_err("text that is not WGSL cannot be reflected, let alone fired");

    match &refused {
        driver_wgpu::serve::Unfired::Unreadable { at, symbol, .. } => {
            assert_eq!(*at, 0, "the launch index, and there is one launch");
            assert_eq!(symbol, SYMBOL, "the refusal names the entrypoint it read");
        }
        other => panic!("expected `Unreadable`, got `{other}`"),
    }
    assert!(
        refused.to_string().contains(SYMBOL),
        "and the message names it, which is the only handle back to the file \
         whose expansion produced the text: {refused}"
    );
}

/// A row whose operands the statement does not state is refused before the
/// device is touched.
///
/// `Unfired::Unplannable` carries an [`Undispatchable`] from the planner, and
/// this is the cheapest way to reach one: a real entrypoint, a real module,
/// and a plan that hands it an operand its kernel row does not declare.
///
/// This refusal was met by accident while checking a different test's guard —
/// pointing a fixture built for `NoModule` at a symbol the tree DOES have
/// produced it immediately — which is the only reason it is cheap. It had sat
/// in the census's "needs a fire built to fail in one specific way" group on
/// the assumption that building one was the expensive part.
#[test]
fn a_row_whose_operands_the_statement_does_not_state_is_unplannable() {
    let Some((device, _held)) = adapter() else {
        return;
    };

    const SYMBOL: &str = "rms_single_row_bfloat16";
    assert!(
        pick(&Embedded, SYMBOL, Capability::Baseline).is_some(),
        "this test needs the module to be FOUND, or it is testing `NoModule`"
    );

    let refused = fire_one(&device, &Embedded, &one_launch(SYMBOL))
        .expect_err("an operand the statement does not state cannot be planned");

    match &refused {
        driver_wgpu::serve::Unfired::Unplannable { at, symbol, .. } => {
            assert_eq!(*at, 0, "the launch index, and there is one launch");
            assert_eq!(
                symbol, SYMBOL,
                "the refusal names the row it could not plan"
            );
        }
        other => panic!("expected `Unplannable`, got `{other}`"),
    }
    let said = refused.to_string();
    assert!(
        said.contains(SYMBOL) && said.contains("operand"),
        "and it says which operand of which symbol, because a planner refusal \
         naming neither is a bug report nobody can act on: {said}"
    );
}

/// Every way a read-out can be refused, and each says which.
///
/// The four `Unread` variants were the rest of the census's "reachable and
/// untested" group, kept there by the same estimate that kept the `Unfired`
/// three — that reaching them meant building a fire. It does not.
/// `serve::logits` takes a device, a buffer and a `Lowered`, and every one of
/// its refusals is decided from the `Readout` the plan states. No dispatch is
/// involved and nothing has to have run.
///
/// `Refused` is the one worth the paragraph. Its check is deliberately made
/// against `lowered.arena_bytes` and NOT against the buffer, so that a caller
/// who allocated a larger arena than the plan asked for is still told when a
/// range runs off the plan's own end. The consequence is that a plan claiming
/// a bigger arena than the buffer it is handed passes the range check and is
/// refused by the DEVICE instead — which is exactly the split this test pins,
/// because the two refusals mean different things to a caller: one is a
/// malformed plan and the other is a device that would not answer.
#[test]
fn the_four_ways_a_read_out_is_refused_each_say_which() {
    let Some((device, _held)) = adapter() else {
        return;
    };

    let readout = |at: usize, rows: u32, vocab: u32, bytes: u32| model_compiler::lower::Readout {
        at,
        rows,
        vocab,
        bytes,
    };
    // The plan is otherwise irrelevant: `logits` reads the exit and the arena
    // size and nothing else.
    let with = |exit: Option<model_compiler::lower::Readout>, arena_bytes: usize| {
        let mut low = one_launch("rms_single_row_bfloat16");
        low.readout = exit;
        low.arena_bytes = arena_bytes;
        low
    };

    let arena = device.zeroed(256).expect("a 256-byte arena");

    // 1. No exit at all. A text that computes something other than a
    //    distribution is a legitimate text; the caller asked the wrong thing.
    match driver_wgpu::serve::logits(&device, &arena, &with(None, 256), &[]) {
        Err(driver_wgpu::serve::Unread::NoExit) => {}
        other => panic!("expected `NoExit`, got {other:?}"),
    }

    // 2. A range that runs off the arena the LOWERING sized.
    match driver_wgpu::serve::logits(&device, &arena, &with(Some(readout(0, 4, 64, 4)), 256), &[]) {
        Err(driver_wgpu::serve::Unread::PastArena { at, extent, arena }) => {
            assert_eq!((at, extent, arena), (0, 1024, 256));
        }
        other => panic!("expected `PastArena`, got {other:?}"),
    }

    // 3. A width this crate does not widen. Two and four are bf16 and f32;
    //    everything else is a plan this reader cannot honour, and guessing
    //    would read two elements as one.
    for odd in [1u32, 3, 8] {
        match driver_wgpu::serve::logits(&device, &arena, &with(Some(readout(0, 1, 4, odd)), 256), &[]) {
            Err(driver_wgpu::serve::Unread::Width(b)) => assert_eq!(b, odd),
            other => panic!("expected `Width({odd})`, got {other:?}"),
        }
    }

    // 4. The device would not give the bytes back: a plan whose arena is
    //    bigger than the buffer it was handed passes the range check above and
    //    is refused here instead.
    let refused =
        driver_wgpu::serve::logits(&device, &arena, &with(Some(readout(0, 4, 256, 4)), 1 << 20), &[]);
    match refused {
        Err(driver_wgpu::serve::Unread::Refused(why)) => assert!(
            !why.to_string().is_empty(),
            "a device refusal that says nothing is the shape this whole error \
             surface exists to avoid"
        ),
        other => panic!("expected `Refused`, got {other:?}"),
    }

    // And the four read differently, which is the point of there being four.
    let said: Vec<String> = [
        driver_wgpu::serve::Unread::NoExit,
        driver_wgpu::serve::Unread::PastArena {
            at: 0,
            extent: 1024,
            arena: 256,
        },
        driver_wgpu::serve::Unread::Width(3),
    ]
    .iter()
    .map(std::string::ToString::to_string)
    .collect();
    assert_eq!(
        said.iter().collect::<std::collections::BTreeSet<_>>().len(),
        said.len(),
        "two of these refusals print the same thing: {said:?}"
    );
}

/// The first ported routine runs on a real adapter and computes what it says.
///
/// `the_first_ported_routine_asks_for_the_grid_its_row_asked_for` proves the
/// body asks for the same GRID the row's `LaunchRule` asked for. This proves
/// the rest of the path: that a `kernels-wgpu` routine, dispatched through
/// `driver_wgpu::encode::Encoder`, reaches the adapter and produces the
/// numbers gemma's PLE join is defined to produce.
///
/// Nothing in `kernels-wgpu` can do this — it names no adapter — and nothing
/// in the table shape needed it, because the driver assembled the dispatch and
/// the row only described it. In the routine shape the body IS the dispatch,
/// so the body is what has to be run.
#[test]
fn the_first_ported_routine_runs_on_this_adapter_and_averages_two_streams() {
    let Some((device, _held)) = adapter() else {
        return;
    };

    /// Two bf16 in one `u32`, low half first — the shader's own packing.
    fn pack(lo: f32, hi: f32) -> u32 {
        (lo.to_bits() >> 16) | ((hi.to_bits() >> 16) << 16)
    }
    /// The low and high bf16 of a word, widened.
    fn unpack(word: u32) -> (f32, f32) {
        (
            f32::from_bits((word & 0xffff) << 16),
            f32::from_bits((word >> 16) << 16),
        )
    }

    const WORDS: usize = 512;
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let proj: Vec<u32> = (0..WORDS)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            pack(i as f32 * 0.5, i as f32 * -0.25)
        })
        .collect();
    let token: Vec<u32> = (0..WORDS)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            pack(i as f32 * 0.125, 1.0)
        })
        .collect();

    let bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let proj_b = device.buffer(&bytes(&proj)).expect("proj");
    let token_b = device.buffer(&bytes(&token)).expect("token");
    let out_b = device
        .zeroed(u64::try_from(WORDS * 4).expect("fits"))
        .expect("out");
    // `PleCombineParams { inv_sqrt2: f32, n: u32 }`.
    let mut params = inv_sqrt2.to_le_bytes().to_vec();
    params.extend_from_slice(&u32::try_from(WORDS).expect("fits").to_le_bytes());
    let params_b = device.buffer(&params).expect("params");

    let mut pipelines = Pipelines::default();
    // THE PARAMS BLOCK IS HANDLE ZERO, because `Handles` mints in ask order
    // and `ctx.params()` is the first thing this body asks for. The operands
    // the caller states are handles 1..3, so `held` is laid out to match.
    let held = [&params_b, &proj_b, &token_b, &out_b];
    // WHAT THE BODY ASKS FOR, ANSWERED. `Env` left the parameter list: the
    // per-layer table is `ctx.params()` and the row count is
    // `ctx.ask::<i32, keys::Rows>()`, so an encoder with nothing behind it
    // refuses both. This is the `answering` channel `lowering::routine`'s
    // `stating` is for, entered from a test instead of from a plan.
    let handles = core::cell::RefCell::new(
        driver_wgpu::lowering::hold::Handles::undivided(&[], &[]),
    );
    let facts = driver_wgpu::lowering::hold::facts(
        "ple_combine_bfloat16",
        // One row of `WORDS` elements: the body's `lanes` is `width * rows`,
        // and the launch this test measures is the flat one it always was.
        1,
        driver_wgpu::dispatch::Geometry::default(),
        1,
        u32::try_from(WORDS).expect("fits"),
        u32::try_from(WORDS).expect("fits"),
    );
    let encoder = driver_wgpu::encode::Encoder::new(
        &device,
        &mut pipelines,
        &Embedded,
        Capability::Baseline,
        &held,
    )
    .answering(&handles, facts);

    let width = i32::try_from(WORDS).expect("fits");
    kernels_wgpu::layout::ple_combine(
        &encoder,
        // The marks are FAT now: an operand carries the rectangle the
        // statement gave it, which is where the body reads its own pitch.
        // `Width` came off 337 parameter lists exactly because the operand
        // beside it already implied it.
        kernels::In {
            ptr: kernels_wgpu::routine::Tensor::new(1),
            rows: 1,
            width,
        },
        kernels::In {
            ptr: kernels_wgpu::routine::Tensor::new(2),
            rows: 1,
            width,
        },
        kernels::Out {
            ptr: kernels_wgpu::routine::Tensor::new(3),
            rows: 1,
            width,
        },
        // The scale came OFF the params block and onto the signature. It is
        // still the same word 0 of the same uniform run once the binder has
        // packed it -- `params` above is built by hand for exactly that
        // reason -- but the routine now states it rather than trusting the
        // caller to have laid it out.
        kernels::routine::Const { v: inv_sqrt2 },
    )
    .expect("the routine dispatches on this adapter");

    let got = device
        .read_at(&out_b, 0, u64::try_from(WORDS * 4).expect("fits"))
        .expect("readback");
    let words: Vec<u32> = got
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut checked = 0usize;
    for (i, word) in words.iter().enumerate() {
        let (glo, ghi) = unpack(*word);
        let (plo, phi) = unpack(proj[i]);
        let (tlo, thi) = unpack(token[i]);
        // The reference rounds to bf16 the way the shader does -- to nearest,
        // ties to even -- and the comparison is exact rather than within a
        // tolerance. A first draft TRUNCATED, and this test caught it: at word
        // 1's high half the shader said 0.53125 and the truncating reference
        // said 0.52734375. `pie_pack_bf16` rounds, and a reference that did
        // not would have made every ported kernel look wrong by a half-ulp.
        let want = |a: f32, b: f32| {
            let bits = ((a + b) * inv_sqrt2).to_bits();
            let round = 0x7fff + ((bits >> 16) & 1);
            f32::from_bits((bits.wrapping_add(round)) & 0xffff_0000)
        };
        assert_eq!(glo, want(plo, tlo), "word {i}, low half");
        assert_eq!(ghi, want(phi, thi), "word {i}, high half");
        checked += 2;
    }
    assert_eq!(checked, WORDS * 2, "every half of every word was compared");
}

/// Two `read_write` bindings into one buffer are LEGAL, and this is the whole
/// reason the shader tree declares no `read` storage binding.
///
/// # The rule, and the exception in it
///
/// `wgpu-core-30.0.0/src/track/mod.rs:333`:
///
/// ```text
/// fn invalid_resource_state<T: ResourceUses>(state: T) -> bool {
///     state.any_exclusive() && !state.bits().is_power_of_two()
/// }
/// ```
///
/// `STORAGE_READ_ONLY` is INCLUSIVE and `STORAGE_READ_WRITE` is EXCLUSIVE, so
/// one buffer bound both ways is two bits with an exclusive one among them —
/// refused. Two `read_write` bindings are the SAME BIT: `is_power_of_two`
/// holds, and the dispatch is fine. That is WebGPU's "usage scope storage
/// exception", and it is the difference between 451 shadow copies per decode
/// and none.
///
/// # Why it is a test and not a paragraph
///
/// The whole shader tree was changed on the strength of it —
/// `kernels-wgpu`'s `no_shader_declares_a_read_only_storage_binding` — and a
/// claim that load-bearing, about behaviour of a device rather than of this
/// code, is exactly the kind that stops being true when a version moves.
/// [`an_arena_bound_both_ways_is_diagnosed_by_name_and_run_anyway`] is its
/// other half: the same two ranges, one binding declared `read`, refused and
/// then shadowed.
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
fn two_read_write_bindings_into_one_buffer_are_legal() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let source = r"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64)
fn twice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&dst)) { dst[i] = src[i] * 2u; }
}
";
    let pipeline = cache
        .get(&device, "twice", Capability::Baseline, source)
        .expect("it builds");
    let align = device.min_storage_offset();
    let n = 64usize;
    let span = n as u64 * 4;
    let stride = span.next_multiple_of(align);
    let mut fill = vec![0u8; (stride + span) as usize];
    for (i, word) in fill.chunks_mut(4).take(n).enumerate() {
        word.copy_from_slice(&u32::try_from(i).expect("small").to_le_bytes());
    }
    let arena = device.buffer(&fill).expect("arena");
    let one = Recorded {
        pipeline,
        buffers: &[
            Bound::within(&arena, 0, span, align).expect("src"),
            Bound::within(&arena, stride, span, align).expect("dst"),
        ],
        uniform: &[],
        groups: [1, 1, 1],
    };
    assert_eq!(
        device.check(&one),
        Ok(()),
        "two `read_write` bindings of one buffer are one usage bit, so \
         nothing here is aliased"
    );
    let ran = device.run_all(&[one]).expect("it dispatches");
    assert_eq!(
        ran.shadowed, 0,
        "nothing was read-only, so there was nothing to copy"
    );
    assert_eq!(ran.buffers, 1);
    let got = device.read_at(&arena, stride, span).expect("readback");
    let words: Vec<u32> = got
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let want: Vec<u32> = (0..n)
        .map(|i| u32::try_from(i * 2).expect("small"))
        .collect();
    assert_eq!(
        words, want,
        "it ran, but it did not read the values that were there"
    );
}

/// Two operands covering the same bytes PARTIALLY is a named refusal.
///
/// # Why this exists now and could not before
///
/// While every read operand was shadowed into scratch, an overlap was
/// harmless: the shader read a copy of what was there before the dispatch,
/// whatever the write did. Now that the tree declares every storage binding
/// `read_write` and nothing is copied, an overlap is a race — WGSL orders the
/// invocations of one dispatch not at all, so which value is read is whatever
/// the scheduler did, and it would be a plausible number rather than an
/// error.
///
/// Disjoint is the ordinary case, an arena launch's input and output.
/// IDENTICAL is the in-place case a kernel authors, where invocation `i`
/// reads and writes element `i`. Partial overlap is what no kernel authors,
/// and no real plan raises it —
/// `a_run_of_decodes_derives_one_lowering_and_says_the_same_thing` and
/// `a_real_fire_is_one_command_buffer_and_shadows_nothing` run 452-launch
/// fires through `check_bindable` and come back clean.
///
/// Neither sibling has this check. Both bind the arena both ways without
/// comment and would run the race.
#[test]
fn two_operands_that_partly_cover_each_other_are_refused_by_name() {
    let Some((device, _held)) = adapter() else {
        return;
    };
    let mut cache = Pipelines::new();
    let source = r"
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(64)
fn twice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&a)) { a[i] = b[i] * 2u; }
}
";
    let pipeline = cache
        .get(&device, "twice", Capability::Baseline, source)
        .expect("it builds");
    let align = device.min_storage_offset();
    let arena = device
        .buffer(&vec![0u8; (align * 4) as usize])
        .expect("arena");

    // `a` covers [0, 2*align) and `b` covers [align, 3*align): they share
    // [align, 2*align), and neither contains the other.
    let one = Recorded {
        pipeline,
        buffers: &[
            Bound::within(&arena, 0, align * 2, align).expect("a"),
            Bound::within(&arena, align, align * 2, align).expect("b"),
        ],
        uniform: &[],
        groups: [1, 1, 1],
    };
    assert_eq!(
        device.check(&one),
        Err(Failed::Overlapping {
            writer: 0,
            other: 1,
            overlap: align..align * 2,
        }),
        "a partial overlap has to be named, or it is a race that returns a \
         number"
    );
    // And `run_all` pays it too -- this is not a diagnosis a caller has to
    // ask for, because there is no workaround that makes it safe.
    assert!(
        device.run_all(&[one]).is_err(),
        "`run_all` ran a dispatch whose two operands overlap"
    );

    // The two ordinary shapes are NOT refused. Without these the check could
    // be refusing everything and this test would not know.
    let disjoint = Recorded {
        pipeline,
        buffers: &[
            Bound::within(&arena, 0, align, align).expect("a"),
            Bound::within(&arena, align, align, align).expect("b"),
        ],
        uniform: &[],
        groups: [1, 1, 1],
    };
    assert_eq!(
        device.check(&disjoint),
        Ok(()),
        "disjoint ranges are normal"
    );
    let in_place = Recorded {
        pipeline,
        buffers: &[
            Bound::within(&arena, 0, align, align).expect("a"),
            Bound::within(&arena, 0, align, align).expect("b"),
        ],
        uniform: &[],
        groups: [1, 1, 1],
    };
    assert_eq!(
        device.check(&in_place),
        Ok(()),
        "an operand bound to the same range twice is the in-place case, which \
         a kernel authors on purpose"
    );
}
