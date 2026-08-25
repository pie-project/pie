//! THE THREE CLAIMS, ON THE CARD IN THIS BOX.
//!
//! `driver-vulkan` spent the whole of R3 and R5 outside the workspace. Its
//! manifest named two crates R3 had deleted, so cargo could not resolve the
//! package and never got as far as compiling it -- and `f80332be8`, which made
//! it a member again, found `cargo check -p driver-vulkan --lib` green on the
//! first try. That is a real result and it is a small one: it says the
//! PORTABLE half parses. `default = []`, so `--lib` compiled `geometry`,
//! `spirv`, `facts`, `phase`, `programs` and `rope` and not one line that
//! touches a GPU.
//!
//! This file is the difference between that and a plane. Three claims, in the
//! order they have to hold, because each is worthless if the one before it is
//! false:
//!
//! 1. a Vulkan device OPENS, through this crate's own [`Device::open`], and
//!    says what it offers;
//! 2. a SPIR-V module out of `kernels-vulkan`'s compiled tree becomes a
//!    `VkPipeline` on that device -- every module in the tree, not a chosen
//!    one, because "the one I picked builds" is the claim that was true right
//!    up until the model that needed the other 664;
//! 3. one kernel DISPATCHES, and the numbers that come back are the numbers
//!    another vendor's silicon computes from the same bytes.
//!
//! # What the numbers are held against
//!
//! The reference is a host model of the shader's own arithmetic, computed in
//! `f64` so that neither device's `f32` reassociation is baked into the thing
//! both are being judged by. That is the weaker of the two references this
//! slice could have used, and it is in the file because it is the one that
//! travels: it needs no CUDA toolkit, no second card, and no golden blob.
//!
//! The stronger one was RUN, and here is the record of it, because a test
//! whose reference is a restatement of the shader cannot notice that the
//! shader and the restatement are wrong together.
//! `kernels-cuda/kernels/norm/rmsnorm.cuh`'s `pie::norm::rmsnorm<bf16, 256>`
//! is the CUDA twin of `kernels-vulkan/kernels/norm/rms.slang`'s
//! `rms_single_row_bfloat16` -- same `y = w · x · rsqrt(mean(x²) + eps)`, same
//! `f32` accumulate, same round-to-nearest-even narrowing (compare that
//! header's `f32_to_bf16` with `common/bf16.slang`'s: they are the same four
//! lines), and `LaunchRule::Rms`'s one-block-per-row-of-256 is the launch both
//! backends give it. Compiled with `nvcc -O2 -arch=sm_89` and run on the SAME
//! L40S as the Vulkan dispatch below, over the same 13 x 461 bf16 inputs this
//! file generates:
//!
//! ```text
//! Vulkan vs the CUDA twin:       5993 / 5993 bf16 words BIT-IDENTICAL
//! Vulkan vs the f64 host model:  worst relative error 2.645e-3, budget 3.906e-3
//! ```
//!
//! Bit-identical was not assumed and is not asserted. The two reductions
//! associate differently -- a subgroup `WaveActiveSum` over four-element
//! strides against a shared-memory tree over 256-element strides -- so an
//! element sitting on a rounding boundary is free to disagree, and at another
//! shape one will. That it came out exact here is a measurement and not a
//! property, and writing it into an assertion would turn a lucky shape into a
//! rule.
//!
//! What it does establish is the one thing the `f64` model cannot establish
//! about itself: that it is not this shader's own mistake written twice. Two
//! independently written kernels, two compilers, two languages and two
//! execution models agree to the bit, and the model agrees with both to within
//! the narrowing.
//!
//! To re-measure it, `PIE_VULKAN_DUMP=<path>` writes this fire's output buffer
//! verbatim, which is what the twin was diffed against. A number quoted in
//! prose that cannot be re-run is a number this tree has no reason to believe.
//!
//! # Why the tolerance is what it is
//!
//! Half a bf16 ulp of the ROW's largest output. bf16 keeps eight significand
//! bits, so a correctly rounded result is within `2^-9` of the exact value and
//! within `2^-9 · max/|v|` of it when scaled by the row's largest -- which is
//! `1/256` at the worst, and that is [`TOLERANCE`]. The `f32` reassociation
//! between the two devices contributes about `1e-7` and is not what sets it.
//!
//! Scaling by the row's own largest and not by `max(|want|, 1.0)` is
//! deliberate: `.wiki/new-driver/vulkan.md` §12 records a floor of one turning
//! a 2% claim into a flat absolute 0.02, which is 7% of an attention value.
//!
//! # Why the sizes are odd
//!
//! `norm/rms.slang` walks a row in chunks of `PIE_GROUP_X · N_READS` = 1024
//! elements, four consecutive per lane. 461 is not a multiple of 1024, so the
//! chunk loop has a partial trip; it is not a multiple of 4, so the unrolled
//! inner read has a partial one too and the `start + i < axis_size` guard is
//! actually exercised. 13 rows is not a multiple of anything. Every one of
//! those was a round number in the suite that shipped the defects §12 records,
//! and a tail that never exists is a tail nothing tests.
//!
//! # Why these are not `#[ignore]`
//!
//! The crate's own idiom, stated at the head of `tests/device.rs` and of
//! `driver-wgpu/tests/device.rs`: an ignored test is skipped on the machine
//! that HAS the hardware too, which is the machine whose failures matter.
//! These are gated on a FEATURE a build box does not turn on, and on a machine
//! that has a card they run in a plain
//! `cargo test -p driver-vulkan --features device --test device_fire`. The
//! third case -- feature on, no device -- prints why and returns, and
//! [`the_runner_states_whether_it_fired`] is not gated at all so that a run
//! which measured nothing says so.
//!
//! # Why `device` and not `native`
//!
//! Because `native` does not build. R5 deleted `model_compiler::lower`,
//! `model_ir::trace::ForwardPlan` and `kernels_vulkan::routine`, and eleven of
//! this crate's modules name them: `--features native` is 40 unresolved-import
//! errors, none of which is in `device`, `spirv`, `geometry`, `facts` or
//! `phase`. `Cargo.toml` says the rest. The half that opens the card was never
//! broken; it was standing behind one feature with the half that was.

#![cfg(feature = "device")]
#![allow(clippy::print_stdout)]

use driver_vulkan::device::{Bound, Device, Pipelines, groups_for};
use driver_vulkan::{Dims, Rule};
use kernels_vulkan::Capability;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Rows the fire covers. Not a multiple of anything.
const ROWS: usize = 13;

/// Elements in one row. Not a multiple of `norm/rms.slang`'s 1024-element
/// chunk, and not a multiple of its four-per-lane inner read either.
const WIDTH: usize = 461;

/// The `eps` inside the reciprocal square root, as both kernels take it.
const EPS: f32 = 1e-5;

/// Half a bf16 ulp, relative to the row's largest output. See the header.
const TOLERANCE: f64 = 1.0 / 256.0;

/// What an untouched output word holds.
///
/// `0x4780` is bf16 for `65536.0`, so a word of two of them is a value this
/// fire cannot produce: every input is in `[-2.1875, 2.1875]`, every gain is
/// under 1, and normalising does not grow a row. **Zero cannot be used** --
/// [`Device::empty`] hands back a buffer of zeros, so a slot nothing wrote and
/// a slot written with a zero would be the same bytes, and a dispatch that
/// never ran would satisfy a check written against zero.
const SENTINEL: u8 = 0x47;

/// The one device this file opens, and the lock that serialises it.
///
/// `None` when there is no device, so a machine without a card skips rather
/// than fails. One device for the file, not one per test, for the reason
/// `tests/device.rs` gives at length: a Vulkan queue and command pool are
/// externally synchronised objects, and this crate's `Device` owns one of
/// each.
static GPU: OnceLock<Option<Mutex<Device>>> = OnceLock::new();

/// Why `Device::open` refused, kept for the test whose subject is that.
static NO_DEVICE: OnceLock<String> = OnceLock::new();

fn gpu() -> Option<MutexGuard<'static, Device>> {
    let held = GPU.get_or_init(|| match Device::open() {
        Ok(d) => Some(Mutex::new(d)),
        Err(e) => {
            eprintln!("skipped: {e}");
            let _ = NO_DEVICE.set(e.to_string());
            None
        }
    });
    // A poisoned lock means an earlier test panicked holding the device.
    // Nothing here leaves it unusable and the panic has already been
    // reported, so the rest run rather than cascading.
    held.as_ref()
        .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// Borrow the shared device, or skip saying why.
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

/// The bf16 narrowing `common/bf16.slang` and `prelude/device.cuh` both do.
///
/// Round to nearest even. A truncating `(bits >> 16) as u16` agrees on most
/// inputs and disagrees on exactly the ones a tolerance check would not
/// notice, and it biases a 461-long accumulate toward zero.
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

/// The activations this fire norms, row-major, `ROWS` x `WIDTH`.
///
/// `37` steps through the residues of `71` and `997` moves each row somewhere
/// unrelated, so neighbouring elements are far apart in value. A ramp would
/// make an off-by-one index a difference of one sixteenth, which no tolerance
/// worth having could see; here it is a difference of about two.
///
/// Every value is a multiple of `1/16` under 2.2, so **every input is
/// bf16-exact** and the comparison is about the kernel's arithmetic rather
/// than about who rounded the inputs.
fn activations() -> Vec<f32> {
    (0..ROWS)
        .flat_map(|r| (0..WIDTH).map(move |i| (((r * 997 + i * 37) % 71) as f32 - 35.0) / 16.0))
        .collect()
}

/// The per-element gain. Also bf16-exact, and not constant across the row --
/// a gain that never varied would hide a shader that dropped `w_stride`.
fn gains() -> Vec<f32> {
    (0..WIDTH).map(|i| 0.5 + (i % 13) as f32 / 32.0).collect()
}

/// `norm/rms.slang`'s twenty-byte push block, in the order the shader's
/// `struct Push` declares it: `eps`, `axis_size`, `w_stride`, `plus_one`,
/// `gain`.
fn push_block() -> Vec<u8> {
    let mut p = Vec::with_capacity(20);
    p.extend_from_slice(&EPS.to_le_bytes());
    p.extend_from_slice(&(WIDTH as u32).to_le_bytes());
    p.extend_from_slice(&1u32.to_le_bytes());
    p.extend_from_slice(&0u32.to_le_bytes());
    p.extend_from_slice(&1.0f32.to_le_bytes());
    p
}

/// One row of the reference, in `f64`.
///
/// The shader's arithmetic and not a tidier equivalent: the sum of squares,
/// the mean, `eps` inside the root, the gain applied to the normalised value.
/// Returned alongside the row's largest magnitude, which is what the tolerance
/// scales by.
fn reference_row(x: &[f32], w: &[f32]) -> (Vec<f64>, f64) {
    let mean = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / WIDTH as f64;
    let inv = (mean + f64::from(EPS)).sqrt().recip();
    let want: Vec<f64> = x
        .iter()
        .zip(w)
        .map(|(v, g)| f64::from(*g) * (f64::from(*v) * inv))
        .collect();
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    (want, scale)
}

/// THE RUNNER SAYS WHETHER IT FIRED, and this is the only test here that is
/// not gated on having a device.
///
/// Every other test in this file returns early when there is none, printing a
/// `skipped:` line that `cargo test` swallows unless someone passed
/// `--nocapture`. So a run that opened a card and a run that touched no GPU
/// whatsoever both print `test result: ok`, and nobody reading the log can
/// tell which they are looking at. `kernels-vulkan`'s suite was caught
/// reporting 48 passed in 0.06 seconds on a box whose ICD is a stub; nothing
/// but the clock said so.
///
/// `PIE_VULKAN_REQUIRE_DEVICE=1` turns the absence into a failure. Same
/// spelling `tests/device.rs` and the `kernels-vulkan` suite use, so one
/// variable covers all three, and what a job that installs a driver ON PURPOSE
/// should set.
#[test]
fn the_runner_states_whether_it_fired() {
    let required = std::env::var_os("PIE_VULKAN_REQUIRE_DEVICE").is_some_and(|v| v != "0");
    match gpu() {
        Some(device) => {
            let name = device.name().to_string();
            let software = device.software();
            drop(device);
            println!(
                "VULKAN DEVICE: PRESENT ({name}){}. The claims in this file were \
                 measured against it.",
                if software { " (software)" } else { "" }
            );
        }
        None => {
            let why = NO_DEVICE.get().map_or("no reason recorded", String::as_str);
            println!("VULKAN DEVICE: ABSENT ({why}).");
            println!("Every claim in this file skipped, so a green run here measured NOTHING.");
            assert!(
                !required,
                "PIE_VULKAN_REQUIRE_DEVICE is set and no device opened: {why}. A \
                 suite that silently skips is what this test exists to prevent"
            );
        }
    }
}

/// CLAIM ONE: a device opens through this crate's own code, and says what it
/// offers.
///
/// Not a formality, and not only because everything below needs it. The facts
/// this prints are the ones `facts::of` hands the engine, which keeps them for
/// the whole run and plans against them -- a wrong one is not caught later, it
/// is believed. So each is held against the thing HERE that would break if the
/// engine believed a wrong one: the alignment against the sub-range bind that
/// enforces it, the tiers against the modules that have to load, the grid
/// limit against the dispatch that has to fit under it.
#[test]
fn a_device_opens_and_states_what_it_offers() {
    let device = gpu!();
    let facts = driver_vulkan::facts::of(&device);

    println!("device:      {}", device.name());
    println!("software:    {}", device.software());
    println!("validated:   {} (a layer is watching)", device.validated());
    println!("tiers:       {:?}", device.tiers());
    println!("max push:    {} bytes", device.max_push());
    println!("max groups:  {:?}", device.max_groups());
    println!("min offset:  {} bytes", device.min_storage_offset());
    println!("unified:     {}", device.unified());
    println!("device-only: {}", device.device_only_memory());
    println!("budget:      {} MiB", device.budget() >> 20);
    println!("facts:       {facts:?}");

    assert_eq!(facts.backend, "vulkan", "the engine matches on this string");
    assert_eq!(
        facts.abi_version,
        driver_api::PIE_DRIVER_ABI_VERSION,
        "a driver that answers a different ABI than it was built against is one \
         the seam must refuse rather than talk to"
    );

    // The alignment is the one field the engine LAYS ITS ARENA OUT with, so
    // it is held against `Bound::at`, which is what refuses a sub-range that
    // does not satisfy it -- the two must be one number, not two that agree.
    let alignment = device.min_storage_offset();
    assert!(
        alignment.is_power_of_two(),
        "`Bound::at` masks rather than divides, which needs a power of two; \
         this device says {alignment}"
    );
    assert!(
        alignment <= u64::from(driver_vulkan::facts::GUARANTEED_STORAGE_ALIGNMENT),
        "the specification's guaranteed maximum for this limit is 256, and a \
         device reporting {alignment} would make `facts::floor` an \
         under-promise rather than a safe one"
    );
    assert_eq!(
        u64::from(facts.storage_alignment),
        alignment,
        "the fact and the limit are the same number said twice"
    );

    // A grid limit under the specification's floor would mean the driver read
    // the wrong field, which is the kind of thing that turns into a silently
    // truncated dispatch rather than an error.
    for (axis, &limit) in device.max_groups().iter().enumerate() {
        assert!(
            limit >= 65535,
            "axis {axis} dispatches at most {limit} workgroups, under the 65535 \
             the specification requires of every implementation"
        );
    }
    assert!(
        device.max_push() >= 128,
        "the specification floor is 128 bytes and `norm/rms.slang`'s block is \
         20; a device under it could not take this crate's push blocks at all"
    );

    assert!(
        device.tiers().contains(&Capability::Baseline),
        "the baseline tier is not optional -- it is the one every entrypoint \
         has, and a device that could load none of it could run no model"
    );
    assert!(
        device.budget() > 0,
        "a device with no host-visible heap is one this crate cannot allocate \
         out of at all, since every buffer it makes is host-visible"
    );
}

/// CLAIM TWO: every module in the compiled tree becomes a pipeline on this
/// device.
///
/// The whole tree and not a sample, which is the difference between this and
/// `the_tier_this_device_selects_is_one_it_can_actually_load` in
/// `tests/device.rs`. A tier is a separate BODY compiled with different
/// extensions, and a module that fails to load does not fail quietly at the
/// call that needs it -- `vkCreateComputePipelines` refuses, and the model
/// that needed exactly that kernel is the one that finds out.
///
/// It goes through [`Pipelines::get`], this crate's own cache, rather than
/// through `ash` directly: what is being measured is that THE DRIVER can build
/// them, including its own reading of each module's binding count and holes.
/// `descriptors` is 0 so the module's own `declared.bindings` decides the
/// layout width, and the push range is the device's maximum so that a layout
/// is never the reason a module is refused -- a range wider than the block a
/// shader declares is legal, and this test is about the MODULE.
#[test]
fn every_entrypoint_in_the_tree_becomes_a_pipeline_on_this_device() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let push = device.max_push();

    let names = kernels_vulkan::entrypoints();
    assert!(
        names.len() >= 400,
        "only {} entrypoints in this build, which is not the tree",
        names.len()
    );

    let mut built = 0usize;
    let mut refused: Vec<String> = Vec::new();
    let mut unresolved: Vec<String> = Vec::new();
    let mut by_tier: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    let mut widest = 0u32;

    let started = std::time::Instant::now();
    for name in &names {
        let Some((code, tier)) = device.module_for(name) else {
            unresolved.push(name.clone());
            continue;
        };
        match cache.get(&device, name, code, push, 0, tier) {
            Ok(pipeline) => {
                widest = widest.max(pipeline.bindings());
                *by_tier.entry(tier.tag()).or_default() += 1;
                built += 1;
            }
            Err(e) => refused.push(format!("{name} ({}): {e}", tier.tag())),
        }
    }
    let took = started.elapsed();

    assert!(
        unresolved.is_empty(),
        "{} entrypoints resolve to no module at any tier this device offers: {}",
        unresolved.len(),
        unresolved.join(", ")
    );
    assert!(
        refused.is_empty(),
        "{} of {} modules did not become a pipeline on {}:\n  {}",
        refused.len(),
        names.len(),
        device.name(),
        refused.join("\n  ")
    );
    assert_eq!(built, names.len(), "every entrypoint or none");
    assert_eq!(
        cache.built(),
        built,
        "the cache holds one pipeline per entrypoint asked for"
    );

    println!(
        "{built} of {} modules built a VkPipeline on {} in {took:?}; by tier \
         {by_tier:?}; widest descriptor set {widest}",
        names.len(),
        device.name()
    );

    cache.clear(&device);
}

/// CLAIM THREE: a kernel fires, and the numbers are right.
///
/// `rms_single_row_bfloat16` over 13 rows of 461, dispatched on the grid this
/// crate computes from the module this crate loaded, checked against the `f64`
/// host model -- and, once, against `kernels-cuda`'s twin on this same card.
/// The header has that measurement and why it is not asserted here.
///
/// The output buffer is filled with [`SENTINEL`] first. A dispatch that
/// covered fewer rows than it claimed would leave the tail holding a value
/// nothing below can produce, and
/// [`a_grid_one_workgroup_short_is_a_failure_this_check_can_see`] is that
/// possibility exercised rather than argued about.
#[test]
fn an_rms_row_this_device_computes_is_the_row_the_reference_computes() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";

    let x = activations();
    let w = gains();
    let xb = bf16_bytes(&x);
    let wb = bf16_bytes(&w);

    // Every input must survive the narrowing untouched, or the comparison
    // below is partly a measurement of who rounded the inputs.
    assert_eq!(
        bf16_read(&xb),
        x,
        "the activations are meant to be bf16-exact"
    );
    assert_eq!(bf16_read(&wb), w, "the gains are meant to be bf16-exact");

    let (code, tier) = device
        .module_for(entrypoint)
        .expect("`rms_single_row_bfloat16` has a module in this build");
    let push = push_block();
    let pipeline = cache
        .get(&device, entrypoint, code, push.len() as u32, 0, tier)
        .expect("the pipeline builds");

    // What the SHADER says about itself, checked before anything is bound to
    // it. `norm/rms.slang` declares three buffers and a five-word push block;
    // a module declaring otherwise is one this test is not describing, and
    // finding that out here is better than finding it out as a wrong number.
    let declared = pipeline.declared();
    assert_eq!(
        declared.bindings, 3,
        "x, w and out_ -- the residual arms' bindings 3 and 4 belong to the \
         other instantiations"
    );
    assert_eq!(
        declared.push_offsets,
        vec![0, 4, 8, 12, 16],
        "eps, axis_size, w_stride, plus_one, gain -- the order `struct Push` \
         declares and the order `push_block` writes"
    );
    assert_eq!(
        declared.local,
        [256, 1, 1],
        "`PIE_GROUP_X` is 256, and the grid below is a division by it"
    );

    let out = device
        .empty((ROWS * WIDTH * 2) as u64)
        .expect("an output buffer");
    device
        .write(&out, &vec![SENTINEL; ROWS * WIDTH * 2])
        .expect("the sentinel goes down before the dispatch");
    let bufs = [
        device.buffer(&xb).expect("x"),
        device.buffer(&wb).expect("w"),
        out,
    ];

    let dims = Dims {
        rows: ROWS as u32,
        width: WIDTH as u32,
        axis: WIDTH as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    assert_eq!(
        groups,
        [ROWS as u32, 1, 1],
        "`Rule::Rms` is one workgroup per row on x, because `norm/rms.slang` \
         takes its row from `gl_WorkGroupID.x` and never mentions y"
    );

    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(pipeline, &bound, &push, groups)
        .expect("the dispatch");

    let raw = device.read(&bufs[2]).expect("read back");
    // The output buffer verbatim, for the CUDA cross-check the header records.
    // It exists so that the strongest claim in this file is one anybody can
    // re-run rather than one they have to take on trust.
    if let Some(to) = std::env::var_os("PIE_VULKAN_DUMP") {
        std::fs::write(to, &raw).expect("the dump path is writable");
    }
    let got = bf16_read(&raw);
    assert_eq!(got.len(), ROWS * WIDTH);

    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for r in 0..ROWS {
        let row = &x[r * WIDTH..(r + 1) * WIDTH];
        let (want, scale) = reference_row(row, &w);
        assert!(scale > 0.0, "row {r} of the reference is all zeros");
        for i in 0..WIDTH {
            let e = (f64::from(got[r * WIDTH + i]) - want[i]).abs() / scale;
            if e > worst {
                worst = e;
                worst_at = (r, i);
            }
        }
    }

    let (r, i) = worst_at;
    assert!(
        worst <= TOLERANCE,
        "row {r} element {i}: the device says {}, the reference says {}, a \
         relative error of {worst:.3e} against a budget of {TOLERANCE:.3e}",
        got[r * WIDTH + i],
        reference_row(&x[r * WIDTH..(r + 1) * WIDTH], &w).0[i]
    );

    println!(
        "{entrypoint} fired {ROWS}x{WIDTH} on {} at tier {}: worst relative \
         error {worst:.3e} against the f64 host model, budget {TOLERANCE:.3e} \
         (worst at row {r} element {i})",
        device.name(),
        tier.tag()
    );

    cache.clear(&device);
    for b in bufs {
        device.free(b);
    }
}

/// THE MUTATION, so that the agreement above is known to be falsifiable.
///
/// A check that has never been seen to fail is a check nobody has any reason
/// to believe. This runs the identical comparison over a grid one workgroup
/// SHORT -- the single defect this crate's whole geometry module exists to
/// prevent, and the one that is completely silent on Vulkan: an undershot grid
/// writes nothing to the rows it does not cover, every call returns success,
/// and no validation layer objects.
///
/// So the last row keeps its [`SENTINEL`], the comparison that passed above
/// fails here, and both facts are asserted. `.wiki/new-driver/vulkan.md` §11
/// is the record of 54 green tests that were green because nothing was
/// checking; this is the shape of the answer to that.
#[test]
fn a_grid_one_workgroup_short_is_a_failure_this_check_can_see() {
    let device = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";

    let x = activations();
    let w = gains();
    let push = push_block();
    let (code, tier) = device.module_for(entrypoint).expect("a module");
    let pipeline = cache
        .get(&device, entrypoint, code, push.len() as u32, 0, tier)
        .expect("the pipeline builds");

    let out = device
        .empty((ROWS * WIDTH * 2) as u64)
        .expect("an output buffer");
    device
        .write(&out, &vec![SENTINEL; ROWS * WIDTH * 2])
        .expect("the sentinel");
    let bufs = [
        device.buffer(&bf16_bytes(&x)).expect("x"),
        device.buffer(&bf16_bytes(&w)).expect("w"),
        out,
    ];

    // The mutation, and the whole of it: the grid this crate computes, minus
    // one workgroup. Nothing else differs from the test above.
    let short = [ROWS as u32 - 1, 1, 1];
    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(pipeline, &bound, &push, short)
        .expect("an undershot dispatch is still a successful one, which is the point");

    let got = bf16_read(&device.read(&bufs[2]).expect("read back"));

    // The rows that WERE covered still agree, so what follows is attributable
    // to the missing workgroup and not to a broken fire.
    for r in 0..ROWS - 1 {
        let (want, scale) = reference_row(&x[r * WIDTH..(r + 1) * WIDTH], &w);
        for i in 0..WIDTH {
            let e = (f64::from(got[r * WIDTH + i]) - want[i]).abs() / scale;
            assert!(
                e <= TOLERANCE,
                "row {r} was inside the short grid and should be untouched by \
                 the mutation, but element {i} is off by {e:.3e}"
            );
        }
    }

    // And the row that was not covered is untouched, which the comparison
    // above would have caught.
    let last = &got[(ROWS - 1) * WIDTH..];
    let sentinel = from_bf16(u16::from_le_bytes([SENTINEL, SENTINEL]));
    assert!(
        last.iter().all(|v| *v == sentinel),
        "the row past the short grid should still hold the sentinel; a device \
         that wrote it anyway would make this whole test meaningless"
    );

    let (want, scale) = reference_row(&x[(ROWS - 1) * WIDTH..], &w);
    let worst = (0..WIDTH)
        .map(|i| (f64::from(last[i]) - want[i]).abs() / scale)
        .fold(0.0f64, f64::max);
    assert!(
        worst > TOLERANCE,
        "the check in `an_rms_row_this_device_computes_is_the_row_the_reference_\
         computes` would have PASSED on a grid one workgroup short -- worst \
         relative error {worst:.3e} against a budget of {TOLERANCE:.3e} -- so \
         it is not evidence of anything"
    );

    println!(
        "one workgroup short: row {} kept its sentinel and the same comparison \
         fails at {worst:.3e} against a budget of {TOLERANCE:.3e}",
        ROWS - 1
    );

    cache.clear(&device);
    for b in bufs {
        device.free(b);
    }
}
