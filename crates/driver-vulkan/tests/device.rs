//! The device half, on a real GPU.
//!
//! `tests/rules.rs` proves the launch arithmetic against the modules as FILES.
//! This proves the rest of it against hardware: that a pipeline built from what
//! a module declares is one the driver accepts, that a grid this crate computed
//! covers the work, and that the numbers that come back are the numbers a
//! host-side reference computes.
//!
//! Skipped, with a reason, when there is no GPU or no modules. That is the
//! normal state of a build machine and not a failure.
//!
//! # A pass here is weaker than it looks without the validation layer
//!
//! Vulkan answers most malformed requests by doing something undefined rather
//! than by failing, so numbers that match are not evidence that a dispatch was
//! legal. [`Device::validated`] reports whether a layer is watching, and
//! `a_layer_is_watching` prints when one is not. To get one:
//!
//! ```text
//! VK_LAYER_PATH=/path/to/explicit_layer.d cargo test -p driver-vulkan --features native
//! ```
//!
//! # One device, shared, behind a lock
//!
//! Every test here used to open its own device, and therefore its own
//! instance. Twice, under the validation layer, the process SIGABRTed after
//! the last test with every test reporting `ok`, no VUID and no message.
//!
//! That is worth an explicit note, because a SIGABRT after a run of `ok`s
//! looks exactly like a test that broke something, and chasing it cost an
//! hour. What settled it at the time was that a clean checkout did the same,
//! and that any single test under the layer was silent.
//!
//! # What is and is not claimed about that
//!
//! The suite now opens ONE device, made once and never destroyed until the
//! process exits, behind a `Mutex` -- a Vulkan command pool and queue are
//! externally synchronised objects, and sharing without the lock would trade
//! a layer bug for a real one.
//!
//! That was worth doing on its own terms: the suite went from 7.4 seconds to
//! 1.15, because opening sixteen devices was most of what it was doing.
//!
//! It is NOT claimed to have fixed the abort. Reverting to a device per test
//! and running the suite twelve more times under the layer produced twelve
//! clean runs, so the failure is intermittent and no longer reproducible on
//! demand -- which means the control that would prove a fix cannot be made
//! to fire. Sharing one instance removes the most plausible cause, a race in
//! the layer's own teardown across concurrently destroyed instances, and
//! that is as much as the evidence supports.

use driver_vulkan::device::{Bound, Device, Failed, Pipelines, groups_for};
use driver_vulkan::{Dims, Rule};
use kernels_vulkan::Capability;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Where a `native` build of `kernels-vulkan` left the modules.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

/// The one device this suite opens, and the lock that serialises it.
///
/// `None` when there is no device to open or no modules to run on it, so
/// that a machine without a GPU skips rather than fails.
static GPU: OnceLock<Option<Mutex<Device>>> = OnceLock::new();

/// A borrow of the shared device, or `None` to skip.
fn gpu() -> Option<MutexGuard<'static, Device>> {
    let held = GPU.get_or_init(|| match Device::open() {
        Ok(d) => Some(Mutex::new(d)),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    });
    // A poisoned lock means an earlier test panicked while holding the
    // device. The device itself is still usable -- nothing this suite does
    // leaves it in a broken state, and a panicking test has already been
    // reported -- so the remaining tests run rather than cascading into a
    // second failure that says nothing.
    held.as_ref()
        .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// Borrow the shared device and the module directory, or skip saying why.
macro_rules! gpu {
    () => {{
        let Some(dir) = SPV_DIR else {
            eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
            return;
        };
        let Some(device) = gpu() else {
            return;
        };
        (device, std::path::Path::new(dir))
    }};
}

/// The bf16 narrowing `common/bf16.glsl` does, in Rust.
///
/// Round to nearest even. A truncating `(bits >> 16) as u16` agrees on most
/// inputs and disagrees on exactly the ones a tolerance check is least likely
/// to notice.
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

/// Read a module by entrypoint name.
fn module(dir: &std::path::Path, entrypoint: &str) -> Vec<u8> {
    let path = dir.join(format!("{entrypoint}.spv"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// A device opens, and says whether anything is checking it.
///
/// Not a formality. Every other test here would pass against a driver that was
/// quietly doing something undefined, so whether a layer is present changes
/// what the rest of this file is evidence FOR, and that belongs in the output
/// rather than in someone's memory of how they ran it.
#[test]
fn a_device_opens_and_says_whether_a_layer_is_watching() {
    let (device, _) = gpu!();
    assert!(!device.name().is_empty());
    assert!(device.max_push() >= 128, "Vulkan guarantees at least 128");
    if device.validated() {
        eprintln!("{}: the validation layer is watching", device.name());
    } else {
        eprintln!(
            "{}: NO validation layer. These tests still compare numbers, but a \
             pass is not evidence that a dispatch was legal -- set VK_LAYER_PATH.",
            device.name()
        );
    }
}

/// A buffer of no bytes is a buffer of four, and the card accepts it.
///
/// `Device::buffer` rounds a zero-length upload up to four bytes, because a
/// zero-sized buffer is illegal Vulkan and an operand a variant never reads
/// still needs a descriptor pointing somewhere. Deleting the round-up changed
/// no test: every buffer in this suite has contents, so the one input the
/// clamp exists for was never sent.
///
/// It is asked here rather than in a unit test because the claim is about
/// what the DRIVER accepts, and the only thing that can answer that is a
/// driver -- with a validation layer watching, which is where an illegal size
/// would be reported.
#[test]
fn a_buffer_of_no_bytes_is_still_a_buffer_the_card_accepts() {
    let (device, _) = gpu!();
    let empty = device.buffer(&[]).expect("an empty upload allocates");
    // Four and not zero, and read back through the same path everything else
    // here uses, so the size is the driver's answer and not this crate's.
    let back = device.read(&empty).expect("an empty buffer reads back");
    assert_eq!(back.len(), 4, "an empty upload did not become four bytes");
    // And it can be BOUND, which is the reason the round-up exists: a
    // descriptor has to point at a range, and a range of nothing is refused
    // one line further down.
    assert!(
        driver_vulkan::device::Bound::at(&device, &empty, 0, 4).is_ok(),
        "a rounded-up buffer cannot be bound, so the round-up bought nothing"
    );
    // The control: the same buffer with a range of nothing is still refused,
    // so the round-up did not turn an empty binding into a legal one.
    assert!(
        matches!(
            driver_vulkan::device::Bound::at(&device, &empty, 0, 0),
            Err(driver_vulkan::device::Failed::Overrun { len: 0, .. })
        ),
        "a zero-length range was accepted"
    );
    device.free(empty);
}

/// A row-wise norm, driven entirely through this crate's own API.
///
/// The end-to-end case: the pipeline's layout comes from what the module
/// declares, the grid comes from [`driver_vulkan::geometry`], and the answer is
/// compared against a host reference. Nothing in the path is the test's own
/// arithmetic except the reference.
#[test]
fn a_row_norm_computes_what_a_host_reference_computes() {
    let (device, dir) = gpu!();

    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";

    // 1024 wide and not a round 1024-of-something: the point is the whole row,
    // and the values are deliberately not a ramp, since neighbouring elements
    // of a ramp are nearly equal and an indexing error would not move the sum.
    let axis = 1024usize;
    let x: Vec<f32> = (0..axis)
        .map(|i| ((i * 37 % 71) as f32 - 35.0) / 16.0)
        .collect();
    let w: Vec<f32> = (0..axis).map(|i| 0.5 + (i % 13) as f32 / 32.0).collect();
    let eps = 1e-5f32;

    let mut params = Vec::new();
    params.extend_from_slice(&eps.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes()); // w_stride
    params.extend_from_slice(&0u32.to_le_bytes()); // plus_one
    params.extend_from_slice(&1.0f32.to_le_bytes()); // gain

    let xb = bf16_bytes(&x);
    let wb = bf16_bytes(&w);
    let bufs = [
        device.buffer(&xb).expect("x"),
        device.buffer(&wb).expect("w"),
        device.buffer(&vec![0u8; axis * 2]).expect("out"),
        device.buffer(&params).expect("params"),
    ];

    let code = module(dir, entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");

    // The grid this crate computes, from the module this crate loaded. One
    // workgroup: the rule is one per axis, and the axis is the row.
    let dims = Dims {
        rows: 1,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    assert_eq!(groups, [1, 1, 1], "one workgroup for one row of one axis");

    let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device.run(pipeline, &bound, &[], groups).expect("dispatch");

    let got = bf16_read(&device.read(&bufs[2]).expect("read back"));

    // The reference reads back the bf16 the DEVICE was given, not the f32 this
    // test started from. Comparing against the f32 would fold the input's own
    // rounding into the tolerance and quietly widen it.
    let xq = bf16_read(&xb);
    let wq = bf16_read(&wb);
    let mean: f32 = xq.iter().map(|v| v * v).sum::<f32>() / axis as f32;
    let inv = 1.0 / (mean + eps).sqrt();

    for (i, (g, (v, gain))) in got.iter().zip(xq.iter().zip(&wq)).enumerate() {
        let want = gain * (v * inv);
        assert!(
            (g - want).abs() <= 8e-3 * want.abs().max(1.0),
            "element {i}: the device says {g}, the reference says {want}"
        );
    }

    cache.clear(&device);
    for b in bufs {
        device.free(b);
    }
}

/// The grid is what makes the answer whole, and one workgroup short is silent.
///
/// The session's central lesson, as a test rather than a comment. The same
/// dispatch is run twice over four rows: once with the grid this crate computes
/// and once with one workgroup removed. The first fills the output; the second
/// leaves the last row holding the zeros the buffer was born with, returns
/// success from every call, and reports nothing.
#[test]
fn a_grid_one_workgroup_short_leaves_the_tail_as_it_found_it() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";

    let axis = 256usize;
    let rows = 4usize;
    let x: Vec<f32> = (0..axis * rows).map(|i| 1.0 + (i % 7) as f32).collect();
    let w = vec![1.0f32; axis];

    let mut params = Vec::new();
    params.extend_from_slice(&1e-5f32.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());

    let code = module(dir, entrypoint);
    let dims = Dims {
        rows: rows as u32,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };

    // Built once, up front: the cache hands out a borrow, so a closure that
    // both builds and dispatches would hold it across two calls.
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");
    let whole = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    assert_eq!(whole, [4, 1, 1], "one workgroup per row");

    let run = |groups: [u32; 3]| -> Vec<f32> {
        let bufs = [
            device.buffer(&bf16_bytes(&x)).expect("x"),
            device.buffer(&bf16_bytes(&w)).expect("w"),
            device.buffer(&vec![0u8; axis * rows * 2]).expect("out"),
            device.buffer(&params).expect("params"),
        ];
        let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
        device.run(pipeline, &bound, &[], groups).expect("dispatch");
        let out = bf16_read(&device.read(&bufs[2]).expect("read back"));
        for b in bufs {
            device.free(b);
        }
        out
    };

    let full = run(whole);
    assert!(
        full.iter().all(|v| *v != 0.0),
        "the whole grid should write every element"
    );

    let short = run([whole[0] - 1, whole[1], whole[2]]);
    let tail = &short[axis * (rows - 1)..];
    assert!(
        tail.iter().all(|v| *v == 0.0),
        "one workgroup short should leave the last row untouched"
    );
    // And the part that DID run is identical, which is what makes this silent:
    // there is no corruption to notice, only an absence.
    assert_eq!(
        &short[..axis * (rows - 1)],
        &full[..axis * (rows - 1)],
        "the rows that ran should be unaffected"
    );

    let _ = run;
    cache.clear(&device);
}

/// A pipeline's layout comes from the MODULE, so an unstated row still loads.
///
/// 292 of the 480 entrypoints name no operands, including `affine_qmm_t` and
/// most of what a model actually runs. A layout built from such a row has no
/// descriptors, and that is not an error return — it is a segfault inside
/// `vkCreateComputePipelines`. Building from the module's own declared bindings
/// is what makes them loadable, and this fires a sample of them.
#[test]
fn an_entrypoint_whose_row_names_no_operands_still_builds_a_pipeline() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();

    let mut unstated = 0;
    for name in kernels_vulkan::entrypoints() {
        let Some(row) = kernels::sig_in(kernels_vulkan::KERNELS, &name) else {
            continue;
        };
        if row.launch != Rule::Unstated {
            continue;
        }
        let path = dir.join(format!("{name}.spv"));
        if !path.exists() {
            continue;
        }
        let code = std::fs::read(&path).expect("readable");
        // The row states no scalars, so the widest legal range is the only
        // safe one: any block the module declares fits inside it, and a range
        // narrower than what the shader reads is rejected.
        let push = device.max_push();
        let pipeline = cache
            .get(&device, &name, &code, push, 0, Capability::Baseline)
            .unwrap_or_else(|e| panic!("`{name}` has no pipeline: {e}"));
        assert_eq!(
            pipeline.bindings(),
            pipeline.declared().bindings,
            "`{name}`'s layout should have a descriptor per declared binding"
        );
        unstated += 1;
        if unstated == 40 {
            break;
        }
    }
    assert_eq!(unstated, 40, "only {unstated} unstated entrypoints loaded");
    cache.clear(&device);
}

/// A dispatch that does not match the module is refused before it is submitted.
///
/// All three are things Vulkan will happily do something undefined about. A
/// short descriptor set leaves the shader reading a descriptor nothing filled;
/// a short push block leaves it reading the previous dispatch's scalars, which
/// are plausible numbers; a zero workgroup count is legal, runs nothing, and
/// returns success.
#[test]
fn a_call_that_does_not_match_the_module_is_refused() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";
    let code = module(dir, entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");

    let b = device.buffer(&[0u8; 64]).expect("a buffer");
    let one = [Bound::whole(&b)];
    assert!(
        matches!(
            device.run(pipeline, &one, &[], [1, 1, 1]),
            Err(Failed::Bindings { .. })
        ),
        "one buffer under a four-binding module should be refused"
    );

    let four = [
        Bound::whole(&b),
        Bound::whole(&b),
        Bound::whole(&b),
        Bound::whole(&b),
    ];
    assert!(
        matches!(
            device.run(pipeline, &four, &[1, 2, 3, 4], [1, 1, 1]),
            Err(Failed::Push { .. })
        ),
        "four push bytes against a zero-byte range should be refused"
    );
    assert!(
        matches!(
            device.run(pipeline, &four, &[], [1, 0, 1]),
            Err(Failed::Vulkan(_))
        ),
        "a dispatch of no workgroups should be refused"
    );

    device.free(b);
    cache.clear(&device);
}

/// A row that lists a buffer its shader never reads still gets a layout for it.
///
/// `layer_scalar_mul_bfloat16` is one of the eleven entrypoints where the two
/// counts disagree: the row lists four buffers and the compiled module
/// decorates three, because glslc drops the `OpDecorate Binding` of one the
/// shader never reads. Building the layout from the module alone gives three
/// descriptors, and the caller -- who has the row, and four buffers to bind --
/// is then refused at `run` for a call that is perfectly legal.
///
/// So this binds all four and requires the dispatch to go through. The
/// opposite mistake is not testable here and does not need to be: a layout
/// SHORTER than the module reads is a segmentation fault inside
/// `vkCreateComputePipelines`, which is why `Pipelines::get` takes the maximum
/// of the two rather than trusting either.
#[test]
fn a_buffer_the_shader_never_reads_is_still_given_a_descriptor() {
    let (device, dir) = gpu!();
    let name = "layer_scalar_mul_bfloat16";
    let path = dir.join(format!("{name}.spv"));
    if !path.exists() {
        eprintln!("skipped: `{name}` was not built");
        return;
    }
    let row = kernels::sig_in(kernels_vulkan::KERNELS, name).expect("the row resolves");
    let stated = kernels_vulkan::buffer_count(row);
    let code = std::fs::read(&path).expect("readable");

    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(
            &device,
            name,
            &code,
            kernels_vulkan::push_size(row),
            stated,
            Capability::Baseline,
        )
        .expect("the pipeline builds");

    assert!(
        stated > pipeline.declared().bindings,
        "`{name}` was chosen because its row ({stated}) outruns its module \
         ({}); if that stopped being true the test proves nothing and should \
         be pointed at another of the eleven",
        pipeline.declared().bindings
    );
    assert_eq!(
        pipeline.bindings(),
        stated,
        "the layout has to cover what the CALLER binds, not what the module \
         happens to read"
    );

    let buffers: Vec<_> = (0..stated)
        .map(|_| device.buffer(&vec![0u8; 256]).expect("buffer"))
        .collect();
    let refs: Vec<Bound<'_>> = buffers.iter().map(Bound::whole).collect();
    let push = vec![0u8; kernels_vulkan::push_size(row) as usize];
    device
        .run(pipeline, &refs, &push, [1, 1, 1])
        .expect("a dispatch binding every buffer the row lists is accepted");

    for b in buffers {
        device.free(b);
    }
    cache.clear(&device);
}

/// Every entrypoint resolves to a module, whatever this device supports.
///
/// The backward-compatibility guarantee, asked of a real device rather than
/// of the directory listing. A tier is an ADDITIONAL module for an entrypoint
/// that already exists -- never a new entrypoint and never a replacement -- so
/// a machine offering nothing optional must still resolve all 480, and a
/// machine offering everything must resolve the same 480 and no more.
///
/// It also asserts the tiers are best-first, because `module_for` takes the
/// first match and an unsorted list would silently prefer the baseline on a
/// device that has a matrix unit: not an error, not a wrong answer, just the
/// whole tier mechanism doing nothing.
#[test]
fn every_entrypoint_resolves_to_a_module_this_device_can_load() {
    let (device, dir) = gpu!();
    let tiers = device.tiers();
    assert!(
        tiers.contains(&Capability::Baseline),
        "the baseline tier is not optional: it is what every entrypoint has"
    );
    let mut sorted = tiers.to_vec();
    sorted.sort_unstable();
    sorted.reverse();
    assert_eq!(tiers, sorted, "the tiers must be offered best first");

    let mut resolved = 0;
    let mut by_tier: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut missing: Vec<String> = Vec::new();
    for name in kernels_vulkan::entrypoints() {
        match device.module_for(dir, &name) {
            Some((_, tier)) => {
                *by_tier.entry(tier.tag().to_string()).or_default() += 1;
                resolved += 1;
            }
            None => missing.push(name),
        }
    }
    assert!(
        missing.is_empty(),
        "{} entrypoints have no module at any tier this device can load: {}",
        missing.len(),
        missing.join(", ")
    );
    assert!(resolved >= 400, "only {resolved} entrypoints resolved");
    eprintln!("{}: {tiers:?}, modules by tier {by_tier:?}", device.name());
}

/// The best module this device can load actually builds a pipeline.
///
/// Resolving a path is not loading it. A tier is a separate BODY compiled with
/// different extensions, so `@coopmat` failing on a device that reports
/// `cooperativeMatrix` is exactly the kind of thing that stays invisible until
/// a particular GPU meets a particular model -- and the feature list that
/// makes it loadable is four names deep, one of which
/// (`vulkanMemoryModelDeviceScope`) is needed by a BASELINE kernel that has
/// nothing to do with matrices.
///
/// A sample rather than the whole table, because building 480 pipelines is a
/// different test with a different runtime; `every_module_this_device_claims_
/// it_can_load_builds_a_pipeline` in `kernels-vulkan` is that one.
#[test]
fn the_tier_this_device_selects_is_one_it_can_actually_load() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let mut built = 0;
    let mut failures: Vec<String> = Vec::new();

    for name in kernels_vulkan::entrypoints().into_iter().take(40) {
        let Some(row) = kernels::sig_in(kernels_vulkan::KERNELS, &name) else {
            continue;
        };
        let Some((path, tier)) = device.module_for(dir, &name) else {
            continue;
        };
        let code = std::fs::read(&path).expect("readable");
        // The row's own numbers when it has them. An unstated row supplies no
        // layout, so the module's own declarations are all there is -- which
        // is why `get` takes the maximum of the two rather than either.
        let (push, descriptors) = if row.operands.is_empty() {
            (device.max_push(), 0)
        } else {
            (
                kernels_vulkan::push_size(row),
                kernels_vulkan::buffer_count(row),
            )
        };
        match cache.get(&device, &name, &code, push, descriptors, tier) {
            Ok(_) => built += 1,
            Err(e) => failures.push(format!("`{name}` at {}: {e}", tier.tag())),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(built >= 20, "only {built} pipelines were built");
    cache.clear(&device);
}

/// The block `pack` builds is the block the shader reads, on the device.
///
/// `tests/rules.rs` compares the row's push layout to the module's `Offset`
/// decorations, which is two DESCRIPTIONS agreeing. This is the other kind of
/// evidence: the bytes this crate packs are handed to a real shader, and the
/// destination it computes from them is checked against one a host computes
/// the same way.
///
/// `kv_append` is the row for it because it exercises both runs at once --
/// five buffers and three scalars, two of them 64-bit and therefore padded --
/// and because its arithmetic is addressing rather than mathematics. A stride
/// read four bytes early is not an approximate answer, it is a write to
/// somewhere else entirely, so the check is exact and the tolerance is zero.
#[test]
fn the_scalars_this_crate_packs_are_the_ones_the_shader_addresses_with() {
    let (device, dir) = gpu!();
    let entrypoint = "kv_append_bfloat16";
    let path = dir.join(format!("{entrypoint}.spv"));
    if !path.exists() {
        eprintln!("skipped: `{entrypoint}` was not built");
        return;
    }
    let row = kernels::sig_in(kernels_vulkan::KERNELS, "kv_append").expect("the row is stated");

    // A cache of `kv_heads` heads, each `seq` slots of `head_dim`. The append
    // writes ONE position, which is what this kernel is: `pos[0]` is a scalar
    // slot and not a per-row table.
    let head_dim = 64usize;
    let kv_heads = 2usize;
    let seq = 8usize;
    let pos = 3u32;

    let k_new: Vec<f32> = (0..kv_heads * head_dim)
        .map(|i| (i % 17) as f32 - 8.0)
        .collect();
    let v_new: Vec<f32> = (0..kv_heads * head_dim)
        .map(|i| (i % 11) as f32 - 5.0)
        .collect();
    let cache_bytes = kv_heads * seq * head_dim * 2;

    let bufs = [
        device.buffer(&bf16_bytes(&k_new)).expect("k_new"),
        device.buffer(&bf16_bytes(&v_new)).expect("v_new"),
        device.buffer(&vec![0u8; cache_bytes]).expect("k_cache"),
        device.buffer(&vec![0u8; cache_bytes]).expect("v_cache"),
        device.buffer(&pos.to_le_bytes()).expect("pos"),
    ];

    // The values in the ROW's order. `pack` decides which of them is a
    // descriptor and which is a push field, and where each lands.
    let call = driver_vulkan::pack(
        row,
        &[
            driver_vulkan::Value::Buffer(0),
            driver_vulkan::Value::Buffer(1),
            driver_vulkan::Value::Buffer(2),
            driver_vulkan::Value::Buffer(3),
            driver_vulkan::Value::Buffer(4),
            driver_vulkan::Value::I32(head_dim as i32),
            driver_vulkan::Value::Usize((seq * head_dim) as u64),
            driver_vulkan::Value::Usize(head_dim as u64),
        ],
    )
    .expect("every operand is the kind the row wants");

    let code = std::fs::read(&path).expect("readable");
    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(
            &device,
            entrypoint,
            &code,
            kernels_vulkan::push_size(row),
            kernels_vulkan::buffer_count(row),
            Capability::Baseline,
        )
        .expect("the pipeline builds");

    let dims = Dims {
        rows: 1,
        head_dim: head_dim as u32,
        kv_heads: kv_heads as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::PerHead, dims, pipeline).expect("a geometry");

    // The descriptor order `pack` chose, and not the order the buffers were
    // created in. They agree here, and the test would still be worth writing
    // if they did not -- that is the mapping being exercised.
    let bound: Vec<Bound<'_>> = call
        .buffers
        .iter()
        .map(|i| Bound::whole(&bufs[*i as usize]))
        .collect();
    device
        .run(pipeline, &bound, &call.push, groups)
        .expect("dispatch");

    let got = bf16_read(&device.read(&bufs[2]).expect("read k_cache back"));
    let want = bf16_read(&bf16_bytes(&k_new));
    for h in 0..kv_heads {
        for d in 0..head_dim {
            let at = h * seq * head_dim + pos as usize * head_dim + d;
            assert_eq!(
                got[at],
                want[h * head_dim + d],
                "head {h} element {d} landed somewhere else, which is what a \
                 stride packed at the wrong offset does"
            );
        }
        // And nowhere else. A stride read four bytes early would still write
        // SOMETHING, at a plausible address, and comparing only the intended
        // slot would call that a pass.
        for s in (0..seq).filter(|s| *s != pos as usize) {
            let at = h * seq * head_dim + s * head_dim;
            assert!(
                got[at..at + head_dim].iter().all(|v| *v == 0.0),
                "head {h} slot {s} was written and should not have been"
            );
        }
    }

    cache.clear(&device);
    for b in bufs {
        device.free(b);
    }
}

/// One arena, four operands at offsets inside it, and the shader addresses
/// each from its own start.
///
/// This is the allocation model a driver actually has. `driver-metal`'s binder
/// resolves every operand to an offset into one arena, and nothing above it
/// allocates per tensor -- so a Vulkan shell that can only bind whole buffers
/// can run the tests in this file and still not run a model.
///
/// What makes it worth a GPU rather than a unit test is that the offset is the
/// device's to honour. `Bound` writes it into the descriptor and the shader
/// never learns it: every index the shader computes is relative to zero. If
/// the descriptor's base were ignored -- or if `range` were `WHOLE_SIZE` and
/// the base were dropped -- every operand would read the arena from the front
/// and this test would get the FIRST row back for all three of them.
///
/// The row chosen is deliberately not the first: reading offset zero when a
/// nonzero one was asked for is exactly the failure, and a test on row 0
/// cannot see it.
#[test]
fn an_operand_at_an_offset_in_one_arena_is_addressed_from_that_offset() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";

    let align = device.min_storage_offset();
    assert!(
        align.is_power_of_two(),
        "the specification requires a power of two and `Bound::at` may mask; \
         this device reports {align}"
    );

    let axis = 256usize;
    let row_bytes = (axis * 2) as u64;
    // Three rows of input, and the one the dispatch is aimed at is the middle.
    let rows: Vec<Vec<f32>> = (0..3)
        .map(|r| {
            (0..axis)
                .map(|i| ((i * 37 % 71) as f32 - 35.0) / 16.0 * (r + 1) as f32)
                .collect()
        })
        .collect();
    let w: Vec<f32> = (0..axis).map(|i| 0.5 + (i % 13) as f32 / 32.0).collect();
    let eps = 1e-5f32;

    let mut params = Vec::new();
    params.extend_from_slice(&eps.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());

    // A suballocator, and the only rule it has is the device's. Rounding UP is
    // the whole of it: an allocator that packs tightly produces offsets the
    // device refuses, which is why the limit has to be asked for rather than
    // assumed.
    let bump = |at: u64, len: u64| -> (u64, u64) { (at, (at + len).next_multiple_of(align)) };
    let mut at = 0u64;
    let (x_at, next) = bump(at, row_bytes * 3);
    at = next;
    let (w_at, next) = bump(at, row_bytes);
    at = next;
    let (out_at, next) = bump(at, row_bytes * 3);
    at = next;
    let (p_at, end) = bump(at, params.len() as u64);

    let mut arena = vec![0u8; end as usize];
    for (r, row) in rows.iter().enumerate() {
        let base = x_at as usize + r * row_bytes as usize;
        arena[base..base + row_bytes as usize].copy_from_slice(&bf16_bytes(row));
    }
    let wb = bf16_bytes(&w);
    arena[w_at as usize..w_at as usize + wb.len()].copy_from_slice(&wb);
    arena[p_at as usize..p_at as usize + params.len()].copy_from_slice(&params);

    let buffer = device.buffer(&arena).expect("one arena for the whole fire");
    let target = 1usize;
    let bound = [
        Bound::at(
            &device,
            &buffer,
            x_at + target as u64 * row_bytes,
            row_bytes,
        )
        .expect("the input row"),
        Bound::at(&device, &buffer, w_at, row_bytes).expect("the gain"),
        Bound::at(
            &device,
            &buffer,
            out_at + target as u64 * row_bytes,
            row_bytes,
        )
        .expect("the output row"),
        Bound::at(&device, &buffer, p_at, params.len() as u64).expect("the parameters"),
    ];
    assert!(
        bound[0].offset() != 0 && bound[2].offset() != 0,
        "a test that binds offset zero cannot see a dropped base"
    );

    let code = module(dir, entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");
    let dims = Dims {
        rows: 1,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    device.run(pipeline, &bound, &[], groups).expect("dispatch");

    let back = device.read(&buffer).expect("read the arena back");
    let got = bf16_read(&back[out_at as usize..(out_at + row_bytes * 3) as usize]);
    let xq = bf16_read(&bf16_bytes(&rows[target]));
    let wq = bf16_read(&wb);
    let mean: f32 = xq.iter().map(|v| v * v).sum::<f32>() / axis as f32;
    let inv = 1.0 / (mean + eps).sqrt();

    for (i, (g, (v, gain))) in got[target * axis..(target + 1) * axis]
        .iter()
        .zip(xq.iter().zip(&wq))
        .enumerate()
    {
        let want = gain * (v * inv);
        assert!(
            (g - want).abs() <= 8e-3 * want.abs().max(1.0),
            "element {i} of the row at offset {} is {g} and the host says {want}",
            bound[0].offset()
        );
    }
    // And the output rows on either side are as the arena was born. A dropped
    // base would have written row 0; a `WHOLE_SIZE` range would let an overrun
    // reach row 2. Neither is visible from the target row alone.
    for other in [0usize, 2] {
        assert!(
            got[other * axis..(other + 1) * axis]
                .iter()
                .all(|v| *v == 0.0),
            "row {other} was written and the dispatch was aimed at row {target}"
        );
    }

    cache.clear(&device);
    device.free(buffer);
}

/// A range this device cannot address from is refused before it is written.
///
/// The refusal exists because the alternative is not an error. Written into a
/// descriptor, an unaligned offset is invalid usage: with a layer it is a
/// message, and WITHOUT one it is undefined behaviour that this driver appears
/// to honour anyway. That is the worst available outcome -- it makes the
/// defect a property of the machine the code was tested on, and it moves when
/// the model does.
#[test]
fn a_range_the_device_cannot_address_from_is_refused_by_this_crate() {
    let (device, _) = gpu!();
    let align = device.min_storage_offset();
    let buffer = device.buffer(&vec![0u8; 4096]).expect("an arena");

    if align > 1 {
        assert!(
            matches!(
                Bound::at(&device, &buffer, align - 1, 16),
                Err(Failed::Unaligned { .. })
            ),
            "an offset one byte before a legal one must be refused, and \
             {align} is what this device asks for"
        );
        assert!(
            Bound::at(&device, &buffer, align, 16).is_ok(),
            "and the legal one next to it must not be"
        );
    }

    // Past the end, and exactly at it. The second is the one a length computed
    // from an off-by-one shape produces, and it is the one `WHOLE_SIZE` would
    // have hidden.
    assert!(matches!(
        Bound::at(&device, &buffer, 0, 4097),
        Err(Failed::Overrun { .. })
    ));
    assert!(matches!(
        Bound::at(&device, &buffer, 4096, 16),
        Err(Failed::Overrun { .. })
    ));
    assert!(
        matches!(
            Bound::at(&device, &buffer, 0, 0),
            Err(Failed::Overrun { .. })
        ),
        "an empty range is illegal Vulkan and is always a width that came out \
         zero"
    );
    // A wrapping sum would land inside the buffer and pass a bound it is
    // nowhere near.
    assert!(matches!(
        // Aligned to 4096, which every alignment a device may report divides,
        // so this is refused for the overrun and not incidentally for the
        // offset.
        Bound::at(&device, &buffer, 0xFFFF_FFFF_FFFF_F000, 4096),
        Err(Failed::Overrun { .. })
    ));

    device.free(buffer);
}

/// The range in a descriptor is what confines an overrun, and it only does
/// that if it is the operand's own extent.
///
/// `VK_WHOLE_SIZE` is the easy thing to write and it means "from here to the
/// end of the buffer". In a one-buffer-per-tensor world that is the same
/// answer. In an ARENA it is not: every operand's range then covers every
/// operand allocated after it, and a shader that writes one element too far
/// writes into the next tensor. Nothing reports it. The next kernel reads a
/// value that was computed, by a real kernel, from real inputs -- it is simply
/// the wrong tensor's.
///
/// This is checkable rather than merely prudent because `robustBufferAccess`
/// is enabled -- the crate docs require it for the tiled GEMM's ragged fetch.
/// With it on, a write outside the bound range is DISCARDED, so the confinement
/// is defined behaviour and not an accident of this driver. The test binds an
/// output range half the row the grid covers and asserts the discarded half
/// went nowhere.
///
/// Written as a control that cannot be skipped: with `range` widened to
/// `WHOLE_SIZE` the whole rest of the arena is in scope and the canary is
/// overwritten.
#[test]
fn an_operand_overrunning_its_range_is_discarded_rather_than_given_to_its_neighbour() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";

    let align = device.min_storage_offset();
    let axis = 256usize;
    let row_bytes = (axis * 2) as u64;

    let x: Vec<f32> = (0..axis)
        .map(|i| ((i * 37 % 71) as f32 - 35.0) / 16.0)
        .collect();
    let w: Vec<f32> = (0..axis).map(|i| 0.5 + (i % 13) as f32 / 32.0).collect();
    let mut params = Vec::new();
    params.extend_from_slice(&1e-5f32.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());

    let x_at = 0u64;
    let w_at = row_bytes.next_multiple_of(align);
    let out_at = (w_at + row_bytes).next_multiple_of(align);
    // The neighbour starts where the BOUND range ends, not where the row
    // ends. That distinction is the test: an operand given half a row still
    // has the other half of its row space allocated to something, and putting
    // the canary a whole row away would leave the overrun landing in the gap
    // between them, where nobody is looking. It was written that way first and
    // the control did not fire.
    let half = row_bytes / 2;
    let canary_at = out_at + half;
    let p_at = (out_at + row_bytes).next_multiple_of(align);
    let end = p_at + params.len() as u64;

    let mut arena = vec![0u8; end as usize];
    arena[x_at as usize..x_at as usize + row_bytes as usize].copy_from_slice(&bf16_bytes(&x));
    arena[w_at as usize..w_at as usize + row_bytes as usize].copy_from_slice(&bf16_bytes(&w));
    arena[p_at as usize..p_at as usize + params.len()].copy_from_slice(&params);
    let canary = bf16_bytes(&vec![7.0f32; axis / 2]);
    arena[canary_at as usize..canary_at as usize + half as usize].copy_from_slice(&canary);

    let buffer = device.buffer(&arena).expect("one arena");

    // Half a row of output, and a grid that covers a whole one. The shader is
    // not being asked to behave; it is being confined.
    let bound = [
        Bound::at(&device, &buffer, x_at, row_bytes).expect("x"),
        Bound::at(&device, &buffer, w_at, row_bytes).expect("w"),
        Bound::at(&device, &buffer, out_at, half).expect("half an output row"),
        Bound::at(&device, &buffer, p_at, params.len() as u64).expect("params"),
    ];

    let code = module(dir, entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");
    let dims = Dims {
        rows: 1,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    device.run(pipeline, &bound, &[], groups).expect("dispatch");

    let back = device.read(&buffer).expect("read the arena back");
    let wrote = bf16_read(&back[out_at as usize..(out_at + half) as usize]);
    assert!(
        wrote.iter().any(|v| *v != 0.0),
        "the half that IS in range must have been written, or this test proves \
         only that the dispatch did nothing"
    );
    let after = bf16_read(&back[canary_at as usize..(canary_at + half) as usize]);
    assert!(
        after.iter().all(|v| *v == 7.0),
        "the neighbouring tensor was overwritten by an operand that ran past \
         its own extent, which is what a `WHOLE_SIZE` range permits"
    );

    cache.clear(&device);
    device.free(buffer);
}

/// A parameter block one word short is refused, because the device will not
/// object and the answer will look fine.
///
/// This is the defect `driver-metal` was found carrying in two kernels: a
/// packed run sized from the text's parameter count while the shader reads its
/// struct's word count. On Metal the shader then read the NEXT dispatch's
/// scalars, which at least varies. Here it is quieter -- `robustBufferAccess`
/// is on for the tiled GEMM's ragged fetch, so the words past the range read
/// as ZERO, and a zero pitch or a zero flag is a value somebody could have
/// meant.
///
/// Run as a control -- with the refusal disabled -- the call is ACCEPTED, 256
/// of 256 outputs come back zero, and the validation layer says nothing about
/// it. That is the measurement this refusal exists for: the layer catches
/// illegal usage, and a range that is legal but too small for what the shader
/// reads is not illegal. Nothing but this check is looking.
#[test]
fn a_parameter_block_short_of_what_the_shader_reads_is_refused() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";

    let axis = 256usize;
    let row_bytes = (axis * 2) as u64;
    let x: Vec<f32> = (0..axis).map(|i| ((i % 23) as f32 - 11.0) / 8.0).collect();
    let w = vec![1.0f32; axis];

    // The block is 20 bytes: eps, axis, w_stride, plus_one, gain. `gain` is
    // last, and a run four bytes short drops it -- which reads as zero, which
    // scales the whole row to zero. A plausible-looking answer that is also
    // visibly wrong is exactly what makes this checkable.
    let mut params = Vec::new();
    params.extend_from_slice(&1e-5f32.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());
    assert_eq!(params.len(), 20, "the block this test is built around");

    let code = module(dir, entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");
    assert_eq!(
        pipeline.declared().block_bytes.get(3),
        Some(&Some(20)),
        "binding 3 is the parameter block and this crate reads it as 20 bytes"
    );

    let bufs = [
        device.buffer(&bf16_bytes(&x)).expect("x"),
        device.buffer(&bf16_bytes(&w)).expect("w"),
        device.buffer(&vec![0u8; row_bytes as usize]).expect("out"),
        device.buffer(&params).expect("params"),
    ];
    let dims = Dims {
        rows: 1,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");

    // What the device does when the range is short, established before the
    // refusal is claimed to be worth having. `gain` is dropped, reads zero,
    // and the whole row comes back zero -- with no error from any call.
    let short = [
        Bound::whole(&bufs[0]),
        Bound::whole(&bufs[1]),
        Bound::whole(&bufs[2]),
        Bound::at(&device, &bufs[3], 0, 16).expect("sixteen bytes is a legal range"),
    ];
    match device.run(pipeline, &short, &[], groups) {
        Err(refused) => assert!(
            matches!(
                refused,
                Failed::Short {
                    binding: 3,
                    needs: 20,
                    given: 16
                }
            ),
            "the short block must be refused for being SHORT, and was refused \
             for {refused}"
        ),
        // Reached only with the refusal removed, which is how the control is
        // run. It reports what the DEVICE did rather than only that this crate
        // failed to stop it: the zeros are the evidence that no layer, no error
        // and no fault stands between a short block and a wrong answer.
        Ok(()) => {
            let got = bf16_read(&device.read(&bufs[2]).expect("read back"));
            let zeros = got.iter().filter(|v| **v == 0.0).count();
            panic!(
                "a 16-byte range under a 20-byte block was accepted, and {zeros} \
                 of {axis} outputs came back zero -- the missing `gain` read as \
                 zero and scaled the row away, with no error from any call"
            );
        }
    }

    // And the full block is not refused, so the check is a floor and not a
    // blanket.
    let full: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    device
        .run(pipeline, &full, &[], groups)
        .expect("the whole block is accepted");
    let got = bf16_read(&device.read(&bufs[2]).expect("read back"));
    assert!(
        got.iter().any(|v| *v != 0.0),
        "with `gain` present the row is not zero, which is what makes its \
         absence visible"
    );

    cache.clear(&device);
    for b in bufs {
        device.free(b);
    }
}

/// A module with a descriptor hole dispatches without one buffer per slot.
///
/// Metal charges nothing for a hole: an argument index nothing is set at is an
/// index the shader does not read. Vulkan looks like it must charge for one,
/// because a descriptor set covers every number up to the highest and there is
/// no way to say a slot is absent -- so a driver would have to find a buffer
/// for a binding no shader reads, and the plan does not name one.
///
/// It does not have to. The specification says descriptors must be valid *if
/// they are accessed*, and this measures that rather than reading it:
/// `affine_qmv_routed` has seven slots and six real bindings, a real lowering
/// states six operands for it, and six is what dispatches.
///
/// # Why it matters beyond saving a buffer
///
/// Counting slots made `affine_qmv_routed` look like a kernel needing a
/// resource the plan does not supply -- `tests/arena.rs` classified it that
/// way before this was measured. Counting decorated bindings makes it a kernel
/// that simply binds. A hole and a driver-owned resource are the same
/// arithmetic and completely different facts.
#[test]
fn a_module_with_a_descriptor_hole_binds_only_what_it_declares() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();

    for entrypoint in [
        // Six of seven: the routed matrix-vector kernel a real MoE text fires.
        "affine_qmv_routed_bfloat16_gs_64_b_4",
        // Five of seven, two holes, and from a different family -- so this is
        // not one shader's quirk.
        "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_64_bn_32",
    ] {
        let code = module(dir, entrypoint);
        let words = driver_vulkan::spirv::words(&code).expect("whole words");
        let declared = driver_vulkan::spirv::declared(&words).expect("well formed");
        let real = declared.bindings as usize - declared.holes();
        assert!(
            real < declared.bindings as usize,
            "`{entrypoint}` has no hole, so it cannot show that one is free"
        );

        let push = declared
            .push_offsets
            .iter()
            .map(|o| *o as usize + 4)
            .max()
            .unwrap_or(0);
        let Ok(pipeline) = cache.get(
            &device,
            entrypoint,
            &code,
            push as u32,
            declared.bindings,
            Capability::Baseline,
        ) else {
            eprintln!("skipped {entrypoint}: this device cannot build it");
            continue;
        };

        // One per DECORATED binding. Asking for one per slot is what this
        // test exists to say a driver does not have to do.
        let bufs: Vec<_> = (0..real)
            .map(|_| device.buffer(&vec![0u8; 65536]).expect("a buffer"))
            .collect();
        let bound: Vec<_> = bufs.iter().map(Bound::whole).collect();
        let answer = device.run(pipeline, &bound, &vec![0u8; push], [1, 1, 1]);

        // The buffers go back before the assertion. A panic that skipped this
        // makes the layer abort inside `vkDestroyDevice`, which replaces the
        // real failure with a SIGABRT nobody can read.
        for b in bufs {
            device.free(b);
        }
        assert!(
            answer.is_ok(),
            "`{entrypoint}` has {} slots and {real} real bindings, and binding \
             the real ones was refused: {answer:?}",
            declared.bindings
        );
    }
    cache.clear(&device);
}

/// One buffer per slot is refused, and so is one too few.
///
/// The count `run` wants is the decorated bindings, which for a holed module
/// is neither the slot count nor anything a caller would guess. Both
/// directions are checked because a driver that accepted the slot count would
/// silently shift every operand past the hole onto the wrong binding -- the
/// descriptor writes are positional, so the shader would read its scales
/// where its weights belong and return a plausible number.
#[test]
fn a_holed_module_refuses_a_buffer_for_every_slot() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "affine_qmv_routed_bfloat16_gs_64_b_4";
    let code = module(dir, entrypoint);
    let words = driver_vulkan::spirv::words(&code).expect("whole words");
    let declared = driver_vulkan::spirv::declared(&words).expect("well formed");
    let real = declared.bindings as usize - declared.holes();
    let push = declared
        .push_offsets
        .iter()
        .map(|o| *o as usize + 4)
        .max()
        .unwrap_or(0);
    let Ok(pipeline) = cache.get(
        &device,
        entrypoint,
        &code,
        push as u32,
        declared.bindings,
        Capability::Baseline,
    ) else {
        eprintln!("skipped: this device cannot build {entrypoint}");
        return;
    };

    let bufs: Vec<_> = (0..declared.bindings)
        .map(|_| device.buffer(&vec![0u8; 4096]).expect("a buffer"))
        .collect();
    let all: Vec<_> = bufs.iter().map(Bound::whole).collect();
    let too_many = device.run(pipeline, &all, &vec![0u8; push], [1, 1, 1]);
    let too_few = device.run(pipeline, &all[..real - 1], &vec![0u8; push], [1, 1, 1]);
    for b in bufs {
        device.free(b);
    }
    cache.clear(&device);

    for (what, answer) in [("one per slot", too_many), ("one short", too_few)] {
        assert!(
            matches!(&answer, Err(Failed::Bindings { module, .. }) if *module == real as u32),
            "{what} was not refused against the {real} bindings this module \
             declares: {answer:?}"
        );
    }
}

/// A rectangle a real plan states, dispatched on a real device.
///
/// `tests/arena.rs` walks all 3992 rectangles the three texts lower and turns
/// 3180 of them into dispatches. Every one of those numbers is arithmetic on
/// a machine with no GPU in it: offsets a compiler chose, bindings a SPIR-V
/// module decorates, a grid a rule computes. None of it proves a driver can
/// actually record one.
///
/// So this takes the plan `qwen3_0_6b` lowers, allocates its arena for real,
/// asks `plan_one` for a rectangle, and submits it under the validation
/// layer. What it checks is not the arithmetic -- the other file does that,
/// against every rectangle rather than one -- but the two claims arithmetic
/// cannot make:
///
/// * `Device::run` accepts what `plan_one` produced. These two compute the
///   binding arity independently, and if they ever disagreed every dispatch
///   the walk plans would be refused at submission, with the whole GPU-free
///   suite still green.
/// * the layer says nothing. A descriptor range past the end of a buffer, an
///   offset the device cannot align to, a push range shorter than the block
///   -- all of these are things the walk cannot see and GPU-AV reports by
///   VUID.
///
/// One rectangle, not all 3180: the arena is a real allocation and the
/// weights are not present, so this binds a zero-filled stand-in for each.
/// That is enough to exercise every path a dispatch takes and not enough to
/// check a number the kernel produced, which is what the kernel tests are
/// for.
#[test]
fn a_rectangle_a_real_plan_states_records_and_submits() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, Row, lower};
    use model_compiler::trace::FireClass;

    let (device, dir) = gpu!();

    // One stand-in for every weight and seam value. Zeros, because nothing
    // here reads a result back.
    let weights = device.buffer(&vec![0u8; 1 << 22]).expect("weights");
    struct Zeros<'a>(&'a driver_vulkan::device::Buffer);
    impl driver_vulkan::binding::Resolve for Zeros<'_> {
        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
            Some(self.0)
        }
        fn named(
            &self,
            _: model_compiler::trace::ValueId,
        ) -> Option<&driver_vulkan::device::Buffer> {
            Some(self.0)
        }
        // The same stand-in for the KV cache and the fire tables. What this
        // test asks is whether a device takes the descriptor set a row's
        // sources produce, not where the driver's own memory comes from.
        fn kv(&self, _: u16, _: bool) -> Option<&driver_vulkan::device::Buffer> {
            Some(self.0)
        }
        fn table(
            &self,
            _: driver_vulkan::binding::FireTable,
        ) -> Option<&driver_vulkan::device::Buffer> {
            Some(self.0)
        }
    }
    let store = Zeros(&weights);

    let mut cache = Pipelines::new();
    let mut ran = 0u32;
    let mut refused = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Both fire classes of all three texts.
    //
    // Started as one text and one class, which reached nine distinct symbols
    // -- five of them ones already known to be waiting on driver-owned
    // resources, so four dispatches, which proves close to nothing. The
    // router, the sorts and the routed GEMMs exist only in a
    // mixture-of-experts plan; the wide GEMMs only in a prefill; the sink
    // attention only in `gpt_oss_20b`. Each addition was worth one to three
    // more symbols and the sweep is what makes the number thirteen.
    for (facts, metal) in [
        (
            LlamaLikeFacts::qwen3_0_6b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            LlamaLikeFacts::qwen3_30b_a3b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            LlamaLikeFacts::gpt_oss_20b(),
            LlamaLikeMetalFacts::gpt_oss_20b(),
        ),
    ] {
        for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 64)] {
            let plan = llama_like_metal(&facts, &metal, class);
            let low = lower(
                &plan,
                &vec![
                    Row {
                        samples: true,
                        ..Row::default()
                    };
                    rows
                ],
                Fire {
                    captures_across_splits: false,
                },
            )
            .expect("the text lowers");

            // A real allocation of exactly what the plan asked for. Exactly, not
            // generously: a buffer larger than the arena would hide a descriptor
            // range that runs off the end of it, and that overrun is the single
            // thing this backend's `extent` exists to prevent.
            let arena_buffer = device
                .buffer(&vec![0u8; low.arena_bytes])
                .expect("the arena allocates");

            let arena = driver_vulkan::binding::Arena {
                buffer: &arena_buffer,
                bytes: low.arena_bytes as u64,
            };

            // The first rectangle of each distinct symbol the plan reaches. One per
            // symbol rather than all 3992 because the point is coverage of the
            // dispatch PATHS, and a plan repeats each symbol once per layer.
            for launch in &low.launches {
                let symbol = low.kernels[launch.kernel as usize].as_str();
                if !seen.insert(symbol.to_owned()) {
                    continue;
                }
                let Ok(code) = std::fs::read(dir.join(format!("{symbol}.spv"))) else {
                    continue;
                };
                let words = driver_vulkan::spirv::words(&code).expect("whole words");
                let declared = driver_vulkan::spirv::declared(&words).expect("well formed");
                let module = driver_vulkan::geometry::Module::named(
                    symbol,
                    [declared.local[0], declared.local[1], declared.local[2]],
                );
                let planned = driver_vulkan::dispatch::plan_one(
                    &low,
                    launch,
                    kernels_vulkan::KERNELS,
                    driver_vulkan::dispatch::Built {
                        module,
                        declared: &declared,
                    },
                    driver_vulkan::dispatch::Sources {
                        arena,
                        resolver: &store,
                        min_offset: device.min_storage_offset(),
                    },
                    driver_vulkan::dispatch::Geometry {
                        q_heads: facts.q_heads,
                        kv_heads: facts.kv_heads,
                        head_dim: facts.head_dim,
                        rotary_dims: facts.head_dim,
                        n_experts: facts.n_experts,
                        experts_per_token: facts.experts_per_token,
                    },
                );
                // The six symbols the GPU-free walk already names as short of a
                // driver-owned resource. Skipped rather than asserted about, because
                // `tests/arena.rs` names them one at a time with counts; repeating
                // that here would give two places to update and one of them would
                // rot.
                let Ok(d) = planned else {
                    continue;
                };

                // The parameter block is the CALLER's to allocate -- `plan_one` is
                // arithmetic and a buffer needs a device. This is the splice, and it
                // is the only thing between a planned dispatch and a submitted one.
                // Spliced at `block_at`, not at the `Params::Block` slot:
                // the first is an index into the DENSE list a descriptor set
                // is written from, and the second is a binding number. They
                // agree on every module in this tree and would not on a
                // module with a hole below its block.
                let block = match (&d.params, d.block_at) {
                    (driver_vulkan::binding::Params::Block { bytes, .. }, Some(at)) => {
                        Some((device.buffer(bytes).expect("the block allocates"), at))
                    }
                    _ => None,
                };
                let mut buffers: Vec<Bound<'_>> = d.buffers.clone();
                if let Some((buf, at)) = &block {
                    buffers.insert(*at, Bound::whole(buf));
                }
                let push: &[u8] = match &d.params {
                    driver_vulkan::binding::Params::Push(b) => b,
                    _ => &[],
                };

                let pipeline = cache
                    .get(
                        &device,
                        symbol,
                        &code,
                        push.len() as u32,
                        buffers.len() as u32,
                        Capability::Baseline,
                    )
                    .expect("the pipeline builds");

                match device.run(pipeline, &buffers, push, d.groups) {
                    Ok(()) => ran += 1,
                    Err(e) => refused.push(format!("{symbol}: {e}")),
                }
                if let Some((buf, _)) = block {
                    device.free(buf);
                }
            }

            device.free(arena_buffer);
        }
    }

    cache.clear(&device);
    device.free(weights);

    // Every dispatch the walk planned, the device took. Stated as "none
    // refused" AND as a floor, because a version of this loop that planned
    // nothing would refuse nothing too.
    assert!(
        refused.is_empty(),
        "{} of {} planned dispatches were refused by the device:\n  {}",
        refused.len(),
        ran as usize + refused.len(),
        refused.join("\n  ")
    );
    // Nineteen: every distinct symbol the three texts reach, in both fire
    // classes, submitted and accepted.
    //
    // Was thirteen, then fourteen. The last five arrived together when the
    // scalar run started being built from the kernel row rather than taken
    // whole, and when this loop started stating the model's geometry --
    // `sdpa_paged_decode` is compiled for a fixed head width and a plan does
    // not carry one.
    //
    // Also the first time `kv_append_paged` and the paged attentions reach a
    // device at all. They are the rows with descriptor holes, the rows that
    // interleave driver numbers among the statement's scalars, and the only
    // rows that put anything on the grid's third dimension.
    //
    // Equality rather than a floor: nineteen is all of them, so this number
    // moving in either direction is news.
    assert_eq!(
        ran, 19,
        "a different number of distinct symbols reached the device"
    );
}

/// A norm a real plan states computes what a host reference computes.
///
/// The test whose absence let the largest defect in this crate through. Every
/// other check here asks whether a dispatch is ACCEPTED; this one asks what it
/// wrote. For 2898 of this tree's 3992 rectangles the two questions had
/// different answers -- the descriptor set was well formed, every range was
/// inside its buffer, the layer was silent, and the operands were in the wrong
/// slots.
///
/// `rms_single_row` is the whole story in one row. `norm/rms.comp` decorates
/// `0=x, 1=w, 2=out`; the kernel row says `In(0), Weight(0), Out(0)`; and
/// `Launch::args` states inputs, then outputs, then weights. Bound
/// positionally, the shader reads the arena range the plan meant for its
/// OUTPUT as the weight, and writes its result over the weight buffer.
///
/// So the reference is computed from the ranges the ROW chose, and compared
/// against the range the row calls the output. Bind positionally and the
/// output range still holds what it was born with.
#[test]
fn a_norm_a_real_plan_states_computes_what_a_host_reference_computes() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, Row, lower};
    use model_compiler::trace::FireClass;

    let (device, dir) = gpu!();
    let symbol = "rms_single_row_bfloat16";

    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the text lowers");

    // Not zeros and not a ramp. Zeros make every wrong binding right, and
    // neighbouring elements of a ramp are near enough that reading one range
    // for another barely moves a sum. This is a per-BYTE pattern, so no two
    // arena offsets hold the same bf16 row.
    let mut fill: Vec<u8> = (0..low.arena_bytes).map(|i| (i * 31 % 251) as u8).collect();
    // bf16 whose exponent bytes are arbitrary can be an infinity or a NaN,
    // and one NaN anywhere in the row makes the whole reference a NaN and the
    // comparison vacuous. Clamping the high byte's exponent keeps every value
    // finite and small without making any two of them equal.
    for hi in fill.iter_mut().skip(1).step_by(2) {
        *hi = (*hi & 0x83) | 0x3c;
    }
    let arena_buffer = device.buffer(&fill).expect("the arena allocates");

    let arena = driver_vulkan::binding::Arena {
        buffer: &arena_buffer,
        bytes: low.arena_bytes as u64,
    };
    let launch = low
        .launches
        .iter()
        .find(|l| low.kernels[l.kernel as usize] == symbol)
        .expect("the text norms");

    // One buffer PER NAME, not one for all of them. A store that answered
    // every name with the same memory cannot tell a binder that resolved this
    // launch's weight from one that resolved any of the other 703 the plan
    // states, because both would read the same bytes and compute the same
    // answer.
    //
    // Every name gets different contents, so binding the wrong one is a wrong
    // number rather than a coincidence.
    let mut weights = driver_vulkan::resources::Weights::new();
    let mut wanted: Vec<String> = Vec::new();
    for a in &low.args[launch.args.start as usize..launch.args.end as usize] {
        if let model_compiler::lower::Arg::Weight(name) = a {
            wanted.push(name.clone());
        }
    }
    assert_eq!(wanted.len(), 1, "the norm states one weight");
    // Two names, so the store holds a decoy as well: an assertion that the
    // binder picked "a buffer in the store" passes on a store of one.
    let decoy = "a.weight.this.launch.does.not.name";
    for (i, name) in wanted.iter().map(String::as_str).chain([decoy]).enumerate() {
        let mut fill: Vec<u8> = (0..1usize << 16)
            .map(|b| ((b * 17 + i * 101) % 241) as u8)
            .collect();
        for hi in fill.iter_mut().skip(1).step_by(2) {
            *hi = (*hi & 0x83) | 0x3c;
        }
        weights.hold(&device, name, &fill).expect("a weight");
    }
    weights.seam(&device, 1 << 16).expect("the seam");

    let code = std::fs::read(dir.join(format!("{symbol}.spv"))).expect("the module is built");
    let words = driver_vulkan::spirv::words(&code).expect("whole words");
    let declared = driver_vulkan::spirv::declared(&words).expect("well formed");
    let d = driver_vulkan::dispatch::plan_one(
        &low,
        launch,
        kernels_vulkan::KERNELS,
        driver_vulkan::dispatch::Built {
            module: driver_vulkan::geometry::Module::named(
                symbol,
                [declared.local[0], declared.local[1], declared.local[2]],
            ),
            declared: &declared,
        },
        driver_vulkan::dispatch::Sources {
            arena,
            resolver: &weights,
            min_offset: device.min_storage_offset(),
        },
        driver_vulkan::dispatch::Geometry::default(),
    )
    .expect("the rectangle plans");

    // What the row chose, read back so the reference and the device see the
    // same bytes. Three buffers: x, w, out -- in the module's order, which is
    // the point.
    let (x, w, out) = (d.buffers[0], d.buffers[1], d.buffers[2]);

    // Which range each slot got, not just that the three agree with each
    // other. Without this the test is self-consistent and blind: it would
    // read slot 1 as the weight because `reorder` put it there, and a
    // `reorder` that swapped the output and the weight would still pass --
    // measured, that exact swap did pass before these three lines.
    //
    // The weight lives in the resolver's buffer and the other two in the
    // arena, which is enough to tell all three apart, and the output is the
    // LAST of the launch's widthed arguments because the trace states inputs
    // before outputs.
    assert!(
        std::ptr::eq(x.buffer(), &arena_buffer) && std::ptr::eq(out.buffer(), &arena_buffer),
        "the norm's input and output are the plan's, so both are ranges of the arena"
    );
    // By NAME. `std::ptr::eq(w.buffer(), <the one buffer>)` was the old form
    // and it only said "the weight came from the store"; with one buffer in
    // the store that is true of every name in the plan. This says the binder
    // resolved the name the launch states, and the decoy beside it in the
    // store means "some name in the store" is not enough to pass.
    assert!(
        std::ptr::eq(
            w.buffer(),
            weights.at(&wanted[0]).expect("the launch's own weight")
        ),
        "the norm's weight is not the buffer held under {}",
        wanted[0]
    );
    assert!(
        !std::ptr::eq(w.buffer(), weights.at(decoy).expect("the decoy")),
        "the norm resolved a name it does not state"
    );
    let widthed: Vec<&model_compiler::lower::Arg> = low.args
        [launch.args.start as usize..launch.args.end as usize]
        .iter()
        .filter(|a| !matches!(a, model_compiler::lower::Arg::Weight(_)))
        .collect();
    let model_compiler::lower::Arg::Arena { at, .. } = widthed[widthed.len() - 1] else {
        panic!("the norm writes into the arena");
    };
    assert_eq!(
        out.offset(),
        *at as u64,
        "slot 2 is the range the trace states last, which is the output"
    );
    let bytes = |b: Bound<'_>| {
        let whole = device.read(b.buffer()).expect("read back");
        whole[b.offset() as usize..(b.offset() + b.len()) as usize].to_vec()
    };
    let xq = bf16_read(&bytes(x));
    let wq = bf16_read(&bytes(w));

    let driver_vulkan::binding::Params::Block { bytes: block, .. } = &d.params else {
        panic!("this module reads its scalars from a buffer");
    };
    // The plan's own scalars, read back rather than invented: `eps` is the
    // first word and the axis the second, and a reference that assumed either
    // would be testing the assumption.
    let eps = f32::from_le_bytes(block[0..4].try_into().unwrap());
    let axis = u32::from_le_bytes(block[4..8].try_into().unwrap()) as usize;
    assert_eq!(xq.len(), axis, "the plan's input is one row of the axis");

    let params = device.buffer(block).expect("the block allocates");
    let mut buffers = d.buffers.clone();
    buffers.insert(d.block_at.expect("a block slot"), Bound::whole(&params));

    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(
            &device,
            symbol,
            &code,
            0,
            buffers.len() as u32,
            Capability::Baseline,
        )
        .expect("the pipeline builds");
    device
        .run(pipeline, &buffers, &[], d.groups)
        .expect("dispatch");

    let got = bf16_read(&bytes(out));
    let mean: f32 = xq.iter().map(|v| v * v).sum::<f32>() / axis as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    let mut moved = 0usize;
    for (i, (g, (v, gain))) in got.iter().zip(xq.iter().zip(&wq)).enumerate() {
        let want = gain * (v * inv);
        assert!(
            (g - want).abs() <= 8e-3 * want.abs().max(1.0),
            "element {i}: the device says {g}, the row's operands say {want}"
        );
        if *g != 0.0 {
            moved += 1;
        }
    }
    // A dispatch that wrote nowhere would agree with a reference of zeros.
    assert!(
        moved > axis / 2,
        "only {moved} of {axis} outputs are non-zero, so the comparison proves little"
    );

    cache.clear(&device);
    for b in [arena_buffer, params] {
        device.free(b);
    }
    weights.close(&device);
}

/// A chain of dispatches recorded once says what the same chain said one at a
/// time.
///
/// [`Device::run`] submits once per dispatch and waits on a fence, so every
/// dispatch is separated from the next by the strongest ordering Vulkan has.
/// A fire cannot afford that -- a real plan states 3992 rectangles -- so
/// `run_all` records them all into one command buffer, where Vulkan gives NO
/// ordering at all unless a barrier states it.
///
/// The chain is deliberate: each norm reads the row the previous one wrote, so
/// nothing here can be right by accident. The reference is the same chain run
/// through `Device::run`, which is the version already proven against a host
/// reference, rather than a second host implementation that could be wrong in
/// the same way.
#[test]
fn a_chain_recorded_once_says_what_the_chain_submitted_one_at_a_time_says() {
    let (device, dir) = gpu!();
    let entrypoint = "rms_single_row_bfloat16";
    let axis = 512usize;
    let links = 8usize;

    // Not a ramp: neighbouring elements of a ramp are near enough that a
    // dispatch reading a stale row would produce nearly the right answer.
    let x: Vec<f32> = (0..axis)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) / 12.0)
        .collect();
    let w: Vec<f32> = (0..axis).map(|i| 0.75 + (i % 11) as f32 / 16.0).collect();
    let mut params = Vec::new();
    params.extend_from_slice(&1e-5f32.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());

    let code = module(dir, entrypoint);
    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");
    let groups = groups_for(
        entrypoint,
        Rule::Rms,
        Dims {
            rows: 1,
            width: axis as u32,
            axis: axis as u32,
            ..Dims::default()
        },
        pipeline,
    )
    .expect("a geometry");

    // `links + 1` rows in one buffer: link `i` reads row `i` and writes row
    // `i + 1`, which is the chaining a plan's arena does.
    let stride = (axis * 2) as u64;
    let mut initial = bf16_bytes(&x);
    initial.resize((links + 1) * axis * 2, 0);
    let wb = device.buffer(&bf16_bytes(&w)).expect("w");
    let pb = device.buffer(&params).expect("params");

    let run_chain = |chained: bool| -> Vec<f32> {
        let rows = device.buffer(&initial).expect("rows");
        let sets: Vec<Vec<Bound<'_>>> = (0..links)
            .map(|i| {
                vec![
                    Bound::at(&device, &rows, i as u64 * stride, stride).expect("in"),
                    Bound::whole(&wb),
                    Bound::at(&device, &rows, (i as u64 + 1) * stride, stride).expect("out"),
                    Bound::whole(&pb),
                ]
            })
            .collect();
        if chained {
            let run: Vec<driver_vulkan::device::Recorded<'_, '_>> = sets
                .iter()
                .map(|b| driver_vulkan::device::Recorded {
                    pipeline,
                    buffers: b,
                    push: &[],
                    groups,
                })
                .collect();
            device.run_all(&run).expect("the chain records and submits");
        } else {
            for b in &sets {
                device.run(pipeline, b, &[], groups).expect("dispatch");
            }
        }
        let out = device.read(&rows).expect("read back");
        device.free(rows);
        bf16_read(&out[links * axis * 2..])
    };

    let one_at_a_time = run_chain(false);
    let recorded = run_chain(true);

    // Bit for bit. The same modules over the same bytes in the same order,
    // so anything but equality is an ordering the recording did not state --
    // a tolerance here would be a place for that to hide.
    assert_eq!(
        recorded, one_at_a_time,
        "the recorded chain and the submitted chain disagree"
    );
    // And the chain went somewhere. Eight norms of a row that started as
    // zeros would also agree, and would prove nothing.
    assert!(
        recorded.iter().filter(|v| **v != 0.0).count() > axis / 2,
        "the last row is mostly zeros, so the comparison proves little"
    );

    cache.clear(&device);
    device.free(wb);
    device.free(pb);
}

/// Every rectangle one real plan states, recorded into one command buffer and
/// submitted once.
///
/// The other plan-driven test here takes the FIRST rectangle of each distinct
/// symbol, which is coverage of the dispatch paths and says nothing about a
/// whole fire. This one takes them all, in the order the plan states them,
/// barriers between, one submit.
///
/// A decode of the smallest text rather than all six lowerings: the point is
/// that the arithmetic holds over a whole plan rather than over a sample, and
/// a prefill of a 30B mixture allocates an arena this would rather not hold
/// alongside a stand-in for every weight.
///
/// What it proves is narrow and worth having: every pipeline builds, every
/// descriptor set is accepted, no dispatch is refused, and the whole run
/// completes inside one fence wait. What it does NOT prove is the numbers --
/// the weights are a stand-in, so the arithmetic is meaningless and only the
/// plumbing is under test. `a_norm_a_real_plan_states_computes_what_a_host_
/// reference_computes` is the one that reads a result.
///
/// The weights are one buffer for all 704 names the plan states, so this
/// cannot tell a binder that resolved `layer.3.q_proj` from one that resolved
/// anything else. Sized rather than fixed only because `Arg::Weight` carries
/// a name and no width, so a store here would be guessing. The norm test
/// holds one buffer per name and checks that identity where a reference makes
/// it mean something.
#[test]
fn a_whole_real_plan_records_into_one_command_buffer_and_submits() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};

    let (device, dir) = gpu!();

    // SIX texts and not one. This test was written against qwen3-0.6B because
    // that was the text to hand, and every claim below -- that a whole plan
    // records, that the batched run agrees with the one-at-a-time run, that a
    // withheld weight is refused by NAME -- was being made about one model's
    // shape. Six run here now: two dense qwen, olmo2 with its own norm
    // placement, mistral, and the two mixtures of experts, whose router and
    // gather rows nothing else on this card reaches.
    //
    // They are affordable: the largest arena among them is four mebibytes,
    // measured, because a lowering sizes activations and not weights.
    let mut fired = 0;
    for (name, facts, metal) in [
        (
            "qwen3_0_6b",
            LlamaLikeFacts::qwen3_0_6b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            "qwen2_5_1_5b",
            LlamaLikeFacts::qwen2_5_1_5b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            "olmo2_1b",
            LlamaLikeFacts::olmo2_1b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            "mistral_7b_v03",
            LlamaLikeFacts::mistral_7b_v03(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            "gpt_oss_20b",
            LlamaLikeFacts::gpt_oss_20b(),
            LlamaLikeMetalFacts::gpt_oss_20b(),
        ),
        (
            "qwen3_30b_a3b",
            LlamaLikeFacts::qwen3_30b_a3b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
    ] {
        fired += whole_plan(&device, dir, name, &facts, &metal);
    }
    // Pinned, because a loop is not evidence that a loop ran: every text here
    // could stop lowering and each `whole_plan` would still be a pass over
    // whatever remained. This is the sum of the six decodes' launches, every
    // one of them recorded and submitted on this card, and it moves when a
    // text does.
    assert_eq!(
        fired, 3136,
        "the six texts fired a different number of rectangles"
    );
}

/// One text's whole decode, fired twice and compared against itself.
///
/// Answers how many rectangles it fired, so the caller can pin a total no
/// individual text's pass would notice going missing.
fn whole_plan(
    device: &Device,
    dir: &std::path::Path,
    name: &str,
    facts: &model::shared::llama_like::forward::facts::LlamaLikeFacts,
    metal: &model::shared::llama_like::forward::facts::LlamaLikeMetalFacts,
) -> usize {
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, Row, lower};
    use model_compiler::trace::FireClass;

    let plan = llama_like_metal(facts, metal, FireClass::Decode);
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the text lowers");

    // Not zeros. Every dispatch in this plan reads the arena, and an arena of
    // zeros makes a plan that ordered itself wrongly agree with one that did
    // not: zero times anything is still zero, whichever order it happens in.
    // Clamped exponents so nothing is an infinity or a NaN, because one NaN
    // reaches every later rectangle and makes the whole comparison vacuous.
    let mut fill: Vec<u8> = (0..low.arena_bytes).map(|i| (i * 31 % 251) as u8).collect();
    for hi in fill.iter_mut().skip(1).step_by(2) {
        *hi = (*hi & 0x83) | 0x3c;
    }
    let arena_buffer = device.buffer(&fill).expect("the arena allocates");

    // A real pool rather than one buffer standing in for everything. The
    // cache is per-layer, so a plan that asked layer 27 for layer 3's keys
    // would get a different buffer here and the same one from a stand-in --
    // and the tables have real lengths, so a dispatch that indexed past the
    // end of one would be a range this device could report rather than an
    // offset inside a buffer big enough to hide it.
    let shape = driver_vulkan::resources::Shape {
        layers: facts.layers as u16,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 4,
        bytes: 2,
    };
    let mut store = driver_vulkan::resources::Pool::open(device, shape).expect("the pool opens");
    store.stand_in(device, 1 << 22).expect("a stand-in");
    // One buffer per NAME, and the plan states 704 of them. Until `Model`
    // existed no resolver in this crate could answer this fire at all: `Pool`
    // answers the cache and the tables and hands every weight the same
    // stand-in, and `Weights` answers names and knows nothing about a cache.
    // This test fired against the stand-in and said so, which meant the whole
    // plan proved its plumbing against a resolver deliberately wrong about
    // half of what a plan states.
    //
    // The sizes are a GUESS, and that is the recorded blocker rather than an
    // oversight: `Arg::Weight` carries a name and no width, so nothing in the
    // plan says how big a tensor is. Four mebibytes covers every projection in
    // this model; it does not cover the embedding table, which is 151936 rows
    // of 1024 -- 311 MiB -- and the only reason that is safe here is that
    // `TokenIds` is all zeros, so the gather reads row zero. This is why a
    // whole plan exercises plumbing rather than computing anything, and it
    // stays that way until a checkpoint loader supplies real sizes.
    let mut weights = driver_vulkan::resources::Weights::new();
    let names: std::collections::BTreeSet<&str> = low
        .args
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Weight(n) => Some(n.as_str()),
            _ => None,
        })
        .collect();
    // 200 and not 500: the floor was set to one text's 704 and olmo2's
    // decode states fewer, which is a difference in DEPTH and not a sign
    // that a plan failed to lower. What the floor is for is a plan that
    // collapsed to a handful of names.
    assert!(
        names.len() > 200,
        "{name}: only {} weight names, so this is not a whole model",
        names.len()
    );
    for n in &names {
        weights
            .hold(device, n, &vec![0u8; 1 << 22])
            .expect("a weight");
    }
    weights.seam(device, 1 << 22).expect("the seam");
    // The fire's tables from `Frame::of` rather than four thousand zeros.
    // The lengths are the fire's own and the page is 3, so the plan runs
    // against a cache arranged the way a server would arrange one rather than
    // against a table of zeros that makes every index correct.
    //
    // It is worth saying what this does NOT buy. A table one entry short was
    // tried here and the validation layer said nothing, GPU-AV included: an
    // overrun inside a bound storage buffer is not a thing Vulkan reports.
    // That is the measured reason `Frame::of` refuses `PastItsPages` itself.
    // There is no later stage that would have.
    let frame = driver_vulkan::resources::Frame::of(
        shape,
        &[driver_vulkan::resources::Request {
            positions: vec![5],
            pages: vec![3, 1],
            samples: Vec::new(),
        }],
    )
    .expect("the fire stages");
    store.stage(device, &frame).expect("the fire's tables");
    // The one a `Frame` does not derive, because it is not a function of the
    // paging: what the rows say. One entry, for the one row this decode has.
    store
        .state(device, driver_vulkan::binding::FireTable::TokenIds, &[0u32])
        .expect("a table");
    // A real ladder rather than the table of zeros this staged before. It
    // changes nothing HERE and is not left in as a decoration: the only two
    // rows in the table that read `RopeFrequencies` are `neox_freqs_mb` and
    // `neox_freqs_decode`, and none of the three texts this crate walks
    // launches either, because only a rescaling deployment needs them. Zeros
    // were still wrong to stage -- an angle of zero is the identity, so a plan
    // that started reading this table would keep passing -- and they were the
    // wrong LENGTH as well, `head_dim` where a ladder is `head_dim / 2`.
    store
        .ladder(device, facts.head_dim, 1_000_000.0, None)
        .expect("the ladder");
    let arena = driver_vulkan::binding::Arena {
        buffer: &arena_buffer,
        bytes: low.arena_bytes as u64,
    };
    let geometry = driver_vulkan::dispatch::Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
    };

    // Through `serve::fire`, which is the call. This test used to assemble
    // the three passes itself -- plan every rectangle, allocate every scalar
    // block, build every pipeline, then record -- and that was the wrong place
    // for it: a test that assembles a fire is testing its own assembly, and
    // the ordering between those passes is not a matter of taste. A block
    // freed while the command buffer still names it is a use-after-free; a
    // pipeline reference taken before the next pipeline is built does not
    // compile. `src/serve.rs` is that shape, with the reasons written down.
    let modules: std::collections::BTreeMap<String, Vec<u8>> = low
        .kernels
        .iter()
        .map(|symbol| {
            let code =
                std::fs::read(dir.join(format!("{symbol}.spv"))).expect("the module is built");
            (symbol.clone(), code)
        })
        .collect();
    // Stated so that a plan that stopped lowering cannot pass by firing
    // nothing.
    assert!(
        low.launches.len() > 250,
        "{name}: only {} launches, so this is not a plan",
        low.launches.len()
    );
    // Nine, measured, for this decode -- the three texts together reach
    // nineteen. Five is the floor rather than nine because the point is that
    // several DIFFERENT modules run, so the pipeline cache is exercised and a
    // fire is not one kernel repeated; pinning the exact number would make
    // this test fail when the model's shape changes, which is not what it is
    // about.
    assert!(
        modules.len() >= 5,
        "{name}: only {} distinct modules, so this is not a plan",
        modules.len()
    );

    let model = driver_vulkan::resources::Model {
        weights: &weights,
        pool: &store,
    };
    let mut cache = Pipelines::new();
    let what = driver_vulkan::serve::Fire {
        arena,
        resolver: &model,
        geometry,
        tier: Capability::Baseline,
        one_at_a_time: false,
    };
    let fired = driver_vulkan::serve::fire(device, &mut cache, &modules, &low, what)
        .unwrap_or_else(|e| panic!("{e}"));
    // Every launch the plan states, in one command buffer. Without this a
    // `fire` that dropped the last rectangle passes, because the comparison
    // below is against the same `fire` and both runs lose the same thing.
    assert_eq!(
        fired,
        driver_vulkan::serve::Fired {
            dispatches: low.launches.len(),
            submissions: 1
        }
    );
    let recorded = device.read(&arena_buffer).expect("the arena reads back");

    // And now the same plan the slow way. `Device::run` submits once per
    // dispatch and waits on a fence, which is the strongest ordering Vulkan
    // has, so this is what the plan MEANS. `run_all` puts them all in one
    // command buffer with a memory barrier between each pair, which is what
    // the plan COSTS.
    //
    // The eight-norm chain already showed that dropping those barriers gives
    // wrong answers on this card while every call returns success and the
    // layer stays silent. This says the same thing over a real plan: five
    // hundred rectangles, twenty distinct kernels, an arena every one of them
    // reads and writes, and a dependency graph nobody wrote down. Removing
    // the barrier fails on every run, at a different byte each time.
    //
    // Worth recording what does NOT fail: setting both access masks to empty
    // and keeping the barrier passed three runs. A `COMPUTE -> COMPUTE`
    // pipeline barrier is an EXECUTION dependency whatever its masks say, and
    // on this card that is the part that matters -- its L2 is coherent, so
    // visibility follows ordering. The masks stay because the specification
    // requires them and a card whose caches are not coherent would need them,
    // but a control that only weakens them is not a control here.
    device
        .write(&arena_buffer, &fill)
        .expect("the arena resets");
    let slow = driver_vulkan::serve::fire(
        device,
        &mut cache,
        &modules,
        &low,
        driver_vulkan::serve::Fire {
            one_at_a_time: true,
            ..what
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    // One submission per dispatch, which is what makes this the reference.
    // Without it a `fire` that ignored the flag would record the same command
    // buffer twice and the comparison below would be against itself.
    assert_eq!(
        slow,
        driver_vulkan::serve::Fired {
            dispatches: low.launches.len(),
            submissions: low.launches.len()
        }
    );
    let one_at_a_time = device.read(&arena_buffer).expect("the arena reads back");

    assert_eq!(
        recorded.len(),
        one_at_a_time.len(),
        "{name}: the same arena both times"
    );
    let differ = recorded
        .iter()
        .zip(&one_at_a_time)
        .position(|(a, b)| a != b);
    assert!(
        differ.is_none(),
        "{name}: the recorded plan and the submitted plan disagree at byte {:?} of the arena",
        differ
    );
    // The plan moved the arena. Two runs that both did nothing agree, and a
    // comparison that would accept that measures the read-back and not the
    // plan.
    let moved = recorded.iter().zip(&fill).filter(|(a, b)| a != b).count();
    assert!(
        moved > low.arena_bytes / 100,
        "{name}: only {moved} of {} arena bytes changed, so the plan barely ran",
        low.arena_bytes
    );

    // Every name is resolved by NAME, not by "some buffer in the store". Drop
    // one of the 704 and the fire must refuse rather than bind a neighbour --
    // and the name dropped is one this plan actually states, so a refusal is
    // about resolution and not about a typo.
    //
    // This is also the control on `Model` itself, and it was measured rather
    // than reasoned: fired against the `Pool` alone -- which is what this test
    // did before -- a plan with `embed` withheld returns `Ok(Fired {
    // dispatches: 452, submissions: 1 })`. All 452 rectangles run, because the
    // pool answers every one of the 704 names with the same stand-in.
    let missing = *names.iter().next().expect("the plan states weights");
    let mut holed = driver_vulkan::resources::Weights::new();
    for n in names.iter().filter(|n| **n != missing) {
        holed
            .hold(device, n, &vec![0u8; 1 << 22])
            .expect("a weight");
    }
    holed.seam(device, 1 << 22).expect("the seam");
    let refused = driver_vulkan::serve::fire(
        device,
        &mut cache,
        &modules,
        &low,
        driver_vulkan::serve::Fire {
            resolver: &driver_vulkan::resources::Model {
                weights: &holed,
                pool: &store,
            },
            ..what
        },
    );
    match refused {
        // By the NAME in the message and not by the variant: every one of the
        // 704 would produce an `Unplannable`, so matching the variant alone
        // would pass on a fire that refused for a different name.
        Err(driver_vulkan::serve::Unfired::Unplannable { why, .. }) => {
            let said = why.to_string();
            let want =
                driver_vulkan::binding::Unbindable::UnknownWeight(missing.to_owned()).to_string();
            assert!(
                said.ends_with(&want),
                "{name}: the fire refused with `{said}`, not for the withheld `{missing}`"
            );
        }
        other => panic!("{name}: a plan missing `{missing}` fired anyway: {other:?}"),
    }
    holed.close(device);

    cache.clear(device);
    device.free(arena_buffer);
    weights.close(device);
    store.close(device);
    low.launches.len()
}

/// A real plan's KV append puts the row where the page table says.
///
/// `kv_append_paged` is the hardest row in the tree and the last one to reach
/// a device. It has six descriptor holes, it interleaves a driver-resolved
/// page size between the statement's two scalars, and both of its destinations
/// are memory no plan mentions. Every earlier test of it asked whether the
/// dispatch was ACCEPTED; nothing had ever read the cache afterwards.
///
/// So this one writes a known row through a real rectangle of a real plan and
/// then looks for it. The page table is deliberately not the identity -- page
/// 3, offset 5 -- because a scatter that ignored the tables entirely would
/// land at slot 0 and an identity table would call that correct.
///
/// The destination arithmetic comes from [`Shape::slot`], which transcribes
/// `attn/kv_write.comp`. `attn/sdpa_paged.comp` computes the same expression
/// from separate source, so the layout is two modules agreeing rather than
/// this crate deciding.
#[test]
fn a_real_plans_kv_append_puts_the_row_where_the_page_table_says() {
    use driver_vulkan::binding::FireTable;
    use driver_vulkan::resources::{Pool, Shape};
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, Row, lower};
    use model_compiler::trace::FireClass;

    let (device, dir) = gpu!();
    let symbol = "kv_append_paged_bfloat16";

    let facts = LlamaLikeFacts::qwen3_0_6b();
    let plan = llama_like_metal(&facts, &LlamaLikeMetalFacts::synthetic(), FireClass::Decode);
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the text lowers");

    let shape = Shape {
        layers: facts.layers as u16,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        // Not a power of two and not the row count: a page size that divided
        // every other number here would let an arithmetic slip cancel.
        page_size: 6,
        pages: 8,
        bytes: 2,
    };
    let mut pool = Pool::open(&device, shape).expect("the cache allocates");
    pool.stand_in(&device, 1 << 22).expect("a stand-in");
    // One row, going to page 3 offset 5 -- neither zero, so a scatter that
    // read neither table would land somewhere this test can tell apart.
    let (page, offset) = (3u32, 5u32);
    pool.state(&device, FireTable::KvWritePage, &[page])
        .expect("the write page");
    pool.state(&device, FireTable::KvWriteOffset, &[offset])
        .expect("the write offset");

    // The arena holds the k and v the append reads, so it is filled with a
    // pattern rather than zeros -- a cache that was never written also holds
    // zeros, and this test would not be able to tell the two apart.
    let mut fill: Vec<u8> = (0..low.arena_bytes).map(|i| (i * 41 % 253) as u8).collect();
    for hi in fill.iter_mut().skip(1).step_by(2) {
        *hi = (*hi & 0x83) | 0x3c;
    }
    let arena_buffer = device.buffer(&fill).expect("the arena allocates");
    let arena = driver_vulkan::binding::Arena {
        buffer: &arena_buffer,
        bytes: low.arena_bytes as u64,
    };

    let launch = low
        .launches
        .iter()
        .find(|l| low.kernels[l.kernel as usize] == symbol)
        .expect("the text appends to a paged cache");
    let layer = launch.layers.start;

    let code = module(dir, symbol);
    let words = driver_vulkan::spirv::words(&code).expect("whole words");
    let declared = driver_vulkan::spirv::declared(&words).expect("well formed");
    let d = driver_vulkan::dispatch::plan_one(
        &low,
        launch,
        kernels_vulkan::KERNELS,
        driver_vulkan::dispatch::Built {
            module: driver_vulkan::geometry::Module::named(
                symbol,
                [declared.local[0], declared.local[1], declared.local[2]],
            ),
            declared: &declared,
        },
        driver_vulkan::dispatch::Sources {
            arena,
            resolver: &pool,
            min_offset: device.min_storage_offset(),
        },
        driver_vulkan::dispatch::Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
    )
    .expect("the rectangle plans");

    // The row's scalar run, read back rather than assumed: the shader's push
    // block is `head_dim, page_size, n_kv_heads` and the row is `Param(0),
    // KvPageSize, Param(1)`. The middle word is the one no statement carries,
    // so finding the pool's page size there is what says `scalars` placed a
    // driver number where the row asked for it.
    let driver_vulkan::binding::Params::Push(push) = &d.params else {
        panic!("this module pushes its scalars");
    };
    assert_eq!(push.len(), 12, "three words");
    let word = |i: usize| u32::from_le_bytes(push[i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!(word(0), shape.head_dim, "the first word is the head width");
    assert_eq!(
        word(1),
        shape.page_size,
        "the second word is the pool's page size, which no statement carries"
    );
    assert_eq!(word(2), shape.kv_heads, "the third word is the head count");

    // What the append is about to read, so the reference is the device's own
    // bytes rather than the pattern this test wrote.
    let k_src = d.buffers[0];
    let src = {
        let whole = device.read(k_src.buffer()).expect("read back");
        whole[k_src.offset() as usize..(k_src.offset() + k_src.len()) as usize].to_vec()
    };

    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(
            &device,
            symbol,
            &code,
            push.len() as u32,
            d.buffers.len() as u32,
            Capability::Baseline,
        )
        .expect("the pipeline builds");
    device
        .run(pipeline, &d.buffers, push, d.groups)
        .expect("dispatch");

    let keys = device
        .read(pool.cache(layer, false).expect("a key cache"))
        .expect("read the cache");
    let values = device
        .read(pool.cache(layer, true).expect("a value cache"))
        .expect("read the cache");

    // Every head of the one row, at the slot the tables named.
    let mut checked = 0usize;
    for h in 0..shape.kv_heads {
        for at in 0..shape.head_dim {
            let to = shape.slot(page, offset, h, at) as usize * 2;
            let from = (h * shape.head_dim + at) as usize * 2;
            assert_eq!(
                &keys[to..to + 2],
                &src[from..from + 2],
                "key head {h} element {at} is not at page {page} offset {offset}"
            );
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        (shape.kv_heads * shape.head_dim) as usize,
        "the whole row is checked"
    );
    // The pattern is not zeros, so a cache that was never written cannot pass
    // the comparison above -- but say so, because a `src` that came back
    // empty would make it vacuous.
    assert!(
        src.iter().filter(|b| **b != 0).count() > src.len() / 4,
        "the source row is mostly zeros, so the comparison proves little"
    );
    // And nothing landed anywhere else. A scatter that wrote every slot, or
    // wrote slot 0 as well, would satisfy every assertion above.
    let written = keys.chunks_exact(2).filter(|c| c != &[0u8, 0]).count();
    assert!(
        written <= (shape.kv_heads * shape.head_dim) as usize,
        "{written} elements of the key cache are non-zero, and one row is {}",
        shape.kv_heads * shape.head_dim
    );
    assert_eq!(
        values.len(),
        keys.len(),
        "the value cache is the same size as the key cache"
    );

    cache.clear(&device);
    device.free(arena_buffer);
    pool.close(&device);
}

/// The append writes a cache the paged attention can read.
///
/// `resources` transcribes one slot expression and says it is a fact because
/// two shaders compute it. This is the test that makes that a measurement:
/// `attn/kv_write.comp` puts six positions into a pool through the page table,
/// and `attn/sdpa_paged.comp` attends over them without either shader or this
/// file ever agreeing on anything but the pool.
///
/// The reference never mentions a slot. It is stated entirely in terms of "the
/// row written at position p", so the only thing carrying the layout between
/// the two halves is the cache itself. If the write and the read disagreed by
/// so much as a head, the attention would be over rows nobody wrote and the
/// comparison would say so.
///
/// The page table is `[3, 1]`, so position 0 lands in page 3 and position 4 in
/// page 1. Descending and not the identity: a read that ignored the table
/// would find page 0 and page 1 in order, which is the arrangement most likely
/// to look correct.
#[test]
fn what_the_append_writes_through_the_page_table_is_what_the_attention_reads() {
    use driver_vulkan::binding::FireTable;
    use driver_vulkan::resources::{Pool, Shape};

    let (device, dir) = gpu!();
    let head_dim = 128usize;
    let shape = Shape {
        layers: 1,
        kv_heads: 1,
        head_dim: head_dim as u32,
        page_size: 4,
        pages: 8,
        bytes: 2,
    };
    // Two pages, out of order, and neither of them page 0.
    let pages = [3u32, 1u32];
    let positions = 6usize;

    let row = |p: usize, salt: usize| -> Vec<f32> {
        (0..head_dim)
            .map(|d| (((p * 7 + d * 13 + salt * 29) % 61) as f32 - 30.0) / 24.0)
            .collect()
    };
    let ks: Vec<Vec<f32>> = (0..positions).map(|p| row(p, 0)).collect();
    let vs: Vec<Vec<f32>> = (0..positions).map(|p| row(p, 1)).collect();
    let q: Vec<f32> = (0..head_dim)
        .map(|d| ((d * 19 % 47) as f32 - 23.0) / 20.0)
        .collect();

    let mut pool = Pool::open(&device, shape).expect("the pool opens");
    let mut cache = Pipelines::new();

    // The write half, one position at a time, because a decode appends one
    // row per fire and the tables it reads are one entry long.
    let append = module(dir, "kv_append_paged_bfloat16");
    let mut push = Vec::new();
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    push.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    push.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());
    for p in 0..positions {
        pool.state(
            &device,
            FireTable::KvWritePage,
            &[pages[p / shape.page_size as usize]],
        )
        .expect("the write page");
        pool.state(
            &device,
            FireTable::KvWriteOffset,
            &[p as u32 % shape.page_size],
        )
        .expect("the write offset");
        let kn = device.buffer(&bf16_bytes(&ks[p])).expect("k_new");
        let vn = device.buffer(&bf16_bytes(&vs[p])).expect("v_new");
        {
            use driver_vulkan::binding::Resolve;
            let bound = [
                Bound::whole(&kn),
                Bound::whole(&vn),
                Bound::whole(pool.kv(0, false).expect("keys")),
                Bound::whole(pool.kv(0, true).expect("values")),
                Bound::whole(pool.table(FireTable::KvWritePage).expect("page")),
                Bound::whole(pool.table(FireTable::KvWriteOffset).expect("offset")),
            ];
            let pipeline = cache
                .get(
                    &device,
                    "kv_append_paged_bfloat16",
                    &append,
                    push.len() as u32,
                    bound.len() as u32,
                    Capability::Baseline,
                )
                .expect("the append builds");
            // One workgroup of 256 covers a 128-wide head; one per head; one
            // per row appended.
            device
                .run(pipeline, &bound, &push, [1, shape.kv_heads, 1])
                .expect("the append dispatches");
        }
        device.free(kn);
        device.free(vn);
    }

    // The read half. The attention is asked for the last position, so it
    // walks every row the appends wrote.
    let q_pos = positions - 1;
    pool.state(&device, FireTable::Positions, &[q_pos as u32])
        .expect("positions");
    pool.state(&device, FireTable::RequestOfToken, &[0])
        .expect("request of token");
    pool.state(&device, FireTable::KvPageIndices, &pages)
        .expect("page indices");
    pool.state(&device, FireTable::KvPageIndptr, &[0, pages.len() as u32])
        .expect("page indptr");
    // `uint8_t` tables, and one zero word is four zero bytes. Masking off,
    // because a mask is a second thing to get wrong and this test is about
    // the pages.
    pool.state(&device, FireTable::AttentionMask, &[0])
        .expect("mask");
    pool.state(&device, FireTable::AttentionMaskEnabled, &[0])
        .expect("mask enabled");

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut spush = Vec::new();
    spush.extend_from_slice(&1i32.to_le_bytes()); // gqa_factor
    spush.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    spush.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());
    spush.extend_from_slice(&scale.to_le_bytes());
    spush.extend_from_slice(&0u32.to_le_bytes()); // mask stride
    spush.extend_from_slice(&0i32.to_le_bytes()); // window: no limit

    let qb = device.buffer(&bf16_bytes(&q)).expect("queries");
    let ob = device.buffer(&vec![0u8; head_dim * 2]).expect("out");
    let symbol = "sdpa_paged_decode_bfloat16_d_128";
    let code = module(dir, symbol);
    {
        use driver_vulkan::binding::Resolve;
        let bound = [
            Bound::whole(&qb),
            Bound::whole(pool.kv(0, false).expect("keys")),
            Bound::whole(pool.kv(0, true).expect("values")),
            Bound::whole(&ob),
            Bound::whole(pool.table(FireTable::Positions).expect("pos")),
            Bound::whole(pool.table(FireTable::RequestOfToken).expect("req")),
            Bound::whole(pool.table(FireTable::KvPageIndices).expect("ix")),
            Bound::whole(pool.table(FireTable::KvPageIndptr).expect("ptr")),
            Bound::whole(pool.table(FireTable::AttentionMask).expect("mask")),
            Bound::whole(
                pool.table(FireTable::AttentionMaskEnabled)
                    .expect("enabled"),
            ),
        ];
        let pipeline = cache
            .get(
                &device,
                symbol,
                &code,
                spush.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the attention builds");
        // One workgroup per query head, one row, and the head width is the
        // local size.
        device
            .run(pipeline, &bound, &spush, [1, 1, 1])
            .expect("the attention dispatches");
    }

    // The reference, in positions rather than slots. Computed from the bf16
    // the device was given, so its own rounding is not folded into the
    // tolerance.
    let qq = bf16_read(&bf16_bytes(&q));
    let kq: Vec<Vec<f32>> = ks.iter().map(|k| bf16_read(&bf16_bytes(k))).collect();
    let vq: Vec<Vec<f32>> = vs.iter().map(|v| bf16_read(&bf16_bytes(v))).collect();
    let scores: Vec<f32> = (0..=q_pos)
        .map(|p| (0..head_dim).map(|d| scale * qq[d] * kq[p][d]).sum::<f32>())
        .collect();
    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f32 = exps.iter().sum();
    let want: Vec<f32> = (0..head_dim)
        .map(|d| (0..=q_pos).map(|p| exps[p] * vq[p][d]).sum::<f32>() / total)
        .collect();

    let got = bf16_read(&device.read(&ob).expect("read back"));
    for (d, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-2 * w.abs().max(1.0),
            "element {d}: the attention says {g}, six appended rows say {w}"
        );
    }
    // The attention attended to more than one row. A softmax that collapsed
    // onto a single position would agree with the reference wherever that
    // position dominates, and the values are close enough in scale that it
    // could. Stated as a spread over the positions the reference used.
    let spread = exps.iter().copied().fold(0.0f32, f32::max) / total;
    assert!(
        spread < 0.9,
        "one position carries {spread} of the softmax, so the sum over pages proves little"
    );

    cache.clear(&device);
    device.free(qb);
    device.free(ob);
    pool.close(&device);
}

/// The two appends put a row in the same place, and the pool is what says
/// where.
///
/// `attn/kv_write.comp` compiles to two shaders from one file. The paged one
/// computes `slot * (kv_heads * head_dim) + h * head_dim + d`. The contiguous
/// one computes `h * k_head_stride + pos * k_seq_stride + d` and takes both
/// strides from the driver. Those two expressions describe the same memory
/// only if the driver hands over `head_dim` and `kv_heads * head_dim`, in that
/// order -- which is what `resources::Pool` says, and what nothing had ever
/// checked. Both numbers were stated from reading the source and exercised by
/// no dispatch.
///
/// So both shaders append the same six rows to two pools of the same shape,
/// the paged one through the identity page table, and the caches are compared
/// byte for byte. There is no tolerance in this test: it is one scatter
/// against another, and a stride wrong by one element moves a whole row.
///
/// `kv_heads` is 2 deliberately. With one head the head stride is multiplied
/// by zero and any value for it passes.
///
/// The push block is the second thing under test. `int head_dim;
/// PIE_STRIDE k_head_stride; PIE_STRIDE k_seq_stride;` is 24 bytes, not 20:
/// `uvec2` aligns to 8, so there are four bytes of padding after the first
/// field. A driver that packs by concatenation writes both strides four bytes
/// low, the shader reads halves of two different numbers, and Vulkan reports
/// nothing because Vulkan does not know what the bytes meant. The offsets here
/// come from `kernels_vulkan::push_layout`, and the module's own SPIR-V
/// decorations say `[0, 8, 16]` independently.
#[test]
fn the_two_appends_put_a_row_in_the_same_place_and_the_pool_says_where() {
    use driver_vulkan::binding::{FireNumber, FireTable, Resolve};
    use driver_vulkan::resources::{Pool, Shape};

    let (device, dir) = gpu!();
    let head_dim = 128usize;
    let shape = Shape {
        layers: 1,
        // Two, so that a wrong head stride has somewhere wrong to go.
        kv_heads: 2,
        head_dim: head_dim as u32,
        page_size: 4,
        pages: 8,
        bytes: 2,
    };
    let heads = shape.kv_heads as usize;
    let positions = 6usize;

    // One row per head per position, all distinct, so that a swap of any two
    // of them shows up.
    let rows = |p: usize, salt: usize| -> Vec<f32> {
        (0..heads * head_dim)
            .map(|i| (((p * 7 + i * 13 + salt * 29) % 61) as f32 - 30.0) / 24.0)
            .collect()
    };
    let ks: Vec<Vec<f32>> = (0..positions).map(|p| rows(p, 0)).collect();
    let vs: Vec<Vec<f32>> = (0..positions).map(|p| rows(p, 1)).collect();

    let mut cache = Pipelines::new();
    let flat = |device: &Device, pool: &Pool| -> (Vec<u8>, Vec<u8>) {
        (
            device.read(pool.kv(0, false).expect("keys")).expect("k"),
            device.read(pool.kv(0, true).expect("values")).expect("v"),
        )
    };

    // The contiguous half. The strides are asked of the pool rather than
    // written here, so this measures `resources` and not the test's own idea
    // of the layout.
    let mut straight = Pool::open(&device, shape).expect("the straight pool");
    let head_stride = straight
        .number(FireNumber::KvHeadStride)
        .expect("a head stride");
    let seq_stride = straight
        .number(FireNumber::KvSeqStride)
        .expect("a sequence stride");
    let plain = module(dir, "kv_append_bfloat16");
    // Four bytes of padding at offset 4, then two 64-bit strides of which the
    // shader reads the low half.
    let mut push = vec![0u8; 24];
    push[0..4].copy_from_slice(&(head_dim as i32).to_le_bytes());
    push[8..12].copy_from_slice(&head_stride.to_le_bytes());
    push[16..20].copy_from_slice(&seq_stride.to_le_bytes());
    for p in 0..positions {
        straight
            .state(&device, FireTable::Positions, &[p as u32])
            .expect("the position");
        let kn = device.buffer(&bf16_bytes(&ks[p])).expect("k_new");
        let vn = device.buffer(&bf16_bytes(&vs[p])).expect("v_new");
        let bound = [
            Bound::whole(&kn),
            Bound::whole(&vn),
            Bound::whole(straight.kv(0, false).expect("keys")),
            Bound::whole(straight.kv(0, true).expect("values")),
            Bound::whole(straight.table(FireTable::Positions).expect("pos")),
        ];
        let pipeline = cache
            .get(
                &device,
                "kv_append_bfloat16",
                &plain,
                push.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the contiguous append builds");
        device
            .run(pipeline, &bound, &push, [1, shape.kv_heads, 1])
            .expect("the contiguous append dispatches");
        device.free(kn);
        device.free(vn);
    }

    // The paged half, through the identity table -- the one arrangement under
    // which the two shaders are supposed to agree.
    let mut paged = Pool::open(&device, shape).expect("the paged pool");
    let scatter = module(dir, "kv_append_paged_bfloat16");
    let mut ppush = Vec::new();
    ppush.extend_from_slice(&(head_dim as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());
    for p in 0..positions {
        paged
            .state(
                &device,
                FireTable::KvWritePage,
                &[p as u32 / shape.page_size],
            )
            .expect("the write page");
        paged
            .state(
                &device,
                FireTable::KvWriteOffset,
                &[p as u32 % shape.page_size],
            )
            .expect("the write offset");
        let kn = device.buffer(&bf16_bytes(&ks[p])).expect("k_new");
        let vn = device.buffer(&bf16_bytes(&vs[p])).expect("v_new");
        let bound = [
            Bound::whole(&kn),
            Bound::whole(&vn),
            Bound::whole(paged.kv(0, false).expect("keys")),
            Bound::whole(paged.kv(0, true).expect("values")),
            Bound::whole(paged.table(FireTable::KvWritePage).expect("page")),
            Bound::whole(paged.table(FireTable::KvWriteOffset).expect("offset")),
        ];
        let pipeline = cache
            .get(
                &device,
                "kv_append_paged_bfloat16",
                &scatter,
                ppush.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the paged append builds");
        device
            .run(pipeline, &bound, &ppush, [1, shape.kv_heads, 1])
            .expect("the paged append dispatches");
        device.free(kn);
        device.free(vn);
    }

    let (sk, sv) = flat(&device, &straight);
    let (pk, pv) = flat(&device, &paged);

    // Non-trivial first. Two caches of zeros are byte-identical, and a
    // comparison that would accept them measures nothing.
    let live = sk.chunks_exact(2).filter(|c| c != &[0u8, 0]).count();
    let want = positions * heads * head_dim;
    assert!(
        live >= want * 9 / 10,
        "{live} elements of the contiguous cache are non-zero and {want} rows were appended"
    );

    assert_eq!(
        sk.len(),
        pk.len(),
        "the two pools have the same shape, so the same size"
    );
    let differ = sk
        .chunks_exact(2)
        .zip(pk.chunks_exact(2))
        .enumerate()
        .find(|(_, (a, b))| a != b);
    assert!(
        differ.is_none(),
        "the two appends disagree about the key cache at element {:?}",
        differ.map(|(i, _)| i)
    );
    assert_eq!(sv, pv, "and about the value cache");

    // The rows are where the pool says, not merely in agreement with each
    // other. Both shaders reading the same wrong layout would agree.
    let read = bf16_read(&sk);
    let seen = bf16_read(&bf16_bytes(&ks[4]));
    for h in 0..heads {
        for d in 0..head_dim {
            let at = shape.slot(4 / shape.page_size, 4 % shape.page_size, h as u32, d as u32);
            assert_eq!(
                read[at as usize],
                seen[h * head_dim + d],
                "position 4, head {h}, channel {d} is not where the shape says"
            );
        }
    }

    cache.clear(&device);
    straight.close(&device);
    paged.close(&device);
}

/// The contiguous decode reads the pool the paged append wrote.
///
/// The other direction of the same two numbers. `attn/sdpa_vector.comp` never
/// sees a page table; it walks the cache by `kv_head * k_head_stride + i *
/// k_seq_stride`, so the driver's strides are the only thing telling it where
/// a position is. If they were the pair the row's comment describes -- a head
/// stride of `max_ctx * head_dim`, which is what a `[head][pos][dim]` pool
/// would want -- this would attend over memory nobody wrote.
///
/// The rows go in through the paged append, which the test above ties to the
/// pool's own `Shape::slot`. So a disagreement here is the read side's, and
/// the reference is stated in positions and heads without naming a slot.
///
/// Four query heads over two key heads, because a grouped read is where a
/// wrong head stride shows: `gqa_factor` of 1 would let any head stride pass
/// for the single head that starts at zero.
///
/// This is also the first dispatch in the tree whose push block carries four
/// 64-bit members. It is 48 bytes with `scale` at 40, and every offset in it
/// is one the naive packed layout gets wrong.
#[test]
fn the_contiguous_decode_reads_the_pool_the_paged_append_wrote() {
    use driver_vulkan::binding::{FireNumber, FireTable, Resolve};
    use driver_vulkan::resources::{Pool, Shape};

    let (device, dir) = gpu!();
    let head_dim = 128usize;
    let shape = Shape {
        layers: 1,
        kv_heads: 2,
        head_dim: head_dim as u32,
        page_size: 4,
        pages: 8,
        bytes: 2,
    };
    let heads = shape.kv_heads as usize;
    let gqa = 2usize;
    let q_heads = heads * gqa;
    let positions = 6usize;

    let rows = |p: usize, salt: usize| -> Vec<f32> {
        (0..heads * head_dim)
            .map(|i| (((p * 7 + i * 13 + salt * 29) % 61) as f32 - 30.0) / 24.0)
            .collect()
    };
    let ks: Vec<Vec<f32>> = (0..positions).map(|p| rows(p, 0)).collect();
    let vs: Vec<Vec<f32>> = (0..positions).map(|p| rows(p, 1)).collect();
    let queries: Vec<f32> = (0..q_heads * head_dim)
        .map(|i| ((i * 19 % 47) as f32 - 23.0) / 20.0)
        .collect();

    let mut pool = Pool::open(&device, shape).expect("the pool");
    let mut cache = Pipelines::new();

    let scatter = module(dir, "kv_append_paged_bfloat16");
    let mut ppush = Vec::new();
    ppush.extend_from_slice(&(head_dim as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());
    for p in 0..positions {
        pool.state(
            &device,
            FireTable::KvWritePage,
            &[p as u32 / shape.page_size],
        )
        .expect("the write page");
        pool.state(
            &device,
            FireTable::KvWriteOffset,
            &[p as u32 % shape.page_size],
        )
        .expect("the write offset");
        let kn = device.buffer(&bf16_bytes(&ks[p])).expect("k_new");
        let vn = device.buffer(&bf16_bytes(&vs[p])).expect("v_new");
        let bound = [
            Bound::whole(&kn),
            Bound::whole(&vn),
            Bound::whole(pool.kv(0, false).expect("keys")),
            Bound::whole(pool.kv(0, true).expect("values")),
            Bound::whole(pool.table(FireTable::KvWritePage).expect("page")),
            Bound::whole(pool.table(FireTable::KvWriteOffset).expect("offset")),
        ];
        let pipeline = cache
            .get(
                &device,
                "kv_append_paged_bfloat16",
                &scatter,
                ppush.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the append builds");
        device
            .run(pipeline, &bound, &ppush, [1, shape.kv_heads, 1])
            .expect("the append dispatches");
        device.free(kn);
        device.free(vn);
    }

    let head_stride = pool
        .number(FireNumber::KvHeadStride)
        .expect("a head stride");
    let seq_stride = pool.number(FireNumber::KvSeqStride).expect("a seq stride");
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // 48 bytes: two ints, four 8-byte strides on 8-byte boundaries, and a
    // float at 40. Written by offset rather than by concatenation.
    let mut push = vec![0u8; 48];
    push[0..4].copy_from_slice(&(gqa as i32).to_le_bytes());
    push[4..8].copy_from_slice(&(positions as i32).to_le_bytes());
    push[8..12].copy_from_slice(&head_stride.to_le_bytes());
    push[16..20].copy_from_slice(&seq_stride.to_le_bytes());
    push[24..28].copy_from_slice(&head_stride.to_le_bytes());
    push[32..36].copy_from_slice(&seq_stride.to_le_bytes());
    push[40..44].copy_from_slice(&scale.to_le_bytes());

    let qb = device.buffer(&bf16_bytes(&queries)).expect("queries");
    let ob = device
        .buffer(&vec![0u8; q_heads * head_dim * 2])
        .expect("out");
    let symbol = "sdpa_vector_decode_bfloat16_d_128";
    let code = module(dir, symbol);
    {
        let bound = [
            Bound::whole(&qb),
            Bound::whole(pool.kv(0, false).expect("keys")),
            Bound::whole(pool.kv(0, true).expect("values")),
            Bound::whole(&ob),
        ];
        let pipeline = cache
            .get(
                &device,
                symbol,
                &code,
                push.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the decode builds");
        // One workgroup per query head, one row; the head width is the local
        // size, so a channel is an invocation.
        device
            .run(pipeline, &bound, &push, [q_heads as u32, 1, 1])
            .expect("the decode dispatches");
    }

    // The reference, in positions and heads. Computed from the bf16 the card
    // was handed, so bf16's own rounding is not charged to the tolerance.
    let qq = bf16_read(&bf16_bytes(&queries));
    let kq: Vec<Vec<f32>> = ks.iter().map(|k| bf16_read(&bf16_bytes(k))).collect();
    let vq: Vec<Vec<f32>> = vs.iter().map(|v| bf16_read(&bf16_bytes(v))).collect();
    let got = bf16_read(&device.read(&ob).expect("read back"));
    let mut spread = 0.0f32;
    for qh in 0..q_heads {
        let kh = qh / gqa;
        let at = kh * head_dim;
        let scores: Vec<f32> = (0..positions)
            .map(|p| {
                (0..head_dim)
                    .map(|d| scale * qq[qh * head_dim + d] * kq[p][at + d])
                    .sum::<f32>()
            })
            .collect();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f32 = exps.iter().sum();
        spread = spread.max(exps.iter().copied().fold(0.0f32, f32::max) / total);
        for d in 0..head_dim {
            let want = (0..positions).map(|p| exps[p] * vq[p][at + d]).sum::<f32>() / total;
            let saw = got[qh * head_dim + d];
            assert!(
                (saw - want).abs() <= 1e-2 * want.abs().max(1.0),
                "query head {qh}, channel {d}: the decode says {saw}, six rows say {want}"
            );
        }
    }
    // And no head's softmax collapsed onto one position, which would agree
    // with the reference wherever that position dominates.
    assert!(
        spread < 0.9,
        "one position carries {spread} of some head's softmax, so the walk proves little"
    );

    cache.clear(&device);
    device.free(qb);
    device.free(ob);
    pool.close(&device);
}

/// Two requests in one fire do not read each other's history.
///
/// `Frame::of` refuses a row whose position reaches past its own request's
/// pages, and the reason given is that the page lists sit end to end, so one
/// entry over is another request's page: resident, aligned, and silently
/// wrong. That is an argument. This is the measurement.
///
/// Two requests share a pool and a fire. Their pages interleave -- request 0
/// owns 5 and 2, request 1 owns 6 and 1 -- so neither one's pages are
/// contiguous and neither one's are ascending. Every table comes from
/// `Frame::of`; nothing here fills one by hand, which is what every other GPU
/// test in this file does and what this exists to stop.
///
/// Each request's rows are distinct from the other's, so an attention that
/// walked the wrong span would answer with the other request's values. The
/// reference is each request's own rows and nothing else.
#[test]
fn two_requests_in_one_fire_do_not_read_each_others_history() {
    use driver_vulkan::binding::{FireTable, Resolve};
    use driver_vulkan::resources::{Frame, Pool, Request, Shape};

    let (device, dir) = gpu!();
    let head_dim = 128usize;
    let shape = Shape {
        layers: 1,
        kv_heads: 1,
        head_dim: head_dim as u32,
        page_size: 4,
        pages: 8,
        bytes: 2,
    };
    // Interleaved, descending within each request, and neither owns page 0.
    let requests = [
        Request {
            positions: (0..6).collect(),
            pages: vec![5, 2],
            samples: Vec::new(),
        },
        Request {
            positions: (0..3).collect(),
            pages: vec![6, 1],
            samples: Vec::new(),
        },
    ];
    let frame = Frame::of(shape, &requests).expect("a stageable fire");
    let rows = frame.rows();

    // A row's contents depend on which request it belongs to, so reading the
    // wrong span gives the wrong answer rather than a similar one.
    let row = |r: usize, p: u32, salt: usize| -> Vec<f32> {
        (0..head_dim)
            .map(|d| (((r * 31 + p as usize * 7 + d * 13 + salt * 29) % 61) as f32 - 30.0) / 24.0)
            .collect()
    };
    let ks: Vec<Vec<f32>> = (0..rows)
        .map(|t| row(frame.request_of_token[t] as usize, frame.positions[t], 0))
        .collect();
    let vs: Vec<Vec<f32>> = (0..rows)
        .map(|t| row(frame.request_of_token[t] as usize, frame.positions[t], 1))
        .collect();
    let queries: Vec<f32> = (0..requests.len() * head_dim)
        .map(|i| ((i * 19 % 47) as f32 - 23.0) / 20.0)
        .collect();

    let mut pool = Pool::open(&device, shape).expect("the pool");
    pool.stage(&device, &frame).expect("the fire's tables");
    let mut cache = Pipelines::new();

    // The append, one row at a time, each with the page and offset the frame
    // worked out for it.
    let scatter = module(dir, "kv_append_paged_bfloat16");
    let mut ppush = Vec::new();
    ppush.extend_from_slice(&(head_dim as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());
    for t in 0..rows {
        pool.state(&device, FireTable::KvWritePage, &[frame.kv_write_page[t]])
            .expect("the write page");
        pool.state(
            &device,
            FireTable::KvWriteOffset,
            &[frame.kv_write_offset[t]],
        )
        .expect("the write offset");
        let kn = device.buffer(&bf16_bytes(&ks[t])).expect("k_new");
        let vn = device.buffer(&bf16_bytes(&vs[t])).expect("v_new");
        let bound = [
            Bound::whole(&kn),
            Bound::whole(&vn),
            Bound::whole(pool.kv(0, false).expect("keys")),
            Bound::whole(pool.kv(0, true).expect("values")),
            Bound::whole(pool.table(FireTable::KvWritePage).expect("page")),
            Bound::whole(pool.table(FireTable::KvWriteOffset).expect("offset")),
        ];
        let pipeline = cache
            .get(
                &device,
                "kv_append_paged_bfloat16",
                &scatter,
                ppush.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the append builds");
        device
            .run(pipeline, &bound, &ppush, [1, shape.kv_heads, 1])
            .expect("the append dispatches");
        device.free(kn);
        device.free(vn);
    }
    // Put the fire's own write tables back, since the loop above replaced
    // them with one row each.
    pool.stage(&device, &frame)
        .expect("the fire's tables again");

    // One decode row per request, each asking for its own last position. The
    // rows are the requests, so `RequestOfToken` is `[0, 1]` -- not the
    // frame's, which describes the rows that were appended.
    pool.state(
        &device,
        FireTable::Positions,
        &requests
            .iter()
            .map(|r| r.positions.len() as u32 - 1)
            .collect::<Vec<_>>(),
    )
    .expect("the query positions");
    pool.state(&device, FireTable::RequestOfToken, &[0, 1])
        .expect("one row per request");

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut spush = Vec::new();
    spush.extend_from_slice(&1i32.to_le_bytes()); // gqa_factor
    spush.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    spush.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());
    spush.extend_from_slice(&scale.to_le_bytes());
    spush.extend_from_slice(&0u32.to_le_bytes()); // mask stride
    spush.extend_from_slice(&0i32.to_le_bytes()); // window

    let qb = device.buffer(&bf16_bytes(&queries)).expect("queries");
    let ob = device
        .buffer(&vec![0u8; requests.len() * head_dim * 2])
        .expect("out");
    let symbol = "sdpa_paged_decode_bfloat16_d_128";
    let code = module(dir, symbol);
    {
        let bound = [
            Bound::whole(&qb),
            Bound::whole(pool.kv(0, false).expect("keys")),
            Bound::whole(pool.kv(0, true).expect("values")),
            Bound::whole(&ob),
            Bound::whole(pool.table(FireTable::Positions).expect("pos")),
            Bound::whole(pool.table(FireTable::RequestOfToken).expect("req")),
            Bound::whole(pool.table(FireTable::KvPageIndices).expect("ix")),
            Bound::whole(pool.table(FireTable::KvPageIndptr).expect("ptr")),
            Bound::whole(pool.table(FireTable::AttentionMask).expect("mask")),
            Bound::whole(
                pool.table(FireTable::AttentionMaskEnabled)
                    .expect("enabled"),
            ),
        ];
        let pipeline = cache
            .get(
                &device,
                symbol,
                &code,
                spush.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the attention builds");
        // One workgroup per query head, one row per request.
        device
            .run(pipeline, &bound, &spush, [1, requests.len() as u32, 1])
            .expect("the attention dispatches");
    }

    let qq = bf16_read(&bf16_bytes(&queries));
    let got = bf16_read(&device.read(&ob).expect("read back"));
    let mut spread = 0.0f32;
    for (r, request) in requests.iter().enumerate() {
        // This request's rows, and only this request's.
        let mine: Vec<usize> = (0..rows)
            .filter(|&t| frame.request_of_token[t] as usize == r)
            .collect();
        assert_eq!(mine.len(), request.positions.len(), "the fire's rows");
        let kq: Vec<Vec<f32>> = mine
            .iter()
            .map(|&t| bf16_read(&bf16_bytes(&ks[t])))
            .collect();
        let vq: Vec<Vec<f32>> = mine
            .iter()
            .map(|&t| bf16_read(&bf16_bytes(&vs[t])))
            .collect();
        let scores: Vec<f32> = kq
            .iter()
            .map(|k| {
                (0..head_dim)
                    .map(|d| scale * qq[r * head_dim + d] * k[d])
                    .sum::<f32>()
            })
            .collect();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f32 = exps.iter().sum();
        spread = spread.max(exps.iter().copied().fold(0.0f32, f32::max) / total);
        for d in 0..head_dim {
            let want = (0..mine.len()).map(|i| exps[i] * vq[i][d]).sum::<f32>() / total;
            let saw = got[r * head_dim + d];
            assert!(
                (saw - want).abs() <= 1e-2 * want.abs().max(1.0),
                "request {r}, channel {d}: the attention says {saw}, its own {} rows say {want}",
                mine.len()
            );
        }
    }
    assert!(
        spread < 0.9,
        "one position carries {spread} of some request's softmax, so the walk proves little"
    );
    // The two requests were asked different questions of different histories,
    // so an attention that answered one of them twice is not a pass.
    let first = &got[..head_dim];
    let second = &got[head_dim..];
    assert!(
        first.iter().zip(second).any(|(a, b)| (a - b).abs() > 1e-3),
        "both requests were answered identically"
    );

    cache.clear(&device);
    device.free(qb);
    device.free(ob);
    pool.close(&device);
}

/// The ladder this driver builds is the one the shader raises.
///
/// `rope/neox.comp` compiles to two shaders from one file. `neox_mb` raises
/// its own ladder -- `exp2(-(i / pair_half) * base)` -- and `neox_freqs_mb`
/// reads one from a buffer. They exist as a pair because a deployment that
/// rescales its ladder (llama-3, YaRN) has no base to state, and the second
/// is the only form that can carry it.
///
/// So the second is handed `rope::frequencies` and both turn the same tensor.
/// A real plan states `base = log2(rope_theta)` -- measured: `2^19.931568` is
/// qwen3's 1_000_000 and `2^17.194603` is gpt-oss's 150_000 -- so the two are
/// the same ladder said two ways, and nothing but this says so.
///
/// `neox_freqs_mb` had never reached a card. None of the three texts this
/// crate walks launches it, because none of them rescales.
///
/// The position is 7 and not 0, because rope at position 0 is the identity
/// and two identities agree whatever ladder either of them holds. That exact
/// failure is recorded in `kernels-vulkan`'s own row comment for
/// `neox_freqs_mb`, which was bare until a test at position zero stopped
/// hiding it.
#[test]
fn the_ladder_this_driver_builds_is_the_one_the_shader_raises() {
    use driver_vulkan::binding::FireTable;
    use driver_vulkan::resources::{Pool, Shape};

    let (device, dir) = gpu!();
    let head_dim = 128usize;
    let heads = 2usize;
    let rows = 3usize;
    let theta = 1_000_000.0f32;
    let pair_half = head_dim / 2;

    // Three rows at three different positions, so a shader that read position
    // zero for every row -- the defect the `neox_freqs_mb` row records --
    // disagrees on two of them.
    let positions = [7u32, 1, 42];
    let n = rows * heads * head_dim;
    let x: Vec<f32> = (0..n)
        .map(|i| (((i * 13 + 5) % 61) as f32 - 30.0) / 24.0)
        .collect();

    let mut pool = Pool::open(
        &device,
        Shape {
            layers: 1,
            kv_heads: 1,
            head_dim: head_dim as u32,
            page_size: 4,
            pages: 2,
            bytes: 2,
        },
    )
    .expect("the pool");
    pool.state(&device, FireTable::Positions, &positions)
        .expect("the positions");
    // Through `Pool::ladder`, which is the call a server makes, rather than
    // through `state` with `rope::words` spelled out here. The two rope
    // shaders are the only rows in the table that read this, so this test is
    // the only place that seam can be checked at all -- and it was written the
    // long way first, which left the call a server actually makes untested.
    pool.ladder(&device, head_dim as u32, theta, None)
        .expect("the ladder");

    let mut cache = Pipelines::new();
    let turn = |cache: &mut Pipelines, symbol: &str, push: &[u8], freqs: bool| -> Vec<f32> {
        let code = module(dir, symbol);
        let xb = device.buffer(&bf16_bytes(&x)).expect("the tensor");
        {
            use driver_vulkan::binding::Resolve;
            let mut bound = vec![
                Bound::whole(&xb),
                Bound::whole(pool.table(FireTable::Positions).expect("pos")),
            ];
            if freqs {
                bound.push(Bound::whole(
                    pool.table(FireTable::RopeFrequencies).expect("freqs"),
                ));
            }
            let pipeline = cache
                .get(
                    &device,
                    symbol,
                    &code,
                    push.len() as u32,
                    bound.len() as u32,
                    Capability::Baseline,
                )
                .expect("the rotation builds");
            // `Rule::Rope`: x is the pair index, y the head, z the row. The
            // shader reads all three off the grid, so this IS the launch rule.
            device
                .run(
                    pipeline,
                    &bound,
                    push,
                    [pair_half as u32, heads as u32, rows as u32],
                )
                .expect("the rotation dispatches");
        }
        let out = bf16_read(&device.read(&xb).expect("read back"));
        device.free(xb);
        out
    };

    // `float scale; float base; int head_dim;`
    let mut raised = Vec::new();
    raised.extend_from_slice(&1.0f32.to_le_bytes());
    raised.extend_from_slice(&theta.log2().to_le_bytes());
    raised.extend_from_slice(&(head_dim as i32).to_le_bytes());
    // `float scale; int head_dim; float mscale;` -- a different block, and
    // `head_dim` is in the middle rather than at the end.
    let mut read = Vec::new();
    read.extend_from_slice(&1.0f32.to_le_bytes());
    read.extend_from_slice(&(head_dim as i32).to_le_bytes());
    read.extend_from_slice(&1.0f32.to_le_bytes());

    let from_base = turn(&mut cache, "neox_mb_bfloat16", &raised, false);
    let from_ladder = turn(&mut cache, "neox_freqs_mb_bfloat16", &read, true);

    assert_eq!(from_base.len(), from_ladder.len(), "the same tensor");
    for (i, (a, b)) in from_base.iter().zip(&from_ladder).enumerate() {
        assert!(
            (a - b).abs() <= 1e-2 * a.abs().max(1e-2),
            "element {i}: the base raises {a} and the ladder reads {b}"
        );
    }
    // Both turned it. Two shaders that each did nothing agree exactly, and
    // rope IS the identity at position zero -- which is how a bare row hid
    // once already.
    let moved = from_base
        .iter()
        .zip(&x)
        .filter(|(a, b)| (*a - bf16_read(&bf16_bytes(&[**b]))[0]).abs() > 1e-3)
        .count();
    assert!(
        moved > n / 2,
        "only {moved} of {n} elements moved, so neither shader rotated"
    );
    // And each row turned by ITS OWN position. `neox_freqs_mb` was once the
    // decode symbol over a multi-row grid, which rotates row zero and leaves
    // the rest; a shader that read `position[0]` for every row would turn all
    // three by the same angle.
    //
    // Stated as an ORDERING rather than as a count. A ladder falls steeply --
    // the last channel is under 1e-5 -- so most of a head barely moves at any
    // position, and "this row moved a lot" is a claim about the ladder's shape
    // and not about the row. How MANY channels move does track the position,
    // monotonically, and three distinct positions give three distinct counts
    // that a shader reading one position cannot produce.
    let moved_in = |r: usize| -> usize {
        let span = r * heads * head_dim..(r + 1) * heads * head_dim;
        from_ladder[span.clone()]
            .iter()
            .zip(&x[span])
            .filter(|(a, b)| (*a - bf16_read(&bf16_bytes(&[**b]))[0]).abs() > 1e-3)
            .count()
    };
    let counts: Vec<usize> = (0..rows).map(moved_in).collect();
    // `positions` is [7, 1, 42], so the order by angle is row 1, row 0, row 2
    // -- deliberately not the row order, so a shader that turned each row by
    // its own INDEX would not produce it either.
    assert!(
        counts[1] < counts[0] && counts[0] < counts[2],
        "rows at positions {positions:?} moved {counts:?} channels, which does not \
         track the position"
    );

    cache.clear(&device);
    pool.close(&device);
}

/// The rows the frame says are read out are the rows the gather moves.
///
/// The readout seam, end to end and on a card. It is the last thing a fire
/// does and the only part of a plan whose output leaves the arena, and until
/// now nothing in this crate had dispatched it.
///
/// `row_gather` is also the ONLY row in the table with an `InPacked` operand,
/// which is a value with no slot of its own: `RequestCount` is not a push word
/// and not a buffer but the second FIELD of the struct `Param(0)` names.
/// Metal's driver appends it to the scalar run, because there the packed slot
/// IS the buffer; Vulkan sends scalars to a push block and binds the struct as
/// a real std430 buffer, so `Binding::Packed` exists to keep the count out of a
/// push word no shader reads. Nothing had ever run it, so "the count goes in
/// the buffer" was a claim about a code path rather than about a result.
///
/// The fire is deliberately mixed -- two decodes and a prefill that reads three
/// of its four rows -- so that the gather's output rows are neither the fire's
/// first `n` nor one per request, and the sampled rows are not contiguous.
#[test]
fn the_rows_the_frame_reads_out_are_the_rows_the_gather_moves() {
    use driver_vulkan::resources::{Frame, Request, Shape};
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, lower};
    use model_compiler::trace::FireClass;

    let (device, dir) = gpu!();
    let symbol = "row_gather_bfloat16";

    let shape = Shape {
        layers: 1,
        kv_heads: 2,
        head_dim: 8,
        page_size: 8,
        pages: 8,
        bytes: 2,
    };
    // Six rows, four readouts, and the readouts are 0, 2, 4, 5 -- so a gather
    // that copied the first four rows, or one row per request, or every row,
    // gives a different answer from this one.
    let requests = [
        Request::of(vec![0], vec![0]),
        Request {
            positions: vec![0, 1, 2, 3],
            pages: vec![1],
            samples: vec![1, 3],
        },
        Request::of(vec![5], vec![2]),
    ];
    let frame = Frame::of(shape, &requests).expect("a fire");
    assert_eq!(
        frame.sampling_indices,
        vec![0, 2, 4, 5],
        "the fire this test means to run"
    );

    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Prefill,
    );
    // `seriation` and not a hand-built row list: the flags the lowering reads
    // and the table the driver stages come from the same frame, which is the
    // agreement `resources`' own unit test makes and this one depends on.
    let low = lower(
        &plan,
        &frame.seriation(),
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the text lowers");
    assert_eq!(
        low.n_requests as usize,
        frame.readouts(),
        "the gather's count is the table's length"
    );

    // Per-byte and not a ramp, with the exponent clamped so no row holds an
    // infinity: the test compares gathered rows against source rows, and two
    // NaNs are never equal, which would fail a correct gather.
    let mut fill: Vec<u8> = (0..low.arena_bytes).map(|i| (i * 37 % 253) as u8).collect();
    for hi in fill.iter_mut().skip(1).step_by(2) {
        *hi = (*hi & 0x83) | 0x3c;
    }
    let arena_buffer = device.buffer(&fill).expect("the arena allocates");
    let arena = driver_vulkan::binding::Arena {
        buffer: &arena_buffer,
        bytes: low.arena_bytes as u64,
    };

    let mut pool = driver_vulkan::resources::Pool::open(&device, shape).expect("a pool");
    pool.stage(&device, &frame).expect("the tables stage");
    pool.stand_in(&device, 1 << 16).expect("a stand-in");

    let launch = low
        .launches
        .iter()
        .find(|l| low.kernels[l.kernel as usize] == symbol)
        .expect("the text gathers");

    let code = module(dir, symbol);
    let words = driver_vulkan::spirv::words(&code).expect("whole words");
    let declared = driver_vulkan::spirv::declared(&words).expect("well formed");
    let d = driver_vulkan::dispatch::plan_one(
        &low,
        launch,
        kernels_vulkan::KERNELS,
        driver_vulkan::dispatch::Built {
            module: driver_vulkan::geometry::Module::named(
                symbol,
                [declared.local[0], declared.local[1], declared.local[2]],
            ),
            declared: &declared,
        },
        driver_vulkan::dispatch::Sources {
            arena,
            resolver: &pool,
            min_offset: device.min_storage_offset(),
        },
        driver_vulkan::dispatch::Geometry::default(),
    )
    .expect("the rectangle plans");

    // The scalars, from the driver's own binding rather than from this test.
    // Both of them: the width the plan states and the count the DRIVER
    // supplies, in one buffer, which is the whole point of `Binding::Packed`.
    let driver_vulkan::binding::Params::Block { bytes: block, .. } = &d.params else {
        panic!("`row_gather` reads its scalars from a buffer, not from push");
    };
    assert_eq!(block.len(), 8, "`RowGatherParams` is two four-byte fields");
    let width = u32::from_le_bytes(block[0..4].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(block[4..8].try_into().unwrap()) as usize;
    // Which FIELD each landed in, not just that both are present. Swapping
    // them gives a gather of `width` rows of `count` elements, which reads and
    // writes real memory and returns success.
    assert_eq!(
        count,
        frame.readouts(),
        "the count is the second field, and it is the frame's readouts"
    );
    assert!(
        width > count,
        "the width is the first field, and a model's hidden size is not four"
    );
    // Nothing rode a push word. The module declares no push block at all, so a
    // count that went there would be dropped in silence.
    assert!(
        declared.push_offsets.is_empty(),
        "`row_gather` declares no push constants, so a scalar sent there is lost"
    );

    let params = device.buffer(block).expect("the block allocates");
    let mut buffers = d.buffers.clone();
    buffers.insert(d.block_at.expect("a block slot"), Bound::whole(&params));

    // Which slot got the table, checked before the dispatch. The rows operand
    // is the third, and it is the only one that is neither the arena nor the
    // block -- so it comes from the resolver, and the resolver's answer for
    // `SamplingIndices` is what `Pool::stage` wrote.
    assert!(
        std::ptr::eq(
            d.buffers[2].buffer(),
            driver_vulkan::binding::Resolve::table(
                &pool,
                driver_vulkan::binding::FireTable::SamplingIndices
            )
            .expect("the pool staged the readout table")
        ),
        "slot 2 is not the sampling table"
    );

    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(
            &device,
            symbol,
            &code,
            0,
            buffers.len() as u32,
            Capability::Baseline,
        )
        .expect("the pipeline builds");
    device
        .run(pipeline, &buffers, &[], d.groups)
        .expect("dispatch");

    let bytes = |b: Bound<'_>| {
        let whole = device.read(b.buffer()).expect("read back");
        whole[b.offset() as usize..(b.offset() + b.len()) as usize].to_vec()
    };
    let src = bf16_read(&bytes(d.buffers[0]));
    let got = bf16_read(&bytes(d.buffers[1]));

    // Row for row against the SOURCE, not against a host reimplementation of a
    // copy: the only thing a gather can get wrong is which row, so the
    // reference has to be the rows themselves.
    for (i, &at) in frame.sampling_indices.iter().enumerate() {
        let want = &src[at as usize * width..(at as usize + 1) * width];
        let have = &got[i * width..(i + 1) * width];
        assert_eq!(
            have, want,
            "output row {i} is not the fire's row {at}, which the frame says it reads"
        );
    }
    // And it did not write past the readouts. The output range is sized for
    // `Dim::Requests`, so a gather that used the row count would run off it --
    // and `count` is the only thing stopping it.
    assert!(
        got.len() >= count * width,
        "the output range holds fewer than the {count} rows the gather writes"
    );

    // The comparison is not vacuous: the sampled rows differ from each other,
    // so "output row i is source row at" is a claim that could fail. Zeros, or
    // a buffer of one repeated row, would satisfy any permutation.
    for i in 1..frame.readouts() {
        assert_ne!(
            &got[..width],
            &got[i * width..(i + 1) * width],
            "readout rows 0 and {i} are identical, so their order proves nothing"
        );
    }

    cache.clear(&device);
    for b in [arena_buffer, params] {
        device.free(b);
    }
    pool.close(&device);
}

/// A fire that cannot run says which launch, and runs when it can.
///
/// The two ways a caller's module store can be wrong, over a real plan. Both
/// are refused in pass one, before anything is recorded, and both have to name
/// the LAUNCH and not only the symbol: a decode of qwen3-0.6B states 452
/// rectangles over 9 distinct symbols, so "`rms_single_row_bfloat16` has no
/// module" does not say which of the fifty-six, and the interesting question
/// about a failure at rectangle 450 is almost always what came before it.
///
/// The symbol withheld is the LAST distinct one the plan reaches -- launch 450
/// of 452 -- so the refusal happens after four hundred rectangles have been
/// planned and their scalar blocks allocated. Withholding the first would
/// refuse at launch 0, which is the only case that cannot have taken anything.
///
/// # What this does NOT check, and why
///
/// That the blocks are freed. `serve::fire` allocates one buffer per dispatch
/// whose scalars live in a storage block and every early return frees them,
/// which is the kind of claim a test should carry -- and it could not be made
/// to fail. Replacing the free on this path with `std::mem::forget` and firing
/// fifty refusals still left the device allocating a 64 MiB buffer afterwards;
/// the blocks are tens of bytes each and this card has twenty-four gigabytes.
/// Finding the ceiling directly was worse: allocating small buffers in a loop
/// until one is refused did not finish in ten minutes, because each is its own
/// device allocation.
///
/// So the free is stated in the code and its ordering is enforced by the borrow
/// checker -- moving it one line earlier does not compile -- and no test here
/// asserts it. A control that cannot be made to fire is not a control, and the
/// alternative was an assertion that passes whatever the code does.
#[test]
fn a_fire_that_cannot_run_says_which_launch() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire as LowerFire, Row, lower};
    use model_compiler::trace::FireClass;

    let (device, dir) = gpu!();

    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        LowerFire {
            captures_across_splits: false,
        },
    )
    .expect("the text lowers");

    let arena_buffer = device
        .buffer(&vec![0u8; low.arena_bytes])
        .expect("the arena allocates");
    let mut store = driver_vulkan::resources::Pool::open(
        &device,
        driver_vulkan::resources::Shape {
            layers: 28,
            kv_heads: 8,
            head_dim: 128,
            page_size: 8,
            pages: 4,
            bytes: 2,
        },
    )
    .expect("the pool opens");
    store.stand_in(&device, 1 << 22).expect("a stand-in");
    let frame = driver_vulkan::resources::Frame::of(
        store.shape(),
        &[driver_vulkan::resources::Request::of(vec![5], vec![3, 1])],
    )
    .expect("the fire stages");
    store.stage(&device, &frame).expect("the fire's tables");
    store
        .state(
            &device,
            driver_vulkan::binding::FireTable::TokenIds,
            &[0u32],
        )
        .expect("the token ids");
    store
        .ladder(&device, 128, 1_000_000.0, None)
        .expect("the ladder");

    let whole: std::collections::BTreeMap<String, Vec<u8>> = low
        .kernels
        .iter()
        .map(|symbol| {
            let code =
                std::fs::read(dir.join(format!("{symbol}.spv"))).expect("the module is built");
            (symbol.clone(), code)
        })
        .collect();

    let mut seen: Vec<&str> = Vec::new();
    for launch in &low.launches {
        let s = low.kernels[launch.kernel as usize].as_str();
        if !seen.contains(&s) {
            seen.push(s);
        }
    }
    let late = *seen.last().expect("the plan names kernels");
    let deep = low
        .launches
        .iter()
        .position(|l| low.kernels[l.kernel as usize] == late)
        .expect("the plan launches it");
    assert!(
        deep > 100,
        "the withheld symbol first appears at launch {deep}, too early to say anything about \
         what a refusal leaves behind"
    );

    let mut absent = whole.clone();
    absent.remove(late);
    let mut broken = whole.clone();
    // Truncated to a length that is not a whole number of words, so
    // `spirv::words` refuses. Cut on a word boundary it would still be a
    // header, and would be refused later and for a different reason.
    broken.insert(late.to_owned(), whole[late][..37].to_vec());

    let what = driver_vulkan::serve::Fire {
        arena: driver_vulkan::binding::Arena {
            buffer: &arena_buffer,
            bytes: low.arena_bytes as u64,
        },
        resolver: &store,
        geometry: driver_vulkan::dispatch::Geometry {
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            rotary_dims: 128,
            n_experts: 0,
            experts_per_token: 0,
        },
        tier: Capability::Baseline,
        one_at_a_time: false,
    };

    let mut cache = Pipelines::new();
    match driver_vulkan::serve::fire(&device, &mut cache, &absent, &low, what) {
        Err(driver_vulkan::serve::Unfired::NoModule { at, symbol }) => {
            assert_eq!(symbol, late);
            assert_eq!(at, deep, "the refusal names the wrong launch");
        }
        other => panic!("a plan with no module for `{late}` fired: {other:?}"),
    }
    match driver_vulkan::serve::fire(&device, &mut cache, &broken, &low, what) {
        Err(driver_vulkan::serve::Unfired::Unreadable { at, symbol, .. }) => {
            assert_eq!(symbol, late);
            assert_eq!(at, deep, "the refusal names the wrong launch");
        }
        // `Unreadable` and not `NoModule`: a store that holds 37 bytes under a
        // symbol is a different mistake from one that holds nothing, and a
        // caller told the module is missing will go looking for a build step
        // that ran.
        other => panic!("a plan with a truncated `{late}` fired: {other:?}"),
    }

    // And the same plan with every module present runs, so both refusals are
    // about the store and not about the plan.
    let fired = driver_vulkan::serve::fire(&device, &mut cache, &whole, &low, what)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(fired.dispatches, low.launches.len());

    cache.clear(&device);
    device.free(arena_buffer);
    store.close(&device);
}

/// The logits a fire leaves are as many rows as the frame asked for, and are
/// not f32.
///
/// The last mile. `serve::fire` moves the arena and stops; the distribution a
/// server samples from is a range of that arena, and until now nothing in this
/// crate could name it.
///
/// Two claims, and the second is the one that carries a defect.
///
/// The row count is the frame's readouts, which is the third crate to compute
/// the same number: the driver stages `sampling_indices`, `model-compiler`
/// computes `n_requests`, and `Readout::rows` states it again. A fire of three
/// requests is used so that "one row" cannot pass by coincidence.
///
/// The element width is bf16 and NOT f32, because `affine_qmv_fast` writes
/// bf16 whatever the text's declared dtype says. `Readout::bytes`' own doc
/// records what a reader that assumed f32 saw: a vocabulary exactly half
/// zeros, which reads as a dead half of a tensor and is really two elements
/// read as one. That is checked here as a claim about a real plan rather than
/// as a comment -- and the arena is filled with a pattern first, so a range
/// that was never written is not mistaken for a distribution.
#[test]
fn the_logits_a_fire_leaves_are_one_row_per_readout_and_are_not_f32() {
    use driver_vulkan::resources::{Frame, Request, Shape};
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire as LowerFire, lower};
    use model_compiler::trace::FireClass;

    let (device, dir) = gpu!();

    let shape = Shape {
        layers: 28,
        kv_heads: 8,
        head_dim: 128,
        page_size: 8,
        pages: 8,
        bytes: 2,
    };
    // Three requests, three readouts. One would let a readout of one row pass
    // whatever it counted.
    let frame = Frame::of(
        shape,
        &[
            Request::of(vec![0], vec![0]),
            Request::of(vec![5], vec![1]),
            Request::of(vec![9], vec![2, 3]),
        ],
    )
    .expect("the fire stages");
    assert_eq!(frame.readouts(), 3);

    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let low = lower(
        &plan,
        &frame.seriation(),
        LowerFire {
            captures_across_splits: false,
        },
    )
    .expect("the text lowers");

    let exit = low.readout.expect("this text states an exit");
    assert_eq!(
        exit.rows as usize,
        frame.readouts(),
        "the exit states {} rows and the frame reads out {}",
        exit.rows,
        frame.readouts()
    );
    assert_eq!(
        exit.bytes, 2,
        "the exit is {} bytes an element, and this plan's lm head writes bf16",
        exit.bytes
    );

    // A pattern and not zeros: the weights are zeros, so the logits are very
    // likely zeros too, and a readout that pointed at a range nothing wrote
    // would be indistinguishable from one that pointed at the answer. With a
    // pattern underneath, "the exit range is no longer the pattern" is a claim
    // that the fire wrote there.
    let fill: Vec<u8> = (0..low.arena_bytes).map(|i| (i * 37 % 251) as u8).collect();
    let arena_buffer = device.buffer(&fill).expect("the arena allocates");

    let mut store = driver_vulkan::resources::Pool::open(&device, shape).expect("the pool opens");
    store.stand_in(&device, 1 << 22).expect("a stand-in");
    store.stage(&device, &frame).expect("the fire's tables");
    store
        .state(
            &device,
            driver_vulkan::binding::FireTable::TokenIds,
            &vec![0u32; frame.rows()],
        )
        .expect("the token ids");
    store
        .ladder(&device, shape.head_dim, 1_000_000.0, None)
        .expect("the ladder");

    let modules: std::collections::BTreeMap<String, Vec<u8>> = low
        .kernels
        .iter()
        .map(|symbol| {
            let code =
                std::fs::read(dir.join(format!("{symbol}.spv"))).expect("the module is built");
            (symbol.clone(), code)
        })
        .collect();

    let mut cache = Pipelines::new();
    driver_vulkan::serve::fire(
        &device,
        &mut cache,
        &modules,
        &low,
        driver_vulkan::serve::Fire {
            arena: driver_vulkan::binding::Arena {
                buffer: &arena_buffer,
                bytes: low.arena_bytes as u64,
            },
            resolver: &store,
            geometry: driver_vulkan::dispatch::Geometry {
                q_heads: 16,
                kv_heads: 8,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 0,
                experts_per_token: 0,
            },
            tier: Capability::Baseline,
            one_at_a_time: false,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));

    let got = driver_vulkan::serve::logits(&device, &arena_buffer, &low).expect("the logits read");
    assert_eq!(got.rows, frame.readouts());
    assert_eq!(got.vocab, exit.vocab as usize);
    assert_eq!(got.values.len(), got.rows * got.vocab);
    assert!(
        got.row(got.rows - 1).is_some(),
        "the last row is addressable"
    );
    assert!(got.row(got.rows).is_none(), "there is no row past the last");

    // The fire wrote there. Compared against what the arena HELD, so a
    // readout pointing at an untouched range fails.
    let before: Vec<f32> = fill[exit.at..exit.at + got.values.len() * 2]
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect();
    let changed = before
        .iter()
        .zip(&got.values)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        changed > got.values.len() / 2,
        "only {changed} of {} logits differ from what the arena held, so the exit may not be \
         where the fire wrote",
        got.values.len()
    );

    // And the defect `Readout::bytes` records, as a measurement. Read the same
    // range four bytes at a time and exactly half the elements are zero,
    // because a bf16 is the TOP half of an f32 and its low half is the next
    // element's -- which, in a vocabulary of mostly small values, is a run of
    // zero mantissas.
    let whole = device.read(&arena_buffer).expect("the arena reads back");
    let as_f32: Vec<f32> = whole[exit.at..exit.at + got.values.len() * 2]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Widened independently, from the same bytes. Without this the only claim
    // about what `logits` RETURNS is that it differs from the pattern, and a
    // control answering with a row of ones did not fire.
    let expect: Vec<f32> = whole[exit.at..exit.at + got.values.len() * 2]
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect();
    assert!(
        expect
            .iter()
            .zip(&got.values)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the logits are not the widening of the exit's own bytes"
    );

    let alive = as_f32.iter().filter(|v| **v != 0.0).count();
    assert!(
        alive * 4 < as_f32.len(),
        "read as f32 the exit is {alive} of {} alive, so this plan's exit may really be f32 and \
         the width check above is checking nothing",
        as_f32.len()
    );

    // The three refusals `logits` makes before it reads anything. All three
    // were unwitnessed -- deleting either check, or checking the range
    // against an arena four kilobytes larger, left the whole suite green,
    // because the only exit ever read here is the one the lowering states and
    // that one is always in range and always two bytes wide. A readout is
    // four numbers out of a compiler; the reader is what stands between a
    // wrong one and a slice of somebody else's arena.
    // Overrunning by EIGHT bytes and not by a page. A comfortable overrun is
    // refused by a check with slack in it just as well as by a correct one,
    // so the margin is the measurement: adding four kilobytes of slack to the
    // comparison leaves a page-sized overrun still refused and this one
    // accepted.
    let extent = exit.rows as usize * exit.vocab as usize * exit.bytes as usize;
    let mut past = low.clone();
    past.readout = Some(model_compiler::lower::Readout {
        at: low.arena_bytes - extent + 8,
        ..exit
    });
    assert!(
        matches!(
            driver_vulkan::serve::logits(&device, &arena_buffer, &past),
            Err(driver_vulkan::serve::Unread::PastArena { .. })
        ),
        "an exit running off the end of the arena was read anyway"
    );
    // Checked against the arena the LOWERING sized and not the buffer, which
    // is a distinction with no symptom until a caller over-allocates. The
    // plan here says its arena ends eight bytes before its own exit does,
    // while the buffer really is big enough -- so a reader that measured
    // against the buffer, or against the plan with any slack in it, reads
    // eight bytes the plan never claimed. Eight and not a page, because the
    // slice bound below is a second, buffer-shaped check that catches any
    // overrun the buffer cannot hold and would otherwise stand in for this
    // one.
    let mut over = low.clone();
    over.arena_bytes = exit.at + extent - 8;
    assert!(
        matches!(
            driver_vulkan::serve::logits(&device, &arena_buffer, &over),
            Err(driver_vulkan::serve::Unread::PastArena { .. })
        ),
        "an exit past the PLAN's arena was read out of a buffer that happened to be bigger"
    );
    // One byte and not three. Three is the width a reader would guess at,
    // and it is caught by the RANGE check instead -- three bytes an element
    // is half again as many bytes as the plan's arena holds -- so it never
    // reaches the width check and witnesses the wrong refusal. One byte
    // fits, which is exactly what makes it dangerous: the range is fine and
    // the elements are read four at a time as f32 anyway.
    let mut odd = low.clone();
    odd.readout = Some(model_compiler::lower::Readout { bytes: 1, ..exit });
    assert!(
        matches!(
            driver_vulkan::serve::logits(&device, &arena_buffer, &odd),
            Err(driver_vulkan::serve::Unread::Width(1))
        ),
        "a one-byte element was read as an f32"
    );
    // The control: the same three clones with nothing changed still read, so
    // the refusals above are about what was changed and not about cloning.
    assert!(
        driver_vulkan::serve::logits(&device, &arena_buffer, &low.clone()).is_ok(),
        "the control read failed"
    );

    cache.clear(&device);
    device.free(arena_buffer);
    store.close(&device);
}

/// A conversation's history is still its own after another conversation has
/// been seated between two of its fires.
///
/// [`Frame::of`] refuses two requests in ONE fire that name the same page, and
/// every earlier test here handed out page numbers by hand. That is the
/// smaller half of the problem: a request is a conversation, and the page it
/// wrote into in one fire must still be its own in the next. Nothing in a
/// plan, a lowering or a frame says so, so a hand-written caller that started
/// at page 0 for every new conversation would pass every check in this crate
/// and silently give two users each other's history.
///
/// Three fires, on purpose. A grows, then B is seated and grows into what a
/// naive caller would hand it -- A's pages -- then A grows again and attends
/// over its whole history. The reference is A's own six rows; if B's append
/// landed anywhere A had written, the softmax weights move and the answer is
/// not close.
///
/// The control that matters is to hand out pages by hand: both conversations
/// starting at page 0 is the whole defect, and with `Book` it cannot happen.
#[test]
fn a_conversation_keeps_its_pages_while_another_is_seated_between_its_fires() {
    use driver_vulkan::binding::{FireTable, Resolve};
    use driver_vulkan::pages::Book;
    use driver_vulkan::resources::{Frame, Pool, Request, Shape};

    let (device, dir) = gpu!();
    let head_dim = 128usize;
    let shape = Shape {
        layers: 1,
        kv_heads: 1,
        head_dim: head_dim as u32,
        page_size: 4,
        pages: 8,
        bytes: 2,
    };

    let mut book = Book::over(shape);
    // A fills exactly one page, B is seated, then A crosses into a second.
    // The middle growth is the one a naive caller gets wrong.
    let a_first = book.grow(1, 4).expect("room for A");
    let b_first = book.grow(2, 4).expect("room for B");
    let a_second = book.grow(1, 2).expect("room for A again");
    assert_eq!(a_first.pages, vec![0]);
    assert_eq!(b_first.pages, vec![1], "B is not given A's page");
    assert_eq!(a_second.pages, vec![0, 2], "A keeps the page it filled");

    // Contents depend on whose row it is, so a clobbered row is a wrong
    // answer rather than a similar one.
    let row = |who: u64, p: u32, salt: usize| -> Vec<f32> {
        (0..head_dim)
            .map(|d| {
                (((who as usize * 41 + p as usize * 7 + d * 13 + salt * 29) % 61) as f32 - 30.0)
                    / 24.0
            })
            .collect()
    };

    let mut pool = Pool::open(&device, shape).expect("the pool");
    let mut cache = Pipelines::new();
    let scatter = module(dir, "kv_append_paged_bfloat16");
    let mut ppush = Vec::new();
    ppush.extend_from_slice(&(head_dim as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    ppush.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());

    // Every appended row, kept so the reference can be built from what was
    // actually written rather than from what the loop meant to write.
    let mut written: Vec<(u64, u32, Vec<f32>, Vec<f32>)> = Vec::new();
    let mut append = |pool: &mut Pool, cache: &mut Pipelines, who: u64, request: &Request| {
        let frame = Frame::of(shape, std::slice::from_ref(request)).expect("the fire stages");
        for t in 0..frame.rows() {
            let p = frame.positions[t];
            let (k, v) = (row(who, p, 0), row(who, p, 1));
            pool.state(&device, FireTable::KvWritePage, &[frame.kv_write_page[t]])
                .expect("the write page");
            pool.state(
                &device,
                FireTable::KvWriteOffset,
                &[frame.kv_write_offset[t]],
            )
            .expect("the write offset");
            let kn = device.buffer(&bf16_bytes(&k)).expect("k_new");
            let vn = device.buffer(&bf16_bytes(&v)).expect("v_new");
            let bound = [
                Bound::whole(&kn),
                Bound::whole(&vn),
                Bound::whole(pool.kv(0, false).expect("keys")),
                Bound::whole(pool.kv(0, true).expect("values")),
                Bound::whole(pool.table(FireTable::KvWritePage).expect("page")),
                Bound::whole(pool.table(FireTable::KvWriteOffset).expect("offset")),
            ];
            let pipeline = cache
                .get(
                    &device,
                    "kv_append_paged_bfloat16",
                    &scatter,
                    ppush.len() as u32,
                    bound.len() as u32,
                    Capability::Baseline,
                )
                .expect("the append builds");
            device
                .run(pipeline, &bound, &ppush, [1, shape.kv_heads, 1])
                .expect("the append dispatches");
            device.free(kn);
            device.free(vn);
            written.push((who, p, k, v));
        }
    };

    append(&mut pool, &mut cache, 1, &a_first);
    append(&mut pool, &mut cache, 2, &b_first);
    append(&mut pool, &mut cache, 1, &a_second);
    assert_eq!(written.len(), 10);

    // A attends over its whole history: one decode row, its own page table.
    let a_now = Request {
        positions: vec![book.tokens(1).expect("A is seated") as u32 - 1],
        pages: book.pages(1).expect("A is seated").to_vec(),
        samples: Vec::new(),
    };
    let a_frame = Frame::of(shape, std::slice::from_ref(&a_now)).expect("A's decode stages");
    pool.stage(&device, &a_frame).expect("A's tables");

    let queries: Vec<f32> = (0..head_dim)
        .map(|i| ((i * 19 % 47) as f32 - 23.0) / 20.0)
        .collect();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut spush = Vec::new();
    spush.extend_from_slice(&1i32.to_le_bytes());
    spush.extend_from_slice(&(shape.page_size as i32).to_le_bytes());
    spush.extend_from_slice(&(shape.kv_heads as i32).to_le_bytes());
    spush.extend_from_slice(&scale.to_le_bytes());
    spush.extend_from_slice(&0u32.to_le_bytes());
    spush.extend_from_slice(&0i32.to_le_bytes());

    let qb = device.buffer(&bf16_bytes(&queries)).expect("queries");
    let ob = device.buffer(&vec![0u8; head_dim * 2]).expect("out");
    let symbol = "sdpa_paged_decode_bfloat16_d_128";
    let code = module(dir, symbol);
    {
        let bound = [
            Bound::whole(&qb),
            Bound::whole(pool.kv(0, false).expect("keys")),
            Bound::whole(pool.kv(0, true).expect("values")),
            Bound::whole(&ob),
            Bound::whole(pool.table(FireTable::Positions).expect("pos")),
            Bound::whole(pool.table(FireTable::RequestOfToken).expect("req")),
            Bound::whole(pool.table(FireTable::KvPageIndices).expect("ix")),
            Bound::whole(pool.table(FireTable::KvPageIndptr).expect("ptr")),
            Bound::whole(pool.table(FireTable::AttentionMask).expect("mask")),
            Bound::whole(
                pool.table(FireTable::AttentionMaskEnabled)
                    .expect("enabled"),
            ),
        ];
        let pipeline = cache
            .get(
                &device,
                symbol,
                &code,
                spush.len() as u32,
                bound.len() as u32,
                Capability::Baseline,
            )
            .expect("the attention builds");
        device
            .run(pipeline, &bound, &spush, [1, 1, 1])
            .expect("the attention dispatches");
    }

    // The reference: A's six rows and nothing B wrote.
    let qq = bf16_read(&bf16_bytes(&queries));
    let mine: Vec<&(u64, u32, Vec<f32>, Vec<f32>)> =
        written.iter().filter(|(who, ..)| *who == 1).collect();
    assert_eq!(mine.len(), 6, "A appended six rows over two fires");
    let scores: Vec<f32> = mine
        .iter()
        .map(|(_, _, k, _)| {
            let kq = bf16_read(&bf16_bytes(k));
            qq.iter().zip(&kq).map(|(a, b)| a * b).sum::<f32>() * scale
        })
        .collect();
    let top = scores.iter().copied().fold(f32::MIN, f32::max);
    let ws: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f32 = ws.iter().sum();
    let want: Vec<f32> = (0..head_dim)
        .map(|d| {
            mine.iter()
                .zip(&ws)
                .map(|((_, _, _, v), w)| bf16_read(&bf16_bytes(v))[d] * w)
                .sum::<f32>()
                / total
        })
        .collect();

    let got = bf16_read(&device.read(&ob).expect("read back"));
    let mut spread = 0.0f32;
    for (d, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 4e-2,
            "channel {d}: attention gave {g} and A's own six rows give {w}"
        );
        spread = spread.max((w - want[0]).abs());
    }
    // A flat answer would pass the comparison whatever the history was.
    assert!(
        spread > 0.1,
        "the reference answer is nearly constant ({spread}), so this comparison proves little"
    );

    device.free(qb);
    device.free(ob);
    cache.clear(&device);
    pool.close(&device);
}

/// Three steps over one deployment: the positions continue, the pages stay
/// put, and the second step builds no pipeline the first did not.
///
/// Everything before this test fired once. A server does not: it carries one
/// pool, one checkpoint and one pipeline cache across thousands of fires while
/// conversations arrive, grow and leave, and the things that can only be wrong
/// ACROSS fires had no test at all.
///
/// The claims are about bookkeeping, not numerics -- the weights are a stand-in,
/// so the distributions mean nothing and are not compared. What is compared:
///
/// - a conversation's positions continue rather than restarting, which is the
///   difference between a decode and a prefill repeated;
/// - its pages are still its own after a second conversation was seated
///   between two of its steps;
/// - the row count follows the turns, and a batch that grows, shrinks, and
///   mixes a prefill with a decode is one code path;
/// - the pipeline cache stops growing, which is the only reason a server can
///   afford to lower every fire.
///
/// Then a prefill and a mixed batch -- one conversation decoding while another
/// prefills -- which is the shape a server actually runs and which nothing in
/// this crate could express before. The comment where they are fired records
/// why they could not, since the reason is a real disagreement in the lowering
/// rather than a missing feature.
///
/// It closes on a refusal: a step asking for more pages than the cache has
/// left is refused with the gap rather than a bare no, and the book is
/// unchanged afterwards.
#[test]
fn a_deployment_fires_step_after_step_and_stops_building_pipelines() {
    use driver_vulkan::binding::Resolve;
    use driver_vulkan::pages::Book;
    use driver_vulkan::resources::{Pool, Shape, Weights};
    use driver_vulkan::turns::{Held, Serving, Turn, Unstepped};
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::trace::FireClass;

    let (device, dir) = gpu!();
    let shape = Shape {
        layers: 28,
        kv_heads: 8,
        head_dim: 128,
        page_size: 8,
        pages: 6,
        bytes: 2,
    };

    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let serving = Serving {
        plan: &plan,
        geometry: driver_vulkan::dispatch::Geometry {
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            rotary_dims: 128,
            n_experts: 0,
            experts_per_token: 0,
        },
        tier: Capability::Baseline,
    };

    let mut book = Book::over(shape);
    let mut pool = Pool::open(&device, shape).expect("the pool");
    pool.stand_in(&device, 1 << 22).expect("a stand-in");
    pool.ladder(&device, shape.head_dim, 1_000_000.0, None)
        .expect("the ladder");
    let mut weights = Weights::new();
    weights
        .seam(&device, 1 << 22)
        .expect("a stand-in checkpoint");
    // One buffer per NAME. `Model` is a pair and not a fallback chain, so an
    // unheld weight is a refusal rather than a stand-in -- which is the point,
    // and means the names must be collected before the first step. They come
    // from a lowering of the same plan, which is the only place they exist:
    // `Arg::Weight` carries a name and no width, so the SIZE below is a guess
    // that works because `TokenIds` is all zeros and every gather reads row 0.
    {
        let probe = model_compiler::lower::lower(
            &plan,
            &[model_compiler::lower::Row::default()],
            model_compiler::lower::Fire {
                captures_across_splits: false,
            },
        )
        .expect("the plan lowers");
        let names: std::collections::BTreeSet<&str> = probe
            .args
            .iter()
            .filter_map(|a| match a {
                model_compiler::lower::Arg::Weight(n) => Some(n.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            names.len() > 500,
            "only {} weight names, so this is not a whole model",
            names.len()
        );
        let block = vec![0u8; 1 << 22];
        for name in names {
            weights.hold(&device, name, &block).expect("a weight");
        }
    }
    let mut cache = Pipelines::new();

    // Loaded from the first step's kernels and never reloaded, which is also
    // what a server does: the module set is the plan's, not the fire's.
    let mut modules: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    let load = |modules: &mut std::collections::BTreeMap<String, Vec<u8>>| {
        for name in std::fs::read_dir(dir).expect("the spirv dir").flatten() {
            let path = name.path();
            if path.extension().is_some_and(|e| e == "spv")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                modules.insert(stem.to_string(), std::fs::read(&path).expect("a module"));
            }
        }
    };
    load(&mut modules);

    let mut held = Held {
        book: &mut book,
        pool: &mut pool,
        weights: &weights,
    };

    // A prefill of four tokens for one conversation.
    let first = serving
        .step(
            &device,
            &mut cache,
            &modules,
            &mut held,
            &[Turn {
                who: 1,
                tokens: vec![0],
            }],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(first.rows, 1);
    assert_eq!(first.logits.rows, 1, "one distribution per turn");
    assert_eq!(held.book.tokens(1), Some(1));
    let a_pages = held.book.pages(1).expect("A is seated").to_vec();
    assert!(first.pipelines > 0);
    assert_eq!(first.fired.submissions, 1);

    // A second conversation, seated between A's two steps -- the case a
    // hand-written caller gets wrong.
    let second = serving
        .step(
            &device,
            &mut cache,
            &modules,
            &mut held,
            &[
                Turn {
                    who: 1,
                    tokens: vec![0],
                },
                Turn {
                    who: 2,
                    tokens: vec![0],
                },
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(second.rows, 2, "one row each");
    // The tables the DEVICE holds are this step's, not the last one's. Read
    // back rather than inferred: a step that staged once and never again
    // passes every other claim here.
    {
        let held_positions = device
            .read(
                held.pool
                    .table(driver_vulkan::binding::FireTable::Positions)
                    .expect("the positions table"),
            )
            .expect("the table reads back");
        let got: Vec<u32> = held_positions
            .chunks_exact(4)
            .take(second.rows)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, second.positions, "the pool holds another step's rows");
    }
    assert_eq!(second.logits.rows, 2, "one distribution per turn");
    assert_eq!(
        second.logits.vocab, first.logits.vocab,
        "the vocabulary does not depend on the batch"
    );
    assert_eq!(held.book.tokens(1), Some(2), "A's positions continued");
    assert_eq!(held.book.tokens(2), Some(1));
    assert_eq!(
        held.book.pages(1).expect("A"),
        a_pages.as_slice(),
        "A's pages survived B being seated"
    );
    assert!(
        !held
            .book
            .pages(2)
            .expect("B")
            .iter()
            .any(|p| a_pages.contains(p)),
        "B was given a page A still holds"
    );
    assert_eq!(
        second.pipelines,
        first.pipelines,
        "the second step built {} pipelines the first did not",
        second.pipelines - first.pipelines
    );

    // A third, to show the batch can shrink again without rebuilding.
    let third = serving
        .step(
            &device,
            &mut cache,
            &modules,
            &mut held,
            &[Turn {
                who: 2,
                tokens: vec![0],
            }],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(third.rows, 1);
    assert_eq!(third.logits.rows, 1);
    assert_eq!(third.pipelines, first.pipelines);
    assert_eq!(held.book.tokens(1), Some(2), "A did not move this step");

    // A prefill: one turn of four tokens, and a mixed batch after it.
    //
    // This is the case that could not run at all until `Serving::step` forced
    // every row to sample, and it is worth stating what was wrong rather than
    // only that it works. qwen3's text spells its epilogue as three plain
    // `OpKind::Launch` ops, not `OpKind::LmHead`, so `Lowerer::epilogue` --
    // which would emit them over the SAMPLED rows -- never runs and the
    // generic path emits them over the whole token window. With only the last
    // row sampling, `n_requests` is 1 and the arena is sized for one row of
    // logits while the head writes four; measured, that asks for 1215488
    // bytes out of 303872 and `binding::extent` refuses the fire. Forcing
    // every row to sample makes `n_requests` the row count, and the worst
    // operand overrun over both shapes below is ZERO bytes.
    //
    // So the distributions are per ROW, and only the last row of a turn has
    // seen the whole prompt. `readout_of` is what says which one that is, and
    // a caller sampling `logits.row(i)` for turn `i` would read a model that
    // had seen one token.
    let prefill = serving
        .step(
            &device,
            &mut cache,
            &modules,
            &mut held,
            &[Turn {
                who: 3,
                tokens: vec![7, 8, 9, 10],
            }],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(prefill.rows, 4);
    assert_eq!(
        prefill.logits.rows, 4,
        "every row samples, so a prefill of four gives four distributions"
    );
    assert_eq!(
        prefill.readout_of,
        vec![3],
        "the turn's answer is its last row"
    );
    assert!(
        prefill.logits.row(prefill.readout_of[0]).is_some(),
        "the turn's own distribution is addressable"
    );
    assert_eq!(held.book.tokens(3), Some(4));
    // The pool's own `SamplingIndices` says what the lowering was told.
    //
    // The frame names one readout for this turn, so `Pool::stage` writes ONE
    // entry -- while the lowering, told every row samples, has `row_gather`
    // read four. That is a sixteen-byte read of a four-byte buffer, and
    // nothing downstream can see it: `Arg::Named` has no extent to check, the
    // descriptor is bound whole, and the validation layer does not report
    // storage-buffer overruns. Checked here because it is checkable nowhere
    // else.
    {
        let table = device
            .read(
                held.pool
                    .table(driver_vulkan::binding::FireTable::SamplingIndices)
                    .expect("the sampling table"),
            )
            .expect("the table reads back");
        let got: Vec<u32> = table
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            got,
            vec![0, 1, 2, 3],
            "the gather would read {} entries out of {} the pool holds",
            prefill.rows,
            got.len()
        );
    }
    assert_eq!(
        prefill.pipelines, first.pipelines,
        "a prefill built a pipeline no decode needed"
    );

    // And a mixed batch: one conversation decoding while another prefills,
    // which is the shape a server actually runs and which no earlier test in
    // this crate could express.
    let mixed = serving
        .step(
            &device,
            &mut cache,
            &modules,
            &mut held,
            &[
                Turn {
                    who: 1,
                    tokens: vec![5],
                },
                Turn {
                    who: 4,
                    tokens: vec![1, 2, 3],
                },
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        mixed.rows, 4,
        "one row for the decode and three for the prefill"
    );
    assert_eq!(mixed.logits.rows, 4);
    assert_eq!(mixed.readout_of.len(), 2);
    assert_ne!(
        mixed.readout_of[0], mixed.readout_of[1],
        "two turns cannot share one distribution"
    );
    // The decode's turn contributed exactly one row, so its readout row is
    // the row whose position is the one the book had. Read from the frame the
    // step reported rather than assumed.
    assert_eq!(
        mixed.positions[mixed.readout_of[0]], 2,
        "the decode's answer is the row at its own position"
    );
    assert_eq!(
        mixed.positions[mixed.readout_of[1]], 2,
        "the prefill's answer is its LAST token, not its first"
    );
    assert_eq!(held.book.tokens(1), Some(3));
    assert_eq!(held.book.tokens(4), Some(3));
    held.book.release(3);
    held.book.release(4);

    // And the refusal. Six pages of eight is 48 tokens; A and B hold four
    // between them, so this cannot fit.
    let before = held.book.spare();
    let refused = serving
        .step(
            &device,
            &mut cache,
            &modules,
            &mut held,
            &[Turn {
                who: 9,
                tokens: vec![0; 400],
            }],
        )
        .expect_err("the cache is far too small for this");
    match refused {
        Unstepped::Unhoused(driver_vulkan::pages::Unhoused::NoPages { wanted, spare }) => {
            assert!(
                wanted > spare,
                "{wanted} wanted and {spare} spare is not a refusal"
            );
            assert_eq!(spare, before);
        }
        other => panic!("the wrong refusal: {other}"),
    }
    assert_eq!(held.book.spare(), before, "a refused growth took a page");
    assert!(
        held.book.pages(9).is_none(),
        "a refused conversation kept a seat"
    );

    cache.clear(&device);
    pool.close(&device);
    weights.close(&device);
}

/// The weight store answers by name, replaces without leaking, and never
/// answers a name it was not given.
///
/// `Weights` has been the resolver under every whole-plan fire in this crate
/// and has never been asked anything on its own. That is not a gap about
/// coverage: the three claims below are each a way for a fire to bind the
/// WRONG buffer and compute a plausible answer, and a whole-plan test cannot
/// see any of them, because it holds one four-megabyte block under every name
/// and so cannot tell the names apart.
///
/// Distinct contents per name, therefore, and the check is on the BYTES rather
/// than on the handle -- a store that returned the right buffer object for the
/// wrong name would pass a comparison of pointers.
///
/// The third claim is the one with teeth. `Model` is a pair, not a fallback
/// chain, and `Weights::named` answers the seam for ANY value id. If `weight`
/// fell back to the seam the same way, a plan naming a weight nobody loaded
/// would bind a buffer of zeros and produce a fire that runs, computes
/// nonsense and refuses nothing.
#[test]
fn a_weight_store_answers_by_name_and_refuses_a_name_it_was_never_given() {
    use driver_vulkan::binding::Resolve;
    use driver_vulkan::resources::Weights;

    let (device, _) = gpu!();
    let mut weights = Weights::new();
    assert!(weights.is_empty());
    assert_eq!(weights.len(), 0);

    let block = |seed: u8| -> Vec<u8> { (0..256u32).map(|i| (i as u8) ^ seed).collect() };
    for (name, seed) in [("layer.0.q", 1u8), ("layer.0.k", 2), ("embed", 3)] {
        weights.hold(&device, name, &block(seed)).expect("a weight");
    }
    assert_eq!(weights.len(), 3);
    assert!(!weights.is_empty());

    // By NAME, and checked on the bytes. Three names that differ only in
    // their last character, so a store keying on a prefix passes nothing.
    for (name, seed) in [("layer.0.q", 1u8), ("layer.0.k", 2), ("embed", 3)] {
        let got = device
            .read(weights.at(name).expect("held under its name"))
            .expect("it reads back");
        assert_eq!(got, block(seed), "`{name}` answered with another's bytes");
        let bound = Resolve::weight(&weights, name).expect("the resolver agrees");
        assert_eq!(
            device.read(bound).expect("it reads back"),
            block(seed),
            "`{name}` binds a different buffer than it holds"
        );
    }

    // Replacing gives the new bytes, keeps the count, and -- the part a
    // reader cannot check from outside -- frees the old buffer rather than
    // stranding it. The count is checked; the free is not, and this says so
    // rather than pretending: nothing in this crate can observe a Vulkan
    // buffer that was allocated and never freed, for the reason
    // `serve::fire`'s own doc records at length.
    weights
        .hold(&device, "embed", &block(9))
        .expect("the replacement");
    assert_eq!(weights.len(), 3, "replacing a name added one");
    assert_eq!(
        device
            .read(weights.at("embed").expect("still held"))
            .expect("read"),
        block(9),
        "the replacement did not take"
    );

    // A name nobody gave it is None, even though a seam exists. The seam
    // answers `named` for any value id on purpose; a `weight` that shared
    // that generosity would bind zeros for a weight a checkpoint forgot and
    // the fire would run.
    weights.seam(&device, 4096).expect("a seam");
    assert!(Resolve::named(&weights, 0).is_some(), "the seam answers");
    assert!(Resolve::named(&weights, 99_999).is_some(), "for any value");
    assert!(
        weights.at("layer.1.q").is_none(),
        "a name never held was answered"
    );
    assert!(
        Resolve::weight(&weights, "layer.1.q").is_none(),
        "an unheld weight fell back to the seam"
    );
    assert!(
        Resolve::weight(&weights, "").is_none(),
        "the empty name was answered"
    );

    weights.close(&device);
}
