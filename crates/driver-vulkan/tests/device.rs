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
/// Whether the device this suite opened is a CPU implementation.
///
/// Set when the device is opened and read without locking, for the reason
/// given there. `false` before the first open, which is the right answer for
/// a suite that has not got a device yet: nothing has been timed either.
static SOFTWARE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn on_software() -> bool {
    *SOFTWARE.get().unwrap_or(&false)
}

fn gpu() -> Option<MutexGuard<'static, Device>> {
    let held = GPU.get_or_init(|| match Device::open() {
        Ok(d) => {
            // Recorded HERE, not read later, because every caller of
            // `on_software` already holds this mutex -- a helper that took it
            // again to ask one bool would deadlock the suite.
            let _ = SOFTWARE.set(d.software());
            Some(Mutex::new(d))
        }
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

/// A wall-clock ceiling, and what it means on a device that has no clock worth
/// asserting against.
///
/// Five tests in this file hold a time. Every one of them is a real regression
/// guard -- they are how "the 370 s KV copy" and "attention reading the
/// history far too many times" were caught -- and every one of them is a
/// statement about a particular piece of hardware, because that is the only
/// way a millisecond ceiling can mean anything.
///
/// Run on Mesa's `llvmpipe` they fail, and they fail for no interesting
/// reason: an LLVM JIT on the host's cores is not the card the number was
/// calibrated on. Deleting them there would be worse than useless -- a suite
/// that skips in silence reports success for work it did not do -- so the
/// ceiling is widened by a stated factor instead, and the widening is
/// announced every time it happens.
///
/// Sixty is not arbitrary. Every defect these budgets were written against is
/// two to four orders of magnitude over its ceiling -- 370 s against 250 ms,
/// 503 ms against 15 ms -- so all of them still fail at sixty times. What is
/// given up is the ability to notice a two-fold regression on a device where
/// a two-fold difference is the host being busy.
fn within_budget(took: std::time::Duration, base: std::time::Duration, what: &str) {
    if !on_software() {
        assert!(took < base, "{what}");
        return;
    }
    const SLACK: u32 = 60;
    let ceiling = base * SLACK;
    eprintln!(
        "SOFTWARE ADAPTER: {took:?} against a hardware budget of {base:?}, \
         checked at {SLACK}x ({ceiling:?}) instead. This is not a timing \
         measurement of anything."
    );
    assert!(
        took < ceiling,
        "{what}\n\n...and this is {SLACK}x the hardware budget on a software \
         adapter, so it is not the calibration that is wrong"
    );
}

/// The bf16 narrowing `common/bf16.slang` does, in Rust.
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

/// A grid past what this device dispatches is refused, and one exactly at the
/// limit is not.
///
/// `maxComputeWorkGroupCount` is the limit with the widest spread in Vulkan
/// and the one this crate had never read. The card measured here answers
/// 2147483647 on x and exactly the specification's floor, 65535, on y and z
/// -- so the refusal is not hypothetical even on a 4090, and a device that
/// answers the floor on all three is common.
///
/// What makes it worth a refusal rather than a comment is what a card does
/// with a grid past the limit: nothing defined. It may dispatch the part that
/// fits and return success, which is an output computed for some of its rows
/// and stale for the others -- fluent, plausible and wrong, which is the
/// class of defect this crate is built against.
///
/// The control is the point of the test. A limit check that refused
/// everything, or that was off by one, would pass the first half; so the
/// second half dispatches a grid of EXACTLY the limit and requires it
/// through. 65535 workgroups of a row norm is real work this card does in
/// under a millisecond, and every workgroup past the four rows the buffer
/// holds writes outside its bound range, which `robustBufferAccess` discards
/// -- the same behaviour the overrun test measures.
#[test]
fn a_grid_past_what_this_device_dispatches_is_refused_and_one_at_the_limit_is_not() {
    let (device, dir) = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";
    let code = module(dir, entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, &code, 0, 0, Capability::Baseline)
        .expect("the pipeline builds");

    let axis = 256usize;
    let limits = device.max_groups();
    let x: Vec<f32> = (0..axis * 4).map(|i| 1.0 + (i % 7) as f32).collect();
    let w = vec![1.0f32; axis];
    let mut params = Vec::new();
    params.extend_from_slice(&1e-5f32.to_le_bytes());
    params.extend_from_slice(&(axis as u32).to_le_bytes());
    params.extend_from_slice(&1u32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&1.0f32.to_le_bytes());
    let xb = device.buffer(&bf16_bytes(&x)).expect("x");
    let wb = device.buffer(&bf16_bytes(&w)).expect("w");
    let ob = device.buffer(&vec![0u8; axis * 4 * 2]).expect("out");
    let pb = device.buffer(&params).expect("params");
    let bound = [
        Bound::whole(&xb),
        Bound::whole(&wb),
        Bound::whole(&ob),
        Bound::whole(&pb),
    ];

    // Every axis, because they are three different limits and a check that
    // compared them all against the first would pass on a card whose x is
    // the widest.
    for axis_of in 0..3 {
        let Some(past) = limits[axis_of].checked_add(1) else {
            // A limit of `u32::MAX` cannot be exceeded by a number that fits
            // in the dispatch call, so there is nothing to refuse.
            continue;
        };
        let mut groups = [1u32; 3];
        groups[axis_of] = past;
        let refused = device.run(pipeline, &bound, &[], groups);
        assert!(
            matches!(
                refused,
                Err(Failed::Grid { axis, groups, limit })
                    if axis as usize == axis_of
                        && groups == past
                        && limit == limits[axis_of]
            ),
            "a grid of {past} on axis {axis_of} against a limit of {} was answered \
             with {refused:?}",
            limits[axis_of]
        );
    }

    // The control, on the narrowest axis this card states, so that the
    // dispatch is one it will really run.
    let at = limits[1].min(65_535);
    device
        .run(pipeline, &bound, &[], [1, at, 1])
        .expect("a grid of exactly the limit is legal and must not be refused");

    for b in [xb, wb, ob, pb] {
        device.free(b);
    }
    cache.clear(&device);
}

/// A row that lists a buffer its shader never reads still gets a layout for it.
///
/// `layer_scalar_mul_bfloat16` is one of the eleven entrypoints where the two
/// counts disagree: the row lists four buffers and the compiled module
/// decorates three, because slangc drops the `OpDecorate Binding` of one the
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
    use model_ir::trace::FireClass;

    let (device, dir) = gpu!();

    // One stand-in for every weight and seam value. Zeros, because nothing
    // here reads a result back.
    let weights = device.buffer(&vec![0u8; 1 << 22]).expect("weights");
    struct Zeros<'a>(&'a driver_vulkan::device::Buffer);
    impl driver_vulkan::binding::Resolve for Zeros<'_> {
        fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
            Some(self.0)
        }
        fn named(&self, _: model_ir::trace::ValueId) -> Option<&driver_vulkan::device::Buffer> {
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
    let mut crossed = 0u32;
    let mut refused = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Both fire classes of all six texts.
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
        (
            LlamaLikeFacts::qwen2_5_1_5b(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (
            LlamaLikeFacts::mistral_7b_v03(),
            LlamaLikeMetalFacts::synthetic(),
        ),
        (LlamaLikeFacts::olmo2_1b(), LlamaLikeMetalFacts::synthetic()),
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
                // A CROSSED symbol is not reached this way any more. This
                // walk is the table path, spelled out: `plan_one` against a
                // `kernel!` row, spliced and submitted by hand. A symbol
                // whose family has landed has no row, so it is counted
                // separately rather than dropped -- the total below is still
                // every symbol these six texts reach, and it still moves only
                // when a text does.
                if driver_vulkan::arm::arm_for(symbol).is_some() {
                    crossed += 1;
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
                        sig: None,
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
    // Twenty-one: every distinct symbol the six texts reach, in both fire
    // classes, submitted and accepted.
    //
    // The twenty-first is `add_bias`, which arrived when the shared text
    // started stating qwen2.5's and gpt-oss's attention biases. It had been
    // written, built and dispatched by hand here for two milestones without
    // a plan ever asking for it; this is the first run in which a real one
    // does.
    //
    // Was nineteen until the compiler stopped sending an MXFP4 expert bank
    // to the unbiased routed symbol. The bank publishes one additive term
    // per output row, `mxfp4_qmv_routed_bias` is the kernel that reads it,
    // and this crate binds and dispatches it unchanged -- so the twentieth
    // row arrived without a line of driver code, which is the point of
    // deriving bindings from the statement rather than from a table.
    //
    // Still the same number after three texts became six, and that is the finding
    // rather than a non-event: qwen2.5, mistral and olmo2 are three more
    // architectures and they reach not one kernel the first three did not.
    // A dense llama-like decode is the same rows at different
    // widths, so what widens this number is a STRUCTURE nothing here has --
    // a mixture of experts did, a sink attention did, a prefill did -- and
    // not another model.
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
    // Then twenty-two, when a rescaled deployment reached these texts:
    // `neox_freqs_mb` is the rotation whose ladder is handed over as a buffer
    // rather than raised from a base, and `rope.rs` builds that buffer.
    //
    // Then twenty-four, when a text quantised at 8 bits arrived upstream: a
    // point carries the bit width, so the same projection at 8 bits names a
    // symbol its 4-bit sibling does not, twice -- once for the GEMM and once
    // for the GEMV.
    //
    // Then twenty-six, when the two `sdpa_paged_tiled` rows stopped being
    // bare axes and started stating their operands: a prefill reaches the
    // tile, a decode reaches the decode kernel, and both are in these texts.
    //
    // Equality rather than a floor: twenty-six is all of them, so this
    // number moving in either direction is news.
    assert_eq!(
        ran + crossed,
        26,
        "a different number of distinct symbols reached the device: {}",
        seen.iter().cloned().collect::<Vec<_>>().join(", ")
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
/// `rms_single_row` is the whole story in one row. `norm/rms.slang` decorates
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
    use model_ir::trace::FireClass;

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
            sig: None,
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
                    symbol: "chained",
                    pipeline,
                    buffers: b,
                    // Empty: no mask, so every slot counts as written and
                    // every pair gets a barrier. This test is about the
                    // chain, and a chain is what the coarse reading records.
                    writes: &[],
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

/// Fires of different sizes, one after another, all say what one fire says.
///
/// A fire no longer makes its own descriptor pool, command buffer and fence.
/// It borrows the device's, resets them, and gives them back -- which took a
/// small fire on this card from 421 microseconds to 35 and is the difference
/// between a driver whose cost is the work and one whose cost is the setup.
///
/// Reuse is exactly where that kind of change goes wrong, and none of the
/// three ways is loud:
///
/// * a descriptor pool that is not RESET runs out of sets, which is a
///   refusal rather than a wrong number, and only on the second fire;
/// * a pool that never GROWS refuses the first fire bigger than the last;
/// * a fence that is not reset is already signalled, so `wait_for_fences`
///   returns at once and the read that follows races the GPU. That one
///   returns success and wrong numbers.
///
/// So the shape of this test is: small, then large, then small again, twenty
/// times, each answer compared bit for bit against the same chain the first
/// time it ran. The large fire is 64 links deep so that a fire nobody waited
/// for has a wide window to be caught in, and the chain is a chain so that a
/// dispatch reading a row the previous one had not written yet cannot agree
/// by accident.
#[test]
fn fires_of_different_sizes_in_a_row_reuse_the_scratch_and_still_agree() {
    let (device, dir) = gpu!();
    let entrypoint = "rms_single_row_bfloat16";
    let axis = 512usize;

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

    let stride = (axis * 2) as u64;
    let wb = device.buffer(&bf16_bytes(&w)).expect("w");
    let pb = device.buffer(&params).expect("params");
    let deepest = 64usize;
    let mut initial = bf16_bytes(&x);
    initial.resize((deepest + 1) * axis * 2, 0);

    let chain = |links: usize| -> Vec<f32> {
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
        let run: Vec<driver_vulkan::device::Recorded<'_, '_>> = sets
            .iter()
            .map(|b| driver_vulkan::device::Recorded {
                symbol: "batched",
                pipeline,
                buffers: b,
                writes: &[],
                push: &[],
                groups,
            })
            .collect();
        device.run_all(&run).expect("the fire records and submits");
        let out = device.read(&rows).expect("read back");
        device.free(rows);
        bf16_read(&out[links * axis * 2..])
    };

    let before = device.pools_made();
    let small = chain(2);
    let large = chain(deepest);
    assert!(
        small.iter().filter(|v| **v != 0.0).count() > axis / 2,
        "the small chain wrote mostly zeros, so the comparison proves little"
    );
    assert!(
        large.iter().filter(|v| **v != 0.0).count() > axis / 2,
        "the deep chain wrote mostly zeros, so the comparison proves little"
    );

    for round in 0..20 {
        // Small AFTER large as well as before it: a pool that grew must still
        // serve a fire that wants less of it, which a pool sized to the
        // request rather than to the high-water mark would not.
        assert_eq!(
            chain(2),
            small,
            "fire {round} of two dispatches disagrees with the first one"
        );
        assert_eq!(
            chain(deepest),
            large,
            "fire {round} of {deepest} dispatches disagrees with the first one"
        );
    }

    // At most two pools for forty-two fires: one for the small shape and one
    // when the deep one asked for more than it held. Often zero, because this
    // suite shares one device and an earlier test has already grown the pool
    // past both shapes -- which is the same statement, made more strongly.
    //
    // A bound rather than an equality for that reason, and it is still the
    // assertion that makes the reuse a CLAIM rather than a hope: everything
    // above passes just as well against a device that builds a fresh pool
    // every fire. That answers correctly, needs forty-two of them, and is
    // what the 421 microseconds were.
    let grew = device.pools_made() - before;
    assert!(
        grew <= 2,
        "42 fires of two shapes needed {grew} descriptor pools, so they are not being reused"
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
    //
    // And BOTH fire classes, not just the decode. A decode is one row, so the
    // whole tiled-GEMM half of the tree -- every `Qmm`, the largest kernel
    // family there is -- was reached by nothing here; a prefill of 64 rows
    // reaches it. It also found the only place on this card where a plan has
    // no single answer (see the determinism probe in `whole_plan`), which a
    // one-row fire cannot show because one row has nothing to tie.
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
        for (class, rows) in [
            (model_ir::trace::FireClass::Decode, 1),
            (model_ir::trace::FireClass::Prefill, 64),
        ] {
            fired += whole_plan(
                &device,
                dir,
                name,
                &facts,
                &metal,
                Wanted {
                    class,
                    rows,
                    embed: 1 << 22,
                    weights: false,
                    compare: true,
                    real: None,
                },
            )
            .fired;
        }
    }
    // Pinned, because a loop is not evidence that a loop ran: every text here
    // could stop lowering and each `whole_plan` would still be a pass over
    // whatever remained. This is the sum of the six texts' launches over BOTH
    // fire classes, every one of them recorded and submitted on this card, and
    // it moves when a text does.
    //
    // It is also, exactly, the 6680 the arena walk in `tests/arena.rs` counts
    // over the same six texts and the same two classes -- which is the point
    // of firing the prefills. Until they were added this test ran half of what
    // that walk measures, so the tiled `Qmm` GEMMs, the largest kernel family
    // in the tree, had never reached the card inside a real plan.
    assert_eq!(
        fired, 6680,
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
    what: Wanted,
) -> Ran {
    let Wanted {
        class,
        rows,
        embed,
        weights: numbers,
        compare,
        real,
    } = what;
    let real_of = |n: &str| -> Option<&'static [u8]> {
        real.map(|m| {
            m.get(n)
                .unwrap_or_else(|| panic!("the checkpoint holds no `{n}`"))
                .as_slice()
        })
    };
    let block_of = |n: &str| {
        if let Some(b) = real_of(n) {
            b.len()
        } else if n.starts_with("embed") {
            embed
        } else {
            1 << 22
        }
    };
    let fill_of = |n: &str| -> Vec<u8> {
        if let Some(b) = real_of(n) {
            return b.to_vec();
        }
        let len = block_of(n);
        if !numbers {
            vec![0u8; len]
        } else if n.ends_with(".scales") || n.ends_with(".zeros") || n.contains("norm") {
            vec![0x3F; len]
        } else {
            // A HASH AND NOT `i * 31 % 251`, which is what this was and which
            // made a whole family of claims vacuous.
            //
            // That expression repeats every 251 bytes. The tied head reads
            // rows of the same buffer, and a row stride coprime with 251 --
            // which every one of these models has -- means vocabulary rows
            // 251 apart are byte-identical. Measured on qwen3-30b-a3b: the
            // top logit of every row of a 16-row prefill was attained by 688
            // DIFFERENT tokens at exactly the same value. Any argmax over
            // that is a report of which index the scan saw first, and
            // `a_routed_prefill_answers_the_same_twice` flipped roughly one
            // run in three because a hundredth of a percent of noise on one
            // row reshuffled the plateau.
            //
            // A multiplicative hash has no period this side of 2^64, so
            // distinct rows get distinct weights and a maximum is a maximum.
            (0..len)
                .map(|i| {
                    let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    (h >> 33) as u8
                })
                .collect()
        }
    };
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, Row, lower};

    let name = &format!("{name}/{class:?}");
    let plan = llama_like_metal(facts, metal, class);
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
        // Sized to the FIRE and not to a decode: a prefill of 64 rows at
        // position 5 needs five pages of 16, and `Frame::of` refuses
        // `PastItsPages` rather than reading the next request's page -- which
        // is the refusal working, and is how this was found.
        page_size: 16,
        pages: 8,
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
        weights.hold(device, n, &fill_of(n)).expect("a weight");
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
            // A prefill's rows are its whole prompt, so the positions run
            // and the page list has to be long enough to hold them. The
            // decode's single row stays at position 5 with a page list that
            // is deliberately not the identity.
            positions: (0..rows as u32).map(|p| p + 5).collect(),
            pages: vec![3, 1, 0, 2, 6, 4, 7, 5],
            samples: Vec::new(),
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
        }],
    )
    .expect("the fire stages");
    store.stage(device, &frame).expect("the fire's tables");
    // The one a `Frame` does not derive, because it is not a function of the
    // paging: what the rows say. One per row, so a prefill states its whole
    // prompt rather than a decode's single token.
    store
        .state(
            device,
            driver_vulkan::binding::FireTable::TokenIds,
            &vec![0u32; rows],
        )
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
    let firing = driver_vulkan::serve::Fire {
        arena,
        resolver: &model,
        geometry,
        tier: Capability::Baseline,
        one_at_a_time: false,
    };
    let before = device.allocations();
    let barriers_before = device.barriers();
    let fired = driver_vulkan::serve::fire(device, &mut cache, &modules, &low, firing)
        .unwrap_or_else(|e| panic!("{e}"));
    // Every launch the plan states, in one command buffer. Without this a
    // `fire` that dropped the last rectangle passes, because the comparison
    // below is against the same `fire` and both runs lose the same thing.
    assert_eq!(
        fired,
        driver_vulkan::serve::Fired {
            dispatches: low.launches.len(),
            submissions: 1,
            blocks: fired.blocks,
            parsed: fired.parsed,
            tiered: fired.tiered,
        }
    );
    // One device allocation for all of them. `blocks` is what this fire used
    // to ask the device for, one buffer each -- 114 of them in a qwen3 decode
    // and over a thousand in a prefill of the 30B mixture -- at a measured
    // 260 microseconds apiece, against a `maxMemoryAllocationCount` that is
    // 4096 on a good many cards. A fire is free to state more rectangles than
    // that, so this was a ceiling and not only a cost.
    //
    // Stated as a comparison against `blocks` rather than as a constant so it
    // says the same thing for every text and both fire classes here, and with
    // a floor so that a lowering which stopped stating blocks could not make
    // it vacuously true.
    assert!(
        fired.blocks > 20,
        "{name}: only {} rectangles state a scalar block, so the count below \
         proves little",
        fired.blocks
    );
    assert_eq!(
        device.allocations() - before,
        1,
        "{name}: a fire of {} scalar blocks made more than one allocation",
        fired.blocks
    );
    // And one module read per distinct symbol, not one per rectangle. The
    // plan states hundreds of rectangles over a handful of symbols, and
    // reading a module is a walk over a few thousand words: measured at 22
    // milliseconds of a 24-millisecond planning pass before the read was
    // cached, against 17 milliseconds of GPU for the whole fire.
    //
    // Against the plan's own count of distinct kernels rather than a
    // constant, so it says the same thing for every text and both classes.
    //
    // It counts the kernel TABLE lookup too, since that moved into the same
    // cache-miss arm: `kernels::sig_in` is a linear scan of every row and
    // then, for a specialised name, of every row's axis points, and a decode
    // was doing it 452 times for nine distinct symbols. There is no second
    // counter because there is no second walk -- one miss, both lookups.
    let symbols: std::collections::BTreeSet<u16> = low.launches.iter().map(|l| l.kernel).collect();
    assert_eq!(
        fired.parsed,
        symbols.len(),
        "{name}: {} rectangles over {} symbols read {} modules",
        fired.dispatches,
        symbols.len(),
        fired.parsed
    );
    assert!(
        fired.dispatches > 4 * symbols.len(),
        "{name}: {} rectangles over {} symbols is too flat for the line above \
         to mean anything",
        fired.dispatches,
        symbols.len()
    );
    // Fewer barriers than rectangles, and more than none.
    //
    // The recording used to put a full pipeline barrier between every pair of
    // dispatches, which is `dispatches - 1` of them, and on this card that was
    // 3.8 milliseconds of a 7.2 millisecond decode -- eight microseconds each,
    // which is what serialising a dispatch that takes two costs. Most
    // neighbouring pairs of a plan do not touch the same bytes, and
    // `device::hazards` is what tells which do.
    //
    // Stated as a range rather than a number because the count is the plan's,
    // and both ends are load-bearing. A `hazards` that answered `true` for
    // everything would be the old recording and would pass every other
    // assertion in this file, including the byte-for-byte comparison below;
    // one that answered `false` for everything would race, which the
    // comparison catches on most runs and not on all of them. Measured over
    // the six texts this helper fires, decode and prefill alike: 311 of 452
    // for qwen3-0.6b, 311 of 480 for qwen2.5-1.5b, 227 of 292 for olmo2-1b,
    // 323 of 452 for mistral-7b, 459 of 604 for gpt-oss-20b and 819 of 1060
    // for the 30B mixture. Between three fifths and four fifths, which is the
    // shape of a transformer: a layer is a chain, and the handful of things
    // inside it that are not -- the three projections, the per-head writes --
    // are what this recovers.
    let barriers = device.barriers() - barriers_before;
    assert!(
        barriers > 0 && barriers < fired.dispatches as u32 - 1,
        "{name}: {barriers} barriers over {} rectangles, which is either the \
         coarse recording or no ordering at all",
        fired.dispatches
    );
    let recorded = device.read(&arena_buffer).expect("the arena reads back");
    // Read HERE and not at the end, because the slow run below overwrites the
    // arena this one left.
    let answer = driver_vulkan::serve::logits(device, &arena_buffer, &low)
        .unwrap_or_else(|e| panic!("{name}: {e}"));

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
    if compare {
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
                ..firing
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
                submissions: low.launches.len(),
                blocks: slow.blocks,
                parsed: slow.parsed,
                tiered: slow.tiered,
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
        // A disagreement here is USUALLY the batching, but it is not necessarily
        // the batching, and asserting straight through would have said the wrong
        // thing. Both mixtures of experts disagree at a 64-row prefill -- and so
        // do two runs of the REFERENCE against each other. `route_sort` builds its
        // permutation through workgroup-scoped atomics, so which of two rows
        // wanting the same expert lands first is whichever lane won the atomic;
        // the gather then writes the same rows to different offsets. One row
        // (every decode here) has nothing to tie, which is why this only appears
        // at a prefill and only in a mixture.
        //
        // So the reference is run AGAIN, and only when the plan proves itself
        // deterministic is byte equality the right claim. This costs nothing when
        // the plan agrees, and it is the difference between "the barrier is
        // wrong" and "the plan has no single answer".
        if differ.is_some() {
            // FOUR MORE RUNS, NOT ONE, and the reason is a flake this suite
            // actually produced. A single re-run was enough for a long time and
            // then failed once in the full suite and passed alone: two draws
            // from a nondeterministic plan can agree by luck, and when they did
            // this reported "the batching is what differs" about a plan whose
            // own answer is not a single answer. One sample cannot distinguish
            // "deterministic" from "did not vary this time".
            //
            // Four costs nothing on a plan that agrees -- this branch is only
            // reached when the batched and unbatched runs already differ, which
            // is the two mixtures at a prefill -- and it makes the wrong
            // conclusion take four coincidences instead of one.
            let mut itself = None;
            let mut samples = Vec::new();
            for _ in 0..4 {
                device
                    .write(&arena_buffer, &fill)
                    .expect("the arena resets");
                driver_vulkan::serve::fire(
                    device,
                    &mut cache,
                    &modules,
                    &low,
                    driver_vulkan::serve::Fire {
                        one_at_a_time: true,
                        ..firing
                    },
                )
                .unwrap_or_else(|e| panic!("{e}"));
                samples.push(device.read(&arena_buffer).expect("the arena reads back"));
            }
            // Every sample against the first run AND against each other: a plan
            // that alternated between two answers would agree with itself
            // pairwise in one ordering and not another.
            for a in std::iter::once(&one_at_a_time).chain(samples.iter()) {
                for b in samples.iter() {
                    itself = itself.or_else(|| a.iter().zip(b).position(|(x, y)| x != y));
                }
            }
            assert!(
                itself.is_some(),
                "{name}: the recorded plan and the submitted plan disagree at byte {:?} of \
             the arena, and five runs of the submitted plan agree with each other, so the \
             batching is what differs",
                differ
            );
            // It reproduces, so it is the plan. Pinned to the two it was measured
            // on: a dense text that started disagreeing would be a real defect
            // and this must not absorb it.
            assert!(
                name.contains("30b_a3b/Prefill") || name.contains("gpt_oss_20b/Prefill"),
                "{name}: disagrees with ITSELF at byte {itself:?}, which was only ever true \
             of a mixture of experts at a prefill"
            );
        }
    }
    // Skipped only by a caller that is comparing two PLANS against each other
    // rather than one plan against itself, and that pays for its own
    // reference. Anything else runs it: it is the crate's only check that
    // batching a command buffer does not change an answer.
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
            ..firing
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
    Ran {
        fired: low.launches.len(),
        answer,
        kernels: low.kernels.clone(),
    }
}

/// What a caller wants one `whole_plan` run to be.
///
/// A struct rather than five more parameters, and the fields are the axes that
/// were each once hardcoded: the class and the row count were `Decode` and 1,
/// the weight byte was 0, and the reference run was unconditional.
#[derive(Clone, Copy)]
struct Wanted {
    /// Which trace the plan comes from.
    class: model_ir::trace::FireClass,
    /// How many rows the fire has.
    rows: usize,
    /// How many bytes the `embed` blocks get.
    ///
    /// Everything else is four mebibytes, which covers every projection in
    /// these models. `embed` is the vocabulary and is also the tied head, and
    /// four mebibytes does NOT cover it: qwen3-0.6B is 151936 rows of 1024 at
    /// four bits, so the head reads 74 mebibytes past the end. That is
    /// invisible on the plumbing runs -- this card returns zero for an
    /// out-of-bounds storage read, `TokenIds` is all zeros so the gather wants
    /// row 0, and a distribution of zeros still records. It is fatal to a
    /// numeric run: the first thing the comparison below found was every logit
    /// at `-0`.
    embed: usize,
    /// Whether the weight blocks hold numbers rather than zeros.
    ///
    /// Zeros for the plumbing runs, where the sizes are a guess and the point
    /// is that every rectangle records. **Numbers for anything that compares
    /// NUMBERS**: with zero weights an affine dequantisation is a constant and
    /// two different matmul kernels agree without computing anything.
    ///
    /// What "numbers" means is not one byte repeated, and that took two
    /// attempts to get right. A block is either packed four-bit data, where
    /// any byte is a legal pair of nibbles, or it is bfloat16 -- the scales,
    /// the zero points and every norm weight -- where most byte patterns are
    /// not usable. `0xA3` repeated makes a scale of `-2^-56`, and every logit
    /// came back `-0`. So the bfloat16 blocks get `0x3F3F`, which is about
    /// 0.75, and the packed blocks get a varying pattern, which is what makes
    /// this comparison able to see a matmul that transposed its operands.
    weights: bool,
    /// Whether to run the plan a second time one dispatch per submission and
    /// require the two to agree.
    ///
    /// The expensive half of this helper. Off only for a caller that is
    /// already firing two plans and comparing those.
    compare: bool,
    /// The bytes a real checkpoint holds, by the name the text binds.
    ///
    /// `None` for every run whose weights are the invented ones above, which
    /// is most of them: a real checkpoint is 335 megabytes off disk and this
    /// helper is called eighteen times.
    ///
    /// When it is `Some`, it overrides BOTH the fill and the block size, so
    /// `embed` above stops mattering -- a real tensor knows how large it is.
    /// Every name the text binds must be in it; a miss is a panic and not a
    /// fallback, because falling back to a guessed block is exactly how an
    /// undersized weight would read as a working load on a card that returns
    /// zero for an overrun.
    real: Option<&'static std::collections::BTreeMap<String, Vec<u8>>>,
}

/// What one `whole_plan` run did.
struct Ran {
    /// Rectangles submitted.
    fired: usize,
    /// The distributions the batched run left in the arena.
    answer: driver_vulkan::serve::Logits,
    /// The distinct symbols the lowering stated.
    ///
    /// Returned so a caller comparing two plans can prove they were two
    /// plans. Without it the cross-kernel comparison below would pass if
    /// `whole_plan` ignored its class and fired the same thing twice, which is
    /// exactly the mistake it exists to rule out.
    kernels: Vec<String>,
}

/// The tiled GEMM and the matrix-vector kernel answer the same prompt the same
/// way.
///
/// `Serving` picks a plan by row count, and that is only sound if the two
/// plans compute the same function. They do not run the same kernels: traced
/// at `FireClass::Prefill` and lowered at sixteen rows, this text states
/// `affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32` and its residual twin where
/// the decode trace states `affine_qmv_fast`. Same weights, same activations,
/// different code -- and nothing in this crate had ever asked whether they
/// agree.
///
/// It is the only NUMERIC check on a matmul anywhere here. Every other whole-
/// plan test compares a plan against itself and so measures ordering; the two
/// host-reference tests are both norms. A matmul that transposed its operands
/// would pass all of them.
///
/// # Why the weights are not zero
///
/// Every other run of `whole_plan` fills each weight block with zeros, because
/// `Arg::Weight` states no width and the sizes are a guess. That is fine for
/// plumbing and useless here: an affine dequantisation of a zero block is a
/// constant, and two matmuls of a constant matrix agree whatever they do with
/// it. So this fills with `0xA3`, and the check below that the distributions
/// are not all one value is what says the fill did its job.
#[test]
fn the_tiled_gemm_answers_the_way_the_vector_kernel_does() {
    use model::shared::llama_like::forward::facts::LlamaLikeFacts;

    let (device, dir) = gpu!();
    for (text, facts, embed, moe) in [
        ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b(), 96 << 20, false),
        // A mixture of experts as well, so the comparison runs with a router,
        // a sort and a gather between the two matmuls rather than a straight
        // residual chain.
        //
        // It does NOT reach the routed GEMM, and that was the reason it was
        // added, so the correction belongs here rather than in a tidier
        // version of the comment. `kernels-vulkan` has `affine_qmm_t_routed`
        // and `mxfp4_qmm_t_routed_bias`; this text at sixteen rows states
        // neither. Its two tiled GEMMs are the same dense
        // `affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32` pair the small qwen
        // states -- the attention projections -- and the expert matmuls stay
        // `affine_qmv_routed` at every row count. So the routed GEMMs have no
        // vector-kernel counterpart in any plan this crate can lower, and
        // there is nothing to compare them against. Recorded as a gap, not
        // closed by relabelling this text as one.
        (
            "qwen3_30b_a3b",
            LlamaLikeFacts::qwen3_30b_a3b(),
            192 << 20,
            true,
        ),
    ] {
        gemm_agrees(&device, dir, text, &facts, embed, moe);
    }
}

/// One text's two matmul kernels, fired and compared.
fn gemm_agrees(
    device: &Device,
    dir: &std::path::Path,
    text: &str,
    facts: &model::shared::llama_like::forward::facts::LlamaLikeFacts,
    embed: usize,
    moe: bool,
) {
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model_ir::trace::FireClass;

    let metal = LlamaLikeMetalFacts::synthetic();
    // THE TILE, and not four and not a round number. Measured over 1, 2, 3,
    // 4, 5, 8, 16, 17, 32 and 64 rows: the prefill plan lowers to zero `qmm`
    // symbols below the tile and two at the tile and above. At four rows this
    // test would fire the same kernels twice and pass without comparing
    // anything -- which is what it silently began doing at sixteen rows the
    // day `QMM_TILE` widened to 32, and is why it is read from the constant
    // now instead of transcribed.
    let rows = model::shared::llama_like::project::QMM_TILE.0 as usize;
    let mut answers = Vec::new();
    for class in [FireClass::Decode, FireClass::Prefill] {
        answers.push(whole_plan(
            device,
            dir,
            text,
            facts,
            &metal,
            Wanted {
                class,
                rows,
                // Over what the head reads: the vocabulary is the one weight
                // in these texts a four-mebibyte guess does not cover.
                embed,
                weights: true,
                compare: false,
                real: None,
            },
        ));
    }
    // TWO PLANS, proved before anything is compared. A helper that ignored
    // its class would fire the same lowering twice and every claim below would
    // hold vacuously.
    assert!(
        !answers[0].kernels.iter().any(|k| k.contains("qmm")),
        "{text}: the decode plan stated a tiled GEMM, so there is nothing here to compare"
    );
    assert_eq!(
        answers[1]
            .kernels
            .iter()
            .filter(|k| k.contains("qmm"))
            .count(),
        2,
        "{text}: the prefill plan stated {:?}, which is not two tiled GEMMs",
        answers[1].kernels
    );
    // WHICH pair, and not just how many. This assertion is why the note above
    // says the routed GEMM is not reached: it was written expecting two routed
    // symbols from the mixture of experts and got the dense pair instead.
    assert_eq!(
        answers[1].kernels.iter().any(|k| k.contains("route_sort")),
        moe,
        "{text}: whether it routes is not what this test was told"
    );
    assert_eq!(
        answers[1]
            .kernels
            .iter()
            .filter(|k| k.contains("qmm") && k.contains("routed"))
            .count(),
        0,
        "{text}: it states a routed GEMM after all, so the note above is wrong and \
         this comparison is covering a kernel it was written to say it does not"
    );
    let (vector, tiled) = (&answers[0].answer, &answers[1].answer);

    assert_eq!(vector.rows, rows, "every row samples");
    assert_eq!(tiled.rows, vector.rows, "the same fire, two plans");
    assert_eq!(
        tiled.vocab, vector.vocab,
        "the vocabulary is not the kernel's"
    );

    // The comparison must be able to fail, and a distribution of one repeated
    // value cannot. This is the control on the weight fill: with zeros it
    // fires.
    let first = vector.values[0];
    assert!(
        vector.values.iter().any(|v| *v != first),
        "{text}: every logit is {first}, so the two kernels agree about nothing"
    );
    assert!(
        vector.values.iter().all(|v| v.is_finite()),
        "{text}: the reference run produced a non-finite logit, so nothing below means anything"
    );

    // RELATIVE, and not byte equality. The two kernels reduce the same
    // products in different orders -- the tile accumulates sixteen rows at a
    // time -- and every intermediate here is bfloat16, which has eight bits of
    // mantissa. Byte equality was tried first and fails on the first row.
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (a, b)) in vector.values.iter().zip(&tiled.values).enumerate() {
        assert!(
            b.is_finite(),
            "{text}: the tiled run produced a non-finite logit at {i}"
        );
        let scale = a.abs().max(b.abs()).max(1e-3);
        let off = (a - b).abs() / scale;
        if off > worst {
            worst = off;
            at = i;
        }
    }
    // Measured: worst is ZERO. Not "close" -- the two kernels agree bit for
    // bit on all 16 * 151936 logits, which is worth stating because it is
    // stronger than this test asks for and because a future card where it
    // stops being exactly zero should not be read as a defect. The tolerance
    // stays at five percent for that reason.
    assert!(
        worst < 0.05,
        "{text}: the two kernels disagree by {worst} at logit {at}: the vector kernel says {} and \
         the tiled one says {}",
        vector.values[at],
        tiled.values[at]
    );

    // AND THE COMPARISON CAN FAIL. A run whose weights are the zeros every
    // other `whole_plan` caller uses gives logits of `-0`, and the same
    // measure puts it at 1.0 -- twenty times the tolerance. Without this the
    // agreement above would be evidence of nothing: a `worst` that came out
    // zero because both sides were zero reads exactly the same.
    let vacuous = whole_plan(
        device,
        dir,
        text,
        facts,
        &metal,
        Wanted {
            class: FireClass::Prefill,
            rows,
            embed,
            weights: false,
            compare: false,
            real: None,
        },
    )
    .answer;
    let apart = vector
        .values
        .iter()
        .zip(&vacuous.values)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-3))
        .fold(0.0f32, f32::max);
    assert!(
        apart > 0.05,
        "{text}: a run with no weights at all is within {apart} of the real one, so this \
         comparison would not have noticed a wrong kernel"
    );
}

/// A routed prefill answers the same twice, though its arena does not.
///
/// This test has now been wrong in both directions, and both corrections are
/// kept because the pair is the actual finding.
///
/// It began by asserting that two runs of one routed fire leave DIFFERENT
/// arenas and the same logits, hunting up to five runs for a difference. Then
/// upstream fixed a real race -- `route_sort`'s `n` was `n_experts * k` where
/// `expert_ids` holds `tokens * k`, so the kernel scanned 112 entries past
/// its region into the `perm` it was concurrently writing -- and the hunt
/// stopped finding anything. So the claim was inverted to byte equality, on
/// the reading that a fixed kernel is a reproducible one, and it passed three
/// times.
///
/// It was luck. The arenas are NOT equal, and the reason is the one this
/// suite had written down all along, in `whole_plan`'s own comment: the
/// permutation is built with workgroup-scoped atomics, so which of two rows
/// wanting the same expert lands first is whichever lane won the atomic. That
/// is a design, not a defect -- the same bump-counter every backend's router
/// uses -- and it survives the fix, because the race and the ordering were
/// two separate things sharing one symptom.
///
/// Measured, when the byte claim finally failed: 75k to 198k of 6.2M arena
/// bytes differ, run to run, and 141 bytes of `route_sort`'s own 512-byte
/// `perm` region are among them. That last number is what makes this an
/// ordering rather than an overrun -- the difference is INSIDE the buffer the
/// router owns, not past it -- and it is why a byte comparison cannot be the
/// claim here.
///
/// So the claim is what the ordering cannot move: the ANSWER. That was first
/// spelled as every row's argmax agreeing, and it was wrong a THIRD time --
/// which is the third correction kept here.
///
/// It flaked, about one run in three, always on row 0. The diagnosis: this
/// fixture's logits sit near 9728, where bf16's seven mantissa bits space the
/// representable values exactly 64 apart, and 3271 different tokens held the
/// top value. A plateau that wide makes an argmax a report of scan order, and
/// the run-to-run difference that reshuffled it was 64 -- one ULP, the
/// smallest difference expressible there. The test called that "the router's
/// ordering reached the ranking".
///
/// Two things came out of chasing it. The fixture's weight fill was
/// `i * 31 % 251`, whose 251-byte period aliases with every one of these
/// models' head row strides, so hundreds of vocabulary rows were byte
/// identical; `whole_plan`'s `fill_of` explains the hash that replaced it.
/// And the ULP itself is not the driver's ordering: it survives
/// `one_at_a_time`, which submits every dispatch on its own fence and is the
/// strongest ordering Vulkan has. What it was, was the router being handed
/// tied scores by a periodic fixture and breaking the tie with an atomic.
///
/// The claim now is EQUALITY OF THE WHOLE DISTRIBUTION, which is strictly
/// stronger than the argmax it replaced and cannot go quiet on a plateau.
/// The relative bound stays beside it as the weaker, more legible statement
/// of the same thing.
///
/// What this no longer catches is an overrun that changes the arena without
/// changing the answer. That is `whole_plan`'s job and it does it properly:
/// it re-runs the reference against itself precisely so it can tell "the
/// batching is wrong" from "the plan has no single answer", which is the
/// distinction this test spent two revisions failing to make.
#[test]
fn a_routed_prefill_answers_the_same_twice() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model_ir::trace::FireClass;

    let (device, dir) = gpu!();
    let facts = LlamaLikeFacts::qwen3_30b_a3b();
    let metal = LlamaLikeMetalFacts::synthetic();
    let want = Wanted {
        class: FireClass::Prefill,
        rows: 16,
        embed: 192 << 20,
        weights: true,
        compare: false,
        real: None,
    };
    let once = whole_plan(&device, dir, "qwen3_30b_a3b", &facts, &metal, want);
    let twice = whole_plan(&device, dir, "qwen3_30b_a3b", &facts, &metal, want);

    // The precondition: this text really does route. Without it the claim
    // below is about a dense model and says nothing.
    assert!(
        once.kernels.iter().any(|k| k.contains("route_sort")),
        "this text states no router, so there is no permutation to shuffle"
    );
    assert!(
        once.answer.values.iter().all(|v| v.is_finite()),
        "a non-finite logit makes both comparisons below vacuous"
    );
    let first = once.answer.values[0];
    assert!(
        once.answer.values.iter().any(|v| *v != first),
        "every logit is {first}, so two runs agreeing means nothing"
    );

    // The comparisons below are over a distribution, so they pass for free on
    // one that is empty or one row wide. A 16-row fire is neither, and saying
    // so here is what keeps the claim from going quiet if `whole_plan` ever
    // stops reading the logits back.
    let rows = once.answer.rows.max(1);
    assert_eq!(
        rows, 16,
        "this fire was asked for 16 rows and answered {rows}, so the per-row \
         comparison below covers something other than the prefill"
    );
    assert_eq!(
        once.answer.values.len(),
        twice.answer.values.len(),
        "two fires of one plan produced distributions of different sizes"
    );
    let per = once.answer.values.len() / rows;
    assert!(
        per > 1000,
        "a {per}-wide row is not a vocabulary, so an argmax over it is not the \
         choice a caller makes"
    );

    // BIT FOR BIT, and the argmax check this replaced is the reason it has
    // to be.
    //
    // These logits sit near 9728, and bf16 has seven explicit mantissa bits,
    // so the representable values there are spaced exactly 64 apart. The
    // consequence is a plateau: 3271 different tokens hold the top value at
    // the same bf16 bucket. An argmax over that reports which index the scan
    // reached first, and a difference of ONE ULP anywhere reshuffles it --
    // which is exactly what used to happen, on row 0 only, in about one run
    // in three. The test named that flip "the router's ordering reached the
    // ranking" and it was a rounding difference of one bf16 step.
    //
    // Equality of the whole distribution is the claim that was wanted. It is
    // strictly stronger than the argmax one -- a router whose permutation
    // reached the arithmetic changes bits long before it changes a winner --
    // and it cannot go quiet on a plateau. It holds because this fire has no
    // order-dependent accumulation left to expose once the fixture stops
    // handing the router tied scores.
    assert_eq!(
        once.answer.values, twice.answer.values,
        "two fires of one routed plan produced different logits. The plan, the
         weights and the arena are identical, so what differs is an ordering:
         either the router's permutation reached the arithmetic, or an expert's
         contribution was accumulated in a different sequence"
    );

    // THE DISTRIBUTION. The blunt half, kept from the original: an argmax can
    // survive a distribution that has drifted badly, so the values are held
    // too.
    let worst = once
        .answer
        .values
        .iter()
        .zip(&twice.answer.values)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-3))
        .fold(0.0f32, f32::max);
    assert!(
        worst < 0.05,
        "the same routed prefill answered differently twice, by {worst}: the router's \
         nondeterminism reaches the distribution and not just the arena it is \
         allowed to reorder"
    );
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
/// `attn/kv_write.slang`. `attn/sdpa_paged.slang` computes the same expression
/// from separate source, so the layout is two modules agreeing rather than
/// this crate deciding.
#[test]
fn a_real_plans_kv_append_puts_the_row_where_the_page_table_says() {
    use driver_vulkan::binding::FireTable;
    use driver_vulkan::resources::{Pool, Shape};
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire, Row, lower};
    use model_ir::trace::FireClass;

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
            sig: None,
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
/// `attn/kv_write.slang` puts six positions into a pool through the page table,
/// and `attn/sdpa_paged.slang` attends over them without either shader or this
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
/// `attn/kv_write.slang` compiles to two shaders from one file. The paged one
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
/// The other direction of the same two numbers. `attn/sdpa_vector.slang` never
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
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
        },
        Request {
            positions: (0..3).collect(),
            pages: vec![6, 1],
            samples: Vec::new(),
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
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
/// `rope/neox.slang` compiles to two shaders from one file. `neox_mb` raises
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
    use model_ir::trace::FireClass;

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
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
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
            sig: None,
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
/// planned and their scalars gathered. Withholding the first would refuse at
/// launch 0, which is the only case that cannot have taken anything.
///
/// # The free, and how it came to be checked here
///
/// That the block buffer is given back on the refusing path used to be stated
/// in the code and asserted nowhere, because it could not be made to fail.
/// Replacing the free with `std::mem::forget` and firing fifty refusals still
/// left the device able to allocate 64 MiB afterwards; the blocks are small
/// and this card has twenty-four gigabytes. Finding the ceiling directly was
/// worse: allocating small buffers in a loop until one was refused did not
/// finish in ten minutes.
///
/// What was missing was not a bigger leak but a witness. `Device` now counts
/// what it hands out and what it takes back, so the claim is one subtraction
/// -- see the assertion after the successful fire -- and `std::mem::forget`
/// fails it immediately. The ordering of the free against the recorded
/// buffers is still the borrow checker's: moving it one line earlier does not
/// compile.
#[test]
fn a_fire_that_cannot_run_says_which_launch() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Fire as LowerFire, Row, lower};
    use model_ir::trace::FireClass;

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
    // What the device holds before any of this, so the three fires below can
    // be asked whether they gave back what they took. See the module note on
    // why this could not be stated until the device counted.
    let held = device.live_buffers();
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

    // Two refusals deep in the plan and one whole fire, and the device is
    // holding exactly what it held before them. A `fire` that kept its block
    // buffer would pass every assertion above -- this card has twenty-four
    // gigabytes and the block is tens of kilobytes, so nothing downstream
    // would ever notice. Measured: `std::mem::forget` in place of the free
    // fails this line by one buffer.
    //
    // The two refusals here are pass-one refusals, which return before the
    // allocation happens and so have nothing to give back; it is the
    // successful fire that takes and returns. Both are under the same claim
    // on purpose, because which refusals come before the allocation is a
    // property of the code and not of the plan, and this line does not have
    // to be revisited when that moves.
    assert_eq!(
        device.live_buffers(),
        held,
        "two refusals and a fire left buffers behind"
    );

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
    use model_ir::trace::FireClass;

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
        mask: Vec::new(),
        traced: false,
        writes: Vec::new(),
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
    use model_ir::trace::FireClass;

    let (device, dir) = gpu!();
    let shape = Shape {
        layers: 28,
        kv_heads: 8,
        head_dim: 128,
        page_size: 8,
        // EIGHT, and it was six. The prefill below is a whole `QMM_TILE` of
        // rows and the tile widened from 16 to 32, so that one turn now needs
        // four pages instead of two and the mixed batch after it had none
        // left. Sixty-four token slots rather than forty-eight; nothing here
        // asserts on the count, only that the seatings below fit.
        pages: 8,
        bytes: 2,
    };

    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let prefill_plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Prefill,
    );
    let serving = Serving {
        plan: &plan,
        prefill: &prefill_plan,
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
        // BOTH plans. The prefill plan states tiled GEMMs where the decode
        // plan states matrix-vector products, and a `Model` is a pair rather
        // than a fallback chain, so a name only the prefill wants is a refusal
        // at the first many-row step. Collected from the union rather than
        // discovered by a failure.
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
        let probe_prefill = model_compiler::lower::lower(
            &prefill_plan,
            &vec![model_compiler::lower::Row::default(); 16],
            model_compiler::lower::Fire {
                captures_across_splits: false,
            },
        )
        .expect("the prefill plan lowers");
        let mut names = names;
        names.extend(probe_prefill.args.iter().filter_map(|a| match a {
            model_compiler::lower::Arg::Weight(n) => Some(n.as_str()),
            _ => None,
        }));
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

    let mut lowerings = driver_vulkan::turns::Lowerings::default();
    let mut held = Held {
        book: &mut book,
        pool: &mut pool,
        weights: &weights,
        lowerings: &mut lowerings,
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
    // ONE more, and exactly one: `sdpa_paged_tiled`. This step lowers the
    // PREFILL plan and the first lowered the decode plan, and the only thing
    // the two disagree about at two rows is which attention kernel they
    // reach. The projections do NOT diverge here -- the tiled GEMM has
    // `bm = 16`, so a prefill at fewer than sixteen rows lowers to exactly
    // the matrix-vector products a decode does. Two rows, three, eight -- all
    // `affine_qmv_fast`; at sixteen two `affine_qmm_t` symbols appear and
    // stay, which is what the sixteen-row step below is for.
    //
    // This was `second.pipelines == first.pipelines` while the two
    // `sdpa_paged_tiled` rows stated nothing but their axes and `dsl::sdpa`
    // could only reach the decode kernel. The count going up by one is the
    // evidence the tiled kernel is now actually selected for a prefill.
    assert_eq!(
        second.pipelines,
        first.pipelines + 1,
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
    // The decode shape again, so nothing new: the count is what the prefill
    // step above left it at, not what the first decode step did.
    assert_eq!(third.pipelines, second.pipelines);
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
    // EXACTLY THE TILE, because the divergence this test ends on is that a
    // prefill of a whole tile builds the two GEMM pipelines a decode never
    // does. A prompt one row short of the tile takes the matrix-vector arm
    // and builds neither -- so this number is `QMM_TILE`'s and not a round
    // sixteen, which is what it was until the tile widened underneath it.
    let tile = model::shared::llama_like::project::QMM_TILE.0;
    let prefill = serving
        .step(
            &device,
            &mut cache,
            &modules,
            &mut held,
            &[Turn {
                who: 3,
                tokens: (0..tile).map(|t| 7 + t).collect(),
            }],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(prefill.rows, tile as usize);
    assert_eq!(
        prefill.logits.rows, tile as usize,
        "every row samples, so a prefill of a whole tile COMPUTES a tile of distributions"
    );
    assert_eq!(
        prefill.readout_of,
        vec![tile as usize - 1],
        "the turn's answer is its last row"
    );
    assert!(
        prefill.logits.row(prefill.readout_of[0]).is_some(),
        "the turn's own distribution is addressable"
    );

    // ...and the fifteen it computed and nobody asked for did not come back.
    //
    // This is the claim that makes a prefill affordable. The lowering says
    // every row samples -- see the comment above -- so the exit really is
    // sixteen distributions wide in the arena, and a 1024-token prompt's exit
    // is 155 million values. Reading all of them cost 0.5 s a step through
    // the copy engine and thirty seconds through the mapping; the turn wants
    // one row of it.
    //
    // Stated as two separate facts because either alone is satisfiable
    // wrongly: `read` alone could name a row `values` does not hold, and a
    // short `values` alone could be a truncated dense read, which would give
    // every request after the first someone else's distribution.
    assert_eq!(
        prefill.logits.read, prefill.readout_of,
        "the rows a step holds are the rows its requests read"
    );
    assert_eq!(
        prefill.logits.values.len(),
        prefill.logits.vocab,
        "a sixteen-row prefill answering one request came back {} values wide, \
         so it is carrying rows nothing will ever address",
        prefill.logits.values.len()
    );
    assert_eq!(held.book.tokens(3), Some(tile as usize));
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
            (0..tile).collect::<Vec<u32>>(),
            "the gather would read {} entries out of {} the pool holds",
            prefill.rows,
            got.len()
        );
    }
    // AND HERE THEY DIVERGE. This step is a whole tile of rows, so it is
    // the first in the crate's serving loop to lower `affine_qmm_t` -- the
    // tiled GEMM, which a deployment carrying only the decode plan would never
    // build and never run. Two of them, measured, and they are two pipelines
    // the three steps before this one did not have.
    //
    // This assertion used to read `prefill.pipelines == first.pipelines`, with
    // the message "a prefill built a pipeline no decode needed". It passed,
    // and it was passing because `Serving` held one plan: the prefill was
    // being answered with sixteen matrix-VECTOR products. The claim was true
    // and it was the wrong claim.
    // Three, not two: the tiled GEMM's two plus `sdpa_paged_tiled`, which the
    // two-row prefill above already built and the one-row decode `first`
    // never does.
    assert_eq!(
        prefill.pipelines,
        first.pipelines + 3,
        "a {tile}-row step built {} pipelines a one-row step had not, and the tiled \
         GEMM is two of them",
        prefill.pipelines - first.pipelines
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

    // SIX MORE DECODES OF A SHAPE ALREADY SEEN, so that the counts below are
    // claims about REPETITION and not about how many shapes this test happens
    // to fire. Without them, disabling the lowering cache moves its number by
    // one -- which a reader could take for a rounding argument. With them it
    // moves by seven.
    let restaged = held.pool.restaged();
    for _ in 0..6 {
        serving
            .step(
                &device,
                &mut cache,
                &modules,
                &mut held,
                &[Turn {
                    who: 1,
                    tokens: vec![9],
                }],
            )
            .unwrap_or_else(|e| panic!("a repeat decode: {e}"));
    }

    // AND THE LOWERINGS WERE NOT RECOMPUTED. This test fired thirteen steps
    // above -- decodes of one row, a prefill of a whole tile, a mixed batch
    // of four rows, and the refusals, which never reach the lowering. Four
    // DISTINCT row shapes reached it, and `lower` ran four times.
    //
    // Stated as a count and not as a duration because a cache that never hits
    // returns exactly the same answers as one that always does; the only
    // difference is 1.38 ms a step, and a wall-clock assertion on a shared
    // box measures the neighbours. See `turns::Lowerings` for why the row
    // shape is a sound key.
    // AND THE STATE TABLES WERE WRITTEN OVER, NOT REALLOCATED.
    //
    // A step writes eleven tables -- the nine `Pool::stage` fills plus the
    // token ids and the sampling identity `Serving::once` states -- and
    // `Pool::state` used to allocate a fresh buffer for each and free the old,
    // sixty-six allocations and sixty-six frees over these six steps, measured at 1.30 ms a step of
    // an 8.1 ms decode. A conversation decoding states the same row count and
    // the same page count for eight tokens at a stretch, so step after step
    // the tables are the same SIZE and can be written over in place.
    //
    // NINE, and each one is a table whose size genuinely changed:
    //
    // * eight at the first of these steps, because the step before it was the
    //   four-row mixed batch and eight of the eleven tables are a different
    //   length at four rows than at one. The two attention-mask tables are
    //   not: they are a byte a row rounded up to a word, so one row and four
    //   rows are both one word;
    // * one more partway through, when this conversation crossed a page
    //   boundary and `kv_page_indices` grew by a word.
    //
    // The other fifty-seven writes went into the buffer already there.
    // Counted and not timed for the reason `Pool::restaged` gives.
    assert_eq!(
        held.pool.restaged() - restaged,
        9,
        "six repeated decodes allocated {} table buffers",
        held.pool.restaged() - restaged
    );

    // `held` borrows `lowerings` mutably and every step above needs it, so
    // the count is read only once nothing will step again. Not `drop`, which
    // clippy rightly points out does nothing to a type with no `Drop`: this
    // is a borrow ending, and `let _ =` is how a reader is told so.
    let _ = held;
    assert_eq!(
        lowerings.lowered(),
        4,
        "the deployment lowered {} times over four distinct row shapes",
        lowerings.lowered()
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

/// The bytes a real checkpoint holds, keyed by the name the text binds.
///
/// `None` when `PIE_CHECKPOINT` names nothing readable, or names something
/// that is not the qwen3-0.6B this fixture is written against. Leaked, because
/// `Wanted::real` is `'static` and because this is read once per process and
/// then held for the life of the suite rather than copied per call.
///
/// # Why this is a straight read and not an executor
///
/// The load plan for a `Binding::MLX_IN_PLACE` target is, measured: 704
/// `Allocate`, 704 `Finalize`, and **six** `BulkExtentWrite`s whose sources
/// tile the whole file from its header to its end at `dst = src - 78296`. So
/// the staging is a verbatim copy of the checkpoint, and a tensor's bytes are
/// exactly its `SourceTensorDecl`'s `[file_offset, +span_bytes)`. Nothing here
/// interprets a transform, and the moment a target asks for one this returns
/// the wrong bytes -- so it asserts that those six are all the writes there
/// are, rather than assuming it.
fn checkpoint_weights() -> Option<&'static std::collections::BTreeMap<String, Vec<u8>>> {
    weights_of(&REALS[0])
}

/// A checkpoint this file can serve, and the text it belongs to.
///
/// Mirrors `tests/checkpoint.rs`'s table -- an integration test cannot import
/// another one -- and carries the two extra facts a FIRE needs and a name
/// comparison does not: the cache shape and the block the seam must be.
struct Real {
    /// The catalog row, taken by id rather than by `catalog::identify`. See
    /// `tests/checkpoint.rs` for the refusal that makes that necessary.
    id: &'static str,
    /// The forward-facts fixture whose text states the names.
    facts: fn() -> model::shared::llama_like::forward::facts::LlamaLikeFacts,
    /// `model.embed_tokens.weight`'s packed shape, which is how a snapshot
    /// says which model it is. Guessing wrong reports the FIXTURE's names as
    /// missing, which reads like a loader defect.
    embed: &'static [i64],
    /// What an independent implementation says this model answers to the
    /// prompt.
    oracle: Oracle,
    /// ...and to the prompt with that answer appended, which is the row the
    /// driver's first DECODE fire produces.
    decoded: Oracle,
}

/// A CPU forward's answer to the [`PERIOD`] prompt, in enough detail to hold a
/// whole distribution against and not so much that it is a golden file.
///
/// Produced by a numpy forward that reads the safetensors directly and
/// dequantizes MLX's 4-bit groups itself -- no code, no kernel and no crate in
/// common with this one. `files/qwen-cpu-reference.py` in the session that
/// wrote this; it is ninety lines and can be written again from the config.
///
/// Eight ranked ids and five fixed indices rather than 151_936 logits: a
/// golden vector of the whole row would be a file nobody could check by
/// reading, and the two things worth pinning are WHICH tokens win and whether
/// the numbers away from the peak are the same numbers.
struct Oracle {
    /// The eight highest-scoring ids, in order.
    top: &'static [u32],
    /// Their logits.
    vals: &'static [f32],
    /// The logits at ids 0, 1_000, 50_000, 100_000 and 151_935 -- chosen for
    /// being spread across the vocabulary and nothing else. Away from the
    /// peak, so a driver that got the argmax right by luck does not.
    probe: &'static [f32],
    /// The row's whole range, which no single logit states.
    span: f32,
}

/// The two real models this file has weights for.
///
/// The second is not a duplicate: qwen2.5 is a different generation with a
/// different role set (no qk-norm), two kv heads instead of eight, and an mlp
/// wide enough that 84 of its weights overflow the block the junk-weight tests
/// hold names under. A driver that had specialised to qwen3 fires this one
/// wrong.
const REALS: &[Real] = &[
    Real {
        id: "qwen3-0.6b",
        facts: model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b,
        embed: &[151_936, 128],
        oracle: Oracle {
            top: &[88_204, 33_032, 62_949, 14, 78_329, 42_746, 57_428, 17_521],
            vals: &[
                20.8004, 15.7309, 15.5539, 15.2734, 15.2257, 14.8423, 14.4461, 14.2924,
            ],
            probe: &[6.3329, -2.3533, -1.5004, 2.0615, 0.1445],
            span: 31.192,
        },
        decoded: Oracle {
            top: &[
                6_100, 16_997, 25_948, 18_062, 6_094, 20_405, 101_203, 65_069,
            ],
            vals: &[
                23.9178, 16.0206, 15.9419, 15.8716, 15.5314, 15.1679, 14.9816, 14.824,
            ],
            probe: &[7.1273, -1.677, 0.4846, 0.6105, 0.953],
            span: 36.8039,
        },
    },
    Real {
        id: "qwen2.5-1.5b",
        facts: model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen2_5_1_5b,
        embed: &[151_936, 192],
        // Run WITH the attention biases, which is now the text this driver is
        // given. It was the biasless run for four commits -- qwen2.5 ships 84
        // bias tensors, the shared Metal text stated none of them, and the
        // driver reproduced that text exactly. See
        // `a_second_real_model_is_served_the_way_the_text_states_it` for what
        // closing it took and why the biasless numbers are still recorded
        // there.
        oracle: Oracle {
            top: &[88_204, 14, 42_746, 271, 62_949, 10_360, 8_680, 5_894],
            vals: &[
                22.8112, 18.8294, 16.9528, 16.2228, 16.0305, 15.8535, 15.6435, 15.4428,
            ],
            probe: &[6.633, 4.047, -1.8166, -0.2853, 0.6392],
            span: 32.904,
        },
        decoded: Oracle {
            top: &[6_100, 16_997, 271, 25_948, 83_646, 927, 198, 4_480],
            vals: &[
                26.3584, 17.8958, 16.4103, 15.7485, 15.3555, 15.3033, 15.2949, 15.1836,
            ],
            probe: &[7.1772, 2.8276, 1.1721, -0.7112, -1.2116],
            span: 40.193,
        },
    },
];

/// The snapshot directories `PIE_CHECKPOINT` names, colon-separated.
fn snapshots() -> Vec<String> {
    match std::env::var("PIE_CHECKPOINT") {
        Ok(v) => v
            .split(':')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// The bytes for one of [`REALS`], from whichever snapshot is that model.
fn weights_of(real: &Real) -> Option<&'static std::collections::BTreeMap<String, Vec<u8>>> {
    use std::sync::OnceLock;
    #[allow(clippy::type_complexity)]
    static HELD: OnceLock<
        std::sync::Mutex<
            std::collections::BTreeMap<
                &'static str,
                Option<&'static std::collections::BTreeMap<String, Vec<u8>>>,
            >,
        >,
    > = OnceLock::new();
    let cache = HELD.get_or_init(Default::default);
    let mut cache = cache.lock().expect("the cache");
    if let Some(held) = cache.get(real.id) {
        return *held;
    }
    let loaded = load_weights(real).map(|m| &*Box::leak(Box::new(m)));
    cache.insert(real.id, loaded);
    loaded
}

/// The read itself, separated so `weights_of` is only the caching.
fn load_weights(real: &Real) -> Option<std::collections::BTreeMap<String, Vec<u8>>> {
    for dir in snapshots() {
        let path = std::path::Path::new(&dir);
        let Ok(meta) = model_loader::checkpoint::read::parse_checkpoint_metadata(path) else {
            continue;
        };
        // The packed width, because the artifact this needs is quantised --
        // the contract refuses a bf16 checkpoint outright. See
        // `tests/checkpoint.rs` for both findings.
        let hidden = meta
            .tensors
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        if hidden != real.embed {
            continue;
        }
        let row = model::catalog::find(real.id)?;
        let config = std::fs::read_to_string(path.join("config.json")).ok()?;
        let encoding = model::encoding::Encoding::from_config_json(&config).ok()?;
        let target = model_loader::plan::StorageTarget::for_backend(
            model_loader::types::BackendKind::Vulkan,
            0,
            1,
        );
        let (plan, _) = model::boot::compile_load_plan_for(
            path,
            &meta,
            &target,
            row,
            &encoding,
            model::boot::Binding::MLX_IN_PLACE,
        )
        .ok()?;
        // THE LOADER'S OWN EXECUTOR, and it took a second model to learn that
        // it was needed.
        //
        // This used to read each tensor's source span out of the file
        // verbatim, on the measured premise that a `Binding::MLX_IN_PLACE`
        // plan is `{Allocate: 704, BulkExtentWrite: 6, Finalize: 704}` whose
        // six writes tile the whole file at `dst = src - 78296`. That premise
        // was asserted rather than assumed -- and qwen2.5 broke it at once:
        // its plan states `{Allocate: 732, BulkExtentWrite: 154, Finalize:
        // 732, TileMap: 535}`. The 535 transforms are what `fused_qkv: true`
        // costs; a verbatim read would have handed the card three separate
        // projections where the text binds one joined weight, which on this
        // card is not a fault but a wrong number.
        //
        // `model_loader::executor::Execution` is a production path -- `pie
        // model convert` materializes artifacts through it -- so running the
        // plan is both less code here and the thing a real driver would do.
        let storage = match model_loader::executor::Execution::new(&plan, path).run() {
            Ok(storage) => storage,
            Err(e) => {
                eprintln!("the loader would not execute `{}`'s plan: {e}", real.id);
                return None;
            }
        };
        let naming = driver_vulkan::names::Naming::mlx();
        let mut out = std::collections::BTreeMap::new();
        for traced in names_a_decode_binds(real) {
            let bytes = naming
                .spellings(&traced)
                .iter()
                .find_map(|s| storage.tensors.get(s.as_str()))
                .unwrap_or_else(|| panic!("`{traced}` resolves to nothing the loader produced"));
            out.insert(traced, bytes.clone());
        }
        return Some(out);
    }
    eprintln!(
        "no snapshot PIE_CHECKPOINT names is the 4-bit `{}` this fixture states",
        real.id
    );
    None
}

/// Every weight name this model's decode plan binds.
///
/// Duplicated from `tests/checkpoint.rs` rather than shared, because an
/// integration test cannot import another one and a `mod` shared between them
/// would drag that file's checkpoint dependencies into every test in this one.
fn names_a_decode_binds(real: &Real) -> Vec<String> {
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_ir::trace::FireClass;

    // The SAME fact set `shelled` builds its text from, and it has to be:
    // `add_bias` decides whether the text states three bias weights a layer,
    // and a loader asked for a shorter list than the shell binds hands the
    // fire an unbound symbol at dispatch time.
    let text = llama_like_metal(
        &(real.facts)(),
        &LlamaLikeMetalFacts {
            add_bias: true,
            ..LlamaLikeMetalFacts::synthetic()
        },
        FireClass::Decode,
    );
    let low = lower(
        &text,
        &[Row::default()],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the plan lowers");
    let names: std::collections::BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) if !n.starts_with("scale.") => Some(n.clone()),
            _ => None,
        })
        .collect();
    names.into_iter().collect()
}

/// A whole plan, fired with the weights a real checkpoint holds.
///
/// # What every other whole-plan test in this file could not say
///
/// They hold one four-megabyte block of invented bytes under all 704 names.
/// That proves the plumbing -- every rectangle records, every barrier is in
/// the right place, two submission shapes agree -- and it proves nothing about
/// the arithmetic, because the numbers going in were never a model's.
///
/// This one reads `mlx-community/Qwen3-0.6B-4bit` off disk, resolves all 704
/// names through [`driver_vulkan::names`], hands each its own tensor at its
/// own real size, and fires. The distribution that comes back is the one this
/// model actually assigns.
///
/// # What it checks, and why not more
///
/// It does not check the logits against a reference implementation, because
/// this crate has none and inventing one would be checking a matmul against a
/// matmul. What it checks is what a wrong load looks like from here:
///
///   - every logit finite, which a bad scale is not (a bf16 exponent read
///     from the wrong half of a word gives infinities within one layer);
///   - the distribution not flat, which is what an all-zero weight gives and
///     what an out-of-bounds read gives on this card;
///   - the same TOKEN from the two matmul kernels, on REAL numbers this time,
///     which is the cross-check that already exists made non-vacuous.
///
/// The zero-weight control fires: with the invented all-zero blocks the
/// distribution spans exactly 0 and the second check reports it.
#[test]
fn a_whole_plan_fires_with_the_weights_a_real_checkpoint_holds() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model_ir::trace::FireClass;

    let (device, dir) = gpu!();
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so a real weight load is unmeasured");
        return;
    };
    assert_eq!(
        real.len(),
        704,
        "the checkpoint answered {} names",
        real.len()
    );

    let facts = LlamaLikeFacts::qwen3_0_6b();
    let metal = LlamaLikeMetalFacts::synthetic();
    // The tile, because that is where the prefill plan starts stating a tiled
    // GEMM -- below it the two classes lower to the same kernels and the
    // comparison at the end would be vacuous. Read from the constant rather
    // than transcribed, for the reason `gemm_agrees` gives at its own `rows`.
    let rows = model::shared::llama_like::project::QMM_TILE.0 as usize;
    let mut answers = Vec::new();
    for class in [FireClass::Decode, FireClass::Prefill] {
        answers.push(whole_plan(
            &device,
            dir,
            "qwen3_0_6b",
            &facts,
            &metal,
            Wanted {
                class,
                rows,
                // Ignored: a real tensor states its own size.
                embed: 0,
                weights: true,
                compare: false,
                real: Some(real),
            },
        ));
    }
    // The premise, same as `gemm_agrees`: two plans, or there is nothing to
    // compare.
    assert!(
        !answers[0].kernels.iter().any(|k| k.contains("qmm")),
        "the decode plan stated a tiled GEMM"
    );
    assert!(
        answers[1].kernels.iter().any(|k| k.contains("qmm")),
        "the prefill plan stated no tiled GEMM"
    );

    for (which, ran) in ["decode", "prefill"].iter().zip(&answers) {
        let logits = &ran.answer.values;
        assert_eq!(
            logits.len(),
            rows * 151936,
            "{which}: {} logits",
            logits.len()
        );
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{which}: a real load produced a value that is not finite"
        );
        // NOT FLAT, which is the failure mode this card hands back silently:
        // an undersized weight reads as zeros, an all-zero weight makes every
        // logit the same, and a distribution of equal numbers still records
        // and still submits.
        let first = &logits[..151936];
        let lo = first.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = first.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi - lo > 1.0,
            "{which}: the whole distribution spans {}, so this is not a model's answer",
            hi - lo
        );
    }

    // AND THE TWO AGREE, on numbers a model actually holds. The existing
    // cross-check fires the same comparison on invented bytes; what this adds
    // is that the agreement survives the dynamic range of real scales, which
    // is where a tiled kernel accumulating in the wrong order would show.
    let (a, b) = (&answers[0].answer.values, &answers[1].answer.values);
    // THE MEASURE IS ABSOLUTE, AND THAT IS A FINDING.
    //
    // The existing cross-check uses a per-element relative difference, and on
    // invented weights it reports exactly 0. On real ones it reports 1.99 --
    // which reads like a broken kernel and is not one. Measured: the largest
    // ABSOLUTE disagreement over all 2_430_976 logits is 0.469, on a
    // distribution spanning 40.4. What the relative measure is reacting to is
    // the 393_717 logits that sit near zero, where dividing by a vanishing
    // magnitude turns a rounding difference into a ratio of two.
    //
    // So the instrument was wrong for this input, not the kernels. A tiled
    // GEMM reduces 1024 bf16 terms in a different order than a vector kernel
    // does, and bf16 carries eight bits of mantissa; a few tenths on a
    // magnitude of twenty is what that costs. Invented weights hid it because
    // every packed block held the same repeating pattern, so every partial
    // sum was the same size and the order stopped mattering.
    let range = {
        let lo = a.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        hi - lo
    };
    let absmax = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        absmax / range < 0.02,
        "the two matmul kernels differ by {absmax} on a distribution spanning {range}"
    );

    // AND THE ONE THAT ACTUALLY MATTERS. A driver's output is a token, and
    // two kernels that agree to a few tenths still disagree about the answer
    // if the top two logits are within those tenths. Measured per row, so a
    // single row's tie does not hide behind fifteen that agree.
    let top = |v: &[f32]| -> Vec<usize> {
        v.chunks(151_936)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .expect("a non-empty row")
                    .0
            })
            .collect()
    };
    assert_eq!(
        top(a),
        top(b),
        "the two matmul kernels choose different tokens"
    );
    eprintln!("real weights: {absmax} apart on a range of {range}, same token every row");
}

/// A real model, shown a pattern, continuing it.
///
/// # Why this and not a prompt
///
/// Everything above fires plans; the one below it fires them with real
/// weights. Neither says the model is THINKING, because both feed token zero
/// and read a distribution nobody interprets. A prompt would need a tokenizer,
/// which this crate has no business owning.
///
/// So the check is one no tokenizer is needed for: **induction**. Show the
/// model a sequence of arbitrary ids, then show it the beginning of the same
/// sequence again, and any transformer that works predicts the continuation --
/// it is the first circuit a language model learns and it is entirely about
/// COPYING, so the ids need not spell anything.
///
/// # What that proves that nothing else here does
///
/// Induction is a claim across positions, so it fails if anything that spans
/// positions is wrong -- and those are exactly the things a single fire cannot
/// check:
///
///   - the rotary angles, since the head must compare position `i` with
///     position `i - period` and a wrong `theta` or a wrong offset destroys
///     the match;
///   - the KV cache's paging, since the earlier occurrence lives in a page
///     written by an earlier fire;
///   - the positions the book hands each step, since an off-by-one restarts
///     the sequence;
///   - the prefill and the decode plans agreeing, since the pattern is
///     prefilled and the continuation is decoded.
///
/// A single wrong one of those still produces finite, non-flat logits.
///
/// # The controls, including the one that did not fire
///
/// Zeroing the weights answers `[151935, 151935, 151935, 151935]` -- the last
/// id in the vocabulary, four times, which is what an argmax over a flat
/// distribution gives. So the claim is about the checkpoint's numbers and not
/// about the plumbing.
///
/// **A wrong rotary theta does NOT break it.** Rebuilding the ladder at
/// 10_000 instead of the 1_000_000 this model was trained with leaves the
/// continuation exactly right.
///
/// # And then it was held against an oracle
///
/// A numpy forward of the same checkpoint -- reading the safetensors directly,
/// dequantizing MLX's 4-bit groups itself, sharing no code and no kernel with
/// this crate -- answers `[88204, 6100, 41777, 2930]` for this prompt, which
/// is argmax-identical to what the card returns over all four greedy steps.
/// So "the pattern" is no longer the only thing this is checked against: an
/// independent implementation agrees token for token. That is a real limit on what this test proves
/// and it is recorded rather than hidden: over 36 positions both bases give
/// angles a head can tell apart, and induction is a copying circuit that
/// matches on CONTENT. So the rotary claim in the list above is the weaker
/// one -- this would catch a ladder that was not applied at all, or one
/// indexed by the wrong position, but not one merely tuned wrongly.
#[test]
fn a_real_model_continues_a_pattern_it_was_shown() {
    let (device, dir) = gpu!();
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so induction is unmeasured");
        return;
    };
    assert_eq!(
        continued(&device, dir, &REALS[0], real, Feeding::Prefilled).tokens,
        PERIOD[2..].to_vec(),
        "the model was shown {PERIOD:?} five times and did not continue it"
    );
}

/// Arbitrary ids, well inside the vocabulary and away from the special tokens
/// at either end.
///
/// What they SPELL does not matter -- induction is a copying circuit -- which
/// is the whole reason this needs no tokenizer.
const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

/// How the prompt reaches the model.
///
/// Three ways to say the same thing to a server, which a server is entitled to
/// assume mean the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feeding {
    /// One step, the whole prompt, which is what a server does with a new
    /// conversation.
    Prefilled,
    /// One step per token, which is what a server does with a conversation it
    /// is already decoding -- and which fires the DECODE plan for every
    /// position rather than the prefill plan once.
    OneAtATime,
    /// Prefilled, but with a second conversation in every batch.
    Alongside,
}

/// What a run of [`continued`] produced.
///
/// The distribution comes back beside the tokens because one caller checks the
/// whole of it against an independent implementation and the others check only
/// which token won. Reading it from the same run they all use is the point: a
/// separate helper that fired its own prompt would be a second setup to drift.
struct Continuation {
    /// The four tokens, greedily.
    tokens: Vec<u32>,
    /// Every logit of the row the FIRST fire answered -- the last row of the
    /// prompt, before anything was fed back.
    first: Vec<f32>,
    /// Every logit of the row the fire AFTER that answered: one token, fed
    /// back, through the decode plan and against a cache the prefill wrote.
    ///
    /// Separate from `first` because they are not the same claim. A prefill
    /// row is computed from tokens the same fire attended over; a decode row
    /// is computed from a cache written by an earlier fire, which is where
    /// paging, the page table and the cache's layout enter -- none of which a
    /// prefill-only comparison can reach.
    second: Vec<f32>,
}

/// A shell serving one of [`REALS`], with its weights held.
///
/// Extracted from `continued` when a second caller wanted the same four
/// setup steps: one place that knows how a text is assembled means a caller
/// cannot get the cache shape from qwen3 while serving qwen2.5.
fn shelled(
    dir: &std::path::Path,
    model: &Real,
    real: &'static std::collections::BTreeMap<String, Vec<u8>>,
    pages: u32,
) -> driver_vulkan::shell::Shell {
    shelled_at_tile(dir, model, real, pages, None, true)
}

/// [`shelled`], with the GEMM's instantiation point overridden.
///
/// `None` is the tile the shared model code states, which is what every
/// other test wants. A tile is stated here only by
/// `the_cooperative_matrix_gemm_answers_what_the_baseline_one_does`, which
/// needs to hold the tile FIXED across its two runs so that the tier is the
/// only thing that differs.
///
/// This used to say the override existed because the default, `(16, 32)`,
/// "deliberately has none" -- no cooperative-matrix module. Both halves of
/// that stopped being true when `QMM_TILE` widened to `(32, 32)`, which is a
/// point that HAS one, and the sentence stayed. Measured on the default
/// path: a 64-row prefill resolves 2 of its 10 symbols above baseline, and a
/// decode resolves 0 of 9 -- the GEMM is built at the tier and the GEMV is
/// not built at it at all. `the_default_tile_reaches_the_tier_in_production`
/// is that measurement kept.
fn shelled_at_tile(
    dir: &std::path::Path,
    model: &Real,
    real: &'static std::collections::BTreeMap<String, Vec<u8>>,
    pages: u32,
    tile: Option<(u32, u32)>,
    tiered: bool,
) -> driver_vulkan::shell::Shell {
    use driver_vulkan::shell::{Deployment, Shell, Text};
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_ir::trace::FireClass;

    let facts = (model.facts)();
    // FROM THE FACTS, not from qwen3's numbers, because the second model this
    // serves has two kv heads where the first has eight. A cache shaped for
    // the wrong model is not refused: it is a page whose rows are read at the
    // wrong stride, which still fires and still returns finite logits.
    //
    // Assembled here rather than derived by the shell for the reason
    // `shell::Text` states: a driver that traced its own text would be a
    // driver with an opinion about which models exist. What the shell does
    // instead is CHECK the four pieces against each other, which is what
    // `a_shell_refuses_a_model_assembled_out_of_two` measures.
    // `synthetic()` is `driver-metal`'s answer sheet, and this backend
    // disagrees with it on exactly one line: `add_bias`. That driver's binder
    // does not resolve `Source::OutWidth`, so the fact set it publishes says
    // it cannot launch `norm::add_bias`; this one can, which is what
    // `a_qwen2_5_decode_matches_a_cpu_oracle_that_adds_the_qkv_biases`
    // measures against a whole distribution.
    let metal = LlamaLikeMetalFacts {
        add_bias: true,
        qmm_tile: tile.unwrap_or(LlamaLikeMetalFacts::synthetic().qmm_tile),
        ..LlamaLikeMetalFacts::synthetic()
    };
    let text = Text {
        decode: llama_like_metal(&facts, &metal, FireClass::Decode),
        prefill: llama_like_metal(&facts, &metal, FireClass::Prefill),
        geometry: driver_vulkan::dispatch::Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: 0,
            experts_per_token: 0,
        },
        layers: facts.layers as u16,
    };

    let mut modules: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(dir).expect("the spirv dir").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "spv")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            // `tiered` false drops every `<symbol>.<tag>` key, which is how
            // a caller asks for the scalar answer on a device that would
            // otherwise load a tiered module. Nothing else can express it:
            // the tier comes from the DEVICE, and a store cannot be asked to
            // pretend the hardware is smaller than it is.
            if tiered || !stem.contains('.') {
                modules.insert(stem.to_string(), std::fs::read(&path).expect("a module"));
            }
        }
    }

    let mut shell = Shell::open(
        text,
        Deployment {
            pages,
            ..Deployment::default()
        },
        modules,
    )
    .unwrap_or_else(|e| panic!("the shell: {e}"));
    for (name, bytes) in real {
        shell.hold(name, bytes).expect("a weight");
    }

    shell
}

/// The four tokens this model produces after being shown [`PERIOD`] five
/// times, fed the given way.
///
/// Returns rather than asserts, because what the callers compare is the three
/// ways against EACH OTHER as much as against the pattern.
fn continued(
    // Not used: `Shell` opens its own device. Held anyway, because it is this
    // suite's lock and dropping it here would let two tests fire at once.
    _device: &Device,
    dir: &std::path::Path,
    model: &Real,
    real: &'static std::collections::BTreeMap<String, Vec<u8>>,
    how: Feeding,
) -> Continuation {
    use driver_vulkan::turns::Turn;

    let mut shell = shelled(dir, model, real, 8);

    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..5 {
        prompt.extend_from_slice(&PERIOD);
    }
    // ...and then the beginning of a sixth repeat, so the next token the model
    // should want is `PERIOD[2]`.
    prompt.push(PERIOD[0]);
    prompt.push(PERIOD[1]);
    // THIRTY-TWO, and the length is deliberate for a reason that has changed.
    // It used to be the only length that ran: a prefill of twenty was refused
    // with `20 rows is not a whole number of 16-row tiles`, and this said a
    // caller above this crate owed the batching. No caller did, so
    // `Serving::tiled` now splits a partial fire into fires the tile covers --
    // measured by
    // `a_prompt_that_is_not_whole_tiles_is_answered_the_way_the_decode_answers_it`.
    //
    // The length stays whole because this test is about something else: three
    // ways of feeding the SAME rows must agree, and a prompt that split would
    // compare a split fire against a decode instead of the prefill against the
    // decode. So this is the unsplit path, kept unsplit on purpose.
    assert_eq!(
        prompt.len() % 16,
        0,
        "the tiled GEMM takes whole 16-row tiles"
    );

    // The second conversation, sixteen tokens so that the batch stays a whole
    // number of tiles, and deliberately NOT the pattern -- a distraction that
    // agreed with A would not distinguish a shared cache from a private one.
    let other: Vec<u32> = (0..16).map(|i| 5_000 + i * 37).collect();

    let argmax = |v: &[f32]| -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a non-empty distribution")
            .0 as u32
    };
    // The row read is the caller's, absolute within the batch. A is put
    // SECOND in every mixed batch below on purpose: a conversation whose
    // answer depended on where in the batch it sat would be a driver that
    // could not be given work in the order it arrived, and A-first would
    // leave A's rows at index 0 either way and never say so.
    let mut fires = 0usize;
    let mut widest = 0usize;
    let mut first: Vec<f32> = Vec::new();
    // A cell rather than a plain local: the closure holds it for the whole
    // run, and the caller below needs to read it BETWEEN calls.
    let latest: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    let mut fire = |turns: &[Turn], a_rows: usize| -> u32 {
        let step = shell.step(turns).unwrap_or_else(|e| panic!("{e}"));
        // The premise of reading row `a_rows - 1`: a batch that came back
        // narrower than the turns asked for would have this reading someone
        // else's distribution, or reading past the end.
        assert_eq!(
            step.rows,
            turns.iter().map(|t| t.tokens.len()).sum::<usize>(),
            "the fire answered a different number of rows than the turns state"
        );
        fires += 1;
        widest = widest.max(step.rows);
        // `row` and not a slice of `values`: a step holds only the rows some
        // request will read (see `serve::Logits::read`), so the arithmetic
        // that used to find row `a_rows - 1` now lands on a different row's
        // distribution or past the end. `row` is the addressing that stays
        // true whichever rows the fire kept.
        let row = step
            .logits
            .row(a_rows - 1)
            .expect("the last row of the fire is a row some request reads");
        if first.is_empty() {
            first = row.to_vec();
        }
        *latest.borrow_mut() = row.to_vec();
        argmax(row)
    };

    let mut got = Vec::new();
    match how {
        Feeding::Prefilled => {
            got.push(fire(
                &[Turn {
                    who: 1,
                    tokens: prompt.clone(),
                }],
                prompt.len(),
            ));
        }
        Feeding::OneAtATime => {
            // Every position through the DECODE plan, one fire each. Only the
            // LAST one's distribution has seen the whole prompt, so the
            // earlier answers are read and dropped -- reading them at all is
            // the point, since a fire whose distribution nobody reads is a
            // fire whose arena could have been anything.
            let mut answer = 0;
            for t in &prompt {
                answer = fire(
                    &[Turn {
                        who: 1,
                        tokens: vec![*t],
                    }],
                    1,
                );
            }
            got.push(answer);
        }
        Feeding::Alongside => {
            got.push(fire(
                &[
                    Turn {
                        who: 2,
                        tokens: other.clone(),
                    },
                    Turn {
                        who: 1,
                        tokens: prompt.clone(),
                    },
                ],
                other.len() + prompt.len(),
            ));
        }
    }

    // Three more, each fed back, so the decode plan and the cache carry the
    // pattern forward rather than the prefill answering everything.
    let mut second: Vec<f32> = Vec::new();
    for (round, filler) in other.iter().take(3).enumerate() {
        let fed = *got.last().expect("a token");
        let (turns, at) = if how == Feeding::Alongside {
            (
                vec![
                    Turn {
                        who: 2,
                        // Whatever B says, as long as it is not what A says.
                        tokens: vec![*filler],
                    },
                    Turn {
                        who: 1,
                        tokens: vec![fed],
                    },
                ],
                2,
            )
        } else {
            (
                vec![Turn {
                    who: 1,
                    tokens: vec![fed],
                }],
                1,
            )
        };
        got.push(fire(&turns, at));
        if round == 0 {
            second = latest.borrow().clone();
        }
    }
    // THE PREMISES, checked after the fact because they are about the whole
    // run rather than any one fire. Without them a helper that quietly
    // ignored `how` would make the comparison between the three ways
    // vacuous -- three identical runs agree perfectly.
    match how {
        Feeding::Prefilled => {
            assert_eq!(fires, 4, "one prefill and three decodes");
            assert_eq!(widest, 32, "the prefill was not one fire");
        }
        Feeding::OneAtATime => {
            assert_eq!(fires, 35, "thirty-two single tokens and three decodes");
            assert_eq!(widest, 1, "something was fed more than one token");
        }
        Feeding::Alongside => {
            assert_eq!(fires, 4, "one prefill and three decodes");
            assert_eq!(widest, 48, "the second conversation was not in the batch");
        }
    }
    // The premise of the decode comparison: a row of the wrong width, or one
    // no fire ever wrote, would be compared against the reference as zeros
    // and read as a driver that answers nothing.
    assert_eq!(
        second.len(),
        first.len(),
        "the decode row is not the width of the prefill's"
    );
    Continuation {
        tokens: got,
        first,
        second,
    }
}

/// The same conversation, said three ways, answered the same way.
///
/// # The claim a server actually needs
///
/// [`a_real_model_continues_a_pattern_it_was_shown`] proves one conversation
/// alone, prefilled. A server never runs that. It runs conversations in
/// batches it did not choose, at row counts that change every step, and it is
/// entitled to assume that a conversation's answer is its own.
///
/// So this fires the same prompt three ways and requires one answer:
///
///   - **prefilled**, thirty-two rows in one fire;
///   - **one token at a time**, thirty-two fires through the DECODE plan --
///     which is a different plan, different kernels, and a KV cache written
///     one row per fire instead of thirty-two at once;
///   - **alongside** a second conversation that shares every batch, every
///     fire, the same arena and the same cache, and says something else.
///
/// # What each one can catch that the others cannot
///
/// The one-at-a-time run is the prefill/decode equivalence. The two plans
/// state different matmuls above sixteen rows and different attention paths,
/// and nothing before this held their ANSWERS against each other on a real
/// model across a real cache.
///
/// The alongside run is page ownership and per-row positions. A batch that
/// let one conversation read another's pages, or that gave row 0 row 32's
/// position, still fires, still records, and still returns finite logits.
///
/// Neither is a claim any single fire can make.
///
/// # The controls
///
/// A's turn is put SECOND in every mixed batch, so its rows do not begin at
/// zero and "the same answer" is not the same offset read twice. Reading its
/// old offset instead answers `[41777, 271, 14190, 11]` -- and the first of
/// those is `PERIOD[4]`, which is exactly right for A's fifteenth row, so the
/// batch is laid out per turn in the order the turns were given and the
/// distraction really is sixteen rows wide.
///
/// One control that does NOT fire, and why it is not weakened: giving B the
/// same conversation id as A is refused by `Frame::of` with `requests 0 and 1
/// both own page 0` rather than producing a wrong number. That is the refusal
/// working, so it is recorded rather than reworked into a numeric failure.
#[test]
fn a_conversation_is_answered_the_same_however_it_reaches_the_driver() {
    let (device, dir) = gpu!();
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so batch independence is unmeasured");
        return;
    };
    let want = PERIOD[2..].to_vec();

    let alone = continued(&device, dir, &REALS[0], real, Feeding::Prefilled).tokens;
    assert_eq!(alone, want, "prefilled");

    let stepped = continued(&device, dir, &REALS[0], real, Feeding::OneAtATime).tokens;
    assert_eq!(
        stepped, alone,
        "the decode plan answers differently than the prefill plan"
    );

    let batched = continued(&device, dir, &REALS[0], real, Feeding::Alongside).tokens;
    assert_eq!(
        batched, alone,
        "a second conversation in the batch changed this one's answer"
    );
}

/// A prompt whose length is not a whole number of GEMM tiles.
///
/// # The fire the driver could not run
///
/// `affine_qmm_t` is compiled at row tiles of 16, 32 and 64, it reads its
/// tile from the grid, and `geometry::eval` refuses a fire whose rows are not
/// a multiple of one -- `PartialTile`. `continued` above records the
/// consequence in its own prompt length: it keeps to 32 tokens and says a
/// caller above this crate owes the batching.
///
/// No caller does. "The capital of France is" is 29 tokens, and a real
/// `pie serve` on this driver refused it at the first projection. So
/// `Serving::tiled` splits such a fire into fires the tile covers, and this
/// is the measurement that the split computes the same model.
///
/// # Why one-at-a-time is the reference
///
/// The single-row path is the one every decode of every conversation takes,
/// it goes through a different plan (`affine_qmv_fast`, not the tiled GEMM),
/// and it cannot be split because there is nothing to split. If the 29-row
/// fire agrees with 29 one-row fires to the last token, the split kept the
/// arithmetic; and the two paths share no launch shape, so they cannot agree
/// by making the same mistake.
///
/// The mutation that matters is the OVERLAP. 29 rows is fired as rows 0..16
/// and rows 13..29, and the second fire's first three rows are rows the first
/// already answered. Keeping them would shift every distribution after row 16
/// by three, so the row a caller reads would be row 25's answer. Measured:
/// passing `0` for the overlap makes the step report THIRTY-TWO rows for a
/// 29-token prompt, which this test names before it reads a distribution --
/// which is why the row count is asserted at all, rather than being taken as
/// obvious from the prompt.
#[test]
fn a_prompt_that_is_not_whole_tiles_is_answered_the_way_the_decode_answers_it() {
    use driver_vulkan::turns::Turn;

    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the partial tile is unmeasured");
        return;
    };
    // Four whole repeats and five tokens of a fifth: 29 rows, which is one
    // 16-row tile and thirteen rows over.
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..4 {
        prompt.extend_from_slice(&PERIOD);
    }
    prompt.extend_from_slice(&PERIOD[..5]);
    assert_eq!(prompt.len(), 29);
    assert_ne!(prompt.len() % 16, 0, "the whole point of the prompt");

    let argmax = |v: &[f32]| -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a non-empty distribution")
            .0 as u32
    };

    let mut shell = shelled(dir, &REALS[0], real, 8);
    let step = shell
        .step(&[Turn {
            who: 1,
            tokens: prompt.clone(),
        }])
        .unwrap_or_else(|e| panic!("the partial-tile prefill: {e}"));
    assert_eq!(step.rows, 29, "the split answered a different fire");
    assert_eq!(
        step.logits.rows, 29,
        "every row samples, so there is one distribution a row"
    );
    let split = step.logits.row(28).expect("the prompt's last row").to_vec();

    // The same tokens, one fire each, on a cache of its own.
    let mut apart = shelled(dir, &REALS[0], real, 8);
    let mut row: Vec<f32> = Vec::new();
    for t in &prompt {
        let one = apart
            .step(&[Turn {
                who: 1,
                tokens: vec![*t],
            }])
            .unwrap_or_else(|e| panic!("the decode: {e}"));
        row = one.logits.values[..one.logits.vocab].to_vec();
    }

    assert_eq!(
        argmax(&split),
        argmax(&row),
        "the split prefill and the decode want different tokens"
    );
    assert_eq!(
        argmax(&split),
        PERIOD[5],
        "the model was shown the pattern and did not continue it"
    );
    // Not bit-exact and not asked to be: the two paths run different kernels
    // over the same weights. Close is the claim -- a shifted row is not close.
    let worst = split
        .iter()
        .zip(&row)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 0.25, "the two paths disagree by {worst}");
}

/// A second real model is served correctly, biases and all.
///
/// # What one model could not say
///
/// Every real-weight claim in this file was made against one checkpoint, and a
/// driver that had specialised to it -- eight kv heads, a 1024-wide hidden, a
/// qk-norm at every layer -- would pass all of them and serve nothing else. So
/// this serves `mlx-community/Qwen2.5-1.5B-Instruct-4bit` through exactly the
/// same helper. It differs in every fact a driver has to get right: **two kv
/// heads instead of eight** (a quarter as wide a cache, 6:1 head grouping
/// rather than 2:1), hidden 1536 rather than 1024, an mlp of 8960 rather than
/// 3072, a FUSED qkv projection, **no qk-norm at all**, and **attention
/// biases**, which nothing else here has -- a different kernel sequence, not
/// merely different sizes. 732 weights against 704.
///
/// # It answered the wrong thing for four commits, and that is the point
///
/// This test used to assert `[5937, 1560, 16925, 43715]` where the pattern
/// wanted `[88204, 6100, 41777, 2930]`, and the difference was three kernels
/// a layer that nobody launched.
///
/// Qwen2.5 has attention biases -- `LlamaLikeFacts::qwen2_5_1_5b` states
/// `qkv_bias: true`, the semantic text and the CUDA text both add
/// `{q,k,v}_proj.bias` to the raw projections, and this checkpoint ships all
/// 84 of them. The shared Metal text ignored the fact entirely, for one
/// reason: `AddBias` lowers to `norm::add_bias_bf16`, no Metal-side kernel
/// added a bias, and a text cannot state an op no kernel implements. So the
/// lowered qwen2.5 decode plan bound 648 weights and not one was a bias, and
/// this driver computed a Qwen2 without them -- correctly, and to the wrong
/// answer.
///
/// # How that stopped being a theory, and then stopped being true
///
/// A CPU reference (`files/qwen-cpu-reference.py` in the session that wrote
/// this: a numpy forward reading the safetensors directly, dequantizing MLX's
/// 4-bit groups itself, sharing no code and no kernel with this crate) was run
/// on the same checkpoint and the same prompt, twice:
///
///   - with the biases:    `[88204, 6100, 41777, 2930]` -- the pattern;
///   - without the biases: `[5937, 1560, 16925, 43715]`.
///
/// The driver matched the second, exactly, four greedy steps deep. That
/// measured the gap instead of arguing it, and it is what the fix was aimed
/// at: a `norm/add_bias` kernel in `kernels-vulkan` AND in `kernels-metal`
/// (coverage is parity, and a text can only name what some kernel on that
/// side implements), `Source::OutWidth` taught to this crate's binder, and a
/// `LlamaLikeMetalFacts::add_bias` capability the text gates the three
/// statements on. The driver now answers the FIRST list.
///
/// The same reference was run on qwen3-0.6B, which has no biases to drop, and
/// agreed with the card there too. So it is an oracle for both models and not
/// a story told about one.
///
/// # Why the biasless answer is still written down
///
/// Because it is the thing this test would produce if the capability were
/// ever quietly turned off -- by a fact set defaulting the other way, by a
/// text that stopped gating on it, by a binder that dropped `OutWidth` and
/// left the run short. Every one of those is a silent wrong answer rather
/// than an error, and none of them changes the shape of anything. Asserting
/// that the answer is the biased one AND is not the biasless one names the
/// regression instead of just failing near it.
///
/// # The controls
///
/// The cache shape is taken from the FACTS rather than from qwen3's numbers,
/// and that is load-bearing: serving qwen2.5 through the eight-head shape is
/// not refused, it just answers differently. The decode plan is fired as well
/// as the prefill one, and the two agree -- which is the claim that would
/// break first if the fused qkv were bound at the wrong width, since prefill
/// and decode state different matmuls.
///
/// And the weights come from `model_loader::executor::Execution` rather than
/// from source spans: this model's plan states 535 `TileMap` transforms for
/// its fused qkv, so the verbatim read that was correct for qwen3 would have
/// handed the card three separate projections.
#[test]
fn a_second_real_model_is_served_the_way_the_text_states_it() {
    let (device, dir) = gpu!();
    let model = &REALS[1];
    let Some(real) = weights_of(model) else {
        eprintln!("no readable 4-bit qwen2.5-1.5b, so the second model is unmeasured");
        return;
    };
    // What a numpy forward of this checkpoint answers WITH the attention
    // biases: the pattern, and the text this driver is now given.
    let biased = PERIOD[2..].to_vec();
    // And what the same forward answers without them. Held here rather than
    // in a comment because it is asserted against: it is what this test
    // produces if the bias statements ever stop being launched, which is a
    // regression nothing else in this file would name.
    let biasless = vec![5_937, 1_560, 16_925, 43_715];
    assert_ne!(
        biasless, biased,
        "the reference did not distinguish the two"
    );

    let prefilled = continued(&device, dir, model, real, Feeding::Prefilled).tokens;
    assert_ne!(
        prefilled, biasless,
        "{} answers what a numpy forward of the SAME text answers with the qkv biases \
         dropped, so the bias statements are not being launched",
        model.id
    );
    assert_eq!(
        prefilled, biased,
        "{} disagrees with a numpy forward of the same text",
        model.id
    );
    // Through the decode plan, one position at a time. A fused qkv bound at
    // the wrong width, or a cache shaped for the wrong head count, breaks this
    // before it breaks the prefill.
    assert_eq!(
        continued(&device, dir, model, real, Feeding::OneAtATime).tokens,
        prefilled,
        "{}'s decode plan answers differently than its prefill plan",
        model.id
    );
}

/// The WHOLE distribution agrees with an independent implementation, not just
/// its argmax.
///
/// # What every other numeric claim in this file settles for
///
/// Which token won. That is the claim a server cares about most and the
/// weakest one a distribution can make: a row whose peak is right and whose
/// tail is noise picks the same greedy token every time and samples nothing
/// like the model. Nothing here had ever held a logit -- as a number -- against
/// anything but this crate's own other kernel.
///
/// So this holds both real models' distributions against a numpy forward that
/// shares no code with this crate: eight ranked ids, their logits, five probes
/// spread across the vocabulary, and the row's range.
///
/// # Two rows, not one
///
/// A prefill row is computed entirely within one fire, from tokens that fire
/// attended over itself. It says nothing about the KV cache: not about what
/// the prefill wrote into it, nor about the page table a later fire reads it
/// through, nor about the decode plan's own attention. So the row after that
/// -- the model's own answer fed back, one token, through the DECODE plan --
/// is held against the reference too, run on the prompt with that answer
/// appended. Everything the cache touches sits between the two.
///
/// # The tolerances, and where they come from
///
/// Logits come back as bf16, which has eight mantissa bits: at a magnitude of
/// ten the representable values are 0.0625 apart, so nothing here can agree
/// more closely than that however right it is. Measured, the card and the
/// reference differ by at most 0.4 on qwen2.5 and 0.06 on qwen3 -- the wider
/// one being the model with a 1536-wide hidden and an 8960-wide mlp, which is
/// more terms summed in a different order. The tolerance is 0.5 absolute,
/// which is six bf16 steps and about 2% of either row's range; a wrong weight,
/// a wrong position or a wrong page moves a logit by whole units.
///
/// The RANKING is checked as a set, not as an order, and only for seven of
/// eight. Measured: qwen2.5's ranks 4 through 6 sit within 0.2 of each other,
/// which is three bf16 steps, so their order is not information -- and its
/// eighth is a genuine swap between two ids 0.16 apart. The top id is checked
/// exactly, because that one has a 0.2 margin on qwen2.5 and 5.0 on qwen3.
///
/// # The controls
///
/// Three, each of which must fail. A row moved by slightly more than the
/// tolerance, which is the only one that says the magnitudes are read at all.
/// Each model's prefill row against its own DECODE reference and back, since
/// nothing else here would notice a decode row that was a copy of the
/// prefill's -- which is what a decode plan that silently re-read the prompt
/// would produce. And each model's rows against the other model's references,
/// without which tolerances this wide could be passing any plausible row.
#[test]
fn both_real_models_agree_with_an_independent_implementation() {
    let (device, dir) = gpu!();
    const PROBE: [usize; 5] = [0, 1_000, 50_000, 100_000, 151_935];
    const SLACK: f32 = 0.5;

    let mut measured = 0usize;
    // Two rows per model: the prompt's last row, and the row after its own
    // answer was fed back.
    let mut rows: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::new();
    for (which, model) in REALS.iter().enumerate() {
        let Some(real) = weights_of(model) else {
            eprintln!(
                "no readable 4-bit {}, so its distribution is unmeasured",
                model.id
            );
            continue;
        };
        let run = continued(&device, dir, model, real, Feeding::Prefilled);
        for (what, row, oracle) in [
            ("prefill", &run.first, &model.oracle),
            ("decode", &run.second, &model.decoded),
        ] {
            let id = format!("{} {what}", model.id);
            assert_eq!(row.len(), 151_936, "{id}: a whole vocabulary");
            assert!(
                row.iter().all(|v| v.is_finite()),
                "{id}: a non-finite logit makes every comparison below vacuous"
            );
            agrees(&id, oracle, row, SLACK, &PROBE);
        }
        rows.push((which, run.first, run.second));
        measured += 1;
    }
    if measured == 0 {
        eprintln!("no real checkpoint, so the oracle is unmeasured");
        return;
    }

    // THE CONTROLS, all of which expect a panic, so the hook is quietened
    // first: a backtrace printed by a check that was SUPPOSED to fail is how a
    // passing run gets read as a broken one.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let missed = |oracle: &Oracle, row: &[f32]| -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agrees("control", oracle, row, SLACK, &PROBE);
        }))
        .is_err()
    };

    // ONE: every logit moved by slightly more than the tolerance. The ranking
    // is untouched, so this is the only control that says the MAGNITUDES are
    // checked at all -- the two below fail at the argmax and would still fail
    // if every number after it were ignored.
    for (which, first, second) in &rows {
        for (what, row, oracle) in [
            ("prefill", first, &REALS[*which].oracle),
            ("decode", second, &REALS[*which].decoded),
        ] {
            let shifted: Vec<f32> = row.iter().map(|v| v + SLACK + 0.1).collect();
            assert!(
                missed(oracle, &shifted),
                "{} {what}: a whole row moved by {} still agreed, so the logits are not \
                 being read",
                REALS[*which].id,
                SLACK + 0.1
            );
        }
    }

    // TWO: each model's prefill row against its OWN decode reference. Both
    // rows come back from the same helper through the same field type, and
    // nothing else here would notice if the decode row were a copy of the
    // prefill's -- which is exactly what a driver whose decode plan silently
    // re-read the prompt would produce.
    for (which, first, second) in &rows {
        assert!(
            missed(&REALS[*which].decoded, first),
            "{}: the prefill row passes the decode reference, so the two rows are not \
             two different fires",
            REALS[*which].id
        );
        assert!(
            missed(&REALS[*which].oracle, second),
            "{}: the decode row passes the prefill reference",
            REALS[*which].id
        );
    }

    // THREE: each row against the OTHER model's reference. Only possible when
    // both were measured, so the reason is printed rather than the claim
    // quietly weakened.
    if rows.len() == 2 {
        for (which, first, second) in &rows {
            let other = &REALS[1 - which];
            assert!(
                missed(&other.oracle, first) && missed(&other.decoded, second),
                "{}'s distribution passes {}'s reference, so this tolerance measures nothing",
                REALS[*which].id,
                other.id
            );
        }
    } else {
        eprintln!("only one model was measured, so the cross-check is unmeasured");
    }
    std::panic::set_hook(hook);
}

/// One row against one [`Oracle`]. Panics with what differed.
fn agrees(id: &str, oracle: &Oracle, row: &[f32], slack: f32, probe: &[usize]) {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|a, b| row[*b].total_cmp(&row[*a]));

    assert_eq!(
        idx[0] as u32, oracle.top[0],
        "{}: the reference's most likely token is not the card's",
        id
    );
    // As a SET and seven of eight: the ranks below the peak are separated by
    // less than a few bf16 steps, so their order carries no information.
    let card: std::collections::BTreeSet<u32> = idx[..8].iter().map(|i| *i as u32).collect();
    let want: std::collections::BTreeSet<u32> = oracle.top.iter().copied().collect();
    let shared = card.intersection(&want).count();
    assert!(
        shared >= 7,
        "{}: only {shared} of the reference's eight most likely tokens are in the card's \
         eight, which is more than a near-tie",
        id
    );
    // The logits themselves, at the ids the reference ranked -- which is a
    // stronger claim than the ranking, since a row could rank the same ids in
    // the same order at entirely wrong magnitudes.
    for (id, want) in oracle.top.iter().zip(oracle.vals) {
        let got = row[*id as usize];
        assert!(
            (got - want).abs() <= slack,
            "{}: logit {id} is {got}, the reference says {want}",
            id
        );
    }
    for (at, want) in probe.iter().zip(oracle.probe) {
        let got = row[*at];
        assert!(
            (got - want).abs() <= slack,
            "{}: logit {at} is {got}, the reference says {want}",
            id
        );
    }
    let (lo, hi) = row
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    assert!(
        (hi - lo - oracle.span).abs() <= slack,
        "{}: the row spans {}, the reference spans {}",
        id,
        hi - lo,
        oracle.span
    );
}

/// Every fact the engine is handed is the one this driver actually keeps.
///
/// # Why this is not "read the limit twice"
///
/// `facts::of` reports numbers it read from the device, so the obvious test --
/// read them again and compare -- would pass for a driver that reported a
/// perfectly accurate limit it then ignored everywhere else. The engine does
/// not use these to describe the hardware; it uses them to decide what to send.
/// So each one is held against the thing in this crate that would break if the
/// engine believed it.
///
/// * `storage_alignment` against `Bound::at`, which refuses a sub-range whose
///   offset it does not divide. An arena laid out on a smaller alignment would
///   be refused a bind; on a larger one it would waste space it did not have
///   to. Both directions are checked.
/// * `page_size` against a pool built at that page size, which must serve.
/// * `unified_memory` against the heaps, which is a different question asked of
///   different data: `deviceType` says what KIND of part this is, the memory
///   types say whether any of its memory is out of the host's reach.
/// * `abi_version` and `backend` against the constants the seam matches on.
#[test]
fn the_facts_the_engine_is_given_are_the_ones_this_driver_keeps() {
    let (device, _dir) = gpu!();
    let facts = driver_vulkan::facts::of(&device);

    assert_eq!(
        facts.abi_version,
        driver_api::PIE_DRIVER_ABI_VERSION,
        "a driver that states an ABI it was not built against is refused at the door"
    );
    assert_eq!(facts.backend, "vulkan", "the string the engine selects on");

    // ALIGNMENT, both ways. `min_storage_offset` is a power of two per the
    // specification, so `+ 1` is never a multiple of it and `* 2` always is.
    let align = u64::from(facts.storage_alignment);
    assert!(align > 0, "an alignment of zero divides nothing");
    assert_eq!(
        align,
        device.min_storage_offset(),
        "the stated alignment is not the one sub-ranges are bound at"
    );
    let buffer = device
        .buffer(&vec![0u8; (align * 8) as usize])
        .expect("a buffer");
    driver_vulkan::device::Bound::at(&device, &buffer, align * 2, align)
        .expect("an offset the stated alignment divides is bindable");
    assert!(
        driver_vulkan::device::Bound::at(&device, &buffer, align + 1, align).is_err(),
        "an offset the stated alignment does NOT divide was bound anyway, so the number \
         the engine lays arenas out on is not the number this driver enforces"
    );
    device.free(buffer);

    // THE PAGE SIZE, against a pool that has to serve at it.
    assert_eq!(
        facts.page_size,
        driver_vulkan::facts::PAGE_SIZE,
        "two spellings of the same constant"
    );
    let shape = driver_vulkan::resources::Shape {
        layers: 1,
        kv_heads: 2,
        head_dim: 64,
        page_size: facts.page_size,
        pages: 4,
        bytes: 2,
    };
    driver_vulkan::resources::Pool::open(&device, shape)
        .expect("a pool at the page size the engine is told to index in units of");

    // UNIFIED MEMORY, against the heaps rather than against `deviceType`.
    assert_eq!(
        facts.unified_memory,
        !device.device_only_memory(),
        "{}: the device's KIND and its memory types disagree about whether the host \
         can see everything",
        device.name()
    );

    // The two that are zero, and stay zero until something implements them.
    // Stated as a claim rather than left unchecked: a non-zero tile map is a
    // promise to accept a sparse residency plan, and nothing here reads one.
    assert_eq!(facts.storage_max_tile_bytes, 0);
    assert_eq!(facts.storage_tile_map_mask, 0);
    // A kernel table fact. If either of these is ever true, `kernels-vulkan`
    // gained a kernel and this test should be the thing that says so.
    assert!(!facts.fp8_native);
    assert!(!facts.native_mxfp4_moe);
}

/// A shell refuses a model whose four pieces came from two models.
///
/// # Why this is the interesting claim about `Shell`
///
/// That it SERVES is proved by every real-model test in this file, all of
/// which now go through it -- `continued` used to assemble a pool, a book, a
/// weights table, two plans, a geometry and a tier by hand, and assembling
/// them is what it no longer does. Reuse is the strongest evidence that a
/// composition is right, and it costs nothing to state.
///
/// What reuse cannot show is the reason the composition is worth having. The
/// pieces have to agree with each other, and nothing before this checked that
/// they did. A cache one layer short of the plan is not refused anywhere
/// downstream: the last layer reads and writes a region belonging to no layer,
/// the fire succeeds, and the logits are finite and wrong. So each pair is
/// broken here on purpose and the refusal is read.
#[test]
fn a_shell_refuses_a_model_assembled_out_of_two() {
    use driver_vulkan::shell::Text;
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_ir::trace::FireClass;

    // No device needed: every check is over the text. Kept in this file
    // anyway, because it is about a type whose other half needs one.
    let metal = LlamaLikeMetalFacts::synthetic();
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let other = LlamaLikeFacts::qwen2_5_1_5b();
    let geometry = |f: &LlamaLikeFacts| driver_vulkan::dispatch::Geometry {
        q_heads: f.q_heads,
        kv_heads: f.kv_heads,
        head_dim: f.head_dim,
        rotary_dims: f.head_dim,
        n_experts: 0,
        experts_per_token: 0,
    };
    let whole = || Text {
        decode: llama_like_metal(&facts, &metal, FireClass::Decode),
        prefill: llama_like_metal(&facts, &metal, FireClass::Prefill),
        geometry: geometry(&facts),
        layers: facts.layers as u16,
    };

    // THE PREMISE. A pair that is right must pass, or every refusal below is
    // a text this driver simply cannot serve.
    whole()
        .servable()
        .unwrap_or_else(|e| panic!("a model assembled from one set of facts: {e}"));

    let refused = |what: &str, text: Text| -> String {
        match text.servable() {
            Err(driver_vulkan::shell::Unopened::Unservable(why)) => why,
            Err(e) => panic!("{what}: refused for the wrong reason: {e}"),
            Ok(()) => panic!("{what}: served"),
        }
    };

    // ONE: two plans from two models. Both are 28-layer llama-likes, so the
    // op counts and the layer depths agree and only the widths do not --
    // which is why the check is over `Dim::Const`.
    let why = refused(
        "a decode plan of qwen3 beside a prefill plan of qwen2.5",
        Text {
            prefill: llama_like_metal(&other, &metal, FireClass::Prefill),
            ..whole()
        },
    );
    assert!(
        why.contains("two models"),
        "refused, but not for being two models: {why}"
    );

    // TWO: a cache a layer short of the plan.
    let why = refused(
        "a cache of 27 layers for a plan of 28",
        Text {
            layers: facts.layers as u16 - 1,
            ..whole()
        },
    );
    assert!(why.contains("layers"), "{why}");

    // THREE: a geometry from the other model. qwen3 has 8 key heads and 16
    // query heads; qwen2.5 has 2 and 12, so 12 does not divide among 8.
    let why = refused(
        "qwen2.5's head counts under qwen3's plan",
        Text {
            geometry: driver_vulkan::dispatch::Geometry {
                q_heads: other.q_heads,
                ..geometry(&facts)
            },
            ..whole()
        },
    );
    assert!(why.contains("divide"), "{why}");

    // FOUR: a rotation wider than the head it turns.
    let why = refused(
        "a rotation of twice the head",
        Text {
            geometry: driver_vulkan::dispatch::Geometry {
                rotary_dims: facts.head_dim * 2,
                ..geometry(&facts)
            },
            ..whole()
        },
    );
    assert!(why.contains("rotation"), "{why}");

    // FIVE: a head dimension no attention kernel serves. 96 is not invented
    // for this test -- it is `phi3_mini`'s, which is why that model cannot be
    // served here and why the refusal has to be at open time rather than at
    // the first fire.
    let why = refused(
        "a head dimension of 96",
        Text {
            geometry: driver_vulkan::dispatch::Geometry {
                head_dim: 96,
                rotary_dims: 96,
                ..geometry(&facts)
            },
            ..whole()
        },
    );
    assert!(why.contains("no attention kernel"), "{why}");
}

/// A conversation forked from another answers the way its source does, and a
/// conversation forked from a DIFFERENT one does not.
///
/// # What this is for
///
/// The engine's `copy_kv`. A branch -- a beam, a retry, two continuations of
/// one system prompt -- otherwise pays a second prefill over tokens the cache
/// already holds. Nothing in this driver could move a page until now.
///
/// # What it measures
///
/// Both halves of a fork, which live in two modules on purpose. `Book::fork`
/// seats the destination and returns the moves; `Pool::copy_page` performs
/// them; only `Shell::fork` has both. A fork that seated without copying
/// produces a conversation attending over WHATEVER those pages last held --
/// zeros on a fresh pool, which is finite, plausible and wrong. So the claim
/// is made numerically, over a whole distribution and not an argmax: two rows
/// that agree to the last bit came from the same cache.
///
/// # The controls
///
/// Three, and each rules out something the equality alone would not:
///
/// 1. **A fork of a different source answers differently.** Without it, a
///    driver that copied nothing and left both conversations reading page 0
///    would pass -- every conversation would agree with every other.
/// 2. **The forked conversation and its source then DIVERGE when fed
///    different tokens.** Without it, a fork that aliased the same pages
///    rather than copying them would pass the first assertion perfectly, and
///    two conversations sharing one cache is precisely the bug being ruled
///    out.
/// 3. **Three refusals**: forking onto a seated conversation, forking from an
///    unseated one, and forking onto itself. The first is the dangerous one --
///    it would drop the destination's pages without telling anybody.
///
/// # The mutations
///
/// Five, each on a different line, and all five are refused at the first
/// comparison -- which is the point: every one of them produces a
/// conversation that still answers, finitely and plausibly.
///
/// 1. `Shell::fork` skips the `copy_page` loop entirely.
/// 2. `Book::fork` seats the destination on the SOURCE's pages, aliasing
///    rather than copying. Refused earlier still, by `Frame::of`: `requests 0
///    and 1 both own page 0`. Recorded rather than reworked, because a
///    refusal is a better outcome than a number.
/// 3. `Pool::copy_page` copies layer 0 only.
/// 4. `Pool::copy_page` copies the keys and not the values.
/// 5. `Book::fork` seats the destination at zero tokens rather than the
///    source's count -- so the pages are right and the length is not.
///
/// Nothing here is a survivor.
#[test]
fn a_forked_conversation_carries_the_history_it_was_forked_from() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so forking is unmeasured");
        return;
    };
    // Four conversations of two pages each at most, and the fork needs the
    // source's pages free a second time.
    let mut shell = shelled(dir, &REALS[0], real, 16);

    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..5 {
        prompt.extend_from_slice(&PERIOD);
    }
    prompt.push(PERIOD[0]);
    prompt.push(PERIOD[1]);
    // The same thirty-two-token prompt the other real-model tests use, and
    // the length is not free: the tiled GEMM takes whole 16-row tiles.
    assert_eq!(prompt.len() % 16, 0, "whole 16-row tiles");
    // Deliberately not the pattern: a distraction that agreed with the
    // pattern would make control 1 vacuous.
    let other: Vec<u32> = (0..16).map(|i| 5_000 + i * 37).collect();

    // A function of the shell rather than a closure over it: this test forks
    // between fires, and a closure that captured `&mut shell` would own it
    // for the whole body.
    fn fire(
        shell: &mut driver_vulkan::shell::Shell,
        turns: &[driver_vulkan::turns::Turn],
    ) -> Vec<Vec<f32>> {
        let step = shell.step(turns).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            step.rows,
            turns.iter().map(|t| t.tokens.len()).sum::<usize>(),
            "the fire answered a different number of rows than the turns state"
        );
        let mut at = 0;
        turns
            .iter()
            .map(|t| {
                // The LAST row of each turn: the only one that has seen the
                // whole of what that turn appended, and -- not by coincidence
                // -- the only row of that turn the fire kept.
                at += t.tokens.len();
                step.logits
                    .row(at - 1)
                    .expect("each turn's last row is the row it reads")
                    .to_vec()
            })
            .collect()
    }
    let turn = |who: u64, tokens: &[u32]| driver_vulkan::turns::Turn {
        who,
        tokens: tokens.to_vec(),
    };

    // Two histories, seated by prefill: 1 holds the pattern, 2 holds the
    // distraction.
    fire(&mut shell, &[turn(1, &prompt), turn(2, &other)]);

    let pages = shell.fork(1, 3).expect("a fork of a seated conversation");
    assert!(pages > 0, "a fork that copied no page is not a fork");
    assert_eq!(
        pages,
        shell.book().pages(1).map_or(0, <[u32]>::len),
        "the fork copied a different number of pages than the source holds"
    );
    let from_other = shell.fork(2, 4).expect("a fork of the distraction");

    // One fire, so nothing between the four rows can differ except the cache
    // each one attends over. 3 is put before 1 so that "the same answer" is
    // not the same offset read twice.
    let rows = fire(
        &mut shell,
        &[
            turn(3, &[PERIOD[2]]),
            turn(1, &[PERIOD[2]]),
            turn(4, &[PERIOD[2]]),
        ],
    );

    let (forked, source, wrong) = (&rows[0], &rows[1], &rows[2]);

    assert_eq!(
        forked, source,
        "the fork answers differently than the conversation it came from"
    );
    // Control 1. Same token, same plan, same batch -- only the copied history
    // differs, so anything but a difference here means no history was copied.
    assert!(
        from_other > 0 && wrong != source,
        "a fork of the DISTRACTION answers exactly what the pattern's fork does, \
         so the pages carry nothing"
    );

    // Control 2. Fed different tokens, the two must part. A fork that handed
    // out the same pages rather than copies would keep answering identically
    // forever, and would have passed everything above.
    let rows = fire(&mut shell, &[turn(3, &[PERIOD[3]]), turn(1, &[9_999])]);
    assert_ne!(
        rows[0], rows[1],
        "the fork and its source still agree after being fed different tokens, \
         so they are sharing one cache rather than holding two"
    );

    // Control 3. The three refusals.
    use driver_vulkan::pages::{Unforkable, Unhoused};
    use driver_vulkan::shell::Unforked;
    let refused = |e: Unforked| match e {
        Unforked::Unhoused(Unhoused::Unforkable(why)) => why,
        other => panic!("expected a refusal about forking, got {other}"),
    };
    assert_eq!(
        refused(shell.fork(1, 3).expect_err("3 is already seated")),
        Unforkable::Taken
    );
    assert_eq!(
        refused(shell.fork(77, 78).expect_err("77 has no history")),
        Unforkable::Absent
    );
    assert_eq!(
        refused(shell.fork(1, 1).expect_err("onto itself")),
        Unforkable::Itself
    );
    // ...and the refusal left 3 alone: it still answers, and from its own
    // history rather than from nothing.
    let after = fire(&mut shell, &[turn(3, &[PERIOD[4]])]);
    assert!(
        after[0].iter().all(|v| v.is_finite()),
        "a refused fork disturbed the conversation it refused to overwrite"
    );
}

/// The engine's `copy_kv` shape -- page moves and row cells -- moves the same
/// bytes a fork does, and a plan with one bad cell moves nothing at all.
///
/// # Why this is separate from the fork test
///
/// They are different verbs on the same machinery, and only one of them is
/// the engine's. `Shell::fork` names a CONVERSATION; the engine's prefix
/// cache names PHYSICAL PAGES and has no conversation id to give. A test of
/// one says nothing about the other's arithmetic: the plan states a page and
/// a row offset, and a row offset is a place inside a page that forking never
/// addresses.
///
/// # What it measures
///
/// Against the cache itself, read back, rather than against logits. A logit
/// comparison would say the copy landed SOMEWHERE right; reading the bytes
/// says which bytes. Both cache sides and a middle layer are checked, since
/// copying only layer 0 is a mutation the fork test already showed is
/// invisible to a single layer's numbers.
///
/// # The controls
///
/// 1. **A plan whose last cell names a page past the pool is refused, and the
///    pages named EARLIER in the same plan are unchanged.** This is the whole
///    reason the plan is walked twice. The C++ this replaces applies the page
///    moves first and notices the bad cell afterwards, leaving a cache that
///    is half somebody else's with no way back.
/// 2. **A foreign memory domain is refused by name.** A plan addressed to
///    another backend's memory, served here, would copy the right bytes into
///    the wrong device's pages.
/// 3. **A mismatched page-id count is refused** rather than zipped to the
///    shorter of the two, which would silently drop moves.
///
/// # The mutations
///
/// Five, and one of them was a live survivor until the test was widened --
/// which is why the survivor is recorded rather than quietly fixed.
///
/// 1. The page moves applied BEFORE the cells are checked, which is the C++
///    order. Caught by control 1.
/// 2. A cell copying to the end of its page instead of one row. **Survived**
///    the first version of this test, because the only cell then had offsets
///    3 and 1 and "the rest of the page" was one row anyway. A second cell at
///    offset 0 with three rows behind it, plus the rows-around assertions,
///    catches it.
/// 3. The cell's source and destination swapped.
/// 4. The domain check removed.
/// 5. The page-pairing check removed.
#[test]
fn the_engine_s_copy_plan_moves_what_it_names_and_nothing_when_it_is_refused() {
    let (device, _dir) = gpu!();
    use driver_vulkan::resources::{Pool, Shape};

    fn as_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    let shape = Shape {
        layers: 3,
        pages: 6,
        page_size: 4,
        kv_heads: 2,
        head_dim: 4,
        bytes: 4,
    };
    let pool = Pool::open(&device, shape).expect("a pool");

    // Every element distinct, and a function of where it is: a copy that
    // landed one page or one row off is then a different number rather than a
    // plausible one.
    let row = shape.row() as usize;
    let per_layer = shape.elements() as usize;
    let mark = |layer: u16, values: bool, slot: usize| -> f32 {
        (1 + layer as usize) as f32 * 1_000_000.0
            + if values { 500_000.0 } else { 0.0 }
            + slot as f32
    };
    for layer in 0..shape.layers {
        for values in [false, true] {
            let filled: Vec<f32> = (0..per_layer).map(|i| mark(layer, values, i)).collect();
            let buffer = pool.cache(layer, values).expect("a layer");
            device.write(buffer, &as_bytes(&filled)).expect("fill");
        }
    }
    let read = |pool: &Pool, layer: u16, values: bool| -> Vec<f32> {
        let bytes = device
            .read(pool.cache(layer, values).expect("a layer"))
            .expect("read");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let before: Vec<Vec<f32>> = (0..shape.layers)
        .flat_map(|l| [read(&pool, l, false), read(&pool, l, true)])
        .collect();

    // One whole page, 0 -> 5, and one row, page 1 row 3 -> page 4 row 1.
    // The row cell crosses both a page and a row boundary on purpose.
    let plan = driver_api::KvCopyPlan {
        src_domain: driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
        dst_domain: driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
        src_page_ids: vec![0],
        dst_page_ids: vec![5],
        cells: vec![
            driver_api::KvMoveCell {
                src_page_id: 1,
                src_token_offset: 3,
                dst_page_id: 4,
                dst_token_offset: 1,
            },
            // A second cell whose two offsets are EQUAL and low. A cell copy
            // that ran to the end of the page instead of one row is a no-op
            // for the cell above -- its offsets are 3 and 1, so "the rest of
            // the page" is one row anyway. This one leaves three rows behind
            // it to be clobbered, and the rows-around assertion sees it.
            driver_api::KvMoveCell {
                src_page_id: 1,
                src_token_offset: 0,
                dst_page_id: 3,
                dst_token_offset: 0,
            },
        ],
        ..driver_api::KvCopyPlan::default()
    };
    // Through the pool directly rather than through a `Shell`, because a
    // shell needs a model and this is arithmetic about a cache.
    let moved = pool.copy_plan(&device, &plan).expect("a sound plan");
    assert_eq!(moved, 3, "one page and two cells");

    for layer in 0..shape.layers {
        for values in [false, true] {
            let got = read(&pool, layer, values);
            let src = shape.slot(0, 0, 0, 0) as usize;
            let dst = shape.slot(5, 0, 0, 0) as usize;
            let n = shape.page_size as usize * row;
            assert_eq!(
                &got[dst..dst + n],
                &(0..n)
                    .map(|i| mark(layer, values, src + i))
                    .collect::<Vec<_>>()[..],
                "layer {layer} values={values}: the page move"
            );
            let src = shape.slot(1, 3, 0, 0) as usize;
            let dst = shape.slot(4, 1, 0, 0) as usize;
            assert_eq!(
                &got[dst..dst + row],
                &(0..row)
                    .map(|i| mark(layer, values, src + i))
                    .collect::<Vec<_>>()[..],
                "layer {layer} values={values}: the row cell"
            );
            // ...the ROWS AROUND the cell in its own page are untouched. A
            // cell copy whose length was the rest of the page rather than one
            // row passes every assertion above -- this is the only one that
            // notices, and it was a live survivor until it was written.
            for (page, keep) in [(4u32, 0u32), (4, 2), (4, 3), (3, 1), (3, 2), (3, 3)] {
                let at = shape.slot(page, keep, 0, 0) as usize;
                assert_eq!(
                    &got[at..at + row],
                    &(0..row)
                        .map(|i| mark(layer, values, at + i))
                        .collect::<Vec<_>>()[..],
                    "layer {layer} values={values}: page {page} row {keep}, which \
                     no cell names"
                );
            }
            // ...and page 2, which the plan never names, is untouched. A copy
            // whose length was a whole layer rather than a page would pass
            // both assertions above.
            let at = shape.slot(2, 0, 0, 0) as usize;
            let n = shape.page_size as usize * row;
            assert_eq!(
                &got[at..at + n],
                &(0..n)
                    .map(|i| mark(layer, values, at + i))
                    .collect::<Vec<_>>()[..],
                "layer {layer} values={values}: a page the plan never named"
            );
        }
    }

    // Control 1. The page move is sound and the cell is not; nothing moves.
    let pool2 = Pool::open(&device, shape).expect("a second pool");
    for layer in 0..shape.layers {
        for values in [false, true] {
            let filled: Vec<f32> = (0..per_layer).map(|i| mark(layer, values, i)).collect();
            device
                .write(
                    pool2.cache(layer, values).expect("a layer"),
                    &as_bytes(&filled),
                )
                .expect("fill");
        }
    }
    let mut bad = plan.clone();
    bad.cells[0].dst_page_id = shape.pages;
    let refused = pool2
        .copy_plan(&device, &bad)
        .expect_err("a cell past the pool");
    assert!(
        format!("{refused:?}").contains("cell 0"),
        "the refusal does not say which cell: {refused:?}"
    );
    let after: Vec<Vec<f32>> = (0..shape.layers)
        .flat_map(|l| [read(&pool2, l, false), read(&pool2, l, true)])
        .collect();
    assert_eq!(
        after, before,
        "a refused plan moved its pages anyway, so the cache is half-copied"
    );

    // Control 2. A foreign domain.
    let mut foreign = plan.clone();
    foreign.src_domain = driver_api::PIE_MEMORY_DOMAIN_METAL_SHARED;
    let refused = pool2
        .copy_plan(&device, &foreign)
        .expect_err("another backend's memory");
    assert!(format!("{refused:?}").contains("domain"), "{refused:?}");

    // Control 3. More sources than destinations.
    let mut lopsided = plan.clone();
    lopsided.src_page_ids.push(2);
    let refused = pool2
        .copy_plan(&device, &lopsided)
        .expect_err("unpaired pages");
    assert!(
        format!("{refused:?}").contains("2 source pages"),
        "{refused:?}"
    );

    pool.close(&device);
    pool2.close(&device);
}

/// The cache can be grown and shrunk under a live conversation without
/// changing a single logit it produces.
///
/// # Why this is the claim
///
/// `driver-metal`'s pool is sparse: it commits and releases pages without
/// moving an address, because Metal binds its heap once. This pool
/// REALLOCATES -- `sparseBinding` is an optional Vulkan feature and this
/// driver's whole premise is running where the optional features are absent.
/// That is only safe because every descriptor here is written during the step
/// that uses it, so no address survives a step for a resize to invalidate.
///
/// That premise is exactly what a test can break. So: prefill a conversation,
/// resize the cache out from under it twice, and require the next token's
/// WHOLE distribution to be bit-identical to the same conversation's on a
/// pool nobody touched. A reallocation that dropped the cache, or copied it
/// at the wrong stride, still answers -- with an argmax that is often still
/// right, which is why the comparison is over every logit.
///
/// # The controls
///
/// 1. **The two runs are the same shell**, one after the other, with the
///    conversation re-seated under a new id. A second shell would compare two
///    devices and two uploads as well as two caches.
/// 2. **A shrink that would drop a held page is refused BY NAME**, saying
///    which conversation and which page -- and the conversation still answers
///    afterwards, so the refusal did not half-apply.
/// 3. **A resize of a pool that is not the KV one is a no-op**, not a
///    refusal: the engine's trim task asks about three pools every tick and
///    two of them hold nothing here.
///
/// # The mutations
///
/// Three, all refused: dropping the copy in `Pool::resize` (the grown cache
/// is zeros); copying `pages` rows instead of `pages * page_size * row`; and
/// letting `Book::resize` shrink past a held page, which strands it and is
/// caught by control 2.
#[test]
fn a_cache_resized_under_a_conversation_does_not_change_its_answer() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so resizing is unmeasured");
        return;
    };
    // Two runs of the same conversation live at once, and the pool is then
    // grown past this and shrunk under it.
    let mut shell = shelled(dir, &REALS[0], real, 10);

    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..5 {
        prompt.extend_from_slice(&PERIOD);
    }
    prompt.push(PERIOD[0]);
    prompt.push(PERIOD[1]);

    fn last(shell: &mut driver_vulkan::shell::Shell, who: u64, tokens: &[u32]) -> Vec<f32> {
        let step = shell
            .step(&[driver_vulkan::turns::Turn {
                who,
                tokens: tokens.to_vec(),
            }])
            .unwrap_or_else(|e| panic!("{e}"));
        step.logits
            .row(step.rows - 1)
            .expect("the last row of the fire is the turn's own")
            .to_vec()
    }
    let resize = |shell: &mut driver_vulkan::shell::Shell, pages: u64| {
        shell
            .resize_pool(&driver_api::PoolResizePlan {
                pool_id: driver_api::PIE_ELASTIC_POOL_KV,
                target_pages: pages,
                ..driver_api::PoolResizePlan::default()
            })
            .unwrap_or_else(|e| panic!("resize to {pages}: {e}"));
    };

    // The undisturbed run first, so the number this compares against was not
    // produced by the machinery under test.
    last(&mut shell, 1, &prompt);
    let want = last(&mut shell, 1, &[PERIOD[2]]);

    // ...and the same conversation again, under a new id, with the cache
    // grown and then shrunk between the prefill and the decode.
    last(&mut shell, 2, &prompt);
    // The highest page anybody holds, plus one: the tightest cache that
    // strands nobody. Conversation 1 is still seated from the run above and
    // holds the low pages, so this is 2's pages and not merely their count.
    let held = shell
        .book()
        .pages(2)
        .expect("the conversation holds pages")
        .iter()
        .copied()
        .max()
        .expect("at least one page")
        + 1;
    assert!(held > 1, "the premise: the conversation holds pages");
    resize(&mut shell, 12);
    assert_eq!(shell.shape().pages, 12, "the pool did not grow");
    // One page past the highest anybody holds: the tightest cache that
    // strands nobody AND leaves the conversation the page its next token
    // needs, since thirty-two tokens fill two sixteen-row pages exactly.
    resize(&mut shell, u64::from(held) + 1);
    assert_eq!(shell.shape().pages, held + 1, "the pool did not shrink");
    let got = last(&mut shell, 2, &[PERIOD[2]]);
    assert_eq!(
        got, want,
        "growing and shrinking the cache changed what the conversation answers"
    );

    // Control 2. A shrink that would drop a page conversation 2 is sitting on.
    let refused = shell
        .resize_pool(&driver_api::PoolResizePlan {
            pool_id: driver_api::PIE_ELASTIC_POOL_KV,
            target_pages: u64::from(held) - 1,
            ..driver_api::PoolResizePlan::default()
        })
        .expect_err("the last page is held");
    let said = format!("{refused}");
    assert!(
        said.contains("conversation 2") && said.contains(&format!("page {}", held - 1)),
        "the refusal does not say who is in the way: {said}"
    );
    assert_eq!(
        shell.shape().pages,
        held + 1,
        "a refused shrink resized the pool anyway"
    );
    // ...and 2 still answers, so nothing half-applied. Its own id, because a
    // fresh conversation would need pages this cache no longer has.
    let again = last(&mut shell, 2, &[PERIOD[3]]);
    assert!(again.iter().all(|v| v.is_finite()));

    // Control 3. A pool this backend does not have.
    let pages = shell.shape().pages;
    shell
        .resize_pool(&driver_api::PoolResizePlan {
            pool_id: driver_api::PIE_ELASTIC_POOL_STATE,
            target_pages: 0,
            ..driver_api::PoolResizePlan::default()
        })
        .expect("a pool that holds nothing is resized by doing nothing");
    assert_eq!(
        shell.shape().pages,
        pages,
        "the state pool's target was applied to the KV pool"
    );

    // Control 4. That same absent pool, asked for STORAGE.
    //
    // Control 3 passing is not enough, and for a while it was all there was:
    // `resize_pool` answered `Ok(())` to every target on a non-KV id, so it
    // passed control 3 by being right about zero and would have passed it just
    // as well by being indiscriminate. The two are told apart only here.
    //
    // Why a lie here is expensive rather than merely untidy: `bootstrap`'s
    // trim task records a successful target in `applied` and then skips that
    // pool on every later tick. So one false `Ok` does not lose one request,
    // it convinces the engine for the rest of the process that a pool with no
    // bytes behind it is holding the pages it asked for.
    for (id, what) in [
        (driver_api::PIE_ELASTIC_POOL_STATE, "the state pool"),
        (driver_api::PIE_ELASTIC_POOL_WORKSPACE, "the workspace pool"),
        (u64::MAX, "an id no driver defines"),
    ] {
        let refused = shell
            .resize_pool(&driver_api::PoolResizePlan {
                pool_id: id,
                target_pages: 8,
                ..driver_api::PoolResizePlan::default()
            })
            .unwrap_err();
        let said = format!("{refused}");
        assert!(
            said.contains('8') && said.contains("no such pool"),
            "the refusal for {what} does not say what could not be done: {said}"
        );
        assert_eq!(
            shell.shape().pages,
            pages,
            "{what}'s target was applied to the KV pool"
        );
    }

    // And the KV pool still answers, so none of those refusals touched it.
    let after = last(&mut shell, 2, &[PERIOD[3]]);
    assert!(after.iter().all(|v| v.is_finite()));
}

/// A decode step answers in tens of milliseconds, not in seconds.
///
/// # Why a timing test exists at all here
///
/// Because a stall in this driver has already shipped once. A KV copy that
/// took 370 seconds was found by running something, not by any test in this
/// file: every correctness gate passed throughout, because the answers were
/// right and only the clock was wrong. This is the guard that class of defect
/// deserves -- an ORDER OF MAGNITUDE tripwire, not a benchmark.
///
/// # The profile behind the ceiling
///
/// Measured on a 4090, release build, no validation layers, qwen3-0.6b at
/// 4 bits, a single conversation decoding one token at a time. 18 ms a step,
/// about 55 tokens a second at this context.
///
/// The shape of the step, taken when it still cost 44 ms:
///
/// | phase | share |
/// |---|---|
/// | `lower` | 0.8-6 ms |
/// | pool stage + state upload | 2.5-6 ms |
/// | the arena (326 KB, allocated per step) | 0.2 ms |
/// | `fire` | the rest |
/// | ...of which descriptor writes, 452 launches | 0.1 ms |
/// | ...of which building the run list | 0.03 ms |
/// | ...of which recording the command buffer | 0.3 ms |
/// | ...of which **submit and wait on the GPU** | **almost all of it** |
///
/// That table used to be followed by "and still true in proportion". It is
/// not, and remeasuring it is what found the arena defect below. At today's
/// 5.4 ms, release, per phase:
///
/// | phase | then | now |
/// |---|---|---|
/// | `lower` | 0.8-6 ms | **0.3 us** |
/// | pool stage | 2.5-6 ms | **3 us** |
/// | state upload | (above) | **0.6 us** |
/// | the arena | 0.2 ms | 0.19 ms |
/// | `fire` | the rest | **5.25 ms, 89%** |
///
/// The two multi-millisecond host phases are gone -- the lowering because it
/// is cached per step, the stage because the pool stopped rewriting tables
/// that had not changed -- and they did not shrink, they collapsed, by three
/// and four orders of magnitude. So the CONCLUSION survived while every
/// number under it stopped being true, which is the failure mode a
/// proportional claim invites.
///
/// The conclusion, restated from the measurement rather than inherited: this
/// crate's own CPU work is now about 0.2 ms of 5.4, and essentially all of
/// that is the arena's allocation. The fire is 89%.
///
/// # What remeasuring it found
///
/// On a DECODE the arena is 326 KB and 0.19 ms, which is where it always was
/// and is dull. On the 384-row prefill in the sibling test it was **35.5 ms
/// of a 167 ms step**, because there the arena is 233 MB -- and it was being
/// zero-filled in system memory and then uploaded whole. See
/// `a_prefills_arena_does_not_cross_the_bus`, and `turns::arena_for` for the
/// route it takes now. The decode profile above could not have found that:
/// the phase is 3% of a decode and 21% of a prefill, and this test only ever
/// looked at the decode.
///
/// # Where the other half went, and four wrong answers on the way
///
/// The device half was 38 ms of the 44, and every kernel in it ran at about
/// 12 GB/s on a card that does roughly a thousand -- the lm_head launch
/// reading its 78 MB of weights, and equally the small launches reading
/// 1.5 MB. The card drew 85 W of 450 at "100% utilisation": resident and
/// stalled.
///
/// Four hypotheses were tested and refuted, and they are recorded because
/// each cost an experiment and because being wrong four times is what
/// eventually pointed at the right thing:
///
/// 1. **Redundant loads in the quantised matvec.** `dot_lane` re-fetches the
///    same packed word once per code -- eight times at 4 bits -- and the
///    scale and bias once per element. Rewriting it around
///    `pie_affine_word_dot`, one word load and one scale per eight MACs,
///    changed the step from 44.4 ms to 44.3 ms. The shader compiler was
///    already hoisting them.
/// 2. **The serial reduction.** `reduce_store` has lane 0 sum 32 shared
///    floats per row while 62 of 64 lanes idle. Deleting the reduction
///    outright -- wrong answers, pure probe -- gave 45.1 ms. It is free.
/// 3. **Barrier serialization.** Every dispatch is separated by a global
///    memory barrier, so no two ever overlap. Removing every barrier -- also
///    wrong answers -- gave 39.6 ms. Worth about 5 ms of 44, real but not the
///    story.
/// 4. **Occupancy in the matvec.** 32 lanes to a row and 64 threads to a
///    workgroup is little to hide memory latency behind. Widening it to 64
///    lanes and 128 threads gave 31.5 ms against 31.8 ms. Nothing.
///
/// The clue was in what all four had in common: 12 GB/s at EVERY launch size
/// and for every kernel, which is not what a shader's decomposition looks
/// like. It is what a bus looks like. `Device::buffer` asked for the first
/// memory type that was `HOST_VISIBLE | HOST_COHERENT` -- and on this card
/// that is type 2, which is system RAM. Every weight, every KV page and every
/// activation lived across PCIe, and 12 GB/s is what PCIe 4.0 x16 gives.
/// Type 4 is `DEVICE_LOCAL` as well, mappable across the whole 24 GB because
/// resizable BAR is on. Preferring it took the step from 31.8 ms to 18.1 ms,
/// and a 1536-token step from 110 ms to 59 ms.
///
/// The kernels were never the ceiling. They may yet be worth improving --
/// nothing above proves they are good, only that they were not what was
/// wrong -- but that is a `kernels-vulkan` question and no longer an urgent
/// one.
///
/// # Why the ceiling is where it is
///
/// 250 ms per step, against 5.4 ms measured in release and 12.0 to 20.1 ms
/// across five debug runs of this suite. Those numbers are not the ones this
/// header carried for most of its life -- it said 18 ms release and about
/// 70 ms debug, from before the wider `QMM_TILE`, the cached per-step
/// lowering and the hazard-analysed barriers landed. A stale figure in a
/// timing note is worse than none: it is what a reader compares against when
/// deciding whether a slowdown is real, and 44 ms would have looked like an
/// improvement.
///
/// The margin is for the shared box rather than the driver, and the spread
/// above is why it stays wide: the same build varies by 1.7x between runs
/// depending on what else holds the card. The number to catch is a stall of
/// seconds, not a slow afternoon, and a tighter bound would be a benchmark --
/// which on a shared machine is a flaky test. 250 ms is twelve times the
/// slowest run observed, which is as tight as that spread allows.
#[test]
fn a_decode_step_does_not_stall() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the step cost is unmeasured");
        return;
    };
    let mut shell = shelled(dir, &REALS[0], real, 64);

    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..4 {
        prompt.extend_from_slice(&PERIOD);
    }
    let step = |shell: &mut driver_vulkan::shell::Shell, tokens: Vec<u32>| {
        shell
            .step(&[driver_vulkan::turns::Turn { who: 1, tokens }])
            .unwrap_or_else(|e| panic!("{e}"));
    };
    step(&mut shell, prompt);
    // Warm: the first decode after a prefill builds pipelines this one reuses,
    // and charging pipeline construction to a steady-state ceiling would
    // measure the cache rather than the fire.
    for _ in 0..2 {
        step(&mut shell, vec![PERIOD[0]]);
    }

    const STEPS: u32 = 8;
    let at = std::time::Instant::now();
    for _ in 0..STEPS {
        step(&mut shell, vec![PERIOD[0]]);
    }
    let each = at.elapsed() / STEPS;
    eprintln!("a decode step: {each:?}");
    within_budget(
        each,
        std::time::Duration::from_millis(250),
        &format!(
            "a decode step took {each:?}, and the measured cost on this card is 5.4 ms in \
             release and 12 to 20 ms in this debug suite: something is stalling, which is \
             how the 370 s KV copy looked from here"
        ),
    );
}

/// Where a decode step's time actually goes, and how that changes with context.
///
/// # Why this test exists
///
/// Because until `PIE_VULKAN_TIMING` existed, every performance number in this
/// crate was wall-clock around a submit, and a decode is four hundred and
/// fifty dispatches. "The step takes 7 ms" does not say which dispatch to go
/// and look at, so the choice of what to optimise next was being made from an
/// argument rather than a measurement.
///
/// The first thing the timestamps said looked like the argument was wrong.
/// The 4-bit qwen3-0.6b on this card, device time per decode step at
/// twenty-four tokens of history:
///
/// | kernel | ms/step | share |
/// | --- | --- | --- |
/// | `affine_qmv_fast` | 2.16 | 48% |
/// | `affine_qmv_fast_residual` | 0.95 | 21% |
/// | `sdpa_paged_decode` | 0.86 | 19% |
/// | `rms_single_row` | 0.30 | 7% |
/// | `neox_mb` | 0.16 | 4% |
/// | `kv_append_paged` | 0.05 | 1% |
/// | `silu_mul` | 0.04 | 1% |
///
/// Attention a fifth, the projections seven tenths -- which reads as a flat
/// contradiction of the standing next target, occupancy in
/// `sdpa_paged_decode`. It is not. Run the same measurement at 384 tokens and
/// it turns over completely:
///
/// | kernel | ms/step | share |
/// | --- | --- | --- |
/// | `sdpa_paged_decode` | 10.14 | 75% |
/// | `affine_qmv*` | 2.92 | 22% |
/// | everything else | 0.51 | 3% |
///
/// Both are true, and neither is the whole cost model. The projections read
/// the weights, which do not grow, so their cost is FIXED per step; attention
/// reads the history, so its cost is LINEAR in it. At twenty-four tokens the
/// fixed part is everything and at 384 it is a fifth. A single number for
/// "where a decode goes" does not exist, and this test is written to hold
/// both ends of that rather than to pick one -- because picking one is how a
/// tuning effort ends up aimed at the phase that was already cheap.
///
/// A caution about the absolute numbers, which is why only shares are
/// asserted below: the tool perturbs what it measures. The submit-and-wait
/// around a short-context fire is 3.47 ms with timing off and 5.78 ms with it
/// on -- see [`Device::timings`] -- so the device milliseconds above are an
/// upper bound. The shares survive because the effect they describe, a fifth
/// against three quarters, is far larger than two thirds of overhead.
///
/// The host was measured separately, with timing OFF and the phases of
/// `run_all` timed against the wall, which is the only way to get it
/// unperturbed. Release, per decode step:
///
/// | phase | short (24 tok) | long (384 tok) |
/// | --- | --- | --- |
/// | argument checks | 0.007 ms | 0.010 ms |
/// | descriptor sets | 0.134 ms | 0.113 ms |
/// | command recording | 0.421 ms | 0.349 ms |
/// | submit and wait | 3.469 ms | 12.961 ms |
/// | everything outside `run_all` | 1.917 ms | 1.867 ms |
/// | wall | 5.949 ms | 15.300 ms |
///
/// The host is **2.48 ms and it does not move with context** -- 42% of a
/// short step and 15% of a long one -- and three quarters of it is outside
/// `run_all` entirely, in the lowering, the plan and the scalar blocks. That
/// is not a defect, it is what a driver costs; it is written down because at
/// short context it is nearly three times the whole attention kernel, and it
/// had never appeared in any measurement here.
///
/// # What is asserted, and why they are ratios
///
/// Absolute milliseconds on a shared box vary by 1.7x between runs, so the
/// claims are about shares, which do not: the shape of a decode is a property
/// of the model and the kernels, not of what else holds the card. The bounds
/// are wide enough to survive any single kernel getting faster and tight
/// enough to fail if an end inverts -- which is the event worth catching,
/// since it would mean the next optimisation belongs somewhere else.
#[test]
fn attention_is_a_fifth_of_a_short_step_and_three_quarters_of_a_long_one() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the step's shape is unmeasured");
        return;
    };
    if std::env::var_os("PIE_VULKAN_TIMING").is_none() {
        eprintln!(
            "PIE_VULKAN_TIMING is not set, so the device wrote no timestamps. \
             Re-run with it set to see where a step goes."
        );
        return;
    }

    // `repeats` copies of PERIOD is the history; `pages` has to hold it.
    let shape = |repeats: usize, pages: u32| -> (f64, f64, f64, std::time::Duration) {
        let mut shell = shelled(dir, &REALS[0], real, pages);
        let mut prompt: Vec<u32> = Vec::new();
        for _ in 0..repeats {
            prompt.extend_from_slice(&PERIOD);
        }
        let step = |shell: &mut driver_vulkan::shell::Shell, tokens: Vec<u32>| {
            shell
                .step(&[driver_vulkan::turns::Turn { who: 1, tokens }])
                .unwrap_or_else(|e| panic!("{e}"));
        };
        step(&mut shell, prompt);
        // Warm: the first decode after a prefill builds pipelines, and the
        // prefill's own dispatches are in these totals until they are
        // subtracted off as the baseline below.
        for _ in 0..2 {
            step(&mut shell, vec![PERIOD[0]]);
        }
        let base = shell.device().timings();
        const STEPS: u32 = 4;
        let at = std::time::Instant::now();
        for _ in 0..STEPS {
            step(&mut shell, vec![PERIOD[0]]);
        }
        let wall = at.elapsed() / STEPS;
        let rows = shell.device().timings();
        assert_eq!(
            shell.device().timings_skipped(),
            0,
            "a fire was too big to time, so these totals are missing part of a step"
        );
        let since = |name: &str| -> f64 {
            let sum = |v: &[(String, f64, u32)]| -> f64 {
                v.iter()
                    .filter(|(k, _, _)| k.starts_with(name))
                    .map(|(_, ms, _)| ms)
                    .sum()
            };
            sum(&rows) - sum(&base)
        };
        let total = since("");
        assert!(total > 0.0, "the device reported no time for four steps");
        for (name, ms, n) in &rows {
            eprintln!("  {name:<48} {ms:>9.3} ms  x{n}");
        }
        eprintln!(
            "  {} tokens: wall {wall:?}/step, device {:.3} ms/step",
            repeats * PERIOD.len(),
            total / f64::from(STEPS)
        );
        (
            since("sdpa_paged_decode") / total,
            since("affine_qmv") / total,
            total / f64::from(STEPS),
            wall,
        )
    };

    let (short_attn, short_proj, short_gpu, short_wall) = shape(4, 64);
    let (long_attn, long_proj, long_gpu, long_wall) = shape(64, 512);
    eprintln!(
        "short: attention {:.0}%, projections {:.0}%, {:.0}% of the wall step off-device\n\
         long:  attention {:.0}%, projections {:.0}%, {:.0}% of the wall step off-device",
        short_attn * 100.0,
        short_proj * 100.0,
        (1.0 - short_gpu / short_wall.as_secs_f64() / 1000.0) * 100.0,
        long_attn * 100.0,
        long_proj * 100.0,
        (1.0 - long_gpu / long_wall.as_secs_f64() / 1000.0) * 100.0,
    );

    // Short context: the fixed cost is the whole cost.
    assert!(
        short_attn < 0.35,
        "at 24 tokens attention is {:.0}% of device time, measured at 19%",
        short_attn * 100.0
    );
    assert!(
        short_proj > 0.45,
        "at 24 tokens the projections are {:.0}% of device time, measured at 69%",
        short_proj * 100.0
    );
    // Long context: attention has overtaken everything, which is what makes
    // its occupancy the target it is written down as.
    assert!(
        long_attn > 0.55,
        "at 384 tokens attention is {:.0}% of device time, measured at 75%: if it \
         really has fallen this far the occupancy work is no longer the next target",
        long_attn * 100.0
    );
    assert!(
        long_proj < 0.40,
        "at 384 tokens the projections are {:.0}% of device time, measured at 22%",
        long_proj * 100.0
    );
    // And the crossover itself, which is the claim neither end makes alone.
    assert!(
        long_attn > short_attn * 2.0,
        "attention went from {:.0}% to {:.0}% of device time over 16x the history; \
         it is supposed to be the part that grows with context",
        short_attn * 100.0,
        long_attn * 100.0
    );
}

/// The timing tool is off unless it is asked for.
///
/// Not a formality. Two `vkCmdWriteTimestamp`s per dispatch and a
/// `vkGetQueryPoolResults` per fire is a cost every user of this driver would
/// pay for a number none of them read, and "it is opt-in" is exactly the kind
/// of claim that is true when written and quietly false a year later.
#[test]
fn timing_costs_nothing_when_it_was_not_asked_for() {
    let (device, _dir) = gpu!();
    if std::env::var_os("PIE_VULKAN_TIMING").is_some() {
        eprintln!("PIE_VULKAN_TIMING is set, so there is nothing to say about it unset");
        return;
    }
    assert!(device.timings().is_empty());
    assert_eq!(device.timings_skipped(), 0);
}

/// A LONG conversation's decode step does not stall either.
///
/// # Why the short one is not enough
///
/// Because the two are bounded by different things, and the sibling above
/// only ever sees twenty-four tokens of context. Decode cost has two parts:
/// a fixed part -- twenty-eight layers of weights read whatever the history
/// -- and a part that grows with the history, which is attention reading
/// every key it has kept. At twenty-four tokens the second part is invisible.
///
/// It was measured, and it was not small. Release, no layers, this card:
///
/// | context | decode step |
/// |---|---|
/// | 24 | 38.0 ms |
/// | 384 | 100.6 ms |
/// | 1536 | 306.4 ms |
///
/// 0.177 ms per token of history then, so at a thousand tokens -- an ordinary
/// conversation -- attention was five sixths of every step, and the crate's
/// whole performance story had been written from a twenty-four token prompt
/// where it looked like a rounding error.
///
/// # What that measurement found
///
/// A decode workgroup in `attn/sdpa_paged.slang` has one thread per head
/// dimension, and every one of those threads was walking the entire query and
/// key vectors to arrive at the same scalar score. A hundred and twenty-eight
/// threads computing one number a hundred and twenty-eight times. Having the
/// workgroup add its own terms in a tree instead:
///
/// | context | before | cooperative | and in VRAM |
/// |---|---|---|---|
/// | 24 | 38.0 ms | 31.8 ms | 18.1 ms |
/// | 384 | 100.6 ms | 49.9 ms | 31.8 ms |
/// | 1536 | 306.4 ms | 110.1 ms | 59.2 ms |
///
/// 0.052 ms per token, and 2.8x faster where it matters; the third column is
/// the memory-type fix `a_decode_step_does_not_stall` describes, which is
/// worth another 1.9x and is a separate finding. Which is why this test
/// exists: the defect was not visible at all from the short context, and
/// nothing would have noticed it coming back.
///
/// # Why the ceiling is where it is, and why 384 and not 1536
///
/// 500 ms, against 15.0 ms measured in release at this context and 21.4 to
/// 32.6 ms across debug runs of this suite. As with the short sibling, the
/// figures this header used to carry -- 31.8 ms release, about 107 ms debug
/// -- predate the `QMM_TILE` widening and the cached lowering, and a stale
/// number in a timing note is what a reader compares a real slowdown
/// against.
///
/// Fifteen times the slowest debug run observed: margin for a shared box
/// rather than for the kernel, and the spread between those runs is itself
/// 1.5x. It would still catch the hundred-and-twenty-eight-fold regression
/// it was written for by a wide margin.
///
/// The context is 384 tokens and not the 1536 the table above measures,
/// because a 1536-token prefill does not finish inside `run_all`'s
/// ten-second fence wait in a debug build with GPU-assisted validation on.
/// That is a property of the layers, not of the driver -- but it is the
/// configuration this suite runs in, and a test that trips a timeout is a
/// test about the timeout. The growth term is perfectly visible at 384: it
/// is two thirds of the step.
///
/// Trying it at 1536 was not wasted. The timeout was reported correctly and
/// then buried, because freeing the fire's scalar block afterwards tripped a
/// validation error this driver treats as fatal, so the process aborted on
/// the consequence and never printed the cause. `Device::run_all` now waits
/// for the device to go idle before a failed fire returns.
#[test]
fn a_long_conversations_decode_step_does_not_stall() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the step cost is unmeasured");
        return;
    };
    let mut shell = shelled(dir, &REALS[0], real, 512);

    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..64 {
        prompt.extend_from_slice(&PERIOD);
    }
    let context = prompt.len();
    let step = |shell: &mut driver_vulkan::shell::Shell, tokens: Vec<u32>| {
        shell
            .step(&[driver_vulkan::turns::Turn { who: 1, tokens }])
            .unwrap_or_else(|e| panic!("{e}"));
    };
    step(&mut shell, prompt);
    for _ in 0..2 {
        step(&mut shell, vec![PERIOD[0]]);
    }

    const STEPS: u32 = 4;
    let at = std::time::Instant::now();
    for _ in 0..STEPS {
        step(&mut shell, vec![PERIOD[0]]);
    }
    let each = at.elapsed() / STEPS;
    eprintln!("a decode step at {context} tokens: {each:?}");
    within_budget(
        each,
        std::time::Duration::from_millis(500),
        &format!(
            "a decode step at {context} tokens of history took {each:?}, and the measured \
             cost is 15 ms in release and 21 to 33 ms in this debug suite: attention is \
             reading the history far too many times, which is exactly the defect this test \
             was written for"
        ),
    );
}

/// The pages a GROW adds read as zeros, not as whatever was in that memory.
///
/// # Why this is its own test
///
/// It was not, and a mutation proved the gap. Deleting the zero-fill from
/// `Pool::resize`'s grow path left all sixty-one tests in this file green,
/// including the one that grows a pool under a live conversation and checks
/// its answer -- because that conversation never READS the pages the grow
/// added, so the garbage in them never reaches a softmax.
///
/// It would reach one later. `sdpa_paged` reads a whole page and lets
/// `kv_len` decide what counts, and the first version of this pool zeroed by
/// construction: it built the new buffer as `vec![0u8; bytes]` on the host.
/// Moving the resize onto `vkCmdCopyBuffer` removed the host buffer and with
/// it, silently, the zeros -- Vulkan does not clear a fresh allocation, and
/// bf16 garbage includes NaN.
///
/// # What this can prove here, and what it cannot
///
/// It proves the OBSERVABLE property, which is the one a reader of the pool
/// cares about: the tail a grow adds reads as zeros. That is asserted on the
/// real pool, over every layer of both halves.
///
/// It does NOT kill the mutation that deletes the `vkCmdFillBuffer`. Removing
/// the fill leaves this test green, because this driver zeroes fresh device
/// memory for its own reasons -- most do, as a process-isolation guarantee --
/// and no fixture available here can produce an allocation that does not. The
/// fill stays because the guarantee is the IMPLEMENTATION's and not the
/// specification's, and this crate runs on whatever card is in front of it.
/// `zero_writes_only_the_range_it_names` below covers the part that is
/// actually this crate's: that `Device::zero` clears what it names and
/// nothing else.
///
/// Stated rather than left as a green tick, because a test whose subject is
/// unfalsifiable on the box it runs on is worth having and worth labelling.
#[test]
fn the_pages_a_grow_adds_are_zero() {
    let (_device, dir) = gpu!();
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so a grown page is unread");
        return;
    };
    let mut shell = shelled(dir, &REALS[0], real, 4);

    // Dirty the pool first, so "zero" cannot be satisfied by an allocator
    // that happened to hand back fresh pages. A conversation writes real KV
    // into the low pages, and the grow below reuses that memory or does not
    // -- either way the tail it adds is the part under test.
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..3 {
        prompt.extend_from_slice(&PERIOD);
    }
    shell
        .step(&[driver_vulkan::turns::Turn {
            who: 1,
            tokens: prompt,
        }])
        .unwrap_or_else(|e| panic!("{e}"));

    let before = shell.shape();
    let kept = before.layer_bytes();
    shell
        .resize_pool(&driver_api::PoolResizePlan {
            pool_id: driver_api::PIE_ELASTIC_POOL_KV,
            target_pages: 8,
            ..driver_api::PoolResizePlan::default()
        })
        .unwrap_or_else(|e| panic!("grow: {e}"));
    let after = shell.shape();
    assert_eq!(after.pages, 8, "the pool did not grow");

    let added = after.layer_bytes() - kept;
    assert!(added > 0, "the premise: a grow adds bytes");
    let pool = shell.pool();
    assert_eq!(
        pool.keys().len(),
        usize::from(after.layers),
        "one key buffer per layer"
    );
    for (which, buffers) in [("keys", pool.keys()), ("values", pool.values())] {
        for (layer, buffer) in buffers.iter().enumerate() {
            // `shell.device()`, not the suite's lock: a `Shell` opens its OWN
            // device, so these buffers were allocated from a different
            // `VkDevice` than `gpu!()` handed back. Reading them through that
            // one is a `commonparent` violation -- this card tolerates it and
            // the validation layer does not.
            let tail = shell
                .device()
                .read_at(buffer, kept, added)
                .unwrap_or_else(|e| panic!("{which} layer {layer}: {e}"));
            if let Some(at) = tail.iter().position(|b| *b != 0) {
                panic!(
                    "{which} layer {layer}: byte {at} of the {added} a grow added is \
                     {:#04x}, not zero. A fresh Vulkan allocation holds whatever was \
                     there, and `sdpa_paged` reads a whole page",
                    tail[at]
                );
            }
        }
    }
}

/// `Device::zero` clears exactly the range it names.
///
/// The primitive under the grow above, tested where it CAN be falsified: on a
/// buffer this test dirtied itself, so "already zero" is not available as an
/// answer. Both edges matter -- a fill that ran long would erase a page the
/// pool still holds, and one that ran short would leave the garbage this
/// exists to remove.
#[test]
fn zero_writes_only_the_range_it_names() {
    let (device, _dir) = gpu!();
    const N: usize = 4096;
    let dirty = vec![0xABu8; N];
    let buffer = device.buffer(&dirty).expect("a buffer");

    device.zero(&buffer, 1024, 2048).expect("zero the middle");
    let back = device.read(&buffer).expect("read it back");
    assert_eq!(back.len(), N);
    assert!(
        back[..1024].iter().all(|b| *b == 0xAB),
        "the bytes before the range were cleared"
    );
    assert!(
        back[1024..3072].iter().all(|b| *b == 0),
        "the range was not cleared"
    );
    assert!(
        back[3072..].iter().all(|b| *b == 0xAB),
        "the bytes after the range were cleared"
    );

    // Both refusals, because a fill that silently rounded to alignment or
    // silently clipped to the buffer would be a partial write reported as a
    // success -- and `vkCmdFillBuffer` requires the alignment rather than
    // handling it.
    let said = format!(
        "{}",
        device
            .zero(&buffer, 1, 8)
            .expect_err("an unaligned offset is refused")
    );
    assert!(said.contains("four-byte aligned"), "{said}");
    let said = format!(
        "{}",
        device
            .zero(&buffer, 0, N as u64 + 4)
            .expect_err("a range past the end is refused")
    );
    assert!(said.contains("in a"), "{said}");
    device.free(buffer);
}

/// A trim costs milliseconds, and a shallow one is not dearer than a deep one.
///
/// # What this replaced, and why the replacement is the interesting part
///
/// This test used to assert the OPPOSITE, and it was right to. The Vulkan
/// seam published `elastic_page_bytes: 0` and `elastic_budget_pages: 0`,
/// which is the pair `bootstrap` reads together before it starts a trim task
/// at all, so `Shell::resize_pool` was implemented, contents-preserving and
/// refusal-safe -- proven by
/// `a_cache_resized_under_a_conversation_does_not_change_its_answer` just
/// above -- and never reached in production.
///
/// The seam's first reason for the zero was "nothing can be given back
/// page-wise", which was false: a shrink frees the old buffers and takes
/// smaller ones. The measured reason was better. `Pool::resize` read every
/// layer's whole old buffer down to HOST memory and wrote the survivors back
/// up, through a mapping that reads at ten megabytes a second because
/// mappable VRAM is write-combined. So the charge was the pool twice, and the
/// delta did not enter it:
///
/// | shrink, 256 pages of qwen3-0.6b | cost |
/// |---|---|
/// | 255 -> 254 (one page) | 2.77 s |
/// | 254 -> 128 (126 pages) | 0.74 s |
///
/// Handing back one page cost nearly four times what handing back half the
/// pool cost, because the deeper cut filled a smaller destination. The
/// cheapest trim that pool offered was the largest one.
///
/// That test carried an instruction: it "still fails hard the day the premise
/// changes ... this assertion would go red, and the failure is the reminder
/// to go and advertise what by then would be true." The premise changed on
/// purpose. `Pool::resize` now moves what survives with `vkCmdCopyBuffer` and
/// zero-fills a grow's tail with `vkCmdFillBuffer`, so nothing crosses the
/// bus and no host memory is held. The same two shrinks:
///
/// | shrink | cost |
/// |---|---|
/// | 255 -> 254 (one page) | 20.5 ms |
/// | 254 -> 128 (126 pages) | 18.6 ms |
/// | 128 -> 256 (a grow) | 20.0 ms |
///
/// A hundred and thirty-five times cheaper, and flat rather than inverted:
/// what is left is dominated by taking and binding fifty-six allocations, not
/// by moving bytes. So the seam now advertises both numbers and the trim task
/// runs.
///
/// # What this asserts, and why so loosely
///
/// Two things, both lower bounds, both an order of magnitude clear of the
/// measurement, because this is a shared box behind two validation layers and
/// an assertion that tracked the number would fail whenever the other tenant
/// got busy:
///
/// - a shrink of any depth completes in under 500 ms, which is the property
///   that makes a ten-second trim tick sane. Measured: about 20 ms.
/// - a one-page shrink is not more than four times a half-pool shrink, which
///   is the INVERSION going away. Measured: 1.10x.
///
/// The second is the one that matters, and it is stated as a ratio rather
/// than as two durations so that it says something about the pool rather than
/// about the box. It goes red if a resize ever again charges for the pool
/// instead of for what it keeps.
#[test]
fn giving_back_one_page_costs_what_giving_back_half_the_pool_costs() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the resize cost is unmeasured");
        return;
    };
    // Big enough that the restage dominates the call's fixed overhead, small
    // enough to leave the box to its other tenant: 256 pages of qwen3-0.6b is
    // about 460 MB of cache.
    const FULL: u64 = 256;
    let mut shell = shelled(dir, &REALS[0], real, FULL as u32);

    let cost = |shell: &mut driver_vulkan::shell::Shell, to: u64| {
        let at = std::time::Instant::now();
        shell
            .resize_pool(&driver_api::PoolResizePlan {
                pool_id: driver_api::PIE_ELASTIC_POOL_KV,
                target_pages: to,
                ..driver_api::PoolResizePlan::default()
            })
            .unwrap_or_else(|e| panic!("resize to {to}: {e}"));
        assert_eq!(
            u64::from(shell.shape().pages),
            to,
            "the pool did not follow"
        );
        at.elapsed()
    };

    // Warm first, and DISCARDED: the first resize of a process pays for host
    // allocator growth the later ones reuse, and charging that to the
    // one-page case would prove this test's point by accident.
    let _ = cost(&mut shell, FULL - 1);

    let one = cost(&mut shell, FULL - 2);
    let half = cost(&mut shell, FULL / 2);

    // A grow, too: the seam advertises elasticity in both directions and the
    // trim task's target rises as well as falls.
    let grow = cost(&mut shell, FULL);

    for (what, took) in [
        ("one-page shrink", one),
        ("half-pool shrink", half),
        ("grow", grow),
    ] {
        within_budget(
            took,
            std::time::Duration::from_millis(500),
            &format!(
                "a {what} took {took:?}. A resize this dear cannot be on a ten-second tick, \
                 so `elastic_page_bytes` and `elastic_budget_pages` at the Vulkan seam are \
                 now overstating this pool and must go back to zero"
            ),
        );
    }
    assert!(
        one <= half * 4,
        "a one-page shrink took {one:?} and a {}-page shrink took {half:?}. A resize is \
         charging for the POOL rather than for what it KEEPS -- which is what the host \
         round-trip did before `Pool::resize` moved to `vkCmdCopyBuffer` -- and the \
         cheapest trim this pool offers is once again the largest one",
        FULL / 2 - 2,
    );
}

/// A growth the machine will not stage is retryable, and changes nothing.
///
/// `Shell::admit` grows the pool to whatever a frame NAMES, because the
/// engine owns page allocation and hands physical pages down. That growth can
/// fail, and until this it failed as a FAULT -- which, now that a driver lane
/// answers its token instead of hanging, means the user's request dies for a
/// condition that clears the moment something is evicted.
///
/// So the failure is classified: [`driver_vulkan::device::Failed::OutOfMemory`]
/// is a scheduling fact and everything else is a fault, and `admit` turns the
/// first into `Launched::Exhausted` -- the variant this crate declared,
/// documented, matched on at the engine seam, and produced nowhere.
///
/// # What this proves and what it cannot
///
/// It proves the classification and the recovery on a REAL pool: the request
/// is refused as retryable, the pool keeps the shape it had, and the
/// conversation sitting in it still answers. It cannot prove the arm where
/// the DEVICE refuses, because provoking that means running a shared machine
/// out of memory to exercise one comparison. That comparison is unit-tested
/// in `device.rs` instead, on the result codes themselves.
///
/// The path forced here is the host half: the staging buffer a resize builds
/// before it allocates. That is the same `Failed::OutOfMemory`, reached by
/// the same call, and it is also the second line of defence that a mutation
/// once found by killing the test binary with SIGABRT -- a `Vec` that cannot
/// be allocated aborts the process rather than returning.
#[test]
fn a_pool_growth_the_host_cannot_stage_is_retryable_rather_than_fatal() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the growth refusal is unmeasured");
        return;
    };
    let mut shell = shelled(dir, &REALS[0], real, 24);

    // A conversation in the pool, so "changed nothing" is a claim about
    // contents and not only about a number.
    let want = {
        let step = shell
            .step(&[driver_vulkan::turns::Turn {
                who: 1,
                tokens: PERIOD[..4].to_vec(),
            }])
            .expect("a first turn");
        step.logits
            .row(step.readout_of[0])
            .expect("a readout row")
            .to_vec()
    };
    let was = shell.shape().pages;

    // Three billion pages: a u32, so the plan is well-formed and the target
    // reaches the pool rather than being refused as unrepresentable. Its
    // staging buffer is tens of terabytes, which `try_reserve_exact` refuses
    // without touching a byte of real memory -- this test costs nothing to
    // run and does not put the machine under pressure.
    let refused = shell
        .resize_pool(&driver_api::PoolResizePlan {
            pool_id: driver_api::PIE_ELASTIC_POOL_KV,
            target_pages: 3_000_000_000,
            ..driver_api::PoolResizePlan::default()
        })
        .expect_err("no host stages a cache that size");
    let driver_vulkan::shell::Unresized::Device(e) = refused else {
        panic!("a growth nothing can stage was refused as a stranded page: {refused}");
    };
    assert!(
        e.is_out_of_memory(),
        "a growth the machine cannot stage was classified as a fault, so the \
         scheduler fails the request instead of evicting and re-posting: {e}"
    );

    assert_eq!(
        shell.shape().pages,
        was,
        "a refused growth resized the pool anyway, so the re-post the \
         classification asks for would run against a half-grown cache"
    );
    // The book was put back too, or the next page handed out is one the cache
    // does not have.
    let again = shell
        .step(&[driver_vulkan::turns::Turn {
            who: 1,
            tokens: PERIOD[..4].to_vec(),
        }])
        .expect("the conversation survives a refused growth");
    let _ = again;
    let fresh = {
        let step = shell
            .step(&[driver_vulkan::turns::Turn {
                who: 7,
                tokens: PERIOD[..4].to_vec(),
            }])
            .expect("a fresh conversation after a refused growth");
        step.logits
            .row(step.readout_of[0])
            .expect("a readout row")
            .to_vec()
    };
    assert_eq!(
        fresh, want,
        "the same prompt answered differently after a refused growth, so the \
         refusal left the pool or the book in a state it did not start in"
    );
}

/// A copy plan whose destination is above the pool GROWS it rather than being
/// refused.
///
/// # What found this, and why nothing else could
///
/// The curated inferlet sweep, on `prefix-tree-kv-cache`, and only when it
/// ran after the other thirty-eight:
///
/// ```text
///   next_token take: channel is poisoned: pipeline: forward failed:
///   pre-launch KV copy rejected: driver-vulkan: page move 0's destination
///   names page 3 row 0, and the pool has 3 pages of 16 rows
/// ```
///
/// Run alone it passed. That is the signature of a driver whose answer
/// depends on what preceded it, and the reason is that this pool is ELASTIC:
/// it holds what the frames so far have needed, not what the scheduler is
/// entitled to hand out. `Shell::admit` knows that and grows to the highest
/// page a frame NAMES. `Shell::copy_kv` is the other door a page number comes
/// through, and it did not: it went straight to `Pool::copy_plan`, whose
/// bounds check is right about the pool as it IS and has no way to know what
/// it could be. So a prefix share aimed one page past the last prefill's
/// high-water mark died, and the conversation died with it.
///
/// # What this measures
///
/// Both directions of the asymmetry, because the fix is not "grow for
/// anything named":
///
/// 1. A DESTINATION above the pool grows it, and the bytes land -- read back
///    from the grown pool and compared against the source page, so a growth
///    that reallocated without carrying the contents over is caught too.
/// 2. A SOURCE above the pool is still REFUSED. This pool only ever grows on
///    demand, so a page it has never held is a page nothing has ever written;
///    growing for it would turn a refusal into a copy of fresh zeros, which
///    is history-shaped silence rather than an error.
///
/// # Mutations this kills
///
/// * The growth removed: the first copy is refused with the sweep's message.
/// * `need` taken over sources as well as destinations: the second half
///   returns `Ok` and the refusal is gone.
/// * The growth done with `resize` to `need` but the pool rebuilt empty: the
///   bytes read back are zeros rather than the source page's.
#[test]
fn a_copy_plan_that_names_a_page_past_the_pool_grows_it_instead_of_refusing() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the copy growth is unmeasured");
        return;
    };
    // Small on purpose: the sweep's pool was three pages because three was
    // all its prefills had asked for.
    let mut shell = shelled(dir, &REALS[0], real, 3);

    // Real history in page 0, so "the bytes land" is about a cache and not
    // about a buffer someone wrote a pattern into.
    shell
        .step(&[driver_vulkan::turns::Turn {
            who: 1,
            tokens: PERIOD[..4].to_vec(),
        }])
        .expect("a first turn");
    let was = shell.shape().pages;
    assert_eq!(was, 3, "this test wants a pool it can name past the end of");

    let layer_bytes = |shell: &driver_vulkan::shell::Shell, page: u32| -> Vec<u8> {
        let shape = shell.shape();
        let buffer = shell.pool().cache(0, false).expect("layer 0 keys");
        let all = shell.device().read(buffer).expect("read the keys");
        let at = shape.slot(page, 0, 0, 0) as usize * shape.bytes as usize;
        let n = shape.page_size as usize * shape.row() as usize * shape.bytes as usize;
        all[at..at + n].to_vec()
    };
    let source = layer_bytes(&shell, 0);
    assert!(
        source.iter().any(|b| *b != 0),
        "page 0 holds no history, so a copy of it proves nothing"
    );

    // The sweep's plan, in miniature: page 0 to a page the pool does not have.
    let moved = shell
        .copy_kv(&driver_api::KvCopyPlan {
            src_domain: driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
            dst_domain: driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
            src_page_ids: vec![0],
            dst_page_ids: vec![3],
            ..driver_api::KvCopyPlan::default()
        })
        .expect("a copy to page 3 of a 3-page pool grows the pool");
    assert_eq!(moved, 1, "one page move");
    assert_eq!(
        shell.shape().pages,
        4,
        "the pool grew to something other than the page the plan named"
    );
    assert_eq!(
        layer_bytes(&shell, 3),
        source,
        "the destination does not hold the source's bytes, so either the copy \
         did not happen or the growth dropped what the pool was holding"
    );
    assert_eq!(
        layer_bytes(&shell, 0),
        source,
        "the growth lost the page the copy read from"
    );

    // The other direction stays a refusal.
    let refused = shell
        .copy_kv(&driver_api::KvCopyPlan {
            src_domain: driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
            dst_domain: driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
            src_page_ids: vec![9],
            dst_page_ids: vec![1],
            ..driver_api::KvCopyPlan::default()
        })
        .expect_err("a source the pool has never held holds no history");
    assert!(
        format!("{refused}").contains("page 9"),
        "the refusal does not name the page that caused it: {refused}"
    );
    assert_eq!(
        shell.shape().pages,
        4,
        "a refused copy grew the pool anyway"
    );
}

/// A frame the ENGINE built answers the same as the driver's own turns.
///
/// # The claim
///
/// There are two page allocators in this system. `Shell::step` uses the
/// driver's own `Book`; `Shell::launch` uses the pages the engine's scheduler
/// chose and does not touch the book at all. This runs one conversation both
/// ways and holds the two distributions against each other BIT FOR BIT.
///
/// # Why bit-for-bit and not a tolerance
///
/// The two paths differ in exactly one thing -- who picked the page numbers --
/// and then converge on the same lowering, the same weights and the same
/// kernels. Any difference at all is a difference in what was read, and the
/// interesting differences are small: a page off by one holds another
/// conversation's keys and the model stays fluent.
///
/// # The controls
///
/// 1. A frame naming a DIFFERENT page for the same tokens must answer
///    differently, or the pages the frame states are not the pages the fire
///    read and this test proves nothing.
/// 2. A frame demanding more pages than the device could ever hold is
///    `Impossible` rather than an error or a wait.
/// 3. A frame whose CSR does not close is refused BEFORE it appends anything,
///    which is checked by firing the conversation again afterwards and
///    getting the same answer as before.
#[test]
fn a_frame_the_engine_built_answers_what_the_driver_s_own_turns_do() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the frame seam is unmeasured");
        return;
    };
    let mut shell = shelled(dir, &REALS[0], real, 24);

    // ── The driver's own path, so the number compared against was not made
    //    by the machinery under test. ──
    // Thirty-two, so the conversation spans two sixteen-row pages: a
    // single-page conversation would let a frame that dropped everything but
    // its first page still answer correctly.
    let prompt: Vec<u32> = PERIOD.iter().copied().cycle().take(32).collect();
    let step = shell
        .step(&[driver_vulkan::turns::Turn {
            who: 1,
            tokens: prompt.clone(),
        }])
        .expect("the prefill");
    let want: Vec<f32> = step
        .logits
        .row(step.readout_of[0])
        .expect("the readout row")
        .to_vec();
    // The pages the book gave conversation 1, which is what the frame below
    // must name for the two paths to be comparable.
    let pages: Vec<u32> = shell.book().pages(1).expect("its pages").to_vec();
    assert!(
        pages.len() >= 2,
        "the premise: the prompt fills more than one page"
    );

    // ── The engine's path, over a conversation the book knows nothing about,
    //    on pages the frame itself names. ──
    let frame = |pages: &[u32]| driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: pages.to_vec(),
        kv_translation_indptr: vec![0, pages.len() as u32],
        required_kv_pages: pages.len() as u32,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: prompt.clone(),
                position_ids: (0..prompt.len() as u32).collect(),
                kv_page_indices: pages.to_vec(),
                kv_page_indptr: vec![0, pages.len() as u32],
                kv_last_page_lens: vec![prompt.len() as u32 % 16],
                qo_indptr: vec![0, prompt.len() as u32],
                sampling_indices: vec![prompt.len() as u32 - 1],
                sampling_indptr: vec![0, 1],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };

    // Pages nobody else holds, so the two runs cannot be the same memory read
    // twice. High ones, since the book hands out low ones first.
    let fresh: Vec<u32> = (20..20 + pages.len() as u32).collect();
    let ran = shell.launch(&frame(&fresh)).expect("the frame");
    let driver_vulkan::frames::Launched::Ran(steps) = ran else {
        panic!("the frame did not run");
    };
    assert_eq!(steps.len(), 1, "one step in, one step out");
    let got = steps[0]
        .logits
        .row(steps[0].readout_of[0])
        .expect("the readout row");
    assert_eq!(
        got, want,
        "the same conversation on scheduler-chosen pages answered differently \
         from the same conversation on book-chosen pages"
    );

    // ── Control 1: the pages a frame names are the pages it reads. ──
    //
    // Same tokens, same everything, different physical pages -- and page 20's
    // keys are still there from the run above, so a fire that ignored the
    // frame's page list would answer identically and this whole test would be
    // measuring nothing.
    let elsewhere: Vec<u32> = (0..pages.len() as u32)
        .map(|i| 20 + pages.len() as u32 + i)
        .collect();
    let ran = shell.launch(&frame(&elsewhere)).expect("the frame");
    let driver_vulkan::frames::Launched::Ran(other) = ran else {
        panic!("the frame did not run");
    };
    let there = other[0]
        .logits
        .row(other[0].readout_of[0])
        .expect("the readout row");
    // Still the SAME answer, because these pages hold this conversation's own
    // freshly-written keys -- which is the point. What must differ is a frame
    // that names pages holding somebody ELSE's keys, below.
    assert_eq!(
        there, want,
        "the same tokens written to different empty pages answered differently"
    );

    // Now the real control: a decode that reads pages it never wrote.
    // A THIRD page: thirty-two tokens fill two sixteen-row pages exactly, so
    // the next token has nowhere to go without one. A decode that omits it is
    // refused before it fires, which is the arithmetic this crate has paid for
    // more than once.
    let held: Vec<u32> = fresh.iter().copied().chain([40]).collect();
    let stale = driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: held.clone(),
        kv_translation_indptr: vec![0, held.len() as u32],
        required_kv_pages: held.len() as u32,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: vec![PERIOD[0]],
                position_ids: vec![prompt.len() as u32],
                kv_page_indices: held.clone(),
                kv_page_indptr: vec![0, held.len() as u32],
                kv_last_page_lens: vec![1],
                qo_indptr: vec![0, 1],
                sampling_indices: vec![0],
                sampling_indptr: vec![0, 1],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };
    let mut wrong = stale.clone();
    // The same decode against pages that were never written at all.
    let empty: Vec<u32> = vec![15, 16, 17];
    wrong.kv_translation.clone_from(&empty);
    wrong.kv_translation_indptr = vec![0, empty.len() as u32];
    wrong.steps[0].plan.kv_page_indices.clone_from(&empty);
    wrong.steps[0].plan.kv_page_indptr = vec![0, empty.len() as u32];

    let history = shell.launch(&stale).expect("the decode with a history");
    let blank = shell.launch(&wrong).expect("the decode without one");
    let (
        driver_vulkan::frames::Launched::Ran(history),
        driver_vulkan::frames::Launched::Ran(blank),
    ) = (history, blank)
    else {
        panic!("a decode did not run");
    };
    assert_ne!(
        history[0].logits.row(history[0].readout_of[0]),
        blank[0].logits.row(blank[0].readout_of[0]),
        "a decode over sixteen tokens of history answered the same as one over \
         pages nothing ever wrote, so the frame's pages are not what attention \
         reads"
    );

    // ── The batched case: two conversations in one frame. ──
    //
    // The single-request runs above cannot see a per-request page CSR at all:
    // with one request, its span IS the whole page list, and a conversion that
    // ignored `kv_page_indptr` entirely answers correctly. Measured -- that
    // mutation survives every assertion above. So the same prompt is fired
    // again beside a SECOND conversation on different pages, and it must
    // still answer `want`.
    let mine: Vec<u32> = vec![50, 51];
    let theirs: Vec<u32> = vec![52, 53];
    let batched = driver_api::FrameSubmission {
        instance_ids: vec![1, 2],
        kv_translation: mine.iter().chain(&theirs).copied().collect(),
        kv_translation_indptr: vec![0, 2, 4],
        required_kv_pages: 4,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: prompt.iter().chain(&prompt).copied().collect(),
                position_ids: (0..prompt.len() as u32)
                    .chain(0..prompt.len() as u32)
                    .collect(),
                kv_page_indices: mine.iter().chain(&theirs).copied().collect(),
                kv_page_indptr: vec![0, 2, 4],
                kv_last_page_lens: vec![0, 0],
                qo_indptr: vec![0, prompt.len() as u32, prompt.len() as u32 * 2],
                // Each request's OWN last row. This said `2 * len - 1` for
                // the second, which is that row counted across the fire --
                // the numbering the driver used to read and nothing ever
                // wrote. A decode envelope resolves its read-out from the
                // instance and never looks at this table, so the wrong value
                // sat here unexamined until the host-wire case below made
                // the same table load-bearing.
                sampling_indices: vec![prompt.len() as u32 - 1, prompt.len() as u32 - 1],
                sampling_indptr: vec![0, 1, 2],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0, 1],
            sub_batch_indptr: vec![0, 2],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1, 2],
            logical_fire_ids: vec![0, 1],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };
    let ran = shell.launch(&batched).expect("the batched frame");
    let driver_vulkan::frames::Launched::Ran(both) = ran else {
        panic!("the batched frame did not run");
    };
    assert_eq!(
        both[0].readout_of.len(),
        2,
        "two requests in, two readouts out"
    );
    for (which, row) in both[0].readout_of.iter().enumerate() {
        assert_eq!(
            both[0].logits.row(*row).expect("a readout row"),
            want,
            "request {which} of a two-request frame answered differently from \
             the same prompt fired alone, so the page CSR was not split at the \
             boundary it names"
        );
    }

    // ── The same two conversations, through the HOST-WIRE class. ──
    //
    // The batch above is a DECODE_ENVELOPE, which resolves its geometry from
    // the instance's channels and never reads the plan's read-out table at
    // all. The engine's other class states everything on the wire, and that
    // is the path `envelope::fill` splits per request -- the path where a
    // read-out row is numbered WITHIN its request rather than across the
    // fire.
    //
    // Nothing here covered it. `vulkan_many_conversations` did, three layers
    // up and a whole engine away, and what it reported was this driver
    // refusing a correct plan: "request 1 reads out row 91, which is not in
    // its own rows 92..184", 91 being request 1's own last row and also
    // request 0's. With the refusal lifted the wrong way, the same numbering
    // hands request 1 request 0's distribution and both conversations answer
    // fluently.
    //
    // So: the same prompt twice, on pages nobody shares, with each request
    // naming its own last row the way `driver::resolve` writes it -- and both
    // must answer what one alone answered.
    let wired = driver_api::FrameSubmission {
        instance_ids: vec![1, 2],
        kv_translation: mine.iter().chain(&theirs).copied().collect(),
        kv_translation_indptr: vec![0, 2, 4],
        required_kv_pages: 4,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: prompt.iter().chain(&prompt).copied().collect(),
                position_ids: (0..prompt.len() as u32)
                    .chain(0..prompt.len() as u32)
                    .collect(),
                // Physical pages: `Shell::launch` takes a plan as stated,
                // and it is `envelope::fill` -- the engine's entry, not this
                // one -- that places working-set pages through the frame.
                kv_page_indices: mine.iter().chain(&theirs).copied().collect(),
                kv_page_indptr: vec![0, 2, 4],
                kv_last_page_lens: vec![0, 0],
                qo_indptr: vec![0, prompt.len() as u32, prompt.len() as u32 * 2],
                // ITS OWN last row, twice -- not `2 * len - 1` for the second.
                sampling_indices: vec![prompt.len() as u32 - 1, prompt.len() as u32 - 1],
                sampling_indptr: vec![0, 1, 2],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0, 1],
            sub_batch_indptr: vec![0, 2],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_HOST],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1, 2],
            logical_fire_ids: vec![0, 1],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };
    let ran = shell
        .launch(&wired)
        .expect("a host-wire frame of two requests");
    let driver_vulkan::frames::Launched::Ran(wire_steps) = ran else {
        panic!("the host-wire frame did not run");
    };
    assert_eq!(
        wire_steps[0].readout_of.len(),
        2,
        "two requests in, two readouts out"
    );
    for (which, row) in wire_steps[0].readout_of.iter().enumerate() {
        assert_eq!(
            wire_steps[0].logits.row(*row).expect("a readout row"),
            want,
            "request {which} of a host-wire batch answered differently from \
             the same prompt fired alone: either its read-out row was taken \
             across the fire instead of within itself, or its pages were"
        );
    }

    // ── Control 2: a demand no device could meet is Impossible, not a wait. ──
    let vast = driver_api::FrameSubmission {
        required_kv_pages: u32::MAX / 2,
        ..frame(&fresh)
    };
    assert!(
        matches!(
            shell.launch(&vast).expect("an answer, not an error"),
            driver_vulkan::frames::Launched::Impossible
        ),
        "a demand of two billion pages was not refused as impossible, so a \
         scheduler would wait for room that cannot exist"
    );

    // ── Control 3: a malformed frame appends nothing. ──
    let mut broken = frame(&fresh);
    // A CSR claiming more rows than there are positions.
    broken.steps[0].plan.qo_indptr = vec![0, prompt.len() as u32 + 4];
    shell
        .launch(&broken)
        .expect_err("a CSR that does not close");
    // ...and the conversation still answers what it did, so nothing was
    // half-written on the way to the refusal.
    let ran = shell.launch(&stale).expect("the decode again");
    let driver_vulkan::frames::Launched::Ran(after) = ran else {
        panic!("the frame did not run");
    };
    assert_eq!(
        after[0].logits.row(after[0].readout_of[0]),
        history[0].logits.row(history[0].readout_of[0]),
        "a refused frame changed the cache on its way out"
    );
}

/// A shell with a device open, a model staged and a cache allocated can be
/// moved to another thread and used there.
///
/// # Why this is worth a device test
///
/// Because `DriverBackend` is one value in the engine's `'static
/// RwLock<Vec<Option<DriverRegistration>>>`, and everything in it must be
/// `Send + Sync`. A `const fn require::<T: Send>()` would say the type
/// implements the trait; this says a shell holding real Vulkan handles, real
/// pages and real weights actually survives the move and still fires after
/// it, which is the thing the engine depends on.
///
/// The channel plane used to live on the shell and made this impossible:
/// `driver::ChannelState` held its cells in a `RefCell` behind an `Rc`. The
/// compiler reported that a crate away, on a static, in a message naming
/// neither this crate nor that field. The plane has since moved beside the
/// shell -- see `programs.rs` -- and its cells moved to `Mutex` behind
/// `Arc`, and this is the end of that chain.
#[test]
fn a_serving_shell_can_be_moved_to_another_thread_and_still_fires() {
    let (device, dir) = gpu!();
    drop(device);
    let model = &REALS[0];
    let Some(real) = weights_of(model) else {
        return;
    };
    let shell = shelled(dir, model, real, 8);

    // The move. If any field of the shell were `!Send` this would not
    // compile, and that is half the assertion; the other half is that it
    // still serves on the far side, since a type can be `Send` and still
    // hold a handle the driver refuses off its creating thread.
    // `who` is a parameter because the two fires must be different
    // conversations: a second fire under the same id continues the first,
    // which is a twelve-token prompt and a different answer.
    fn top_of(shell: &mut driver_vulkan::shell::Shell, who: u64, tokens: &[u32]) -> usize {
        let step = shell
            .step(&[driver_vulkan::turns::Turn {
                who,
                tokens: tokens.to_vec(),
            }])
            .unwrap_or_else(|e| panic!("{e}"));
        step.logits
            .row(step.rows - 1)
            .expect("the turn's last row")
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a non-empty distribution")
            .0
    }

    let (mut shell, top) = std::thread::spawn(move || {
        let mut shell = shell;
        let top = top_of(&mut shell, 0, &PERIOD);
        (shell, top)
    })
    .join()
    .expect("the shell crossed a thread boundary");

    // ...and it still serves on the original thread, after the round trip,
    // and says the same thing: a shell that had lost its cache or its
    // pipelines to the move would answer a different token here.
    let again = top_of(&mut shell, 1, &PERIOD);
    assert_eq!(
        top, again,
        "the shell answered {top} on the far thread and {again} on this one, \
         so something in it did not survive the move"
    );
}

/// The channel plane is `Send`, and a registration outlives a thread move.
///
/// # What this is guarding
///
/// The plane is the portable half -- host memory, no device -- so its
/// acceptance rules are `driver`'s own unit tests' business. What is this
/// crate's business is that `Programs` can be held by something that lives
/// in the engine's `'static` registry: the seam owns one from `create`, and
/// a `!Send` one would not compile there.
///
/// The close is run on the far side rather than merely carrying the value,
/// because the interior mutability is the part that changed: a `RefCell`
/// would have compiled if only the *outer* pointer had become an `Arc`.
#[test]
fn the_channel_plane_can_be_moved_to_another_thread_and_used_there() {
    let mut programs = driver_vulkan::programs::Programs::new();
    let binding = programs
        .register_channel(&driver_api::ChannelRegistrationPlan {
            channel_id: 3,
            dtype: driver_api::PIE_CHANNEL_DTYPE_F32,
            shape: vec![4],
            capacity: 2,
            host_role: driver_api::PIE_CHANNEL_HOST_ROLE_WRITER,
            seeded: false,
            extern_dir: driver_api::PIE_CHANNEL_EXTERN_NONE,
            extern_name: Vec::new(),
            driver_id: 0,
            reader_wait_id: 0,
            writer_wait_id: 0,
        })
        .expect("a well-formed channel");
    assert!(
        binding.mirror_bytes >= u64::from(binding.cell_bytes) * u64::from(binding.capacity),
        "the ring is smaller than the cells it claims: {binding:?}"
    );

    let mut programs = std::thread::spawn(move || {
        let mut programs = programs;
        programs.close_channel(3);
        // Twice is not an error -- teardown races both ways.
        programs.close_channel(3);
        programs
    })
    .join()
    .expect("the plane crossed a thread boundary");

    programs
        .register_channel(&driver_api::ChannelRegistrationPlan {
            channel_id: 3,
            dtype: driver_api::PIE_CHANNEL_DTYPE_F32,
            shape: vec![4],
            capacity: 2,
            host_role: driver_api::PIE_CHANNEL_HOST_ROLE_WRITER,
            seeded: false,
            extern_dir: driver_api::PIE_CHANNEL_EXTERN_NONE,
            extern_name: Vec::new(),
            driver_id: 0,
            reader_wait_id: 0,
            writer_wait_id: 0,
        })
        .expect("the close released the id, so it is free again");
}

/// A frame naming a plan feature this driver does not implement is refused
/// through the real `launch`, and refused BEFORE the cache is touched.
///
/// # Why this is worth a device test
///
/// `frames::unserved_in` is unit tested and that covers which fields it
/// names. What only a device can show is the second half: that the refusal
/// happens at ADMISSION, before any key is appended. A driver that refused
/// on the third step of a three-step frame would have written the first two
/// steps' keys, and the scheduler's retry of the same frame -- which is what
/// a scheduler does with a refusal it can fix -- would append them twice, so
/// the same conversation would carry every prefix token in duplicate and
/// answer fluently.
///
/// So this fires a good frame, refuses a bad one, and fires the good one
/// again in a fresh conversation on the SAME pages, asserting the two good
/// answers agree. If the refused frame had written anything, the second read
/// of those pages would differ.
#[test]
fn a_frame_naming_an_unserved_feature_is_refused_before_the_cache_moves() {
    let (device, dir) = gpu!();
    drop(device);
    let Some(real) = weights_of(&REALS[0]) else {
        return;
    };
    let mut shell = shelled(dir, &REALS[0], real, 24);

    let prompt: Vec<u32> = PERIOD.to_vec();
    let pages: Vec<u32> = vec![17, 18];
    let frame = |max_layers: Option<u32>| driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: pages.clone(),
        kv_translation_indptr: vec![0, pages.len() as u32],
        required_kv_pages: pages.len() as u32,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: prompt.clone(),
                position_ids: (0..prompt.len() as u32).collect(),
                kv_page_indices: pages.clone(),
                kv_page_indptr: vec![0, pages.len() as u32],
                kv_last_page_lens: vec![prompt.len() as u32 % 16],
                qo_indptr: vec![0, prompt.len() as u32],
                sampling_indices: vec![prompt.len() as u32 - 1],
                sampling_indptr: vec![0, 1],
                // The one field under test. Everything else is a frame the
                // suite already fires, so a failure here is about this.
                max_layers,
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };

    let top = |shell: &mut driver_vulkan::shell::Shell| {
        let driver_vulkan::frames::Launched::Ran(steps) =
            shell.launch(&frame(None)).expect("a plain frame")
        else {
            panic!("the plain frame did not run");
        };
        let step = &steps[0];
        step.logits
            .row(step.rows - 1)
            .expect("the frame's last row")
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a non-empty distribution")
            .0
    };

    let before = top(&mut shell);

    let refused = shell
        .launch(&frame(Some(4)))
        .expect_err("a truncated depth this driver would run past");
    let text = format!("{refused}");
    assert!(
        text.contains("max_layers"),
        "the refusal does not name the field: {text}"
    );

    let after = top(&mut shell);
    assert_eq!(
        before, after,
        "the refused frame wrote to the cache on its way out, so a retry of \
         it would read pages that already hold its keys"
    );
}

/// The tile PRODUCTION states reaches the cooperative-matrix build.
///
/// `the_cooperative_matrix_gemm_answers_what_the_baseline_one_does` overrides
/// both the tile and the tier, which is right for what it asks -- it needs
/// two runs differing in one thing -- and is exactly why it cannot answer
/// this. It would pass unchanged on a build where `QMM_TILE` named a point
/// with no cooperative-matrix module, because it never asks the default what
/// it resolves.
///
/// Measured, at the tile shipped today: a 64-row prefill resolves 2 of its 10
/// symbols above baseline, and they are the two that should be --
/// `affine_qmm_t` and `affine_qmm_t_residual`, both at `bm_32_bn_32`. Every
/// other symbol in the fire is a norm, a rope, a gather, the paged attention
/// or the GEMV, and none of those has a cooperative-matrix build.
///
/// That gap is not hypothetical. All 146 coopmat modules were unreachable on
/// every device for the whole life of this crate, and nothing failed: the
/// device reported the tier, the shell set it, the pipeline cache keyed on
/// it. A forced-tile test would not have noticed, and the constant has moved
/// once already -- `(16, 32)` to `(32, 32)` -- with a comment about its
/// coopmat modules left behind that had become false in both halves.
///
/// So this asks the shipped configuration, and states the numbers rather
/// than just "more than zero", because the interesting failure is a DROP.
#[test]
fn the_default_tile_reaches_the_tier_in_production() {
    let (device, dir) = gpu!();
    if !device
        .tiers()
        .contains(&kernels_vulkan::Capability::Coopmat)
    {
        eprintln!("SKIP: this device does not offer cooperativeMatrix");
        return;
    }
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the tier is unmeasured");
        return;
    };

    // A whole number of tiles, or the GEMM's `TokensMultipleOf` guard sends
    // every projection down the GEMV arm and the answer is 0 for a reason
    // that has nothing to do with the tier.
    //
    // READ FROM THE SAME PLACE `shelled` READS IT. There are deliberately two
    // statements of this tile -- `project::QMM_TILE` and the fixture's
    // `qmm_tile`, written out rather than read so that a fixture and a
    // projection can be COMPARED, which is how the last widening was caught.
    // The first version of this test sized its prompt from the constant while
    // the shell it built took the fixture's, so a mutation of the constant
    // changed nothing and the failure message named a tile that was not the
    // one that ran.
    let tile = model::shared::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic()
        .qmm_tile
        .0 as usize;
    let mut prompt: Vec<u32> = Vec::new();
    while prompt.len() < 2 * tile {
        prompt.push(PERIOD[prompt.len() % PERIOD.len()]);
    }
    assert_eq!(prompt.len() % tile, 0);

    let mut shell = shelled(dir, &REALS[0], real, 256);
    let prefill = shell
        .step(&[driver_vulkan::turns::Turn {
            who: 1,
            tokens: prompt,
        }])
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        prefill.fired.tiered > 0,
        "a {}-row prefill at the shipped tile of {tile} resolved NO module \
         above baseline: the tile this driver actually serves does not reach \
         the cooperative-matrix build, however green the forced-tile \
         comparison is. Parsed {} symbols.",
        2 * tile,
        prefill.fired.parsed
    );

    // A decode is GEMV, and `affine_qmv_fast` has no cooperative-matrix build
    // and never will -- the tier is a matrix unit and a matvec has no matrix.
    // Stated so that a day when it DOES resolve something is a day someone
    // looks, rather than a silently better number.
    let decode = shell
        .step(&[driver_vulkan::turns::Turn {
            who: 1,
            tokens: vec![PERIOD[0]],
        }])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        decode.fired.tiered, 0,
        "a decode resolved {} modules above baseline out of {} symbols; the \
         GEMV arm has no cooperative-matrix build, so either one was added or \
         a decode is now taking the GEMM arm",
        decode.fired.tiered, decode.fired.parsed
    );
}

/// The cooperative-matrix GEMM answers what the scalar one does, on a real
/// checkpoint, through the whole driver.
///
/// # Why this test had to exist the moment the tier started working
///
/// `Modules::code` used to take only a symbol, and every module store here is
/// keyed by file stem, so the cooperative-matrix build of an entrypoint --
/// stored under `<symbol>.coopmat` -- was named by nothing and loaded by
/// nobody. All 146 of them were dead. `serve.rs` carries that story; this is
/// the consequence for testing.
///
/// Turning them on is not a small change. It swaps the arithmetic of the
/// largest kernel family in the tree for a different one: a subgroup matrix
/// unit with `float16_t` A and B operands, accumulating through shared
/// memory. `kernels-vulkan` checks that kernel against its own baseline in
/// isolation, which is worth having and is not the same claim as this one --
/// a kernel can be right about a matrix and wrong about which bytes of a real
/// quantised weight it was handed.
///
/// And the existing suite could not have caught it. Every other test here
/// runs at the tile the shared model code states, `(16, 32)`, and 16 is the
/// one row tile `quant/qmm_t.slang` deliberately does NOT compile a
/// cooperative-matrix module for. **The whole device suite passed with the
/// tier switched on and exercised precisely zero of it.** This is the only
/// place in this crate where a cooperative-matrix module runs against a real
/// model at all.
///
/// # What is held fixed
///
/// The tile and the tier are two knobs and this moves ONE. Both runs of a
/// pair state the same tile, and the control gets a module store with every
/// `<symbol>.<tag>` key removed, so it falls back to the scalar module on
/// the same device, over the same weights, through the same launches.
/// Comparing `(16, 32)` against `(32, 32)` instead would have changed the
/// batching and the grid as well, and any difference would have been
/// unattributable.
///
/// Both tiles that HAVE a cooperative-matrix build are covered, 32 and 64,
/// because the reason to care about this claim is mostly the case for
/// widening `QMM_TILE` -- and 64 is the width the prefill measurement in
/// `crates/model` points at. The two tiles agree on the error to the last
/// digit reported below, which is itself worth something: the discrepancy is
/// the operand format and not the tiling.
///
/// # Why this is a tolerance and not an equality
///
/// It was written as an equality first, on the strength of `qmm_t.slang`'s
/// header saying the dequantisation is kept byte-for-byte identical. It is --
/// and the MULTIPLY is not. The scalar path multiplies in fp32; the matrix
/// unit takes `float16_t` operands. So the two disagree by a little, and the
/// question is only how much.
///
/// Measured over qwen3-0.6b's 151,936 logits, at both tiles: **greatest
/// difference 0.25, mean 0.036, against a largest logit of 25.6** -- and the
/// same argmax. The
/// bound below is 0.5, twice the observed error; the run is deterministic --
/// same weights, same prompt, same launches -- so the margin is there for a
/// different device's matrix unit rounding differently, not for run-to-run
/// noise.
///
/// The argmax assertion is the one that matters. A sampler reads the
/// ranking, and a tier that changed which token won would be a different
/// model however small its residuals looked.
#[test]
fn the_cooperative_matrix_gemm_answers_what_the_baseline_one_does() {
    let (device, dir) = gpu!();
    if !device
        .tiers()
        .contains(&kernels_vulkan::Capability::Coopmat)
    {
        eprintln!("SKIP: this device does not offer cooperativeMatrix");
        return;
    }
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the tier is unmeasured");
        return;
    };

    // EXACTLY TWO WHOLE TILES, and the count is the point.
    //
    // `llama_like/forward/mod.rs` puts the GEMM behind a
    // `TokensMultipleOf(tile)` guard, because `qmm_t` has no row argument and
    // reads its tile from the grid. A prompt of 48 rows at a tile of 32 is
    // not a multiple of it, so every projection takes the GEMV arm instead --
    // and a run that never launches `affine_qmm_t` compares the
    // cooperative-matrix tier against itself. Measured, on the way here: at
    // 48 rows the two runs below agreed to the last bit and the only symbols
    // resolved were `affine_qmv_fast*`.
    let mut prompt: Vec<u32> = Vec::new();
    while prompt.len() < 64 {
        prompt.push(PERIOD[prompt.len() % PERIOD.len()]);
    }
    assert_eq!(
        prompt.len() % 32,
        0,
        "the GEMM arm needs a whole number of tiles"
    );

    let answer = |tile: u32, tiered: bool| {
        let mut shell = shelled_at_tile(dir, &REALS[0], real, 256, Some((tile, 32)), tiered);
        let step = shell
            .step(&[driver_vulkan::turns::Turn {
                who: 1,
                tokens: prompt.clone(),
            }])
            .unwrap_or_else(|e| panic!("{e}"));
        (
            step.logits
                .row(step.readout_of[0])
                .expect("a readout row")
                .to_vec(),
            step.fired.tiered,
        )
    };

    // BOTH tiles that have a cooperative-matrix build, because the value of
    // this claim is mostly to whoever widens `QMM_TILE`, and 64 is where the
    // measurement says they should widen it to. 64 rows is a whole number of
    // either.
    for tile in [32u32, 64] {
        let (scalar, scalar_tiered) = answer(tile, false);
        let (matrix, matrix_tiered) = answer(tile, true);

        // DIRECTLY, before comparing any numbers.
        //
        // The comparison below infers that the cooperative-matrix module
        // loaded, from the two runs disagreeing. That is a proxy, and when it
        // failed -- three times, intermittently, on a shared box -- it could
        // only report that the two agreed, which says nothing about why.
        // `Fired::tiered` counts the symbols that resolved above Baseline, so
        // this says whether the tier ran instead of guessing from its
        // effects, and a future failure names which half broke.
        assert_eq!(
            scalar_tiered, 0,
            "the untiered run resolved {scalar_tiered} module(s) above baseline \
             at a tile of {tile}, so it is not the control it claims to be"
        );
        assert!(
            matrix_tiered > 0,
            "the tiered run resolved NO module above baseline at a tile of \
             {tile}: the cooperative-matrix build was not reached, which is \
             the defect this test exists for. Check `Modules::code` still \
             takes a tier and that the store holds `*.coopmat` entries"
        );

        assert!(
            scalar.iter().all(|v| v.is_finite()) && matrix.iter().all(|v| v.is_finite()),
            "a run produced non-finite logits, so the comparison below would be \
         between two kinds of nothing"
        );
        assert_ne!(
            scalar, matrix,
            "the two runs agree to the last bit at a tile of {tile}, which means \
         the cooperative-matrix module was NOT loaded -- the exact defect this \
         test exists for. Check `Modules::code` still takes a tier."
        );

        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("a logit")
                .0
        };
        assert_eq!(
            top(&scalar),
            top(&matrix),
            "the cooperative-matrix tier chose a different token than the scalar \
         one. A sampler reads the ranking, so this is a different model and \
         not a rounding difference"
        );

        let worst = scalar
            .iter()
            .zip(&matrix)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 0.5,
            "the cooperative-matrix GEMM differs from the scalar one by {worst}, \
         where 0.172 was measured and fp16 operands explain about that much. \
         Something beyond precision changed: the dequantisation point, the \
         tile the grid states, or the operand order"
        );
    }
}

/// At the tile the shared model code actually states, the two module stores
/// answer identically -- because at a row tile of 16 there is no
/// cooperative-matrix module to reach.
///
/// The control for the test above, and it is worth as much as that test is.
/// A difference between two runs proves nothing unless the same comparison
/// can also come out equal: if the store-stripping harness were simply
/// perturbing something -- a different allocation order, a different pipeline
/// cache shape -- it would perturb this run too.
///
/// It also pins the claim `serve.rs` makes about the tier walk from the other
/// side. `Capability::PREFERENCE` is walked from the device's tier DOWN, and
/// on this GPU that walk starts at `Coopmat` for every symbol in the plan.
/// The 146 cooperative-matrix modules in the tree are all `affine_qmm_t*` and
/// `sdpa_paged_mma*`, and `quant/qmm_t.slang` deliberately compiles none at a
/// row tile of 16 -- its header explains that a 16-row tile does not amortise
/// a 16x16x16 matrix operation, which was measured and confirmed. So the walk
/// runs, finds nothing above baseline, and falls through. Equal answers are
/// what that looks like from outside.
///
/// This is why the pre-existing device suite could not have caught the tier
/// being turned on: every other test here runs at this point.
#[test]
fn at_the_default_tile_the_tier_has_nothing_to_reach() {
    let (device, dir) = gpu!();
    if !device
        .tiers()
        .contains(&kernels_vulkan::Capability::Coopmat)
    {
        eprintln!("SKIP: this device does not offer cooperativeMatrix");
        return;
    }
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the tier is unmeasured");
        return;
    };

    // The same 64 tokens the test above uses, so the only difference between
    // the two tests is the tile.
    let mut prompt: Vec<u32> = Vec::new();
    while prompt.len() < 64 {
        prompt.push(PERIOD[prompt.len() % PERIOD.len()]);
    }

    let answer = |tiered: bool| {
        let mut shell = shelled_at_tile(dir, &REALS[0], real, 256, Some((16, 32)), tiered);
        let step = shell
            .step(&[driver_vulkan::turns::Turn {
                who: 1,
                tokens: prompt.clone(),
            }])
            .unwrap_or_else(|e| panic!("{e}"));
        step.logits
            .row(step.readout_of[0])
            .expect("a readout row")
            .to_vec()
    };

    assert_eq!(
        answer(false),
        answer(true),
        "stripping the tiered modules changed the answer at a tile of 16, \
         where no cooperative-matrix module is compiled. Either `qmm_t.slang` \
         gained a bm_16 build -- in which case the test above should move to \
         it -- or the harness perturbs something other than the tier, which \
         would make that test's difference unattributable"
    );
}

/// How much of the tiered build any text in this tree can actually name, and
/// how much of THAT the shipped tile can reach.
///
/// # Why a census and not a behaviour
///
/// The milestone before this one found that `Modules::code` took a symbol and
/// no tier, so every `<symbol>.<tag>` module was named by nothing. That is
/// fixed and two tests above measure the fix on real weights. This one
/// answers the question the fix raised and did not settle: **granted the
/// resolver now walks the tiers, how many modules does the walk have to walk
/// to?**
///
/// Three numbers, and the third is the one that matters.
///
/// * **185 tiered modules are compiled** -- 146 at `coopmat`, 39 at `fp16`.
/// * **52 belong to an entrypoint any text names.** `model_compiler`'s
///   `dsl::metal` emits exactly two quantised-GEMM stems, `affine_qmm_t` and
///   `affine_qmm_t_residual` (`dsl.rs`, the `format!` at each). Every other
///   tiered module -- `_bias`, `_splitk`, `_strided`, all seven
///   `*_fp16_precast*` families, and `sdpa_paged_mma{,_sink}` -- is stamped
///   for a symbol no plan in this repository states. That includes ALL 39
///   fp16 ones, so the fp16 tier is unreachable by NAMING, which no resolver
///   change can lift.
/// * **Zero are reachable at the tile the shipped constant states.**
///   `QMM_TILE` is `(16, 32)` and `quant/qmm_t.slang` compiles no
///   cooperative-matrix build at a row tile of 16, deliberately.
///
/// So the honest account of the resolver fix is: it unblocked an eight-fold
/// prefill win and, on its own, takes none of it. Both halves are needed and
/// the other half lives in `crates/model`. That is recorded at `QMM_TILE`
/// itself, with the measurement and the tradeoff.
///
/// # What this pins
///
/// The day someone widens `QMM_TILE`, or compiles a `bm_16` cooperative build,
/// or has a text name `affine_qmm_t_bias`, this goes red -- and every one of
/// those is a moment to re-read the paragraph above rather than a moment to
/// update a number quietly. It needs no GPU: it is a question about the build
/// tree and the lowering, and both are readable from here.
#[test]
fn the_tiered_builds_this_driver_can_actually_reach() {
    use model_dsl as dsl;

    let Some(dir) = SPV_DIR else {
        eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
        return;
    };
    let mut tiered: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).expect("the spirv dir").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "spv")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            && stem.contains('.')
        {
            tiered.push(stem.to_string());
        }
    }
    let at = |tag: &str| tiered.iter().filter(|s| s.ends_with(tag)).count();
    assert_eq!(at(".coopmat"), 146, "a different number of coopmat builds");
    assert_eq!(at(".fp16"), 39, "a different number of fp16 builds");

    // A module belongs to a text-nameable launch when its stem is exactly
    // `<stem>_bfloat16_gs_<g>_b_<b>_bm_<bm>_bn_<bn>`: `dsl::metal` writes the
    // two stems, `affine_point` writes the middle, `affine_gemm_point` writes
    // the tile. Split on the middle rather than testing a prefix, because
    // `affine_qmm_t` is a prefix of `affine_qmm_t_bias` and of every other
    // family, and a prefix test would count all of them.
    let reachable = |tile: (u32, u32), tag: &str| -> usize {
        let suffix = dsl::metal::affine_gemm_point(dsl::WeightRepr::Bf16, 0, tile)
            .split_once("_bm_")
            .map(|(_, t)| format!("_bm_{t}"))
            .expect("the gemm point states a tile");
        tiered
            .iter()
            .filter(|s| s.ends_with(tag))
            .filter(|s| {
                let base = s.trim_end_matches(tag).trim_end_matches('.');
                let Some((stem, rest)) = base.split_once("_bfloat16_gs_") else {
                    return false;
                };
                ["affine_qmm_t", "affine_qmm_t_residual"].contains(&stem) && rest.ends_with(&suffix)
            })
            .count()
    };

    let (bm, bn) = model::shared::llama_like::project::QMM_TILE;
    assert_eq!(
        (bm, bn),
        (32, 32),
        "the shipped GEMM tile moved again. The count below is a count of \
         what one particular tile reaches, so re-derive it rather than \
         editing it: this test's whole point is that a tile and a build tree \
         can disagree silently"
    );
    // TWELVE, and it was zero. This is the number this test was written to
    // watch: for as long as `QMM_TILE` was `(16, 32)`, none of the 146
    // cooperative-matrix builds in the tree could be named by any text, so
    // the resolver's tier machinery ran nothing whatever the device
    // advertised. Widening the tile is what made it reachable, and
    // `crates/model`'s `QMM_TILE` carries the 4.5x that came with it.
    //
    // Twelve rather than two, because a stem is stamped once per
    // `(group x bits)` point and a text picks the point from the checkpoint
    // it loaded. The directory holds more at this tile carrying `_wm_` and
    // `_wn_` suffixes; those are not counted, and correctly so -- no text can
    // name them, because `affine_gemm_point` writes a tile and stops.
    assert_eq!(
        reachable((bm, bn), ".coopmat"),
        12,
        "the shipped tile no longer reaches a cooperative-matrix build, which \
         is the state this driver spent its whole life in without saying so"
    );
    assert_eq!(
        reachable((bm, bn), ".fp16"),
        0,
        "an fp16 build became reachable at the shipped tile"
    );

    // The tile that was shipped, kept as the control: it is the reason the
    // count above is a fact about the TILE and not about the matcher.
    assert_eq!(
        reachable((16, 32), ".coopmat"),
        0,
        "a cooperative-matrix build now exists at a row tile of 16, which \
         means `qmm_t.slang` grew a bm_16 module. Measure it: the last time \
         that was tried, 91 generated twins were worth nothing and were \
         discarded"
    );
    assert_eq!(
        reachable((64, 32), ".coopmat"),
        12,
        "tile 64 lost its builds"
    );
    assert_eq!(
        reachable((32, 32), ".fp16"),
        0,
        "the fp16 tier is unreachable by naming: all 39 of its builds are \
         `*_fp16_precast*` entrypoints and no text states one"
    );
}

/// A buffer reads back at an offset what was written there, through the DMA.
///
/// # What this is guarding
///
/// `Device::buffer` prefers the memory type that is both `DEVICE_LOCAL` and
/// `HOST_VISIBLE`, which on this card is mappable VRAM behind resizable BAR.
/// That preference is worth five times the decode rate and it is not in
/// question. What was never examined is the OTHER direction: mappable VRAM is
/// write-combined, and reading it back through the mapping is uncached,
/// unprefetched and one PCIe round trip deep.
///
/// Measured, on a 1024-token prefill of qwen3-0.6B before this changed:
///
/// | phase | before | after |
/// |---|---|---|
/// | allocate and zero the 334 MB arena | 82 ms | 82 ms |
/// | every dispatch of every layer | 588 ms | 588 ms |
/// | read the answer back | **32 967 ms** | **220 ms** |
/// | widen the logits to f32 | 278 ms | 278 ms |
/// | the whole step | **33 847 ms** | **1 107 ms** |
///
/// Ninety-eight per cent of a prefill was one `memcpy` from uncached memory,
/// at ten megabytes a second on a bus that does twelve gigabytes. The copy
/// engine reads the same memory at the bus's rate into host-cached system
/// memory, and the host then reads THAT at the cache's.
///
/// So this test asks three things of `read_at`, and the third is the one that
/// would go quiet: the bytes are right, an offset is honoured, and the read
/// actually went through the copy engine rather than through a mapping.
#[test]
fn a_read_of_device_memory_goes_through_the_copy_engine() {
    let (device, _) = gpu!();

    // Big enough that the two paths are not the same number. At the measured
    // ten megabytes a second the mapped path would need six seconds for this
    // and the staged one needs tens of milliseconds.
    let bytes: Vec<u8> = (0..64 << 20u32).map(|i| (i % 251) as u8).collect();
    let before = device.staged();
    let buffer = device.buffer(&bytes).expect("a 64 MiB buffer");

    let whole = device.read(&buffer).expect("read it back");
    assert_eq!(whole.len(), bytes.len(), "a short read");
    assert_eq!(whole, bytes, "the bytes came back changed");

    // An offset, and one that is not a multiple of anything: a staged copy
    // states `src_offset` on the copy region and a mapped one adds it to the
    // pointer, and a path that dropped it would still return the right NUMBER
    // of bytes.
    let at = 1_000_003usize;
    let len = 4096usize;
    let part = device
        .read_at(&buffer, at as u64, len as u64)
        .expect("read a slice back");
    assert_eq!(
        part,
        &bytes[at..at + len],
        "the offset was not where the read started"
    );

    assert!(
        at as u64 + len as u64 <= buffer.size(),
        "this test's own arithmetic left the buffer"
    );
    assert!(
        device.read_at(&buffer, buffer.size(), 1).is_err(),
        "a read that starts at the end of the buffer was allowed"
    );
    assert!(
        device.read_at(&buffer, 0, buffer.size() + 1).is_err(),
        "a read longer than the buffer was allowed"
    );

    // THE ONE THAT WOULD GO QUIET.
    //
    // Both reads above answer correctly whichever path they took, so nothing
    // else in this file would notice the day `read_at` stopped staging --
    // except the wall clock of every prefill, which no test watches.
    //
    // Two, and not "more than none": the two reads that were given a real
    // range. `Buffer::local` is what decides, so a device with no device-local
    // host-visible type would fail here, and that is the correct answer for
    // it -- such a part's buffers are in system memory and the mapped read is
    // the fast one. This card has one.
    assert_eq!(
        device.staged() - before,
        2,
        "a read of device-local memory did not go through the copy engine"
    );

    device.free(buffer);
}

/// Every row a request NAMES comes back in the read-out.
///
/// # Why this claim, on this harness
///
/// `serve::logits_of` reads ONE span covering the rows it was asked for, and
/// `Logits::row` answers `None` for any other. So a request naming rows the
/// readback did not cover is a `None` four layers from where it is felt --
/// which is exactly what a first port of multi-readout support produced:
/// `request 0 reads row 30 of a read-out of 34 rows`.
///
/// That claim needs no real weights. The stand-in checkpoint makes every row
/// the SAME distribution, so nothing here can say the rows are distinct -- and
/// nothing here needs to. Coverage is the half that was wrong.
///
/// The distinctness half belongs with the real-weight tests and cannot run on
/// a machine without a pre-quantised snapshot, which is every machine this has
/// been run on. `driver-wgpu`'s `tests/serving.rs` has it, against a bf16
/// checkpoint it quantises through the load plan -- a difference in what the
/// two crates' harnesses accept, not in what they could prove.
#[test]
fn every_row_a_request_names_is_in_the_read_out() {
    use driver_vulkan::pages::Book;
    use driver_vulkan::resources::{Pool, Request, Shape, Weights};
    use driver_vulkan::turns::{Held, Serving};
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_ir::trace::FireClass;

    let (device, dir) = gpu!();
    let shape = Shape {
        layers: 28,
        kv_heads: 8,
        head_dim: 128,
        page_size: 8,
        pages: 8,
        bytes: 2,
    };
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let prefill_plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Prefill,
    );
    let serving = Serving {
        plan: &plan,
        prefill: &prefill_plan,
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
    {
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (text, rows) in [(&plan, 1usize), (&prefill_plan, 32)] {
            let probe = model_compiler::lower::lower(
                text,
                &vec![model_compiler::lower::Row::default(); rows],
                model_compiler::lower::Fire {
                    captures_across_splits: false,
                },
            )
            .expect("the plan lowers");
            names.extend(probe.args.iter().filter_map(|a| match a {
                model_compiler::lower::Arg::Weight(n) => Some(n.clone()),
                _ => None,
            }));
        }
        let block = vec![0u8; 1 << 22];
        for name in &names {
            weights.hold(&device, name, &block).expect("a weight");
        }
    }
    let mut cache = Pipelines::new();
    let mut modules: std::collections::BTreeMap<String, Vec<u8>> =
        std::collections::BTreeMap::new();
    for name in std::fs::read_dir(dir).expect("the spirv dir").flatten() {
        let path = name.path();
        if path.extension().is_some_and(|e| e == "spv")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            modules.insert(stem.to_string(), std::fs::read(&path).expect("a module"));
        }
    }
    let mut lowerings = driver_vulkan::turns::Lowerings::default();
    let mut held = Held {
        book: &mut book,
        pool: &mut pool,
        weights: &weights,
        lowerings: &mut lowerings,
    };

    // Sixteen rows over two eight-row pages, reading out three of them --
    // which is a speculative verifier's shape, and the shape this driver
    // refused by name until the rows could be found again.
    let rows = 16u32;
    let mut request = Request::of((0..rows).collect(), vec![0, 1]);
    let named: Vec<u32> = vec![rows - 3, rows - 2, rows - 1];
    request.samples = named.clone();
    let tokens: Vec<u32> = vec![0; rows as usize];

    let step = serving
        .over(
            &device,
            &mut cache,
            &modules,
            &mut held,
            std::slice::from_ref(&request),
            &[tokens.as_slice()],
        )
        .unwrap_or_else(|e| panic!("the fire: {e}"));

    for &at in &named {
        assert!(
            step.logits.row(at as usize).is_some(),
            "row {at} was named by the request and is not in the read-out of {} \
             rows -- the readback did not cover it",
            step.logits.rows
        );
    }
    // The control: a row NOBODY named is absent, so the assertion above is
    // about the naming and not about a readback that happens to hold
    // everything.
    assert!(
        step.logits.row(0).is_none(),
        "row 0 was not named and came back anyway, so this test cannot see \
         what it is for"
    );
}

/// `Device::copy_within` moves the same bytes whichever route it takes, and a
/// large move costs the submission rather than the bytes.
///
/// # Why this test exists
///
/// Because `copy_within` had exactly one route -- a `memmove` through the
/// mapping -- and it was chosen for being CORRECT, which it is. The cost was
/// never the argument, and the cost was the whole problem: this card's
/// mappable VRAM is write-combined, so the load side of that `memmove` runs
/// uncached at some thirty megabytes a second. `Pool::copy_plan` calls this
/// once per layer per half for every page the engine moves, so a prefix share
/// and a fork both paid it. It is the same defect `Pool::resize` had, and it
/// outlived that fix by hiding behind a doc comment that explained why the
/// host route was correct without ever asking what it cost.
///
/// # What it measures
///
/// Both routes, against a host-computed expectation, and both edges of the
/// choice between them:
///
/// 1. **Disjoint** ranges, which take the copy engine -- every move a pool
///    actually makes, because a page move names two different pages.
/// 2. **Overlapping** ranges, which must keep the documented `memmove`
///    promise. A `vkCmdCopyBuffer` whose regions overlap within one buffer is
///    undefined, so this is the case the mapping is retained for; if the
///    disjointness test were ever written backwards, this is what says so.
/// 3. **The cost**, as an order-of-magnitude tripwire and not a benchmark. A
///    megabyte took 33 ms through the mapping and 27 us on the copy engine,
///    so a 15 ms ceiling is comfortably clear of the fixed route and nowhere
///    near the variable one. It started at 5 ms and was raised once, after a
///    full-suite run on a contended box tripped it while the same test passed
///    solo twice: the copy engine's cost is a SUBMISSION, so it moves with
///    whatever else is submitting, and 15 ms still leaves the host route --
///    which contention would only slow further -- caught by more than double.
///
/// The overlap is deliberately a FORWARD one -- destination above source,
/// overlapping by half -- because that is the direction a naive byte-at-a-time
/// copy corrupts, and `std::ptr::copy` is the reason it does not.
#[test]
fn a_copy_within_a_buffer_moves_the_same_bytes_by_either_route() {
    let (device, _dir) = gpu!();
    const N: usize = 1 << 20;
    // A pattern where every byte says which position it came from, so a copy
    // that lands at the wrong offset is caught rather than a copy that lands
    // wholly elsewhere.
    let source: Vec<u8> = (0..N * 3).map(|i| (i % 251) as u8).collect();

    // 1. Disjoint: the copy engine's path.
    let disjoint = device.buffer(&source).expect("a buffer");
    let began = std::time::Instant::now();
    device
        .copy_within(&disjoint, 0, (N * 2) as u64, N as u64)
        .expect("a disjoint copy");
    let took = began.elapsed();
    let back = device.read(&disjoint).expect("read it back");
    assert_eq!(
        &back[N * 2..],
        &source[..N],
        "a disjoint copy did not land the source's bytes at the destination"
    );
    assert_eq!(
        &back[..N * 2],
        &source[..N * 2],
        "a disjoint copy wrote outside the range it was given"
    );
    within_budget(
        took,
        std::time::Duration::from_millis(15),
        &format!(
            "a megabyte took {took:?}. Through the mapping it takes about 33 ms and \
             on the copy engine about 27 us, so this is the host route back"
        ),
    );

    // 2. Overlapping, forwards: the mapping's path, and the `memmove` promise.
    let overlapping = device.buffer(&source).expect("a buffer");
    device
        .copy_within(&overlapping, 0, (N / 2) as u64, N as u64)
        .expect("an overlapping copy");
    let back = device.read(&overlapping).expect("read it back");
    let mut expected = source.clone();
    expected.copy_within(0..N, N / 2);
    assert_eq!(
        &back[..N * 2],
        &expected[..N * 2],
        "an overlapping forward copy did not behave as a memmove"
    );
}

/// Moving a page of a real cache costs milliseconds, not tens of them.
///
/// # Why a timing test exists for this
///
/// Because `copy_kv` is not a rare verb. Every prefix share and every fork
/// goes through `Pool::copy_page`, which calls `Device::copy_within` once per
/// layer per half -- fifty-six times for this model, at thirty-two kilobytes
/// each. That was a `memmove` through a write-combined mapping, so each of
/// those fifty-six calls paid an uncached read, and the whole page cost about
/// sixty-three milliseconds.
///
/// Nothing went red. The bytes were always right, which is exactly how the
/// same defect in `Pool::resize` survived: a correct route nobody had timed.
/// Measured here on the 4090, eight pages moved back to back:
///
/// ```text
///   through the mapping   502.6 ms   (62.8 ms a page)
///   on the copy engine     27.3 ms   ( 3.4 ms a page)
/// ```
///
/// # What it measures, and what the ceiling is for
///
/// Eight moves rather than one, because the FIRST move of a run carries the
/// pool's first touch and reads about seventy milliseconds on either route --
/// a one-page test would therefore pass with the host route in place and
/// prove nothing. Eight amortise that away, and the two routes are then
/// eighteen times apart.
///
/// The ceiling is 150 ms for the eight, which is five times the copy engine's
/// measured cost and a third of the mapping's. That gap is deliberately wide:
/// this box is shared, so the number moves, and an order-of-magnitude
/// tripwire that never fires spuriously is worth more than a benchmark that
/// does. Reverting `copy_within` to the mapping fails it by 3.4x.
#[test]
fn moving_a_page_costs_milliseconds_rather_than_tens_of_them() {
    let (_device, dir) = gpu!();
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so a page move is unmeasured");
        return;
    };
    let shell = shelled(dir, &REALS[0], real, 32);
    let pool = shell.pool();

    // The first move of a fresh pool is the pool's first touch, on either
    // route. Spend it before the clock starts.
    // The shell's own device. `gpu!()`'s is this suite's lock and belongs to
    // no buffer here; copying between pool buffers through it names two
    // `VkDevice`s in one `vkCmdCopyBuffer`, which is what the validation
    // layer aborted on and what made this timing meaningless.
    let owner = shell.device();
    pool.copy_page(owner, 0, 1).expect("a warming move");

    let began = std::time::Instant::now();
    for page in 0..8u32 {
        pool.copy_page(owner, page, page + 8).expect("a page move");
    }
    let took = began.elapsed();

    within_budget(
        took,
        std::time::Duration::from_millis(150),
        &format!(
            "eight page moves took {took:?}. On the copy engine they take about \
             27 ms and through the mapping about 503 ms, so this is the host \
             route back in `Device::copy_within`"
        ),
    );
}

/// A prefill's arena is filled by the card, not shipped to it.
///
/// # What this counts and why it is not a timing test
///
/// The arena is the scratch a fire writes its intermediates and its logits
/// into, sized `rows * vocab * 4` for the head. For a decode that is 326 KB.
/// For a 384-row prefill of qwen3-0.6b it is **233 megabytes**, because every
/// row samples (see `turns::once` for why) and the vocabulary is 151,936.
///
/// It used to be made with `Device::buffer(&vec![0u8; n])`: a zero-filled
/// `Vec` in system memory, then uploaded whole. Both halves were paid.
/// Measured on the 384-row prefill, release: the arena phase cost **35.5 ms
/// of a 167 ms step**, and `Device::empty` plus a `vkCmdFillBuffer` does the
/// same thing in **1.6 ms**. The bus was not misbehaving -- `Device::write`
/// runs at the 10 GB/s it documents, and 233 MB at 10 GB/s is 23 ms. Sending
/// zeros over a bus at all was the mistake.
///
/// The obvious tripwire would be a stopwatch, and this crate has one of those
/// for `copy_page` because there the two routes move IDENTICAL bytes and only
/// the clock can tell them apart. Here there is a better witness: whether the
/// bytes crossed the bus at all. `Device::uploaded` counts them, so this
/// asserts a fact about traffic rather than about a shared machine's mood --
/// no ceiling to tune, and no flake.
///
/// # Why the bound is the arena's own size
///
/// A step uploads real things besides the arena: the fire tables, the token
/// ids, the sampling indices. Those are kilobytes. So the assertion is that
/// the step's uploads are a small fraction of the arena rather than zero, and
/// the arena's size is measured here and printed rather than assumed -- if a
/// future lowering makes the arena small, the test says so instead of passing
/// on a bound that has stopped meaning anything.
///
/// Mutation: restoring `device.buffer(&vec![0u8; low.arena_bytes])` in
/// `turns::arena_for` uploads the whole arena and fails here by two orders of
/// magnitude.
#[test]
fn a_prefills_arena_does_not_cross_the_bus() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the arena's route is unmeasured");
        return;
    };
    let mut shell = shelled(dir, &REALS[0], real, 512);

    // Warm first, so that the weights' own upload -- which IS a legitimate
    // host write, and is hundreds of megabytes -- is not counted against the
    // step being measured.
    shell
        .step(&[driver_vulkan::turns::Turn {
            who: 1,
            tokens: PERIOD.to_vec(),
        }])
        .unwrap_or_else(|e| panic!("{e}"));

    let rows = 384;
    let mut prompt: Vec<u32> = Vec::new();
    while prompt.len() < rows {
        prompt.extend_from_slice(&PERIOD);
    }
    prompt.truncate(rows);

    let before = shell.device().uploaded();
    shell
        .step(&[driver_vulkan::turns::Turn {
            who: 2,
            tokens: prompt,
        }])
        .unwrap_or_else(|e| panic!("{e}"));
    let uploaded = shell.device().uploaded() - before;

    // What the arena for that prefill would have been. Not read out of the
    // lowering -- which this test has no handle on -- but computed the way
    // the lowering sizes it, and then sanity-checked against being trivial.
    let arena = rows as u64 * 151_936 * 4;
    eprintln!("a {rows}-row prefill uploaded {uploaded} bytes; its arena is {arena}");
    assert!(
        arena > 200 * 1024 * 1024,
        "the arena is only {arena} bytes, so this test no longer separates the routes"
    );
    assert!(
        uploaded < arena / 100,
        "a {rows}-row prefill uploaded {uploaded} bytes, which is most of its \
         {arena}-byte arena: the arena is being zero-filled on the host and \
         shipped over the bus again"
    );
}

/// Opening a cache does not send the cache to the card.
///
/// # Why this counts bytes
///
/// `Pool::open` zeroes every layer-half, and it must: a cache that came up
/// holding the last model's rows would produce attention over sequences
/// nobody asked about, and the attention would look plausible. The question
/// is only WHERE the zeros are made.
///
/// It used to make them on the host -- one `vec![0u8; layer_bytes]`, uploaded
/// to each of the `2 * layers` buffers. Correct, and it cost the whole cache
/// in bus traffic. Measured on the 28-layer, 512-page pool below: **939 MB
/// uploaded and 162 ms**, against **0 bytes and 36 ms** through
/// `Device::empty` plus a `vkCmdFillBuffer` each.
///
/// This pool is small. A serving pool is sized to fill the card, so on a
/// 24 GB 4090 the old route spent seconds of startup sending zeros to memory
/// that writes them itself at its own bandwidth.
///
/// The striking part is that `Pool::resize` already did it the right way --
/// its grow path zeroes the new tail with `Device::zero` -- three hundred
/// lines from a function doing the same job the other way. Neither could
/// notice, because both produce a cache full of zeros.
///
/// So the assertion is exact rather than timed: opening a cache uploads
/// NOTHING. There is no ceiling to tune and nothing for a shared box to make
/// flaky, and restoring the host route fails it by 939 megabytes.
#[test]
fn opening_a_cache_uploads_nothing() {
    let (device, _dir) = gpu!();
    let shape = driver_vulkan::resources::Shape {
        layers: 28,
        kv_heads: 8,
        head_dim: 128,
        page_size: 16,
        pages: 512,
        bytes: 2,
    };
    let before = device.uploaded();
    let at = std::time::Instant::now();
    let pool = driver_vulkan::resources::Pool::open(&device, shape).expect("open");
    let took = at.elapsed();
    let uploaded = device.uploaded() - before;
    let whole = shape.layer_bytes() * 2 * u64::from(shape.layers);
    eprintln!("opening a {whole}-byte cache took {took:?} and uploaded {uploaded} bytes");
    pool.close(&device);

    // The cache is worth checking too: a shape that had quietly become tiny
    // would make the claim below true and meaningless.
    assert!(
        whole > 512 * 1024 * 1024,
        "the cache is only {whole} bytes, so this no longer separates the routes"
    );
    assert_eq!(
        uploaded, 0,
        "opening a {whole}-byte cache uploaded {uploaded} bytes: it is being \
         zeroed on the host and shipped over the bus"
    );
}

/// A frame that binds a page above the pool grows it, and answers the same as
/// one fired on a pool that was already large enough.
///
/// # The door this closes
///
/// `Shell::copy_kv` had this defect and was fixed one commit earlier; this is
/// the same question asked of the other door. A frame's growth came from
/// `kv_translation` and `required_kv_pages` -- both of which are the engine's
/// STATEMENTS about the pages -- and not from `kv_page_indices`, which is the
/// list the driver actually binds. The two can differ: the translation is
/// empty whenever nothing was moved, and only one of the engine's two
/// batch-assembly paths folds the page list into the declared high-water.
///
/// Measured on a three-page pool, before the fix:
///
/// ```text
///   a step of this frame did not run:
///   Unstageable(NoSuchPage { request: 0, page: 7, pages: 3 })
/// ```
///
/// That is not a silent wrong answer -- the bounds check in `Request::stage`
/// caught it, which is why it stays exactly where it is. It is a REQUEST
/// KILLED for a page the scheduler was entitled to hand out and the pool
/// could have grown to hold, which is the same fault the copy door had and
/// the same one `admit` exists to prevent.
///
/// # What is measured
///
/// The growth, its size, and the ANSWER -- the same decode fired on the same
/// page 7 from a pool that started with room for it. A growth that reallocated
/// without carrying the pool's contents, or that renumbered the pages, gives
/// a different row.
#[test]
fn a_frame_binding_a_page_above_the_pool_grows_it_rather_than_dying() {
    let (device, dir) = gpu!();
    let _ = &device;
    let Some(real) = checkpoint_weights() else {
        eprintln!("no readable 4-bit qwen3-0.6b, so the frame growth is unmeasured");
        return;
    };

    // Page 7, declared as one page, with nothing translated: well-formed, and
    // above a pool that only three prefills' worth of frames have grown.
    let held: Vec<u32> = vec![7];
    let frame = driver_api::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: Vec::new(),
        kv_translation_indptr: vec![0, 0],
        required_kv_pages: 1,
        steps: vec![driver_api::StepSubmission {
            plan: driver_api::LaunchPlan {
                token_ids: PERIOD[..4].to_vec(),
                position_ids: (0..4).collect(),
                kv_page_indices: held.clone(),
                kv_page_indptr: vec![0, held.len() as u32],
                kv_last_page_lens: vec![4],
                qo_indptr: vec![0, 4],
                sampling_indices: vec![3],
                sampling_indptr: vec![0, 1],
                ..driver_api::LaunchPlan::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![0],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
            region_row_indptr: Vec::new(),
            region_sig: Vec::new(),
            region_k: Vec::new(),
        }],
    };

    let fired = |pages: u32| -> (Vec<f32>, u32) {
        let mut shell = shelled(dir, &REALS[0], real, pages);
        let ran = shell
            .launch(&frame)
            .expect("a frame naming page 7 is servable from any pool that can hold it");
        let driver_vulkan::frames::Launched::Ran(steps) = ran else {
            panic!("a frame the pool can grow to hold was refused: {ran:?}");
        };
        let row = steps[0]
            .logits
            .row(steps[0].readout_of[0])
            .expect("a readout row")
            .to_vec();
        (row, shell.shape().pages)
    };

    let (grown, pages) = fired(3);
    assert_eq!(
        pages, 8,
        "the pool did not grow to cover the page the frame binds"
    );
    let (roomy, was) = fired(8);
    assert_eq!(was, 8, "the roomy pool grew when it did not need to");
    assert_eq!(
        grown, roomy,
        "the same tokens on the same page answered differently depending on \
         whether the pool had to grow first, so the growth does not leave the \
         cache where a fire expects it"
    );
}

/// A child process that opens a device and says which one it got.
///
/// Ignored, because it is not a test: it is the other half of
/// `the_device_that_opens_is_the_one_that_was_chosen`, which re-executes this
/// binary to read the answer under a different environment. Doing it in a
/// child rather than with `set_var` is not fastidiousness -- every other test
/// in this file opens a device, and a variable flipped in this process would
/// change which device THEY ran on, at whatever moment the scheduler chose.
#[test]
#[ignore = "the child half of the device-choice proof"]
fn report_which_device_opened() {
    let (device, _) = gpu!();
    eprintln!("OPENED: {}", device.name());
}

/// The device that opens is the one that was chosen, and the choice is not
/// the loader's enumeration order.
///
/// `Device::finish` used to be `devices.first()`. The Vulkan specification
/// places no order on `vkEnumeratePhysicalDevices`, and the machine this crate
/// was written on offers TWO devices -- an RTX 4090 and a `llvmpipe` software
/// rasteriser from Mesa's `lvp_icd.json`. Every number in this crate was
/// measured on the card because the loader happened to list it first. Had that
/// ever changed, the whole suite would have moved onto a CPU implementation,
/// passed, and said nothing.
///
/// So this asks the real loader on the real machine, twice. The default open
/// must land on the ranked best; naming a device must land on THAT one. On a
/// box with one device the two answers are the same and the test still holds,
/// which is why it asserts a relation rather than a name.
#[test]
fn the_device_that_opens_is_the_one_that_was_chosen() {
    fn open_with(pin: Option<&str>) -> Option<String> {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "report_which_device_opened",
            "--ignored",
            "--exact",
            "--nocapture",
        ]);
        match pin {
            Some(v) => cmd.env("PIE_VULKAN_DEVICE", v),
            None => cmd.env_remove("PIE_VULKAN_DEVICE"),
        };
        let out = cmd.output().expect("cannot re-execute this test binary");
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .find_map(|l| l.strip_prefix("OPENED: ").map(str::to_string))
    }

    let Some(default) = open_with(None) else {
        eprintln!("skipped: no device opens here at all");
        return;
    };
    eprintln!("default: {default}");

    // A software adapter must never be the DEFAULT choice while anything else
    // can compute. Asked by pinning it: if it opens when named, it was there
    // to be chosen, and the default declining it is then a decision.
    for pipe in ["llvmpipe", "lavapipe", "swiftshader"] {
        let Some(got) = open_with(Some(pipe)) else {
            continue;
        };
        assert!(
            got.to_ascii_lowercase().contains(pipe),
            "PIE_VULKAN_DEVICE={pipe} opened {got}. The override must take the \
             name it is given or refuse -- silently opening a different device \
             turns a deliberate cross-check into a measurement of the wrong \
             thing"
        );
        assert_ne!(
            got, default,
            "this machine offers a software adapter ({got}) and it is ALSO \
             what opens by default. Something is picking by enumeration order \
             again"
        );
        eprintln!("software adapter present and declined by default: {got}");
    }

    // And the default is reachable by name, which is the other direction: a
    // ranking that returned something the override cannot address would mean
    // the two paths disagree about what the device list even is.
    let pinned = open_with(Some(&default));
    assert_eq!(
        pinned.as_deref(),
        Some(default.as_str()),
        "naming the device that opens by default did not open it"
    );

    assert_eq!(
        open_with(Some("no device is called this")),
        None,
        "an override naming nothing opened SOMETHING. A run that asked for one \
         device and quietly got another looks exactly like a measurement"
    );
}
