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

//! # THIRTY-EIGHT TESTS STOOD HERE, AND THE CUT WAS ONE LINE OF REASONING
//!
//! Every one of them reached the legacy execution path -- `model::shared`'s
//! traced texts, `model_compiler::lower`, `shell::Shell`, `turns::Serving`,
//! `dispatch::Geometry` or `replay` -- and all six of those are deleted. This
//! file is a third of the 12,154 lines it was, and what is left is every test
//! that measures the DEVICE: a buffer, a descriptor, a pipeline, a grid, a
//! pool, a page, a number that comes back -- plus two that did not exist
//! before, because until `serve::run` there was nothing to join the executor to
//! the card with.
//!
//! They are named here in four groups, because "the serving loop is gone" is
//! not a reason to lose the list of what it was checked for. Anything a
//! `Program` walk owes back is on it.
//!
//! **THE FIRE PATH, THROUGH A LOWERED PLAN (13).** `a_rectangle_a_real_plan_
//! states_records_and_submits`, `a_norm_a_real_plan_states_computes_what_a_
//! host_reference_computes`, `a_whole_real_plan_records_into_one_command_
//! buffer_and_submits`, `a_real_plans_kv_append_puts_the_row_where_the_page_
//! table_says`, `the_rows_the_frame_reads_out_are_the_rows_the_gather_moves`,
//! `a_fire_that_cannot_run_says_which_launch`, `the_logits_a_fire_leaves_are_
//! one_row_per_readout_and_are_not_f32`, `every_row_a_request_names_is_in_the_
//! read_out`, `the_tiled_gemm_answers_the_way_the_vector_kernel_does`,
//! `a_routed_prefill_answers_the_same_twice`, `the_default_tile_reaches_the_
//! tier_in_production`, `the_cooperative_matrix_gemm_answers_what_the_baseline_
//! one_does`, `at_the_default_tile_the_tier_has_nothing_to_reach`, plus the
//! `whole_plan`, `plan_by_routine`, `gemm_agrees` and `Binding` scaffolding
//! they shared.
//!
//! Two of those are the crate's sharpest claims and are owed back by name:
//! **the logits of a real fire are one row per READOUT and are NOT f32** (a
//! reader that assumed four bytes got a vocabulary exactly half zeros), and
//! **a fire that cannot run says WHICH LAUNCH** -- a plan states the same
//! symbol hundreds of times and a refusal that named only the symbol is not a
//! refusal anyone can act on. `serve::Unfired` still carries the launch index
//! for exactly that reason, and `serve::run` raises it -- so the second claim
//! is kept and only the first is owed.
//!
//! **A REAL CHECKPOINT, END TO END (6).** `a_whole_plan_fires_with_the_weights_
//! a_real_checkpoint_holds`, `a_real_model_continues_a_pattern_it_was_shown`,
//! `a_conversation_is_answered_the_same_however_it_reaches_the_driver`,
//! `a_prompt_that_is_not_whole_tiles_is_answered_the_way_the_decode_answers_
//! it`, `a_second_real_model_is_served_the_way_the_text_states_it`,
//! `both_real_models_agree_with_an_independent_implementation`, and the `Real`
//! / `REALS` / `Oracle` / `continued` / `shelled` fixtures under them. THIS IS
//! THE MEASUREMENT THE WHOLE CRATE EXISTS TO MAKE and nothing replaces it: two
//! real models, loaded from disk, answering the same tokens an independent
//! implementation answers. It cannot be ported until a lane binds --
//! `baker::mod`'s `every_catalog_row_traces_for_this_plane_and_none_binds_yet`
//! is the test that says when.
//!
//! **THE SERVING LOOP AND THE BOOK (15).** `a_deployment_fires_step_after_step_
//! and_stops_building_pipelines`, `a_forked_conversation_carries_the_history_
//! it_was_forked_from`, `a_cache_resized_under_a_conversation_does_not_change_
//! its_answer`, `a_decode_step_does_not_stall`, `a_long_conversations_decode_
//! step_does_not_stall`, `the_projections_dominate_both_steps_now_that_the_
//! decode_splits_its_keys`, `a_decode_layer_is_eleven_ordered_stages_and_the_
//! ordering_is_the_cost`, `giving_back_one_page_costs_what_giving_back_half_
//! the_pool_costs`, `a_pool_growth_the_host_cannot_stage_is_retryable_rather_
//! than_fatal`, `a_copy_plan_that_names_a_page_past_the_pool_grows_it_instead_
//! of_refusing`, `a_serving_shell_can_be_moved_to_another_thread_and_still_
//! fires`, `the_pages_a_grow_adds_are_zero`, `a_shell_refuses_a_model_
//! assembled_out_of_two`, `moving_a_page_costs_milliseconds_rather_than_tens_
//! of_them` and `a_prefills_arena_does_not_cross_the_bus`. All fifteen went
//! with `shell::Shell`, whose verbs are the engine seam's;
//! `resources::Pool::ceiling` and `Pool::copy_rows` carry the paragraphs that
//! used to point at it.
//!
//! The last two are BANDWIDTH claims and are the ones a rebuilt serving loop
//! should reinstate first: a page move costs milliseconds rather than tens of
//! them because `vkCmdCopyBuffer` runs on the copy engine, and a prefill's
//! arena never crosses the bus at all.
//!
//! **THE ENGINE'S FRAME (3) AND THE TIER PROBE (1).** `a_frame_the_engine_
//! built_answers_what_the_driver_s_own_turns_do`, `a_frame_naming_an_unserved_
//! feature_is_refused_before_the_cache_moves`, `a_frame_binding_a_page_above_
//! the_pool_grows_it_rather_than_dying` -- all three matched on
//! `frames::Launched`, which `shell::launch` was the only constructor of --
//! and `the_tiered_builds_this_driver_can_actually_reach`, which read
//! `model_dsl` through a dependency this manifest never declared and so could
//! not have compiled even with the lowering in place.
//!
//! Two measurements from the timing group are worth carrying out of it,
//! because they are numbers and not code: a decode layer is **eleven ordered
//! stages** and the ordering is the cost, and the projections dominate both
//! steps now that the decode splits its keys.

use driver_vulkan::device::{Bound, Device, Failed, Pipelines, groups_for};
use driver_vulkan::{Dims, Rule};
use kernels_vulkan::Capability;
use std::sync::{Mutex, MutexGuard, OnceLock};

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

/// Why `Device::open` refused, kept for the one test that reports it.
///
/// `gpu()` printed this and dropped it, which is fine for a skip message and
/// no use to a test whose whole subject is whether the suite ran at all.
static NO_DEVICE: OnceLock<String> = OnceLock::new();

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
            let _ = NO_DEVICE.set(e.to_string());
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

/// Borrow the shared device, or skip saying why.
///
/// This used to hand back a module DIRECTORY beside the device, from
/// `option_env!("PIE_KERNELS_VULKAN_SPV_DIR")`. The modules are in the rlib
/// now, so what is left of that check is whether this build compiled any --
/// which is the same question, asked of the thing that has the answer.
macro_rules! gpu {
    () => {{
        if !kernels_vulkan::embedded() {
            eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
            return;
        }
        let Some(device) = gpu() else {
            return;
        };
        device
    }};
}

/// Whether this run measured anything at all, said out loud.
///
/// The 69 `gpu!()` tests in this file print `skipped:` and return when there
/// is no device, which is the right shape for a suite that has to stay green
/// on the machines that build `model-ir`. It has one bad property: a run that
/// exercised the whole driver and a run that touched no GPU whatsoever both
/// print
///
/// ```text
/// test result: ok. 39 passed
/// ```
///
/// and `cargo test` hides a passing test's stdout, so the `skipped:` lines
/// that would have said which are not shown either. The sibling
/// `kernels-vulkan` suite was caught reporting 48 passed in 0.06 seconds on a
/// box whose Vulkan ICD is a stub; nothing but the clock said so.
///
/// This test is not gated, because its whole job is to run everywhere.
/// `PIE_VULKAN_REQUIRE_DEVICE=1` turns the absence into a failure -- the same
/// spelling `kernels-vulkan` uses, so one variable covers both suites, and
/// what any job that installs a driver ON PURPOSE should set.
///
/// Both this suite's ways of measuring nothing would produce the same vacuous
/// green, but only one of them is reachable. A build with no modules is the
/// other, and it cannot happen HERE: this target carries
/// `required-features = ["native"]`, `native` pulls in `kernels-vulkan/native`,
/// and that crate's `build.rs` PANICS when it cannot run `slangc` rather than
/// emitting an empty set. So `kernels_vulkan::embedded()` is always true in
/// this binary -- the check `gpu!()` makes above it is belt and braces -- and
/// a test asserting on it here would be one more guard that cannot fire.
/// `kernels-vulkan`'s own suite has no `required-features`, so it checks both.
#[test]
fn the_runner_states_whether_it_has_a_device() {
    let required = std::env::var_os("PIE_VULKAN_REQUIRE_DEVICE").is_some_and(|v| v != "0");
    // Counted off this file rather than written down, so the number cannot
    // become a lie the next time a test is added. The needle is split because
    // an undivided one MATCHES ITSELF: this literal is in the text
    // `include_str!` reads.
    let needle = concat!("= gpu", "!();");
    let gated = include_str!("device.rs").matches(needle).count();
    // THIRTY, AND IT READ SIXTY. The floor is not a target, it is the line
    // below which this file has stopped being the device suite -- and the cut
    // that deleted the legacy walk took thirty-eight tests with it, thirty of
    // which opened a card. Lowering the number is the honest half of that
    // deletion: leaving it at sixty would have made this test fail for the
    // rest of the suite's life, and raising it to exactly what the file holds
    // would make it fail every time somebody adds one.
    assert!(
        gated >= 30,
        "found {gated} device-gated tests by reading this file, which is not what it contains"
    );

    match gpu() {
        Some(device) => {
            let name = device.name().to_string();
            let kind = if on_software() { " (software)" } else { "" };
            drop(device);
            println!(
                "VULKAN DEVICE: PRESENT ({name}){kind}. The {gated} device-gated tests here ran against it."
            );
        }
        None => {
            let why = NO_DEVICE.get().map_or("no reason recorded", String::as_str);
            println!("VULKAN DEVICE: ABSENT ({why}).");
            println!(
                "All {gated} device-gated tests in this file skipped, so a green `--test device` here measured NOTHING."
            );
            assert!(
                !required,
                "PIE_VULKAN_REQUIRE_DEVICE is set and no device opened: {why}. A suite that silently skips is what this test exists to prevent"
            );
        }
    }
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

/// The baseline module for an entrypoint, out of the rlib.
fn module(entrypoint: &str) -> &'static [u8] {
    kernels_vulkan::code(entrypoint, Capability::Baseline)
        .unwrap_or_else(|| panic!("`{entrypoint}` has no baseline module in this build"))
}

/// A device opens, and says whether anything is checking it.
///
/// Not a formality. Every other test here would pass against a driver that was
/// quietly doing something undefined, so whether a layer is present changes
/// what the rest of this file is evidence FOR, and that belongs in the output
/// rather than in someone's memory of how they ran it.
#[test]
fn a_device_opens_and_says_whether_a_layer_is_watching() {
    let device = gpu!();
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
    let device = gpu!();
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
    let device = gpu!();

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
    // THREE BUFFERS AND A PUSH RANGE. `RmsParams` was a storage struct at
    // binding 3 until the binder rows retired; the same five words are a
    // `[[vk::push_constant]]` block now, so the bytes this test already
    // builds go to `Device::run`'s push argument and the module declares one
    // descriptor fewer. Same ABI, same order, a different carrier.
    let bufs = [
        device.buffer(&xb).expect("x"),
        device.buffer(&wb).expect("w"),
        device.buffer(&vec![0u8; axis * 2]).expect("out"),
    ];

    let code = module(entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
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
    device
        .run(pipeline, &bound, &params, groups)
        .expect("dispatch");

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
    let device = gpu!();
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

    let code = module(entrypoint);
    let dims = Dims {
        rows: rows as u32,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };

    // Built once, up front: the cache hands out a borrow, so a closure that
    // both builds and dispatches would hold it across two calls.
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
        .expect("the pipeline builds");
    let whole = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    assert_eq!(whole, [4, 1, 1], "one workgroup per row");

    let run = |groups: [u32; 3]| -> Vec<f32> {
        let bufs = [
            device.buffer(&bf16_bytes(&x)).expect("x"),
            device.buffer(&bf16_bytes(&w)).expect("w"),
            device.buffer(&vec![0u8; axis * rows * 2]).expect("out"),
        ];
        let bound: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
        device
            .run(pipeline, &bound, &params, groups)
            .expect("dispatch");
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

/// A pipeline's layout comes from the MODULE and from nothing else.
///
/// 292 of the 481 entrypoints named no operands, including `affine_qmm_t` and
/// most of what a model actually runs. A layout built from such a row has no
/// descriptors, and that is not an error return — it is a segfault inside
/// `vkCreateComputePipelines`. Building from the module's own declared bindings
/// is what makes them loadable, and this fires a sample of them.
///
/// This used to SELECT its sample by asking each entrypoint's row for
/// `Rule::Unstated`, which is how "names no operands" was said while there was
/// a table. There is no table: `KERNELS` is empty, the ask answered `None`
/// every time, and the loop loaded nothing while still reporting a pass shape
/// — it failed only because the sample size is asserted, which is the reason
/// that assertion is here.
///
/// The filter is gone rather than repointed, and the test is STRONGER for it:
/// what it was ever about is that the row has no say in the layout, so the
/// honest sample is "the first forty entrypoints this build produced", stated
/// with no reference to a row at all. A routine states operands the way every
/// other one does, so there is no longer a category of entrypoint that names
/// none — the category was a column.
#[test]
fn an_entrypoint_whose_row_names_no_operands_still_builds_a_pipeline() {
    let device = gpu!();
    let mut cache = Pipelines::new();

    let mut unstated = 0;
    for name in kernels_vulkan::entrypoints() {
        let Some(code) = kernels_vulkan::code(&name, Capability::Baseline) else {
            continue;
        };
        // Nothing here states the scalars, so the widest legal range is the
        // only safe one: any block the module declares fits inside it, and a
        // range narrower than what the shader reads is rejected.
        let push = device.max_push();
        let pipeline = cache
            .get(&device, &name, code, push, 0, Capability::Baseline)
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
    let device = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";
    let code = module(entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
        .expect("the pipeline builds");

    let b = device.buffer(&[0u8; 64]).expect("a buffer");
    // The block this module's push range holds: five words, and the range
    // check refuses any other length. It is built here rather than filled
    // with a plausible number, because two of the three calls below have to
    // get PAST the push check to reach the thing they are about.
    let block = [0u8; 20];

    let one = [Bound::whole(&b)];
    assert!(
        matches!(
            device.run(pipeline, &one, &block, [1, 1, 1]),
            Err(Failed::Bindings { .. })
        ),
        "one buffer under a three-binding module should be refused"
    );

    let three = [Bound::whole(&b), Bound::whole(&b), Bound::whole(&b)];
    // BOTH DIRECTIONS, and the short one is the dangerous one: a range the
    // shader reads past leaves it holding whatever the previous dispatch
    // pushed, which is a plausible number rather than a fault.
    assert!(
        matches!(
            device.run(pipeline, &three, &[1, 2, 3, 4], [1, 1, 1]),
            Err(Failed::Push { .. })
        ),
        "four push bytes against a twenty-byte range should be refused"
    );
    assert!(
        matches!(
            device.run(pipeline, &three, &[0u8; 24], [1, 1, 1]),
            Err(Failed::Push { .. })
        ),
        "twenty-four push bytes against a twenty-byte range should be refused"
    );
    assert!(
        matches!(
            device.run(pipeline, &three, &block, [1, 0, 1]),
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
    let device = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "rms_single_row_bfloat16";
    let code = module(entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
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
    let bound = [Bound::whole(&xb), Bound::whole(&wb), Bound::whole(&ob)];

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
        let refused = device.run(pipeline, &bound, &params, groups);
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
        .run(pipeline, &bound, &params, [1, at, 1])
        .expect("a grid of exactly the limit is legal and must not be refused");

    for b in [xb, wb, ob] {
        device.free(b);
    }
    cache.clear(&device);
}

// `mod rows` STOOD HERE: two `KernelSig` values written down verbatim,
// `norm::layer_scalar_mul` and `attn::kv_append`, recovered from the commits
// that deleted their `kernel!` rows so that the two GPU tests below kept
// exercising the exact cases each was found against.
//
// They are gone because `driver-vulkan` no longer names `KernelSig` anywhere
// -- `src/lowering.rs` and `kernels_vulkan::{bindings, buffer_count,
// push_layout, push_size}` were the last readers and are deleted -- and a
// fixture is not worth being the reason a whole vocabulary has to stay
// compiled. `kernels-wgpu` is the last crate with rows, and Stage 5 deletes
// the macro; a driver holding two rows in a test module would block it.
//
// Both numbers each test took from a row are TRANSCRIBED at the use site now,
// beside the shader line they come from, which is the same trade
// `rules.rs`'s `PARAM_BLOCKS` makes and for the same reason: a number read
// off the source and checked on a device is evidence, and a number derived
// the way the thing under test derives it is not.

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
    let device = gpu!();
    let name = "layer_scalar_mul_bfloat16";
    let Some(code) = kernels_vulkan::code(name, Capability::Baseline) else {
        eprintln!("skipped: `{name}` was not built");
        return;
    };
    // Four, from the row this entrypoint was stated with:
    // `x <- In(0)`, `scalar <- Weight(0)`, `out <- Out(0)`, `params <-
    // Param(0)`. The module decorates three, which is the whole point.
    let stated = 4u32;

    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(
            &device,
            name,
            code,
            // `norm/layer_scalar.slang` declares no push block; every
            // scalar rides the `params` buffer.
            0,
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
    let push: Vec<u8> = Vec::new();
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
    let device = gpu!();
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
        match device.module_for(&name) {
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
    let device = gpu!();
    let mut cache = Pipelines::new();
    let mut built = 0;
    let mut failures: Vec<String> = Vec::new();

    for name in kernels_vulkan::entrypoints().into_iter().take(40) {
        let Some((code, tier)) = device.module_for(&name) else {
            continue;
        };
        // Nothing outside the module states a layout for these forty.
        //
        // This asked each entrypoint's row for a push size and a buffer count
        // and SKIPPED the entrypoint when there was no row -- which, once
        // `KERNELS` emptied, was every one of them: the loop built nothing and
        // the run reported `only 0 pipelines were built`.
        //
        // The row is not repointed at the arm registry, because the number an
        // arm would give is the number the module already declares and
        // comparing a thing to itself proves nothing. What this test is about
        // is the TIER -- that the module `module_for` picks is one this device
        // can really load -- so it now states the widest legal push and no
        // descriptors, and `get` takes the maximum of that and the module's
        // own declarations, which is the path a crossed symbol takes in
        // production too.
        let (push, descriptors) = (device.max_push(), 0);
        match cache.get(&device, &name, code, push, descriptors, tier) {
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
    let device = gpu!();
    let entrypoint = "kv_append_bfloat16";
    let Some(code) = kernels_vulkan::code(entrypoint, Capability::Baseline) else {
        eprintln!("skipped: `{entrypoint}` was not built");
        return;
    };
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

    // `driver_vulkan::pack(row, &[Value::Buffer(0), .., Value::I32(head_dim),
    // Value::Usize(seq * head_dim), Value::Usize(head_dim)])` STOOD HERE. It
    // read a `KernelSig`'s operand kinds and decided which value was a
    // descriptor and which a push field. `src/lowering.rs` is deleted -- there
    // is no row to read -- and what packs a routine's scalars now is
    // `binding::params_from` feeding `encode::Encoder`, which `binding.rs`'s
    // and `encode.rs`'s own tests check and which
    // `a_rectangle_a_real_plan_states_records_and_submits` submits.
    //
    // So the block is TRANSCRIBED, from `attn/kv_write.slang`'s
    // `struct Push { int head_dim; PIE_STRIDE k_head_stride; PIE_STRIDE
    // k_seq_stride; }` and `PIE_STRIDE` being `uint2`: an `int` at 0, four
    // bytes of padding to the 8-byte alignment a `uint2` wants, and two
    // eight-byte strides at 8 and 16. Twenty-four bytes.
    //
    // That is the point of keeping this test at all rather than folding it
    // into the walk. A layout DERIVED the same way as the thing it checks
    // agrees with itself; this one is read off the shader source, handed to a
    // real GPU, and judged by where the write landed. A stride four bytes
    // early is not an approximate answer, it is a write to somewhere else, and
    // the assertions below are exact on both the slot written and the ones
    // that must not be.
    let mut push = Vec::new();
    push.extend_from_slice(&(head_dim as i32).to_le_bytes());
    push.extend_from_slice(&[0u8; 4]);
    push.extend_from_slice(&((seq * head_dim) as u64).to_le_bytes());
    push.extend_from_slice(&(head_dim as u64).to_le_bytes());
    assert_eq!(push.len(), 24, "`Push` is an int, a pad, and two `uint2`");
    let call_buffers: [u32; 5] = [0, 1, 2, 3, 4];

    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(
            &device,
            entrypoint,
            code,
            push.len() as u32,
            call_buffers.len() as u32,
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

    // Densely numbered from zero in the shader's own binding order:
    // `k_new`, `v_new`, `k_dst`, `v_dst`, `pos`.
    let bound: Vec<Bound<'_>> = call_buffers
        .iter()
        .map(|i| Bound::whole(&bufs[*i as usize]))
        .collect();
    device
        .run(pipeline, &bound, &push, groups)
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
    let device = gpu!();
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
    ];
    assert!(
        bound[0].offset() != 0 && bound[2].offset() != 0,
        "a test that binds offset zero cannot see a dropped base"
    );

    let code = module(entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
        .expect("the pipeline builds");
    let dims = Dims {
        rows: 1,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    // The scalars ride a push range now, not a fourth descriptor. The arena
    // still HOLDS them at `p_at`, because the point of this test is that the
    // three operand ranges are addressed from their own offsets and a layout
    // with nothing after the output would put the last one at the end.
    device
        .run(pipeline, &bound, &params, groups)
        .expect("dispatch");

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
    let device = gpu!();
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
    let device = gpu!();
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
    ];

    let code = module(entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
        .expect("the pipeline builds");
    let dims = Dims {
        rows: 1,
        width: axis as u32,
        axis: axis as u32,
        ..Dims::default()
    };
    let groups = groups_for(entrypoint, Rule::Rms, dims, pipeline).expect("a geometry");
    device
        .run(pipeline, &bound, &params, groups)
        .expect("dispatch");

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

/// A parameter run one word short is refused, because the device will not
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
///
/// THE CARRIER MOVED AND THE DEFECT DID NOT. `RmsParams` was a storage struct
/// at binding 3 when this was written, and the short range was a `Bound::at`
/// of sixteen bytes over a twenty-byte block. The five words are a push range
/// now, so the short run is a sixteen-byte slice, and the refusal is
/// `Failed::Push` rather than `Failed::Block`. What a short push leaves the
/// shader reading is WORSE than what a short storage range left it reading:
/// `robustBufferAccess` zeroed the words past a descriptor, and push memory is
/// simply whatever the last dispatch on this command buffer wrote there, which
/// is a plausible number rather than a visible zero.
#[test]
fn a_parameter_block_short_of_what_the_shader_reads_is_refused() {
    let device = gpu!();
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

    let code = module(entrypoint);
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
        .expect("the pipeline builds");
    assert_eq!(
        pipeline.push(),
        20,
        "the module's push range is the five-word block this test is built \
         around"
    );

    let bufs = [
        device.buffer(&bf16_bytes(&x)).expect("x"),
        device.buffer(&bf16_bytes(&w)).expect("w"),
        device.buffer(&vec![0u8; row_bytes as usize]).expect("out"),
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
    let short: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
    match device.run(pipeline, &short, &params[..16], groups) {
        Err(refused) => assert!(
            matches!(
                refused,
                Failed::Push {
                    range: 20,
                    given: 16
                }
            ),
            "the short run must be refused for being SHORT, and was refused \
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
        .run(pipeline, &full, &params, groups)
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
    let device = gpu!();
    let mut cache = Pipelines::new();

    for entrypoint in [
        // Six of seven: the routed matrix-vector kernel a real MoE text fires.
        "affine_qmv_routed_bfloat16_gs_64_b_4",
        // Five of seven, two holes, and from a different family -- so this is
        // not one shader's quirk.
        "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_64_bn_32",
    ] {
        let code = module(entrypoint);
        let words = driver_vulkan::spirv::words(code).expect("whole words");
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
            code,
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
    let device = gpu!();
    let mut cache = Pipelines::new();
    let entrypoint = "affine_qmv_routed_bfloat16_gs_64_b_4";
    let code = module(entrypoint);
    let words = driver_vulkan::spirv::words(code).expect("whole words");
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
        code,
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
    let device = gpu!();
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

    let code = module(entrypoint);
    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
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

    let run_chain = |chained: bool| -> Vec<f32> {
        let rows = device.buffer(&initial).expect("rows");
        let sets: Vec<Vec<Bound<'_>>> = (0..links)
            .map(|i| {
                vec![
                    Bound::at(&device, &rows, i as u64 * stride, stride).expect("in"),
                    Bound::whole(&wb),
                    Bound::at(&device, &rows, (i as u64 + 1) * stride, stride).expect("out"),
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
                    push: &params,
                    groups,
                })
                .collect();
            device.run_all(&run).expect("the chain records and submits");
        } else {
            for b in &sets {
                device.run(pipeline, b, &params, groups).expect("dispatch");
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
    let device = gpu!();
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

    let code = module(entrypoint);
    let mut cache = Pipelines::new();
    let pipeline = cache
        .get(&device, entrypoint, code, 20, 0, Capability::Baseline)
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
                push: &params,
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

    let device = gpu!();
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
    let append = module("kv_append_paged_bfloat16");
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
                    append,
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
    let code = module(symbol);
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
                code,
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

    let device = gpu!();
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
    let plain = module("kv_append_bfloat16");
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
                plain,
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
    let scatter = module("kv_append_paged_bfloat16");
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
                scatter,
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

    let device = gpu!();
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

    let scatter = module("kv_append_paged_bfloat16");
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
                scatter,
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
    let code = module(symbol);
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
                code,
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

    let device = gpu!();
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
    let scatter = module("kv_append_paged_bfloat16");
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
                scatter,
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
    let code = module(symbol);
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
                code,
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

    let device = gpu!();
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
        let code = module(symbol);
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
                    code,
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

    let device = gpu!();
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
    let scatter = module("kv_append_paged_bfloat16");
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
                    scatter,
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
    let code = module(symbol);
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
                code,
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

    let device = gpu!();
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
    let device = gpu!();
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
    let device = gpu!();
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

/// The timing tool is off unless it is asked for.
///
/// Not a formality. Two `vkCmdWriteTimestamp`s per dispatch and a
/// `vkGetQueryPoolResults` per fire is a cost every user of this driver would
/// pay for a number none of them read, and "it is opt-in" is exactly the kind
/// of claim that is true when written and quietly false a year later.
#[test]
fn timing_costs_nothing_when_it_was_not_asked_for() {
    let device = gpu!();
    if std::env::var_os("PIE_VULKAN_TIMING").is_some() {
        eprintln!("PIE_VULKAN_TIMING is set, so there is nothing to say about it unset");
        return;
    }
    assert!(device.timings().is_empty());
    assert_eq!(device.timings_skipped(), 0);
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
    let device = gpu!();
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
    let device = gpu!();

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
    let device = gpu!();
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
    let device = gpu!();
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
    let device = gpu!();
    eprintln!("OPENED: {}", device.name());
}

/// Does the loader offer any device that is not a CPU implementation?
///
/// Asked of `ash` directly rather than of [`Device`], and that is the whole
/// point: the caller is checking that the ranking declined a software adapter,
/// so a premise taken from the ranking would make the check circular. This
/// reads `deviceType` off the loader, which is the one signal the crate under
/// test has no hand in.
///
/// `false` when there is no loader, no instance or no device -- a machine that
/// cannot enumerate has no hardware device for the caller's purposes, and the
/// caller's other arm already handles having nothing to open.
fn any_hardware_device() -> bool {
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        return false;
    };
    // Vulkan 1.0 is deliberate: this asks only for the device list and one
    // property struct, both core since 1.0, so an ICD that refuses a newer
    // version still answers.
    let app = ash::vk::ApplicationInfo::default().api_version(ash::vk::API_VERSION_1_0);
    let info = ash::vk::InstanceCreateInfo::default().application_info(&app);
    let Ok(instance) = (unsafe { entry.create_instance(&info, None) }) else {
        return false;
    };
    let found = unsafe { instance.enumerate_physical_devices() }.is_ok_and(|devices| {
        devices.iter().any(|&d| {
            unsafe { instance.get_physical_device_properties(d) }.device_type
                != ash::vk::PhysicalDeviceType::CPU
        })
    });
    unsafe { instance.destroy_instance(None) };
    found
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
///
/// One relation it asserted did NOT hold there, and the first machine to say
/// so was a container whose only Vulkan device is Mesa's `llvmpipe`: "a
/// software adapter must never be the default" is true only while something
/// else can compute, and this test read the two cases as one. It failed, with
/// a message blaming enumeration order, on a box behaving correctly -- the
/// software adapter was the default because it was the ONLY device, which is
/// the one time picking it is right.
///
/// The condition it was missing cannot be asked of `Device`: whether a
/// hardware device exists is the premise of the ranking under test, so taking
/// the answer from the ranking would make the check circular and it would
/// pass by construction. [`any_hardware_device`] asks the LOADER instead, by
/// `deviceType`, which is the independent signal.
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
    let hardware = any_hardware_device();

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
        // "While anything else can compute" is the half of the rule that was
        // unwritten. With no hardware device on the machine there is nothing
        // to prefer, so the software adapter opening by default is correct
        // and this relation has nothing to say.
        if !hardware {
            eprintln!(
                "the only Vulkan device here is a software adapter ({got}), so \
                 the default HAS to be it -- the ranking is not cross-checked \
                 on this machine"
            );
            continue;
        }
        assert_ne!(
            got, default,
            "this machine offers a software adapter ({got}) and a hardware \
             device, and the software one is ALSO what opens by default. \
             Something is picking by enumeration order again"
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

// ── A `Program` WALKED ONTO THE CARD ────────────────────────────────────
//
// THE CLAIM THIS FILE WAS SHORT OF. Everything above measures the device
// half -- a buffer, a descriptor, a pipeline, a grid -- and
// `tests/the_walk_is_the_program.rs` measures the executor half against a mock
// `Encode` with no adapter in the process. Nothing joined them, so "the baker
// path works" was two green suites and an inference.
//
// What follows is the join, and it is deliberately the SMALLEST program that
// can be one: one statement, `norm.rmsnorm`, walked out of a `Program` by
// `walk::fire::Fire`, encoded by `baker::encode::Encoder`, recorded and
// submitted by `serve::run`, and compared against an `f64` host reference.
//
// It is small because the alternative is not available: no catalog row binds
// a lane on this plane -- every SKU refuses at `gemm.matmul`, there being no
// dense matmul in this shader tree at all -- so `baker::mod`'s
// `every_catalog_row_traces_for_this_plane_and_none_binds_yet` is the test that
// says when a real text can be walked instead. Until then the Program is stated
// by hand, exactly as `the_walk_is_the_program.rs` states its own and for the
// same reason.

/// Rows in the walked fire. Three, so that a per-row grid is distinguishable
/// from a single-block one: `rms_single_row_bfloat16` is one workgroup per row
/// and a driver that dispatched `[1,1,1]` would leave rows 1 and 2 holding
/// whatever the arena was born with.
const WALK_ROWS: i32 = 3;

/// The normalised row's width, in elements.
///
/// 461 on purpose. It is prime, so it is not a multiple of the module's 256
/// workgroup and the tail lane is a real one; and `461 * 2` is 922 bytes, which
/// is not a multiple of any alignment a device reports, so the arena's second
/// rectangle has to be PLACED rather than assumed.
const WALK_WIDTH: u64 = 461;

/// Where the walked fire's output rectangle starts, per row.
///
/// 1024 and not `align16(922)`, and the difference is the finding
/// `tests/arena.rs` carries: `model_compiler::program::carve` rounds to 16, a
/// conformant Vulkan device may ask for 256, and a value's byte base is its
/// offset TIMES the fire's rows. A multiple of 256 is a multiple of 256 however
/// many rows multiply it, so stating this one by hand is what keeps this test
/// about the walk instead of about the alignment. When the carve is 256 again,
/// this constant can be `align16(WALK_WIDTH * 2)` and mean the same thing.
const WALK_PITCH: u64 = 1024;

/// The epsilon inside the root, matching `tests/device_fire.rs`.
const WALK_EPS: f32 = 1e-5;

/// The arena's allocation, as the walk names it.
const WALK_ARENA: driver_vulkan::baker::marks::BufferId = driver_vulkan::baker::marks::BufferId(0);
/// The weight arena's.
const WALK_BANKS: driver_vulkan::baker::marks::BufferId = driver_vulkan::baker::marks::BufferId(1);

/// The fire's staged planes: none of them.
///
/// A `norm.rmsnorm` statement names no cache, no slab and no runtime table, so
/// every answer here is `None` -- and `None` is the honest one rather than a
/// stand-in. A scan handed a null carry answers fluently and wrongly; the same
/// is true of a table of zeros, which is why `Pools` is a trait a driver
/// implements rather than a struct with defaults.
struct NoStaging;

impl driver_vulkan::baker::stage::Pools for NoStaging {
    fn kv(&self, _layer: u32, _values: bool) -> Option<driver_vulkan::baker::marks::Slice> {
        None
    }

    fn slab(
        &self,
        _layer: u32,
        _which: driver_vulkan::baker::stage::Slab,
    ) -> Option<driver_vulkan::baker::marks::Slice> {
        None
    }

    fn kv_geometry(&self) -> driver_vulkan::baker::stage::KvGeometry {
        driver_vulkan::baker::stage::KvGeometry {
            page_size: 0,
            seq_stride: 0,
            head_stride: 0,
            kv_heads: 0,
            head_dim: 0,
        }
    }

    fn table(
        &self,
        _which: driver_vulkan::baker::stage::FireTable,
    ) -> Option<driver_vulkan::baker::marks::Slice> {
        None
    }
}

/// The one-statement plan the walk runs.
fn walk_plan() -> model_ir::plan::Plan {
    use model_ir::plan::{Cond, Op, Param, Plan, Seam, Shard, ValueDef};
    Plan {
        name: "one-rmsnorm".into(),
        plane: model_ir::kernels::Backend::Vulkan,
        facts: vec!["qo_one".into()],
        params: vec![Param {
            name: "norm.weight".into(),
            shape: vec![WALK_WIDTH],
            shard: Shard::Replicated,
            repr: "dense".into(),
        }],
        caches: Vec::new(),
        values: vec![ValueDef::Runtime("token_ids".into()), ValueDef::Stmt(0)],
        ops: vec![Op {
            kernel: "norm.rmsnorm".into(),
            inputs: vec![0],
            outputs: vec![1],
            weights: vec!["norm.weight".into()],
            params: vec![WALK_EPS.to_bits().into()],
            cache: None,
            layer: Some(0),
            cond: Cond::Always,
        }],
        seams: vec![Seam {
            seam: model_ir::seam::OUT.name.to_string(),
            values: vec![1],
            layer: None,
        }],
    }
}

/// The `Program` for it, stated rather than bound.
///
/// `bound` cannot size this one: a result is sizable only if its width rule
/// does not read an operand's rectangle, and `norm.rmsnorm`'s does. Value 0 is
/// the tower's input, which nothing upstream produces here -- so the two slots
/// are stated at the rectangles a real carve would have given them.
fn walk_program() -> model_compiler::program::Program {
    use model_compiler::program::{Call, Dt, Program, Rows, Slot, Step};
    let arena = |offset: u64| Slot::Arena {
        offset,
        rows: Rows::Fire,
        width: WALK_WIDTH,
        dtype: Dt::Bf16,
    };
    Program {
        words: vec![0],
        steps: vec![Step {
            op: 0,
            call: Call::Point("norm.rmsnorm".into()),
        }],
        slots: vec![arena(0), arena(WALK_PITCH)],
        row_pitch: WALK_PITCH * 2,
    }
}

/// The activations the walked fire norms, row-major.
///
/// `tests/device_fire.rs`'s generator, at this file's shape and for its reason:
/// `37` steps through the residues of `71` and `997` moves each row somewhere
/// unrelated, so neighbouring elements are far apart in value and an off-by-one
/// index is a difference of about two rather than of one sixteenth. Every value
/// is a multiple of `1/16` under 2.2, so **every input is bf16-exact** and the
/// comparison is about the kernel's arithmetic rather than about who rounded
/// the inputs.
fn walk_activations() -> Vec<f32> {
    (0..WALK_ROWS as u64)
        .flat_map(|r| {
            (0..WALK_WIDTH).map(move |i| (((r * 997 + i * 37) % 71) as f32 - 35.0) / 16.0)
        })
        .collect()
}

/// The per-element gain. Also bf16-exact, and not constant across the row -- a
/// gain that never varied would hide a shader that dropped `w_stride`.
fn walk_gains() -> Vec<f32> {
    (0..WALK_WIDTH)
        .map(|i| 0.5 + (i % 13) as f32 / 32.0)
        .collect()
}

fn walk_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn walk_from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

fn walk_bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| walk_to_bf16(*x).to_le_bytes())
        .collect()
}

/// A STATEMENT BECOMES A DISPATCH, AND THE DISPATCH IS THE ONE THE MODULE
/// DECLARES.
///
/// Host-side and needs no card, because what it checks is the join between the
/// executor and the reflection: the claim body states TOTAL INVOCATIONS and the
/// module declares `[numthreads]`, and only `serve::run` has read both. Three
/// rows of 461 through `per_axis` is `3 * 256` lanes, which over a workgroup of
/// 256 is three groups -- one per row, which is what
/// `rms_single_row_bfloat16` is written to be.
///
/// It also pins the ARTIFACT the body composed: at `Capability::Baseline` the
/// tier walk has one element, so the file is `"{entrypoint}.spv"` and can be
/// asserted rather than merely checked non-empty.
#[test]
fn a_statement_walks_into_the_dispatch_its_module_declares() {
    use driver_vulkan::baker::encode::Encoder;
    use driver_vulkan::baker::marks::Slice;
    use driver_vulkan::baker::walk::{Extent, Fire};

    if !kernels_vulkan::module::embedded() {
        eprintln!("skipped: no embedded modules in this build");
        return;
    }
    let plan = walk_plan();
    let program = walk_program();
    let banks = walk_banks();
    let pools = NoStaging;
    let fire = Fire::over(
        &plan,
        &program,
        Extent {
            arena: Slice {
                buffer: WALK_ARENA,
                at: 0,
                bytes: WALK_PITCH * 2 * WALK_ROWS as u64,
            },
            rows: WALK_ROWS,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let encoder = Encoder::over(&fire.bindings, &fire.cursor, Capability::Baseline);
    fire.walk(&encoder)
        .unwrap_or_else(|why| panic!("the one-statement walk refused: {why}"));
    let dispatches = encoder.finish();

    assert_eq!(dispatches.len(), 1, "one statement, one dispatch");
    let d = &dispatches[0];
    assert_eq!(d.symbol, "rms_single_row_bfloat16");
    assert_eq!(
        d.file, "rms_single_row_bfloat16.spv",
        "at baseline the tier walk has one element, so the body's artifact is \
         the bare entrypoint"
    );
    assert_eq!(
        d.lanes,
        [WALK_ROWS as u32 * 256, 1, 1],
        "`per_axis` states one workgroup's worth of lanes per axis, and the \
         axis is the whole row"
    );
    assert_eq!(d.args.len(), 3, "x, the gain, and y");
    assert_eq!(
        d.params.len(),
        5,
        "`norm/rms.slang` declares eps, axis_size, w_stride, plus_one, gain"
    );

    // AND THE GRID IS THE DIVISION, which is `serve::run`'s to make because it
    // is the side that read the module.
    let code = kernels_vulkan::module::at(d.file).expect("the tree stamps it");
    let words = driver_vulkan::spirv::words(code).expect("readable SPIR-V");
    let declared = driver_vulkan::spirv::declared(&words).expect("a readable declaration");
    assert_eq!(declared.local, [256, 1, 1]);
    assert_eq!(
        d.lanes[0].div_ceil(declared.local[0]),
        WALK_ROWS as u32,
        "one workgroup per row is what this module is written to be"
    );
}

/// The banks this fixture binds: one dense bf16 gain vector.
fn walk_banks() -> std::collections::BTreeMap<String, driver_vulkan::baker::Bank> {
    use driver_vulkan::baker::Bank;
    use driver_vulkan::baker::marks::Slice;
    [(
        "norm.weight".to_string(),
        Bank {
            slice: Slice {
                buffer: WALK_BANKS,
                at: 0,
                bytes: WALK_WIDTH * 2,
            },
            shape: vec![WALK_WIDTH],
            dtype: model::produce::Dtype::Bf16,
            repr: "dense".to_string(),
        },
    )]
    .into_iter()
    .collect()
}

/// THE PRIZE: a `Program` walked end to end, on this card, and the numbers are
/// the numbers.
///
/// Statement -> `Fire::walk` -> generated dispatch -> `kernels_vulkan::norm`'s
/// claim body -> `Encode::fire` -> `baker::dispatch::Dispatch` -> `serve::run`
/// -> `vkCmdDispatch`. Every link in that chain is exercised, and the last one
/// is checked against an `f64` reference computed the way the shader computes:
/// the sum of squares, the mean, `eps` INSIDE the root, the gain applied to the
/// normalised value.
///
/// The tolerance scales by the row's largest magnitude and is one bf16 ulp
/// (2^-8, about 3.9e-3). A tighter one would be measuring bf16's rounding and
/// a looser one would accept a kernel that had dropped the gain.
#[test]
fn a_program_walked_onto_this_card_computes_what_the_reference_computes() {
    use driver_vulkan::baker::encode::Encoder;
    use driver_vulkan::baker::marks::Slice;
    use driver_vulkan::baker::walk::{Extent, Fire};
    use driver_vulkan::serve::{Embedded, run};

    let device = gpu!();

    // ── the walk, with no device in it ────────────────────────────────
    let plan = walk_plan();
    let program = walk_program();
    let banks = walk_banks();
    let pools = NoStaging;
    let arena_bytes = WALK_PITCH * 2 * WALK_ROWS as u64;
    let fire = Fire::over(
        &plan,
        &program,
        Extent {
            arena: Slice {
                buffer: WALK_ARENA,
                at: 0,
                bytes: arena_bytes,
            },
            rows: WALK_ROWS,
            requests: 1,
            layers: 1,
        },
        &banks,
        &pools,
    );
    let encoder = Encoder::over(&fire.bindings, &fire.cursor, Capability::Baseline);
    fire.walk(&encoder)
        .unwrap_or_else(|why| panic!("the walk refused: {why}"));
    let dispatches = encoder.finish();

    // ── the device, which the walk knew nothing about ─────────────────
    let x = walk_activations();
    let w = walk_gains();
    let mut arena = vec![0u8; arena_bytes as usize];
    let xb = walk_bf16_bytes(&x);
    arena[..xb.len()].copy_from_slice(&xb);

    let arena_buf = device.buffer(&arena).expect("an arena");
    let bank_buf = device.buffer(&walk_bf16_bytes(&w)).expect("a gain vector");
    let mut pipelines = Pipelines::new();

    let fired = run(
        &device,
        &mut pipelines,
        &Embedded,
        &[&arena_buf, &bank_buf],
        &dispatches,
        Capability::Baseline,
    );
    let fired = match fired {
        Ok(f) => f,
        Err(why) => {
            device.free(arena_buf);
            device.free(bank_buf);
            panic!("the walked program did not fire: {why}");
        }
    };
    assert_eq!(fired.dispatches, 1);
    assert_eq!(fired.submissions, 1, "one command buffer for the whole run");
    assert_eq!(
        fired.blocks, 0,
        "`rms_single_row_bfloat16` declares a push block, so nothing is staged \
         in a storage struct"
    );
    assert_eq!(fired.parsed, 1, "one distinct module read");
    assert_eq!(
        fired.tiered, 0,
        "the walk was encoded at `Baseline`, so no body could have reached above it"
    );

    // ── what the card left where the program said it would ────────────
    //
    // At the slot's own byte base, `offset * rows`, and not at a place this
    // test chose: reading anywhere else would pass over a driver that had
    // written the right numbers to the wrong rectangle.
    let at = WALK_PITCH * WALK_ROWS as u64;
    let len = WALK_WIDTH * 2 * WALK_ROWS as u64;
    let raw = device.read_at(&arena_buf, at, len).expect("read back");
    device.free(arena_buf);
    device.free(bank_buf);

    let got: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| walk_from_bf16(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    assert_eq!(got.len(), (WALK_WIDTH * WALK_ROWS as u64) as usize);

    let width = WALK_WIDTH as usize;
    let mut worst = 0.0f64;
    for r in 0..WALK_ROWS as usize {
        let row = &x[r * width..(r + 1) * width];
        let mean = row
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / width as f64;
        let inv = (mean + f64::from(WALK_EPS)).sqrt().recip();
        let scale = row
            .iter()
            .zip(&w)
            .fold(0.0f64, |m, (v, g)| {
                m.max((f64::from(*g) * f64::from(*v) * inv).abs())
            })
            .max(1e-9);
        for i in 0..width {
            let want = f64::from(w[i]) * (f64::from(row[i]) * inv);
            let err = (f64::from(got[r * width + i]) - want).abs() / scale;
            worst = worst.max(err);
        }
    }
    // One bf16 ulp. The mantissa is eight bits, so a relative error of 2^-8 is
    // the smallest difference the format can represent at all.
    assert!(
        worst < 2f64.powi(-8),
        "the walked program's worst relative error is {worst:e}, past one bf16 ulp"
    );
    eprintln!(
        "a Program walked onto this card: {} row(s) of {WALK_WIDTH}, worst relative error {worst:e}",
        WALK_ROWS
    );
}
