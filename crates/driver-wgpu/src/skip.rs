//! What a wgpu test says when it checked nothing.
//!
//! # Why a test helper is in the library
//!
//! `PIE_WGPU_REQUIRE_DEVICE` already existed, and the workflow already set
//! it, and one test in one file read it. `tests/cooperative.rs` and
//! `tests/serving.rs` open their own adapter -- for reasons each states, and
//! both are good ones -- and when none answered they printed a line `cargo
//! test` swallows and returned. The switch CI sets to prove the llvmpipe
//! install took could not see either of them.
//!
//! A mechanism that lives in one test file is a mechanism for one test file.
//! This is the same ten lines `driver_metal::skip` is, under this backend's
//! own switch, in the one place every test binary here can reach.
//!
//! # The two halves, and why the distinction has to be drawn
//!
//! `PIE_WGPU_REQUIRE_DEVICE` is only worth setting if it can stay green on a
//! runner that has done everything asked of it. llvmpipe is a complete WGSL
//! implementation and offers no cooperative matrix, so a suite that made
//! "this adapter has no cooperative matrix" fatal under the switch would be
//! permanently red on the only runner that runs it, and the switch would be
//! turned off within a week.
//!
//! So: [`skipped`] is for a gate that could have run on some other machine --
//! no adapter answered -- and is fatal under the switch. [`inapplicable`] is
//! for one that could not run anywhere, because the thing it needs is absent
//! from this adapter rather than from this run, and is never fatal.
//!
//! # A third state, and the number that made it necessary
//!
//! `tests/serving.rs` holds the quietest skip in this crate. Every test that
//! wants real weights funnels through one `Option`, and when it was `None`
//! each took its own `else { return }` and was reported `ok` without printing
//! the word SKIP at all -- so not even a human reading `--nocapture` had a
//! word to search for.
//!
//! Routed through [`skipped`] and measured: **23 of that file's 26 tests were
//! passing without measuring anything.** Three were doing the work.
//!
//! That number is why it does not stay under `PIE_WGPU_REQUIRE_DEVICE`. The
//! workflow installs `mesa-vulkan-drivers` and then sets that switch, so the
//! runner has been ASKED for an adapter and failing when it has none is
//! holding it to its word. Nothing has ever asked that runner for a
//! checkpoint. Making it red for something it was never told to provide is
//! how the switch that does work gets unset.
//!
//! So [`unmeasured`] is the third state: it says the same sentence, it is
//! fatal under its own `PIE_WGPU_REQUIRE_WEIGHTS`, and no workflow sets that
//! yet. The gate exists so the day someone points `PIE_CHECKPOINT` at a
//! runner, one variable turns 23 reported passes into 23 measurements.

/// State that this gate checked nothing, and that somewhere else it could
/// have.
///
/// Fatal under `PIE_WGPU_REQUIRE_DEVICE`, which is the whole point: the
/// workflow installs `mesa-vulkan-drivers` immediately before these steps for
/// the express purpose of guaranteeing an adapter, and a green step is not
/// evidence the guarantee held.
#[allow(
    clippy::print_stdout,
    reason = "saying a gate checked nothing is the job"
)]
pub fn skipped(why: &str) {
    assert!(
        !std::env::var_os("PIE_WGPU_REQUIRE_DEVICE").is_some_and(|v| v != "0"),
        "`PIE_WGPU_REQUIRE_DEVICE` is set and this gate measured nothing: \
         {why}. Provide what it names, or unset the switch and accept that \
         this suite reports `ok` for work it did not do."
    );
    println!("SKIP: {why}");
}

/// State that this gate's PREMISE does not hold on this adapter.
///
/// Never fatal. No runner setup fixes an adapter that does not implement
/// cooperative matrix, and a switch that goes red on facts nobody can change
/// is a switch that gets unset.
#[allow(
    clippy::print_stdout,
    reason = "saying a gate checked nothing is the job"
)]
pub fn inapplicable(why: &str) {
    println!("SKIP: {why}");
}

/// State that this gate had no weights to measure.
///
/// Fatal under `PIE_WGPU_REQUIRE_WEIGHTS`, which no workflow sets. A
/// checkpoint is provisionable, so this is not [`inapplicable`]; but it is
/// not [`skipped`] either, because the runner that sets
/// `PIE_WGPU_REQUIRE_DEVICE` was asked for an adapter and never for a model,
/// and a switch that fails on what was never requested is a switch that gets
/// removed rather than satisfied.
///
/// The measurement that separated these: with this routed through `skipped`,
/// `tests/serving.rs` reported `3 passed; 23 failed`. Twenty-three of its
/// twenty-six tests had been green without touching a weight.
#[allow(
    clippy::print_stdout,
    reason = "saying a gate checked nothing is the job"
)]
pub fn unmeasured(why: &str) {
    assert!(
        !std::env::var_os("PIE_WGPU_REQUIRE_WEIGHTS").is_some_and(|v| v != "0"),
        "`PIE_WGPU_REQUIRE_WEIGHTS` is set and this gate measured nothing: \
         {why}. `PIE_CHECKPOINT` names the directory to look in."
    );
    println!("SKIP: {why}");
}
