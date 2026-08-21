//! What a metal test says when it checked nothing.
//!
//! # Why a test helper is in the library
//!
//! `PIE_METAL_NO_SKIP` is one mechanism, and it is only worth having if
//! there is one of it. The first attempt put these two functions in
//! `driver-metal/tests/common/mod.rs`, which reaches every test in this
//! crate and no test outside it -- and `engine/tests/metal_seam.rs` opens
//! a Metal device eight times and hand-wrote the same skip eight times,
//! under the same flag, invisible to it. A `tests/common` module cannot
//! be the one definition of anything that crosses a crate boundary.
//!
//! So it lives here, where a `dev-dependency` can read it. It is ten lines,
//! it reads one environment variable, and it is the reason a metal run can
//! be made to prove it measured something.

/// State that this gate checked nothing, and why.
///
/// # Every skip in the crate goes through here, and that is the point
///
/// There are thirty-odd of them and five reasons: no checkpoint named, no
/// Metal 4 device, no measurement taken for this checkpoint, this device
/// cannot hold this checkpoint, and this checkpoint does not have the feature
/// the gate is about. All five are honest — a gate that cannot run should not
/// PASS, and asserting a llama reference against gemma reports the rig as a
/// driver defect — and all five print to stderr and let the harness say `ok`.
///
/// Which is fine when a human reads the stderr and fatal when nobody does.
/// `cargo test` reports "19 passed" for a run in which nineteen gates printed
/// SKIP and compared nothing, and that report is indistinguishable from the
/// one where a real device ran the whole suite. The elapsed time is the only
/// tell, and it is not in the summary.
///
/// So `PIE_METAL_NO_SKIP` turns every one of them into a failure. It is opt-in
/// rather than the default because the default has to stay green on a Linux
/// box with no Metal at all — that is what `#[ignore]` and the SKIPs are for —
/// and because the reasons are genuinely different in kind: only the runner
/// knows whether "no measurement for this checkpoint" is a gap it accepts.
/// What it buys is a run that CANNOT lie about having checked something:
/// point it at a checkpoint on a machine with a device, and either every gate
/// compares something or the suite is red.
///
/// Measured on this machine before it existed: the whole target on
/// `Qwen3.6-35B-A3B-4bit` (19 GB, the largest checkpoint here) reports 19
/// passed and skips nothing for the device, so the staging ceiling this was
/// written against does not bite at that size today. "Today" and "at that
/// size" are exactly the qualifications that make the flag worth having.
#[allow(
    clippy::print_stderr,
    reason = "saying a gate checked nothing is the job"
)]
pub fn skipped(why: &str) {
    assert!(
        std::env::var_os("PIE_METAL_NO_SKIP").is_none(),
        "SKIP under PIE_METAL_NO_SKIP: {why}"
    );
    eprintln!("SKIP: {why}");
}

/// State that this gate's PREMISE does not hold for this checkpoint.
///
/// The other half of the split, and the reason [`skipped`] can be made fatal
/// at all. Some gates come in exclusive pairs — a checkpoint either rescales
/// its rope ladder or it does not, and the two gates that check the two lanes
/// cannot both run against one snapshot. No runner setup fixes that, so
/// failing on it under `PIE_METAL_NO_SKIP` would make the flag permanently
/// red and therefore useless.
///
/// The distinction is whose gap it is. A [`skipped`] gate could have run:
/// point at a checkpoint, run on a device with room, take the measurement.
/// One of these could not, and saying so is a different sentence.
#[allow(
    clippy::print_stderr,
    reason = "saying a gate checked nothing is the job"
)]
pub fn inapplicable(why: &str) {
    eprintln!("SKIP: {why}");
}
