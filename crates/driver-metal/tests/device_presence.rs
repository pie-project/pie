//! Whether the machine running this has a Metal 4 device, stated out loud.
//!
//! Every device test in this crate opens the same way:
//!
//! ```ignore
//! let Ok(context) = Context::new() else {
//!     eprintln!("SKIP: no Metal 4 device");
//!     return;
//! };
//! ```
//!
//! That is the right shape -- a test that cannot run should not fail -- and
//! it has one bad property: a test that skipped and a test that measured a
//! thousand numbers print the same `ok`. The `driver-metal` job's own
//! comments already name this failure for a different case:
//!
//!   "The name says what the green tick means, because a job that tested
//!    nothing and a job that tested the portable half look identical."
//!
//! The device tests violate exactly that. `device_attention`, `device_gdn`
//! and `device_kernels` between them assert several thousand numbers off real
//! hardware, and until this file existed CI ran none of them and said nothing
//! about why. Adding `--include-ignored` to the job makes them run wherever a
//! device exists; this test is what makes the OTHER case legible, so a build
//! that measured nothing says so instead of looking like a build that
//! measured everything.
//!
//! It is not `#[ignore]`d, because its whole job is to run everywhere.
//!
//! # The switch
//!
//! `PIE_METAL_REQUIRE_DEVICE=1` turns the absence into a failure. Nothing
//! sets it yet, and the reason is that nobody knows the answer: GitHub's
//! `macos-latest` now maps to macOS 26, which is the OS Metal 4 needs, but
//! whether a hosted runner's virtualised Apple GPU actually vends a Metal 4
//! device has never been checked by anything in this tree. The honest thing
//! is to make the runner say, once, in a log -- and then set the variable and
//! keep it true.

#![cfg(feature = "metal-4")]

use driver_metal::device::Context;

/// Say whether there is a device, and fail only if someone asked for one.
#[test]
fn the_runner_states_whether_it_has_a_metal_4_device() {
    let required = std::env::var_os("PIE_METAL_REQUIRE_DEVICE").is_some_and(|v| v != "0");
    match Context::new() {
        Ok(_) => {
            println!(
                "METAL 4 DEVICE: PRESENT. The `#[ignore]`d device tests in this crate \
                 will measure real hardware when run with `--include-ignored`."
            );
        }
        Err(why) => {
            println!(
                "METAL 4 DEVICE: ABSENT ({why}). Every `#[ignore]`d device test in this \
                 crate skips, so a green run of `device_attention`, `device_gdn` or \
                 `device_kernels` on this machine measured NOTHING."
            );
            assert!(
                !required,
                "`PIE_METAL_REQUIRE_DEVICE` is set and there is no Metal 4 device: \
                 {why}. Either this runner lost a capability it used to have, or the \
                 variable is set somewhere it should not be -- a device test that \
                 silently skips is the failure this whole file exists to prevent"
            );
        }
    }
}
