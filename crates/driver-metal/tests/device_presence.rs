//! Whether the machine running this has a Metal 4 device, stated out loud.
//!
//! A skipped device test and one that measured real hardware both print an
//! `ok`, so `device_attention`, `device_gdn` and `device_kernels` give no
//! sign of whether they ran on real hardware. This test is not `#[ignore]`d
//! — its job is to run everywhere and say which case applies.
//!
//! `PIE_METAL_REQUIRE_DEVICE=1` turns an absent device into a failure.
//! Nothing sets it yet: whether a hosted `macos-latest` runner actually
//! vends a Metal 4 device has not been confirmed.

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
