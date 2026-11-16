#![cfg(target_vendor = "apple")]

//! The device's rates as pie's Metal kernels reach them, for a profile's
//! `[rates]` ([`engine_metal::rates`], the step `pie tune metal --rates`
//! runs). Ignored: run by hand on the device whose file it fills.
//!
//!     cargo test --release -p engine-metal --test engine_metal \
//!         the_device_rates -- --ignored --nocapture

#[test]
#[ignore = "a measurement for a device file, run by hand"]
fn the_device_rates() {
    if !engine_metal::device::present() {
        return;
    }
    let measured =
        engine_metal::rates::measure(None, &mut |line| println!("{line}")).expect("the rates");
    print!("{}", measured.block());
}
