//! The two ceilings, read off the real machine.
//!
//! The arithmetic is unit-tested against numbers a machine will not produce
//! on demand. What is left for hardware is that the device and the kernel
//! answer at all, and that the two answers are about the same machine.

#![allow(clippy::print_stdout)]

use driver_metal::Error;
use driver_metal::device::{Context, Memory};

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => {
            driver_metal::skip::skipped("no Metal 4 device, so no allocation was measured");
            None
        }
        Err(e) => panic!("context: {e}"),
    }
}

#[test]
fn a_probe_describes_one_machine() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    let memory = Memory::probe(&context);
    println!("{memory:?}");

    assert!(
        memory.installed >= 1 << 33,
        "{} installed",
        memory.installed
    );
    assert!(
        memory.device_working_set > 0,
        "the device would not say what it will hold"
    );
    assert!(
        memory.device_working_set <= memory.installed,
        "the device claims it will hold more than the machine has"
    );
    assert!(
        memory.reclaimable < memory.installed,
        "everything installed is reclaimable, which would mean nothing is in use"
    );
    assert!(
        memory.wired > 0 && memory.wired < memory.installed,
        "wired = {}",
        memory.wired
    );

    // On Apple silicon the working set is a flat fraction of installed RAM,
    // so it is a ceiling and not a measurement. A machine with most of its
    // memory already spoken for would still report it.
    let fraction = memory.wired_fraction().expect("both numbers");
    assert!((0.0..1.0).contains(&fraction), "{fraction}");

    let ceiling = memory.ceiling().expect("a machine that answers");
    assert_eq!(
        ceiling,
        memory.device_working_set.min(memory.reclaimable),
        "the tighter of the two did not win"
    );
    assert!(memory.headroom(0));
    assert!(!memory.headroom(memory.installed * 2));
}

/// A probe is a value, so refusing a model too big for the GPU needs no
/// override installed anywhere -- which is the whole reason it is a value.
#[test]
fn an_imaginary_machine_needs_no_hook() {
    let tiny = Memory {
        device_working_set: 1 << 20,
        reclaimable: 1 << 30,
        wired: 0,
        installed: 8 << 30,
    };
    assert!(
        !tiny.headroom(19 << 30),
        "a 19 GiB model fit in a 1 MiB GPU"
    );
    assert_eq!(tiny.ceiling(), Some(1 << 20));

    // And the reverse: a device that would hold it on a machine that would
    // not. The C++ needs a `device_working_set_is_forced` predicate here
    // because its two numbers come from different worlds when one is forced;
    // both of these came from the same struct literal.
    let busy = Memory {
        device_working_set: 24 << 30,
        reclaimable: 2 << 30,
        ..tiny
    };
    assert!(!busy.headroom(19 << 30));
    assert_eq!(busy.ceiling(), Some(2 << 30));
}
