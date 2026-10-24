#![cfg(target_vendor = "apple")]

//! Every Metal unit kernels-metal ships compiles on this device, whether or not
//! a test fires one of its entries: a unit is compiled whole, so an entry no
//! test reaches still breaks every launch of its neighbours. A unit whose
//! entries a launch stamps at run time compiles to no function; it still has
//! to compile.

use engine_metal::device::{Context, Pipelines};

#[test]
fn every_unit_compiles() {
    let Ok(device) = Context::bind() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let pipelines = Pipelines::new();
    let mut failed = Vec::new();
    let mut units = 0;
    for (file, _) in kernels_metal::SOURCES {
        if !file.ends_with(".metal") {
            continue;
        }
        units += 1;
        if let Err(why) = pipelines.entrypoints(&device, file) {
            failed.push(format!("{file}: {why}"));
        }
    }
    assert!(units > 0, "kernels-metal ships no unit");
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
