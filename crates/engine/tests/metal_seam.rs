//! The Metal seam: can it be selected, and does it say what it cannot serve?
//!
//! A backend that cannot be selected teaches nothing. This checks the half
//! that works — the seam opens from a boot document and dispatches through
//! the contract — and that the half that does not refuses **by name** rather
//! than by absence, panic, or a plausible wrong answer.
//!
//! # What this file used to check, and why it does not
//!
//! It checked the device's stated facts (`unified_memory`, `page_size`, the
//! two arithmetic flags), that `copy_kv`/`resize_pool` refused "before
//! load_model" with a pool-shaped message, and that `load_model` took exactly
//! one descriptor. All three were about `driver_metal::serve::Shell`, which
//! went with the string-plan stack — see `engine::driver::backend::metal`'s
//! header. What is left to check is the shape of the refusal, which is the
//! part a caller depends on.

#![cfg(all(feature = "driver-metal", target_vendor = "apple"))]

use engine::driver::backend::open;

/// A driver that answers `Unsupported` is still a driver: it registers, it
/// names itself, and every verb refuses with the verb in the message.
#[test]
fn the_metal_seam_opens_and_refuses_every_verb_by_name() {
    let mut backend = open::metal(b"{}").expect("the seam opens without a device");
    assert_eq!(backend.kind(), "metal");
    assert!(
        backend.device_facts().is_none(),
        "nothing here has bound a device, and the facts should say so rather \
         than let a scheduler discover it at the first fire"
    );
    assert!(
        backend.export_kv_handle().is_none(),
        "there is no pool to export"
    );

    for error in [
        backend
            .fire(&Default::default())
            .expect_err("there is no shell to fire"),
        backend
            .copy_kv(&Default::default())
            .expect_err("there is no pool to copy within"),
        backend
            .resize_pool(&driver_api::PoolResize {
                pool: driver_api::Pool::Kv,
                target_pages: 0,
                map_ranges: Vec::new(),
                unmap_ranges: Vec::new(),
            })
            .expect_err("there is no pool to resize"),
        backend
            .encode(&mut Default::default())
            .expect_err("there is no encoder"),
    ] {
        let driver_api::DriverError::Unsupported { verb, driver } = &error else {
            panic!("a seam with no shell refuses as `Unsupported`, not as {error}");
        };
        assert_eq!(*driver, "metal", "a refusal names the driver that made it");
        assert!(!verb.is_empty(), "and the verb it was about");
    }
}

/// The boot document is read HERE, and a document that does not parse is not
/// an error.
#[test]
fn the_seam_reads_the_boot_document_and_tolerates_one_that_says_nothing() {
    assert!(open::metal(b"").is_ok(), "an empty document states no id");
    assert!(
        open::metal(b"[model]\nid = \"qwen3\"\n").is_ok(),
        "and one that states an id opens the same way"
    );
}
