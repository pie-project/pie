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
//! one descriptor. All three were about `engine_metal::serve::Shell`, which
//! went with the string-plan stack — see `engine_metal::boot`'s header, which
//! is where that history now lives along with the reader itself. What is left
//! to check is the shape of the refusal, which is the part a caller depends
//! on.

#![cfg(all(feature = "engine-metal", target_vendor = "apple"))]

use runtime::engine::backend::open;

/// An engine that answers `Unsupported` is still an engine: it registers, it
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
            .encode(&mut Default::default())
            .expect_err("there is no encoder"),
    ] {
        let engine::Error::Unsupported { verb, engine } = &error else {
            panic!("a seam with no shell refuses as `Unsupported`, not as {error}");
        };
        assert_eq!(*engine, "metal", "a refusal names the engine that made it");
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
