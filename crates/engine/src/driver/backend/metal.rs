//! Opening a Metal device from a boot config.
//!
//! # What this file stopped being
//!
//! It was a door onto a shell that did not exist: a `MetalDriver` this crate
//! defined itself, one field (`[model] id`, carried so the refusals could
//! name it), and every verb of the contract answering
//! `DriverError::Unsupported`. Its own header said what was missing —
//! *"a device to bind, a checkpoint to land, pools to reserve, and a command
//! buffer to encode onto"* — and named `driver-cuda/serve.rs` as the shape
//! it would take.
//!
//! It took that shape. `driver-metal` is the whole shell now: `device/`,
//! `weights.rs`, `store/`, the arena, the resident inputs, the windows,
//! `serve.rs`, and the guest-program plane beside them. So this file is what
//! `cuda.rs` beside it is, and for the same reasons decision 13 gives —
//! **the `Driver` impl is the shell's**, in the crate that owns the device,
//! and this module selects one rather than adapting one.
//!
//! # What is left, and it is less than the CUDA door's
//!
//! `cuda.rs` reads two things out of the boot TOML: which device, and how
//! much of a fire to record. Neither exists here. Metal selects with
//! `MTLCreateSystemDefaultDevice` and a Mac has one GPU, so there is no
//! ordinal to parse; and design §6 puts no capture on this plane at all
//! (*"no record.rs: dispatch is encode-only, so `EagerSink` per fire IS
//! encoding"*), so there is no mode to choose. What is left is handing the
//! shell the load door (`crate::driver::load::contract_for`) it cannot state
//! for itself — and taking the document anyway, because a seam that refused
//! to be handed one would be the second thing entitled to an opinion about
//! the file's shape.

use anyhow::{Result, anyhow};
use driver_metal::{DeviceBoot, Metal};

/// Open the system's default Metal device.
///
/// # Errors
///
/// A boot document that is not UTF-8 or not TOML. Binding the device itself
/// happens at [`Driver::load`](driver_api::Driver::load), not here:
/// `Shell::load` is one call that binds, bakes and lands, and there is
/// nothing to bind before a plan says what to bake.
pub fn open(config_bytes: &[u8]) -> Result<Metal> {
    // PARSED AND NOT READ, deliberately. Nothing in this document reaches
    // the metal shell today — see the module doc — but parsing it is what
    // makes a malformed boot file fail HERE, at the door, rather than
    // somewhere later that has nothing to do with it.
    let _doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| anyhow!("the metal boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| anyhow!("the metal boot config is not TOML: {error}"))?;
    Ok(Metal::new(DeviceBoot::default(), crate::driver::load::contract_for))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_boot_document_that_says_nothing_about_this_driver_still_opens() {
        // The ordinary case: a document about some other role, or an empty
        // one. There is no key this seam requires.
        assert!(open(b"").is_ok());
        assert!(open(b"[model]\nid = \"qwen35-d0.8b\"\n").is_ok());
    }

    #[test]
    fn a_boot_document_that_is_not_toml_is_refused_at_the_door() {
        assert!(open(b"this is not = = toml").is_err());
    }
}
