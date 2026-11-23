//! Where Vulkan keeps what a load armed ([`engine_gpu::armed`]), the build
//! that keys it, and the records the build carries.

/// The version of the layout a recording's commands and its fire block take
/// ([`crate::encode`], [`crate::device::Image`], [`crate::record`]): bumped,
/// with [`ENCODER_LAYOUT`] set again, by a change that moves it. It keys what
/// a load armed beside the kernels' code, so a change elsewhere in the
/// engine keeps every record.
pub const ENCODING: u32 = 1;

/// The hash of the encoder's layout files as [`ENCODING`] was set for them
/// (`build.rs`, `PIE_ENCODER_LAYOUT`): a test holds the two together.
pub const ENCODER_LAYOUT: u64 = 0xff03_512e_f87f_14be;

/// What keys what a load armed: [`ENCODING`] and the code of every kernel
/// the load built (each module's SPIR-V, by its key, in key order).
#[must_use]
pub fn build(pipelines: &crate::device::Pipelines) -> u64 {
    let mut keys = pipelines.built_since(0);
    keys.sort_unstable();
    keys.dedup();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &byte in bytes {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&ENCODING.to_le_bytes());
    for key in keys {
        eat(key.as_bytes());
        eat(&[0]);
        eat(kernels_slang::spirv(key).unwrap_or_default());
        eat(&[0]);
    }
    hash
}

// The committed records (`profiles/bodies/<key>`), by key (`build.rs`).
include!(concat!(env!("OUT_DIR"), "/committed_armed.rs"));

/// The directory a load keeps its records in: `vulkan-bodies` beside the
/// pipeline cache, in the engine's cache. None where the boot names no
/// cache.
#[must_use]
pub fn dir(boot: &crate::api::DeviceBoot) -> Option<std::path::PathBuf> {
    let cache = boot.pipeline_cache.as_deref()?.parent()?;
    Some(cache.join("vulkan-bodies"))
}

/// `pie tune export`: every record of this build the home's cache keeps,
/// copied into `out/bodies` (the engine's `profiles/`). The keys copied.
pub fn export(cache_dir: &std::path::Path, out: &std::path::Path) -> std::io::Result<Vec<String>> {
    engine_gpu::armed::export(&cache_dir.join("vulkan-bodies"), &out.join("bodies"))
}

#[cfg(test)]
mod tests {
    /// A change to the files that lay a recording's commands and its fire
    /// block bumps [`super::ENCODING`] and sets [`super::ENCODER_LAYOUT`]
    /// again: a record armed under another layout must not be taken.
    #[test]
    fn the_encoder_layout_moves_only_with_its_version() {
        let now = u64::from_str_radix(env!("PIE_ENCODER_LAYOUT"), 16).expect("16 hex digits");
        assert_eq!(
            now,
            super::ENCODER_LAYOUT,
            "the encoder's layout files changed (encode.rs, device/block.rs, record.rs): if what a \
             recording's commands or its fire block hold moved, bump `armed::ENCODING`; either \
             way set `armed::ENCODER_LAYOUT` to {now:#018x}"
        );
    }
}
