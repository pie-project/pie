//! Where Vulkan keeps what a load armed ([`engine_gpu::armed`]), the build
//! that keys it, and the records the build carries.

/// The hash of the code a Vulkan walk records: this crate's sources, the
/// shared shell's and the Slang kernels' (`build.rs`).
pub const BUILD: u64 = match u64::from_str_radix(env!("PIE_ARMED_BUILD"), 16) {
    Ok(hash) => hash,
    Err(_) => panic!("build.rs writes 16 hex digits"),
};

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
    engine_gpu::armed::export(&cache_dir.join("vulkan-bodies"), &out.join("bodies"), BUILD)
}
