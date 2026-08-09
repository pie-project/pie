//! The boot document, read once.
//!
//! # Why the engine reads this and a driver does not
//!
//! The boot TOML is the ENGINE's format. Three seams said so in their own
//! words — `driver-metal`'s *"a boot TOML is the engine's format. A driver
//! that parsed it would be the second thing entitled to an opinion about the
//! file's shape, and the two would drift"*, and `driver-vulkan`'s repeat of
//! the same sentence — and then each of them parsed it separately.
//!
//! So there were three readers of one format inside one crate, each
//! rebuilding the identical chain (`from_utf8` → `parse::<toml::Table>` →
//! `get("model")?.get(key)`), each with its own idea of what a missing key
//! means. The sentence they all quote is an argument for exactly one reader,
//! and this is it.
//!
//! # What is NOT here, and why that is not an oversight
//!
//! **The defaults.** [`BootConfig`] answers what the document SAYS, in
//! `Option`s. What to do when a key is absent is the seam's, because it is
//! backend-specific: `driver-vulkan` refuses a boot with no module directory
//! (SPIR-V is a build product that has to be found on disk) while
//! `driver-wgpu` cannot be given that key at all (its shaders are in the
//! rlib). A shared default would have to be a guess on behalf of a device
//! this module has never heard of.
//!
//! **The environment fallbacks.** `PIE_KERNELS_VULKAN_SPV_DIR` names one
//! backend in the variable itself, and belongs with the seam that names it.
//!
//! **`[driver]` and `[batching]`.** Those are the DRIVER's knobs —
//! `runahead`, `kv_cache_dtype`, `device`, `tp_size`, `calibrate_planner` —
//! and `driver-cuda` parses them in `crate::boot` on its own side, which is
//! right: they are facts about how that device should behave, and this crate
//! has no opinion about any of them. What is shared is `[model]`, which
//! describes the DEPLOYMENT, and that is what this reads.
//!
//! # A document that does not parse is not an error
//!
//! Two of the four backends are handed a PATH where the other two are handed
//! a document (`worker`'s `embedded_driver` says so at the call site). A path
//! is not TOML, so it parses to nothing and every key reads absent — which is
//! the same answer as an empty document and is what those seams already
//! relied on. Refusing here would break the two that pass a path.

use std::path::PathBuf;

/// What the boot document states, for the keys the engine's own seams read.
///
/// Every field is an `Option` because every field is a question about what
/// the operator wrote, not about what a backend should do — see the module
/// header for where the defaults live instead.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BootConfig {
    /// `[model] config` — a checkpoint's `config.json`, for HF snapshots
    /// whose config does not ride inside the checkpoint.
    pub config: Option<PathBuf>,
    /// `[model] id` — the operator's answer to "which model is this".
    ///
    /// An OVERRIDE and not a selector: a checkpoint is matched to a catalog
    /// row by its tensors, and this exists for the one case tensors cannot
    /// settle — two shape-identical rows where the operator knows which
    /// download this is.
    pub model_id: Option<String>,
    /// `[model] kernels` — a directory of pre-built kernel modules.
    ///
    /// Only `driver-vulkan` has one to look for: SPIR-V is a build product
    /// that has to be found on disk, where `kernels-wgpu` ships its shaders
    /// in the rlib and `driver-metal` and `driver-cuda` compile their own.
    pub kernels: Option<PathBuf>,
    /// `[model] kv_pages` — how many KV pages the pool is opened with.
    ///
    /// `None` means the seam's own default. Zero is treated as absent rather
    /// than honoured: it parses, and it is not a cache.
    pub kv_pages: Option<u32>,
}

impl BootConfig {
    /// Read the document.
    ///
    /// Never fails. Bytes that are not UTF-8, or are not TOML, or are a PATH
    /// rather than a document, all read as "the operator stated nothing" —
    /// which is what the seams handed a path already depended on.
    #[must_use]
    pub fn parse(config_bytes: &[u8]) -> Self {
        let Some(table) = std::str::from_utf8(config_bytes)
            .ok()
            .and_then(|text| text.parse::<toml::Table>().ok())
        else {
            return Self::default();
        };
        // Annotated because `?` inside a closure needs the closure's return
        // type to be inferrable, and every caller below immediately maps it
        // to a different type.
        let model = |key: &str| -> Option<&toml::Value> { table.get("model")?.get(key) };
        Self {
            config: model("config")
                .and_then(toml::Value::as_str)
                .map(PathBuf::from),
            model_id: model("id").and_then(toml::Value::as_str).map(str::to_owned),
            kernels: model("kernels")
                .and_then(toml::Value::as_str)
                .map(PathBuf::from),
            // Zero is dropped HERE rather than at each reader. It parses and
            // is not a cache, and a seam that took it literally would open a
            // pool that can hold no context — which is indistinguishable from
            // a driver that failed to open one.
            kv_pages: model("kv_pages")
                .and_then(toml::Value::as_integer)
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&n| n != 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BootConfig;

    /// The boot document the worker writes is the boot document this reads.
    ///
    /// Two crates, one document, and no shared type between them:
    /// `worker`'s `wgpu_startup_toml` puts `kv_pages` under `[model]`, and
    /// this reads it from there. Nothing in the compiler connects the two --
    /// a key moved to `[batching]`, which is where the Metal writer keeps its
    /// page geometry and so the obvious thing to copy, would read as absent
    /// with no complaint, and an operator who asked for a bigger cache would
    /// get the seam's default and no sign the number was ignored.
    ///
    /// The literal TOML is that document's shape written out by hand on
    /// purpose. Calling the worker's writer would be the same mistake as
    /// sharing a type: the point is that the two agree, and a test that
    /// derives one from the other cannot say so.
    #[test]
    fn the_boot_document_the_worker_writes_is_the_one_this_reads() {
        assert_eq!(
            BootConfig::parse(
                br#"[model]
kv_pages = 4096
"#
            )
            .kv_pages,
            Some(4096)
        );
    }

    /// The shapes that are NOT an error, each for its own reason.
    ///
    /// A hand-written config that says nothing, and a document that does not
    /// parse at all -- which is what a PATH looks like from in here, and the
    /// reason two of the four seams are handed the text instead.
    #[test]
    fn a_document_that_states_nothing_and_one_that_is_a_path_both_read_empty() {
        assert_eq!(BootConfig::parse(b"[model]\n"), BootConfig::default());
        assert_eq!(
            BootConfig::parse(b"/home/someone/.pie/launch/0/driver.toml"),
            BootConfig::default()
        );
        assert_eq!(BootConfig::parse(b""), BootConfig::default());
    }

    /// Zero pages reads as absent, so a seam falls back to its own default.
    ///
    /// It parses, and it is not a cache. Dropping it here is what stops four
    /// seams from each having to remember to.
    #[test]
    fn zero_pages_is_not_a_cache() {
        assert_eq!(BootConfig::parse(b"[model]\nkv_pages = 0\n").kv_pages, None);
        // And a value that cannot be a page count at all.
        assert_eq!(
            BootConfig::parse(b"[model]\nkv_pages = -8\n").kv_pages,
            None
        );
    }

    /// Every `[model]` key the engine's seams read, from one document.
    #[test]
    fn the_model_table_answers_every_key_a_seam_asks_for() {
        let boot = BootConfig::parse(
            br#"[model]
id = "qwen3-0.6b"
config = "/models/qwen3/config.json"
kernels = "/build/spv"
kv_pages = 2048

[driver]
runahead = true
"#,
        );
        assert_eq!(boot.model_id.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(
            boot.config,
            Some(std::path::PathBuf::from("/models/qwen3/config.json"))
        );
        assert_eq!(boot.kernels, Some(std::path::PathBuf::from("/build/spv")));
        assert_eq!(boot.kv_pages, Some(2048));
        // `[driver]` is the DRIVER's table and this does not touch it; see the
        // module header. `driver-cuda` reads it on its own side.
    }
}
