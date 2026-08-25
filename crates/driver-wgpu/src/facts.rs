//! What this driver tells the engine about the device it opened.
//!
//! Every backend's door answers two verbs first: create the driver, and state
//! the device's facts. The engine keeps the answer for the whole run, so a
//! fact stated wrong here is believed. The alignment is MEASURED, because the
//! same build runs over Vulkan, Metal, D3D12 and a browser; the guaranteed
//! FLOOR of 256 is what a plan must satisfy to bind anywhere. `fp8_native` and
//! `native_mxfp4_moe` are stated false because they ask whether this driver
//! has a KERNEL, and `kernels-wgpu`'s table has neither format.

/// The facts, given the one number that is a property of the device.
///
/// `storage_alignment` is the opened adapter's
/// `min_storage_buffer_offset_alignment`: a parameter, not a query, so the
/// whole answer is checkable with no adapter, and it is the number
/// [`Bound::within`](crate::binding::Bound::within) refuses a sub-range for not having.
///
/// # Panics
///
/// Never.
#[must_use]
pub fn of(storage_alignment: u32, unified: bool) -> driver_api::DeviceFacts {
    driver_api::DeviceFacts {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        backend: BACKEND.to_string(),
        unified_memory: unified,
        // See the module doc: a table fact, not a device one.
        fp8_native: false,
        native_mxfp4_moe: false,
        storage_alignment,
        // NOT a tile map: `model-loader` executes the loader's `TileMap`
        // instructions host-side and what reaches this driver is bytes. WebGPU
        // has no sparse binding, so a non-zero value could not be honoured.
        storage_max_tile_bytes: 0,
        storage_tile_map_mask: 0,
        page_size: PAGE_SIZE,
    }
}

/// What this backend calls itself in the handshake.
///
/// The engine matches on this string to pick a backend. `wgpu` and not
/// `webgpu`: it names the crate the shell is written against, and it is
/// distinct from `vulkan` even when this backend runs over Vulkan.
pub const BACKEND: &str = "wgpu";

/// Rows per KV page.
///
/// Sixteen because the tiled GEMM is compiled at `bm = 16`: a page of any
/// other size would put a prefill's row count out of step with the tile count,
/// which the dispatch refuses rather than pads. The engine's `kv_translation`
/// indices are in units of this, so a scheduler assuming 16 against a driver
/// using 32 would address every page after the first at the wrong row.
pub const PAGE_SIZE: u32 = 16;

/// The storage-buffer offset alignment WebGPU guarantees, in bytes.
///
/// `wgpu::Limits::downlevel_defaults().min_storage_buffer_offset_alignment`,
/// restated so the portable half can name it without `wgpu` present. A real
/// adapter may report a SMALLER one, so a plan whose offsets all divide 256
/// binds everywhere and one that divides only the local card's number binds
/// here.
pub const GUARANTEED_STORAGE_ALIGNMENT: u32 = 256;

/// The uniform-buffer offset alignment WebGPU guarantees, in bytes.
///
/// The same 256, named separately because it is a different limit that happens
/// to share a value: folding the two into one constant would not notice an
/// implementation that raised one.
pub const GUARANTEED_UNIFORM_ALIGNMENT: u32 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants that are shared with something else, held against it.
    #[test]
    fn the_stated_page_size_is_the_one_the_tiles_need() {
        // `dispatch.rs` refuses a prefill whose rows are not a whole number
        // of 16-row tiles, so a page of any other size is refused when full.
        assert_eq!(
            PAGE_SIZE, 16,
            "the tiled GEMM takes 16-row tiles, so a page holds 16 rows"
        );
        assert!(
            PAGE_SIZE.is_power_of_two(),
            "the engine indexes in units of this"
        );
        // The tile the table's narrowest entrypoint names is that same
        // sixteen, read off the row rather than restated.
        let narrowest = kernels_wgpu::entrypoints()
            .into_iter()
            .filter_map(|name| {
                let rest = &name[name.find("_bm_")? + 4..];
                rest.chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
                    .parse::<u32>()
                    .ok()
            })
            .min()
            .expect("the table has tiled GEMM entrypoints");
        assert_eq!(
            narrowest, PAGE_SIZE,
            "a page holds one tile's worth of rows, and the table now says {narrowest}"
        );
    }

    /// The whole answer, with no adapter -- which is the point of the shape.
    #[test]
    fn the_facts_are_answerable_without_a_device() {
        let facts = of(256, false);
        assert_eq!(facts.backend, "wgpu");
        assert_eq!(facts.abi_version, driver_api::PIE_DRIVER_ABI_VERSION);
        assert_eq!(facts.page_size, PAGE_SIZE);
        assert_eq!(facts.storage_alignment, 256);
        assert!(!facts.unified_memory);
        // Table facts, not device ones. A backend that grew an fp8 kernel
        // would flip these HERE, next to the table that gained it.
        assert!(!facts.fp8_native);
        assert!(!facts.native_mxfp4_moe);
        // Not a residency plan, and on this backend not one that could be:
        // WebGPU has no sparse binding.
        assert_eq!(facts.storage_max_tile_bytes, 0);
        assert_eq!(facts.storage_tile_map_mask, 0);
    }

    /// The alignment is forwarded and not overridden: a device reporting a
    /// coarser granularity than the guaranteed floor is legal, and clamping to
    /// 256 would promise the engine an offset this driver would then refuse.
    #[test]
    fn a_coarser_device_alignment_reaches_the_engine_unchanged() {
        assert_eq!(of(1024, true).storage_alignment, 1024);
        assert!(of(1024, true).unified_memory);
    }
}
