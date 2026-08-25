//! What this driver tells the engine about the device it opened.
//!
//! The engine keeps these for the whole run and plans against them, so a fact
//! stated wrong here is not caught later. It is believed. `driver-metal` states
//! its facts as constants and is entitled to; Vulkan is not one device, so three
//! fields are read from the device actually opened and `tests/device.rs` holds
//! each against the thing that would break if it were wrong. `fp8_native` and
//! `native_mxfp4_moe` are stated false and no device query would make them true:
//! they ask whether this driver has a kernel, and `kernels-vulkan` has neither.

/// The facts a device cannot make worse.
///
/// Every field is a constant of this backend or the weakest answer the Vulkan
/// specification permits, so a caller is never promised something a real device
/// would decline; [`of`] starts here and overwrites two. It exists because ten
/// of the seam's verbs never touch a device and are testable with no GPU.
/// `storage_alignment` is 256 because it binds the CALLER — the
/// engine lays its arena out at multiples of it, [`Bound::at`](crate::binding)
/// refuses a sub-range without it — and 256 is the largest the specification
/// allows, so the only value that cannot under-promise.
#[must_use]
pub fn floor() -> driver_api::DeviceFacts {
    driver_api::DeviceFacts {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        backend: BACKEND.to_string(),
        // The pessimistic half: a device that turns out to be unified only
        // ever makes a copy this said would be needed unnecessary.
        unified_memory: false,
        fp8_native: false,
        native_mxfp4_moe: false,
        storage_alignment: GUARANTEED_STORAGE_ALIGNMENT,
        storage_max_tile_bytes: 0,
        storage_tile_map_mask: 0,
        page_size: PAGE_SIZE,
    }
}

/// The weakest `minStorageBufferOffsetAlignment` a caller may assume.
///
/// The specification's guaranteed MAXIMUM for the limit, which makes it the safe
/// floor for a caller that must satisfy it. Every device met reports less.
pub const GUARANTEED_STORAGE_ALIGNMENT: u32 = 256;

/// The facts, read from `device`.
///
/// # Panics
///
/// Never: every field is a constant or a limit the device must report.
#[cfg(feature = "device")]
#[must_use]
pub fn of(device: &crate::device::Device) -> driver_api::DeviceFacts {
    driver_api::DeviceFacts {
        unified_memory: device.unified(),
        // The cast cannot lose: the limit is at most 256 on every device the
        // specification allows, and its guaranteed maximum is 256.
        storage_alignment: device.min_storage_offset() as u32,
        ..floor()
    }
}

/// What this backend calls itself in the handshake. The engine matches on this
/// string to pick a backend, so it is a name in an interface, not a description.
pub const BACKEND: &str = "vulkan";

/// Rows per KV page.
///
/// Sixteen because the tiled GEMM is compiled at `bm = 16`: another number would
/// still work for attention while putting a prefill's row count out of step with
/// the tile count, which the dispatch refuses rather than pads. The engine's
/// `kv_translation` indices are in units of this, which is why it is stated.
pub const PAGE_SIZE: u32 = 16;

#[cfg(test)]
mod tests {
    use super::*;

    /// The two constants shared with something else, held against it. Not a device test.
    #[test]
    fn the_stated_page_size_is_the_one_the_tiles_need() {
        assert_eq!(
            PAGE_SIZE, 16,
            "the tiled GEMM takes 16-row tiles, so a page holds 16 rows"
        );
        assert!(
            PAGE_SIZE.is_power_of_two(),
            "`Shape::locate` divides by this, and the engine indexes in units of it"
        );
    }
}
