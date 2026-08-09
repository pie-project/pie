//! What this driver tells the engine about the device it opened.
//!
//! # The first thing the seam asks
//!
//! Every backend's door answers two verbs before anything else can happen:
//! create the driver, and state the device's facts. The engine keeps the
//! answer for the whole run and plans against it -- `driver/backend.rs` clones
//! it out of `create` and hands it to the scheduler -- so a fact stated wrong
//! here is not caught later by anything. It is believed.
//!
//! # Measured, not declared
//!
//! `driver-metal` states its facts as constants, and it is entitled to: it
//! serves one vendor's parts and Apple's alignment is Apple's alignment.
//! Vulkan is not one device. The same binary runs on a discrete NVIDIA card, an
//! integrated Intel one, a phone and a software implementation, and three of
//! the fields below differ between them. So they are read from the device that
//! was actually opened, and [`tests/device.rs`] holds each one against the
//! thing in this crate that would break if it were wrong -- which is a
//! stronger check than holding it against the same limit read twice.
//!
//! # The two that are stated rather than measured, and why
//!
//! `fp8_native` and `native_mxfp4_moe` are false, and no device query would
//! make them true. They do not ask what the hardware can do; they ask whether
//! this driver has a kernel that does it. `kernels-vulkan`'s table is the
//! answer and neither format is in it. A driver that reported `fp8_native`
//! from a device extension would be promising a kernel that does not exist.

/// The facts, read from `device`.
///
/// # Panics
///
/// Never. Every field is either a constant or a limit the specification
/// requires the device to report.
#[cfg(feature = "native")]
#[must_use]
pub fn of(device: &crate::device::Device) -> driver_api::DeviceFacts {
    driver_api::DeviceFacts {
        abi_version: driver_api::PIE_DRIVER_ABI_VERSION,
        backend: BACKEND.to_string(),
        unified_memory: device.unified(),
        // See the module doc: a table fact, not a device one.
        fp8_native: false,
        native_mxfp4_moe: false,
        // `minStorageBufferOffsetAlignment`, which is the alignment
        // `Bound::at` refuses a sub-range for not having. Stating a different
        // number here would mean the engine's arena laid tensors out at
        // offsets this driver then declined to bind.
        //
        // The cast cannot lose: the limit is at most 256 on every device the
        // specification allows, and its guaranteed maximum is 256.
        storage_alignment: device.min_storage_offset() as u32,
        // NOT a tile map. Both are zero on `driver-metal` for the same reason
        // they are zero here: the loader's `TileMap` instructions are executed
        // host-side by `model-loader`, and what reaches this driver is bytes.
        // A non-zero `storage_max_tile_bytes` would be a promise to accept a
        // sparse residency plan nothing here implements.
        storage_max_tile_bytes: 0,
        storage_tile_map_mask: 0,
        page_size: PAGE_SIZE,
    }
}

/// What this backend calls itself in the handshake.
///
/// The engine matches on this string to pick a backend, so it is a name in an
/// interface rather than a description.
pub const BACKEND: &str = "vulkan";

/// Rows per KV page.
///
/// Sixteen because the tiled GEMM is compiled at `bm = 16`: a page that held
/// some other number of rows would still work for attention and would put a
/// prefill's row count out of step with the tile count, which the dispatch
/// refuses rather than pads. So this is one number in two places and the
/// device suite holds them equal.
///
/// The engine's `kv_translation` indices are in units of this, which is why it
/// is in the handshake at all: a scheduler that assumed 16 against a driver
/// that used 32 would address every page after the first at the wrong row.
pub const PAGE_SIZE: u32 = 16;

#[cfg(test)]
mod tests {
    use super::*;

    /// The two constants that are shared with something else, held against it.
    ///
    /// Not a device test: neither depends on which device is open, and a check
    /// that needs a GPU to run is a check that does not run on most machines.
    #[test]
    fn the_stated_page_size_is_the_one_the_tiles_need() {
        // `dispatch.rs` refuses a prefill whose rows are not a whole number of
        // 16-row tiles. A page of any other size would let a caller fill one
        // page exactly and be refused for it.
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
