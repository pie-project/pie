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
//! # Measured, not declared, and here that is not optional
//!
//! `driver-metal` states its facts as constants and is entitled to: it serves
//! one vendor's parts. `driver-vulkan` measures, because the same binary runs
//! on a discrete card, an integrated one and a software implementation.
//!
//! WebGPU is Vulkan's case widened. The same build runs over Vulkan, Metal,
//! D3D12 and -- the point of the backend -- a browser, where the alignment is
//! whatever the page's implementation decided. So the alignment is measured,
//! and the specification's guaranteed FLOOR of 256 is the number a plan has to
//! satisfy to bind anywhere rather than only here.
//!
//! # Why the measured half takes a number and not a device
//!
//! [`of`] is handed a `u32` alignment. `driver-vulkan`'s counterpart takes a
//! `&Device` and is therefore untestable without a GPU -- its own facts test
//! can only check the two constants. Passing the number instead means the
//! whole answer is checkable here, and the device half's job shrinks to asking
//! `wgpu::Limits::min_storage_buffer_offset_alignment` and forwarding it,
//! which is not a place a defect can hide.
//!
//! # The two that are stated rather than measured, and why
//!
//! `fp8_native` and `native_mxfp4_moe` are false, and no adapter query would
//! make them true. They do not ask what the hardware can do; they ask whether
//! this driver has a KERNEL that does it. `kernels-wgpu`'s table is the answer
//! and neither format is in it. A driver that reported `fp8_native` from a
//! device feature bit would be promising a kernel that does not exist.

/// The facts, given the one number that is a property of the device.
///
/// `storage_alignment` is
/// `wgpu::Limits::min_storage_buffer_offset_alignment` from the adapter that
/// was actually opened. It is a parameter rather than a device query for the
/// reason the module doc gives: it makes the whole answer testable with no
/// adapter, and it is the number
/// [`binding::Bound::within`](crate::binding::Bound::within) refuses a
/// sub-range for not having, so stating a different one here would mean the
/// engine's arena laid tensors out at offsets this driver then declined to
/// bind.
///
/// `unified` is the other device fact: whether the adapter reports an
/// integrated or a software device, which is what decides if a staging copy is
/// a copy or a formality.
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
        // NOT a tile map. Both are zero on both siblings for the reason they
        // are zero here: the loader's `TileMap` instructions are executed
        // host-side by `model-loader`, and what reaches this driver is bytes.
        // A non-zero `storage_max_tile_bytes` would be a promise to accept a
        // sparse residency plan nothing here implements -- and WebGPU has no
        // sparse binding at all, so it is a promise this backend could not
        // keep even if something did.
        storage_max_tile_bytes: 0,
        storage_tile_map_mask: 0,
        page_size: PAGE_SIZE,
    }
}

/// What this backend calls itself in the handshake.
///
/// The engine matches on this string to pick a backend, so it is a name in an
/// interface rather than a description. `wgpu` and not `webgpu`: it names the
/// crate the shell is written against, which is what a deployment selects, and
/// it is distinct from `vulkan` even though this backend may well be running
/// over Vulkan -- the engine is choosing a SHELL, not a driver stack.
pub const BACKEND: &str = "wgpu";

/// Rows per KV page.
///
/// Sixteen because the tiled GEMM is compiled at `bm = 16`: a page that held
/// some other number of rows would still work for attention and would put a
/// prefill's row count out of step with the tile count, which the dispatch
/// refuses rather than pads. `kernels-wgpu`'s tile axis is `kernels-metal`'s
/// row for row -- `_bm_16_bn_16`, `_bm_32_bn_32`, `_bm_64_bn_64` -- so the
/// narrowest tile is 16 here as it is on both siblings, and this is one number
/// in two places.
///
/// The engine's `kv_translation` indices are in units of this, which is why it
/// is in the handshake at all: a scheduler that assumed 16 against a driver
/// that used 32 would address every page after the first at the wrong row.
pub const PAGE_SIZE: u32 = 16;

/// The storage-buffer offset alignment WebGPU guarantees, in bytes.
///
/// `wgpu::Limits::downlevel_defaults().min_storage_buffer_offset_alignment`,
/// restated so the portable half can name it without `wgpu` present. A real
/// adapter may report a SMALLER one -- the limit is a maximum a caller may
/// request, and 256 is the value every implementation must accept -- so a plan
/// whose offsets all divide 256 binds everywhere, and one that only divides
/// the local card's number binds here.
pub const GUARANTEED_STORAGE_ALIGNMENT: u32 = 256;

/// The uniform-buffer offset alignment WebGPU guarantees, in bytes.
///
/// The same 256, and named separately because it is a different limit that
/// happens to share a value: `min_uniform_buffer_offset_alignment` is what a
/// launch's parameter block has to start at, and a shell that folded the two
/// into one constant would not notice an implementation that raised one.
pub const GUARANTEED_UNIFORM_ALIGNMENT: u32 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants that are shared with something else, held against it.
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
            "the engine indexes in units of this"
        );
        // And the tile the table's narrowest entrypoint names is that same
        // sixteen, read off the row rather than restated. `driver-vulkan`
        // holds this in its device suite, where it needs a GPU; here it is a
        // string in a table and needs nothing.
        let narrowest = kernels_wgpu::entrypoints()
            .into_iter()
            .filter_map(|e| crate::geometry::Tile::from_entrypoint(&e).map(|t| t.rows))
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

    /// The alignment is forwarded and not overridden.
    ///
    /// A device reporting a coarser granularity than the guaranteed floor is
    /// legal, and a driver that clamped to 256 would tell the engine it may
    /// place a tensor at an offset this driver would then refuse.
    #[test]
    fn a_coarser_device_alignment_reaches_the_engine_unchanged() {
        assert_eq!(of(1024, true).storage_alignment, 1024);
        assert!(of(1024, true).unified_memory);
    }
}
