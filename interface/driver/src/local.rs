//! Direct local FFI descriptors for embedded CUDA and Metal drivers.
//!
//! All borrowed pointers in request descriptors are valid only for the
//! duration of the foreign call that receives the enclosing descriptor.
//! Drivers must copy any metadata they need past the return boundary.
//!
//! ABI evolution is version-gated, not `struct_size`-gated: the runtime and
//! driver must reject unknown `abi_version` values on the top-level extensible
//! descriptors below. Explicit `reserved*` fields pin the current C layout and
//! reserve space for append-only evolution without relying on implicit padding.

use std::ffi::c_void;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::geometry::GeometryClass;

/// Current direct local ABI version.
///
/// v12: finalized launches can be prepared into a driver-owned elastic-memory
/// lease and then launched or released exactly once.
pub const PIE_DRIVER_ABI_VERSION: u32 = 12;
pub const PIE_MODEL_COMPONENT_FULL: u32 = 0;
pub const PIE_MODEL_COMPONENT_TEXT: u32 = 1;
pub const PIE_MODEL_COMPONENT_ENCODE: u32 = 2;

/// Success.
pub const PIE_STATUS_OK: i32 = 0;
/// Descriptor validation failed synchronously.
pub const PIE_STATUS_INVALID_ARGUMENT: i32 = -1;
/// ABI version or layout mismatch.
pub const PIE_STATUS_BAD_ABI_VERSION: i32 = -2;
/// The requested operation is not implemented by the driver.
pub const PIE_STATUS_UNSUPPORTED: i32 = -3;
/// The target object is closed or otherwise unavailable.
pub const PIE_STATUS_CLOSED: i32 = -4;
/// The driver encountered an internal failure after accepting the call.
pub const PIE_STATUS_DRIVER_ERROR: i32 = -5;

/// The finalized launch was admitted and `lease_id` is valid.
pub const PIE_LAUNCH_PREPARE_READY: u32 = 0;
/// The launch may fit later after physical budget is released.
pub const PIE_LAUNCH_PREPARE_EXHAUSTED: u32 = 1;
/// The launch can never fit within the driver's physical budget ceiling.
pub const PIE_LAUNCH_PREPARE_IMPOSSIBLE: u32 = 2;

// Literal values so cbindgen emits plain macros; the assert pins them to the
// Rust enum.
pub const PIE_GEOMETRY_CLASS_HOST: u32 = 0;
pub const PIE_GEOMETRY_CLASS_DECODE_ENVELOPE: u32 = 1;
pub const PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY: u32 = 2;
const _: () = {
    assert!(PIE_GEOMETRY_CLASS_HOST == GeometryClass::Host as u32);
    assert!(PIE_GEOMETRY_CLASS_DECODE_ENVELOPE == GeometryClass::DecodeEnvelope as u32);
    assert!(PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY == GeometryClass::DeviceGeometry as u32);
};

/// Reset the recurrent-state slot before executing the request.
pub const PIE_RS_FLAG_RESET: u8 = 1;
/// Fold buffered recurrent-state data into the slot after the pass.
pub const PIE_RS_FLAG_FOLD: u8 = 2;

/// Concrete F32 channel element type.
pub const PIE_CHANNEL_DTYPE_F32: u8 = 0;
/// Concrete I32 channel element type.
pub const PIE_CHANNEL_DTYPE_I32: u8 = 1;
/// Concrete U32 channel element type.
pub const PIE_CHANNEL_DTYPE_U32: u8 = 2;
/// Concrete boolean channel element type.
pub const PIE_CHANNEL_DTYPE_BOOL: u8 = 3;
/// Driver-resolved activation channel element type.
pub const PIE_CHANNEL_DTYPE_ACT: u8 = 4;

/// Channel has no host endpoint.
pub const PIE_CHANNEL_HOST_ROLE_NONE: u8 = 0;
/// Host produces values consumed by the device program.
pub const PIE_CHANNEL_HOST_ROLE_WRITER: u8 = 1;
/// Device program produces values consumed by the host.
pub const PIE_CHANNEL_HOST_ROLE_READER: u8 = 2;

/// Channel is private to one bound instance.
pub const PIE_CHANNEL_EXTERN_NONE: u8 = 0;
/// Bound program consumes an externally produced channel.
pub const PIE_CHANNEL_EXTERN_IMPORT: u8 = 1;
/// Bound program produces an externally consumed channel.
pub const PIE_CHANNEL_EXTERN_EXPORT: u8 = 2;

/// Memory domain tag for local KV residency copies.
pub type PieMemoryDomain = u32;

/// Page-locked host memory.
pub const PIE_MEMORY_DOMAIN_HOST_PINNED: PieMemoryDomain = 0;
/// CUDA device memory on `*_device_ordinal`.
pub const PIE_MEMORY_DOMAIN_CUDA_DEVICE: PieMemoryDomain = 1;
/// ROCm device memory on `*_device_ordinal`.
pub const PIE_MEMORY_DOMAIN_ROCM_DEVICE: PieMemoryDomain = 2;
/// Metal shared CPU/GPU memory.
pub const PIE_MEMORY_DOMAIN_METAL_SHARED: PieMemoryDomain = 3;
/// Metal private device memory.
pub const PIE_MEMORY_DOMAIN_METAL_PRIVATE: PieMemoryDomain = 4;
pub const PIE_ELASTIC_POOL_KV: u64 = 0;
pub const PIE_ELASTIC_POOL_STATE: u64 = 1;
pub const PIE_ELASTIC_POOL_WORKSPACE: u64 = 2;

/// Opaque embedded-driver handle.
pub type PieDriver = c_void;

/// Runtime completion callback, callable from any foreign thread.
///
/// `ctx` is an opaque runtime-owned pointer, forwarded unchanged from
/// [`PieRuntimeCallbacks::ctx`]. Embedded drivers may invoke this callback from
/// any foreign thread after an operation has been accepted.
pub type PieRuntimeNotifyFn =
    Option<unsafe extern "C" fn(ctx: *mut c_void, wait_id: u64, epoch: u64)>;

/// Borrowed immutable byte slice.
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieBytes {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieMutBytes {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Borrowed immutable `u8` slice.
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieU8Slice {
    pub ptr: *const u8,
    pub len: usize,
}

/// Borrowed immutable `u32` slice.
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieU32Slice {
    pub ptr: *const u32,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieU32MutSlice {
    pub ptr: *mut u32,
    pub len: usize,
}

/// Borrowed immutable `u64` slice.
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieU64Slice {
    pub ptr: *const u64,
    pub len: usize,
}

/// Terminal completion outcome published by the native driver.
pub type PieTerminalOutcome = u32;

/// The operation has not reached a terminal state yet.
pub const PIE_TERMINAL_OUTCOME_PENDING: PieTerminalOutcome = 0;
/// The operation completed successfully.
pub const PIE_TERMINAL_OUTCOME_SUCCESS: PieTerminalOutcome = 1;
/// The operation completed unsuccessfully.
pub const PIE_TERMINAL_OUTCOME_FAILED: PieTerminalOutcome = 2;
/// The accepted work item committed no effects and must be attempted again.
pub const PIE_TERMINAL_OUTCOME_RETRY: PieTerminalOutcome = 3;

/// Host-visible terminal control cell.
///
/// The `outcome` word is published with release semantics by the driver and
/// read with acquire semantics by the runtime.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PieTerminalCell {
    pub outcome: PieTerminalOutcome,
    /// Reserved; must be zero.
    pub reserved0: u32,
}

/// Borrowed immutable slice of terminal-cell pointers.
///
/// The slice storage may be null only when `len == 0`. Each element must point
/// at a distinct, properly aligned [`PieTerminalCell`] that remains stable until
/// the corresponding accepted operation retires.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieTerminalCellPtrSlice {
    pub ptr: *const *mut PieTerminalCell,
    pub len: usize,
}

/// Persistent channel endpoint registration descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieChannelDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    pub channel_id: u64,
    pub shape: PieU32Slice,
    /// One of `PIE_CHANNEL_DTYPE_*`.
    pub dtype: u8,
    /// One of `PIE_CHANNEL_HOST_ROLE_*`.
    pub host_role: u8,
    /// Must be 0 or 1.
    pub seeded: u8,
    /// One of `PIE_CHANNEL_EXTERN_*`.
    pub extern_dir: u8,
    pub capacity: u32,
    /// Reserved; must be zero.
    pub reserved1: u32,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
    /// Canonical extern binding name. Empty for private channels.
    pub extern_name: PieBytes,
}

impl Default for PieChannelDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            channel_id: 0,
            shape: PieU32Slice::default(),
            dtype: PIE_CHANNEL_DTYPE_F32,
            host_role: PIE_CHANNEL_HOST_ROLE_NONE,
            seeded: 0,
            extern_dir: PIE_CHANNEL_EXTERN_NONE,
            capacity: 0,
            reserved1: 0,
            reader_wait_id: 0,
            writer_wait_id: 0,
            extern_name: PieBytes::default(),
        }
    }
}

/// Stable driver-owned host endpoint returned by channel registration.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieChannelEndpointBinding {
    pub channel_id: u64,
    pub mirror_base: u64,
    pub word_base: u64,
    pub mirror_bytes: u64,
    pub word_bytes: u64,
    pub cell_bytes: u32,
    pub capacity: u32,
    pub head_word_index: u32,
    pub tail_word_index: u32,
    pub poison_word_index: u32,
    pub closed_word_index: u32,
}

/// One channel-value payload used for PTIR seeds and host puts.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieChannelValueDesc {
    pub channel_id: u64,
    pub bytes: PieBytes,
}

/// Borrowed immutable slice of [`PieChannelValueDesc`].
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieChannelValueDescSlice {
    pub ptr: *const PieChannelValueDesc,
    pub len: usize,
}

/// Flattened mask words with request and row partitions.
///
/// `request_indptr` partitions rows per request; `word_indptr` partitions the
/// packed `u32` words per row.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieMaskWordsDesc {
    pub request_indptr: PieU32Slice,
    pub word_indptr: PieU32Slice,
    pub words: PieU32Slice,
}

/// A single KV cell move expressed in physical page/token coordinates.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PieKvMoveCell {
    pub dst_page_id: u32,
    pub dst_token_offset: u32,
    pub src_page_id: u32,
    pub src_token_offset: u32,
}

/// Borrowed immutable slice of [`PieKvMoveCell`].
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieKvMoveCellSlice {
    pub ptr: *const PieKvMoveCell,
    pub len: usize,
}

/// One recurrent-state slot copy range.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PieStateCopyRange {
    pub src_slot_id: u32,
    pub dst_slot_id: u32,
    pub src_token_offset: u32,
    pub dst_token_offset: u32,
    pub token_count: u32,
}

/// Borrowed immutable slice of [`PieStateCopyRange`].
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieStateCopyRangeSlice {
    pub ptr: *const PieStateCopyRange,
    pub len: usize,
}

/// One sparse pool page range to map or unmap.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiePoolRange {
    pub page_index: u64,
    pub page_count: u64,
}

/// Borrowed immutable slice of [`PiePoolRange`].
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PiePoolRangeSlice {
    pub ptr: *const PiePoolRange,
    pub len: usize,
}

/// Runtime-owned callbacks passed at driver creation time.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PieRuntimeCallbacks {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    /// Opaque runtime-owned context pointer forwarded to [`PieRuntimeNotifyFn`].
    pub ctx: *mut c_void,
    /// Mandatory for embedded native-driver creation.
    pub notify: PieRuntimeNotifyFn,
}

impl Default for PieRuntimeCallbacks {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            ctx: std::ptr::null_mut(),
            notify: None,
        }
    }
}

/// Payload-free asynchronous completion target.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieCompletion {
    pub wait_id: u64,
    pub target_epoch: u64,
    /// Stable terminal control cell for value-less operations. Launch batches
    /// use the per-member `PieLaunchDesc::terminal_cells` instead.
    pub terminal_cell: *mut PieTerminalCell,
}

unsafe impl Send for PieCompletion {}
unsafe impl Sync for PieCompletion {}

/// Driver creation descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PieDriverCreateDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    pub config_bytes: PieBytes,
    pub runtime: PieRuntimeCallbacks,
}

impl Default for PieDriverCreateDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            config_bytes: PieBytes::default(),
            runtime: PieRuntimeCallbacks::default(),
        }
    }
}

/// Driver-owned JSON payload returned from a cold boot call.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieDriverCaps {
    pub json_bytes: *const u8,
    pub json_len: usize,
}

/// Blocking model-load descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieModelLoadDesc {
    pub abi_version: u32,
    /// One of `PIE_MODEL_COMPONENT_*`.
    pub component: u32,
    /// Compiler source hash expected by this runtime.
    pub compiler_version: u64,
    /// Serialized, versioned LoadPlan. Empty plans are invalid.
    pub load_plan_bytes: PieBytes,
    /// UTF-8 path to the driver-local checkpoint payload root.
    pub snapshot_dir: PieBytes,
}

impl Default for PieModelLoadDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            component: PIE_MODEL_COMPONENT_FULL,
            compiler_version: 0,
            load_plan_bytes: PieBytes::default(),
            snapshot_dir: PieBytes::default(),
        }
    }
}

/// Static program registration descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieProgramDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    /// Stable C3 registration/cache key; canonical bytes are only needed on first registration.
    pub program_hash: u64,
    pub canonical_bytes: PieBytes,
    pub sidecar_bytes: PieBytes,
}

impl Default for PieProgramDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            program_hash: 0,
            canonical_bytes: PieBytes::default(),
            sidecar_bytes: PieBytes::default(),
        }
    }
}

/// Per-instance bind descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieInstanceDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    /// Runtime-derived geometry class; the driver verifies this against the
    /// registered trace and echoes it in `PieInstanceBinding`.
    pub geometry_class: u32,
    /// Reserved; must be zero.
    pub reserved1: u32,
    pub program_id: u64,
    pub requested_instance_id: u64,
    pub pacing_wait_id: u64,
    pub channel_ids: PieU64Slice,
    pub seed_values: PieChannelValueDescSlice,
}

impl Default for PieInstanceDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            geometry_class: PIE_GEOMETRY_CLASS_HOST,
            reserved1: 0,
            program_id: 0,
            requested_instance_id: 0,
            pacing_wait_id: 0,
            channel_ids: PieU64Slice::default(),
            seed_values: PieChannelValueDescSlice::default(),
        }
    }
}

/// Driver-assigned identity returned from `*_bind_instance`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieInstanceBinding {
    pub instance_id: u64,
    pub geometry_class: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
}

/// One batched launch descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieLaunchDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    /// Bound instance ids, one per fire/program in scheduler order.
    pub instance_ids: PieU64Slice,
    /// Stable terminal control cell addresses, one per `instance_ids` member.
    pub terminal_cells: PieTerminalCellPtrSlice,
    pub token_ids: PieU32Slice,
    pub position_ids: PieU32Slice,
    pub kv_page_indices: PieU32Slice,
    pub kv_page_indptr: PieU32Slice,
    pub kv_last_page_lens: PieU32Slice,
    pub qo_indptr: PieU32Slice,
    /// Folded recurrent-state slot per resolved `qo_indptr` row. Empty for
    /// pure-attention launches; never indexed by `instance_ids`.
    pub rs_slot_ids: PieU32Slice,
    /// Flags parallel to `rs_slot_ids`.
    pub rs_slot_flags: PieU8Slice,
    pub rs_fold_lens: PieU32Slice,
    pub rs_buffer_slot_ids: PieU32Slice,
    pub rs_buffer_slot_indptr: PieU32Slice,
    pub masks: PieMaskWordsDesc,
    /// Model readout rows, flattened across the batch.
    pub sampling_indices: PieU32Slice,
    /// Request → readout-row CSR parallel to the batch.
    pub sampling_indptr: PieU32Slice,
    pub context_ids: PieU64Slice,
    /// Boolean `0`/`1`; any other value is invalid.
    pub single_token_mode: u8,
    /// Boolean `0`/`1`; any other value is invalid.
    pub has_user_mask: u8,
    /// Reserved; must be zero.
    pub reserved_flags: [u8; 2],
    /// Exclusive physical KV page high-water required before this launch.
    pub required_kv_pages: u32,
    pub image_indptr: PieU32Slice,
    pub image_grids: PieU32Slice,
    pub image_anchor_positions: PieU32Slice,
    pub image_pixels: PieBytes,
    pub image_pixel_indptr: PieU32Slice,
    pub image_mrope_positions: PieU32Slice,
    pub image_mrope_indptr: PieU32Slice,
    pub image_patch_positions: PieU32Slice,
    pub image_anchor_rows: PieU32Slice,
    pub audio_features: PieBytes,
    pub audio_feature_indptr: PieU32Slice,
    pub audio_anchor_rows: PieU32Slice,
    pub audio_indptr: PieU32Slice,
    pub embed_rows: PieBytes,
    pub embed_indptr: PieU32Slice,
    pub embed_shapes: PieU32Slice,
    pub embed_dtypes: PieU8Slice,
    pub embed_anchor_rows: PieU32Slice,
    pub embed_block_indptr: PieU32Slice,
    pub kv_len: PieU32Slice,
    pub kv_len_device: PieU64Slice,
    /// Per-instance WorkingSet page translation, flattened across the batch:
    /// entry `i` of an instance's segment is the PHYSICAL KV page id backing
    /// WorkingSet-relative page index `i` for THIS fire (committed mapping
    /// overlaid with the fire's prepared write targets). The driver maps any
    /// WorkingSet-relative page reference it resolves from device channels
    /// (`Pages` / `WSlot` descriptor ports) through this table; guests never
    /// see physical ids (kv_refact.md, flattened-table model). An empty
    /// segment means the instance's channel geometry is already physical
    /// (legacy) or absent.
    pub kv_translation: PieU32Slice,
    /// CSR partition of `kv_translation`, one segment per `instance_ids`
    /// entry (`len == instance_ids.len + 1` when present, else empty).
    pub kv_translation_indptr: PieU32Slice,
    /// Program → wire-request attribution CSR (`len == instance_ids.len + 1`
    /// when present, else empty): program `p` owns the wire request rows
    /// `[row_indptr[p], row_indptr[p+1])` of `qo_indptr`/`kv_page_indptr`/
    /// `sampling_indptr`. A device-geometry program's span is its empty
    /// wire placeholder row; the driver substitutes its channel-resolved
    /// geometry for that span when composing the forward batch.
    pub ptir_program_row_indptr: PieU32Slice,
    pub ptir_kv_write_lower_bounds: PieU64Slice,
    pub ptir_kv_write_upper_bounds: PieU64Slice,
    /// Immutable logical-fire ids, one per instance.
    pub logical_fire_ids: PieU64Slice,
    /// Dense-channel sequence tickets, CSR-partitioned per instance.
    pub channel_expected_head: PieU64Slice,
    pub channel_expected_tail: PieU64Slice,
    pub channel_ticket_indptr: PieU32Slice,
}

impl Default for PieLaunchDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            instance_ids: PieU64Slice::default(),
            terminal_cells: PieTerminalCellPtrSlice::default(),
            token_ids: PieU32Slice::default(),
            position_ids: PieU32Slice::default(),
            kv_page_indices: PieU32Slice::default(),
            kv_page_indptr: PieU32Slice::default(),
            kv_last_page_lens: PieU32Slice::default(),
            qo_indptr: PieU32Slice::default(),
            rs_slot_ids: PieU32Slice::default(),
            rs_slot_flags: PieU8Slice::default(),
            rs_fold_lens: PieU32Slice::default(),
            rs_buffer_slot_ids: PieU32Slice::default(),
            rs_buffer_slot_indptr: PieU32Slice::default(),
            masks: PieMaskWordsDesc::default(),
            sampling_indices: PieU32Slice::default(),
            sampling_indptr: PieU32Slice::default(),
            context_ids: PieU64Slice::default(),
            single_token_mode: 0,
            has_user_mask: 0,
            reserved_flags: [0; 2],
            required_kv_pages: 0,
            image_indptr: PieU32Slice::default(),
            image_grids: PieU32Slice::default(),
            image_anchor_positions: PieU32Slice::default(),
            image_pixels: PieBytes::default(),
            image_pixel_indptr: PieU32Slice::default(),
            image_mrope_positions: PieU32Slice::default(),
            image_mrope_indptr: PieU32Slice::default(),
            image_patch_positions: PieU32Slice::default(),
            image_anchor_rows: PieU32Slice::default(),
            audio_features: PieBytes::default(),
            audio_feature_indptr: PieU32Slice::default(),
            audio_anchor_rows: PieU32Slice::default(),
            audio_indptr: PieU32Slice::default(),
            embed_rows: PieBytes::default(),
            embed_indptr: PieU32Slice::default(),
            embed_shapes: PieU32Slice::default(),
            embed_dtypes: PieU8Slice::default(),
            embed_anchor_rows: PieU32Slice::default(),
            embed_block_indptr: PieU32Slice::default(),
            kv_len: PieU32Slice::default(),
            kv_len_device: PieU64Slice::default(),
            kv_translation: PieU32Slice::default(),
            kv_translation_indptr: PieU32Slice::default(),
            ptir_program_row_indptr: PieU32Slice::default(),
            ptir_kv_write_lower_bounds: PieU64Slice::default(),
            ptir_kv_write_upper_bounds: PieU64Slice::default(),
            logical_fire_ids: PieU64Slice::default(),
            channel_expected_head: PieU64Slice::default(),
            channel_expected_tail: PieU64Slice::default(),
            channel_ticket_indptr: PieU32Slice::default(),
        }
    }
}

/// Result of synchronously preparing one finalized launch.
///
/// `lease_id` is nonzero only for [`PIE_LAUNCH_PREPARE_READY`]. A ready lease
/// is consumed by exactly one `*_launch_prepared` or `*_release_launch` call.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieLaunchPrepareResult {
    pub outcome: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    pub lease_id: u64,
    /// Monotonic physical-budget generation observed by this attempt.
    pub budget_generation: u64,
    /// Independently rounded physical pages required by the finalized launch.
    pub required_pages: u64,
    /// Current physical budget in the same page units.
    pub budget_pages: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieEncodeDesc {
    pub abi_version: u32,
    pub reserved0: u32,
    pub image_grids: PieU32Slice,
    pub image_pixels: PieBytes,
    pub image_pixel_indptr: PieU32Slice,
    pub image_patch_positions: PieU32Slice,
    pub image_anchor_rows: PieU32Slice,
    pub audio_features: PieBytes,
    pub audio_feature_indptr: PieU32Slice,
    pub audio_anchor_rows: PieU32Slice,
    pub output_rows: PieMutBytes,
    pub output_row_indptr: PieU32MutSlice,
}

impl Default for PieEncodeDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            image_grids: PieU32Slice::default(),
            image_pixels: PieBytes::default(),
            image_pixel_indptr: PieU32Slice::default(),
            image_patch_positions: PieU32Slice::default(),
            image_anchor_rows: PieU32Slice::default(),
            audio_features: PieBytes::default(),
            audio_feature_indptr: PieU32Slice::default(),
            audio_anchor_rows: PieU32Slice::default(),
            output_rows: PieMutBytes::default(),
            output_row_indptr: PieU32MutSlice::default(),
        }
    }
}

/// Direct KV-copy descriptor.
///
/// Whole-page residency copies use the `src_page_ids`/`dst_page_ids` slices. The
/// optional `cells` slice carries per-token device KV moves for Design-B compaction.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieKvCopyDesc {
    pub abi_version: u32,
    pub src_domain: PieMemoryDomain,
    pub src_device_ordinal: u32,
    pub dst_domain: PieMemoryDomain,
    pub dst_device_ordinal: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    pub src_page_ids: PieU32Slice,
    pub dst_page_ids: PieU32Slice,
    pub cells: PieKvMoveCellSlice,
}

impl Default for PieKvCopyDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            src_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            src_device_ordinal: 0,
            dst_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            dst_device_ordinal: 0,
            reserved0: 0,
            src_page_ids: PieU32Slice::default(),
            dst_page_ids: PieU32Slice::default(),
            cells: PieKvMoveCellSlice::default(),
        }
    }
}

/// Direct recurrent-state copy descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieStateCopyDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    pub slot_ranges: PieStateCopyRangeSlice,
}

impl Default for PieStateCopyDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            slot_ranges: PieStateCopyRangeSlice::default(),
        }
    }
}

/// Direct pool-resize descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PiePoolResizeDesc {
    pub abi_version: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
    pub pool_id: u64,
    pub target_pages: u64,
    pub map_ranges: PiePoolRangeSlice,
    pub unmap_ranges: PiePoolRangeSlice,
}

impl Default for PiePoolResizeDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            pool_id: 0,
            target_pages: 0,
            map_ranges: PiePoolRangeSlice::default(),
            unmap_ranges: PiePoolRangeSlice::default(),
        }
    }
}

/// ABI descriptor validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieAbiValidationError {
    status: i32,
    message: &'static str,
}

impl PieAbiValidationError {
    pub const fn new(status: i32, message: &'static str) -> Self {
        Self { status, message }
    }

    pub const fn status(self) -> i32 {
        self.status
    }

    pub const fn message(self) -> &'static str {
        self.message
    }
}

impl fmt::Display for PieAbiValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (status {})", self.message, self.status)
    }
}

impl std::error::Error for PieAbiValidationError {}

pub type PieAbiValidationResult = Result<(), PieAbiValidationError>;

const fn abi_version_error() -> PieAbiValidationError {
    PieAbiValidationError::new(
        PIE_STATUS_BAD_ABI_VERSION,
        "descriptor abi_version does not match PIE_DRIVER_ABI_VERSION",
    )
}

const fn invalid_argument(message: &'static str) -> PieAbiValidationError {
    PieAbiValidationError::new(PIE_STATUS_INVALID_ARGUMENT, message)
}

const fn bool_field_is_valid(value: u8) -> bool {
    value <= 1
}

/// Returns true when `outcome` is a valid [`PieTerminalOutcome`] discriminant.
pub const fn pie_terminal_outcome_is_valid(outcome: PieTerminalOutcome) -> bool {
    matches!(
        outcome,
        PIE_TERMINAL_OUTCOME_PENDING
            | PIE_TERMINAL_OUTCOME_SUCCESS
            | PIE_TERMINAL_OUTCOME_FAILED
            | PIE_TERMINAL_OUTCOME_RETRY
    )
}

/// Returns true when `domain` is a valid [`PieMemoryDomain`] discriminant.
pub const fn pie_memory_domain_is_valid(domain: PieMemoryDomain) -> bool {
    matches!(
        domain,
        PIE_MEMORY_DOMAIN_HOST_PINNED
            | PIE_MEMORY_DOMAIN_CUDA_DEVICE
            | PIE_MEMORY_DOMAIN_ROCM_DEVICE
            | PIE_MEMORY_DOMAIN_METAL_SHARED
            | PIE_MEMORY_DOMAIN_METAL_PRIVATE
    )
}

/// Validates a top-level ABI version tag.
pub const fn validate_pie_abi_version(abi_version: u32) -> PieAbiValidationResult {
    if abi_version == PIE_DRIVER_ABI_VERSION {
        Ok(())
    } else {
        Err(abi_version_error())
    }
}

fn validate_reserved_zero(name: &'static str, value: u32) -> PieAbiValidationResult {
    if value == 0 {
        Ok(())
    } else {
        Err(invalid_argument(name))
    }
}

fn validate_reserved_bytes_zero(name: &'static str, value: &[u8]) -> PieAbiValidationResult {
    if value.iter().all(|byte| *byte == 0) {
        Ok(())
    } else {
        Err(invalid_argument(name))
    }
}

fn validate_bool_field(name: &'static str, value: u8) -> PieAbiValidationResult {
    if bool_field_is_valid(value) {
        Ok(())
    } else {
        Err(invalid_argument(name))
    }
}

fn checked_slice_bytes<T>(len: usize, name: &'static str) -> PieAbiValidationResult {
    if len.checked_mul(std::mem::size_of::<T>()).is_some() {
        Ok(())
    } else {
        Err(invalid_argument(name))
    }
}

fn validate_slice_ptr<T>(ptr: *const T, len: usize, name: &'static str) -> PieAbiValidationResult {
    checked_slice_bytes::<T>(len, name)?;
    if len != 0 && ptr.is_null() {
        Err(invalid_argument(name))
    } else {
        Ok(())
    }
}

fn validate_mut_ptr<T>(ptr: *mut T, name: &'static str) -> PieAbiValidationResult {
    if ptr.is_null() {
        Err(invalid_argument(name))
    } else {
        Ok(())
    }
}

fn validate_bytes(bytes: PieBytes, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(bytes.ptr, bytes.len, name)
}

fn validate_mut_bytes(bytes: PieMutBytes, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(bytes.ptr.cast_const(), bytes.len, name)
}

fn validate_terminal_cell_ptr(
    ptr: *mut PieTerminalCell,
    name: &'static str,
) -> PieAbiValidationResult {
    if ptr.is_null() {
        return Err(invalid_argument(name));
    }
    if (ptr as usize) % std::mem::align_of::<PieTerminalCell>() != 0 {
        return Err(invalid_argument(name));
    }
    Ok(())
}

fn validate_u8_slice(slice: PieU8Slice, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_u32_slice(slice: PieU32Slice, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_u32_mut_slice(slice: PieU32MutSlice, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr.cast_const(), slice.len, name)
}

fn validate_u64_slice(slice: PieU64Slice, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_terminal_cell_ptr_slice(
    slice: PieTerminalCellPtrSlice,
    name: &'static str,
) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)?;
    if slice.len == 0 {
        return Ok(());
    }
    let ptrs = unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) };
    for &cell in ptrs {
        validate_terminal_cell_ptr(cell, name)?;
    }
    Ok(())
}

fn validate_channel_value_desc_slice(
    slice: PieChannelValueDescSlice,
    name: &'static str,
) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_kv_move_cell_slice(
    slice: PieKvMoveCellSlice,
    name: &'static str,
) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_state_copy_range_slice(
    slice: PieStateCopyRangeSlice,
    name: &'static str,
) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_pool_range_slice(
    slice: PiePoolRangeSlice,
    name: &'static str,
) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_row_count_u32(
    len: usize,
    name: &'static str,
    outer_count: usize,
    allow_empty: bool,
) -> PieAbiValidationResult {
    if len == 0 && allow_empty {
        return Ok(());
    }
    if len == outer_count {
        Ok(())
    } else {
        Err(invalid_argument(name))
    }
}

fn validate_indptr_shape(
    len: usize,
    name: &'static str,
    outer_count: usize,
    allow_empty: bool,
) -> PieAbiValidationResult {
    if len == 0 && allow_empty {
        return Ok(());
    }
    if len == outer_count.saturating_add(1) {
        Ok(())
    } else {
        Err(invalid_argument(name))
    }
}

unsafe fn as_u32_slice<'a>(
    slice: PieU32Slice,
    name: &'static str,
) -> Result<&'a [u32], PieAbiValidationError> {
    validate_u32_slice(slice, name)?;
    if slice.len == 0 {
        Ok(&[])
    } else {
        Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) })
    }
}

unsafe fn as_channel_values<'a>(
    slice: PieChannelValueDescSlice,
    name: &'static str,
) -> Result<&'a [PieChannelValueDesc], PieAbiValidationError> {
    validate_channel_value_desc_slice(slice, name)?;
    if slice.len == 0 {
        Ok(&[])
    } else {
        Ok(unsafe { std::slice::from_raw_parts(slice.ptr, slice.len) })
    }
}

unsafe fn validate_nested_channel_values(
    slice: PieChannelValueDescSlice,
    slice_name: &'static str,
    bytes_name: &'static str,
) -> PieAbiValidationResult {
    for value in unsafe { as_channel_values(slice, slice_name)? } {
        validate_bytes(value.bytes, bytes_name)?;
    }
    Ok(())
}

unsafe fn validate_csr(
    indptr: PieU32Slice,
    indptr_name: &'static str,
    values_len: usize,
    outer_count: usize,
    allow_empty: bool,
) -> PieAbiValidationResult {
    validate_indptr_shape(indptr.len, indptr_name, outer_count, allow_empty)?;
    let indptr_values = unsafe { as_u32_slice(indptr, indptr_name)? };
    if indptr_values.is_empty() {
        return Ok(());
    }
    if indptr_values[0] != 0 {
        return Err(invalid_argument(indptr_name));
    }
    let mut previous = 0u32;
    for &value in indptr_values {
        if value < previous {
            return Err(invalid_argument(indptr_name));
        }
        previous = value;
    }
    let last = previous as usize;
    if last > values_len {
        return Err(invalid_argument(indptr_name));
    }
    Ok(())
}

/// Validates runtime callbacks carried in [`PieDriverCreateDesc`].
pub fn validate_runtime_callbacks(callbacks: &PieRuntimeCallbacks) -> PieAbiValidationResult {
    validate_pie_abi_version(callbacks.abi_version)?;
    validate_reserved_zero("runtime reserved0 must be zero", callbacks.reserved0)?;
    if callbacks.notify.is_none() {
        return Err(invalid_argument(
            "runtime notify callback must be non-null at driver create",
        ));
    }
    Ok(())
}

/// Validates a driver-create descriptor.
pub fn validate_driver_create_desc(desc: &PieDriverCreateDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("create reserved0 must be zero", desc.reserved0)?;
    validate_bytes(desc.config_bytes, "create config_bytes ptr/len mismatch")?;
    validate_runtime_callbacks(&desc.runtime)
}

/// Validates a model-load descriptor.
pub fn validate_model_load_desc(desc: &PieModelLoadDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    if desc.component > PIE_MODEL_COMPONENT_ENCODE {
        return Err(invalid_argument("model load component is invalid"));
    }
    validate_bytes(
        desc.load_plan_bytes,
        "model load load_plan_bytes ptr/len mismatch",
    )?;
    validate_bytes(
        desc.snapshot_dir,
        "model load snapshot_dir ptr/len mismatch",
    )?;
    if desc.load_plan_bytes.len == 0 {
        return Err(invalid_argument(
            "model load requires non-empty load_plan_bytes",
        ));
    }
    if desc.compiler_version == 0 {
        return Err(invalid_argument(
            "model load requires a nonzero compiler_version",
        ));
    }
    if desc.snapshot_dir.len == 0 {
        return Err(invalid_argument(
            "model load requires non-empty snapshot_dir",
        ));
    }
    Ok(())
}

/// Validates a program-registration descriptor.
pub fn validate_program_desc(desc: &PieProgramDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("program reserved0 must be zero", desc.reserved0)?;
    validate_bytes(
        desc.canonical_bytes,
        "program canonical_bytes ptr/len mismatch",
    )?;
    validate_bytes(desc.sidecar_bytes, "program sidecar_bytes ptr/len mismatch")
}

/// Validates an instance-bind descriptor.
pub unsafe fn validate_instance_desc(desc: &PieInstanceDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("instance reserved0 must be zero", desc.reserved0)?;
    if GeometryClass::try_from(desc.geometry_class).is_err() {
        return Err(invalid_argument("instance geometry_class is invalid"));
    }
    validate_reserved_zero("instance reserved1 must be zero", desc.reserved1)?;
    validate_u64_slice(desc.channel_ids, "instance channel_ids ptr/len mismatch")?;
    unsafe {
        validate_nested_channel_values(
            desc.seed_values,
            "instance seed_values ptr/len mismatch",
            "instance seed_values bytes ptr/len mismatch",
        )
    }
}

/// Validates a persistent channel endpoint registration descriptor.
pub unsafe fn validate_channel_desc(desc: &PieChannelDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("channel reserved0 must be zero", desc.reserved0)?;
    validate_reserved_zero("channel reserved1 must be zero", desc.reserved1)?;
    if desc.channel_id == 0 {
        return Err(invalid_argument("channel id must be nonzero"));
    }
    validate_u32_slice(desc.shape, "channel shape ptr/len mismatch")?;
    if desc.dtype > PIE_CHANNEL_DTYPE_ACT {
        return Err(invalid_argument("channel dtype is invalid"));
    }
    if desc.host_role > PIE_CHANNEL_HOST_ROLE_READER {
        return Err(invalid_argument("channel host_role is invalid"));
    }
    validate_bool_field("channel seeded must be 0 or 1", desc.seeded)?;
    if desc.extern_dir > PIE_CHANNEL_EXTERN_EXPORT {
        return Err(invalid_argument("channel extern_dir is invalid"));
    }
    if desc.capacity == 0 {
        return Err(invalid_argument("channel capacity must be nonzero"));
    }
    if desc.reader_wait_id == 0
        || desc.writer_wait_id == 0
        || desc.reader_wait_id == desc.writer_wait_id
    {
        return Err(invalid_argument(
            "channel reader and writer wait ids must be nonzero and distinct",
        ));
    }
    validate_bytes(desc.extern_name, "channel extern_name ptr/len mismatch")?;
    if desc.extern_dir == PIE_CHANNEL_EXTERN_NONE {
        if desc.extern_name.len != 0 {
            return Err(invalid_argument(
                "private channel must not have an extern name",
            ));
        }
    } else if desc.extern_name.len == 0
        || desc.host_role != PIE_CHANNEL_HOST_ROLE_NONE
        || desc.seeded != 0
    {
        return Err(invalid_argument(
            "extern channel requires a name, no host role, and no seed",
        ));
    }
    if desc.shape.len != 0 {
        let shape = unsafe { std::slice::from_raw_parts(desc.shape.ptr, desc.shape.len) };
        if shape.contains(&0) {
            return Err(invalid_argument("channel shape dimensions must be nonzero"));
        }
    }
    Ok(())
}

/// Validates a driver-owned channel endpoint binding.
pub fn validate_channel_endpoint_binding(
    binding: &PieChannelEndpointBinding,
    desc: &PieChannelDesc,
) -> PieAbiValidationResult {
    if binding.channel_id != desc.channel_id {
        return Err(invalid_argument("channel binding id mismatch"));
    }
    if binding.mirror_base == 0 || binding.word_base == 0 {
        return Err(invalid_argument("channel binding bases must be nonzero"));
    }
    if binding.cell_bytes == 0 || binding.capacity != desc.capacity {
        return Err(invalid_argument("channel binding geometry mismatch"));
    }
    let ring_cells = u64::from(desc.capacity)
        .checked_add(1)
        .ok_or_else(|| invalid_argument("channel binding ring size overflow"))?;
    let expected_mirror = u64::from(binding.cell_bytes)
        .checked_mul(ring_cells)
        .ok_or_else(|| invalid_argument("channel binding mirror size overflow"))?;
    if binding.mirror_bytes < expected_mirror
        || binding.word_bytes < 4 * std::mem::size_of::<u64>() as u64
    {
        return Err(invalid_argument("channel binding storage is undersized"));
    }
    let indices = [
        binding.head_word_index,
        binding.tail_word_index,
        binding.poison_word_index,
        binding.closed_word_index,
    ];
    if indices
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != indices.len()
        || indices
            .iter()
            .any(|&index| u64::from(index) * 8 >= binding.word_bytes)
    {
        return Err(invalid_argument("channel binding word layout is invalid"));
    }
    Ok(())
}

/// Validates a driver-owned instance-binding record returned from bind.
pub fn validate_instance_binding(binding: &PieInstanceBinding) -> PieAbiValidationResult {
    if binding.instance_id == 0 {
        return Err(invalid_argument("instance binding id must be nonzero"));
    }
    if GeometryClass::try_from(binding.geometry_class).is_err() {
        return Err(invalid_argument(
            "instance binding geometry_class is invalid",
        ));
    }
    validate_reserved_zero("instance binding reserved0 must be zero", binding.reserved0)?;
    Ok(())
}

/// Validates an operation completion descriptor.
pub fn validate_completion(
    completion: PieCompletion,
    require_terminal_cell: bool,
) -> PieAbiValidationResult {
    if completion.wait_id == 0 {
        return Err(invalid_argument("completion wait_id must be nonzero"));
    }
    if completion.target_epoch == 0 {
        return Err(invalid_argument("completion target_epoch must be nonzero"));
    }
    if require_terminal_cell {
        validate_terminal_cell_ptr(
            completion.terminal_cell,
            "completion terminal_cell must be non-null and aligned",
        )?;
    } else if !completion.terminal_cell.is_null() {
        return Err(invalid_argument(
            "launch completion terminal_cell must be null",
        ));
    }
    Ok(())
}

/// Validates a launch descriptor.
///
/// # Safety
///
/// This walks foreign `indptr` slices and nested channel-value descriptors, so
/// every non-null pointer/length pair in `desc` must reference readable memory
/// for the declared element count.
pub unsafe fn validate_launch_desc(desc: &PieLaunchDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("launch reserved0 must be zero", desc.reserved0)?;
    validate_u64_slice(desc.instance_ids, "launch instance_ids ptr/len mismatch")?;
    validate_terminal_cell_ptr_slice(
        desc.terminal_cells,
        "launch terminal_cells ptr/len mismatch",
    )?;
    validate_u32_slice(desc.token_ids, "launch token_ids ptr/len mismatch")?;
    validate_u32_slice(desc.position_ids, "launch position_ids ptr/len mismatch")?;
    validate_u32_slice(
        desc.kv_page_indices,
        "launch kv_page_indices ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.kv_last_page_lens,
        "launch kv_last_page_lens ptr/len mismatch",
    )?;
    validate_u8_slice(desc.rs_slot_flags, "launch rs_slot_flags ptr/len mismatch")?;
    validate_u32_slice(desc.rs_slot_ids, "launch rs_slot_ids ptr/len mismatch")?;
    validate_u32_slice(desc.rs_fold_lens, "launch rs_fold_lens ptr/len mismatch")?;
    validate_u32_slice(
        desc.rs_buffer_slot_ids,
        "launch rs_buffer_slot_ids ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.sampling_indices,
        "launch sampling_indices ptr/len mismatch",
    )?;
    validate_u64_slice(desc.context_ids, "launch context_ids ptr/len mismatch")?;
    validate_bool_field(
        "launch single_token_mode must be 0 or 1",
        desc.single_token_mode,
    )?;
    validate_bool_field("launch has_user_mask must be 0 or 1", desc.has_user_mask)?;
    validate_reserved_bytes_zero("launch reserved_flags must be zero", &desc.reserved_flags)?;
    validate_u32_slice(desc.image_grids, "launch image_grids ptr/len mismatch")?;
    validate_u32_slice(
        desc.image_anchor_positions,
        "launch image_anchor_positions ptr/len mismatch",
    )?;
    validate_bytes(desc.image_pixels, "launch image_pixels ptr/len mismatch")?;
    validate_u32_slice(
        desc.image_mrope_positions,
        "launch image_mrope_positions ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.image_patch_positions,
        "launch image_patch_positions ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.image_anchor_rows,
        "launch image_anchor_rows ptr/len mismatch",
    )?;
    validate_bytes(
        desc.audio_features,
        "launch audio_features ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.audio_anchor_rows,
        "launch audio_anchor_rows ptr/len mismatch",
    )?;
    validate_bytes(desc.embed_rows, "launch embed_rows ptr/len mismatch")?;
    validate_u32_slice(desc.embed_indptr, "launch embed_indptr ptr/len mismatch")?;
    validate_u32_slice(desc.embed_shapes, "launch embed_shapes ptr/len mismatch")?;
    validate_u8_slice(desc.embed_dtypes, "launch embed_dtypes ptr/len mismatch")?;
    validate_u32_slice(
        desc.embed_anchor_rows,
        "launch embed_anchor_rows ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.embed_block_indptr,
        "launch embed_block_indptr ptr/len mismatch",
    )?;
    validate_u32_slice(desc.kv_len, "launch kv_len ptr/len mismatch")?;
    validate_u64_slice(desc.kv_len_device, "launch kv_len_device ptr/len mismatch")?;
    validate_u32_slice(
        desc.kv_translation,
        "launch kv_translation ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.kv_translation_indptr,
        "launch kv_translation_indptr ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.ptir_program_row_indptr,
        "launch ptir_program_row_indptr ptr/len mismatch",
    )?;
    validate_u64_slice(
        desc.ptir_kv_write_lower_bounds,
        "launch ptir_kv_write_lower_bounds ptr/len mismatch",
    )?;
    validate_u64_slice(
        desc.ptir_kv_write_upper_bounds,
        "launch ptir_kv_write_upper_bounds ptr/len mismatch",
    )?;
    validate_u64_slice(
        desc.logical_fire_ids,
        "launch logical_fire_ids ptr/len mismatch",
    )?;
    validate_u64_slice(
        desc.channel_expected_head,
        "launch channel_expected_head ptr/len mismatch",
    )?;
    validate_u64_slice(
        desc.channel_expected_tail,
        "launch channel_expected_tail ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.channel_ticket_indptr,
        "launch channel_ticket_indptr ptr/len mismatch",
    )?;

    let request_count = desc.instance_ids.len;
    let wire_row_count = desc.qo_indptr.len.saturating_sub(1);
    if desc.kv_translation.len != 0 && desc.kv_translation_indptr.len == 0 {
        return Err(invalid_argument(
            "launch kv_translation_indptr is required when translation values are present",
        ));
    }
    if desc.channel_expected_head.len != 0 && desc.channel_ticket_indptr.len == 0 {
        return Err(invalid_argument(
            "launch channel_ticket_indptr is required when ticket values are present",
        ));
    }
    validate_row_count_u32(
        desc.terminal_cells.len,
        "launch terminal_cells length must match batch size",
        request_count,
        false,
    )?;
    if request_count != 0 {
        let instance_ids =
            unsafe { std::slice::from_raw_parts(desc.instance_ids.ptr, desc.instance_ids.len) };
        let terminal_cells =
            unsafe { std::slice::from_raw_parts(desc.terminal_cells.ptr, desc.terminal_cells.len) };
        for index in 0..request_count {
            if instance_ids[..index].contains(&instance_ids[index]) {
                return Err(invalid_argument(
                    "launch instance_ids must not contain duplicates",
                ));
            }
            if terminal_cells[..index].contains(&terminal_cells[index]) {
                return Err(invalid_argument(
                    "launch terminal_cells must point to distinct cells",
                ));
            }
        }
    }
    if desc.position_ids.len != desc.token_ids.len {
        return Err(invalid_argument(
            "launch position_ids length must match token_ids length",
        ));
    }
    unsafe {
        validate_csr(
            desc.qo_indptr,
            "launch qo_indptr malformed",
            desc.token_ids.len,
            wire_row_count,
            true,
        )?;
        validate_csr(
            desc.kv_page_indptr,
            "launch kv_page_indptr malformed",
            desc.kv_page_indices.len,
            wire_row_count,
            true,
        )?;
        validate_csr(
            desc.rs_buffer_slot_indptr,
            "launch rs_buffer_slot_indptr malformed",
            desc.rs_buffer_slot_ids.len,
            wire_row_count,
            true,
        )?;
        validate_csr(
            desc.sampling_indptr,
            "launch sampling_indptr malformed",
            desc.sampling_indices.len,
            wire_row_count,
            true,
        )?;
        validate_csr(
            desc.image_indptr,
            "launch image_indptr malformed",
            desc.image_anchor_positions.len,
            wire_row_count,
            true,
        )?;
        validate_csr(
            desc.audio_indptr,
            "launch audio_indptr malformed",
            desc.audio_anchor_rows.len,
            wire_row_count,
            true,
        )?;
        validate_csr(
            desc.embed_block_indptr,
            "launch embed_block_indptr malformed",
            desc.embed_dtypes.len,
            wire_row_count,
            true,
        )?;
        validate_csr(
            desc.embed_indptr,
            "launch embed_indptr malformed",
            desc.embed_rows.len,
            desc.embed_dtypes.len,
            true,
        )?;
        validate_csr(
            desc.kv_translation_indptr,
            "launch kv_translation_indptr malformed",
            desc.kv_translation.len,
            request_count,
            true,
        )?;
        validate_csr(
            desc.channel_ticket_indptr,
            "launch channel_ticket_indptr malformed",
            desc.channel_expected_head.len,
            request_count,
            true,
        )?;
    }
    if desc.embed_shapes.len != desc.embed_dtypes.len.saturating_mul(2)
        || desc.embed_anchor_rows.len != desc.embed_dtypes.len
    {
        return Err(invalid_argument(
            "launch embedding shapes/dtypes/anchors must describe the same blocks",
        ));
    }
    if desc.embed_dtypes.len != 0 {
        let dtypes =
            unsafe { std::slice::from_raw_parts(desc.embed_dtypes.ptr, desc.embed_dtypes.len) };
        if dtypes.iter().any(|dtype| *dtype != 2) {
            return Err(invalid_argument(
                "launch precomputed embeddings currently require bf16 dtype tag 2",
            ));
        }
    }
    if desc.channel_expected_head.len != desc.channel_expected_tail.len {
        return Err(invalid_argument(
            "launch channel ticket head/tail vectors must be parallel",
        ));
    }
    if desc.channel_ticket_indptr.len != 0 {
        let indptr = unsafe {
            std::slice::from_raw_parts(
                desc.channel_ticket_indptr.ptr,
                desc.channel_ticket_indptr.len,
            )
        };
        if indptr.last().copied().unwrap_or_default() as usize != desc.channel_expected_head.len {
            return Err(invalid_argument(
                "launch channel_ticket_indptr must cover every ticket",
            ));
        }
    }
    validate_row_count_u32(
        desc.logical_fire_ids.len,
        "launch logical_fire_ids length must match batch size",
        request_count,
        true,
    )?;
    if desc.ptir_program_row_indptr.len != 0 {
        unsafe {
            validate_csr(
                desc.ptir_program_row_indptr,
                "launch ptir_program_row_indptr malformed",
                wire_row_count,
                request_count,
                false,
            )?;
        }
        if desc.ptir_kv_write_lower_bounds.len != desc.ptir_kv_write_upper_bounds.len
            || (desc.ptir_kv_write_lower_bounds.len != 0
                && desc.ptir_kv_write_lower_bounds.len != request_count)
        {
            return Err(invalid_argument(
                "launch PTIR KV write bounds must have one pair per instance",
            ));
        }
        if desc.ptir_kv_write_lower_bounds.len != 0 {
            let lower = unsafe {
                std::slice::from_raw_parts(
                    desc.ptir_kv_write_lower_bounds.ptr,
                    desc.ptir_kv_write_lower_bounds.len,
                )
            };
            let upper = unsafe {
                std::slice::from_raw_parts(
                    desc.ptir_kv_write_upper_bounds.ptr,
                    desc.ptir_kv_write_upper_bounds.len,
                )
            };
            if lower.iter().zip(upper).any(|(lower, upper)| lower > upper) {
                return Err(invalid_argument("launch PTIR KV write bounds are inverted"));
            }
        }
    }

    validate_row_count_u32(
        desc.kv_last_page_lens.len,
        "launch kv_last_page_lens length must match resolved row count",
        wire_row_count,
        true,
    )?;
    if desc.rs_slot_ids.len != desc.rs_slot_flags.len {
        return Err(invalid_argument(
            "launch rs_slot_ids and rs_slot_flags lengths must match",
        ));
    }
    if desc.rs_fold_lens.len != 0 && desc.rs_fold_lens.len != desc.rs_slot_ids.len {
        return Err(invalid_argument(
            "launch rs_fold_lens length must match rs_slot_ids",
        ));
    }
    if desc.qo_indptr.len != 0
        && desc.rs_slot_ids.len != 0
        && desc.rs_slot_ids.len != wire_row_count
    {
        return Err(invalid_argument(
            "launch rs slot vector length must match resolved qo rows",
        ));
    }
    if desc.rs_slot_flags.len != 0 {
        let flags =
            unsafe { std::slice::from_raw_parts(desc.rs_slot_flags.ptr, desc.rs_slot_flags.len) };
        if flags
            .iter()
            .any(|flag| flag & !(PIE_RS_FLAG_RESET | PIE_RS_FLAG_FOLD) != 0)
        {
            return Err(invalid_argument(
                "launch rs_slot_flags contains unknown bits",
            ));
        }
    }
    validate_row_count_u32(
        desc.context_ids.len,
        "launch context_ids length must match resolved row count",
        wire_row_count,
        true,
    )?;
    validate_row_count_u32(
        desc.kv_len.len,
        "launch kv_len length must match resolved row count",
        wire_row_count,
        true,
    )?;
    if desc.kv_len_device.len > 1 {
        return Err(invalid_argument(
            "launch kv_len_device must carry zero or one device pointer",
        ));
    }

    if desc.image_grids.len % 3 != 0 {
        return Err(invalid_argument(
            "launch image_grids length must be divisible by 3",
        ));
    }
    if desc.image_anchor_positions.len != desc.image_grids.len / 3 {
        return Err(invalid_argument(
            "launch image_anchor_positions length must match image count",
        ));
    }
    if desc.image_anchor_rows.len != desc.image_anchor_positions.len {
        return Err(invalid_argument(
            "launch image_anchor_rows length must match image count",
        ));
    }
    unsafe {
        validate_csr(
            desc.image_pixel_indptr,
            "launch image_pixel_indptr malformed",
            desc.image_pixels.len,
            desc.image_anchor_positions.len,
            true,
        )?;
        validate_csr(
            desc.image_mrope_indptr,
            "launch image_mrope_indptr malformed",
            desc.image_mrope_positions.len,
            desc.image_anchor_positions.len,
            true,
        )?;
        validate_csr(
            desc.audio_feature_indptr,
            "launch audio_feature_indptr malformed",
            desc.audio_features.len,
            desc.audio_anchor_rows.len,
            true,
        )?;
    }

    if desc.image_patch_positions.len % 2 != 0 {
        return Err(invalid_argument(
            "launch image_patch_positions length must be divisible by 2",
        ));
    }
    if desc.audio_indptr.len != 0 && desc.audio_anchor_rows.len == 0 && desc.audio_features.len != 0
    {
        return Err(invalid_argument(
            "launch audio anchors must be present when audio features are present",
        ));
    }

    unsafe {
        validate_csr(
            desc.masks.request_indptr,
            "launch masks.request_indptr malformed",
            desc.masks.word_indptr.len.saturating_sub(1),
            wire_row_count,
            true,
        )?;
        let row_count = {
            let request_indptr = as_u32_slice(
                desc.masks.request_indptr,
                "launch masks.request_indptr malformed",
            )?;
            request_indptr.last().copied().unwrap_or(0) as usize
        };
        validate_csr(
            desc.masks.word_indptr,
            "launch masks.word_indptr malformed",
            desc.masks.words.len,
            row_count,
            true,
        )?;
    }

    Ok(())
}

/// # Safety
/// All descriptor pointers must remain readable/writable for the declared lengths.
pub unsafe fn validate_encode_desc(desc: &PieEncodeDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("encode reserved0 must be zero", desc.reserved0)?;
    validate_u32_slice(desc.image_grids, "encode image_grids ptr/len mismatch")?;
    validate_bytes(desc.image_pixels, "encode image_pixels ptr/len mismatch")?;
    validate_u32_slice(
        desc.image_pixel_indptr,
        "encode image_pixel_indptr ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.image_patch_positions,
        "encode image_patch_positions ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.image_anchor_rows,
        "encode image_anchor_rows ptr/len mismatch",
    )?;
    validate_bytes(
        desc.audio_features,
        "encode audio_features ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.audio_feature_indptr,
        "encode audio_feature_indptr ptr/len mismatch",
    )?;
    validate_u32_slice(
        desc.audio_anchor_rows,
        "encode audio_anchor_rows ptr/len mismatch",
    )?;
    validate_mut_bytes(desc.output_rows, "encode output_rows ptr/len mismatch")?;
    validate_u32_mut_slice(
        desc.output_row_indptr,
        "encode output_row_indptr ptr/len mismatch",
    )?;
    let images = desc.image_anchor_rows.len;
    let clips = desc.audio_anchor_rows.len;
    if images + clips == 0
        || desc.output_row_indptr.len != images + clips + 1
        || desc.output_rows.len == 0
        || desc.output_rows.len % std::mem::size_of::<u16>() != 0
    {
        return Err(invalid_argument(
            "encode media descriptor shapes are inconsistent",
        ));
    }
    if images == 0 {
        if desc.image_grids.len != 0
            || desc.image_pixels.len != 0
            || desc.image_pixel_indptr.len != 0
            || desc.image_patch_positions.len != 0
        {
            return Err(invalid_argument(
                "encode image payload requires image anchors",
            ));
        }
    } else {
        if desc.image_grids.len != images.saturating_mul(3)
            || desc.image_pixel_indptr.len != images + 1
            || desc.image_pixels.len == 0
            || desc.image_pixels.len % std::mem::size_of::<f32>() != 0
            || desc.image_patch_positions.len == 0
            || desc.image_patch_positions.len % 2 != 0
        {
            return Err(invalid_argument("encode image metadata is inconsistent"));
        }
        let pixel_indptr = unsafe {
            as_u32_slice(
                desc.image_pixel_indptr,
                "encode image_pixel_indptr ptr/len mismatch",
            )?
        };
        if pixel_indptr.first().copied() != Some(0)
            || pixel_indptr.last().copied() != Some(desc.image_pixels.len as u32)
            || !pixel_indptr.windows(2).all(|window| {
                window[0] <= window[1]
                    && window[0] % std::mem::size_of::<f32>() as u32 == 0
                    && window[1] % std::mem::size_of::<f32>() as u32 == 0
            })
        {
            return Err(invalid_argument(
                "encode image_pixel_indptr must exactly partition aligned pixels",
            ));
        }
    }
    if clips == 0 {
        if desc.audio_features.len != 0 || desc.audio_feature_indptr.len != 0 {
            return Err(invalid_argument(
                "encode audio payload requires audio anchors",
            ));
        }
    } else {
        if desc.audio_feature_indptr.len != clips + 1
            || desc.audio_features.len == 0
            || desc.audio_features.len % std::mem::size_of::<f32>() != 0
        {
            return Err(invalid_argument("encode audio metadata is inconsistent"));
        }
        let audio_indptr = unsafe {
            as_u32_slice(
                desc.audio_feature_indptr,
                "encode audio_feature_indptr ptr/len mismatch",
            )?
        };
        if audio_indptr.first().copied() != Some(0)
            || audio_indptr.last().copied() != Some(desc.audio_features.len as u32)
            || !audio_indptr.windows(2).all(|window| {
                window[0] < window[1]
                    && window[0] % std::mem::size_of::<f32>() as u32 == 0
                    && window[1] % std::mem::size_of::<f32>() as u32 == 0
            })
        {
            return Err(invalid_argument(
                "encode audio_feature_indptr must exactly partition aligned features",
            ));
        }
    }
    Ok(())
}

/// Validates a KV copy descriptor.
pub fn validate_kv_copy_desc(desc: &PieKvCopyDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("kv copy reserved0 must be zero", desc.reserved0)?;
    if !pie_memory_domain_is_valid(desc.src_domain) {
        return Err(invalid_argument("kv copy src_domain is invalid"));
    }
    if !pie_memory_domain_is_valid(desc.dst_domain) {
        return Err(invalid_argument("kv copy dst_domain is invalid"));
    }
    validate_u32_slice(desc.src_page_ids, "kv copy src_page_ids ptr/len mismatch")?;
    validate_u32_slice(desc.dst_page_ids, "kv copy dst_page_ids ptr/len mismatch")?;
    validate_kv_move_cell_slice(desc.cells, "kv copy cells ptr/len mismatch")?;
    if desc.src_page_ids.len != desc.dst_page_ids.len {
        return Err(invalid_argument(
            "kv copy src_page_ids and dst_page_ids lengths must match",
        ));
    }
    Ok(())
}

/// Validates a state-copy descriptor.
pub fn validate_state_copy_desc(desc: &PieStateCopyDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("state copy reserved0 must be zero", desc.reserved0)?;
    validate_state_copy_range_slice(desc.slot_ranges, "state copy slot_ranges ptr/len mismatch")
}

/// Validates a pool-resize descriptor.
pub fn validate_pool_resize_desc(desc: &PiePoolResizeDesc) -> PieAbiValidationResult {
    validate_pie_abi_version(desc.abi_version)?;
    validate_reserved_zero("pool resize reserved0 must be zero", desc.reserved0)?;
    validate_pool_range_slice(desc.map_ranges, "pool resize map_ranges ptr/len mismatch")?;
    validate_pool_range_slice(
        desc.unmap_ranges,
        "pool resize unmap_ranges ptr/len mismatch",
    )
}

/// Validates a driver-owned launch-prepare result.
pub fn validate_launch_prepare_result(result: &PieLaunchPrepareResult) -> PieAbiValidationResult {
    validate_reserved_zero(
        "launch prepare result reserved0 must be zero",
        result.reserved0,
    )?;
    if result.outcome > PIE_LAUNCH_PREPARE_IMPOSSIBLE {
        return Err(invalid_argument("launch prepare result outcome is invalid"));
    }
    if (result.outcome == PIE_LAUNCH_PREPARE_READY) != (result.lease_id != 0) {
        return Err(invalid_argument(
            "launch prepare result lease_id does not match outcome",
        ));
    }
    Ok(())
}

/// Validates non-null out-parameters used by the native driver entrypoints.
pub fn validate_create_out_params(caps: *mut PieDriverCaps) -> PieAbiValidationResult {
    validate_mut_ptr(caps, "driver create caps output pointer must be non-null")
}

unsafe extern "C" {
    pub fn pie_cuda_create(
        desc: *const PieDriverCreateDesc,
        caps: *mut PieDriverCaps,
    ) -> *mut PieDriver;
    pub fn pie_cuda_load_model(
        driver: *mut PieDriver,
        load: *const PieModelLoadDesc,
        caps: *mut PieDriverCaps,
    ) -> i32;
    pub fn pie_cuda_register_program(
        driver: *mut PieDriver,
        program: *const PieProgramDesc,
        program_id: *mut u64,
    ) -> i32;
    pub fn pie_cuda_register_channel(
        driver: *mut PieDriver,
        channel: *const PieChannelDesc,
        binding: *mut PieChannelEndpointBinding,
    ) -> i32;
    pub fn pie_cuda_bind_instance(
        driver: *mut PieDriver,
        instance: *const PieInstanceDesc,
        binding: *mut PieInstanceBinding,
    ) -> i32;
    pub fn pie_cuda_launch(
        driver: *mut PieDriver,
        launch: *const PieLaunchDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_cuda_prepare_launch(
        driver: *mut PieDriver,
        launch: *const PieLaunchDesc,
        result: *mut PieLaunchPrepareResult,
    ) -> i32;
    pub fn pie_cuda_launch_prepared(
        driver: *mut PieDriver,
        launch: *const PieLaunchDesc,
        lease_id: u64,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_cuda_release_launch(driver: *mut PieDriver, lease_id: u64) -> i32;
    pub fn pie_cuda_encode(
        driver: *mut PieDriver,
        encode: *const PieEncodeDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_cuda_copy_kv(
        driver: *mut PieDriver,
        copy: *const PieKvCopyDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_cuda_copy_state(
        driver: *mut PieDriver,
        copy: *const PieStateCopyDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_cuda_resize_pool(
        driver: *mut PieDriver,
        resize: *const PiePoolResizeDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_cuda_close_instance(driver: *mut PieDriver, instance_id: u64) -> i32;
    pub fn pie_cuda_close_channel(driver: *mut PieDriver, channel_id: u64) -> i32;
    pub fn pie_cuda_destroy(driver: *mut PieDriver);
}

unsafe extern "C" {
    pub fn pie_metal_create(
        desc: *const PieDriverCreateDesc,
        caps: *mut PieDriverCaps,
    ) -> *mut PieDriver;
    pub fn pie_metal_load_model(
        driver: *mut PieDriver,
        load: *const PieModelLoadDesc,
        caps: *mut PieDriverCaps,
    ) -> i32;
    pub fn pie_metal_register_program(
        driver: *mut PieDriver,
        program: *const PieProgramDesc,
        program_id: *mut u64,
    ) -> i32;
    pub fn pie_metal_register_channel(
        driver: *mut PieDriver,
        channel: *const PieChannelDesc,
        binding: *mut PieChannelEndpointBinding,
    ) -> i32;
    pub fn pie_metal_bind_instance(
        driver: *mut PieDriver,
        instance: *const PieInstanceDesc,
        binding: *mut PieInstanceBinding,
    ) -> i32;
    pub fn pie_metal_launch(
        driver: *mut PieDriver,
        launch: *const PieLaunchDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_metal_prepare_launch(
        driver: *mut PieDriver,
        launch: *const PieLaunchDesc,
        result: *mut PieLaunchPrepareResult,
    ) -> i32;
    pub fn pie_metal_launch_prepared(
        driver: *mut PieDriver,
        launch: *const PieLaunchDesc,
        lease_id: u64,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_metal_release_launch(driver: *mut PieDriver, lease_id: u64) -> i32;
    pub fn pie_metal_encode(
        driver: *mut PieDriver,
        encode: *const PieEncodeDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_metal_copy_kv(
        driver: *mut PieDriver,
        copy: *const PieKvCopyDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_metal_copy_state(
        driver: *mut PieDriver,
        copy: *const PieStateCopyDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_metal_resize_pool(
        driver: *mut PieDriver,
        resize: *const PiePoolResizeDesc,
        completion: PieCompletion,
    ) -> i32;
    pub fn pie_metal_close_instance(driver: *mut PieDriver, instance_id: u64) -> i32;
    pub fn pie_metal_close_channel(driver: *mut PieDriver, channel_id: u64) -> i32;
    pub fn pie_metal_destroy(driver: *mut PieDriver);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::NonNull;

    fn committed_header() -> String {
        std::fs::read_to_string(format!(
            "{}/include/pie_driver_abi.h",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("read committed pie_driver_abi.h")
    }

    fn cbindgen_config() -> String {
        std::fs::read_to_string(format!(
            "{}/cbindgen/cbindgen.toml",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("read cbindgen.toml")
    }

    fn cbindgen_main() -> String {
        std::fs::read_to_string(format!(
            "{}/cbindgen/src/main.rs",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("read pie-driver-abi-cbindgen main.rs")
    }

    fn header_block<'a>(header: &'a str, name: &str) -> &'a str {
        let start = format!("typedef struct {name} {{");
        let end = format!("}} {name};");
        let from = header
            .find(&start)
            .unwrap_or_else(|| panic!("missing header block start for {name}"));
        let tail = &header[from..];
        let to = tail
            .find(&end)
            .unwrap_or_else(|| panic!("missing header block end for {name}"));
        &tail[..to + end.len()]
    }

    unsafe extern "C" fn dummy_notify(_: *mut std::ffi::c_void, _: u64, _: u64) {}

    fn valid_runtime_callbacks() -> PieRuntimeCallbacks {
        PieRuntimeCallbacks {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            ctx: std::ptr::null_mut(),
            notify: Some(dummy_notify),
        }
    }

    #[test]
    fn descriptor_defaults_use_current_abi_version() {
        assert_eq!(
            PieRuntimeCallbacks::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(
            PieDriverCreateDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(
            PieModelLoadDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(
            PieProgramDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(
            PieInstanceDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(PieLaunchDesc::default().abi_version, PIE_DRIVER_ABI_VERSION);
        assert_eq!(PieEncodeDesc::default().abi_version, PIE_DRIVER_ABI_VERSION);
        assert_eq!(PieKvCopyDesc::default().abi_version, PIE_DRIVER_ABI_VERSION);
        assert_eq!(
            PieStateCopyDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(
            PiePoolResizeDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(PieRuntimeCallbacks::default().reserved0, 0);
        assert_eq!(PieDriverCreateDesc::default().reserved0, 0);
        assert_eq!(
            PieModelLoadDesc::default().component,
            PIE_MODEL_COMPONENT_FULL
        );
        assert_eq!(PieProgramDesc::default().reserved0, 0);
        assert_eq!(PieInstanceDesc::default().reserved0, 0);
        assert_eq!(
            PieInstanceDesc::default().geometry_class,
            PIE_GEOMETRY_CLASS_HOST
        );
        assert_eq!(PieInstanceDesc::default().reserved1, 0);
        assert_eq!(PieLaunchDesc::default().reserved0, 0);
        assert_eq!(PieEncodeDesc::default().reserved0, 0);
        assert_eq!(PieLaunchDesc::default().reserved_flags, [0; 2]);
        assert_eq!(PieKvCopyDesc::default().reserved0, 0);
        assert_eq!(PieStateCopyDesc::default().reserved0, 0);
        assert_eq!(PiePoolResizeDesc::default().reserved0, 0);
    }

    #[test]
    fn batch_launch_defaults_are_empty_borrowed_views() {
        let launch = PieLaunchDesc::default();
        assert!(launch.instance_ids.ptr.is_null());
        assert_eq!(launch.instance_ids.len, 0);
        assert!(launch.image_pixels.ptr.is_null());
        assert_eq!(launch.image_pixels.len, 0);
        assert_eq!(launch.single_token_mode, 0);
        assert_eq!(launch.has_user_mask, 0);
    }

    #[test]
    fn instance_binding_defaults_to_host_geometry() {
        let instance = PieInstanceDesc::default();
        assert!(instance.channel_ids.ptr.is_null());
        assert_eq!(instance.channel_ids.len, 0);
        assert!(instance.seed_values.ptr.is_null());
        assert_eq!(instance.seed_values.len, 0);

        assert_eq!(PieInstanceBinding::default().instance_id, 0);
        assert_eq!(
            PieInstanceBinding::default().geometry_class,
            PIE_GEOMETRY_CLASS_HOST
        );
        assert_eq!(PieInstanceBinding::default().reserved0, 0);
    }

    #[test]
    fn instance_geometry_class_is_validated() {
        let mut instance = PieInstanceDesc::default();
        instance.geometry_class = u32::MAX;
        assert!(unsafe { validate_instance_desc(&instance) }.is_err());

        let binding = PieInstanceBinding {
            instance_id: 1,
            geometry_class: u32::MAX,
            reserved0: 0,
        };
        assert!(validate_instance_binding(&binding).is_err());
    }

    #[test]
    fn completion_layout_is_stable() {
        assert_eq!(std::mem::size_of::<PieCompletion>(), 24);
        assert_eq!(std::mem::align_of::<PieCompletion>(), 8);
    }

    #[test]
    fn encode_layout_and_validation_are_stable() {
        assert_eq!(std::mem::size_of::<PieEncodeDesc>(), 168);
        assert_eq!(std::mem::align_of::<PieEncodeDesc>(), 8);

        let grids = [1u32, 1, 1];
        let pixels = [0u8; 4];
        let pixel_indptr = [0u32, 4];
        let patch_positions = [0u32, 0];
        let anchors = [0u32];
        let mut output = [0u8; 2];
        let mut output_indptr = [0u32; 2];
        let desc = PieEncodeDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            image_grids: PieU32Slice {
                ptr: grids.as_ptr(),
                len: grids.len(),
            },
            image_pixels: PieBytes {
                ptr: pixels.as_ptr(),
                len: pixels.len(),
            },
            image_pixel_indptr: PieU32Slice {
                ptr: pixel_indptr.as_ptr(),
                len: pixel_indptr.len(),
            },
            image_patch_positions: PieU32Slice {
                ptr: patch_positions.as_ptr(),
                len: patch_positions.len(),
            },
            image_anchor_rows: PieU32Slice {
                ptr: anchors.as_ptr(),
                len: anchors.len(),
            },
            audio_features: PieBytes::default(),
            audio_feature_indptr: PieU32Slice::default(),
            audio_anchor_rows: PieU32Slice::default(),
            output_rows: PieMutBytes {
                ptr: output.as_mut_ptr(),
                len: output.len(),
            },
            output_row_indptr: PieU32MutSlice {
                ptr: output_indptr.as_mut_ptr(),
                len: output_indptr.len(),
            },
        };
        unsafe { validate_encode_desc(&desc) }.unwrap();

        let bad_indptr = [0u32, 3];
        let malformed = PieEncodeDesc {
            image_pixel_indptr: PieU32Slice {
                ptr: bad_indptr.as_ptr(),
                len: 2,
            },
            ..desc
        };
        assert!(unsafe { validate_encode_desc(&malformed) }.is_err());
    }

    #[test]
    fn completion_validator_enforces_terminal_cell_mode() {
        let mut cell = PieTerminalCell {
            outcome: PIE_TERMINAL_OUTCOME_PENDING,
            reserved0: 0,
        };
        let control = PieCompletion {
            wait_id: 1,
            target_epoch: 1,
            terminal_cell: &mut cell,
        };
        validate_completion(control, true).unwrap();
        assert!(validate_completion(control, false).is_err());

        let launch = PieCompletion {
            terminal_cell: std::ptr::null_mut(),
            ..control
        };
        validate_completion(launch, false).unwrap();
        assert!(validate_completion(launch, true).is_err());
    }

    #[test]
    fn launch_validator_rejects_duplicate_members_and_terminal_cells() {
        let instance_ids = [7u64, 7];
        let mut terminal_cells = [
            PieTerminalCell {
                outcome: PIE_TERMINAL_OUTCOME_PENDING,
                reserved0: 0,
            },
            PieTerminalCell {
                outcome: PIE_TERMINAL_OUTCOME_PENDING,
                reserved0: 0,
            },
        ];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: instance_ids.as_ptr(),
                len: instance_ids.len(),
            },
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: terminal_ptrs.as_ptr(),
                len: terminal_ptrs.len(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert!(err.message().contains("instance_ids"));

        let distinct_instance_ids = [7u64, 8];
        let duplicate_terminal_ptrs = [terminal_ptrs[0], terminal_ptrs[0]];
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: distinct_instance_ids.as_ptr(),
                len: distinct_instance_ids.len(),
            },
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: duplicate_terminal_ptrs.as_ptr(),
                len: duplicate_terminal_ptrs.len(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert!(err.message().contains("distinct"));
    }

    #[test]
    fn validators_reject_null_pointer_with_nonzero_len_for_every_slice_type() {
        assert_eq!(
            validate_bytes(
                PieBytes {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "bytes",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            validate_u8_slice(
                PieU8Slice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "u8",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            validate_u32_slice(
                PieU32Slice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "u32",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            validate_u64_slice(
                PieU64Slice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "u64",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            validate_channel_value_desc_slice(
                PieChannelValueDescSlice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "channel values",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            validate_kv_move_cell_slice(
                PieKvMoveCellSlice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "cells",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            validate_state_copy_range_slice(
                PieStateCopyRangeSlice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "ranges",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            validate_pool_range_slice(
                PiePoolRangeSlice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "pool ranges",
            )
            .unwrap_err()
            .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn validators_reject_slice_byte_overflow() {
        let u64_err = validate_u64_slice(
            PieU64Slice {
                ptr: NonNull::<u64>::dangling().as_ptr(),
                len: usize::MAX,
            },
            "u64 overflow",
        )
        .unwrap_err();
        assert_eq!(u64_err.status(), PIE_STATUS_INVALID_ARGUMENT);

        let channel_value_err = validate_channel_value_desc_slice(
            PieChannelValueDescSlice {
                ptr: NonNull::<PieChannelValueDesc>::dangling().as_ptr(),
                len: usize::MAX,
            },
            "channel value overflow",
        )
        .unwrap_err();
        assert_eq!(channel_value_err.status(), PIE_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn create_validator_rejects_wrong_abi_and_missing_notify() {
        let config = [1u8, 2, 3];
        let mut desc = PieDriverCreateDesc {
            abi_version: PIE_DRIVER_ABI_VERSION + 1,
            reserved0: 0,
            config_bytes: PieBytes {
                ptr: config.as_ptr(),
                len: config.len(),
            },
            runtime: valid_runtime_callbacks(),
        };
        assert_eq!(
            validate_driver_create_desc(&desc).unwrap_err().status(),
            PIE_STATUS_BAD_ABI_VERSION
        );

        desc.abi_version = PIE_DRIVER_ABI_VERSION;
        desc.runtime.notify = None;
        let err = validate_driver_create_desc(&desc).unwrap_err();
        assert_eq!(err.status(), PIE_STATUS_INVALID_ARGUMENT);
        assert!(
            err.message()
                .contains("runtime notify callback must be non-null")
        );
    }

    #[test]
    fn model_load_validator_requires_program_and_snapshot() {
        let mut desc = PieModelLoadDesc::default();
        assert!(validate_model_load_desc(&desc).is_err());
        let program = [1u8];
        let snapshot = b"/tmp/model";
        desc.load_plan_bytes = PieBytes {
            ptr: program.as_ptr(),
            len: program.len(),
        };
        desc.snapshot_dir = PieBytes {
            ptr: snapshot.as_ptr(),
            len: snapshot.len(),
        };
        desc.compiler_version = 1;
        validate_model_load_desc(&desc).unwrap();
        desc.component = PIE_MODEL_COMPONENT_ENCODE + 1;
        assert!(validate_model_load_desc(&desc).is_err());
    }

    #[test]
    fn kv_copy_validator_rejects_invalid_memory_domain() {
        let desc = PieKvCopyDesc {
            src_domain: 99,
            ..PieKvCopyDesc::default()
        };
        let err = validate_kv_copy_desc(&desc).unwrap_err();
        assert_eq!(err.status(), PIE_STATUS_INVALID_ARGUMENT);
        assert!(err.message().contains("src_domain"));
    }

    #[test]
    fn launch_validator_rejects_malformed_qo_indptr() {
        let instance_ids = [11u64, 12];
        let mut terminal_cells = [PieTerminalCell::default(), PieTerminalCell::default()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        let tokens = [1u32, 2, 3];
        let positions = [4u32, 5, 6];
        let qo_indptr = [0u32, 4];
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: instance_ids.as_ptr(),
                len: instance_ids.len(),
            },
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: terminal_ptrs.as_ptr(),
                len: terminal_ptrs.len(),
            },
            token_ids: PieU32Slice {
                ptr: tokens.as_ptr(),
                len: tokens.len(),
            },
            position_ids: PieU32Slice {
                ptr: positions.as_ptr(),
                len: positions.len(),
            },
            qo_indptr: PieU32Slice {
                ptr: qo_indptr.as_ptr(),
                len: qo_indptr.len(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert_eq!(err.status(), PIE_STATUS_INVALID_ARGUMENT);
        assert!(err.message().contains("qo_indptr"));
    }

    #[test]
    fn launch_validator_rejects_malformed_mask_relationships() {
        let instance_ids = [41u64];
        let mut terminal_cells = [PieTerminalCell::default()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        let tokens = [7u32];
        let positions = [9u32];
        let qo_indptr = [0u32, 1];
        let mask_request_indptr = [1u32, 1];
        let mask_word_indptr = [0u32];
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: instance_ids.as_ptr(),
                len: instance_ids.len(),
            },
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: terminal_ptrs.as_ptr(),
                len: terminal_ptrs.len(),
            },
            token_ids: PieU32Slice {
                ptr: tokens.as_ptr(),
                len: tokens.len(),
            },
            position_ids: PieU32Slice {
                ptr: positions.as_ptr(),
                len: positions.len(),
            },
            qo_indptr: PieU32Slice {
                ptr: qo_indptr.as_ptr(),
                len: qo_indptr.len(),
            },
            masks: PieMaskWordsDesc {
                request_indptr: PieU32Slice {
                    ptr: mask_request_indptr.as_ptr(),
                    len: mask_request_indptr.len(),
                },
                word_indptr: PieU32Slice {
                    ptr: mask_word_indptr.as_ptr(),
                    len: mask_word_indptr.len(),
                },
                words: PieU32Slice::default(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert_eq!(err.status(), PIE_STATUS_INVALID_ARGUMENT);
        assert!(err.message().contains("masks.request_indptr"));
    }

    #[test]
    fn launch_validator_rejects_image_count_mismatches() {
        let instance_ids = [51u64];
        let mut terminal_cells = [PieTerminalCell::default()];
        let terminal_ptrs = terminal_cells
            .each_mut()
            .map(|cell| cell as *mut PieTerminalCell);
        let tokens = [1u32];
        let positions = [2u32];
        let qo_indptr = [0u32, 1];
        let image_indptr = [0u32, 1];
        let image_grids = [1u32, 2, 3];
        let image_anchor_positions = [7u32];
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: instance_ids.as_ptr(),
                len: instance_ids.len(),
            },
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: terminal_ptrs.as_ptr(),
                len: terminal_ptrs.len(),
            },
            token_ids: PieU32Slice {
                ptr: tokens.as_ptr(),
                len: tokens.len(),
            },
            position_ids: PieU32Slice {
                ptr: positions.as_ptr(),
                len: positions.len(),
            },
            qo_indptr: PieU32Slice {
                ptr: qo_indptr.as_ptr(),
                len: qo_indptr.len(),
            },
            image_indptr: PieU32Slice {
                ptr: image_indptr.as_ptr(),
                len: image_indptr.len(),
            },
            image_grids: PieU32Slice {
                ptr: image_grids.as_ptr(),
                len: image_grids.len(),
            },
            image_anchor_positions: PieU32Slice {
                ptr: image_anchor_positions.as_ptr(),
                len: image_anchor_positions.len(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert_eq!(err.status(), PIE_STATUS_INVALID_ARGUMENT);
        assert!(err.message().contains("image_anchor_rows"));
    }

    #[test]
    fn launch_validator_rejects_invalid_boolean_flags() {
        let launch = PieLaunchDesc {
            single_token_mode: 2,
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert_eq!(err.status(), PIE_STATUS_INVALID_ARGUMENT);
        assert!(err.message().contains("single_token_mode"));
    }

    #[test]
    fn launch_validator_accepts_resolved_request_rs_vectors() {
        let instance_ids = [71u64];
        let mut terminal = PieTerminalCell::default();
        let terminal_cells = [&mut terminal as *mut PieTerminalCell];
        let token_ids = [10u32, 11];
        let position_ids = [0u32, 0];
        let qo_indptr = [0u32, 1, 2];
        let program_row_indptr = [0u32, 2];
        let rs_slot_ids = [3u32, 4];
        let rs_slot_flags = [PIE_RS_FLAG_RESET, 0];
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: instance_ids.as_ptr(),
                len: instance_ids.len(),
            },
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: terminal_cells.as_ptr(),
                len: terminal_cells.len(),
            },
            token_ids: PieU32Slice {
                ptr: token_ids.as_ptr(),
                len: token_ids.len(),
            },
            position_ids: PieU32Slice {
                ptr: position_ids.as_ptr(),
                len: position_ids.len(),
            },
            qo_indptr: PieU32Slice {
                ptr: qo_indptr.as_ptr(),
                len: qo_indptr.len(),
            },
            ptir_program_row_indptr: PieU32Slice {
                ptr: program_row_indptr.as_ptr(),
                len: program_row_indptr.len(),
            },
            rs_slot_ids: PieU32Slice {
                ptr: rs_slot_ids.as_ptr(),
                len: rs_slot_ids.len(),
            },
            rs_slot_flags: PieU8Slice {
                ptr: rs_slot_flags.as_ptr(),
                len: rs_slot_flags.len(),
            },
            ..PieLaunchDesc::default()
        };
        unsafe { validate_launch_desc(&launch) }.unwrap();

        let mismatched_flags = PieLaunchDesc {
            rs_slot_flags: PieU8Slice {
                ptr: rs_slot_flags.as_ptr(),
                len: 1,
            },
            ..launch
        };
        assert!(
            unsafe { validate_launch_desc(&mismatched_flags) }
                .unwrap_err()
                .message()
                .contains("lengths must match")
        );

        let fold_lens = [0u32, 0, 0];
        let mismatched_fold = PieLaunchDesc {
            rs_fold_lens: PieU32Slice {
                ptr: fold_lens.as_ptr(),
                len: fold_lens.len(),
            },
            ..launch
        };
        assert!(
            unsafe { validate_launch_desc(&mismatched_fold) }
                .unwrap_err()
                .message()
                .contains("rs_fold_lens")
        );

        let one_slot = [3u32];
        let one_flag = [0u8];
        let wrong_resolved_count = PieLaunchDesc {
            rs_slot_ids: PieU32Slice {
                ptr: one_slot.as_ptr(),
                len: one_slot.len(),
            },
            rs_slot_flags: PieU8Slice {
                ptr: one_flag.as_ptr(),
                len: one_flag.len(),
            },
            ..launch
        };
        assert!(
            unsafe { validate_launch_desc(&wrong_resolved_count) }
                .unwrap_err()
                .message()
                .contains("resolved qo rows")
        );
    }

    #[test]
    fn launch_validator_rejects_malformed_ticket_and_translation_csr() {
        let translation = [7u32];
        let launch = PieLaunchDesc {
            kv_translation: PieU32Slice {
                ptr: translation.as_ptr(),
                len: translation.len(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert!(err.message().contains("kv_translation_indptr"));

        let heads = [0u64];
        let ticket_indptr = [0u32, 1];
        let launch = PieLaunchDesc {
            channel_expected_head: PieU64Slice {
                ptr: heads.as_ptr(),
                len: heads.len(),
            },
            channel_ticket_indptr: PieU32Slice {
                ptr: ticket_indptr.as_ptr(),
                len: ticket_indptr.len(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert!(
            err.message().contains("channel_ticket_indptr") || err.message().contains("head/tail")
        );

        let instance_ids = [1u64];
        let mut terminal = PieTerminalCell::default();
        let terminal_cells = [&mut terminal as *mut PieTerminalCell];
        let heads = [0u64, 1];
        let tails = [0u64, 1];
        let undercovered = [0u32, 1];
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: instance_ids.as_ptr(),
                len: instance_ids.len(),
            },
            terminal_cells: PieTerminalCellPtrSlice {
                ptr: terminal_cells.as_ptr(),
                len: terminal_cells.len(),
            },
            channel_expected_head: PieU64Slice {
                ptr: heads.as_ptr(),
                len: heads.len(),
            },
            channel_expected_tail: PieU64Slice {
                ptr: tails.as_ptr(),
                len: tails.len(),
            },
            channel_ticket_indptr: PieU32Slice {
                ptr: undercovered.as_ptr(),
                len: undercovered.len(),
            },
            ..PieLaunchDesc::default()
        };
        let err = unsafe { validate_launch_desc(&launch) }.unwrap_err();
        assert!(err.message().contains("cover every ticket"));
    }

    #[test]
    fn cbindgen_config_stays_pinned_and_excludes_transfer_types() {
        let config = cbindgen_config();
        let main_rs = cbindgen_main();
        assert!(!config.contains("parse.expand"));
        assert!(!config.contains("RUSTC_BOOTSTRAP"));
        assert!(!main_rs.contains("RUSTC_BOOTSTRAP"));

        for excluded in [
            "\"KvDtype\"",
            "\"KvLayoutKind\"",
            "\"KvLayout\"",
            "\"MemoryDomain\"",
            "\"KvRegion\"",
            "\"KvHandle\"",
        ] {
            assert!(
                config.contains(excluded),
                "missing cbindgen exclude {excluded}"
            );
        }
    }

    #[test]
    fn generated_header_matches_refined_surface() {
        let header = committed_header();
        let runtime = header_block(&header, "PieRuntimeCallbacks");
        let create = header_block(&header, "PieDriverCreateDesc");
        let load = header_block(&header, "PieModelLoadDesc");
        let program = header_block(&header, "PieProgramDesc");
        let instance = header_block(&header, "PieInstanceDesc");
        let binding = header_block(&header, "PieInstanceBinding");
        let terminal_cell = header_block(&header, "PieTerminalCell");
        let terminal_cell_ptr_slice = header_block(&header, "PieTerminalCellPtrSlice");
        let launch = header_block(&header, "PieLaunchDesc");
        let encode = header_block(&header, "PieEncodeDesc");
        let completion = header_block(&header, "PieCompletion");
        let kv_copy = header_block(&header, "PieKvCopyDesc");
        let state_copy = header_block(&header, "PieStateCopyDesc");
        let pool_resize = header_block(&header, "PiePoolResizeDesc");

        assert!(header.contains("typedef uint32_t PieTerminalOutcome;"));
        assert!(terminal_cell.contains("PieTerminalOutcome outcome;"));
        assert!(terminal_cell.contains("uint32_t reserved0;"));
        assert!(terminal_cell_ptr_slice.contains("struct PieTerminalCell *const *ptr;"));
        assert!(launch.contains("struct PieU64Slice instance_ids;"));
        assert!(launch.contains("struct PieTerminalCellPtrSlice terminal_cells;"));
        assert!(
            !launch.contains("host_put"),
            "ABI v2: launch descriptors carry no channel values"
        );
        assert!(completion.contains("struct PieTerminalCell *terminal_cell;"));
        assert!(launch.contains("uint32_t reserved0;"));
        assert!(launch.contains("uint8_t reserved_flags[2];"));
        assert!(launch.contains("uint32_t required_kv_pages;"));
        assert!(runtime.contains("uint32_t reserved0;"));
        assert!(runtime.contains("PieRuntimeNotifyFn notify;"));
        assert!(create.contains("uint32_t reserved0;"));
        assert!(load.contains("struct PieBytes load_plan_bytes;"));
        assert!(load.contains("struct PieBytes snapshot_dir;"));
        assert!(load.contains("uint64_t compiler_version;"));
        assert!(load.contains("uint32_t component;"));
        assert!(program.contains("uint64_t program_hash;"));
        assert!(program.contains("uint32_t reserved0;"));
        assert!(instance.contains("struct PieU64Slice channel_ids;"));
        assert!(launch.contains("struct PieBytes embed_rows;"));
        assert!(launch.contains("struct PieU32Slice embed_indptr;"));
        assert!(encode.contains("struct PieMutBytes output_rows;"));
        assert!(encode.contains("struct PieU32MutSlice output_row_indptr;"));
        assert!(encode.contains("struct PieBytes audio_features;"));
        assert!(!program.contains("channel_ids"));
        assert!(binding.contains("uint64_t instance_id;"));
        assert!(!header.contains("typedef struct PieChannelBinding"));
        assert!(!header.contains("PieChannelWait"));
        assert!(kv_copy.contains("uint32_t reserved0;"));
        assert!(kv_copy.contains("struct PieU32Slice src_page_ids;"));
        assert!(kv_copy.contains("struct PieU32Slice dst_page_ids;"));
        assert!(state_copy.contains("uint32_t reserved0;"));
        assert!(state_copy.contains("struct PieStateCopyRangeSlice slot_ranges;"));
        assert!(pool_resize.contains("uint32_t reserved0;"));
        assert!(!header.contains("struct_size"));

        for needle in [
            "PIE_SAMPLER_",
            "PIE_SAMPLING_BINDING_",
            "logit_masks",
            "sampler_",
            "sampling_program_",
            "sampling_input_",
            "sampling_late_",
            "sampling_binding_",
            "adapter_bindings",
            "spec_token_ids",
            "spec_position_ids",
            "spec_indptr",
            "output_spec_flags",
            "PieF32Slice",
            "PieAdapterBinding",
            "KvDtype",
            "KvLayoutKind",
            "KvLayout",
            "KvRegion",
            "KvHandle",
        ] {
            assert!(
                !header.contains(needle),
                "generated header unexpectedly contains excluded name {needle}"
            );
        }
    }
}
