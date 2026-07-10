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

/// Current direct local ABI version.
pub const PIE_DRIVER_ABI_VERSION: u32 = 1;

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

/// Reset the recurrent-state slot before executing the request.
pub const PIE_RS_FLAG_RESET: u8 = 1;
/// Fold buffered recurrent-state data into the slot after the pass.
pub const PIE_RS_FLAG_FOLD: u8 = 2;

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

/// Borrowed immutable `u64` slice.
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieU64Slice {
    pub ptr: *const u64,
    pub len: usize,
}

/// Per-channel runtime wait ids registered at bind time.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieChannelWait {
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
}

/// Borrowed immutable slice of [`PieChannelWait`].
///
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieChannelWaitSlice {
    pub ptr: *const PieChannelWait,
    pub len: usize,
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

/// Stable bound layout for one host-visible channel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieChannelBinding {
    pub channel_id: u64,
    pub cell_bytes: u32,
    pub capacity: u32,
    pub mirror_offset: u64,
    pub head_word_index: u32,
    pub tail_word_index: u32,
    pub poison_word_index: u32,
    pub reserved: u32,
}

/// Slice of driver-owned [`PieChannelBinding`] records.
///
/// The pointer remains valid until `close_instance` for the bound instance.
/// `ptr` may be null only when `len == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieChannelBindingSlice {
    pub ptr: *const PieChannelBinding,
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    /// Must be zero in ABI v1.
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
}

/// Driver creation descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PieDriverCreateDesc {
    pub abi_version: u32,
    /// Must be zero in ABI v1.
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

/// Cold JSON capability payload returned from `*_create`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieDriverCaps {
    pub json_bytes: *const u8,
    pub json_len: usize,
}

/// Static program registration descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieProgramDesc {
    pub abi_version: u32,
    /// Must be zero in ABI v1.
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
    /// Must be zero in ABI v1.
    pub reserved0: u32,
    pub program_id: u64,
    pub requested_instance_id: u64,
    pub pacing_wait_id: u64,
    pub channel_waits: PieChannelWaitSlice,
    pub channel_ids: PieU64Slice,
    pub seed_values: PieChannelValueDescSlice,
}

impl Default for PieInstanceDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            program_id: 0,
            requested_instance_id: 0,
            pacing_wait_id: 0,
            channel_waits: PieChannelWaitSlice::default(),
            channel_ids: PieU64Slice::default(),
            seed_values: PieChannelValueDescSlice::default(),
        }
    }
}

/// Stable addresses and counts returned from `*_bind_instance`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PieInstanceBinding {
    pub instance_id: u64,
    pub frame_base: u64,
    pub mirror_base: u64,
    pub word_base: u64,
    pub channel_count: u32,
    pub word_count: u32,
    pub frame_bytes: u64,
    pub mirror_bytes: u64,
    pub word_bytes: u64,
    /// Driver-owned channel layout table, valid until `close_instance`.
    pub channels: PieChannelBindingSlice,
}

/// One batched launch descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieLaunchDesc {
    pub abi_version: u32,
    /// Must be zero in ABI v1.
    pub reserved0: u32,
    /// Bound instance ids, one per fire/program in scheduler order.
    pub instance_ids: PieU64Slice,
    pub token_ids: PieU32Slice,
    pub position_ids: PieU32Slice,
    pub kv_page_indices: PieU32Slice,
    pub kv_page_indptr: PieU32Slice,
    pub kv_last_page_lens: PieU32Slice,
    pub qo_indptr: PieU32Slice,
    pub rs_slot_ids: PieU32Slice,
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
    /// Must be zero in ABI v1.
    pub reserved_flags: [u8; 6],
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
    /// Flattened PTIR host-put values for all launched instances.
    pub ptir_host_put_values: PieChannelValueDescSlice,
    /// CSR partition of `ptir_host_put_values`, one segment per `instance_ids` entry.
    pub host_put_indptr: PieU32Slice,
    pub kv_len: PieU32Slice,
    pub kv_len_device: PieU64Slice,
}

impl Default for PieLaunchDesc {
    fn default() -> Self {
        Self {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            instance_ids: PieU64Slice::default(),
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
            reserved_flags: [0; 6],
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
            ptir_host_put_values: PieChannelValueDescSlice::default(),
            host_put_indptr: PieU32Slice::default(),
            kv_len: PieU32Slice::default(),
            kv_len_device: PieU64Slice::default(),
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
    /// Must be zero in ABI v1.
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
    /// Must be zero in ABI v1.
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
    /// Must be zero in ABI v1.
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

fn validate_u8_slice(slice: PieU8Slice, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_u32_slice(slice: PieU32Slice, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_u64_slice(slice: PieU64Slice, name: &'static str) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_channel_wait_slice(
    slice: PieChannelWaitSlice,
    name: &'static str,
) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_channel_value_desc_slice(
    slice: PieChannelValueDescSlice,
    name: &'static str,
) -> PieAbiValidationResult {
    validate_slice_ptr(slice.ptr, slice.len, name)
}

fn validate_channel_binding_slice(
    slice: PieChannelBindingSlice,
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
    validate_channel_wait_slice(
        desc.channel_waits,
        "instance channel_waits ptr/len mismatch",
    )?;
    validate_u64_slice(desc.channel_ids, "instance channel_ids ptr/len mismatch")?;
    unsafe {
        validate_nested_channel_values(
            desc.seed_values,
            "instance seed_values ptr/len mismatch",
            "instance seed_values bytes ptr/len mismatch",
        )
    }
}

/// Validates a driver-owned instance-binding record returned from bind.
pub fn validate_instance_binding(binding: &PieInstanceBinding) -> PieAbiValidationResult {
    validate_channel_binding_slice(
        binding.channels,
        "instance binding channels ptr/len mismatch",
    )
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
    unsafe {
        validate_nested_channel_values(
            desc.ptir_host_put_values,
            "launch ptir_host_put_values ptr/len mismatch",
            "launch ptir_host_put_values bytes ptr/len mismatch",
        )
    }?;
    validate_u32_slice(desc.kv_len, "launch kv_len ptr/len mismatch")?;
    validate_u64_slice(desc.kv_len_device, "launch kv_len_device ptr/len mismatch")?;

    let request_count = desc.instance_ids.len;
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
            request_count,
            true,
        )?;
        validate_csr(
            desc.kv_page_indptr,
            "launch kv_page_indptr malformed",
            desc.kv_page_indices.len,
            request_count,
            true,
        )?;
        validate_csr(
            desc.rs_buffer_slot_indptr,
            "launch rs_buffer_slot_indptr malformed",
            desc.rs_buffer_slot_ids.len,
            request_count,
            true,
        )?;
        validate_csr(
            desc.sampling_indptr,
            "launch sampling_indptr malformed",
            desc.sampling_indices.len,
            request_count,
            true,
        )?;
        validate_csr(
            desc.host_put_indptr,
            "launch host_put_indptr malformed",
            desc.ptir_host_put_values.len,
            request_count,
            true,
        )?;
        validate_csr(
            desc.image_indptr,
            "launch image_indptr malformed",
            desc.image_anchor_positions.len,
            request_count,
            true,
        )?;
        validate_csr(
            desc.audio_indptr,
            "launch audio_indptr malformed",
            desc.audio_anchor_rows.len,
            request_count,
            true,
        )?;
    }

    validate_row_count_u32(
        desc.kv_last_page_lens.len,
        "launch kv_last_page_lens length must match batch size",
        request_count,
        true,
    )?;
    validate_row_count_u32(
        desc.rs_slot_ids.len,
        "launch rs_slot_ids length must match batch size",
        request_count,
        true,
    )?;
    validate_row_count_u32(
        desc.rs_slot_flags.len,
        "launch rs_slot_flags length must match batch size",
        request_count,
        true,
    )?;
    validate_row_count_u32(
        desc.rs_fold_lens.len,
        "launch rs_fold_lens length must match batch size",
        request_count,
        true,
    )?;
    validate_row_count_u32(
        desc.context_ids.len,
        "launch context_ids length must match batch size",
        request_count,
        true,
    )?;
    validate_row_count_u32(
        desc.kv_len.len,
        "launch kv_len length must match batch size",
        request_count,
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
            request_count,
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

/// Validates non-null out-parameters used by the native driver entrypoints.
pub fn validate_create_out_params(caps: *mut PieDriverCaps) -> PieAbiValidationResult {
    validate_mut_ptr(caps, "driver create caps output pointer must be non-null")
}

unsafe extern "C" {
    pub fn pie_cuda_create(
        desc: *const PieDriverCreateDesc,
        caps: *mut PieDriverCaps,
    ) -> *mut PieDriver;
    pub fn pie_cuda_register_program(
        driver: *mut PieDriver,
        program: *const PieProgramDesc,
        program_id: *mut u64,
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
    pub fn pie_cuda_destroy(driver: *mut PieDriver);
}

unsafe extern "C" {
    pub fn pie_metal_create(
        desc: *const PieDriverCreateDesc,
        caps: *mut PieDriverCaps,
    ) -> *mut PieDriver;
    pub fn pie_metal_register_program(
        driver: *mut PieDriver,
        program: *const PieProgramDesc,
        program_id: *mut u64,
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
            PieProgramDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(
            PieInstanceDesc::default().abi_version,
            PIE_DRIVER_ABI_VERSION
        );
        assert_eq!(PieLaunchDesc::default().abi_version, PIE_DRIVER_ABI_VERSION);
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
        assert_eq!(PieProgramDesc::default().reserved0, 0);
        assert_eq!(PieInstanceDesc::default().reserved0, 0);
        assert_eq!(PieLaunchDesc::default().reserved0, 0);
        assert_eq!(PieLaunchDesc::default().reserved_flags, [0; 6]);
        assert_eq!(PieKvCopyDesc::default().reserved0, 0);
        assert_eq!(PieStateCopyDesc::default().reserved0, 0);
        assert_eq!(PiePoolResizeDesc::default().reserved0, 0);
    }

    #[test]
    fn batch_launch_defaults_are_empty_borrowed_views() {
        let launch = PieLaunchDesc::default();
        assert!(launch.instance_ids.ptr.is_null());
        assert_eq!(launch.instance_ids.len, 0);
        assert!(launch.host_put_indptr.ptr.is_null());
        assert_eq!(launch.host_put_indptr.len, 0);
        assert!(launch.image_pixels.ptr.is_null());
        assert_eq!(launch.image_pixels.len, 0);
        assert_eq!(launch.single_token_mode, 0);
        assert_eq!(launch.has_user_mask, 0);
    }

    #[test]
    fn instance_binding_defaults_keep_channel_mapping_at_bind() {
        let instance = PieInstanceDesc::default();
        assert!(instance.channel_ids.ptr.is_null());
        assert_eq!(instance.channel_ids.len, 0);
        assert!(instance.seed_values.ptr.is_null());
        assert_eq!(instance.seed_values.len, 0);

        let binding = PieInstanceBinding::default();
        assert!(binding.channels.ptr.is_null());
        assert_eq!(binding.channels.len, 0);
    }

    #[test]
    fn completion_layout_is_stable() {
        assert_eq!(std::mem::size_of::<PieCompletion>(), 16);
        assert_eq!(std::mem::align_of::<PieCompletion>(), 8);
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
            validate_channel_wait_slice(
                PieChannelWaitSlice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "waits",
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
            validate_channel_binding_slice(
                PieChannelBindingSlice {
                    ptr: std::ptr::null(),
                    len: 1,
                },
                "bindings",
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
        let tokens = [1u32, 2, 3];
        let positions = [4u32, 5, 6];
        let qo_indptr = [0u32, 2];
        let launch = PieLaunchDesc {
            instance_ids: PieU64Slice {
                ptr: instance_ids.as_ptr(),
                len: instance_ids.len(),
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
        let program = header_block(&header, "PieProgramDesc");
        let instance = header_block(&header, "PieInstanceDesc");
        let binding = header_block(&header, "PieInstanceBinding");
        let launch = header_block(&header, "PieLaunchDesc");
        let kv_copy = header_block(&header, "PieKvCopyDesc");
        let state_copy = header_block(&header, "PieStateCopyDesc");
        let pool_resize = header_block(&header, "PiePoolResizeDesc");

        assert!(launch.contains("struct PieU64Slice instance_ids;"));
        assert!(launch.contains("struct PieU32Slice host_put_indptr;"));
        assert!(launch.contains("uint32_t reserved0;"));
        assert!(launch.contains("uint8_t reserved_flags[6];"));
        assert!(runtime.contains("uint32_t reserved0;"));
        assert!(runtime.contains("PieRuntimeNotifyFn notify;"));
        assert!(create.contains("uint32_t reserved0;"));
        assert!(program.contains("uint64_t program_hash;"));
        assert!(program.contains("uint32_t reserved0;"));
        assert!(instance.contains("struct PieU64Slice channel_ids;"));
        assert!(!program.contains("channel_ids"));
        assert!(header.contains("typedef struct PieChannelBinding {"));
        assert!(header.contains("typedef struct PieChannelBindingSlice {"));
        assert!(binding.contains("struct PieChannelBindingSlice channels;"));
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
