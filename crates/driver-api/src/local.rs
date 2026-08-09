//! The records a driver ANSWERS with, and the constants both sides name.
//!
//! # What this module stopped being
//!
//! It was "direct local FFI descriptors for embedded CUDA and Metal drivers":
//! twenty-eight `#[repr(C)]` structs, four `{ptr, len}` slice views, an opaque
//! `PieDriver = c_void`, an `unsafe extern "C"` notify callback, and seven
//! validators whose job was to check that a foreign caller had filled them in.
//! Every one of those existed because the driver on the far side was C++.
//!
//! None of them is. `driver-cuda` was the last shell reached through a C
//! linkage — thirteen `pie_cuda_*` free functions and a `*mut PieDriver` — and
//! it is a Rust type with Rust methods now, so the descriptors it took apart
//! with `slice_of` are the owned types in [`plan`](crate::plan) and
//! [`submission`](crate::submission) that the engine already had. A round trip
//! through `#[repr(C)]` between two Rust crates in one workspace bought
//! nothing and cost the types.
//!
//! # What is left, and why each one is
//!
//! Five records and the constants. They survive because they are genuinely
//! shared vocabulary rather than a marshalling shape:
//!
//! - [`TerminalCell`] — a location two threads share. It keeps `#[repr(C)]`
//!   and it is the only survivor that earns it: `StepSubmission::terminal_cells`
//!   holds `*mut TerminalCell`, `worker`'s executor holds a `Vec` of them, and
//!   a driver publishes into one while the engine reads it. The `AtomicU32`
//!   inside is what makes that sentence checkable.
//! - [`ChannelBinding`] and [`InstanceBinding`] — what a driver ANSWERS from
//!   `register_channel` and `bind_instance`. All four device drivers name
//!   both, which is what makes them vocabulary; the descriptors they used to
//!   answer *to* are gone.
//! - [`KvMoveCell`], [`StateCopyRange`], [`PoolRange`] — the rows inside a
//!   copy or resize plan. `serde`-derived, because the remote protocol carries
//!   them to another node.
//!
//! The `#[repr(C)]` on the last four is gone: nothing lays them out for a
//! foreign reader any more, and leaving the attribute on would have been a
//! claim about a boundary that no longer exists.
//!
//! The validators went the same way. Five of the seven checked descriptors
//! that no longer exist; the two that remain
//! ([`validate_channel_endpoint_binding`], [`validate_instance_binding`])
//! check what a DRIVER answered, which is the direction the type system
//! cannot check and therefore the only direction still worth a validator.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use serde::{Deserialize, Serialize};

use crate::geometry::GeometryClass;

/// Current direct local ABI version.
///
/// v14 (Project Venus): the launch unit is the sealed **frame** — one
/// `FrameSubmission` carries k forward steps that the driver executes as one
/// closed system with a single completion. Frame-invariant tables (lane
/// roster, WorkingSet page translation, frame-union KV admission demand) are
/// hoisted out of the per-step sections. Admission is folded into the launch
/// call itself ([`PIE_STATUS_EXHAUSTED`] / [`PIE_STATUS_IMPOSSIBLE`]); the
/// v12 prepare/lease surface and the v13 `settle_defer` lever are deleted.
///
/// v17 (phase 3′): `PieProgramDesc::region_analysis` — the per-region bind
/// verdicts and intrinsic side-table analysis the CUDA driver derives for
/// itself in `region_support.hpp`. Additive, and empty means "not supplied",
/// but the struct grew, so drivers and workers ship together.
/// v21: `LaunchPlan::rs_buffer_read_*` — the buffered prefix a fire must
/// REPLAY before its own tokens, so a recurrence can start from
/// `folded ⊕ replay(buffer)` instead of only from the folded boundary.
/// Separate from the write CSR because a write may allocate a slab and a read
/// must not. Additive, and an empty read side means "nothing to replay", but
/// the struct grew, so drivers and workers ship together.
/// v22: `LaunchPlan::rs_buffer_heads` — where each row's logical buffer
/// token 0 physically sits. A fold absorbs tokens off the front of the buffer
/// but can only release WHOLE covered pages, and `fold_granularity` is 1 while
/// a buffer page is the KV page size, so a fold routinely lands mid-page and
/// the survivors keep their offsets. Every buffer span the driver walks is
/// therefore `head + logical`. Zero for a buffer that was never partially
/// folded, which is why this was invisible until the replay path landed.
/// v23: `PIE_RS_FLAG_BUFFER_WRITE` — a new bit in `rs_slot_flags` marking a
/// row whose buffer span is a WRITE. Orthogonal to `PIE_RS_FLAG_FOLD`: a pass
/// may scatter its own tokens into the buffer AND fold a prefix of the result
/// in one go, and the two flags together are what tell a write-and-fold (run
/// the extended `[buffered | new]` layout, snapshot the state at
/// `rs_fold_lens[r]`) apart from a pure commit (whose rows ARE the replay).
/// No struct grew, but an older driver rejects the unknown bit, so drivers and
/// workers ship together.
/// v24: `PIE_RS_FLAG_FOLD_LEN_DEVICE` — a new bit in `rs_slot_flags` marking a
/// row whose fold length the WORKER DOES NOT KNOW. The value lives in the
/// `rs_fold_len` descriptor port, which the driver resolves at compose time,
/// so a speculative decode's accepted count never has to round-trip through
/// the host between the fire that computes it and the fire that folds it.
/// `rs_fold_lens[r]` is a placeholder for such a row and MUST be ignored; the
/// driver clamps the resolved value to the row's replay length. No struct
/// grew, but an older driver rejects the unknown bit, so drivers and workers
/// ship together.
pub const PIE_DRIVER_ABI_VERSION: u32 = 25;
pub const PIE_MODEL_COMPONENT_FULL: u32 = 0;
pub const PIE_MODEL_COMPONENT_TEXT: u32 = 1;
pub const PIE_MODEL_COMPONENT_ENCODE: u32 = 2;

/// MXFP4 MoE lowering request. Discriminants match `PieLoaderMxfp4MoeRequest`;
/// the driver forwards the value to the loader unchanged.
pub const PIE_MXFP4_MOE_AUTO: u32 = 0;
pub const PIE_MXFP4_MOE_ROUTED_DECODE: u32 = 1;
pub const PIE_MXFP4_MOE_NATIVE_GEMM: u32 = 2;
pub const PIE_MXFP4_MOE_EAGER_BF16: u32 = 3;

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
/// Frame admission is full right now; the frame may fit after physical
/// budget is released. The engine re-posts later.
pub const PIE_STATUS_EXHAUSTED: i32 = -6;
/// The frame can never fit within the driver's physical budget ceiling.
pub const PIE_STATUS_IMPOSSIBLE: i32 = -7;

// Literal values, and the assert pins them to the
// Rust enum.
/// Sentinel for `LaunchOp::channel` on ops that touch no channel.
pub const PIE_NO_CHANNEL: u32 = u32::MAX;

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
/// The pass SCATTERS its own tokens into the buffer. Orthogonal to `FOLD`: a
/// pass may write the buffer and fold a prefix of the result in one go, and
/// the two together are what distinguishes a write-and-fold (which runs the
/// extended `[buffered | new]` layout and snapshots the state at
/// `rs_fold_lens[r]`) from a pure commit (whose rows ARE the replay, gathered
/// straight from the slabs).
pub const PIE_RS_FLAG_BUFFER_WRITE: u8 = 4;
/// This row's fold length is NOT host-known. `rs_fold_lens[r]` is a
/// placeholder and must be ignored; the real value comes from the
/// `rs_fold_len` descriptor port, which the driver resolves once the fire that
/// computes it has completed, and clamps to the row's replay length.
pub const PIE_RS_FLAG_FOLD_LEN_DEVICE: u8 = 8;

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
pub type DeviceDomain = u32;

/// Page-locked host memory.
pub const PIE_MEMORY_DOMAIN_HOST_PINNED: DeviceDomain = 0;
/// CUDA device memory on `*_device_ordinal`.
pub const PIE_MEMORY_DOMAIN_CUDA_DEVICE: DeviceDomain = 1;
/// ROCm device memory on `*_device_ordinal`.
pub const PIE_MEMORY_DOMAIN_ROCM_DEVICE: DeviceDomain = 2;
/// Metal shared CPU/GPU memory.
pub const PIE_MEMORY_DOMAIN_METAL_SHARED: DeviceDomain = 3;
/// Metal private device memory.
pub const PIE_MEMORY_DOMAIN_METAL_PRIVATE: DeviceDomain = 4;
/// Vulkan device memory on `*_device_ordinal`.
///
/// One arm and not two, unlike Metal's shared/private pair: a Vulkan driver
/// chooses its heap from the memory types the device reports, and whether
/// that memory is also host-visible is a property of the DEVICE rather than
/// of the allocation -- an integrated GPU and a discrete card with resizable
/// BAR both report host-visible device-local memory. A caller that needs to
/// know asks the driver's facts for `unified_memory`.
pub const PIE_MEMORY_DOMAIN_VULKAN_DEVICE: DeviceDomain = 5;
/// WebGPU device memory, allocated through `wgpu`.
///
/// Its own tag even though the API underneath it is Vulkan, Metal or D3D12,
/// because the tag is a discriminator and not a description: the engine stamps
/// it on every `KvCopyPlan` and a driver refuses a plan whose ends are not its
/// own, so a `wgpu` shell answering [`PIE_MEMORY_DOMAIN_VULKAN_DEVICE`] would
/// accept a plan naming a `driver-vulkan` pool's pages. Two allocators, two
/// buffers, one number, and a copy between unrelated pools is exactly what the
/// tag exists to refuse.
///
/// One arm and not one per sibling API: `wgpu` picks its backend at runtime and
/// can pick a different one between two runs on one machine, so a per-API tag
/// would be a fact this driver cannot state at boot. Whether the memory is also
/// host-visible is answered by the facts' `unified_memory`, as it is for
/// Vulkan.
pub const PIE_MEMORY_DOMAIN_WEBGPU_DEVICE: DeviceDomain = 6;
pub const PIE_ELASTIC_POOL_KV: u64 = 0;
pub const PIE_ELASTIC_POOL_STATE: u64 = 1;
pub const PIE_ELASTIC_POOL_WORKSPACE: u64 = 2;


/// Terminal completion outcome published by the native driver.
pub type TerminalOutcome = u32;

/// The operation has not reached a terminal state yet.
pub const PIE_TERMINAL_OUTCOME_PENDING: TerminalOutcome = 0;
/// The operation completed successfully.
pub const PIE_TERMINAL_OUTCOME_SUCCESS: TerminalOutcome = 1;
/// The operation completed unsuccessfully.
pub const PIE_TERMINAL_OUTCOME_FAILED: TerminalOutcome = 2;
/// The accepted work item committed no effects and must be attempted again.
pub const PIE_TERMINAL_OUTCOME_RETRY: TerminalOutcome = 3;

/// Host-visible terminal control cell.
///
/// The `outcome` word is published with release semantics by the driver and
/// read with acquire semantics by the runtime — so it IS an [`AtomicU32`],
/// and declaring it as one is what makes that sentence checkable.
///
/// It was a plain `u32` for as long as a C++ driver wrote it, because a
/// `_Atomic uint32_t` and a `u32` had to agree on layout and the Rust side
/// took the weaker of the two. `AtomicU32` has the layout of `u32`, so the
/// repr is unchanged and nothing about the cost is — what changed is that
/// four call sites (`engine`'s completion reader and its remote publisher,
/// `worker`'s executor, `driver-cuda`'s state fence) each re-derived the
/// atomic view with `AtomicU32::from_ptr(addr_of!(..))`, and a fifth
/// (`OwnedTerminalCell::reset`) wrote the word NON-atomically through
/// `*ptr = ..` while a driver could still be publishing into it.
///
/// The type no longer derives `Copy`/`Clone`/`PartialEq`: a cell is a
/// location that two threads share, not a value that can be copied out from
/// under one of them. Read it with [`Self::load`].
#[repr(C)]
#[derive(Debug, Default)]
pub struct TerminalCell {
    pub outcome: AtomicU32,
    /// Reserved; must be zero.
    pub reserved0: u32,
}

impl TerminalCell {
    /// A fresh cell, not yet published to.
    #[must_use]
    pub const fn pending() -> Self {
        Self {
            outcome: AtomicU32::new(PIE_TERMINAL_OUTCOME_PENDING),
            reserved0: 0,
        }
    }

    /// The published outcome, acquiring whatever the publisher released.
    pub fn load(&self) -> TerminalOutcome {
        self.outcome.load(Ordering::Acquire)
    }

    /// Publish a terminal outcome, releasing the effects that preceded it.
    pub fn publish(&self, outcome: TerminalOutcome) {
        self.outcome.store(outcome, Ordering::Release);
    }

    /// Return the cell to [`PIE_TERMINAL_OUTCOME_PENDING`] before reuse.
    ///
    /// Released rather than relaxed for the same reason a publication is: a
    /// recycled cell is handed to a driver that must not observe the previous
    /// tenant's outcome.
    pub fn reset(&self) {
        self.outcome
            .store(PIE_TERMINAL_OUTCOME_PENDING, Ordering::Release);
    }
}


/// Stable driver-owned host endpoint returned by channel registration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChannelBinding {
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


/// A single KV cell move expressed in physical page/token coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvMoveCell {
    pub dst_page_id: u32,
    pub dst_token_offset: u32,
    pub src_page_id: u32,
    pub src_token_offset: u32,
}

/// One recurrent-state slot copy range.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateCopyRange {
    pub src_slot_id: u32,
    pub dst_slot_id: u32,
    pub src_token_offset: u32,
    pub dst_token_offset: u32,
    pub token_count: u32,
}

/// One sparse pool page range to map or unmap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRange {
    pub page_index: u64,
    pub page_count: u64,
}


/// What an emitted kernel is for. The driver switches on this to decide which
/// launch path a compiled entry belongs to, so it never has to re-derive from
/// the plan what the host already decided.
pub const PIE_KERNEL_SINGLETON: u32 = 0;
pub const PIE_KERNEL_FUSED: u32 = 1;
pub const PIE_KERNEL_GROUPED: u32 = 2;
pub const PIE_KERNEL_READINESS: u32 = 3;
pub const PIE_KERNEL_COMMIT: u32 = 4;


/// The region can be bound as a second-party region.
pub const PIE_REGION_SECOND_PARTY_SUPPORTED: u32 = 1 << 0;
/// The region is a well-formed generated region.
pub const PIE_REGION_GENERATED_VALID: u32 = 1 << 1;


// ── the launch package ──
//
// Everything below is the program itself, in the shape a driver executes it:
// channels to allocate, ports to bind, per-stage op DAGs to launch, and the
// per-region plan the emitted kernels were generated from.
//
// It is deliberately *not* PTIR. A driver reading this table never sees a
// container, a sidecar, a hash to check, or a wire format to parse — those are
// the compiler's business, and the compiler has already validated all of them
// (`ptir-refactor.md` §2.3). Records are flat and POD; nested `Pie*Slice`
// fields point at host-owned arrays that outlive the registration call.

/// Where a value comes from. Mirrors `tensor_ir` value sources.
pub const PIE_VALUE_CONST: u8 = 0;
pub const PIE_VALUE_INTRINSIC: u8 = 1;
pub const PIE_VALUE_CHANNEL_TAKE: u8 = 2;
pub const PIE_VALUE_CHANNEL_READ: u8 = 3;
pub const PIE_VALUE_OP_RESULT: u8 = 4;

/// The channel is pre-filled with a seed cell at instantiation.
pub const PIE_CHANNEL_SEEDED: u8 = 1 << 0;
/// The host reads or writes this channel.
pub const PIE_CHANNEL_HOST_VISIBLE: u8 = 1 << 1;
/// The host is the *reader* — the channel is a program output.
pub const PIE_CHANNEL_HOST_READER: u8 = 1 << 2;

/// No op in the pass touches this channel, so a fire has nothing to wait for.
pub const PIE_READINESS_UNTOUCHED: u8 = 0;
/// The first op to touch this channel in pass order takes or reads it, so a
/// fire is ready only while the ring is non-empty.
pub const PIE_READINESS_NEEDS_FULL: u8 = 1;
/// The first op to touch this channel in pass order puts to it, so a fire is
/// ready only while the ring is non-full.
pub const PIE_READINESS_NEEDS_EMPTY: u8 = 2;

/// The region is served by a generated kernel.
pub const PIE_REGION_GENERATED: u8 = 0;
/// The region is served by a vendor or second-party library call.
pub const PIE_REGION_LIBRARY: u8 = 1;

/// A dimension is a literal extent, not a symbolic one.
pub const PIE_EXTENT_STATIC: u8 = 0xff;

/// The stage reads the `query` intrinsic.
pub const PIE_STAGE_REQUIRES_QUERY: u32 = 1 << 0;
/// The stage reads the `layer` intrinsic.
pub const PIE_STAGE_REQUIRES_LAYER: u32 = 1 << 1;
/// The stage reads the `attn_score` intrinsic.
pub const PIE_STAGE_REQUIRES_ATTN_SCORE: u32 = 1 << 2;
/// The stage names a second-party kernel.
pub const PIE_STAGE_REQUIRES_KERNEL_CALL: u32 = 1 << 3;
/// The stage writes the `attn_page_mask` sink.
pub const PIE_STAGE_REQUIRES_PAGE_MASK: u32 = 1 << 4;
/// The stage reads multi-token-prediction draft rows.
pub const PIE_STAGE_REQUIRES_MTP_ROWS: u32 = 1 << 5;
/// Every op in the stage is coverable by the grouped launch path, and its
/// intrinsics and runtime extents are ones that path supports. When clear,
/// `error` says why and the stage must take the fused path.
pub const PIE_STAGE_GROUPED_VALID: u32 = 1 << 6;
/// The stage writes the `lora` sink.
pub const PIE_STAGE_REQUIRES_LORA: u32 = 1 << 7;


/// Driver-assigned identity returned from `*_bind_instance`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InstanceBinding {
    pub instance_id: u64,
    pub geometry_class: u32,
    /// Reserved; must be zero.
    pub reserved0: u32,
}

/// `StepSubmission::region_sig` bit: the region's members carry
/// multi-token qo windows (the ragged window class).
pub const PIE_REGION_SIG_MULTI_TOKEN: u32 = 1 << 0;

/// `StepSubmission::region_sig` bit: attention-stage hook programs.
pub const PIE_REGION_SIG_HOOK: u32 = 1 << 1;

/// `StepSubmission::region_sig` bit: a user (custom) attention mask.
pub const PIE_REGION_SIG_MASK: u32 = 1 << 2;

/// `StepSubmission::region_sig` bit: a depth truncation (the region's k
/// is `region_k`).
pub const PIE_REGION_SIG_TRUNCATED: u32 = 1 << 3;

/// `StepSubmission::region_sig` bit: a span-grouped correction (lora)
/// program. Window-free — never a seriation term — but the depth
/// split's decline rules consult it (a lane carrying BOTH correction
/// and truncation is the PQ-tree class, refused), so the table states
/// it (③b: the words' decline rules become derivable).
pub const PIE_REGION_SIG_LORA: u32 = 1 << 4;

/// `StepSubmission::region_sig` bit: the region's hook programs write the
/// `attn_page_mask` sink (Track B page substitution) — such a hook needs
/// the full-R paged decode path, so the banded-depth derivation excludes
/// it.
pub const PIE_REGION_SIG_HOOK_PAGE_MASK: u32 = 1 << 5;

/// `StepSubmission::planned_hook_free_prefix_rows`'s "no plan sent"
/// sentinel. Not zero: zero is a legitimate planned value ("no fast
/// prefix" — an all-hooked step).
pub const PIE_HOOK_FREE_PREFIX_UNPLANNED: u32 = u32::MAX;

/// `StepSubmission::planned_unmasked_prefix_rows`'s "no plan sent"
/// sentinel (zero is a legitimate planned value: an all-masked step).
pub const PIE_UNMASKED_PREFIX_UNPLANNED: u32 = u32::MAX;

/// `StepSubmission::planned_max_layers`'s "full model" sentinel (zero is
/// never a legitimate depth).
pub const PIE_MAX_LAYERS_FULL: u32 = u32::MAX;

/// `StepSubmission::planned_full_depth_rows`'s "no depth split" sentinel
/// (zero would mean an all-truncated composed fire, a legal future
/// value).
pub const PIE_FULL_DEPTH_UNPLANNED: u32 = u32::MAX;

/// ABI descriptor validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationError {
    status: i32,
    message: &'static str,
}

impl ValidationError {
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

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (status {})", self.message, self.status)
    }
}

impl std::error::Error for ValidationError {}

pub type ValidationResult = Result<(), ValidationError>;

const fn abi_version_error() -> ValidationError {
    ValidationError::new(
        PIE_STATUS_BAD_ABI_VERSION,
        "descriptor abi_version does not match PIE_DRIVER_ABI_VERSION",
    )
}

const fn invalid_argument(message: &'static str) -> ValidationError {
    ValidationError::new(PIE_STATUS_INVALID_ARGUMENT, message)
}

/// Returns true when `outcome` is a valid [`TerminalOutcome`] discriminant.
pub const fn pie_terminal_outcome_is_valid(outcome: TerminalOutcome) -> bool {
    matches!(
        outcome,
        PIE_TERMINAL_OUTCOME_PENDING
            | PIE_TERMINAL_OUTCOME_SUCCESS
            | PIE_TERMINAL_OUTCOME_FAILED
            | PIE_TERMINAL_OUTCOME_RETRY
    )
}

/// Returns true when `domain` is a valid [`DeviceDomain`] discriminant.
pub const fn pie_memory_domain_is_valid(domain: DeviceDomain) -> bool {
    matches!(
        domain,
        PIE_MEMORY_DOMAIN_HOST_PINNED
            | PIE_MEMORY_DOMAIN_CUDA_DEVICE
            | PIE_MEMORY_DOMAIN_ROCM_DEVICE
            | PIE_MEMORY_DOMAIN_METAL_SHARED
            | PIE_MEMORY_DOMAIN_METAL_PRIVATE
            | PIE_MEMORY_DOMAIN_VULKAN_DEVICE
            | PIE_MEMORY_DOMAIN_WEBGPU_DEVICE
    )
}

/// Validates a top-level ABI version tag.
pub const fn validate_pie_abi_version(abi_version: u32) -> ValidationResult {
    if abi_version == PIE_DRIVER_ABI_VERSION {
        Ok(())
    } else {
        Err(abi_version_error())
    }
}

fn validate_reserved_zero(name: &'static str, value: u32) -> ValidationResult {
    if value == 0 {
        Ok(())
    } else {
        Err(invalid_argument(name))
    }
}


/// Validates a driver-owned channel endpoint binding against the plan it
/// answers.
///
/// Takes the owned [`ChannelRegistrationPlan`](crate::plan::ChannelRegistrationPlan)
/// rather than a `#[repr(C)]` mirror of it. The mirror existed so a C++ driver
/// could be handed `{ptr, len}` views of the plan's `Vec`s; every driver is
/// Rust now, so the plan itself crosses and this reads the two fields it needs
/// off it.
///
/// # Errors
///
/// A binding whose identity, capacity, storage size or word layout does not
/// answer the plan.
pub fn validate_channel_endpoint_binding(
    binding: &ChannelBinding,
    plan: &crate::plan::ChannelRegistrationPlan,
) -> ValidationResult {
    if binding.channel_id != plan.channel_id {
        return Err(invalid_argument("channel binding id mismatch"));
    }
    if binding.mirror_base == 0 || binding.word_base == 0 {
        return Err(invalid_argument("channel binding bases must be nonzero"));
    }
    if binding.cell_bytes == 0 || binding.capacity != plan.capacity {
        return Err(invalid_argument("channel binding geometry mismatch"));
    }
    let ring_cells = u64::from(plan.capacity)
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
pub fn validate_instance_binding(binding: &InstanceBinding) -> ValidationResult {
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

