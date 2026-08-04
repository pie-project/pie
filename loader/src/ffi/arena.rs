//! The arena that keeps a [`PieLoaderPlan`]'s slices alive.
//!
//! Every pointer the driver reads points into buffers owned here. The C++ side
//! this replaces used `std::deque` for its backing stores specifically because
//! `push_back` must not invalidate addresses already handed out; the Rust
//! equivalent is to allocate each run as its own boxed slice, which is stable by
//! construction and needs no growth discipline at all.
//!
//! [`PlanArena`] is built once, then frozen: `PlanArena::finish` leaks it into
//! a raw pointer stored in `PieLoaderPlan::owner`, and `pie_loader_release`
//! reclaims it. Between those two points nothing mutates, so handing the plan to
//! another thread is sound.

use crate::plan::{DestExtent, Extent, LoadPlan, SourceExtent, StorageInstr};
use crate::types::{Encoding, QuantScheme, Visibility};

use super::types::*;

/// Owns the backing storage for one plan.
///
/// Field order matters only in that `plan_*` vectors are the ones the exported
/// slices point at; the scalar vectors below them back the nested slices inside
/// those views.
#[derive(Default)]
pub struct PlanArena {
    strings: Vec<Box<[u8]>>,
    u32_runs: Vec<Box<[u32]>>,
    i64_runs: Vec<Box<[i64]>>,
    dim_runs: Vec<Box<[PieLoaderDimSpecView]>>,
    files: Vec<PieLoaderCheckpointFileView>,
    sources: Vec<PieLoaderSourceTensorView>,
    tensors: Vec<PieLoaderTensorDeclView>,
    buffers: Vec<PieLoaderBufferDeclView>,
    instrs: Vec<PieLoaderStorageInstrView>,
    schedule: Vec<u32>,
    attachments: Vec<PieLoaderQuantAttachmentView>,
    groups: Vec<PieLoaderGroupView>,
    bindings: Vec<Box<[PieLoaderSourceBindingView]>>,
    /// Group plans, each built by [`build`] into an arena of its own.
    ///
    /// A nested plan cannot share this arena: the exported slices point at
    /// *these* vectors, and a second plan's instructions would have to live in
    /// the same one, which would make either plan's slice cover both. Owning
    /// the children as whole plans instead costs one indirection and makes
    /// [`release`] recursive, which is the honest shape -- a group's plan is a
    /// plan, and it is freed like one.
    children: Vec<*mut PieLoaderPlan>,
}

impl PlanArena {
    fn store_str(&mut self, value: &str) -> PieLoaderBytes {
        let boxed: Box<[u8]> = value.as_bytes().to_vec().into_boxed_slice();
        let view = PieLoaderBytes {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.strings.push(boxed);
        view
    }

    fn store_u32(&mut self, values: impl IntoIterator<Item = u32>) -> PieLoaderU32Slice {
        let boxed: Box<[u32]> = values.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let view = PieLoaderU32Slice {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.u32_runs.push(boxed);
        view
    }

    fn store_i64(&mut self, values: &[i64]) -> PieLoaderI64Slice {
        let boxed: Box<[i64]> = values.to_vec().into_boxed_slice();
        let view = PieLoaderI64Slice {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.i64_runs.push(boxed);
        view
    }

    fn store_dims(
        &mut self,
        values: impl IntoIterator<Item = PieLoaderDimSpecView>,
    ) -> PieLoaderDimSpecSlice {
        let boxed: Box<[PieLoaderDimSpecView]> =
            values.into_iter().collect::<Vec<_>>().into_boxed_slice();
        let view = PieLoaderDimSpecSlice {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.dim_runs.push(boxed);
        view
    }

    fn stride(&mut self, stride: &Extent) -> PieLoaderStridedExtentView {
        let dims = self.store_dims(stride.dims.iter().map(|dim| PieLoaderDimSpecView {
            count: dim.count,
            src_stride: dim.src_stride,
            dst_stride: dim.dst_stride,
        }));
        PieLoaderStridedExtentView {
            base_offset: stride.base_offset,
            element_bytes: stride.element_bytes,
            dims,
        }
    }

    fn source_extent(&mut self, source: &SourceExtent) -> PieLoaderSourceExtentView {
        let stride = self.stride(&source.stride);
        PieLoaderSourceExtentView {
            file_id: source.file_id.0,
            tensor_id: source.tensor_id.0,
            file_offset: source.file_offset,
            span_bytes: source.span_bytes,
            stride,
            dtype: source.dtype.into(),
        }
    }

    fn dest_extent(&mut self, dest: &DestExtent) -> PieLoaderDestExtentView {
        let stride = self.stride(&dest.stride);
        PieLoaderDestExtentView {
            buffer_id: dest.buffer.0,
            offset: dest.offset,
            stride,
        }
    }

    /// Freeze the arena and publish the plan header.
    ///
    /// The returned `PieLoaderPlan` is itself boxed so the driver holds a stable
    /// pointer; `owner` points at the arena. Both are reclaimed together by
    /// [`release`].
    fn finish(self, plan: &LoadPlan, cache_key: &str) -> *mut PieLoaderPlan {
        let files = PieLoaderCheckpointFileSlice {
            ptr: self.files.as_ptr(),
            len: self.files.len(),
        };
        let sources = PieLoaderSourceTensorSlice {
            ptr: self.sources.as_ptr(),
            len: self.sources.len(),
        };
        let tensors = PieLoaderTensorDeclSlice {
            ptr: self.tensors.as_ptr(),
            len: self.tensors.len(),
        };
        let buffers = PieLoaderBufferDeclSlice {
            ptr: self.buffers.as_ptr(),
            len: self.buffers.len(),
        };
        let instrs = PieLoaderStorageInstrSlice {
            ptr: self.instrs.as_ptr(),
            len: self.instrs.len(),
        };
        let schedule = PieLoaderU32Slice {
            ptr: self.schedule.as_ptr(),
            len: self.schedule.len(),
        };
        let attachments = PieLoaderQuantAttachmentSlice {
            ptr: self.attachments.as_ptr(),
            len: self.attachments.len(),
        };
        let groups = PieLoaderGroupSlice {
            ptr: self.groups.as_ptr(),
            len: self.groups.len(),
        };
        // Rendered here rather than driver-side so the loader stays the only
        // place that knows how to name its own instructions.
        let mut arena = self;
        let cache_key = arena.store_str(cache_key);
        let summary = arena.store_str(&crate::dump::describe(plan));
        let stats_json = arena.store_str(&crate::dump::plan_stats_json(plan));
        let owner = Box::into_raw(Box::new(arena)).cast::<std::ffi::c_void>();
        Box::into_raw(Box::new(PieLoaderPlan {
            files,
            sources,
            tensors,
            buffers,
            instrs,
            schedule,
            memory: PieLoaderMemoryPlanView {
                persistent_bytes: plan.memory.persistent_bytes,
                temporary_peak_bytes: plan.memory.temporary_peak_bytes,
                transform_scratch_peak_bytes: plan.memory.transform_scratch_peak_bytes,
                checkpoint_read_bytes: plan.memory.checkpoint_read_bytes,
                device_write_bytes: plan.memory.device_write_bytes,
            },
            compiler_version: plan.compiler_version,
            target: (&plan.target).into(),
            attachments,
            cache_key,
            summary,
            stats_json,
            groups,
            owner,
        }))
    }
}

/// Split an [`Encoding`] into the three flat fields the POD views carry.
///
/// Mirrors the C++ `encoding()` helper: `Raw` leaves the quant fields at their
/// resting values rather than filling in scheme defaults, so a reader can tell
/// "unquantized" from "quantized at 8 bits" without consulting `encoding_kind`.
fn split_encoding(
    encoding: &Encoding,
) -> (
    PieLoaderEncodingKind,
    PieLoaderDType,
    PieLoaderQuantScheme,
    u8,
    u32,
) {
    match encoding {
        Encoding::Raw(dtype) => (
            PieLoaderEncodingKind::Raw,
            (*dtype).into(),
            PieLoaderQuantScheme::None,
            0,
            0,
        ),
        Encoding::Quant(spec) => (
            PieLoaderEncodingKind::Quant,
            spec.logical_dtype.into(),
            spec.scheme.into(),
            spec.bits_per_element,
            spec.group_size,
        ),
    }
}

fn quant_or_none(scheme: Option<QuantScheme>) -> PieLoaderQuantScheme {
    scheme.map_or(PieLoaderQuantScheme::None, Into::into)
}

/// Flatten one [`StorageInstr`] into the union view.
///
/// The synthesized fields below are not incidental — they are the contract the
/// C++ executor already reads, so they are reproduced verbatim from the parser
/// that is being deleted:
///
/// * `ExtentWrite` republishes `dest.buffer_id` as the instruction's
///   `buffer_id`, so a scheduler can find the written buffer without branching
///   on `kind`.
/// * `BulkExtentWrite` has no `DestExtent` in the Rust IR — only a raw offset —
///   so one is synthesized as a single flat byte run (`element_bytes = 1`, one
///   dim of `span_bytes`) against the sentinel buffer, which is what makes the
///   arena-relative write representable in the same struct as a buffer-relative
///   one.
/// * `CreateView` and `Finalize` publish their scalar operands as
///   one-element `input_buffers`/`output_buffers` runs, so a consumer walking
///   dataflow edges never has to special-case them.
fn flatten_instr(arena: &mut PlanArena, instr: &StorageInstr) -> PieLoaderStorageInstrView {
    use PieLoaderStorageOp as Op;
    let (id, op) = match instr {
        StorageInstr::Allocate { id, buffer } => (
            id,
            Op::Allocate {
                buffer_id: buffer.0,
            },
        ),
        StorageInstr::Fill { id, buffer } => (
            id,
            Op::Fill {
                buffer_id: buffer.0,
            },
        ),
        StorageInstr::ExtentWrite { id, source, dest } => {
            let source = arena.source_extent(source);
            (
                id,
                Op::ExtentWrite {
                    source,
                    dest: arena.dest_extent(dest),
                },
            )
        }
        StorageInstr::BulkExtentWrite {
            id,
            source,
            dest_offset,
        } => (
            id,
            Op::BulkExtentWrite {
                source: arena.source_extent(source),
                dest_offset: *dest_offset,
            },
        ),
        StorageInstr::TileMap {
            id,
            kind,
            source,
            dest,
            inputs,
            outputs,
            tile,
            transform,
        } => {
            let (source, has_source) = match source {
                Some(source) => (arena.source_extent(source), true),
                None => (PieLoaderSourceExtentView::default(), false),
            };
            let (dest, has_dest) = match dest {
                Some(dest) => (arena.dest_extent(dest), true),
                None => (PieLoaderDestExtentView::default(), false),
            };
            let repack = transform.repack;
            (
                id,
                Op::TileMap {
                    tile_kind: (*kind).into(),
                    source,
                    has_source,
                    dest,
                    has_dest,
                    input_buffers: arena.store_u32(inputs.iter().map(|b| b.0)),
                    output_buffers: arena.store_u32(outputs.iter().map(|b| b.0)),
                    rows_per_tile: tile.rows_per_tile,
                    transform_fusion: transform.fusion.into(),
                    transform_from: quant_or_none(transform.from),
                    transform_to: quant_or_none(transform.to),
                    repack_layout: repack.map_or(PieLoaderRepackLayout::None, |r| r.layout.into()),
                    transform_batch: repack.map_or(0, |r| r.batch),
                    transform_source_rows: repack.map_or(0, |r| r.source_rows),
                    transform_target_rows: repack.map_or(0, |r| r.target_rows),
                    transform_source_cols: repack.map_or(0, |r| r.source_cols),
                    transform_target_cols: repack.map_or(0, |r| r.target_cols),
                    transform_scratch_bytes: transform.scratch_bytes,
                    transform_metadata_source: transform
                        .metadata_source
                        .map_or(PIE_LOADER_NO_TENSOR, |id| id.0),
                    transform_scale_factor_bits: transform.scale_factor_bits,
                    transform_scale_blocks: arena.store_i64(&transform.scale_blocks),
                },
            )
        }
        StorageInstr::CreateView {
            id,
            input,
            output,
            view,
        } => (
            id,
            Op::CreateView {
                input_buffer: input.0,
                output_buffer: output.0,
                view: arena.dest_extent(view),
            },
        ),
        StorageInstr::Finalize { id, tensor, name } => (
            id,
            Op::Finalize {
                buffer_id: tensor.0,
                name: arena.store_str(name),
            },
        ),
    };
    PieLoaderStorageInstrView { id: id.0, op }
}

/// The cache key to ship with a plan that was not compiled for a snapshot.
///
/// A key is still a key: an empty one would collide across every model, so the
/// callers that have no snapshot path to hash — `plan_view` and its tests — name
/// this instead of inventing one each.
pub const UNKEYED: &str = "0000000000000000";

/// Convert a compiled [`LoadPlan`] into the POD form the driver walks.
///
/// The caller owns the result and must hand it to [`release`].
pub fn build(plan: &LoadPlan, cache_key: &str) -> *mut PieLoaderPlan {
    let mut arena = PlanArena::default();

    arena.tensors.reserve(plan.tensors.len());
    for tensor in &plan.tensors {
        let name = arena.store_str(&tensor.name);
        let shape = arena.store_i64(&tensor.shape);
        let (encoding_kind, dtype, quant_scheme, quant_bits_per_element, quant_group_size) =
            split_encoding(&tensor.encoding);
        arena.tensors.push(PieLoaderTensorDeclView {
            dtype,
            id: tensor.id.0,
            name,
            encoding_kind,
            quant_scheme,
            quant_bits_per_element,
            quant_group_size,
            shape,
            alignment: tensor.alignment,
            visibility: match tensor.visibility {
                Visibility::Public => PieLoaderVisibility::Public,
                Visibility::Internal => PieLoaderVisibility::Internal,
            },
        });
    }

    arena.files.reserve(plan.files.len());
    for file in &plan.files {
        let path = arena.store_str(&file.path);
        arena.files.push(PieLoaderCheckpointFileView {
            id: file.id.0,
            path,
            size_bytes: file.size_bytes,
            format: file.format.into(),
        });
    }

    arena.sources.reserve(plan.sources.len());
    for source in &plan.sources {
        let name = arena.store_str(&source.name);
        let shape = arena.store_i64(&source.shape);
        let (encoding_kind, dtype, quant_scheme, quant_bits_per_element, quant_group_size) =
            split_encoding(&source.encoding);
        arena.sources.push(PieLoaderSourceTensorView {
            dtype,
            id: source.id.0,
            name,
            file_id: source.file_id.0,
            file_offset: source.file_offset,
            span_bytes: source.span_bytes,
            encoding_kind,
            quant_scheme,
            quant_bits_per_element,
            quant_group_size,
            shape,
        });
    }

    arena.buffers.reserve(plan.buffers.len());
    for buffer in &plan.buffers {
        arena.buffers.push(PieLoaderBufferDeclView {
            id: buffer.id.0,
            tensor_id: buffer.tensor.map_or(PIE_LOADER_NO_BUFFER, |t| t.0),
            has_tensor: buffer.tensor.is_some(),
            bytes: buffer.bytes,
            alignment: buffer.alignment,
            temporary: buffer.temporary,
            has_persistent_offset: buffer.persistent_offset.is_some(),
            persistent_offset: buffer.persistent_offset.unwrap_or(0),
        });
    }

    arena.instrs.reserve(plan.instrs.len());
    for instr in &plan.instrs {
        let view = flatten_instr(&mut arena, instr);
        arena.instrs.push(view);
    }

    arena.schedule = plan.schedule.iter().map(|id| id.0).collect();

    arena.attachments.reserve(plan.attachments.len());
    for attachment in &plan.attachments {
        arena.attachments.push(PieLoaderQuantAttachmentView {
            tensor_id: attachment.tensor.0,
            scale_tensor_id: attachment.scale_tensor.0,
            zero_point_tensor_id: attachment
                .zero_point_tensor
                .map_or(PIE_LOADER_NO_TENSOR, |id| id.0),
            granularity: PieLoaderQuantGranularity::from(attachment.granularity),
            group_size: attachment.group_size,
            channel_axis: attachment.channel_axis,
            scale_form: PieLoaderScaleForm::from(attachment.scale_form),
        });
    }

    arena.groups.reserve(plan.groups.len());
    for group in &plan.groups {
        // Built first: `store_str` borrows `arena` mutably, and the child plan
        // is an independent allocation either way.
        let child = build(&group.plan, UNKEYED);
        arena.children.push(child);

        let name = arena.store_str(&group.name);
        let per_instance = group.bindings.first().map_or(0, Vec::len);
        let flat: Box<[PieLoaderSourceBindingView]> = group
            .bindings
            .iter()
            .flatten()
            .map(|binding| PieLoaderSourceBindingView {
                instr_id: binding.instr.0,
                file_id: binding.file_id.0,
                tensor_id: binding.tensor_id.0,
                file_offset: binding.file_offset,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let bindings = PieLoaderSourceBindingSlice {
            ptr: flat.as_ptr(),
            len: flat.len(),
        };
        arena.bindings.push(flat);
        arena.groups.push(PieLoaderGroupView {
            name,
            arity: group.arity,
            plan: child,
            bindings,
            bindings_per_instance: per_instance,
        });
    }

    arena.finish(plan, cache_key)
}

/// Reclaim a plan produced by [`build`].
///
/// # Safety
///
/// `plan` must be a pointer returned by [`build`] and not yet released.
pub unsafe fn release(plan: *mut PieLoaderPlan) {
    if plan.is_null() {
        return;
    }
    let boxed = unsafe { Box::from_raw(plan) };
    if !boxed.owner.is_null() {
        drop(unsafe { Box::from_raw(boxed.owner.cast::<PlanArena>()) });
    }
}

/// A group's plans are freed with the plan that owns them.
///
/// Implemented on the arena rather than in [`release`] so that the recursion
/// happens exactly where the ownership is recorded -- a child added to
/// `children` is reclaimed whether it was reached through `release` or through
/// a parent arena being dropped.
impl Drop for PlanArena {
    fn drop(&mut self) {
        for child in std::mem::take(&mut self.children) {
            unsafe { release(child) };
        }
    }
}

/// The arena is immutable once published, and every pointer it hands out is into
/// storage it exclusively owns, so a plan can be read from any thread and freed
/// from a thread other than the one that built it. Ranks load in parallel
/// (`runtime/engine/src/driver/backend/cuda.rs:331-345`), so this is a
/// requirement, not a convenience.
unsafe impl Send for PlanArena {}
unsafe impl Sync for PlanArena {}
unsafe impl Send for PieLoaderPlan {}
unsafe impl Sync for PieLoaderPlan {}

/// Read a plan's contents back as safe Rust slices, for tests that assert the
/// POD form reproduces the plan it was built from.
#[cfg(test)]
pub(crate) mod view {
    use super::*;

    pub unsafe fn instrs(plan: *const PieLoaderPlan) -> &'static [PieLoaderStorageInstrView] {
        let plan = unsafe { &*plan };
        if plan.instrs.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.instrs.ptr, plan.instrs.len) }
    }

    pub unsafe fn tensors(plan: *const PieLoaderPlan) -> &'static [PieLoaderTensorDeclView] {
        let plan = unsafe { &*plan };
        if plan.tensors.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.tensors.ptr, plan.tensors.len) }
    }

    pub unsafe fn files(plan: *const PieLoaderPlan) -> &'static [PieLoaderCheckpointFileView] {
        let plan = unsafe { &*plan };
        if plan.files.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.files.ptr, plan.files.len) }
    }

    pub unsafe fn sources(plan: *const PieLoaderPlan) -> &'static [PieLoaderSourceTensorView] {
        let plan = unsafe { &*plan };
        if plan.sources.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.sources.ptr, plan.sources.len) }
    }

    pub unsafe fn buffers(plan: *const PieLoaderPlan) -> &'static [PieLoaderBufferDeclView] {
        let plan = unsafe { &*plan };
        if plan.buffers.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.buffers.ptr, plan.buffers.len) }
    }

    pub unsafe fn schedule(plan: *const PieLoaderPlan) -> &'static [u32] {
        let plan = unsafe { &*plan };
        if plan.schedule.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(plan.schedule.ptr, plan.schedule.len) }
    }

    pub unsafe fn bytes(value: &PieLoaderBytes) -> &'static str {
        if value.ptr.is_null() {
            return "";
        }
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(value.ptr, value.len)) }
    }

    pub unsafe fn u32s(value: &PieLoaderU32Slice) -> &'static [u32] {
        if value.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(value.ptr, value.len) }
    }

    pub unsafe fn i64s(value: &PieLoaderI64Slice) -> &'static [i64] {
        if value.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(value.ptr, value.len) }
    }

    pub unsafe fn dims(value: &PieLoaderDimSpecSlice) -> &'static [PieLoaderDimSpecView] {
        if value.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(value.ptr, value.len) }
    }
}
