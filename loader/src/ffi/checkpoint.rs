//! The checkpoint, as a value the driver can read before it writes a contract.
//!
//! `plan = compile(source_facts, program, target)` has three inputs, and this
//! module is the first of them. A driver cannot author a contract without
//! knowing what the files contain: whether this checkpoint ships a fused
//! `qkv_proj` or three separate projections, whether the MoE experts are one
//! stacked tensor or one per expert, what the on-disk extents of a packed AWQ
//! weight actually are. Every one of those questions is answered by the tensor
//! table, and every one of them is what the loader's own `arch/` passes were
//! reading the table for.
//!
//! # Why a handle and not a directory
//!
//! The obvious API is `compile(snapshot_dir, contract, target)`, with the
//! loader parsing the headers itself. That reads the checkpoint twice — once
//! for the driver to write the contract against, once inside the compile — and
//! two reads of a file that can change between them can disagree. Opening the
//! checkpoint once and passing the handle makes the source facts a *value*, so
//! the contract and the plan are provably about the same bytes.
//!
//! It also makes the split in `checkpoint::read` visible from C++: this is the
//! only entry point that touches the filesystem before execution.

use std::path::PathBuf;

use crate::checkpoint::CheckpointMetadata;
use crate::ffi::contract::{PieLoaderEncodingSpec, write_encoding};
use crate::ffi::types::{
    PieLoaderBytes, PieLoaderCheckpointFileSlice, PieLoaderCheckpointFileView,
    PieLoaderCheckpointFormat, PieLoaderI64Slice,
};

/// One tensor as it appears in the checkpoint.
///
/// The encoding is the full [`PieLoaderEncodingSpec`], not the flattened form
/// `PieLoaderSourceTensorView` uses: a driver reads this and writes it straight
/// back into a contract node, so a lossy summary here would be a contract that
/// cannot name what it found.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderRawTensorView {
    pub id: u32,
    pub name: PieLoaderBytes,
    pub file_id: u32,
    pub file_offset: u64,
    pub span_bytes: u64,
    /// Extents as the file declares them, before any sharding.
    pub shape: PieLoaderI64Slice,
    pub encoding: PieLoaderEncodingSpec,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderRawTensorSlice {
    pub ptr: *const PieLoaderRawTensorView,
    pub len: usize,
}

impl Default for PieLoaderRawTensorSlice {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

/// An opened checkpoint.
///
/// Freed with [`pie_loader_close_checkpoint`]. Immutable between open and
/// close, so it may be read from several threads — which is the case that
/// matters, because the ranks of a TP group compile in parallel from one
/// checkpoint.
///
/// [`pie_loader_close_checkpoint`]: crate::ffi::entry::pie_loader_close_checkpoint
#[repr(C)]
#[derive(Debug)]
pub struct PieLoaderCheckpoint {
    pub files: PieLoaderCheckpointFileSlice,
    /// Every tensor in every file, in the order the reader found them.
    pub tensors: PieLoaderRawTensorSlice,
    owner: *mut std::ffi::c_void,
}

/// Owns the backing storage for one opened checkpoint, and the Rust metadata
/// the compiler actually consumes.
///
/// Keeping `metadata` here rather than re-deriving it from the POD views is
/// what makes the handle a single source of truth: the contract the driver
/// writes and the plan the loader compiles are about the same parse.
pub(super) struct CheckpointArena {
    pub(super) metadata: CheckpointMetadata,
    pub(super) snapshot_dir: PathBuf,
    strings: Vec<Box<[u8]>>,
    shapes: Vec<Box<[i64]>>,
    files: Vec<PieLoaderCheckpointFileView>,
    tensors: Vec<PieLoaderRawTensorView>,
}

impl CheckpointArena {
    fn store_str(&mut self, value: &str) -> PieLoaderBytes {
        let boxed: Box<[u8]> = value.as_bytes().to_vec().into_boxed_slice();
        let view = PieLoaderBytes {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.strings.push(boxed);
        view
    }

    fn store_i64(&mut self, values: &[i64]) -> PieLoaderI64Slice {
        let boxed: Box<[i64]> = values.to_vec().into_boxed_slice();
        let view = PieLoaderI64Slice {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.shapes.push(boxed);
        view
    }
}

/// Marshal a parsed checkpoint into a handle.
pub(super) fn build(
    metadata: CheckpointMetadata,
    snapshot_dir: PathBuf,
) -> *mut PieLoaderCheckpoint {
    let mut arena = CheckpointArena {
        metadata,
        snapshot_dir,
        strings: Vec::new(),
        shapes: Vec::new(),
        files: Vec::new(),
        tensors: Vec::new(),
    };

    let files = arena.metadata.files.clone();
    for file in &files {
        let path = arena.store_str(&file.path);
        arena.files.push(PieLoaderCheckpointFileView {
            id: file.id.0,
            path,
            size_bytes: file.size_bytes,
            format: PieLoaderCheckpointFormat::from(file.format),
        });
    }

    // `weights()`, not `tensors`: what crosses this boundary is what a driver
    // will plan, copy and upload. A pie artifact stores its compiled tokenizer
    // and model descriptor as `dense` `u8` objects under `__meta__/`, which are
    // indistinguishable from raw `u8` weights except by name, and no driver
    // knows the prefix -- neither this repo's Metal nor its CUDA schema
    // mentions it. Handing them over made every family contract fail on
    // `__meta__/model/descriptor` the first time it opened a `.zt`, and the
    // alternative -- teaching each family to skip a prefix that is not part of
    // any model -- puts container bookkeeping in the one place that is supposed
    // to be nothing but the family's tensor names.
    //
    // Nothing is lost: no FFI entry point exposes metadata objects, and the
    // consumers that want them (`worker::weights`, `pie model convert`) read
    // them from the Rust `Checkpoint` directly via `meta_object`.
    let tensors: Vec<_> = arena.metadata.weights().cloned().collect();
    for tensor in &tensors {
        let name = arena.store_str(&tensor.name);
        let shape = arena.store_i64(&tensor.shape);
        let encoding = write_encoding(&tensor.encoding);
        arena.tensors.push(PieLoaderRawTensorView {
            id: tensor.id.0,
            name,
            file_id: tensor.file_id.0,
            file_offset: tensor.file_offset,
            span_bytes: tensor.span_bytes,
            shape,
            encoding,
        });
    }

    let header = PieLoaderCheckpoint {
        files: PieLoaderCheckpointFileSlice {
            ptr: arena.files.as_ptr(),
            len: arena.files.len(),
        },
        tensors: PieLoaderRawTensorSlice {
            ptr: arena.tensors.as_ptr(),
            len: arena.tensors.len(),
        },
        owner: std::ptr::null_mut(),
    };
    let owner = Box::into_raw(Box::new(arena)).cast::<std::ffi::c_void>();
    Box::into_raw(Box::new(PieLoaderCheckpoint { owner, ..header }))
}

/// Recover the arena a handle points at.
///
/// # Safety
///
/// `handle` is a live pointer from [`pie_loader_open_checkpoint`].
pub(super) unsafe fn arena_of<'a>(handle: *const PieLoaderCheckpoint) -> &'a CheckpointArena {
    unsafe { &*(*handle).owner.cast::<CheckpointArena>() }
}

/// Free a handle and everything it points at.
///
/// # Safety
///
/// `handle` is null, or a pointer from [`pie_loader_open_checkpoint`] that has
/// not already been closed.
pub(super) unsafe fn release(handle: *mut PieLoaderCheckpoint) {
    if handle.is_null() {
        return;
    }
    let boxed = unsafe { Box::from_raw(handle) };
    if !boxed.owner.is_null() {
        drop(unsafe { Box::from_raw(boxed.owner.cast::<CheckpointArena>()) });
    }
}
