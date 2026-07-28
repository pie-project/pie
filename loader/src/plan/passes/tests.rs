//! The pass pipeline's own checks.
//!
//! These test the passes against a hand-built plan rather than through a
//! contract, which is the point: a plan can be malformed in ways no contract
//! can express, and the validators exist for exactly those.

use super::rewrite::try_merge_bulk_extent_write;
use super::validate::{validate_persistent_layout, validate_target_support};
use crate::extent::Extent;
use crate::plan::{
    BufferDecl, DestExtent, LoadPlan, SourceExtent, StorageInstr, StorageTarget, TileMapKind,
    TileSpec, TransformSpec,
};
use crate::types::{BackendKind, BufferId, FileId, InstrId, TensorId};

fn operand(id: u32, bytes: u64, alignment: u32, offset: Option<u64>) -> BufferDecl {
    BufferDecl {
        id: BufferId(id),
        tensor: Some(TensorId(id)),
        bytes,
        alignment,
        temporary: false,
        persistent_offset: offset,
    }
}

fn program_with(buffers: Vec<BufferDecl>) -> LoadPlan {
    let mut plan = LoadPlan::empty(StorageTarget {
        preferred_alignment: 256,
        ..StorageTarget::default()
    });
    plan.buffers = buffers;
    plan
}

#[test]
fn accepts_aligned_disjoint_operands() {
    let mut plan = program_with(vec![
        operand(0, 256, 1, Some(0)),
        operand(1, 256, 1, Some(256)),
    ]);
    assert!(validate_persistent_layout(&mut plan).is_ok());
}

#[test]
fn rejects_misaligned_operand_base() {
    // 128 is not a multiple of the fixture target's 256-byte alignment.
    let mut plan = program_with(vec![operand(0, 64, 1, Some(128))]);
    assert!(validate_persistent_layout(&mut plan).is_err());
}

#[test]
fn rejects_overlapping_operands() {
    // [0,512) and [256,512) overlap; both bases are 256-aligned.
    let mut plan = program_with(vec![
        operand(0, 512, 1, Some(0)),
        operand(1, 256, 1, Some(256)),
    ]);
    assert!(validate_persistent_layout(&mut plan).is_err());
}

#[test]
fn rejects_view_escaping_backing() {
    let mut plan = program_with(vec![operand(0, 64, 256, Some(0))]);
    plan.instrs.push(StorageInstr::CreateView {
        id: InstrId(0),
        input: BufferId(0),
        output: BufferId(1),
        view: DestExtent {
            buffer: BufferId(1),
            offset: 32,
            stride: Extent::byte_run(64),
        },
    });
    // Window [32, 96) escapes the 64-byte backing buffer.
    assert!(validate_persistent_layout(&mut plan).is_err());
}

#[test]
fn bulk_merge_respects_target_tile_bound() {
    let make = |id, file_offset, dest_offset| StorageInstr::BulkExtentWrite {
        id: InstrId(id),
        source: SourceExtent {
            file_id: FileId(0),
            tensor_id: TensorId(id),
            file_offset,
            span_bytes: 8,
            stride: Extent::byte_run(8),
        },
        dest_offset,
    };
    let mut first = make(0, 0, 0);
    let second = make(1, 8, 8);
    assert!(!try_merge_bulk_extent_write(&mut first, &second, 8).unwrap());
    assert!(try_merge_bulk_extent_write(&mut first, &second, 16).unwrap());
}

#[test]
fn target_transform_matrix_matches_host_and_metal_executors() {
    let tile = |kind| StorageInstr::TileMap {
        id: InstrId(0),
        kind,
        source: None,
        dest: None,
        inputs: Vec::new(),
        outputs: Vec::new(),
        tile: TileSpec {
            max_tile_bytes: 1,
            rows_per_tile: 0,
        },
        transform: TransformSpec::default(),
    };

    let mut host = LoadPlan::empty(StorageTarget::default());
    host.instrs.push(tile(TileMapKind::Cast));
    assert!(validate_target_support(&mut host).is_ok());
    host.instrs[0] = tile(TileMapKind::Transcode);
    assert!(validate_target_support(&mut host).is_err());

    let mut metal = LoadPlan::empty(StorageTarget {
        backend: BackendKind::Metal,
        tile_map_mask: crate::plan::METAL_TILE_MAP_MASK,
        ..StorageTarget::default()
    });
    metal.instrs.push(tile(TileMapKind::Cast));
    assert!(validate_target_support(&mut metal).is_err());
}
