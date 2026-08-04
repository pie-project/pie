//! The pass pipeline's own checks.
//!
//! These test the passes against a hand-built plan rather than through a
//! contract, which is the point: a plan can be malformed in ways no contract
//! can express, and the validators exist for exactly those.

use super::rewrite::try_merge_bulk_extent_write;
use super::validate::{validate_fill_order, validate_persistent_layout, validate_target_support};
use crate::extent::Extent;
use crate::plan::pass::{Pass, Stage, run_passes};
use crate::plan::{
    BufferDecl, DestExtent, LoadPlan, SourceExtent, StorageInstr, StorageTarget, TileMapKind,
    TileSpec, TransformSpec,
};
use crate::types::{BackendKind, BufferId, DType, FileId, InstrId, TensorId};

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
            dtype: DType::U8,
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
    assert!(validate_target_support(&mut metal).is_ok());
    // Advertising a kind is not advertising every transform inside it: the
    // Metal executor encodes to MLX affine U4 and nothing else, and a plan that
    // asked it for the default (`None`) scheme would find no encoder.
    metal.instrs[0] = tile(TileMapKind::Encode);
    assert!(validate_target_support(&mut metal).is_err());
    metal.instrs[0] = tile(TileMapKind::Repack);
    assert!(validate_target_support(&mut metal).is_err());
}

/// Two persistent buffers, so that an arena-relative write has to be matched to
/// the one it actually lands in.
///
/// `A` owns `[0, 256)` and `B` owns `[256, 512)`. The instruction table is
/// fixed and dense — fill `A`, fill `B`, then the write — and only the
/// *schedule* varies, which is the thing the invariant is about.
fn fill_order_plan(write: StorageInstr, schedule: &[u32]) -> LoadPlan {
    let mut plan = program_with(vec![
        operand(0, 256, 1, Some(0)),
        operand(1, 256, 1, Some(256)),
    ]);
    plan.instrs = vec![
        StorageInstr::Fill {
            id: InstrId(0),
            buffer: BufferId(0),
        },
        StorageInstr::Fill {
            id: InstrId(1),
            buffer: BufferId(1),
        },
        write,
    ];
    plan.schedule = schedule.iter().map(|id| InstrId(*id)).collect();
    plan
}

/// A bulk write into `B`'s arena window, as the coalesce pass emits it.
fn bulk_into_b() -> StorageInstr {
    StorageInstr::BulkExtentWrite {
        id: InstrId(2),
        source: SourceExtent {
            file_id: FileId(0),
            tensor_id: TensorId(0),
            file_offset: 0,
            span_bytes: 256,
            stride: Extent::byte_run(256),
            dtype: DType::U8,
        },
        dest_offset: 256,
    }
}

/// The write lands in `B`, and `B` is zeroed after it — the fill would eat the
/// bytes just copied.
///
/// This is the case the check used to decide by picking an arbitrary key out of
/// a `HashMap`: with `A` zeroed first and `B` last, whichever of the two the
/// iterator happened to yield decided the answer, so the pass caught this about
/// half the time and was reproducible neither way.
#[test]
fn rejects_a_bulk_write_into_a_buffer_zeroed_after_it() {
    let mut plan = fill_order_plan(bulk_into_b(), &[0, 2, 1]);
    assert!(validate_fill_order(&mut plan).is_err());
}

/// The mirror: `A` is zeroed late, but the write goes to `B`, which was zeroed
/// first. Nothing is wrong, and a check that matched by identity rather than by
/// overlap would reject a correct plan.
#[test]
fn accepts_a_bulk_write_beside_a_buffer_zeroed_after_it() {
    let mut plan = fill_order_plan(bulk_into_b(), &[1, 2, 0]);
    assert!(validate_fill_order(&mut plan).is_ok());
}

/// The pipeline is partitioned: every rewrite, then every check.
///
/// A validator proves something about the plan the compiler hands back, and
/// that proof survives only if nothing rewrites afterwards. The property is
/// invisible in `all()` — it is just the order the lines happen to be in — so
/// it is asserted here and enforced in `run_passes`.
#[test]
fn every_rewrite_comes_before_every_check() {
    let stages: Vec<_> = super::all().iter().map(|pass| pass.stage).collect();
    let first_check = stages.iter().position(|s| *s == Stage::Check);
    assert!(first_check.is_some(), "the pipeline has no validators");
    assert!(
        stages[first_check.unwrap()..]
            .iter()
            .all(|s| *s == Stage::Check),
        "a rewrite runs after a validator: {:?}",
        super::all()
            .iter()
            .map(|p| (p.name, p.stage))
            .collect::<Vec<_>>()
    );
}

/// And the rule is enforced, not merely asserted about today's list.
#[test]
fn a_rewrite_scheduled_after_a_check_is_refused() {
    fn nothing(_: &mut LoadPlan) -> crate::error::Result<usize> {
        Ok(0)
    }
    let bad = [
        Pass {
            name: "check",
            stage: Stage::Check,
            run: nothing,
        },
        Pass {
            name: "late-rewrite",
            stage: Stage::Rewrite,
            run: nothing,
        },
    ];
    let mut plan = fill_order_plan(bulk_into_b(), &[0, 1, 2]);
    let err = run_passes(&mut plan, &bad).unwrap_err().to_string();
    assert!(err.contains("late-rewrite"), "{err}");
}

/// A validator that reports a rewrite is not a validator.
#[test]
fn a_check_that_reports_a_rewrite_is_refused() {
    fn rewrote(_: &mut LoadPlan) -> crate::error::Result<usize> {
        Ok(1)
    }
    let bad = [Pass {
        name: "lying-check",
        stage: Stage::Check,
        run: rewrote,
    }];
    let mut plan = fill_order_plan(bulk_into_b(), &[0, 1, 2]);
    let err = run_passes(&mut plan, &bad).unwrap_err().to_string();
    assert!(err.contains("lying-check"), "{err}");
}
