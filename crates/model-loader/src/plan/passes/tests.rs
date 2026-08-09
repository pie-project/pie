//! The pass pipeline's own checks.
//!
//! These test the passes against a hand-built plan rather than through a
//! contract, which is the point: a plan can be malformed in ways no contract
//! can express, and the validators exist for exactly those.

use super::rewrite::try_merge_bulk_extent_write;
use super::validate::{
    validate_fill_order, validate_kernel_operands, validate_persistent_layout,
    validate_target_support,
};
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
        ty: crate::contract::TensorType::raw(vec![bytes as i64], DType::U8),
        bytes,
        alignment,
        temporary: false,
        persistent_offset: offset,
        scratch_offset: None,
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

// ---------------------------------------------------------------------------
// The scratch region: what a slot may be shared with, and what it may not.
//
// This is the allocator whose mistakes are silent. Two operands placed at one
// offset while both are live is not a crash and not a refusal — it is a weight
// that loads and holds the other one's bytes. So the reuse rule is tested
// directly rather than inferred from a plan that happens not to provoke it.
// ---------------------------------------------------------------------------

/// A host-owned buffer: something `place_in_scratch` may place.
fn host_operand(id: u32, bytes: u64) -> BufferDecl {
    BufferDecl {
        temporary: true,
        tensor: None,
        ..operand(id, bytes, 1, None)
    }
}

/// A schedule that touches `buffer` at each of `steps`, padded with fills of a
/// resident buffer so the positions are what the test says they are.
fn schedule_touching(plan: &mut LoadPlan, steps: &[(u32, usize)], length: usize) {
    let mut instrs: Vec<StorageInstr> = (0..length)
        .map(|at| StorageInstr::Allocate {
            id: InstrId(u32::try_from(at).unwrap()),
            // Buffer 0 is the resident one every fixture below declares; an
            // Allocate of it is an instruction that names nothing else.
            buffer: BufferId(0),
        })
        .collect();
    for (buffer, at) in steps {
        instrs[*at] = StorageInstr::ExtentWrite {
            id: InstrId(u32::try_from(*at).unwrap()),
            source: SourceExtent {
                file_id: FileId(0),
                tensor_id: TensorId(0),
                file_offset: 0,
                span_bytes: 8,
                stride: Extent::byte_run(8),
                dtype: DType::U8,
            },
            dest: DestExtent {
                buffer: BufferId(*buffer),
                offset: 0,
                stride: Extent::byte_run(8),
            },
        };
    }
    plan.schedule = (0..length)
        .map(|at| InstrId(u32::try_from(at).unwrap()))
        .collect();
    plan.instrs = instrs;
}

#[test]
fn scratch_slots_are_reused_by_operands_that_are_never_live_together() {
    // The property `llama_dense_cuda_runtime_fp8` depends on: fourteen staged
    // operands, one slot, because each is written immediately before the
    // transform that reads it. Summing them instead would ask the caller for
    // 2.4 MB where 360 kB is enough — and for a real checkpoint, for a second
    // model's worth of device memory.
    let mut plan = program_with(vec![
        operand(0, 1024, 1, Some(0)),
        host_operand(1, 512),
        host_operand(2, 256),
    ]);
    schedule_touching(&mut plan, &[(1, 2), (2, 6)], 8);
    let placed = super::arena::place_in_scratch(&mut plan, &[BufferId(1), BufferId(2)]).unwrap();

    assert_eq!(placed, 2);
    let (first, second) = (
        plan.buffers[1].scratch_offset.unwrap(),
        plan.buffers[2].scratch_offset.unwrap(),
    );
    assert_eq!(first, second, "disjoint live ranges share a slot");
    assert!(
        first >= 1024,
        "and the slot sits behind the resident tensors, not inside them"
    );
}

#[test]
fn two_operands_live_at_once_never_share_bytes() {
    // The same allocator, asked the question it must not get wrong. Buffer 1
    // is touched at steps 2 and 6, buffer 2 at step 4 — so 2 is live inside
    // 1's range and the two cannot be at one address.
    let mut plan = program_with(vec![
        operand(0, 1024, 1, Some(0)),
        host_operand(1, 512),
        host_operand(2, 256),
    ]);
    schedule_touching(&mut plan, &[(1, 2), (2, 4), (1, 6)], 8);
    super::arena::place_in_scratch(&mut plan, &[BufferId(1), BufferId(2)]).unwrap();

    let (a, a_end) = {
        let d = &plan.buffers[1];
        let o = d.scratch_offset.unwrap();
        (o, o + d.bytes)
    };
    let (b, b_end) = {
        let d = &plan.buffers[2];
        let o = d.scratch_offset.unwrap();
        (o, o + d.bytes)
    };
    assert!(
        a >= b_end || b >= a_end,
        "overlapping lives were placed at overlapping bytes: [{a}, {a_end}) and \
         [{b}, {b_end})"
    );
    // And the arena grew by exactly what the two need, aligned — not by the
    // model.
    plan.memory.persistent_bytes = 1024;
    super::memory::recompute_memory_plan(&mut plan).unwrap();
    assert_eq!(plan.memory.arena_bytes(), a_end.max(b_end));
}

#[test]
fn a_buffer_that_gets_zeroed_is_left_on_the_host() {
    // `hoist-bulk-arena-writes` runs after placement and lifts every `Fill`
    // into a prologue. A fill that sat between two users of one slot would end
    // up in front of both, so a filled buffer must not be given a shared slot
    // at all — the transform reading it loses its kernel, which is the
    // conservative half of a rule whose other half is silent corruption.
    let mut plan = program_with(vec![operand(0, 1024, 1, Some(0)), host_operand(1, 512)]);
    schedule_touching(&mut plan, &[(1, 2)], 4);
    plan.instrs[1] = StorageInstr::Fill {
        id: InstrId(1),
        buffer: BufferId(1),
    };
    let placed = super::arena::place_in_scratch(&mut plan, &[BufferId(1)]).unwrap();
    assert_eq!(placed, 0);
    assert!(plan.buffers[1].scratch_offset.is_none());
}

#[test]
fn a_view_is_placed_through_the_buffer_it_windows() {
    // A view owns no bytes, so what has to go in the arena is its BASE.
    // Placing the view instead is a no-op that leaves the operand on the host
    // — invisible in the plan and total at run time.
    let mut plan = program_with(vec![
        operand(0, 1024, 1, Some(0)),
        host_operand(1, 512),
        BufferDecl {
            bytes: 0,
            ..host_operand(2, 0)
        },
    ]);
    schedule_touching(&mut plan, &[(1, 2)], 4);
    plan.instrs[3] = StorageInstr::CreateView {
        id: InstrId(3),
        input: BufferId(1),
        output: BufferId(2),
        view: DestExtent {
            buffer: BufferId(2),
            offset: 0,
            stride: Extent::byte_run(256),
        },
    };
    // Asked about the VIEW, and only the view.
    let placed = super::arena::place_in_scratch(&mut plan, &[BufferId(2)]).unwrap();
    assert_eq!(placed, 1);
    assert!(
        plan.buffers[1].scratch_offset.is_some(),
        "the base is what occupies bytes"
    );
    assert!(plan.buffers[2].scratch_offset.is_none());
}

/// A kernel names a device, and a device can only reach the arena.///
/// The refusal §5.4 of `.wiki/fix/loader.md` asks for: a plan that says
/// `kernel = …` over an operand with no arena offset is a plan that will load
/// correctly and about a hundred times slower, and nothing downstream can
/// tell. Compiled plans cannot be in this state — `stage-device-transforms`
/// puts the operands there — so the case has to be built by hand, which is
/// exactly what a validator is for.
#[test]
fn rejects_a_named_kernel_over_a_host_operand() {
    let named = |inputs: Vec<BufferId>| StorageInstr::TileMap {
        id: InstrId(0),
        kind: TileMapKind::Encode,
        source: None,
        dest: None,
        inputs,
        outputs: vec![BufferId(0)],
        tile: TileSpec {
            max_tile_bytes: 1,
            rows_per_tile: 0,
        },
        transform: TransformSpec {
            kernel: Some("quant::quantize_bf16_to_fp8_e4m3_per_channel".to_string()),
            ..TransformSpec::default()
        },
    };

    // Buffer 1 is host-owned: no persistent offset, no scratch offset.
    let mut plan = program_with(vec![operand(0, 256, 1, Some(0)), operand(1, 256, 1, None)]);
    plan.instrs.push(named(vec![BufferId(1)]));
    let refusal = validate_kernel_operands(&mut plan).expect_err("must refuse");
    assert!(
        format!("{refusal}").contains("no arena offset"),
        "the refusal names what is wrong: {refusal}"
    );

    // The same instruction over an operand the arena's SCRATCH region holds is
    // fine, which is what makes the line above a check rather than a ban on
    // kernels.
    let mut staged = program_with(vec![operand(0, 256, 1, Some(0)), operand(1, 256, 1, None)]);
    staged.buffers[1].persistent_offset = None;
    staged.buffers[1].scratch_offset = Some(256);
    staged.instrs.push(named(vec![BufferId(1)]));
    assert!(validate_kernel_operands(&mut staged).is_ok());
}

/// The executor's old policy, as a plan-level fact: a kernel-bearing transform
/// that still reads the checkpoint is refused by name instead of being
/// silently run on the host.
#[test]
fn rejects_a_named_kernel_that_still_reads_the_checkpoint() {
    let mut plan = program_with(vec![operand(0, 256, 1, Some(0))]);
    plan.instrs.push(StorageInstr::TileMap {
        id: InstrId(0),
        kind: TileMapKind::Encode,
        source: Some(SourceExtent {
            file_id: FileId(0),
            tensor_id: TensorId(0),
            file_offset: 0,
            span_bytes: 256,
            stride: Extent::byte_run(256),
            dtype: DType::BF16,
        }),
        dest: None,
        inputs: Vec::new(),
        outputs: vec![BufferId(0)],
        tile: TileSpec {
            max_tile_bytes: 1,
            rows_per_tile: 0,
        },
        transform: TransformSpec {
            kernel: Some("quant::quantize_bf16_to_fp8_e4m3_per_channel".to_string()),
            ..TransformSpec::default()
        },
    });
    let refusal = validate_kernel_operands(&mut plan).expect_err("must refuse");
    assert!(
        format!("{refusal}").contains("reads the checkpoint"),
        "{refusal}"
    );
}

#[test]
fn bulk_merge_respects_target_tile_bound() {    let make = |id, file_offset, dest_offset| StorageInstr::BulkExtentWrite {
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
