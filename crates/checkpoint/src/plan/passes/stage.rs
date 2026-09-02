//! Device-side scratch: gives a transform's operands somewhere on the device
//! to be. A transform reading the checkpoint gets an
//! [`StorageInstr::ExtentWrite`] into a staging buffer and then reads that;
//! an intermediate a device transform reads or writes is placed in the
//! arena's scratch region. Whether a rewrite is worth making is decided by
//! asking [`kernel_for`] about the instruction as it would be once rewritten.
//!
//! [`ArenaBacking::run_tile_map`]: crate::executor::arena::ArenaBacking::run_tile_map

use crate::error::Result;
use crate::extent::Extent;
use crate::plan::index::{PlanIndex, instr_by_id};
use crate::plan::passes::tile::{TileMapFacts, facts_of, kernel_for};
use crate::plan::{BufferDecl, DestExtent, LoadPlan, SourceExtent, StorageInstr};
use crate::types::{BufferId, Encoding, InstrId};

pub(super) fn stage_device_transforms(program: &mut LoadPlan) -> Result<usize> {
    // A target with no kernels has nothing to stage for; skip walking a host plan.
    if program.target.tile_map_mask == 0 {
        return Ok(0);
    }
    let index = PlanIndex::new(program);
    let old = program.instrs.clone();
    let schedule = program.schedule.clone();

    // Pass one: decide, reading only.
    let mut decisions = Vec::with_capacity(schedule.len());
    let mut wanted: Vec<BufferId> = Vec::new();
    for id in &schedule {
        let instr = instr_by_id(&old, *id)?;
        let decision = decide(program, &index, instr)?;
        if let Some(decision) = &decision {
            wanted.extend(decision.arena_operands.iter().copied());
        }
        decisions.push(decision);
    }
    if decisions.iter().all(Option::is_none) {
        return Ok(0);
    }

    // Pass two: rewrite. Each staged transform gains a buffer to read and an
    // `ExtentWrite` that fills it, immediately before the transform.
    let mut rewritten: Vec<StorageInstr> = Vec::with_capacity(old.len() + decisions.len() * 2);
    let mut staged = 0usize;
    for (id, decision) in schedule.iter().zip(decisions) {
        let instr = instr_by_id(&old, *id)?.clone();
        let Some(decision) = decision else {
            rewritten.push(instr);
            continue;
        };
        let Some(stage) = decision.stage else {
            rewritten.push(instr);
            continue;
        };
        let buffer = declare_staging_buffer(program, &stage)?;
        wanted.push(buffer);
        rewritten.push(StorageInstr::Allocate {
            id: InstrId(0),
            buffer,
        });
        rewritten.push(StorageInstr::ExtentWrite {
            id: InstrId(0),
            source: stage.source.clone(),
            dest: DestExtent {
                buffer,
                offset: 0,
                stride: Extent::byte_run(stage.source.span_bytes),
            },
        });
        rewritten.push(read_from_buffer(instr, buffer)?);
        staged += 1;
    }
    super::rewrite::rewrite_program_instrs(program, rewritten)?;
    let placed = super::arena::place_in_scratch(program, &wanted)?;
    Ok(staged + placed)
}

/// What this pass decided about one instruction.
struct Decision {
    /// The checkpoint extent to stage, or `None` when the transform already
    /// reads a buffer and only needs that buffer placed.
    stage: Option<Staging>,
    /// Operands that have to be arena spans for the backing to be offered
    /// this transform at all.
    arena_operands: Vec<BufferId>,
}

/// A checkpoint extent, and the buffer type the bytes will be read back as.
struct Staging {
    source: SourceExtent,
    shape: Vec<i64>,
    encoding: Encoding,
    alignment: u32,
}

/// Would the device run this transform, and what does that need?
///
/// `None` means "leave it alone", which is the answer for every transform the
/// target has no row for — the host runs those and always did.
fn decide(program: &LoadPlan, index: &PlanIndex, instr: &StorageInstr) -> Result<Option<Decision>> {
    let StorageInstr::TileMap {
        source,
        dest,
        inputs,
        outputs,
        transform,
        ..
    } = instr
    else {
        return Ok(None);
    };
    let Some(facts) = facts_of(program, index, instr) else {
        return Ok(None);
    };
    let operands = || {
        let mut all: Vec<BufferId> = inputs.iter().chain(outputs).copied().collect();
        all.extend(dest.as_ref().map(|dest| dest.buffer));
        all
    };

    let Some(source) = source else {
        // Already reads a buffer; only question is whether it's device-addressable.
        let facts = TileMapFacts {
            operands_in_arena: true,
            ..facts
        };
        return Ok(kernel_for(&facts, &program.target).map(|_| Decision {
            stage: None,
            arena_operands: operands(),
        }));
    };

    // A block-scaled `Encode` reads its input's per-group factors out of the
    // checkpoint while it works, so its source is not merely bytes to move.
    if transform.metadata_source.is_some() {
        return Ok(None);
    }
    // `SourceExtent::dtype` is the logical dtype (e.g. BF16 for an MXFP4
    // payload), not the raw bytes, so only a raw-encoded source can stage.
    let Some(raw) = index.source(program, source.tensor_id) else {
        return Ok(None);
    };
    if !matches!(raw.encoding, Encoding::Raw(_)) {
        return Ok(None);
    }
    let Some(primary) = outputs.first() else {
        return Ok(None);
    };
    let out = program.buffer(*primary)?;
    // Destination's logical shape when writing a whole buffer, else the
    // extent's own counts for a window write.
    let shape = match dest {
        Some(dest) => dest.stride.dims.iter().map(|dim| dim.count).collect(),
        None => out.ty.shape.clone(),
    };
    let elements = crate::types::tensor_elements(&shape).unwrap_or(0);
    if elements == 0 || elements.saturating_mul(source.dtype.bytes_ceil()) != source.span_bytes {
        // Extent doesn't cover a whole operand of the declared shape.
        return Ok(None);
    }

    // The instruction as it would be: reading a dense buffer of exactly these
    // bytes rather than a file.
    let staged_facts = TileMapFacts {
        has_source: false,
        compact_source: true,
        source_dtype: Some(source.dtype),
        // in_place: a staging buffer is never the destination.
        // operands_in_arena: what this pass is making true.
        in_place: false,
        operands_in_arena: true,
        ..facts
    };
    if kernel_for(&staged_facts, &program.target).is_none() {
        return Ok(None);
    }
    Ok(Some(Decision {
        stage: Some(Staging {
            source: source.clone(),
            shape,
            encoding: Encoding::Raw(source.dtype),
            alignment: out.alignment,
        }),
        arena_operands: operands(),
    }))
}

/// Point a `TileMap` at a buffer instead of at the checkpoint. The staged
/// buffer goes first in `inputs`, where every reader expects the payload.
fn read_from_buffer(instr: StorageInstr, buffer: BufferId) -> Result<StorageInstr> {
    let StorageInstr::TileMap {
        id,
        kind,
        dest,
        mut inputs,
        outputs,
        tile,
        transform,
        ..
    } = instr
    else {
        return Err(crate::error::Error::Internal(
            "stage-device-transforms rewrote something that is not a TileMap".to_string(),
        ));
    };
    inputs.insert(0, buffer);
    Ok(StorageInstr::TileMap {
        id,
        kind,
        source: None,
        dest,
        inputs,
        outputs,
        tile,
        transform,
    })
}

fn declare_staging_buffer(program: &mut LoadPlan, stage: &Staging) -> Result<BufferId> {
    let id = BufferId(
        u32::try_from(program.buffers.len())
            .map_err(|_| crate::error::Error::Contract("too many buffers".to_string()))?,
    );
    program.buffers.push(BufferDecl {
        id,
        // Not a tensor: a staging buffer is the arena's, not the contract's.
        tensor: None,
        ty: crate::contract::TensorType::new(stage.shape.clone(), stage.encoding.clone()),
        bytes: stage.source.span_bytes,
        alignment: stage.alignment.max(program.target.preferred_alignment),
        // Reusable, freed at its last use, not part of what the load leaves behind.
        temporary: true,
        persistent_offset: None,
        scratch_offset: None,
    });
    Ok(id)
}
