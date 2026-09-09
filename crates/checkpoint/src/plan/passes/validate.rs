use std::collections::HashMap;

use crate::error::{Error, OrOverflow, Result};
use crate::plan::geometry::extent_storage_bytes;
use crate::plan::index::instr_by_id;
use crate::plan::{LoadPlan, StorageInstr, TileMapKind};
use crate::types::{
    BackendKind, BufferId, Encoding, QuantScheme, RepackLayout, TensorId, Visibility,
};

pub(super) fn validate_fill_order(program: &mut LoadPlan) -> Result<usize> {
    let mut filled: HashMap<BufferId, usize> = HashMap::new();
    for (at, id) in program.schedule.iter().enumerate() {
        if let StorageInstr::Fill { buffer, .. } = instr_by_id(&program.instrs, *id)? {
            filled.insert(*buffer, at);
        }
    }
    if filled.is_empty() {
        return Ok(0);
    }

    let mut windows = Vec::new();
    for (buffer, fill_at) in &filled {
        let decl = program.buffer(*buffer)?;
        let Some(start) = decl.persistent_offset else {
            continue;
        };
        windows.push(FilledWindow {
            start,
            end: start
                .checked_add(decl.bytes)
                .or_overflow("filled persistent buffer window overflow")?,
            buffer: *buffer,
            fill_at: *fill_at,
        });
    }
    windows.sort_by_key(|window| (window.start, window.buffer.0));

    for (at, id) in program.schedule.iter().enumerate() {
        let instr = instr_by_id(&program.instrs, *id)?;

        let named: &[BufferId] = match instr {
            StorageInstr::ExtentWrite { dest, .. } | StorageInstr::GatherWrite { dest, .. } => {
                std::slice::from_ref(&dest.buffer)
            }
            StorageInstr::TileMap { outputs, .. } => outputs.as_slice(),
            _ => &[],
        };
        for buffer in named {
            if let Some(fill_at) = filled.get(buffer)
                && *fill_at > at
            {
                return Err(late_fill(*buffer, at, *fill_at));
            }
        }

        if let StorageInstr::BulkExtentWrite {
            source,
            dest_offset,
            ..
        } = instr
        {
            check_arena_write(&windows, *dest_offset, source.span_bytes, at)?;
        }
    }
    Ok(0)
}

struct FilledWindow {
    start: u64,
    end: u64,
    buffer: BufferId,
    fill_at: usize,
}

fn check_arena_write(
    windows: &[FilledWindow],
    dest_offset: u64,
    bytes: u64,
    at: usize,
) -> Result<()> {
    let end = dest_offset
        .checked_add(bytes)
        .or_overflow("arena write window overflow")?;
    for window in windows {
        if dest_offset < window.end && window.start < end && window.fill_at > at {
            return Err(late_fill(window.buffer, at, window.fill_at));
        }
    }
    Ok(())
}

fn late_fill(buffer: BufferId, at: usize, fill_at: usize) -> Error {
    Error::Internal(format!(
        "buffer {} is written at step {at} but not zeroed until step {fill_at}",
        buffer.0
    ))
}

pub(super) fn validate_target_support(program: &mut LoadPlan) -> Result<usize> {
    for instr in &program.instrs {
        let StorageInstr::TileMap {
            kind, transform, ..
        } = instr
        else {
            continue;
        };
        let advertised = program.target.tile_map_mask & kind.capability_bit() != 0;
        let supported = advertised
            && (matches!(
                kind,
                TileMapKind::Cast
                    | TileMapKind::Reblock
                    | TileMapKind::Scale
                    | TileMapKind::Bias
                    | TileMapKind::Unary
            ) || (*kind == TileMapKind::Encode
                && matches!(
                    transform.to,
                    Some(
                        QuantScheme::Fp8E4M3
                            | QuantScheme::Int8Symmetric
                            | QuantScheme::Mxfp4E2M1E8M0
                            | QuantScheme::MlxAffineU4
                    )
                ))
                || (*kind == TileMapKind::Decode
                    && transform
                        .from
                        .is_some_and(|scheme| scheme.is_self_contained()))
                || (*kind == TileMapKind::Repack
                    && program.target.native_mxfp4_moe
                    && transform.repack.is_some_and(|repack| {
                        matches!(
                            repack.layout,
                            RepackLayout::MarlinMxfp4Weight | RepackLayout::MarlinMxfp4Scale
                        )
                    }))
                || (*kind == TileMapKind::Repack
                    && transform.repack.is_some_and(|repack| {
                        matches!(
                            repack.layout,
                            RepackLayout::TiledAffineU4Weight | RepackLayout::TiledAffineFactor
                        )
                    })));
        if !supported {
            if *kind == TileMapKind::Encode {
                return Err(Error::Unsupported(format!(
                    "this load would quantize on the way in ({:?}->{:?}), and a serving \
                     plan does not convert: the stored form IS the served form. Run \
                     `pie model import` on the source checkpoint — the encode runs there, \
                     once, and the artifact it writes holds the codes this load wants.",
                    transform.from, transform.to
                )));
            }
            if *kind == TileMapKind::Unary {
                return Err(Error::Unsupported(format!(
                    "this load would apply {:?} on the way in, and a serving plan does \
                     not: the function is of the checkpoint's values, so it is paid once \
                     per weight rather than once per boot. Run `pie model import` on the \
                     source checkpoint — the function runs there, and the artifact it \
                     writes holds the values this load reads.",
                    transform.unary
                )));
            }
            if *kind == TileMapKind::Repack {
                return Err(Error::Unsupported(format!(
                    "this load would relayout a weight plane on the way in ({:?}), and a \
                     serving plan does not: a repack is paid once per weight, not once per \
                     boot. Run `pie model import` on the source checkpoint — the \
                     relabelling runs there, and the artifact it writes holds the plane in \
                     the order the kernel reads.",
                    transform.repack.map(|repack| repack.layout)
                )));
            }
            return Err(Error::Unsupported(format!(
                "{:?} target does not support {:?} TileMap ({:?}->{:?})",
                program.target.backend, kind, transform.from, transform.to
            )));
        }
    }
    Ok(0)
}

pub(super) fn validate_bound_encodings(program: &mut LoadPlan) -> Result<usize> {
    if program.target.backend == BackendKind::Unknown {
        return Ok(0);
    }
    for tensor in &program.tensors {
        if tensor.visibility != Visibility::Public {
            continue;
        }
        let Encoding::Quant(spec) = &tensor.encoding else {
            continue;
        };
        if spec.scheme.is_self_contained() && !binds_block(program.target.backend, spec.scheme) {
            return Err(Error::Unsupported(format!(
                "{}: a {:?} target would bind `{}` as {:?}, whose scales live \
                 inside the payload and which no kernel of that backend reads. \
                 It has to be decoded, and this plan does not decode it",
                tensor.name, program.target.backend, tensor.name, spec.scheme
            )));
        }
    }
    Ok(0)
}

fn binds_block(backend: BackendKind, scheme: QuantScheme) -> bool {
    match backend {
        BackendKind::Cuda | BackendKind::Vulkan | BackendKind::Wgpu => matches!(
            scheme,
            QuantScheme::GgufQ2K
                | QuantScheme::GgufQ3K
                | QuantScheme::GgufQ4K
                | QuantScheme::GgufQ5K
                | QuantScheme::GgufQ6K
        ),
        BackendKind::Metal | BackendKind::Unknown => false,
    }
}

pub(super) fn validate_scale_factors(program: &mut LoadPlan) -> Result<usize> {
    for instr in &program.instrs {
        let StorageInstr::TileMap {
            kind,
            source,
            inputs,
            transform,
            ..
        } = instr
        else {
            continue;
        };
        if *kind != TileMapKind::Scale || transform.scale_blocks.is_empty() {
            continue;
        }
        let wanted = 1 + usize::from(source.is_none());
        if inputs.len() != wanted {
            return Err(Error::Contract(format!(
                "per-group Scale has {} input operands, expected {wanted} \
                 (payload then factors)",
                inputs.len()
            )));
        }
    }
    Ok(0)
}

pub(super) fn validate_kernel_operands(program: &mut LoadPlan) -> Result<usize> {
    for instr in &program.instrs {
        let StorageInstr::TileMap {
            kind,
            source,
            dest,
            inputs,
            outputs,
            transform,
            ..
        } = instr
        else {
            continue;
        };
        let Some(kernel) = transform.kernel.as_deref() else {
            continue;
        };
        if source.is_some() {
            return Err(Error::Contract(format!(
                "{kind:?} names kernel `{kernel}` but reads the checkpoint, whose \
                 bytes are on the host; stage-device-transforms should have given \
                 it a buffer to read"
            )));
        }
        let mut operands: Vec<BufferId> = inputs.iter().chain(outputs).copied().collect();
        operands.extend(dest.as_ref().map(|dest| dest.buffer));
        for operand in operands {
            if !in_arena(program, operand)? {
                return Err(Error::Contract(format!(
                    "{kind:?} names kernel `{kernel}` but operand buffer {} has no \
                     arena offset, so a backing could not be told where it is",
                    operand.0
                )));
            }
        }
    }
    Ok(0)
}

fn in_arena(program: &LoadPlan, id: BufferId) -> Result<bool> {
    let mut id = id;
    for _ in 0..MAX_VIEW_HOPS {
        let decl = program.buffer(id)?;
        if decl.arena_offset().is_some() {
            return Ok(true);
        }
        let base = program.instrs.iter().find_map(|instr| match instr {
            StorageInstr::CreateView { input, output, .. } if *output == id => Some(*input),
            _ => None,
        });
        match base {
            Some(base) => id = base,
            None => return Ok(false),
        }
    }
    Ok(false)
}

const MAX_VIEW_HOPS: usize = 16;

pub(super) fn validate_persistent_layout(program: &mut LoadPlan) -> Result<usize> {
    let mut claimed: HashMap<TensorId, BufferId> = HashMap::new();
    for buffer in &program.buffers {
        let Some(tensor) = buffer.tensor else {
            continue;
        };
        if let Some(first) = claimed.insert(tensor, buffer.id) {
            return Err(Error::Contract(format!(
                "buffers {} and {} both claim tensor {}",
                first.0, buffer.id.0, tensor.0
            )));
        }
    }
    let mut spans: Vec<(u64, u64, u32)> = Vec::new();
    for buffer in &program.buffers {
        let Some(offset) = buffer.persistent_offset else {
            continue;
        };
        let alignment = u64::from(
            buffer
                .alignment
                .max(program.target.preferred_alignment)
                .max(1),
        );
        if offset % alignment != 0 {
            return Err(Error::Contract(format!(
                "persistent buffer {} base offset {} violates operand alignment {}",
                buffer.id.0, offset, alignment
            )));
        }
        let end = offset
            .checked_add(buffer.bytes)
            .or_overflow("persistent arena offset overflow")?;
        spans.push((offset, end, buffer.id.0));
    }
    spans.sort_by_key(|span| span.0);
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(Error::Contract(format!(
                "persistent buffers {} and {} overlap in the arena: [{}, {}) vs [{}, {})",
                pair[0].2, pair[1].2, pair[0].0, pair[0].1, pair[1].0, pair[1].1
            )));
        }
    }
    for instr in &program.instrs {
        let StorageInstr::CreateView { input, view, .. } = instr else {
            continue;
        };
        let backing = program.buffer(*input)?;
        let extent = extent_storage_bytes(&view.stride)?;
        let end = view
            .offset
            .checked_add(extent)
            .or_overflow("CreateView window overflow")?;
        if end > backing.bytes {
            return Err(Error::Contract(format!(
                "CreateView window [{}, {}) escapes backing buffer {} ({} bytes)",
                view.offset, end, backing.id.0, backing.bytes
            )));
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::binds_block;
    use crate::types::{BackendKind, QuantScheme};

    #[test]
    fn validate_every_case() {
        the_k_quants_bind_on_every_backend_with_a_stored_block_point();
        the_small_blocks_and_the_lattices_bind_nowhere();
    }

    fn the_k_quants_bind_on_every_backend_with_a_stored_block_point() {
        let k_quants = [
            QuantScheme::GgufQ2K,
            QuantScheme::GgufQ3K,
            QuantScheme::GgufQ4K,
            QuantScheme::GgufQ5K,
            QuantScheme::GgufQ6K,
        ];
        for scheme in k_quants {
            for backend in [BackendKind::Cuda, BackendKind::Vulkan, BackendKind::Wgpu] {
                assert!(
                    binds_block(backend, scheme),
                    "{backend:?} has a {scheme:?} point and should bind it as stored"
                );
            }
            assert!(
                !binds_block(BackendKind::Metal, scheme),
                "Metal has no stored-block point and must decode {scheme:?}"
            );
        }
    }

    fn the_small_blocks_and_the_lattices_bind_nowhere() {
        for scheme in [
            QuantScheme::GgufQ4_0,
            QuantScheme::GgufQ4_1,
            QuantScheme::GgufQ5_0,
            QuantScheme::GgufQ5_1,
            QuantScheme::GgufQ8_0,
            QuantScheme::GgufIq4Nl,
            QuantScheme::GgufIq4Xs,
        ] {
            for backend in [
                BackendKind::Cuda,
                BackendKind::Vulkan,
                BackendKind::Wgpu,
                BackendKind::Metal,
            ] {
                assert!(
                    !binds_block(backend, scheme),
                    "{backend:?} reads no {scheme:?} block as stored"
                );
            }
        }
    }
}
