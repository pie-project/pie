//! Strided-extent geometry: build, narrow and measure the source/dest
//! extents that back every copy instruction, plus plan-wide id lookups.

use super::passes::instr_id_of;
use super::*;

pub(super) fn narrow_repack_source(
    mut source: SourceView,
    spec: RepackSpec,
) -> Result<(SourceView, RepackSpec), CompileError> {
    let mut narrowed = spec;
    let valid_rows = if narrowed.valid_rows == 0 {
        narrowed.target_rows
    } else {
        narrowed.valid_rows
    };
    narrowed.valid_rows = valid_rows;

    if source.shape.len() < 2 {
        return Err(CompileError::InvalidInput(
            "Repack source must have batch and row axes".to_string(),
        ));
    }
    if source.shape[0] != i64::from(narrowed.batch)
        || source.shape[1] != i64::from(narrowed.source_rows)
    {
        return Err(CompileError::InvalidInput(format!(
            "Repack source shape {:?} does not match batch/source_rows {:?}/{}",
            source.shape, narrowed.batch, narrowed.source_rows
        )));
    }

    let (row_start, row_count) = match narrowed.row_map {
        crate::types::RowMap::Identity => (narrowed.source_row_offset, valid_rows),
        crate::types::RowMap::Even | crate::types::RowMap::Odd => {
            let start = narrowed.source_row_offset.checked_mul(2).ok_or_else(|| {
                CompileError::InvalidInput("Repack row offset overflow".to_string())
            })?;
            let rows = valid_rows.checked_mul(2).ok_or_else(|| {
                CompileError::InvalidInput("Repack row count overflow".to_string())
            })?;
            (start, rows)
        }
    };
    if row_start != 0 || row_count != narrowed.source_rows {
        source = narrow_source_axis(source, Axis(1), i64::from(row_start), i64::from(row_count))?;
        narrowed.source_rows = row_count;
        narrowed.source_row_offset = 0;
    }

    match narrowed.layout {
        RepackLayout::MarlinMxfp4Weight => {
            if source.shape.len() != 4 || source.shape[3] != 16 {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Weight Repack source must be [B, R, K/32, 16], got {:?}",
                    source.shape
                )));
            }
            if !narrowed.source_col_offset.is_multiple_of(32)
                || !narrowed.source_cols.is_multiple_of(32)
                || !narrowed.source_stride_cols.is_multiple_of(32)
            {
                return Err(CompileError::InvalidInput(
                    "MarlinMxfp4Weight source narrowing requires 32-wide MXFP4 group alignment"
                        .to_string(),
                ));
            }
            let group_start = narrowed.source_col_offset / 32;
            let group_count = narrowed.source_cols / 32;
            if source.shape[2] != i64::from(narrowed.source_stride_cols / 32) {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Weight source group axis {:?} does not match stride cols {}",
                    source.shape, narrowed.source_stride_cols
                )));
            }
            if group_start != 0 || narrowed.source_cols != narrowed.source_stride_cols {
                source = narrow_source_axis(
                    source,
                    Axis(2),
                    i64::from(group_start),
                    i64::from(group_count),
                )?;
                narrowed.source_stride_cols = narrowed.source_cols;
                narrowed.source_col_offset = 0;
            }
        }
        RepackLayout::MarlinMxfp4Scale => {
            if source.shape.len() != 3 {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Scale Repack source must be [B, R, groups], got {:?}",
                    source.shape
                )));
            }
            if source.shape[2] != i64::from(narrowed.source_stride_cols) {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Scale source group axis {:?} does not match stride cols {}",
                    source.shape, narrowed.source_stride_cols
                )));
            }
            if narrowed.source_col_offset != 0
                || narrowed.source_cols != narrowed.source_stride_cols
            {
                source = narrow_source_axis(
                    source,
                    Axis(2),
                    i64::from(narrowed.source_col_offset),
                    i64::from(narrowed.source_cols),
                )?;
                narrowed.source_stride_cols = narrowed.source_cols;
                narrowed.source_col_offset = 0;
            }
        }
        RepackLayout::DenseRowGather => {
            if source.shape.len() != 2 {
                return Err(CompileError::InvalidInput(format!(
                    "DenseRowGather Repack source must be [B, R], got {:?}",
                    source.shape
                )));
            }
        }
        RepackLayout::None => {}
    }

    Ok((source, narrowed))
}

pub(super) fn buffer_bytes(program: &LoadPlan, id: BufferId) -> Result<u64, CompileError> {
    program
        .buffers
        .get(id.0 as usize)
        .filter(|buffer| buffer.id == id)
        .or_else(|| program.buffers.iter().find(|buffer| buffer.id == id))
        .map(|buffer| buffer.bytes)
        .ok_or_else(|| CompileError::InvalidInput(format!("buffer {} is missing", id.0)))
}

/// Resolve a scheduled instruction by id: index directly when ids are dense (the
/// common case), else fall back to a linear scan. Used by every pass that walks
/// `program.schedule` against an instruction slice.
pub(super) fn instr_by_id(
    instrs: &[StorageInstr],
    id: InstrId,
) -> Result<&StorageInstr, CompileError> {
    instrs
        .get(id.0 as usize)
        .filter(|instr| instr_id_of(instr) == id)
        .or_else(|| instrs.iter().find(|instr| instr_id_of(instr) == id))
        .ok_or_else(|| CompileError::InvalidInput(format!("scheduled instr {} is missing", id.0)))
}

pub(super) fn repack_stage_bytes(spec: RepackSpec) -> Result<u64, CompileError> {
    match spec.layout {
        RepackLayout::MarlinMxfp4Weight => {
            let elems = u64::from(spec.target_rows)
                .checked_mul(u64::from(spec.target_cols))
                .ok_or_else(|| {
                    CompileError::InvalidInput("MXFP4 repack stage size overflow".to_string())
                })?;
            Ok(elems.div_ceil(2))
        }
        RepackLayout::MarlinMxfp4Scale | RepackLayout::DenseRowGather | RepackLayout::None => Ok(0),
    }
}

pub(super) fn extent_storage_bytes(extent: &StridedExtent) -> Result<u64, CompileError> {
    tensor_nbytes(
        &extent.dims.iter().map(|dim| dim.count).collect::<Vec<_>>(),
        u64::from(extent.element_bytes),
    )
    .ok_or_else(|| CompileError::InvalidInput("extent byte size overflow".to_string()))
}

pub(super) fn strided_physical_source_bytes(extent: &StridedExtent) -> Result<u64, CompileError> {
    let mut max_offset = extent.base_offset;
    for dim in &extent.dims {
        if dim.count < 0 || dim.src_stride < 0 {
            return Err(CompileError::InvalidInput(
                "negative source extent dimension or stride".to_string(),
            ));
        }
        if dim.count == 0 {
            return Ok(0);
        }
        let count = u64::try_from(dim.count - 1)
            .map_err(|_| CompileError::InvalidInput("source extent count overflow".to_string()))?;
        let stride = u64::try_from(dim.src_stride)
            .map_err(|_| CompileError::InvalidInput("source extent stride overflow".to_string()))?;
        max_offset = max_offset
            .checked_add(count.checked_mul(stride).ok_or_else(|| {
                CompileError::InvalidInput("source extent byte overflow".to_string())
            })?)
            .ok_or_else(|| CompileError::InvalidInput("source extent byte overflow".to_string()))?;
    }
    max_offset
        .checked_add(u64::from(extent.element_bytes))
        .ok_or_else(|| CompileError::InvalidInput("source extent byte overflow".to_string()))
}

pub(super) fn compact_extent(shape: &[i64], element_bytes: u64) -> StridedExtent {
    let mut stride = i64::try_from(element_bytes).unwrap_or(i64::MAX);
    let mut dims = Vec::with_capacity(shape.len());
    for dim in shape.iter().rev() {
        dims.push(DimSpec {
            count: *dim,
            src_stride: stride,
            dst_stride: stride,
        });
        stride = stride.saturating_mul(*dim);
    }
    dims.reverse();
    StridedExtent {
        base_offset: 0,
        element_bytes: u32::try_from(element_bytes).unwrap_or(u32::MAX),
        dims,
    }
}

pub(super) fn byte_extent(bytes: u64) -> StridedExtent {
    StridedExtent {
        base_offset: 0,
        element_bytes: 1,
        dims: vec![DimSpec {
            count: i64::try_from(bytes).unwrap_or(i64::MAX),
            src_stride: 1,
            dst_stride: 1,
        }],
    }
}

pub(super) fn full_dest_extent(
    buffer: BufferId,
    decl: &TensorDecl,
) -> Result<DestExtent, CompileError> {
    Ok(DestExtent {
        buffer,
        offset: 0,
        stride: storage_extent_for_shape(&decl.shape, &decl.encoding)?,
    })
}

pub(super) fn storage_extent_for_shape(
    shape: &[i64],
    encoding: &Encoding,
) -> Result<StridedExtent, CompileError> {
    if let Some(element_bytes) = encoding_dense_element_bytes(encoding) {
        return Ok(compact_extent(shape, element_bytes));
    }
    Ok(byte_extent(encoding_nbytes(shape, encoding).ok_or_else(
        || CompileError::InvalidInput("packed extent byte size overflow".to_string()),
    )?))
}

fn selected_source_extent(
    source: &StridedExtent,
    shape: &[i64],
    encoding: &Encoding,
) -> Result<StridedExtent, CompileError> {
    let Some(element_bytes) = encoding_dense_element_bytes(encoding) else {
        return storage_extent_for_shape(shape, encoding);
    };
    if source.dims.len() != shape.len() {
        return Err(CompileError::InvalidInput(format!(
            "source stride rank {} does not match selected shape rank {}",
            source.dims.len(),
            shape.len()
        )));
    }
    let dest = compact_extent(shape, element_bytes);
    let dims = source
        .dims
        .iter()
        .zip(shape.iter())
        .zip(dest.dims.iter())
        .map(|((dim, count), dest_dim)| DimSpec {
            count: *count,
            src_stride: dim.src_stride,
            dst_stride: dest_dim.dst_stride,
        })
        .collect();
    Ok(StridedExtent {
        base_offset: source.base_offset,
        element_bytes: u32::try_from(element_bytes).unwrap_or(u32::MAX),
        dims,
    })
}

fn narrow_source_axis(
    mut source: SourceView,
    axis: Axis,
    start: i64,
    length: i64,
) -> Result<SourceView, CompileError> {
    let axis_index = axis.0 as usize;
    if axis_index >= source.shape.len() {
        return Err(CompileError::InvalidInput(format!(
            "source slice axis {} out of range for shape {:?}",
            axis.0, source.shape
        )));
    }
    if start < 0 || length < 0 || start + length > source.shape[axis_index] {
        return Err(CompileError::InvalidInput(format!(
            "source slice [{start}, {}) on axis {} exceeds shape {:?}",
            start + length,
            axis.0,
            source.shape
        )));
    }
    let old_stride = source.stride.clone();
    let can_preserve_strides = encoding_dense_element_bytes(&source.encoding).is_some()
        && old_stride.dims.len() == source.shape.len();
    let axis_stride_bytes = if can_preserve_strides {
        u64::try_from(old_stride.dims[axis_index].src_stride).map_err(|_| {
            CompileError::InvalidInput("negative source stride in slice lowering".to_string())
        })?
    } else {
        dense_axis_stride_bytes(&source.shape, axis, &source.encoding)?
    };
    source.offset_bytes = source
        .offset_bytes
        .checked_add(
            u64::try_from(start)
                .ok()
                .and_then(|start| start.checked_mul(axis_stride_bytes))
                .ok_or_else(|| {
                    CompileError::InvalidInput("source slice offset overflow".to_string())
                })?,
        )
        .ok_or_else(|| CompileError::InvalidInput("source slice offset overflow".to_string()))?;
    source.shape[axis_index] = length;
    source.stride = if can_preserve_strides {
        selected_source_extent(&old_stride, &source.shape, &source.encoding)?
    } else {
        storage_extent_for_shape(&source.shape, &source.encoding)?
    };
    Ok(source)
}

fn dense_axis_stride_bytes(
    shape: &[i64],
    axis: Axis,
    encoding: &Encoding,
) -> Result<u64, CompileError> {
    let axis = axis.0 as usize;
    if axis >= shape.len() {
        return Err(CompileError::InvalidInput(format!(
            "axis {} out of range for shape {:?}",
            axis, shape
        )));
    }
    let suffix_elements = shape[axis + 1..].iter().try_fold(1u64, |acc, dim| {
        let dim = u64::try_from(*dim).ok()?;
        acc.checked_mul(dim)
    });
    match encoding {
        Encoding::Raw(dtype) => suffix_elements
            .and_then(|elements| elements.checked_mul(dtype.bytes()))
            .ok_or_else(|| CompileError::InvalidInput("dense stride overflow".to_string())),
        Encoding::Quant(spec) => {
            let spec = spec.clone().normalized();
            let suffix = suffix_elements
                .ok_or_else(|| CompileError::InvalidInput("dense stride overflow".to_string()))?;
            let bits = suffix
                .checked_mul(u64::from(spec.bits_per_element))
                .ok_or_else(|| CompileError::InvalidInput("packed stride overflow".to_string()))?;
            if bits % 8 != 0 {
                return Err(CompileError::InvalidInput(format!(
                    "packed {:?} select on axis {} is not byte-aligned",
                    spec.scheme, axis
                )));
            }
            Ok(bits / 8)
        }
    }
}

pub(super) fn dtype_to_quant_marker(dtype: DType) -> QuantScheme {
    match dtype {
        DType::F8E4M3 => QuantScheme::Fp8E4M3,
        DType::F8E5M2 => QuantScheme::Fp8E5M2,
        DType::I8 | DType::U8 => QuantScheme::Int8Symmetric,
        _ => QuantScheme::None,
    }
}

/// Whether a lazy source view is a plain dense window on its checkpoint tensor.
///
/// Gather piece offsets and strides are expressed in the input's own dense
/// layout, so they can be rebased onto a view that is itself dense — a shard of
/// a shard stays lazy — but not onto one that already skips. A strided view has
/// to be materialized before it can be gathered from again.
pub(super) fn source_is_dense(source: &SourceView) -> Result<bool, CompileError> {
    Ok(source.stride == storage_extent_for_shape(&source.shape, &source.encoding)?)
}

/// Split one [`GatherPiece`] into the source and destination extents of an
/// `ExtentWrite`.
///
/// The innermost dimension is always the contiguous block, so it becomes
/// `element_bytes` and the outer dimensions become the loop nest. The
/// destination of a piece is a single contiguous range by construction — the
/// fold in `contract::compile` refuses to widen a loop that skips in the
/// destination — which is exactly what the executors require of `dest.stride`.
pub(super) fn gather_extents(
    piece: &GatherPiece,
) -> Result<(StridedExtent, StridedExtent), CompileError> {
    let bytes = piece.bytes();
    if piece.is_contiguous() {
        return Ok((byte_extent(bytes), byte_extent(bytes)));
    }
    let (inner, outer) = piece
        .dims
        .split_last()
        .ok_or_else(|| CompileError::InvalidInput("gather piece has no extent".to_string()))?;
    if inner.src_stride != 1 || inner.dst_stride != 1 {
        return Err(CompileError::InvalidInput(
            "gather piece has no contiguous inner block".to_string(),
        ));
    }
    let element_bytes = u32::try_from(inner.count).map_err(|_| {
        CompileError::InvalidInput("gather piece inner block exceeds 4 GiB".to_string())
    })?;
    let mut dense = i64::from(element_bytes);
    let mut source_dims = Vec::with_capacity(outer.len());
    let mut dest_dims = Vec::with_capacity(outer.len());
    for dim in outer.iter().rev() {
        if dim.dst_stride != dense {
            return Err(CompileError::InvalidInput(
                "gather piece writes a non-contiguous destination".to_string(),
            ));
        }
        source_dims.push(DimSpec {
            count: dim.count,
            src_stride: dim.src_stride,
            dst_stride: dense,
        });
        dest_dims.push(DimSpec {
            count: dim.count,
            src_stride: dense,
            dst_stride: dense,
        });
        dense = dense.checked_mul(dim.count).ok_or_else(|| {
            CompileError::InvalidInput("gather piece extent overflow".to_string())
        })?;
    }
    source_dims.reverse();
    dest_dims.reverse();
    Ok((
        StridedExtent {
            base_offset: 0,
            element_bytes,
            dims: source_dims,
        },
        StridedExtent {
            base_offset: 0,
            element_bytes,
            dims: dest_dims,
        },
    ))
}
