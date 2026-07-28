//! Strided-extent geometry: build, narrow and measure the source/dest
//! extents that back every copy instruction.

use crate::error::{Error, OrOverflow, Result};
use crate::extent::{Dim, Extent};
use crate::plan::DestExtent;
use crate::plan::build::SourceView;
use crate::types::{
    Axis, BufferId, Encoding, RepackLayout, RepackSpec, TensorDecl, encoding_dense_element_bytes,
    encoding_nbytes, tensor_nbytes,
};

pub(super) fn narrow_repack_source(
    mut source: SourceView,
    spec: RepackSpec,
) -> Result<(SourceView, RepackSpec)> {
    let mut narrowed = spec;
    let valid_rows = if narrowed.valid_rows == 0 {
        narrowed.target_rows
    } else {
        narrowed.valid_rows
    };
    narrowed.valid_rows = valid_rows;

    if source.shape.len() < 2 {
        return Err(Error::Contract(
            "Repack source must have batch and row axes".to_string(),
        ));
    }
    if source.shape[0] != i64::from(narrowed.batch)
        || source.shape[1] != i64::from(narrowed.source_rows)
    {
        return Err(Error::Contract(format!(
            "Repack source shape {:?} does not match batch/source_rows {:?}/{}",
            source.shape, narrowed.batch, narrowed.source_rows
        )));
    }

    let (row_start, row_count) = match narrowed.row_map {
        crate::types::RowMap::Identity => (narrowed.source_row_offset, valid_rows),
        crate::types::RowMap::Even | crate::types::RowMap::Odd => {
            let start = narrowed
                .source_row_offset
                .checked_mul(2)
                .or_overflow("Repack row offset overflow")?;
            let rows = valid_rows
                .checked_mul(2)
                .or_overflow("Repack row count overflow")?;
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
                return Err(Error::Contract(format!(
                    "MarlinMxfp4Weight Repack source must be [B, R, K/32, 16], got {:?}",
                    source.shape
                )));
            }
            if !narrowed.source_col_offset.is_multiple_of(32)
                || !narrowed.source_cols.is_multiple_of(32)
                || !narrowed.source_stride_cols.is_multiple_of(32)
            {
                return Err(Error::Contract(
                    "MarlinMxfp4Weight source narrowing requires 32-wide MXFP4 group alignment"
                        .to_string(),
                ));
            }
            let group_start = narrowed.source_col_offset / 32;
            let group_count = narrowed.source_cols / 32;
            if source.shape[2] != i64::from(narrowed.source_stride_cols / 32) {
                return Err(Error::Contract(format!(
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
                return Err(Error::Contract(format!(
                    "MarlinMxfp4Scale Repack source must be [B, R, groups], got {:?}",
                    source.shape
                )));
            }
            if source.shape[2] != i64::from(narrowed.source_stride_cols) {
                return Err(Error::Contract(format!(
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
                return Err(Error::Contract(format!(
                    "DenseRowGather Repack source must be [B, R], got {:?}",
                    source.shape
                )));
            }
        }
        RepackLayout::None => {}
    }

    Ok((source, narrowed))
}

pub(super) fn repack_stage_bytes(spec: RepackSpec) -> Result<u64> {
    match spec.layout {
        RepackLayout::MarlinMxfp4Weight => {
            let elems = u64::from(spec.target_rows)
                .checked_mul(u64::from(spec.target_cols))
                .or_overflow("MXFP4 repack stage size overflow")?;
            Ok(elems.div_ceil(2))
        }
        RepackLayout::MarlinMxfp4Scale | RepackLayout::DenseRowGather | RepackLayout::None => Ok(0),
    }
}

pub(super) fn extent_storage_bytes(extent: &Extent) -> Result<u64> {
    tensor_nbytes(
        &extent.dims.iter().map(|dim| dim.count).collect::<Vec<_>>(),
        u64::from(extent.element_bytes),
    )
    .or_overflow("extent byte size overflow")
}

pub(super) fn strided_physical_source_bytes(extent: &Extent) -> Result<u64> {
    let mut max_offset = extent.base_offset;
    for dim in &extent.dims {
        if dim.count < 0 || dim.src_stride < 0 {
            return Err(Error::Contract(
                "negative source extent dimension or stride".to_string(),
            ));
        }
        if dim.count == 0 {
            return Ok(0);
        }
        let count = u64::try_from(dim.count - 1).or_overflow("source extent count overflow")?;
        let stride = u64::try_from(dim.src_stride).or_overflow("source extent stride overflow")?;
        max_offset = max_offset
            .checked_add(
                count
                    .checked_mul(stride)
                    .or_overflow("source extent byte overflow")?,
            )
            .or_overflow("source extent byte overflow")?;
    }
    max_offset
        .checked_add(u64::from(extent.element_bytes))
        .or_overflow("source extent byte overflow")
}

pub(super) fn full_dest_extent(buffer: BufferId, decl: &TensorDecl) -> Result<DestExtent> {
    Ok(DestExtent {
        buffer,
        offset: 0,
        stride: storage_extent_for_shape(&decl.shape, &decl.encoding)?,
    })
}

pub(super) fn storage_extent_for_shape(shape: &[i64], encoding: &Encoding) -> Result<Extent> {
    if let Some(element_bytes) = encoding_dense_element_bytes(encoding) {
        return Ok(Extent::dense(shape, element_bytes));
    }
    Ok(Extent::byte_run(
        encoding_nbytes(shape, encoding).or_overflow("packed extent byte size overflow")?,
    ))
}

fn selected_source_extent(source: &Extent, shape: &[i64], encoding: &Encoding) -> Result<Extent> {
    let Some(element_bytes) = encoding_dense_element_bytes(encoding) else {
        return storage_extent_for_shape(shape, encoding);
    };
    if source.dims.len() != shape.len() {
        return Err(Error::Contract(format!(
            "source stride rank {} does not match selected shape rank {}",
            source.dims.len(),
            shape.len()
        )));
    }
    let dest = Extent::dense(shape, element_bytes);
    let dims = source
        .dims
        .iter()
        .zip(shape.iter())
        .zip(dest.dims.iter())
        .map(|((dim, count), dest_dim)| Dim {
            count: *count,
            src_stride: dim.src_stride,
            dst_stride: dest_dim.dst_stride,
        })
        .collect();
    Ok(Extent {
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
) -> Result<SourceView> {
    let axis_index = axis.0 as usize;
    if axis_index >= source.shape.len() {
        return Err(Error::Contract(format!(
            "source slice axis {} out of range for shape {:?}",
            axis.0, source.shape
        )));
    }
    if start < 0 || length < 0 || start + length > source.shape[axis_index] {
        return Err(Error::Contract(format!(
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
        u64::try_from(old_stride.dims[axis_index].src_stride)
            .map_err(|_| Error::Contract("negative source stride in slice lowering".to_string()))?
    } else {
        dense_axis_stride_bytes(&source.shape, axis, &source.encoding)?
    };
    source.offset_bytes = source
        .offset_bytes
        .checked_add(
            u64::try_from(start)
                .ok()
                .and_then(|start| start.checked_mul(axis_stride_bytes))
                .or_overflow("source slice offset overflow")?,
        )
        .or_overflow("source slice offset overflow")?;
    source.shape[axis_index] = length;
    source.stride = if can_preserve_strides {
        selected_source_extent(&old_stride, &source.shape, &source.encoding)?
    } else {
        storage_extent_for_shape(&source.shape, &source.encoding)?
    };
    Ok(source)
}

fn dense_axis_stride_bytes(shape: &[i64], axis: Axis, encoding: &Encoding) -> Result<u64> {
    let axis = axis.0 as usize;
    if axis >= shape.len() {
        return Err(Error::Contract(format!(
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
            .or_overflow("dense stride overflow"),
        Encoding::Quant(spec) => {
            let spec = spec.clone().normalized();
            let suffix = suffix_elements.or_overflow("dense stride overflow")?;
            let bits = suffix
                .checked_mul(u64::from(spec.bits_per_element))
                .or_overflow("packed stride overflow")?;
            if bits % 8 != 0 {
                return Err(Error::Contract(format!(
                    "packed {:?} select on axis {} is not byte-aligned",
                    spec.scheme, axis
                )));
            }
            Ok(bits / 8)
        }
    }
}
