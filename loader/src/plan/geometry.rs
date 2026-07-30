//! Strided-extent geometry: build and measure the source/dest extents that
//! back every copy instruction.

use crate::error::{Error, OrOverflow, Result};
use crate::extent::Extent;
use crate::plan::DestExtent;
use crate::types::{
    BufferId, Encoding, RepackLayout, RepackSpec, TensorDecl, encoding_dense_element_bytes,
    encoding_nbytes, tensor_nbytes,
};

pub(super) fn repack_stage_bytes(spec: RepackSpec) -> Result<u64> {
    match spec.layout {
        RepackLayout::MarlinMxfp4Weight => {
            let elems = u64::from(spec.target_rows)
                .checked_mul(u64::from(spec.target_cols))
                .or_overflow("MXFP4 repack stage size overflow")?;
            Ok(elems.div_ceil(2))
        }
        RepackLayout::MarlinMxfp4Scale => Ok(0),
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
