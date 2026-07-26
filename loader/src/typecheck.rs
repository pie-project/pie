use crate::error::CompileError;
use crate::ir::{GatherPiece, LayoutExpr, LayoutPlan};
use crate::types::{
    DType, Encoding, QuantScheme, RepackLayout, RepackSpec, RowMap, TensorDecl, encoding_nbytes,
};

pub fn typecheck(plan: &LayoutPlan) -> Result<Vec<TensorDecl>, CompileError> {
    let mut inferred = Vec::with_capacity(plan.exprs.len());
    for (index, expr) in plan.exprs.iter().enumerate() {
        let decl = infer_expr(plan, &inferred, index, expr)?;
        inferred.push(decl);
    }
    for output in &plan.outputs {
        if output.0 as usize >= inferred.len() {
            return Err(CompileError::InvalidInput(format!(
                "layout output expr {} is out of range",
                output.0
            )));
        }
    }
    Ok(inferred)
}

fn infer_expr(
    _plan: &LayoutPlan,
    inferred: &[TensorDecl],
    index: usize,
    expr: &LayoutExpr,
) -> Result<TensorDecl, CompileError> {
    match expr {
        LayoutExpr::Source { decl, .. } => Ok(decl.clone()),
        LayoutExpr::Gather {
            inputs,
            pieces,
            zero_fill,
            decl,
        } => {
            let output_bytes = encoding_nbytes(&decl.shape, &decl.encoding).ok_or_else(|| {
                CompileError::InvalidInput(format!("Gather expr {index} size overflow"))
            })? as i64;
            let mut input_bytes = Vec::with_capacity(inputs.len());
            for input in inputs {
                let source = input_decl(inferred, input.0, index)?;
                input_bytes.push(encoding_nbytes(&source.shape, &source.encoding).ok_or_else(
                    || {
                        CompileError::InvalidInput(format!(
                            "Gather expr {index} input size overflow"
                        ))
                    },
                )? as i64);
            }
            if pieces.is_empty() && !zero_fill {
                return Err(CompileError::InvalidInput(format!(
                    "Gather expr {index} copies nothing and fills nothing"
                )));
            }
            let mut copied = 0i64;
            for (piece_index, piece) in pieces.iter().enumerate() {
                let where_ = format!("Gather expr {index} piece {piece_index}");
                let limit = *input_bytes.get(piece.input as usize).ok_or_else(|| {
                    CompileError::InvalidInput(format!("{where_} names input {}", piece.input))
                })?;
                let (src_extent, dst_extent, bytes) = piece_extents(piece, &where_)?;
                copied = copied.checked_add(bytes).ok_or_else(|| {
                    CompileError::InvalidInput(format!("{where_} byte count overflows"))
                })?;
                if piece.src_offset as i64 + src_extent > limit {
                    return Err(CompileError::InvalidInput(format!(
                        "{where_} reads past the end of its input"
                    )));
                }
                if piece.dst_offset as i64 + dst_extent > output_bytes {
                    return Err(CompileError::InvalidInput(format!(
                        "{where_} writes past the end of the output"
                    )));
                }
            }
            // The pieces are disjoint by construction, so a byte count is a
            // coverage proof: without a fill they must tile the output exactly.
            if !zero_fill && copied != output_bytes {
                return Err(CompileError::InvalidInput(format!(
                    "Gather expr {index} covers {copied} of {output_bytes} bytes \
                     with no zero fill declared"
                )));
            }
            if copied > output_bytes {
                return Err(CompileError::InvalidInput(format!(
                    "Gather expr {index} writes {copied} bytes into {output_bytes}"
                )));
            }
            Ok(decl.clone())
        }
        LayoutExpr::Cast { input, dtype, decl } => {
            let source = input_decl(inferred, input.0, index)?;
            if !matches!(source.encoding, Encoding::Raw(_)) {
                return Err(CompileError::InvalidInput(format!(
                    "Cast expr {index} input must be raw"
                )));
            }
            let mut expected = source.clone();
            expected.encoding = Encoding::Raw(*dtype);
            expect_decl(index, decl, expected)
        }
        LayoutExpr::Decode {
            scheme,
            data,
            metadata,
            decl,
        } => {
            let source = input_decl(inferred, data.0, index)?;
            require_quant_scheme(index, "Decode", source, *scheme)?;
            for meta in metadata {
                let _ = input_decl(inferred, meta.0, index)?;
            }
            let mut expected = source.clone();
            let logical_dtype = match &source.encoding {
                Encoding::Quant(spec) => spec.logical_dtype,
                Encoding::Raw(_) => unreachable!(),
            };
            expected.encoding = Encoding::Raw(logical_dtype);
            if decl.same_runtime_contract(&expected) {
                return Ok(decl.clone());
            }
            if let Encoding::Raw(dtype) = decl.encoding
                && dtype.is_float()
                && logical_dtype.is_float()
            {
                expected.encoding = Encoding::Raw(dtype);
                return expect_decl(index, decl, expected);
            }
            expect_decl(index, decl, expected)
        }
        LayoutExpr::Encode {
            scheme,
            input,
            metadata_outputs: _,
            decl,
        } => {
            let source = input_decl(inferred, input.0, index)?;
            if !matches!(source.encoding, Encoding::Raw(_)) {
                return Err(CompileError::InvalidInput(format!(
                    "Encode expr {index} input must be raw"
                )));
            }
            match &decl.encoding {
                Encoding::Quant(spec) if spec.scheme == *scheme => Ok(decl.clone()),
                _ => Err(CompileError::InvalidInput(format!(
                    "Encode expr {index} output is not {:?}",
                    scheme
                ))),
            }
        }
        LayoutExpr::Transcode {
            from,
            to,
            data,
            metadata,
            metadata_outputs: _,
            decl,
        } => {
            let source = input_decl(inferred, data.0, index)?;
            require_quant_scheme(index, "Transcode", source, *from)?;
            for meta in metadata {
                let _ = input_decl(inferred, meta.0, index)?;
            }
            match &decl.encoding {
                Encoding::Quant(spec) if spec.scheme == *to => Ok(decl.clone()),
                _ => Err(CompileError::InvalidInput(format!(
                    "Transcode expr {index} output is not {:?}",
                    to
                ))),
            }
        }
        LayoutExpr::Repack { input, spec, decl } => {
            let source = input_decl(inferred, input.0, index)?;
            validate_repack_spec(index, source, decl, *spec)?;
            Ok(decl.clone())
        }
        LayoutExpr::Realize {
            input,
            runtime_name,
            decl,
        } => {
            if runtime_name.is_empty() {
                return Err(CompileError::InvalidInput(format!(
                    "Realize expr {index} has empty runtime name"
                )));
            }
            let expected = input_decl(inferred, input.0, index)?;
            if !decl.same_runtime_contract(expected) {
                return Err(CompileError::InvalidInput(format!(
                    "Realize expr {index} decl does not match input"
                )));
            }
            Ok(decl.clone())
        }
    }
}

/// `(source extent, destination extent, bytes moved)` of one rectangular copy.
///
/// The extent of a strided walk is the distance from its first byte to its
/// last, which is what a bounds check needs; the byte count is the product of
/// the counts, which is what a coverage check needs. They differ whenever the
/// copy skips.
fn piece_extents(piece: &GatherPiece, where_: &str) -> Result<(i64, i64, i64), CompileError> {
    if piece.dims.is_empty() {
        return Err(CompileError::InvalidInput(format!(
            "{where_} has no extent"
        )));
    }
    let mut src_extent = 1i64;
    let mut dst_extent = 1i64;
    let mut bytes = 1i64;
    for dim in &piece.dims {
        if dim.count <= 0 {
            return Err(CompileError::InvalidInput(format!(
                "{where_} has a non-positive count"
            )));
        }
        if dim.src_stride < 0 || dim.dst_stride < 0 {
            return Err(CompileError::InvalidInput(format!(
                "{where_} walks backwards"
            )));
        }
        src_extent += (dim.count - 1) * dim.src_stride;
        dst_extent += (dim.count - 1) * dim.dst_stride;
        bytes = bytes
            .checked_mul(dim.count)
            .ok_or_else(|| CompileError::InvalidInput(format!("{where_} byte count overflows")))?;
    }
    Ok((src_extent, dst_extent, bytes))
}

fn input_decl(
    inferred: &[TensorDecl],
    input: u32,
    index: usize,
) -> Result<&TensorDecl, CompileError> {
    if input as usize >= index {
        return Err(CompileError::InvalidInput(format!(
            "expr {index} references non-dominating input {input}"
        )));
    }
    inferred.get(input as usize).ok_or_else(|| {
        CompileError::InvalidInput(format!("expr {index} input {input} is out of range"))
    })
}

fn validate_repack_spec(
    index: usize,
    source: &TensorDecl,
    decl: &TensorDecl,
    spec: RepackSpec,
) -> Result<(), CompileError> {
    if spec.layout == RepackLayout::None {
        return Err(CompileError::InvalidInput(format!(
            "Repack expr {index} has no target layout"
        )));
    }
    if spec.batch == 0 || spec.source_rows == 0 || spec.target_rows == 0 {
        return Err(CompileError::InvalidInput(format!(
            "Repack expr {index} has invalid row dimensions"
        )));
    }
    let valid_rows = if spec.valid_rows == 0 {
        spec.target_rows
    } else {
        spec.valid_rows
    };
    if valid_rows > spec.target_rows {
        return Err(CompileError::InvalidInput(format!(
            "Repack expr {index} valid_rows exceeds target_rows"
        )));
    }
    let source_col_end = spec
        .source_col_offset
        .checked_add(spec.source_cols)
        .ok_or_else(|| {
            CompileError::InvalidInput(format!("Repack expr {index} column offset overflow"))
        })?;
    if spec.source_cols == 0
        || spec.target_cols == 0
        || spec.target_cols < spec.source_cols
        || spec.source_stride_cols < spec.source_cols
        || spec.source_col_offset > spec.source_stride_cols
        || source_col_end > spec.source_stride_cols
    {
        return Err(CompileError::InvalidInput(format!(
            "Repack expr {index} has invalid column dimensions"
        )));
    }
    match spec.row_map {
        RowMap::Identity => {
            if spec.source_row_offset + valid_rows > spec.source_rows {
                return Err(CompileError::InvalidInput(format!(
                    "Repack expr {index} row offset exceeds source rows"
                )));
            }
        }
        RowMap::Even | RowMap::Odd => {
            if !spec.source_rows.is_multiple_of(2) {
                return Err(CompileError::InvalidInput(format!(
                    "Repack expr {index} even/odd row maps require an even source row count"
                )));
            }
            let required = spec
                .source_row_offset
                .checked_add(valid_rows)
                .and_then(|rows| rows.checked_mul(2))
                .ok_or_else(|| {
                    CompileError::InvalidInput(format!("Repack expr {index} row offset overflow"))
                })?;
            if required > spec.source_rows {
                return Err(CompileError::InvalidInput(format!(
                    "Repack expr {index} even/odd row offset exceeds source rows"
                )));
            }
        }
    }
    match spec.layout {
        RepackLayout::MarlinMxfp4Weight => {
            if !spec.source_cols.is_multiple_of(8)
                || !spec.target_cols.is_multiple_of(8)
                || !spec.source_stride_cols.is_multiple_of(8)
                || !spec.source_col_offset.is_multiple_of(8)
            {
                return Err(CompileError::InvalidInput(format!(
                    "Repack expr {index} Marlin MXFP4 weight columns and offsets must be divisible by 8"
                )));
            }
            match &decl.encoding {
                Encoding::Quant(q) if q.scheme == QuantScheme::Mxfp4E2M1E8M0 => {}
                _ => {
                    return Err(CompileError::InvalidInput(format!(
                        "Repack expr {index} Marlin MXFP4 weight output must be MXFP4 quantized"
                    )));
                }
            }
        }
        RepackLayout::MarlinMxfp4Scale => {
            if !matches!(decl.encoding, Encoding::Raw(DType::U8)) {
                return Err(CompileError::InvalidInput(format!(
                    "Repack expr {index} Marlin MXFP4 scale output must be raw U8"
                )));
            }
            if u64::from(spec.target_rows) * u64::from(spec.target_cols) % 64 != 0 {
                return Err(CompileError::InvalidInput(format!(
                    "Repack expr {index} Marlin MXFP4 scale output requires total scale count divisible by 64"
                )));
            }
        }
        RepackLayout::DenseRowGather => {
            if source.encoding != decl.encoding {
                return Err(CompileError::InvalidInput(format!(
                    "Repack expr {index} DenseRowGather cannot change encoding"
                )));
            }
        }
        RepackLayout::None => unreachable!(),
    }
    Ok(())
}

fn expect_decl(
    index: usize,
    actual: &TensorDecl,
    mut expected: TensorDecl,
) -> Result<TensorDecl, CompileError> {
    expected.id = actual.id;
    expected.name = actual.name.clone();
    if !actual.same_runtime_contract(&expected) {
        return Err(CompileError::InvalidInput(format!(
            "expr {index} declared {:?} but inferred {:?}",
            actual, expected
        )));
    }
    Ok(actual.clone())
}

fn require_quant_scheme(
    index: usize,
    op: &'static str,
    source: &TensorDecl,
    scheme: QuantScheme,
) -> Result<(), CompileError> {
    match &source.encoding {
        Encoding::Quant(spec) if spec.scheme == scheme => Ok(()),
        Encoding::Quant(spec) => Err(CompileError::InvalidInput(format!(
            "{op} expr {index} expected {:?}, got {:?}",
            scheme, spec.scheme
        ))),
        Encoding::Raw(_) => Err(CompileError::InvalidInput(format!(
            "{op} expr {index} input must be quantized"
        ))),
    }
}
