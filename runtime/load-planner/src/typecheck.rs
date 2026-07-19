use crate::error::CompileError;
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::types::{
    DType, Encoding, Layout, QuantScheme, RepackLayout, RepackSpec, RowMap, Sharding, TensorDecl,
    encoding_nbytes,
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
        LayoutExpr::ByteSpans { spans, decl } => {
            if spans.is_empty() {
                return Err(CompileError::InvalidInput(format!(
                    "ByteSpans expr {index} has no spans"
                )));
            }
            let total_bytes = encoding_nbytes(&decl.shape, &decl.encoding).ok_or_else(|| {
                CompileError::InvalidInput(format!("ByteSpans expr {index} size overflow"))
            })?;
            for (span_index, span) in spans.iter().enumerate() {
                if span.span_bytes == 0 {
                    return Err(CompileError::InvalidInput(format!(
                        "ByteSpans expr {index} span {span_index} is empty"
                    )));
                }
                let end = span
                    .dest_offset_bytes
                    .checked_add(span.span_bytes)
                    .ok_or_else(|| {
                        CompileError::InvalidInput(format!(
                            "ByteSpans expr {index} span {span_index} offset overflow"
                        ))
                    })?;
                if end > total_bytes {
                    return Err(CompileError::InvalidInput(format!(
                        "ByteSpans expr {index} span {span_index} exceeds output size"
                    )));
                }
            }
            Ok(decl.clone())
        }
        LayoutExpr::Select {
            input,
            axis,
            start,
            length,
            decl,
        } => {
            let mut expected = input_decl(inferred, input.0, index)?.clone();
            let axis = axis.0 as usize;
            if axis >= expected.shape.len() || *start < 0 || *length <= 0 {
                return Err(CompileError::InvalidInput(format!(
                    "Select expr {index} has invalid axis/range"
                )));
            }
            if start + length > expected.shape[axis] {
                return Err(CompileError::InvalidInput(format!(
                    "Select expr {index} range [{}:{}) exceeds dim {}",
                    start,
                    start + length,
                    expected.shape[axis]
                )));
            }
            require_byte_aligned_slice(index, "Select", &expected, axis, *start, *length)?;
            expected.shape[axis] = *length;
            expect_decl(index, decl, expected)
        }
        LayoutExpr::Partition {
            input,
            axis,
            parts,
            index: rank,
            decl,
        } => {
            if *parts == 0 || *rank >= *parts {
                return Err(CompileError::InvalidInput(format!(
                    "Partition expr {index} has invalid parts/index"
                )));
            }
            let mut expected = input_decl(inferred, input.0, index)?.clone();
            let partition_axis = *axis;
            let axis = axis.0 as usize;
            if axis >= expected.shape.len() || expected.shape[axis] % i64::from(*parts) != 0 {
                return Err(CompileError::InvalidInput(format!(
                    "Partition expr {index} axis {} cannot be split into {} parts",
                    axis, parts
                )));
            }
            expected.shape[axis] /= i64::from(*parts);
            expected.sharding = Sharding {
                axis: Some(partition_axis),
                world: *parts,
                rank: *rank,
            };
            expect_decl(index, decl, expected)
        }
        LayoutExpr::Join { inputs, axis, decl } => {
            if inputs.is_empty() {
                return Err(CompileError::InvalidInput(format!(
                    "Join expr {index} has no inputs"
                )));
            }
            let mut expected = input_decl(inferred, inputs[0].0, index)?.clone();
            let axis = axis.0 as usize;
            if axis >= expected.shape.len() {
                return Err(CompileError::InvalidInput(format!(
                    "Join expr {index} axis {} out of range",
                    axis
                )));
            }
            expected.shape[axis] = 0;
            for input in inputs {
                let current = input_decl(inferred, input.0, index)?;
                require_compatible(index, "Join", &expected, current, Some(axis))?;
                expected.shape[axis] += current.shape[axis];
            }
            expect_decl(index, decl, expected)
        }
        LayoutExpr::Stack { inputs, axis, decl } => {
            if inputs.is_empty() {
                return Err(CompileError::InvalidInput(format!(
                    "Stack expr {index} has no inputs"
                )));
            }
            let first = input_decl(inferred, inputs[0].0, index)?.clone();
            for input in inputs.iter().skip(1) {
                let current = input_decl(inferred, input.0, index)?;
                require_compatible(index, "Stack", &first, current, None)?;
                if first.shape != current.shape {
                    return Err(CompileError::InvalidInput(format!(
                        "Stack expr {index} input shape mismatch"
                    )));
                }
            }
            let axis = axis.0 as usize;
            if axis > first.shape.len() {
                return Err(CompileError::InvalidInput(format!(
                    "Stack expr {index} axis {} out of range",
                    axis
                )));
            }
            let mut expected = first;
            expected.shape.insert(axis, inputs.len() as i64);
            expect_decl(index, decl, expected)
        }
        LayoutExpr::Unzip {
            input,
            axis,
            outputs,
        } => {
            let source = input_decl(inferred, input.0, index)?;
            let axis = axis.0 as usize;
            if axis >= source.shape.len() || outputs.len() != source.shape[axis] as usize {
                return Err(CompileError::InvalidInput(format!(
                    "Unzip expr {index} output count does not match axis size"
                )));
            }
            for output in outputs {
                let mut expected = source.clone();
                expected.shape.remove(axis);
                if !output.same_runtime_contract(&expected) {
                    return Err(CompileError::InvalidInput(format!(
                        "Unzip expr {index} output decl mismatch"
                    )));
                }
            }
            Ok(source.clone())
        }
        LayoutExpr::Reorder { input, perm, decl } => {
            let source = input_decl(inferred, input.0, index)?;
            if perm.len() != source.shape.len() {
                return Err(CompileError::InvalidInput(format!(
                    "Reorder expr {index} rank mismatch"
                )));
            }
            let mut seen = vec![false; perm.len()];
            let mut expected = source.clone();
            expected.shape.clear();
            for axis in perm {
                let axis = *axis as usize;
                if axis >= seen.len() || seen[axis] {
                    return Err(CompileError::InvalidInput(format!(
                        "Reorder expr {index} has invalid permutation"
                    )));
                }
                seen[axis] = true;
                expected.shape.push(source.shape[axis]);
            }
            expect_decl(index, decl, expected)
        }
        LayoutExpr::View {
            input,
            layout,
            axis,
            start,
            length,
            decl,
        } => {
            let mut expected = input_decl(inferred, input.0, index)?.clone();
            if let Some(axis) = axis {
                let axis_index = axis.0 as usize;
                if axis_index >= expected.shape.len() {
                    return Err(CompileError::InvalidInput(format!(
                        "View expr {index} axis {} out of range",
                        axis.0
                    )));
                }
                if *start < 0 || *length <= 0 || start + length > expected.shape[axis_index] {
                    return Err(CompileError::InvalidInput(format!(
                        "View expr {index} range [{}, {}) exceeds dim {}",
                        start,
                        start + length,
                        expected.shape[axis_index]
                    )));
                }
                require_byte_aligned_slice(index, "View", &expected, axis_index, *start, *length)?;
                expected.shape[axis_index] = *length;
            }
            expected.layout = layout.clone();
            expected.alignment = layout.alignment;
            expect_decl(index, decl, expected)
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
        LayoutExpr::Attach {
            data,
            metadata,
            decl,
        } => {
            let expected = input_decl(inferred, data.0, index)?.clone();
            for meta in metadata {
                let _ = input_decl(inferred, meta.0, index)?;
            }
            expect_decl(index, decl, expected)
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
            if spec.source_rows % 2 != 0 {
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
            if spec.source_cols % 8 != 0
                || spec.target_cols % 8 != 0
                || spec.source_stride_cols % 8 != 0
                || spec.source_col_offset % 8 != 0
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

fn require_compatible(
    index: usize,
    op: &'static str,
    expected: &TensorDecl,
    current: &TensorDecl,
    ignored_axis: Option<usize>,
) -> Result<(), CompileError> {
    if expected.encoding != current.encoding || expected.sharding != current.sharding {
        return Err(CompileError::InvalidInput(format!(
            "{op} expr {index} input encoding/sharding mismatch"
        )));
    }
    if expected.shape.len() != current.shape.len() {
        return Err(CompileError::InvalidInput(format!(
            "{op} expr {index} input rank mismatch"
        )));
    }
    for (axis, (lhs, rhs)) in expected.shape.iter().zip(&current.shape).enumerate() {
        if Some(axis) != ignored_axis && lhs != rhs {
            return Err(CompileError::InvalidInput(format!(
                "{op} expr {index} input shape mismatch on axis {axis}"
            )));
        }
    }
    Ok(())
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

fn require_byte_aligned_slice(
    index: usize,
    op: &'static str,
    source: &TensorDecl,
    axis: usize,
    start: i64,
    length: i64,
) -> Result<(), CompileError> {
    let Encoding::Quant(spec) = &source.encoding else {
        return Ok(());
    };
    let spec = spec.clone().normalized();
    if spec.bits_per_element >= 8 {
        return Ok(());
    }
    let suffix_elements = source.shape[axis + 1..].iter().try_fold(1u64, |acc, dim| {
        let dim = u64::try_from(*dim).ok()?;
        acc.checked_mul(dim)
    });
    let suffix_elements = suffix_elements.ok_or_else(|| {
        CompileError::InvalidInput(format!("{op} expr {index} packed slice overflows"))
    })?;
    let stride_bits = suffix_elements
        .checked_mul(u64::from(spec.bits_per_element))
        .ok_or_else(|| {
            CompileError::InvalidInput(format!("{op} expr {index} packed slice overflows"))
        })?;
    let start = u64::try_from(start).map_err(|_| {
        CompileError::InvalidInput(format!("{op} expr {index} has negative packed start"))
    })?;
    let length = u64::try_from(length).map_err(|_| {
        CompileError::InvalidInput(format!("{op} expr {index} has negative packed length"))
    })?;
    let start_bits = start.checked_mul(stride_bits).ok_or_else(|| {
        CompileError::InvalidInput(format!("{op} expr {index} packed slice overflows"))
    })?;
    let span_bits = length.checked_mul(stride_bits).ok_or_else(|| {
        CompileError::InvalidInput(format!("{op} expr {index} packed slice overflows"))
    })?;
    if start_bits % 8 != 0 || span_bits % 8 != 0 {
        return Err(CompileError::InvalidInput(format!(
            "{op} expr {index} packed {:?} slice is not byte-aligned",
            spec.scheme
        )));
    }
    if let Some(channel_axis) = spec.channel_axis
        && channel_axis.0 as usize == axis
    {
        let group = u64::from(spec.normalized_group_size());
        if group > 1 && (start % group != 0 || length % group != 0) {
            return Err(CompileError::InvalidInput(format!(
                "{op} expr {index} packed {:?} slice is not group-aligned",
                spec.scheme
            )));
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn _layout_compatible(_layout: &Layout) -> bool {
    true
}
