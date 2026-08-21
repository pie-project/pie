use model_loader::checkpoint::RawTensor;
use model_loader::contract::{Expr, TensorType};
use model_loader::error::Error;
use model_loader::types::{Axis, DType, Encoding, QuantScheme, QuantSpec};

use super::builder::{Builder, is_raw};
use super::policy::RuntimeQuant;

pub fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub fn has_wrapper_member(raw_name: &str, member: &str) -> bool {
    raw_name.starts_with(member)
        || raw_name
            .strip_prefix("model.")
            .is_some_and(|rest| rest.starts_with(member))
}

pub fn decoder_member(raw_name: &str) -> Option<&str> {
    for prefix in ["model.language_model.", "language_model.model."] {
        if let Some(rest) = raw_name.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

pub fn already_lowered(raw_name: &str) -> bool {
    for table in [
        "shared_embedding.",
        "embed_tokens.",

        "embed_tokens_per_layer.",
        "per_layer_model_projection.",
        "per_layer_projection_norm.",
    ] {
        if raw_name.starts_with(table) {
            return true;
        }
    }
    if raw_name == "final_norm.weight" {
        return true;
    }

    let Some(tail) = raw_name.strip_prefix("layers.") else {
        return false;
    };
    let Some(dot) = tail.find('.') else {
        return false;
    };
    let index = &tail[..dot];
    !index.is_empty() && index.chars().all(|c| c.is_ascii_digit())
}

pub fn int4_requested(b: &Builder<'_>, schema: &str) -> Result<bool, Error> {
    match b.runtime_quant() {
        RuntimeQuant::None => Ok(false),
        RuntimeQuant::Int4 => Ok(true),
        other => fail(format!(
            "Metal {schema}: runtime_quant={other:?} has no encoder here; these \
             kernels read MLX affine, so `int4` is the only request they can serve"
        )),
    }
}

pub fn push_direct(b: &mut Builder<'_>, raw: &RawTensor, output: String) {
    if is_raw(&raw.encoding, DType::F16) || is_raw(&raw.encoding, DType::F32) {
        let bf16 = Encoding::Raw(DType::BF16);
        b.define(
            output,
            Expr::src(&raw.name).cast(bf16.clone()),
            bf16,
            Some(raw.shape.clone()),
        );
        return;
    }
    b.define(
        output,
        Expr::src(&raw.name),
        raw.encoding.clone(),
        Some(raw.shape.clone()),
    );
}

pub fn affine_encoding(bits: u32, group_size: u32) -> Encoding {
    Encoding::Quant(QuantSpec {
        scheme: if bits == 4 {
            QuantScheme::MlxAffineU4
        } else {
            QuantScheme::Int8Asymmetric
        },
        logical_dtype: DType::BF16,
        bits_per_element: bits as u8,
        group_size,
        channel_axis: Some(Axis(1)),
    })
}

pub const AFFINE_GROUP: i64 = 64;

pub fn push_mlx_affine_stacked(
    b: &mut Builder<'_>,
    raw: &RawTensor,
    scales: &RawTensor,
    biases: &RawTensor,
    declared_bits_hint: i64,
    declared_group_size: i64,
    output: String,
) -> Result<(), Error> {
    if raw.shape.len() < 2 || scales.shape.len() != raw.shape.len() || biases.shape != scales.shape
    {
        return fail(format!(
            "MLX affine triplet '{}' has incompatible shapes",
            raw.name
        ));
    }
    if !is_raw(&raw.encoding, DType::U32) {
        return fail(format!(
            "MLX affine triplet '{}' is not the U32 words this format packs into",
            raw.name
        ));
    }
    let mut rows = 1i64;
    for (index, extent) in raw.shape[..raw.shape.len() - 1].iter().enumerate() {
        if *extent != scales.shape[index] {
            return fail(format!(
                "MLX affine triplet '{}' disagrees with its scales on the stacked axes",
                raw.name
            ));
        }
        rows *= extent;
    }
    let groups = *scales.shape.last().expect("rank checked above");
    if groups <= 0 {
        return fail(format!("MLX affine triplet '{}' has no groups", raw.name));
    }

    let mut logical_cols;
    let mut bits = declared_bits_hint;
    if declared_group_size > 0 {
        logical_cols = groups * declared_group_size;
        let packed_bits = raw.shape.last().expect("rank checked") * 32;
        if logical_cols <= 0 || packed_bits % logical_cols != 0 {
            return fail(format!(
                "MLX affine triplet '{}' cannot derive a width from groups of {}",
                raw.name, declared_group_size
            ));
        }
        bits = packed_bits / logical_cols;
    } else {
        logical_cols = 0;
    }
    if bits != 4 && bits != 8 {
        return fail(format!(
            "MLX affine triplet '{}' has an unsupported width ({bits} bits)",
            raw.name
        ));
    }
    if declared_group_size <= 0 {

        logical_cols = raw.shape.last().expect("rank checked") * (32 / bits);
        if logical_cols % groups != 0 {
            return fail(format!(
                "MLX affine triplet '{}' cannot derive a group size",
                raw.name
            ));
        }
    }
    let group_size = u32::try_from(logical_cols / groups)
        .map_err(|_| Error::Contract("MLX affine group size does not fit u32".into()))?;

    let encoding = affine_encoding(bits as u32, group_size);
    b.define(
        output,
        Expr::src(&raw.name).transmute(TensorType::new(vec![rows, logical_cols], encoding.clone())),
        encoding,
        Some(vec![rows, logical_cols]),
    );
    Ok(())
}

pub fn push_mlx_affine_declared(
    b: &mut Builder<'_>,
    raw: &RawTensor,
    scales: &RawTensor,
    biases: &RawTensor,
    declared_bits: i64,
    declared_group_size: i64,
    output: String,
) -> Result<(), Error> {
    let bits = if declared_bits > 0 { declared_bits } else { 4 };
    push_mlx_affine_stacked(b, raw, scales, biases, bits, declared_group_size, output)
}

pub fn push_mlx_mxfp4_stacked(
    b: &mut Builder<'_>,
    raw: &RawTensor,
    scales: &RawTensor,
    output: String,
) -> Result<(), Error> {
    if raw.shape.len() < 2 || scales.shape.len() != raw.shape.len() {
        return fail(format!(
            "MXFP4 pair '{}' and its scales differ in rank",
            raw.name
        ));
    }
    if !is_raw(&scales.encoding, DType::U8) {
        return fail(format!(
            "MXFP4 pair '{}' has scales that are not the U8 E8M0 block exponents \
             this format stores",
            raw.name
        ));
    }
    let mut rows = 1i64;
    for (index, extent) in raw.shape[..raw.shape.len() - 1].iter().enumerate() {
        if *extent != scales.shape[index] {
            return fail(format!(
                "MXFP4 pair '{}' disagrees with its scales on the stacked axes",
                raw.name
            ));
        }
        rows *= extent;
    }
    let groups = *scales.shape.last().expect("rank checked");
    if groups <= 0 || *raw.shape.last().expect("rank checked") != groups * 4 {
        return fail(format!(
            "MXFP4 pair '{}' packs {} words against {groups} blocks, and eight \
             nibbles to a word over 32-element blocks needs {}",
            raw.name,
            raw.shape.last().expect("rank checked"),
            groups * 4
        ));
    }
    let cols = groups * 32;

    let encoding = Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Mxfp4E2M1E8M0,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    });
    b.define(
        output,
        Expr::src(&raw.name).transmute(TensorType::new(vec![rows, cols], encoding.clone())),
        encoding,
        Some(vec![rows, cols]),
    );
    Ok(())
}

pub fn push_encoded_affine(
    b: &mut Builder<'_>,
    value: Expr,
    rows: i64,
    cols: i64,
    output: String,
) -> Result<(), Error> {
    if cols % AFFINE_GROUP != 0 {
        return fail(format!(
            "Metal: '{output}' has {cols} columns, which these group-64 kernels \
             cannot quantize"
        ));
    }
    let encoding = affine_encoding(4, AFFINE_GROUP as u32);
    b.define(
        output,
        value.cast(encoding.clone()),
        encoding,
        Some(vec![rows, cols]),
    );
    Ok(())
}

pub fn mxfp4_values(
    b: &mut Builder<'_>,
    blocks: Expr,
    scales: Expr,
    rows: i64,
    groups: i64,
    scales_tensor: String,
) -> Expr {
    let cols = groups * 32;
    let group_shape = vec![rows, groups];
    let e8m0 = Encoding::Raw(DType::E8M0);
    let declared = b.define(
        scales_tensor.clone(),
        scales.transmute(TensorType::new(group_shape.clone(), e8m0.clone())),
        e8m0,
        Some(group_shape),
    );
    b.mark_internal(declared);

    let quant = Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Mxfp4E2M1E8M0,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    });
    blocks
        .transmute(TensorType::new(vec![rows, cols], quant))
        .scale_per_block(Expr::out(&scales_tensor))
}

pub fn routed_expert_member(
    raw_name: &str,
    member: &str,
    schema: &str,
    has_shared_expert: bool,
) -> Result<Option<String>, Error> {
    const SWITCH: &str = "mlp.switch_mlp.";
    if let Some(rest) = member.strip_prefix(SWITCH) {
        return Ok(Some(format!("mlp.experts.{rest}")));
    }
    if has_shared_expert {
        for ok in ["mlp.shared_expert.", "mlp.shared_expert_gate."] {
            if member.starts_with(ok) {
                return Ok(Some(member.to_string()));
            }
        }
    }
    for shared in [
        "mlp.shared_expert.",
        "mlp.shared_expert_gate.",
        "mlp.shared_experts.",
    ] {
        if member.starts_with(shared) {
            return fail(format!(
                "Metal {schema} schema has no shared expert, but '{raw_name}' is one: \
                 this driver would load it and never read it, running the routed \
                 mixture alone"
            ));
        }
    }
    const EXPERTS: &str = "mlp.experts.";
    if let Some(rest) = member.strip_prefix(EXPERTS)
        && rest.chars().next().is_some_and(|c| c.is_ascii_digit())
    {
        return fail(format!(
            "Metal {schema} schema needs the routed experts stacked on axis 0 \
             (one `mlp.experts.gate_proj` per layer, expert-major), but \
             '{raw_name}' is per-expert"
        ));
    }
    Ok(None)
}

pub fn layer_member<'n>(
    rest: &'n str,
    schema: &str,
    raw_name: &str,
) -> Result<(&'n str, &'n str), Error> {
    const LAYERS: &str = "layers.";
    let Some(tail) = rest.strip_prefix(LAYERS) else {
        return fail(format!(
            "Metal {schema} schema has no declared mapping or skip for '{raw_name}'"
        ));
    };
    let Some(dot) = tail.find('.') else {
        return fail(format!(
            "Metal {schema} layer tensor '{raw_name}' is malformed"
        ));
    };
    let layer = &tail[..dot];
    if layer.is_empty() || !layer.chars().all(|c| c.is_ascii_digit()) {
        return fail(format!(
            "Metal {schema} layer tensor '{raw_name}' has an invalid layer index"
        ));
    }
    Ok((layer, &tail[dot + 1..]))
}

pub type RenameRule<'r> = &'r dyn Fn(&Builder<'_>, &str) -> Result<Option<String>, Error>;

fn unpacks_to_bf16(encoding: &Encoding) -> bool {
    matches!(
        encoding,
        Encoding::Quant(spec)
            if spec.scheme.is_self_contained() && spec.logical_dtype == DType::BF16
    )
}

pub fn author_mlx_file(
    b: &mut Builder<'_>,
    schema: &str,
    rename: RenameRule<'_>,
) -> Result<(), Error> {
    let quant_bits = i64::from(b.encoding().bits);
    let quant_group = i64::from(b.encoding().group_size);
    let encode_floats = int4_requested(b, schema)?;

    if !encode_floats && !b.tensors().iter().any(|t| t.name.ends_with(".scales")) {
        return fail(format!(
            "Metal {schema} needs quantized weights: this checkpoint carries no \
             `.scales` tensors, so it is unquantized (bf16/fp16), and the Metal \
             driver binds every projection through its affine-U4 path. Either \
             build it with `--quant int4` to encode the weights now, or use a \
             pre-quantized repo -- the `mlx-community/*-4bit` conversions are \
             the ones this path is built for."
        ));
    }
    let mut declared = 0usize;
    for raw in b.tensors().to_vec() {
        let Some(output) = rename(b, &raw.name)? else {
            continue;
        };
        if raw.name.ends_with(".weight") && is_raw(&raw.encoding, DType::U32) {
            let base = &raw.name[..raw.name.len() - ".weight".len()];
            let (Some(scales), Some(biases)) = (
                b.find(&format!("{base}.scales")),
                b.find(&format!("{base}.biases")),
            ) else {
                return fail(format!(
                    "Metal affine-U4 weight '{}' is missing scales or biases",
                    raw.name
                ));
            };
            push_mlx_affine_declared(b, raw, scales, biases, quant_bits, quant_group, output)?;
        } else if encode_floats && raw.name.ends_with(".weight") && raw.shape.len() == 2 {
            let value = if is_raw(&raw.encoding, DType::BF16) {
                Expr::src(&raw.name)
            } else if is_raw(&raw.encoding, DType::F16)
                || is_raw(&raw.encoding, DType::F32)
                || unpacks_to_bf16(&raw.encoding)
            {

                let widened = format!("{output}.bf16");
                let bf16 = Encoding::Raw(DType::BF16);
                let declared = b.define(
                    widened.clone(),
                    Expr::src(&raw.name).cast(bf16.clone()),
                    bf16,
                    Some(raw.shape.clone()),
                );
                b.mark_internal(declared);
                Expr::out(&widened)
            } else {
                push_direct(b, raw, output);
                declared += 1;
                continue;
            };
            push_encoded_affine(b, value, raw.shape[0], raw.shape[1], output)?;
        } else {
            push_direct(b, raw, output);
        }
        declared += 1;
    }
    if declared == 0 {
        return fail(format!("Metal {schema} schema found no decoder tensors"));
    }
    Ok(())
}
