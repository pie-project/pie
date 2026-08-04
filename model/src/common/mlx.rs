//! The Metal driver's lowering toolkit: bind-in-place, MLX names.
//!
//! Ported from `driver/metal/src/model/contract_detail.hpp`. Where the CUDA
//! lowering fuses, shards and requantizes, this one renames and binds what
//! the file holds — [`Naming::Mlx`](crate::common::policy::Naming) selects
//! it, and the same family authors serve both by branching on the policy.
//!
//! The one transform family here is the MLX quantization vocabulary:
//! affine-U4/U8 triplets (`.weight`/`.scales`/`.biases`) declared by
//! transmute, shipped MXFP4 pairs declared without decoding, and — for the
//! projections a published checkpoint left in BF16 — a load-time encode into
//! the affine layout the matvecs read.

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::{Expr, TensorType};
use pie_loader::error::Error;
use pie_loader::types::{Axis, DType, Encoding, QuantScheme, QuantSpec};

use super::builder::{Builder, is_raw};

pub(crate) fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// Whether `raw_name` is `member` under the optional `model.` wrapper.
pub fn has_wrapper_member(raw_name: &str, member: &str) -> bool {
    raw_name.starts_with(member)
        || raw_name
            .strip_prefix("model.")
            .is_some_and(|rest| rest.starts_with(member))
}

/// The text decoder's member, with whichever wrapper prefix spelled it
/// stripped.
///
/// `model.language_model.*` (HF) and `language_model.model.*` (`mlx_lm`) are
/// the two spellings; they SWAP the two words rather than one merely adding
/// a prefix. Only the prefix differs — everything downstream sees the same
/// member string either way.
pub fn decoder_member(raw_name: &str) -> Option<&str> {
    for prefix in ["model.language_model.", "language_model.model."] {
        if let Some(rest) = raw_name.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

/// Declare a tensor where it lies, casting the widths no Metal kernel reads.
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

/// The encoding this driver's quantized matvecs read.
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

/// The columns this driver's kernels group under one scale.
pub const AFFINE_GROUP: i64 = 64;

/// Declare an MLX affine weight whose leading axes are a STACK.
///
/// A sparse-MoE checkpoint stores one tensor per projection with the expert
/// on axis 0 — `[n_experts, out, in/pack]` — rather than `n_experts`
/// matrices. Rank 2 is the stacked case with an empty stack, so there is one
/// implementation.
///
/// Three numbers — width, group, packed columns — and the shapes pin only
/// their product. Exactly one has to be told: given the group (what
/// `config.json` states for the whole file) the width is derived, which
/// reads `mlx_lm`'s per-tensor overrides for free; given the width (gpt-oss
/// states no quantization at all) the group is derived instead.
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
        // gpt-oss states no quantization at all, so here the width is the
        // told number and the group is the derived one — the same equation
        // solved for the other unknown.
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

/// [`push_mlx_affine_stacked`] with the historical 4-bit default for a
/// config that declares nothing.
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

/// Declare an MXFP4 weight the checkpoint SHIPPED, without decoding it.
///
/// `mlx_lm` writes MXFP4 as a `.weight` of U32 — eight nibbles to a
/// little-endian word — beside a U8 `.scales` of E8M0 block exponents, and
/// no `.biases`. This is a transmute and not a decode: the bytes staged into
/// the heap are the checkpoint's own.
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

/// Declare a weight the LOADER quantizes, rather than one the checkpoint
/// shipped quantized: a `cast` to the affine encoding, whose encode writes
/// `<stem>.scales` and `<stem>.biases` beside its output as part of the same
/// pass.
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

/// The BF16 values behind an MXFP4 `_blocks`/`_scales` pair.
///
/// Two nodes and no kernel of this driver's own: the contract says the
/// packed bytes are E2M1 nibbles under E8M0 block scales, and the loader's
/// dequantizer turns that declaration into values. The scales have to be
/// *declared* before they can be scaled by, so this leaves an internal
/// tensor behind under `scales_tensor`.
pub fn mxfp4_values(
    b: &mut Builder<'_>,
    blocks: Expr,
    scales: Expr,
    rows: i64,
    cols: i64,
    scales_tensor: String,
) -> Result<Expr, Error> {
    if cols % 32 != 0 {
        return fail(format!(
            "MXFP4 tensor '{scales_tensor}' has {cols} columns, which is not a \
             whole number of 32-element blocks"
        ));
    }
    let groups = vec![rows, cols / 32];
    let e8m0 = Encoding::Raw(DType::E8M0);
    if let Some(declared) = b.define(
        scales_tensor.clone(),
        scales.transmute(TensorType::new(groups.clone(), e8m0.clone())),
        e8m0,
        Some(groups),
    ) {
        b.mark_internal(declared);
    }

    let quant = Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Mxfp4E2M1E8M0,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    });
    Ok(blocks
        .transmute(TensorType::new(vec![rows, cols], quant))
        .scale_per_block(Expr::out(&scales_tensor)))
}

/// The one rule every routed family's mixture is named by.
///
/// A routed FFN must arrive with its experts STACKED on axis 0, which is
/// what `affine_qmv_routed` indexes. Two spellings are accepted
/// (`mlp.switch_mlp.*` from `mlx_lm`, `mlp.experts.*` from the fused HF
/// export); the unstacked bank and — for a family that computes none — the
/// shared expert are refused rather than skipped, because skipping is what
/// silently produces the wrong model.
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
    if let Some(rest) = member.strip_prefix(EXPERTS) {
        if rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            return fail(format!(
                "Metal {schema} schema needs the routed experts stacked on axis 0 \
                 (one `mlp.experts.gate_proj` per layer, expert-major), but \
                 '{raw_name}' is per-expert"
            ));
        }
    }
    Ok(None)
}

/// Split `layers.N.member` off a decoder-relative name, validating the index.
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

/// The shared authoring loop for the affine-triplet families (llama,
/// qwen3.5, gemma4): rename every tensor, pair the U32 weights with their
/// scales and biases, cast the widths no kernel reads, and refuse a
/// checkpoint that declares nothing.
///
/// `rename` answers with the runtime name, `None` to skip, or an error for a
/// tensor the schema has no opinion on — the same trichotomy every Metal
/// header states.
pub fn author_mlx_file(
    b: &mut Builder<'_>,
    schema: &str,
    rename: &dyn Fn(&Builder<'_>, &str) -> Result<Option<String>, Error>,
) -> Result<(), Error> {
    let quant_bits = i64::from(b.facts().mlx_quant_bits);
    let quant_group = i64::from(b.facts().mlx_quant_group_size);
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
