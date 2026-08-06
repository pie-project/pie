//! The Metal driver's lowering toolkit: bind-in-place, MLX names.
//!
//! Ported from `driver/metal/src/model/contract_detail.hpp`. Where the CUDA
//! lowering fuses, shards and requantizes, this one renames and binds what
//! the file holds — [`Naming::Mlx`](crate::policy::Naming) selects
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
use super::policy::RuntimeQuant;

/// Refuse, naming the tensor. `pub` for the same reason `builder::fail` is:
/// an MLX generation refuses for its own reasons and the message has to read
/// the same however it got there.
pub fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
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

/// Is this the name a Metal lowering already produced?
///
/// The bind path's namespace is closed and small — `layers.<n>.*` plus a
/// handful of layer-less tables — and it is disjoint from the checkpoint
/// namespace, where every decoder tensor arrives under `model.` or one of the
/// two `language_model.` spellings. That disjointness is what makes an
/// identity arm a fact rather than a guess: a name in this set did not come
/// from a checkpoint.
///
/// Why it is needed at all: a serve boot re-authors the contract from the
/// names its checkpoint holds, so an artifact `pie model build` wrote —
/// whose tensors ARE the runtime tensors — is fed back through the very rename
/// that produced them. Without an identity arm each schema refuses its own
/// output (`no declared mapping or skip for 'final_norm.weight'`) and the
/// artifact cannot boot. `f(f(x)) == f(x)` is the property, and it is what
/// makes import → optimize → serve a pipeline rather than three dead ends.
///
/// `lm_head.` is deliberately absent. It is the one name living in both
/// namespaces, and every family already answers it before reaching here —
/// untied it maps to itself, tied it becomes `shared_embedding` — so an
/// identity arm covering it would silently break the tied case for real
/// checkpoints.
///
/// What this does NOT do is re-run the refusals `routed_expert_member` applies
/// to a checkpoint's expert banks. Those describe shapes a *checkpoint* can
/// arrive in; a lowered artifact holds only what the author already accepted,
/// and a hand-written one that did not would fail at bind for want of the
/// stacked bank instead.
pub fn already_lowered(raw_name: &str) -> bool {
    for table in [
        "shared_embedding.",
        "embed_tokens.",
        // gemma4's per-layer embedding table and its two projections:
        // layer-less, and they keep their own names through the lowering.
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
    // `layers.<digits>.` and nothing looser. The digits are what separate a
    // lowered name from a checkpoint that merely begins with the same word.
    let Some(tail) = raw_name.strip_prefix("layers.") else {
        return false;
    };
    let Some(dot) = tail.find('.') else {
        return false;
    };
    let index = &tail[..dot];
    !index.is_empty() && index.chars().all(|c| c.is_ascii_digit())
}

/// Which requantization this lowering can serve, and the refusal for the rest.
///
/// `None` and `Int4` are the only two answers these kernels have: they read
/// MLX affine, and `Encode` is implemented for `MlxAffineU4` on both the host
/// and the driver. Anything else is CUDA's vocabulary, and it is refused
/// rather than ignored — authoring an unquantized contract for `--quant int8`
/// would hand back an artifact whose name and bytes disagree. That was the
/// state this field was in: plumbed through three layers to authors that never
/// looked at it.
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

/// A family's naming rule: raw checkpoint name in, runtime name out —
/// `Ok(None)` for a tensor the schema has no opinion on, `Err` for one it
/// refuses.
pub type RenameRule<'r> = &'r dyn Fn(&Builder<'_>, &str) -> Result<Option<String>, Error>;

/// The shared authoring loop for the affine-triplet families (llama,
/// qwen3.5, gemma4): rename every tensor, pair the U32 weights with their
/// scales and biases, cast the widths no kernel reads, and refuse a
/// checkpoint that declares nothing.
///
/// `rename` answers with the runtime name, `None` to skip, or an error for a
/// tensor the schema has no opinion on — the same trichotomy every Metal
/// header states.
///
/// `RuntimeQuant::Int4` adds one arm: a rank-2 `.weight` the checkpoint left
/// in a float type is encoded to affine-U4 instead of declared as values. It
/// is the same rule gpt-oss applies unconditionally — its published checkpoint
/// mixes MXFP4 experts with BF16 attention, and the matvecs are quantized — and
/// the same `Encode` op, so what runs at load without a request and offline
/// under `pie model build --quant int4` is one transform, not two. Rank is
/// what separates these from the norms, which must stay values.
///
/// F16 and F32 are cast to BF16 first rather than encoded where they lie. The
/// two executors would not agree otherwise: the host reads the operand's
/// declared dtype, while the driver's `encode_mlx_affine_u4` is handed a byte
/// width and reads every 2-byte element as BF16 — so an F16 weight would be
/// quantized correctly offline and from misread bits at load. `Cast` is in
/// both tile-map masks and is what `push_direct` already does with these two
/// widths, so the chain costs nothing and removes the disagreement.
pub fn author_mlx_file(
    b: &mut Builder<'_>,
    schema: &str,
    rename: RenameRule<'_>,
) -> Result<(), Error> {
    let quant_bits = i64::from(b.facts().quant_bits);
    let quant_group = i64::from(b.facts().quant_group_size);
    let encode_floats = int4_requested(b, schema)?;
    // These three families bind every projection through Metal's affine-U4
    // path (`push_quant` in `driver/metal/src/loader/heap_bind.cpp` asks for
    // `.weight`/`.scales`/`.biases` unconditionally). A bf16 checkpoint has no
    // `.scales` at all, so without a request to encode them it used to author a
    // contract that looked fine and then died deep in the loader with `llama
    // bind: unstaged weight embed_tokens.scales` -- a message about an internal
    // staging table, from which nothing about the actual problem is
    // recoverable. Say it here, where the checkpoint is still in view.
    //
    // Only when nothing will supply those tensors: under `--quant int4` the
    // floats are encoded below, which is the other, equally valid way to serve
    // an unquantized release.
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
            } else if is_raw(&raw.encoding, DType::F16) || is_raw(&raw.encoding, DType::F32) {
                // Two tile maps, not one expression: `Cast` needs a kernel, so
                // a cast feeding an encode has to leave a buffer behind for the
                // encode to read rather than nest inside it.
                let widened = format!("{output}.bf16");
                let bf16 = Encoding::Raw(DType::BF16);
                if let Some(declared) = b.define(
                    widened.clone(),
                    Expr::src(&raw.name).cast(bf16.clone()),
                    bf16,
                    Some(raw.shape.clone()),
                ) {
                    b.mark_internal(declared);
                }
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

#[cfg(test)]
mod tests {
    use super::{already_lowered, routed_expert_member};

    /// `f(f(x)) == f(x)`, checked on the names the four Metal schemas emit.
    ///
    /// The property is what lets a serve boot re-author from an artifact whose
    /// tensors are already the runtime tensors. Stated over `already_lowered`
    /// rather than over each family's rename because the four call it at the
    /// same point and differ only in what they do *before* it.
    #[test]
    fn the_runtime_namespace_is_recognized_and_the_checkpoint_one_is_not() {
        for lowered in [
            "layers.0.self_attn.q_proj.weight",
            "layers.31.mlp.experts.gate_proj.scales",
            "layers.7.input_layernorm.weight",
            "final_norm.weight",
            "embed_tokens.weight",
            "shared_embedding.weight",
            "embed_tokens_per_layer.weight",
            "per_layer_model_projection.weight",
            "per_layer_projection_norm.weight",
        ] {
            assert!(already_lowered(lowered), "{lowered} is a runtime name");
        }
        for checkpoint in [
            // Every decoder tensor arrives under a wrapper. That is the
            // disjointness the identity arm rests on.
            "model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "language_model.model.norm.weight",
            "model.norm.weight",
            "model.embed_tokens.weight",
            // `lm_head.` is excluded on purpose: tied, it is NOT an identity.
            "lm_head.weight",
            // `layers.` without an index is not a lowered name, and neither is
            // a bare word that merely starts with it.
            "layers.weight",
            "layers.attn.weight",
            "layersomething.weight",
            "layers.",
        ] {
            assert!(
                !already_lowered(checkpoint),
                "{checkpoint} is not a runtime name"
            );
        }
    }

    /// The refusal that used to be pinned from Metal's C++ side
    /// (`llama_decode_step_test.cpp`), ported here when the C++ author died:
    /// a shared-expert block this driver does not compute, or an unstacked
    /// per-expert bank, is an error naming the tensor — never a silent skip,
    /// because skipping is what runs the routed mixture alone and produces
    /// fluent text that is not the checkpoint's.
    #[test]
    fn routed_banks_map_and_unservable_experts_are_refused() {
        let call = |member: &str, has_shared: bool| {
            routed_expert_member(
                &format!("model.layers.3.{member}"),
                member,
                // The schema name reaches this function only to be quoted in
                // a diagnostic, so the test states a name rather than a
                // family: `common/` names no family, and the guard in
                // `tests/common_is_thin.rs` holds it to that.
                "test",
                has_shared,
            )
        };
        // The stacked routed bank, in both spellings, still maps.
        assert!(matches!(
            call("mlp.experts.gate_proj.weight", false),
            Ok(None)
        ));
        assert_eq!(
            call("mlp.switch_mlp.gate_proj.weight", false).unwrap(),
            Some("mlp.experts.gate_proj.weight".to_string()),
            "mlx_lm's spelling of the same bank maps"
        );
        // Per-expert tensors are refused, not skipped.
        let err = call("mlp.experts.0.gate_proj.weight", false).unwrap_err();
        assert!(err.to_string().contains("per-expert"), "{err}");
        // A shared-expert block for a family that computes none is refused...
        let err = call("mlp.shared_experts.gate_up_proj.weight", false).unwrap_err();
        assert!(err.to_string().contains("shared expert"), "{err}");
        // ...and accepted verbatim for one that does.
        assert_eq!(
            call("mlp.shared_expert.gate_proj.weight", true).unwrap(),
            Some("mlp.shared_expert.gate_proj.weight".to_string())
        );
    }
}
