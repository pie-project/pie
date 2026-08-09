//! The Metal driver's lowering toolkit: bind-in-place, MLX names.
//!
//! Ported from `crates/driver-metal/csrc/src/model/contract_detail.hpp`. Where the CUDA
//! lowering fuses, shards and requantizes, this one renames and binds what
//! the file holds — [`Naming::Mlx`](crate::shared::policy::Naming) selects
//! it, and the same family authors serve both by branching on the policy.
//!
//! The one transform family here is the MLX quantization vocabulary:
//! affine-U4/U8 triplets (`.weight`/`.scales`/`.biases`) declared by
//! transmute, shipped MXFP4 pairs declared without decoding, and — for the
//! projections a published checkpoint left in BF16 — a load-time encode into
//! the affine layout the matvecs read.

use model_loader::checkpoint::RawTensor;
use model_loader::contract::{Expr, TensorType};
use model_loader::error::Error;
use model_loader::types::{Axis, DType, Encoding, QuantScheme, QuantSpec};

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
///
/// The `* 32` in that arithmetic is the one thing the shapes cannot say:
/// it is the width of a word, and it is only right if the tensor is U32.
/// The sole caller gates on that before calling, but the function is what
/// the arithmetic belongs to, so it checks rather than assumes — the same
/// way [`push_mlx_mxfp4_stacked`] checks its scales are the U8 exponents
/// it reads them as.
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
///
/// The width arrives as a GROUP count, not a column count. Every caller
/// reads it off the packed tensor's own group axis and would have had to
/// multiply by 32 to state columns, so taking columns meant a runtime
/// check that the number divided back -- a refusal no caller could
/// reach, guarding an arithmetic identity. Stated this way the whole
/// number of blocks is the only representable width.
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
    let quant_bits = i64::from(b.encoding().bits);
    let quant_group = i64::from(b.encoding().group_size);
    let encode_floats = int4_requested(b, schema)?;
    // These three families bind every projection through Metal's affine-U4
    // path (`push_quant` in `crates/driver-metal/csrc/src/loader/heap_bind.cpp` asks for
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

#[cfg(test)]
mod tests {
    use super::{
        Builder, DType, Encoding, Error, RawTensor, RenameRule, already_lowered, author_mlx_file,
        layer_member, routed_expert_member,
    };
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::{Policy, RuntimeQuant};
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;
    use model_loader::types::{BackendKind, FileId, TensorId};

    const HIDDEN: i64 = 64;
    const ROWS: i64 = 32;

    fn tensor(t: &mut Vec<RawTensor>, name: &str, shape: Vec<i64>, encoding: Encoding) {
        let elements: i64 = shape.iter().product();
        t.push(RawTensor {
            id: TensorId(u32::try_from(t.len()).expect("a small fixture")),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: u64::try_from(elements).unwrap_or(0),
            shape,
            encoding,
        });
    }

    /// A rename that answers for anything ending in `.weight`, so a test can
    /// pick what the LOOP does without also testing a family's table.
    fn passthrough(_b: &Builder<'_>, name: &str) -> Result<Option<String>, Error> {
        Ok(Some(name.to_string()))
    }

    fn run(
        tensors: Vec<RawTensor>,
        quant: RuntimeQuant,
        rename: RenameRule<'_>,
    ) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        // 4 bits over groups of 32: the width `push_mlx_affine_declared`
        // would otherwise assume, stated so the fixture is not resting on
        // the fallback the encoding's own doc calls expensive.
        let enc = StoredEncoding {
            method: "affine".to_string(),
            bits: 4,
            group_size: 32,
        };
        let target = StorageTarget::for_backend(BackendKind::Metal, 0, 1);
        let policy = Policy {
            runtime_quant: quant,
            ..Policy::default()
        };
        let shape = LoadShape {
            layers: 1,
            head_dim: 0,
            n_experts: 0,
            mamba_groups: 0,
            kv_shared_layers: 0,
            tied_embeddings: true,
        };
        let mut b = Builder::new(&meta, "mlx-test", shape, &enc, &target, &policy);
        author_mlx_file(&mut b, "Test", rename)?;
        b.finish()
    }

    fn refusal(tensors: Vec<RawTensor>, quant: RuntimeQuant, rename: RenameRule<'_>) -> String {
        match run(tensors, quant, rename) {
            Ok(_) => panic!("expected a refusal"),
            Err(e) => e.to_string(),
        }
    }

    /// One quantized projection: the affine-U4 triplet Metal binds.
    fn affine_triplet() -> Vec<RawTensor> {
        let mut t = Vec::new();
        tensor(
            &mut t,
            "layers.0.self_attn.q_proj.weight",
            vec![ROWS, HIDDEN / 8],
            Encoding::Raw(DType::U32),
        );
        tensor(
            &mut t,
            "layers.0.self_attn.q_proj.scales",
            vec![ROWS, HIDDEN / 32],
            Encoding::Raw(DType::BF16),
        );
        tensor(
            &mut t,
            "layers.0.self_attn.q_proj.biases",
            vec![ROWS, HIDDEN / 32],
            Encoding::Raw(DType::BF16),
        );
        t
    }

    fn names(c: &ModelContract) -> Vec<&str> {
        c.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    // ── The refusal that replaced a message from inside the loader ───────

    /// An unquantized checkpoint with no `--quant int4` is refused HERE,
    /// where the checkpoint is still in view. The Metal binder asks every
    /// projection for `.scales` unconditionally, so without this the load
    /// authored cleanly and then died on `unstaged weight embed_tokens.scales`
    /// -- a sentence about an internal staging table, from which nothing
    /// about the actual problem is recoverable.
    #[test]
    fn an_unquantized_checkpoint_is_refused_before_the_binder_sees_it() {
        let mut t = Vec::new();
        tensor(
            &mut t,
            "layers.0.self_attn.q_proj.weight",
            vec![ROWS, HIDDEN],
            Encoding::Raw(DType::BF16),
        );
        let m = refusal(t, RuntimeQuant::None, &passthrough);
        assert!(m.contains("needs quantized weights"), "{m}");
        // The refusal has to carry the way OUT, not just the diagnosis:
        // both remedies are named.
        assert!(m.contains("--quant int4"), "{m}");
        assert!(m.contains("4bit"), "{m}");
        assert!(m.contains("Test"), "the refusal names the schema: {m}");
    }

    /// The same checkpoint under `--quant int4` is accepted, because the
    /// floats are encoded below. The guard reads a REQUEST, not a file
    /// property, and that is the whole reason it is two conditions.
    #[test]
    fn the_same_checkpoint_loads_when_the_weights_will_be_encoded() {
        let mut t = Vec::new();
        tensor(
            &mut t,
            "layers.0.self_attn.q_proj.weight",
            vec![ROWS, HIDDEN],
            Encoding::Raw(DType::BF16),
        );
        let c = run(t, RuntimeQuant::Int4, &passthrough).expect("int4 encodes the floats");
        assert!(names(&c).contains(&"layers.0.self_attn.q_proj.weight"));
    }

    /// One `.scales` anywhere is enough to pass the gate. Stated because
    /// the check is an `any`, not a per-tensor rule -- a partly quantized
    /// checkpoint gets through here and is caught tensor by tensor below.
    #[test]
    fn a_single_scales_tensor_anywhere_satisfies_the_gate() {
        let c = run(affine_triplet(), RuntimeQuant::None, &passthrough).expect("a quantized file");
        assert!(names(&c).contains(&"layers.0.self_attn.q_proj.weight"));
    }

    // ── What the loop does with each kind of tensor ──────────────────────

    /// A packed U32 weight whose companions are missing is refused rather
    /// than pushed direct. Metal would bind the U32 bytes as if they were
    /// the weight -- there is no shape check that would notice, since a
    /// packed `[rows, cols/8]` is a perfectly ordinary tensor.
    #[test]
    fn a_packed_weight_missing_its_scales_is_refused() {
        let mut t = affine_triplet();
        t.retain(|r| r.name != "layers.0.self_attn.q_proj.scales");
        // A second projection keeps its scales, so the file is still
        // "quantized" as the gate above measures it -- an `any`, not a
        // per-tensor rule. Without this decoy the gate fires first and the
        // test asserts nothing about the loop.
        tensor(
            &mut t,
            "layers.0.mlp.down_proj.scales",
            vec![ROWS, 2],
            Encoding::Raw(DType::BF16),
        );
        let m = refusal(t, RuntimeQuant::None, &passthrough);
        assert!(m.contains("missing scales or biases"), "{m}");
        assert!(
            m.contains("q_proj.weight"),
            "the refusal names the tensor: {m}"
        );
    }

    /// And the biases half. Separate test because the two are one `let
    /// else` over a tuple, and a fixture damaging only one of them
    /// exercises only one probe.
    #[test]
    fn a_packed_weight_missing_its_biases_is_refused() {
        let mut t = affine_triplet();
        t.retain(|r| !r.name.ends_with(".biases"));
        // The gate above passes -- `.scales` is still present -- so this
        // really is the loop's own refusal and not the unquantized one.
        let m = refusal(t, RuntimeQuant::None, &passthrough);
        assert!(m.contains("missing scales or biases"), "{m}");
    }

    /// An F32 weight under `--quant int4` is CAST before it is encoded,
    /// and the cast lands in its own tensor rather than nesting inside the
    /// encode: `Cast` needs a kernel, so it has to leave a buffer behind
    /// for the encode to read. The intermediate is marked internal, which
    /// is what keeps it out of the runtime namespace.
    #[test]
    fn a_float32_weight_is_widened_through_a_named_intermediate() {
        let mut t = Vec::new();
        tensor(
            &mut t,
            "layers.0.mlp.down_proj.weight",
            vec![ROWS, HIDDEN],
            Encoding::Raw(DType::F32),
        );
        let c = run(t, RuntimeQuant::Int4, &passthrough).expect("int4 encodes the floats");
        let widened = c
            .tensors
            .iter()
            .find(|x| x.name == "layers.0.mlp.down_proj.weight.bf16")
            .expect("the cast leaves a buffer behind");
        assert_eq!(widened.encoding, Encoding::Raw(DType::BF16));
        assert_eq!(
            widened.visibility,
            model_loader::contract::Visibility::Internal
        );
        // Declaring the F32 source AS bf16 would satisfy every line above
        // and read every element wrong, so the node itself is the assertion.
        assert!(
            matches!(&widened.expr, model_loader::contract::Expr::Cast { .. }),
            "the intermediate converts rather than relabels: {:?}",
            widened.expr
        );
    }

    /// A BF16 weight takes the same path with no cast at all -- there is
    /// nothing to widen. Pinned against the F32 case because the two
    /// differ only in whether the intermediate exists.
    #[test]
    fn a_bfloat16_weight_is_encoded_with_no_intermediate() {
        let mut t = Vec::new();
        tensor(
            &mut t,
            "layers.0.mlp.down_proj.weight",
            vec![ROWS, HIDDEN],
            Encoding::Raw(DType::BF16),
        );
        let c = run(t, RuntimeQuant::Int4, &passthrough).expect("int4 encodes the floats");
        assert!(!names(&c).contains(&"layers.0.mlp.down_proj.weight.bf16"));
        assert!(names(&c).contains(&"layers.0.mlp.down_proj.weight"));
    }

    /// A 2-D `.weight` of a dtype the encoder cannot read is pushed
    /// straight through instead of encoded. It still COUNTS as declared --
    /// the `continue` skips the encode, not the tally -- which is what
    /// keeps a checkpoint of only such tensors from hitting the
    /// "no decoder tensors" refusal below.
    #[test]
    fn a_weight_the_encoder_cannot_read_is_pushed_through_and_still_counts() {
        // The ONLY tensor in the file, so that "it still counts" is what
        // the assertion rests on: with the tally skipped, `declared` stays
        // zero and the empty-contract refusal below fires instead.
        let mut t = Vec::new();
        tensor(
            &mut t,
            "layers.0.mlp.down_proj.weight",
            vec![ROWS, HIDDEN],
            Encoding::Raw(DType::U8),
        );
        let c =
            run(t, RuntimeQuant::Int4, &passthrough).expect("an unencodable weight is not fatal");
        assert_eq!(names(&c), vec!["layers.0.mlp.down_proj.weight"]);
    }

    /// A 1-D tensor -- a norm -- is never encoded, whatever was requested.
    /// The rank condition is what separates a projection from a scale
    /// vector, and both end in `.weight`.
    #[test]
    fn a_one_dimensional_weight_is_never_encoded() {
        // Under int4, so the encode branch is LIVE and the rank is the only
        // thing keeping the norm out of it. At `RuntimeQuant::None` the
        // branch is dead and this test would assert nothing -- it did, and
        // a control deleting the rank check was silent.
        let mut t = Vec::new();
        tensor(
            &mut t,
            "layers.0.input_layernorm.weight",
            vec![HIDDEN],
            Encoding::Raw(DType::BF16),
        );
        let c = run(t, RuntimeQuant::Int4, &passthrough).expect("a norm needs no quantization");
        let norm = c
            .tensors
            .iter()
            .find(|x| x.name == "layers.0.input_layernorm.weight")
            .expect("the norm");
        assert_eq!(norm.encoding, Encoding::Raw(DType::BF16));
    }

    /// A rename that answers `None` for everything leaves the loop with
    /// nothing declared, and THAT is refused. Without it the Metal load
    /// would produce an empty contract and the driver would report every
    /// tensor missing, one at a time, instead of the one fact that
    /// explains all of them.
    #[test]
    fn a_schema_that_recognizes_nothing_is_refused_rather_than_left_empty() {
        fn nothing(_b: &Builder<'_>, _name: &str) -> Result<Option<String>, Error> {
            Ok(None)
        }
        let m = refusal(affine_triplet(), RuntimeQuant::None, &nothing);
        assert!(m.contains("found no decoder tensors"), "{m}");
        assert!(m.contains("Test"), "the refusal names the schema: {m}");
    }

    /// A rename's own refusal is propagated, not swallowed into the
    /// "found no decoder tensors" one. The `?` is the difference between a
    /// family naming the tensor it could not place and the loop reporting
    /// that the family placed nothing.
    #[test]
    fn a_rename_that_refuses_is_reported_by_its_own_words() {
        fn refuse(_b: &Builder<'_>, name: &str) -> Result<Option<String>, Error> {
            super::fail(format!("the schema will not have '{name}'"))
        }
        let m = refusal(affine_triplet(), RuntimeQuant::None, &refuse);
        assert!(m.contains("will not have"), "{m}");
        assert!(!m.contains("found no decoder tensors"), "{m}");
    }

    // ── The layer index every Metal schema splits off ────────────────────

    /// The three refusals of `layer_member` say three different things,
    /// and the difference is the whole point: a reader told "no declared
    /// mapping" goes looking for a table entry, and a reader told
    /// "malformed" does not.
    #[test]
    fn the_three_layer_index_refusals_are_told_apart() {
        let unmapped = layer_member("blocks.0.attn", "Test", "raw")
            .expect_err("not under layers.")
            .to_string();
        assert!(
            unmapped.contains("no declared mapping or skip"),
            "{unmapped}"
        );

        let no_dot = layer_member("layers.0", "Test", "raw")
            .expect_err("no member after the index")
            .to_string();
        assert!(no_dot.contains("is malformed"), "{no_dot}");

        let not_a_number = layer_member("layers.first.attn", "Test", "raw")
            .expect_err("an index that is not digits")
            .to_string();
        assert!(
            not_a_number.contains("invalid layer index"),
            "{not_a_number}"
        );

        let empty = layer_member("layers..attn", "Test", "raw")
            .expect_err("an empty index")
            .to_string();
        assert!(empty.contains("invalid layer index"), "{empty}");
    }

    /// The member is everything after the FIRST dot, dots included -- a
    /// projection is `self_attn.q_proj.weight`, not `self_attn`.
    #[test]
    fn the_member_keeps_every_dot_after_the_index() {
        let (layer, member) =
            layer_member("layers.31.self_attn.q_proj.weight", "Test", "raw").expect("a projection");
        assert_eq!(layer, "31");
        assert_eq!(member, "self_attn.q_proj.weight");
    }

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

#[cfg(test)]
mod quant_tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::Policy;
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::contract::{ModelContract, Visibility};
    use model_loader::plan::StorageTarget;
    use model_loader::types::TensorId;

    fn raw(name: &str, shape: Vec<i64>, encoding: Encoding) -> RawTensor {
        RawTensor {
            id: TensorId(0),
            name: name.to_string(),
            file_id: model_loader::types::FileId(0),
            file_offset: 0,
            span_bytes: shape.iter().product::<i64>().max(0) as u64,
            shape,
            encoding,
        }
    }

    fn u32t(name: &str, shape: Vec<i64>) -> RawTensor {
        raw(name, shape, Encoding::Raw(DType::U32))
    }

    fn u8t(name: &str, shape: Vec<i64>) -> RawTensor {
        raw(name, shape, Encoding::Raw(DType::U8))
    }

    /// Run `author` against a builder and hand back the contract it wrote.
    fn author(author: impl FnOnce(&mut Builder<'_>) -> Result<(), Error>) -> ModelContract {
        try_author(author).expect("the author was expected to succeed")
    }

    fn try_author(
        author: impl FnOnce(&mut Builder<'_>) -> Result<(), Error>,
    ) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let enc = StoredEncoding::dense();
        let policy = Policy::default();
        let shape = LoadShape::dense(1, 128, false);
        let mut b = Builder::new(&meta, "test-row", shape, &enc, &target, &policy);
        author(&mut b)?;
        b.finish()
    }

    fn declared(contract: &ModelContract, name: &str) -> (Vec<i64>, Encoding) {
        let t = contract
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("'{name}' was not declared"));
        (
            t.shape.clone().expect("a declared shape"),
            t.encoding.clone(),
        )
    }

    fn spec(encoding: &Encoding) -> QuantSpec {
        match encoding {
            Encoding::Quant(spec) => spec.clone(),
            other => panic!("expected a quantized encoding, got {other:?}"),
        }
    }

    fn message(err: Error) -> String {
        match err {
            Error::Contract(msg) => msg,
            other => panic!("expected a contract refusal, got {other:?}"),
        }
    }

    // ─── The two unknowns ────────────────────────────────────

    /// A config that states its group size fixes the logical width, and the
    /// bit depth falls out of how many words the file actually holds.
    #[test]
    fn a_stated_group_size_derives_the_width() {
        // 256 logical columns at 4 bits = 32 U32 words; at 8 bits = 64.
        for (words, bits, scheme) in [
            (32, 4u8, QuantScheme::MlxAffineU4),
            (64, 8u8, QuantScheme::Int8Asymmetric),
        ] {
            let contract = author(|b| {
                push_mlx_affine_stacked(
                    b,
                    &u32t("w", vec![512, words]),
                    &u32t("s", vec![512, 4]),
                    &u32t("bi", vec![512, 4]),
                    0,
                    64,
                    "out".into(),
                )
            });
            let (shape, encoding) = declared(&contract, "out");
            assert_eq!(shape, vec![512, 256], "{bits} bits");
            let spec = spec(&encoding);
            assert_eq!(spec.bits_per_element, bits);
            assert_eq!(spec.group_size, 64);
            assert_eq!(spec.scheme, scheme);
            assert_eq!(spec.channel_axis, Some(Axis(1)));
        }
    }

    /// gpt-oss states no quantization at all, so the width is the told
    /// number and the group is the derived one — the same equation solved
    /// for the other unknown.
    #[test]
    fn a_stated_width_derives_the_group_size() {
        // 32 words hold 256 logical columns at 4 bits and 128 at 8, so the
        // same file yields a different group at each width.
        for (bits, groups, cols, group_size) in [(4i64, 8i64, 256i64, 32u32), (8, 4, 128, 32)] {
            let contract = author(|b| {
                push_mlx_affine_stacked(
                    b,
                    &u32t("w", vec![512, 32]),
                    &u32t("s", vec![512, groups]),
                    &u32t("bi", vec![512, groups]),
                    bits,
                    0,
                    "out".into(),
                )
            });
            let (shape, encoding) = declared(&contract, "out");
            assert_eq!(shape, vec![512, cols], "{bits} bits");
            assert_eq!(spec(&encoding).group_size, group_size, "{bits} bits");
        }
    }

    /// A config that declares nothing gets 4 bits, which is what every
    /// published `mlx_lm` affine checkpoint holds.
    #[test]
    fn declaring_nothing_means_four_bits() {
        let stated = author(|b| {
            push_mlx_affine_declared(
                b,
                &u32t("w", vec![512, 32]),
                &u32t("s", vec![512, 8]),
                &u32t("bi", vec![512, 8]),
                0,
                0,
                "out".into(),
            )
        });
        let told = author(|b| {
            push_mlx_affine_stacked(
                b,
                &u32t("w", vec![512, 32]),
                &u32t("s", vec![512, 8]),
                &u32t("bi", vec![512, 8]),
                4,
                0,
                "out".into(),
            )
        });
        assert_eq!(declared(&stated, "out"), declared(&told, "out"));
    }

    /// A stacked expert bank folds its leading axes into rows rather than
    /// declaring a rank-3 tensor.
    #[test]
    fn a_stacked_bank_folds_its_leading_axes_into_rows() {
        let contract = author(|b| {
            push_mlx_affine_stacked(
                b,
                &u32t("w", vec![8, 512, 32]),
                &u32t("s", vec![8, 512, 8]),
                &u32t("bi", vec![8, 512, 8]),
                4,
                0,
                "out".into(),
            )
        });
        assert_eq!(declared(&contract, "out").0, vec![8 * 512, 256]);
    }

    // ─── The refusals ────────────────────────────────────────

    #[test]
    fn a_triplet_that_does_not_line_up_is_refused_by_name() {
        let cases: [(&str, RawTensor, RawTensor, RawTensor, i64, i64, &str); 8] = [
            (
                "a vector has no group axis",
                u32t("w", vec![32]),
                u32t("s", vec![8]),
                u32t("bi", vec![8]),
                4,
                0,
                "incompatible shapes",
            ),
            (
                // The width of a word is the one number the shapes do
                // not carry, so a BF16 tensor would be read as holding
                // twice the columns it holds.
                "the words are not U32",
                raw("w", vec![512, 32], Encoding::Raw(DType::BF16)),
                u32t("s", vec![512, 8]),
                u32t("bi", vec![512, 8]),
                4,
                0,
                "not the U32 words this format packs into",
            ),
            (
                "the scales are a rank short",
                u32t("w", vec![512, 32]),
                u32t("s", vec![512]),
                u32t("bi", vec![512]),
                4,
                0,
                "incompatible shapes",
            ),
            (
                "the biases do not match the scales",
                u32t("w", vec![512, 32]),
                u32t("s", vec![512, 8]),
                u32t("bi", vec![512, 4]),
                4,
                0,
                "incompatible shapes",
            ),
            (
                "the stacked axes disagree",
                u32t("w", vec![8, 512, 32]),
                u32t("s", vec![4, 512, 8]),
                u32t("bi", vec![4, 512, 8]),
                4,
                0,
                "disagrees with its scales on the stacked axes",
            ),
            (
                "there are no groups",
                u32t("w", vec![512, 32]),
                u32t("s", vec![512, 0]),
                u32t("bi", vec![512, 0]),
                4,
                0,
                "has no groups",
            ),
            (
                "the stated group size does not divide the words",
                u32t("w", vec![512, 32]),
                u32t("s", vec![512, 5]),
                u32t("bi", vec![512, 5]),
                0,
                64,
                "cannot derive a width from groups of 64",
            ),
            (
                "the derived group size does not divide the columns",
                u32t("w", vec![512, 32]),
                u32t("s", vec![512, 7]),
                u32t("bi", vec![512, 7]),
                4,
                0,
                "cannot derive a group size",
            ),
        ];
        for (why, w, s, bi, bits, group, expected) in cases {
            let err =
                try_author(|b| push_mlx_affine_stacked(b, &w, &s, &bi, bits, group, "o".into()))
                    .expect_err(why);
            let msg = message(err);
            assert!(msg.contains(expected), "{why}: {msg}");
            assert!(
                msg.contains('w'),
                "{why}: the refusal names the tensor: {msg}"
            );
        }
    }

    /// Only 4 and 8 bits have kernels; a width that derives to anything
    /// else is refused rather than declared and read wrongly.
    #[test]
    fn only_four_and_eight_bit_widths_are_served() {
        for (words, group, bits) in [(32i64, 128i64, 2i64), (32, 16, 16)] {
            let err = try_author(|b| {
                push_mlx_affine_stacked(
                    b,
                    &u32t("w", vec![512, words]),
                    &u32t("s", vec![512, 4]),
                    &u32t("bi", vec![512, 4]),
                    0,
                    group,
                    "o".into(),
                )
            })
            .expect_err("an unservable width");
            assert!(
                message(err).contains(&format!("unsupported width ({bits} bits)")),
                "{words} words at group {group}"
            );
        }
    }

    // ─── Shipped MXFP4 ───────────────────────────────────────

    /// Eight nibbles to a little-endian word: a 32-element block is four
    /// words, and the declaration is a transmute of the checkpoint's own
    /// bytes rather than a decode.
    #[test]
    fn a_shipped_mxfp4_pair_is_declared_without_decoding() {
        let contract = author(|b| {
            push_mlx_mxfp4_stacked(
                b,
                &u32t("w", vec![8, 512, 32]),
                &u8t("s", vec![8, 512, 8]),
                "out".into(),
            )
        });
        let (shape, encoding) = declared(&contract, "out");
        assert_eq!(shape, vec![8 * 512, 256]);
        let spec = spec(&encoding);
        assert_eq!(spec.scheme, QuantScheme::Mxfp4E2M1E8M0);
        assert_eq!(spec.bits_per_element, 4);
        assert_eq!(spec.group_size, 32);
        assert_eq!(spec.logical_dtype, DType::BF16);
    }

    #[test]
    fn an_mxfp4_pair_that_does_not_line_up_is_refused_by_name() {
        let cases: [(&str, RawTensor, RawTensor, &str); 4] = [
            (
                "a vector has no block axis",
                u32t("w", vec![32]),
                u8t("s", vec![8]),
                "differ in rank",
            ),
            (
                "the scales are not the E8M0 exponents",
                u32t("w", vec![512, 32]),
                u32t("s", vec![512, 8]),
                "not the U8 E8M0 block exponents",
            ),
            (
                "the stacked axes disagree",
                u32t("w", vec![8, 512, 32]),
                u8t("s", vec![4, 512, 8]),
                "disagrees with its scales on the stacked axes",
            ),
            (
                "the words do not cover the blocks",
                u32t("w", vec![512, 30]),
                u8t("s", vec![512, 8]),
                "packs 30 words against 8 blocks",
            ),
        ];
        for (why, w, s, expected) in cases {
            let err = try_author(|b| push_mlx_mxfp4_stacked(b, &w, &s, "o".into())).expect_err(why);
            let msg = message(err);
            assert!(msg.contains(expected), "{why}: {msg}");
            assert!(
                msg.contains('w'),
                "{why}: the refusal names the tensor: {msg}"
            );
        }
    }

    #[test]
    fn an_mxfp4_pair_with_no_blocks_is_refused() {
        let err = try_author(|b| {
            push_mlx_mxfp4_stacked(
                b,
                &u32t("w", vec![512, 0]),
                &u8t("s", vec![512, 0]),
                "o".into(),
            )
        })
        .expect_err("no blocks");
        assert!(message(err).contains("packs 0 words against 0 blocks"));
    }

    // ─── Load-time encode ────────────────────────────────────

    /// A projection a checkpoint left in BF16 is quantized on load, not
    /// transmuted: the contract carries a `cast`, and the encode writes the
    /// scales and biases beside its output.
    #[test]
    fn a_bf16_projection_is_encoded_at_load_time() {
        let contract = author(|b| push_encoded_affine(b, Expr::src("w"), 512, 256, "out".into()));
        let (shape, encoding) = declared(&contract, "out");
        assert_eq!(shape, vec![512, 256]);
        let spec = spec(&encoding);
        assert_eq!(spec.group_size, AFFINE_GROUP as u32);
        assert_eq!(spec.bits_per_element, 4);
    }

    #[test]
    fn a_width_these_group_64_kernels_cannot_quantize_is_refused() {
        let err = try_author(|b| push_encoded_affine(b, Expr::src("w"), 512, 100, "out".into()))
            .expect_err("100 is not a multiple of 64");
        let msg = message(err);
        assert!(msg.contains("'out' has 100 columns"), "{msg}");
        assert!(msg.contains("group-64"), "{msg}");
    }

    // ─── MXFP4 values ────────────────────────────────────────

    /// The scales have to be declared before they can be scaled by, so the
    /// pair leaves an internal tensor behind.
    ///
    /// The width is a group count: `8` groups is `256` columns, and there
    /// is no third width to refuse, which is why this is the only test
    /// the function has.
    #[test]
    fn mxfp4_values_leaves_its_scales_declared_and_internal() {
        let mut captured = None;
        let contract = author(|b| {
            captured = Some(mxfp4_values(
                b,
                Expr::src("blocks"),
                Expr::src("scales"),
                512,
                256 / 32,
                "out.scales".into(),
            ));
            Ok(())
        });
        let (shape, encoding) = declared(&contract, "out.scales");
        assert_eq!(shape, vec![512, 256 / 32]);
        assert_eq!(encoding, Encoding::Raw(DType::E8M0));
        assert!(
            contract
                .tensors
                .iter()
                .any(|t| t.name == "out.scales" && t.visibility == Visibility::Internal),
            "the scales are an intermediate, not a runtime tensor"
        );
        assert!(captured.is_some(), "the values expression came back");
    }
}
