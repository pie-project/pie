//! What the Qwen3.5 hybrid families bind.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/qwen3_5/qwen3_5_contract.hpp`. The
//! dense hybrid needs one real thing beyond the generic rules: the Gated
//! DeltaNet tensors stack `[K | K | V]` on axis 0, so a uniform row shard
//! cuts across the block boundaries and hands a rank part of K where it
//! needs V. The MoE hybrid adds the shared-expert join and the per-expert
//! stacks.

use model_loader::checkpoint::RawTensor;
use model_loader::contract::Expr;
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, QuantScheme};

use crate::shared::builder::{Builder, is_raw};
use crate::shared::mlx;
use crate::shared::moe::hf_moe_expert_stacks;

/// qwen3_5, qwen3_5_text: a dense hybrid decoder under the usual names.
pub fn author_qwen3_5(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    // The vision-language checkpoints nest the decoder; the text-only ones
    // do not. Both are this row, so the prefix is asked for rather than
    // declared.
    b.decoder_layer_prefix_any_of(&["model.language_model.layers.", "model.layers."]);
    gdn_kkv_blocked_shards(b)?;
    gdn_fp32_parameters(b)?;
    // The speculative-decoding head is a full-attention layer with the same
    // projection names, so it wants the same join. Checkpoints without one
    // make this a no-op.
    b.also_join_module("mtp.layers.0.");
    mtp_int8_lm_head(b)?;
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// qwen3_moe, qwen3_5_moe, qwen3_5_moe_text.
///
/// Deliberately no `dense_fused_projection_joins`: this bind path reads
/// q/k/v separately, and the MLP lives in the experts, so there is no
/// layer-level `gate_proj`/`up_proj` pair to join.
pub fn author_qwen3_5_moe(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    b.decoder_layer_prefix_any_of(&["model.language_model.layers.", "model.layers."]);
    gdn_kkv_blocked_shards(b)?;
    gdn_fp32_parameters(b)?;
    // The MoE decode runs through flashinfer's CUTLASS grouped GEMM, which
    // reads fc1's output as [linear|gate]; the checkpoint stores [gate|up].
    // Both the pre-fused and the per-expert stacking paths publish in the
    // order the bound driver expects. `moe/flashinfer_moe.hpp` is where
    // that order is written down — "fc1 weights must be stacked as
    // [up; gate], not pie's usual [gate; up]" — and the two have to agree.
    const GATE_SECOND: bool = true;
    b.fused_moe_gate_up_tp_slices(GATE_SECOND)?;
    shared_expert_gate_up_joins(b);
    hf_moe_expert_stacks(b, GATE_SECOND, false)?;
    b.publish_remaining()
}

/// This rank's `[K/T | K/T | V/T]` view of a `[2K + V, ...]` tensor.
///
/// Returned with its declared shape, because the extent depends on whether
/// the world divides each block and only `local_extent` knows.
fn gdn_kkv_blocked(b: &Builder<'_>, raw: &RawTensor, k_dim: i64, v_dim: i64) -> (Expr, Vec<i64>) {
    let src = || Expr::src(&raw.name);
    let (key_lo, key_rows) = b.band(src(), 0, 0, k_dim);
    let (key_hi, _) = b.band(src(), 0, k_dim, k_dim);
    let (value, value_rows) = b.band(src(), 0, 2 * k_dim, v_dim);
    let mut shape = raw.shape.clone();
    shape[0] = 2 * key_rows + value_rows;
    (Expr::concat(0, vec![key_lo, key_hi, value]), shape)
}

/// Shard the Gated DeltaNet tensors whose leading axis stacks `[K | K | V]`.
///
/// `linear_attn.in_proj_qkv.weight` and `conv1d.weight` both stack two key
/// blocks and one value block on axis 0. Take each block's
/// band and join them: every rank gets its own `[K/T | K/T | V/T]`, which is
/// what the GDN kernels address. Without this the loader has no shard axis
/// for these names and leaves them replicated, so every rank loads the whole
/// tensor and the driver slices it afterwards with device-to-device copies.
///
/// `K` and `V` come from the checkpoint, not from a config field: `in_proj_z`
/// is `[V, hidden]`, and `in_proj_qkv` is `[2K + V, hidden]`, so the pair
/// determines both.
fn gdn_kkv_blocked_shards(b: &mut Builder<'_>) -> Result<(), Error> {
    if b.target().tp_size <= 1 {
        return Ok(());
    }
    for layer in 0..b.shape().layers {
        let la = format!("{}{layer}.linear_attn.", b.decoder_layer_prefix_value());
        let (Some(qkv), Some(z)) = (
            b.find(&b.source_name(&format!("{la}in_proj_qkv.weight"))),
            b.find(&b.source_name(&format!("{la}in_proj_z.weight"))),
        ) else {
            continue;
        };
        if qkv.shape.is_empty() || z.shape.is_empty() {
            continue;
        }
        let v_dim = z.shape[0];
        let conv_dim = qkv.shape[0];
        if conv_dim <= v_dim || (conv_dim - v_dim) % 2 != 0 {
            continue;
        }
        let k_dim = (conv_dim - v_dim) / 2;
        for leaf in ["in_proj_qkv.weight", "conv1d.weight"] {
            let Some(raw) = b.find(&b.source_name(&format!("{la}{leaf}"))) else {
                continue;
            };
            if raw.shape.is_empty() || raw.shape[0] != conv_dim {
                continue;
            }
            let (expr, shape) = gdn_kkv_blocked(b, raw, k_dim, v_dim);
            let id = raw.id;
            let encoding = raw.encoding.clone();
            b.define(b.output_name(&raw.name), expr, encoding, Some(shape));
            b.consume(id);
        }
    }
    Ok(())
}

/// Widen the two gated-delta-net parameters the kernels read as fp32.
///
/// `A_log` and the gated RMSNorm's weight enter the GDN kernels as
/// `const float*`, but HF ships them fp32 on Qwen3.5-4B and **bf16** on
/// Qwen3.6-35B-A3B. Only these two: `dt_bias` sits beside them in the same
/// module and is read as bf16, so a suffix match any looser than this list
/// would silently widen it.
///
/// The `already fp32` branch is not an optimization; it is required. A
/// `Cast` to the encoding its operand already has is refused (a node may not
/// denote exactly its operand), which is what makes the two checkpoint
/// conventions impossible to paper over with one unconditional cast.
fn gdn_fp32_parameters(b: &mut Builder<'_>) -> Result<(), Error> {
    for raw in b.tensors().to_vec() {
        if ![".linear_attn.A_log", ".linear_attn.norm.weight"]
            .iter()
            .any(|tail| raw.name.ends_with(tail))
        {
            continue;
        }
        let bf16 = is_raw(&raw.encoding, DType::BF16);
        if !bf16 && !is_raw(&raw.encoding, DType::F32) {
            continue;
        }
        let axis = b.shard_axis(&raw.name)?;
        let (expr, local) = b.shard(Expr::src(&raw.name), raw.shape.clone(), axis);
        let f32enc = Encoding::Raw(DType::F32);
        let expr = if bf16 {
            expr.cast(f32enc.clone())
        } else {
            expr
        };
        b.define(b.output_name(&raw.name), expr, f32enc, Some(local));
        b.consume(raw.id);
    }
    Ok(())
}

/// Publish an int8 view of `lm_head` for the speculative head to read.
///
/// The draft step and the main path read the *same* head, at different
/// precisions. So this is not a re-encode: both views are published, and
/// `quantized_view` leaves the bf16 original alone. A tied checkpoint has no
/// `lm_head.weight`; the head is `embed_tokens`, which is what the bind
/// resolves to and therefore what gets quantized.
fn mtp_int8_lm_head(b: &mut Builder<'_>) -> Result<(), Error> {
    if !b.knobs().qwen35_mtp_int8_lm_head || b.find("mtp.fc.weight").is_none() {
        return Ok(());
    }
    // The decoder prefix varies (the VL checkpoints nest it), so the tied
    // fallback matches the suffix.
    let head = b.find("lm_head.weight").or_else(|| {
        b.tensors()
            .iter()
            .copied()
            .find(|raw| raw.name.ends_with(".embed_tokens.weight"))
    });
    // Only a bf16 source: a checkpoint that already ships a quantized head
    // wants that head, not a second encoding of it.
    let Some(head) = head else {
        return Ok(());
    };
    if !is_raw(&head.encoding, DType::BF16) {
        return Ok(());
    }
    let name = head.name.clone();
    b.quantized_view(&name, "mtp.lm_head".to_string(), QuantScheme::Int8Symmetric)?;
    Ok(())
}

/// Join the shared expert's gate and up projections the MoE forward reads
/// pre-fused, and optionally the scalar gate row after them.
///
/// The sources are **not** consumed. Unlike the Gated DeltaNet join, both
/// unfused projections stay live: the fold-into-routed path reads them
/// separately, and which path runs is a per-step decision. So this slab is
/// additive, exactly like the Kimi and DSv4 expert stacks.
///
/// Only bf16 sources, because the bind had exactly one converter and a
/// checkpoint that ships this pair quantized wants the quantized kernels.
/// The scalar gate is replicated, not sharded, so it is named directly while
/// gate and up take the column-parallel split.
fn shared_expert_gate_up_join(b: &mut Builder<'_>, layer_prefix: &str) {
    let lp = format!("{layer_prefix}mlp.shared_expert");
    let (Some(gate), Some(up)) = (
        b.find(&b.source_name(&format!("{lp}.gate_proj.weight"))),
        b.find(&b.source_name(&format!("{lp}.up_proj.weight"))),
    ) else {
        return;
    };
    if !is_raw(&gate.encoding, DType::BF16) || !is_raw(&up.encoding, DType::BF16) {
        return;
    }
    if gate.shape.len() != 2 || up.shape.len() != 2 || gate.shape[1] != up.shape[1] {
        return;
    }

    let gate_local = b.split(Expr::src(&gate.name), 0);
    let up_local = b.split(Expr::src(&up.name), 0);
    let rows = b.local_extent(gate.shape[0]) + b.local_extent(up.shape[0]);

    // The scalar gate stays its own tensor. Folding its row into this slab
    // was `PIE_QWEN35_FUSED_SHARED_SCALAR_GATE`, and the arm that read the
    // folded `gate_up_gate_proj` is gone from the forward
    // (`qwen35_fused_shared_scalar_gate_enabled()` is `false`), so a contract
    // that published it would name a tensor nothing binds.
    b.define(
        b.output_name(&format!("{lp}.gate_up_proj.weight")),
        Expr::concat(0, vec![gate_local, up_local]),
        gate.encoding.clone(),
        Some(vec![rows, gate.shape[1]]),
    );
}

/// Every module that carries a shared expert: the decoder layers and, when
/// the checkpoint ships one, the speculative-decoding block. The MTP layer
/// is not under `decoder_layer_prefix`, and its bind runs the same fusion.
fn shared_expert_gate_up_joins(b: &mut Builder<'_>) {
    for layer in 0..b.shape().layers {
        let prefix = format!("{}{layer}.", b.decoder_layer_prefix_value());
        shared_expert_gate_up_join(b, &prefix);
    }
    shared_expert_gate_up_join(b, "mtp.layers.0.");
}

/// The Metal lowering: rename for MLX's binder, bind in place. Ported from
/// `crates/driver-metal/csrc/src/model/qwen3_5/qwen3_5_contract.hpp`; also answers for
/// `qwen3_next` and `qwen3_6`, the mlx-side spellings of the same hybrid.
pub fn author_qwen3_5_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    let has_lm_head = b.tensors().iter().any(|raw| {
        raw.name.starts_with("lm_head.") || raw.name.starts_with("language_model.lm_head.")
    });
    let tied = b.shape().tied_embeddings && !has_lm_head;
    mlx::author_mlx_file(b, "Qwen3.5", &move |_, raw_name| {
        qwen3_5_mlx_name(raw_name, tied)
    })
}

fn qwen3_5_mlx_name(raw_name: &str, tied: bool) -> Result<Option<String>, Error> {
    // Not the text decoder. The vision tower has the same two spellings as
    // the decoder does; `mtp.` is the multi-token-prediction head, which this
    // driver does not run.
    for skip in ["visual.", "vision_tower.", "mtp."] {
        if mlx::has_wrapper_member(raw_name, skip) {
            return Ok(None);
        }
    }
    // The output projection: spelled bare by the HF release and under the
    // wrapper by the mlx repack. Untied it keeps its own name; tied it lands
    // on `shared_embedding` beside the table it IS.
    for head in ["lm_head.", "language_model.lm_head."] {
        if let Some(tail) = raw_name.strip_prefix(head) {
            return Ok(Some(if tied {
                format!("shared_embedding.{tail}")
            } else {
                format!("lm_head.{tail}")
            }));
        }
    }
    // Its own output is a valid input: see `mlx::already_lowered`. After the
    // head arm above, because that one is not an identity when tied.
    if mlx::already_lowered(raw_name) {
        return Ok(Some(raw_name.to_string()));
    }
    // The same two-spellings fact `author_qwen3_5` declares, and for the same
    // reason: the vision-language checkpoints nest the decoder under
    // `language_model.` and the text-only ones (`qwen3_5_text`) do not, and
    // both are this row. `mlx::decoder_member` knows only the nested pair
    // because gemma-4, its other caller, has no un-nested release; the bare
    // `model.` wrapper is added here rather than there so gemma-4 keeps
    // refusing a spelling it never ships. Safe after the arms above: the
    // vision tower and the MTP head are skipped under this same wrapper.
    let Some(decoder) = mlx::decoder_member(raw_name).or_else(|| raw_name.strip_prefix("model."))
    else {
        return mlx::fail(format!(
            "Metal Qwen3.5 schema has no declared mapping or skip for '{raw_name}'"
        ));
    };
    if let Some(tail) = decoder.strip_prefix("embed_tokens.") {
        return Ok(Some(if tied {
            format!("shared_embedding.{tail}")
        } else {
            format!("embed_tokens.{tail}")
        }));
    }
    if decoder == "norm.weight" {
        return Ok(Some("final_norm.weight".to_string()));
    }
    let (layer, member) = mlx::layer_member(decoder, "Qwen3.5", raw_name)?;
    if let Some(renamed) = mlx::routed_expert_member(
        raw_name, member, "Qwen3.5", /*has_shared_expert=*/ true,
    )? {
        return Ok(Some(format!("layers.{layer}.{renamed}")));
    }
    Ok(Some(format!("layers.{layer}.{member}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::{FamilyKnobs, Policy};
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;
    use model_loader::types::{BackendKind, FileId, TensorId};

    const HIDDEN: i64 = 64;
    const K: i64 = 16;
    const V: i64 = 32;
    const LA: &str = "model.layers.0.linear_attn.";

    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }
    fn f32e() -> Encoding {
        Encoding::Raw(DType::F32)
    }

    struct Ck(Vec<RawTensor>);

    impl Ck {
        fn new() -> Self {
            Self(Vec::new())
        }
        fn push(mut self, name: &str, shape: &[i64], encoding: Encoding) -> Self {
            let elements: i64 = shape.iter().product();
            self.0.push(RawTensor {
                id: TensorId(u32::try_from(self.0.len()).expect("a small fixture")),
                name: name.to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: u64::try_from(elements * 2).unwrap_or(0),
                shape: shape.to_vec(),
                encoding,
            });
            self
        }
    }

    /// A dense hybrid layer: the three `[2K + V, ...]` GDN tensors plus the
    /// `in_proj_z` that says where the value block starts.
    fn gdn_checkpoint() -> Ck {
        let conv = 2 * K + V;
        Ck::new()
            .push(&format!("{LA}in_proj_qkv.weight"), &[conv, HIDDEN], bf16())
            .push(&format!("{LA}in_proj_z.weight"), &[V, HIDDEN], bf16())
            .push(&format!("{LA}conv1d.weight"), &[conv, 1, 4], bf16())
            .push(&format!("{LA}A_log"), &[V], f32e())
            .push(&format!("{LA}norm.weight"), &[V], bf16())
            .push(&format!("{LA}dt_bias"), &[V], bf16())
            .push("model.norm.weight", &[HIDDEN], bf16())
    }

    fn target(tp_size: u32) -> StorageTarget {
        StorageTarget {
            backend: BackendKind::Cuda,
            tp_rank: 0,
            tp_size,
            max_tile_bytes: 1 << 20,
            preferred_alignment: 256,
            tile_map_mask: model_loader::plan::CUDA_TILE_MAP_MASK,
            ..StorageTarget::default()
        }
    }

    fn run(
        ck: Ck,
        tp_size: u32,
        policy: &Policy,
        author: impl FnOnce(&mut Builder<'_>) -> Result<(), Error>,
    ) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: ck.0,
        };
        let enc = StoredEncoding::dense();
        let t = target(tp_size);
        let mut b = Builder::new(
            &meta,
            "qwen3.5-test",
            LoadShape::dense(1, 128, true),
            &enc,
            &t,
            policy,
        );
        author(&mut b)?;
        b.finish()
    }

    fn dense(ck: Ck, tp_size: u32) -> Result<ModelContract, Error> {
        run(ck, tp_size, &Policy::default(), author_qwen3_5)
    }

    fn shape_of<'a>(contract: &'a ModelContract, name: &str) -> Option<&'a Vec<i64>> {
        contract
            .tensors
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| t.shape.as_ref())
    }

    // ─── the [K | K | V] shard ───────────────────────────────────────

    /// At one rank there is nothing to split, and the tensors are
    /// published as they lie.
    #[test]
    fn a_single_rank_leaves_the_gdn_tensors_alone() {
        let contract = dense(gdn_checkpoint(), 1).expect("the fixture authors");
        assert_eq!(
            shape_of(&contract, &format!("{LA}in_proj_qkv.weight")),
            Some(&vec![2 * K + V, HIDDEN])
        );
    }

    /// Across two ranks each block is banded separately and rejoined, so
    /// a rank holds `[K/2 | K/2 | V/2]` rather than the top half of the
    /// stack — which would hand it all of K and none of V.
    #[test]
    fn two_ranks_each_get_their_own_share_of_every_block() {
        let contract = dense(gdn_checkpoint(), 2).expect("the fixture authors");
        for (name, trailing) in [
            (format!("{LA}in_proj_qkv.weight"), vec![HIDDEN]),
            (format!("{LA}conv1d.weight"), vec![1, 4]),
        ] {
            let mut expected = vec![K + V / 2];
            expected.extend(trailing);
            assert_eq!(
                shape_of(&contract, &name),
                Some(&expected),
                "{name} is this rank's [K/2 | K/2 | V/2]"
            );
        }
    }

    /// The block widths come from the checkpoint, not from a config
    /// field, so a pair that cannot state them is left alone rather than
    /// split on a guess.
    #[test]
    fn a_stack_that_does_not_state_its_blocks_is_left_replicated() {
        let conv = 2 * K + V;
        for (case, ck) in [
            (
                "there is no in_proj_z to measure V by",
                Ck::new()
                    .push(&format!("{LA}in_proj_qkv.weight"), &[conv, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
            ),
            (
                "the value block is the whole stack",
                Ck::new()
                    .push(&format!("{LA}in_proj_qkv.weight"), &[conv, HIDDEN], bf16())
                    .push(&format!("{LA}in_proj_z.weight"), &[conv, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
            ),
            (
                "what is left over does not halve into two key blocks",
                Ck::new()
                    .push(&format!("{LA}in_proj_qkv.weight"), &[conv, HIDDEN], bf16())
                    .push(&format!("{LA}in_proj_z.weight"), &[V + 1, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
            ),
        ] {
            let contract = dense(ck, 2).unwrap_or_else(|e| panic!("{case}: {e}"));
            assert_eq!(
                shape_of(&contract, &format!("{LA}in_proj_qkv.weight")),
                Some(&vec![conv, HIDDEN]),
                "{case}: the stack was published whole"
            );
        }
        // A stack with no extents at all cannot be measured either, and
        // asking `shape[0]` of it must not be how we find that out.
        let ck = Ck::new()
            .push(&format!("{LA}in_proj_qkv.weight"), &[], bf16())
            .push(&format!("{LA}in_proj_z.weight"), &[V, HIDDEN], bf16())
            .push("model.norm.weight", &[HIDDEN], bf16());
        let contract = dense(ck, 2).expect("the fixture authors");
        assert!(
            contract
                .tensors
                .iter()
                .any(|t| t.name == format!("{LA}in_proj_qkv.weight")),
            "published, not banded"
        );
    }

    /// A companion whose leading extent is not the stack's is not one of
    /// these tensors, whatever it is called.
    #[test]
    fn a_companion_of_the_wrong_height_is_not_banded() {
        let conv = 2 * K + V;
        let ck = Ck::new()
            .push(&format!("{LA}in_proj_qkv.weight"), &[conv, HIDDEN], bf16())
            .push(&format!("{LA}in_proj_z.weight"), &[V, HIDDEN], bf16())
            .push(&format!("{LA}conv1d.weight"), &[conv + 8, 1, 4], bf16())
            .push("model.norm.weight", &[HIDDEN], bf16());
        let contract = dense(ck, 2).expect("the fixture authors");
        assert_eq!(
            shape_of(&contract, &format!("{LA}conv1d.weight")),
            Some(&vec![conv + 8, 1, 4]),
            "left to the generic rule"
        );
    }

    // ─── the two fp32 parameters ─────────────────────────────────────

    /// Both conventions land on fp32, and `dt_bias` beside them does
    /// not — the suffix list is exact for that reason.
    #[test]
    fn only_the_two_named_parameters_are_widened() {
        for (case, encoding) in [("HF ships it bf16", bf16()), ("or fp32", f32e())] {
            let given = encoding.clone();
            let ck = Ck::new()
                .push(&format!("{LA}A_log"), &[V], encoding.clone())
                .push(&format!("{LA}norm.weight"), &[V], encoding.clone())
                .push(&format!("{LA}dt_bias"), &[V], encoding)
                .push("model.norm.weight", &[HIDDEN], bf16());
            let contract = dense(ck, 1).expect("the fixture authors");
            let of = |name: &str| {
                contract
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .map(|t| t.encoding.clone())
            };
            assert_eq!(of(&format!("{LA}A_log")), Some(f32e()), "{case}");
            assert_eq!(of(&format!("{LA}norm.weight")), Some(f32e()), "{case}");
            assert_eq!(
                of(&format!("{LA}dt_bias")),
                Some(given),
                "{case}: dt_bias sits in the same module and is left as it lies"
            );
        }
    }

    /// A parameter that is neither bf16 nor fp32 is left where it lies:
    /// the cast this pass writes has exactly two operands it knows.
    #[test]
    fn a_parameter_of_a_third_width_is_not_cast() {
        let ck = Ck::new()
            .push(&format!("{LA}A_log"), &[V], Encoding::Raw(DType::F16))
            .push("model.norm.weight", &[HIDDEN], bf16());
        let contract = dense(ck, 1).expect("the fixture authors");
        assert_eq!(
            contract
                .tensors
                .iter()
                .find(|t| t.name == format!("{LA}A_log"))
                .map(|t| t.encoding.clone()),
            Some(Encoding::Raw(DType::F16))
        );
    }

    // ─── the speculative head's int8 view ────────────────────────────

    fn mtp_policy() -> Policy {
        Policy {
            knobs: FamilyKnobs {
                qwen35_mtp_int8_lm_head: true,
                ..FamilyKnobs::default()
            },
            ..Policy::default()
        }
    }

    fn untied_head() -> Ck {
        Ck::new()
            .push("mtp.fc.weight", &[HIDDEN, HIDDEN], bf16())
            .push("lm_head.weight", &[128, HIDDEN], bf16())
            .push("model.embed_tokens.weight", &[128, HIDDEN], bf16())
            .push("model.norm.weight", &[HIDDEN], bf16())
    }

    /// Both views are published. This is not a re-encode: the draft step
    /// reads int8 and the main path reads the same head in bf16, so the
    /// original has to survive.
    #[test]
    fn the_speculative_head_gets_an_int8_view_beside_the_bf16_one() {
        let contract =
            run(untied_head(), 1, &mtp_policy(), author_qwen3_5).expect("the fixture authors");
        let int8 = contract
            .tensors
            .iter()
            .find(|t| t.name == "mtp.lm_head")
            .expect("the int8 view was published");
        match &int8.encoding {
            Encoding::Quant(spec) => assert_eq!(spec.scheme, QuantScheme::Int8Symmetric),
            other => panic!("expected an int8 encoding, got {other:?}"),
        }
        assert_eq!(
            contract
                .tensors
                .iter()
                .find(|t| t.name == "lm_head.weight")
                .map(|t| t.encoding.clone()),
            Some(bf16()),
            "the main path's head is left alone"
        );
    }

    /// A tied checkpoint has no `lm_head.weight`; the head IS the
    /// embedding table, which is what the bind resolves to and therefore
    /// what gets quantized.
    #[test]
    fn a_tied_checkpoint_quantizes_the_table_the_head_resolves_to() {
        let ck = Ck::new()
            .push("mtp.fc.weight", &[HIDDEN, HIDDEN], bf16())
            .push("model.embed_tokens.weight", &[128, HIDDEN], bf16())
            .push("model.norm.weight", &[HIDDEN], bf16());
        let contract = run(ck, 1, &mtp_policy(), author_qwen3_5).expect("the fixture authors");
        assert!(
            contract.tensors.iter().any(|t| t.name == "mtp.lm_head"),
            "the suffix fallback found the table"
        );
    }

    /// Three ways this pass declines, each for its own reason.
    #[test]
    fn the_int8_view_is_declined_rather_than_forced() {
        for (case, ck, policy) in [
            (
                "the knob is off, which is the default",
                untied_head(),
                Policy::default(),
            ),
            (
                "there is no speculative head to read it",
                Ck::new()
                    .push("lm_head.weight", &[128, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
                mtp_policy(),
            ),
            (
                "the checkpoint already ships a quantized head",
                Ck::new()
                    .push("mtp.fc.weight", &[HIDDEN, HIDDEN], bf16())
                    .push(
                        "lm_head.weight",
                        &[128, HIDDEN],
                        Encoding::Raw(DType::F8E4M3),
                    )
                    .push("model.norm.weight", &[HIDDEN], bf16()),
                mtp_policy(),
            ),
            (
                "there is no head at all",
                Ck::new()
                    .push("mtp.fc.weight", &[HIDDEN, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
                mtp_policy(),
            ),
        ] {
            let contract =
                run(ck, 1, &policy, author_qwen3_5).unwrap_or_else(|e| panic!("{case}: {e}"));
            assert!(
                !contract.tensors.iter().any(|t| t.name == "mtp.lm_head"),
                "{case}: nothing was published"
            );
        }
    }

    // ─── the shared expert's join ────────────────────────────────────

    const SE: &str = "model.layers.0.mlp.shared_expert";

    fn moe_checkpoint() -> Ck {
        Ck::new()
            .push(&format!("{SE}.gate_proj.weight"), &[V, HIDDEN], bf16())
            .push(&format!("{SE}.up_proj.weight"), &[V, HIDDEN], bf16())
            .push(&format!("{SE}.down_proj.weight"), &[HIDDEN, V], bf16())
            .push("model.norm.weight", &[HIDDEN], bf16())
    }

    fn moe(ck: Ck, tp_size: u32) -> Result<ModelContract, Error> {
        run(ck, tp_size, &Policy::default(), author_qwen3_5_moe)
    }

    /// The slab is additive: both unfused projections stay live, because
    /// the fold-into-routed path reads them separately and which path
    /// runs is a per-step decision.
    #[test]
    fn the_shared_expert_join_is_additive() {
        let contract = moe(moe_checkpoint(), 1).expect("the fixture authors");
        assert_eq!(
            shape_of(&contract, &format!("{SE}.gate_up_proj.weight")),
            Some(&vec![2 * V, HIDDEN])
        );
        for kept in ["gate_proj", "up_proj"] {
            assert!(
                contract
                    .tensors
                    .iter()
                    .any(|t| t.name == format!("{SE}.{kept}.weight")),
                "{kept} stays live"
            );
        }
    }

    /// Three ways the join declines, all silently — this pass is
    /// optional and a checkpoint without the pair is not an error.
    #[test]
    fn a_shared_expert_the_join_cannot_serve_is_passed_over() {
        for (case, ck) in [
            (
                "there is no pair",
                Ck::new()
                    .push(&format!("{SE}.gate_proj.weight"), &[V, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
            ),
            (
                "the pair is not bf16",
                Ck::new()
                    .push(
                        &format!("{SE}.gate_proj.weight"),
                        &[V, HIDDEN],
                        Encoding::Raw(DType::F8E4M3),
                    )
                    .push(&format!("{SE}.up_proj.weight"), &[V, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
            ),
            (
                "the two do not agree on their input width",
                Ck::new()
                    .push(&format!("{SE}.gate_proj.weight"), &[V, HIDDEN], bf16())
                    .push(&format!("{SE}.up_proj.weight"), &[V, HIDDEN / 2], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
            ),
            (
                "a projection is not a matrix",
                Ck::new()
                    .push(&format!("{SE}.gate_proj.weight"), &[V], bf16())
                    .push(&format!("{SE}.up_proj.weight"), &[V, HIDDEN], bf16())
                    .push("model.norm.weight", &[HIDDEN], bf16()),
            ),
        ] {
            let contract = moe(ck, 1).unwrap_or_else(|e| panic!("{case}: {e}"));
            assert!(
                shape_of(&contract, &format!("{SE}.gate_up_proj.weight")).is_none(),
                "{case}: nothing was joined"
            );
        }
    }

    // ─── the Metal names ─────────────────────────────────────────────

    #[test]
    fn the_head_lands_where_tying_says() {
        for (tied, expected) in [(false, "lm_head.weight"), (true, "shared_embedding.weight")] {
            for spelling in ["lm_head.weight", "language_model.lm_head.weight"] {
                assert_eq!(
                    qwen3_5_mlx_name(spelling, tied).expect("a declared name"),
                    Some(expected.to_string()),
                    "{spelling} at tied={tied}"
                );
            }
        }
    }

    #[test]
    fn the_embedding_lands_where_tying_says() {
        for (tied, expected) in [
            (false, "embed_tokens.weight"),
            (true, "shared_embedding.weight"),
        ] {
            assert_eq!(
                qwen3_5_mlx_name("model.language_model.embed_tokens.weight", tied)
                    .expect("a declared name"),
                Some(expected.to_string())
            );
        }
    }

    #[test]
    fn the_final_norm_is_renamed_and_a_layer_keeps_its_index() {
        assert_eq!(
            qwen3_5_mlx_name("model.language_model.norm.weight", false).expect("a declared name"),
            Some("final_norm.weight".to_string())
        );
        assert_eq!(
            qwen3_5_mlx_name(
                "model.language_model.layers.3.self_attn.q_proj.weight",
                false
            )
            .expect("a declared name"),
            Some("layers.3.self_attn.q_proj.weight".to_string())
        );
    }

    /// All three wrappers this row ships under land on the same name.
    ///
    /// `author_qwen3_5` declares two of them (`model.language_model.` for
    /// the vision-language releases, bare `model.` for `qwen3_5_text`), and
    /// `mlx_lm` writes the third by swapping the two words. A Metal load of
    /// a text-only checkpoint used to be refused at its very first tensor
    /// because this schema knew only the nested pair.
    #[test]
    fn every_wrapper_this_row_ships_under_lands_on_one_name() {
        for wrapper in ["model.language_model.", "language_model.model.", "model."] {
            assert_eq!(
                qwen3_5_mlx_name(&format!("{wrapper}norm.weight"), false).expect("a declared name"),
                Some("final_norm.weight".to_string()),
                "{wrapper}"
            );
            assert_eq!(
                qwen3_5_mlx_name(&format!("{wrapper}layers.3.mlp.down_proj.weight"), false)
                    .expect("a declared name"),
                Some("layers.3.mlp.down_proj.weight".to_string()),
                "{wrapper}"
            );
        }
    }

    /// The vision tower and the speculative head are not the text
    /// decoder, under either of the two spellings the decoder has.
    #[test]
    fn what_this_driver_does_not_run_is_skipped() {
        for skipped in [
            "visual.blocks.0.attn.qkv.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "vision_tower.blocks.0.attn.qkv.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "model.mtp.fc.weight",
        ] {
            assert_eq!(
                qwen3_5_mlx_name(skipped, false).expect("a declared skip"),
                None,
                "{skipped}"
            );
        }
    }

    /// The mixture members the Metal schema renames, and the one it keeps.
    ///
    /// MLX stacks the routed experts under `mlp.switch_mlp`; the Metal
    /// schema binds them as `mlp.experts`. The shared expert and its gate
    /// keep their names, because this row HAS a shared expert -- the same
    /// names on a row without one are refused, which is why the flag is
    /// passed rather than assumed.
    ///
    /// Untested, the failure is quiet in the worst way: a routed member
    /// left under `switch_mlp` is a name the kernel never binds, so the
    /// shared expert generates alone and the model is a fraction of
    /// itself rather than a load error.
    #[test]
    fn the_routed_experts_are_renamed_and_the_shared_one_is_kept() {
        for (raw, want) in [
            (
                "model.language_model.layers.3.mlp.switch_mlp.gate_proj.weight",
                "layers.3.mlp.experts.gate_proj.weight",
            ),
            (
                "model.language_model.layers.3.mlp.switch_mlp.down_proj.scales",
                "layers.3.mlp.experts.down_proj.scales",
            ),
            (
                "model.language_model.layers.3.mlp.shared_expert.up_proj.weight",
                "layers.3.mlp.shared_expert.up_proj.weight",
            ),
            (
                "model.language_model.layers.3.mlp.shared_expert_gate.weight",
                "layers.3.mlp.shared_expert_gate.weight",
            ),
        ] {
            assert_eq!(
                qwen3_5_mlx_name(raw, false).expect("a declared name"),
                Some(want.to_string()),
                "{raw}"
            );
        }
    }

    /// The two spellings the stacked schema cannot serve are refused
    /// HERE rather than at the bind.
    ///
    /// `routed_expert_member` refuses two shapes and this text carries
    /// the refusal out with a `?`. That `?` had never fired, which
    /// matters because the two shapes it rejects are exactly the two a
    /// real checkpoint arrives in:
    ///
    /// * the PLURAL `mlp.shared_experts.` -- deepseek's spelling. It is
    ///   not qwen-3.5's, and the singular is, so the two differ by one
    ///   letter and the load must not read one as the other.
    /// * PER-EXPERT numbering, `mlp.experts.0.gate_proj`, which is what
    ///   an unstacked MLX conversion produces. The Metal routed matvec
    ///   reads ONE tensor per layer, expert-major on axis 0, so a
    ///   per-expert name has no bank to land in.
    ///
    /// Both would otherwise reach the builder as unknown names and be
    /// reported against a symbol instead of against the conversion.
    #[test]
    fn the_two_expert_spellings_this_schema_cannot_serve_are_refused_by_name() {
        for (raw, needle) in [
            (
                "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
                "no shared expert",
            ),
            (
                "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
                "per-expert",
            ),
        ] {
            let err = qwen3_5_mlx_name(raw, false)
                .expect_err("the stacked schema cannot serve this spelling")
                .to_string();
            assert!(err.contains(needle), "{raw}: {err}");
            assert!(err.contains(raw), "the refusal names the tensor: {err}");
        }
    }

    /// A name with no mapping and no skip is refused rather than passed
    /// through, which is what makes the two lists above exhaustive.
    #[test]
    fn a_name_with_neither_a_mapping_nor_a_skip_is_refused() {
        let err =
            qwen3_5_mlx_name("transformer.h.0.ln_1.weight", false).expect_err("no arm claims this");
        assert!(
            err.to_string().contains("no declared mapping or skip")
                && err.to_string().contains("transformer.h.0.ln_1.weight"),
            "{err}"
        );
    }

    /// The lowering's own output is a valid input, which is what lets a
    /// contract be re-authored over one it already produced.
    #[test]
    fn an_already_lowered_name_is_an_identity() {
        assert_eq!(
            qwen3_5_mlx_name("layers.0.self_attn.q_proj.weight", false).expect("a declared name"),
            Some("layers.0.self_attn.q_proj.weight".to_string())
        );
    }
}
