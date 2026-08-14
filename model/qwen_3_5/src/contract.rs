//! What the Qwen3.5 hybrid families bind.
//!
//! Ported from `driver/cuda/src/model/qwen3_5/qwen3_5_contract.hpp`. The
//! dense hybrid needs one real thing beyond the generic rules: the Gated
//! DeltaNet tensors stack `[K | K | V]` on axis 0, so a uniform row shard
//! cuts across the block boundaries and hands a rank part of K where it
//! needs V. The MoE hybrid adds the shared-expert join and the per-expert
//! stacks.

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::{Expr, TensorType};
use pie_loader::error::Error;
use pie_loader::types::{Axis, DType, Encoding, QuantScheme, QuantSpec};

use pie_model_common::builder::{Builder, is_raw};
use pie_model_common::mlx;
use pie_model_common::moe::hf_moe_expert_stacks;

/// qwen3_5, qwen3_5_text: a dense hybrid decoder under the usual names.
pub fn author_qwen3_5(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    // The vision-language checkpoints nest the decoder; the text-only ones
    // do not. Both are this row, so the prefix is asked for rather than
    // declared.
    b.decoder_layer_prefix_any_of(&["model.language_model.layers.", "model.layers."]);
    gdn_fp8_dequant(b)?;
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
    gdn_fp8_dequant(b)?;
    gdn_kkv_blocked_shards(b)?;
    gdn_fp32_parameters(b)?;
    // The MoE decode runs through flashinfer's CUTLASS grouped GEMM, which
    // reads fc1's output as [linear|gate]; the checkpoint stores [gate|up].
    // Both the pre-fused and the per-expert stacking paths publish in the
    // order the bound driver expects — `qwen35_moe_gate_up_swapped()` on the
    // forward side is this same constant, and the two have to agree.
    const GATE_SECOND: bool = true;
    b.fused_moe_gate_up_tp_slices(GATE_SECOND)?;
    shared_expert_gate_up_joins(b);
    hf_moe_expert_stacks(b, GATE_SECOND, false)?;
    b.publish_remaining()
}

/// Dequantize the FP8 Gated DeltaNet projections to bf16 at load time.
///
/// The CUDA forward feeds every `linear_attn` projection to `gemm_act_x_w`
/// as a bare bf16 tensor — the layer struct has `QuantMeta` ports for the
/// attention and MLP projections only — so a shipped F8E4M3 GDN weight
/// cannot serve quantized. It can serve *dequantized*: the checkpoint pairs
/// each projection with a 2-D `weight_scale_inv` block-scale tensor, and
/// `Expr::Scale`/`ScaleFactor::PerBlock` is exactly `weight = fp8 * scale`
/// executed by the load-time dequant kernel both executors implement. The
/// factors are declared as an `Internal` tensor at their checkpoint dtype
/// (BF16 or F32; the CUDA engine widens BF16 factors itself), so the
/// driver's bind table sees only the bf16 weight under its usual name.
///
/// Sharding folds in *before* the multiply, on scale-block boundaries only.
/// `in_proj_qkv` stacks `[K | K | V]` on axis 0 and each band splits per
/// rank, so its rows keep their scale rows only when `K` and `V` are whole
/// multiples of `row_block * tp`; the other projections split on their usual
/// axis, which keeps its blocks only when tp divides the scale extent. A
/// split that lands inside a 128-row block has no representable scale slice,
/// so it is refused rather than approximated — as is a projection whose
/// scale companion is missing altogether.
fn gdn_fp8_dequant(b: &mut Builder<'_>) -> Result<(), Error> {
    let tp = i64::from(b.target().tp_size.max(1));
    for raw in b.tensors().to_vec() {
        if !raw.name.contains(".linear_attn.") || !is_raw(&raw.encoding, DType::F8E4M3) {
            continue;
        }
        let refuse = |b: &Builder<'_>, why: String| -> Result<(), Error> {
            Err(Error::Contract(format!(
                "model_type='{}' tp_size={}: '{}' ships F8E4M3, and the Gated \
                 DeltaNet path runs bf16 (no quant-scale port), but the weight \
                 cannot be dequantized at load: {why}",
                b.facts().model_type,
                b.target().tp_size,
                raw.name,
            )))
        };
        if !raw.name.ends_with(".weight") || raw.shape.len() != 2 {
            return refuse(
                b,
                "only 2-D `.weight` projections have a block-scale pairing".to_string(),
            );
        }
        let scale_name = format!("{}_scale_inv", raw.name);
        let Some(scale) = b.find(&scale_name) else {
            return refuse(b, "its `weight_scale_inv` companion is missing".to_string());
        };
        // BF16 or F32 only: the factors ship at their checkpoint dtype (a
        // cast in the plan would hand the CUDA engine an operand it cannot
        // type — a staged buffer is untyped bytes to it), and its dequant
        // reads exactly these two.
        if scale.shape.len() != 2
            || !(is_raw(&scale.encoding, DType::BF16) || is_raw(&scale.encoding, DType::F32))
        {
            return refuse(
                b,
                format!(
                    "'{scale_name}' must be a 2-D F32/BF16 scale tensor, \
                     got {:?} {:?}",
                    scale.encoding, scale.shape
                ),
            );
        }
        let mut block = [0i64; 2];
        for axis in 0..2 {
            let (w, s) = (raw.shape[axis], scale.shape[axis]);
            if s <= 0 || w % s != 0 {
                return refuse(
                    b,
                    format!(
                        "scale shape {:?} is not a whole blocking of weight \
                         shape {:?}",
                        scale.shape, raw.shape
                    ),
                );
            }
            block[axis] = w / s;
        }
        // The payload's bytes reread as a block-scaled scheme: what tells the
        // planner this `Scale` also *decodes* (`TransformSpec::from`), and the
        // executors to unpack E4M3 before the multiply.
        let fp8 = Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::BF16,
            bits_per_element: 8,
            group_size: u32::try_from(block[1]).unwrap_or(0),
            channel_axis: Some(Axis(1)),
        });
        let is_qkv = raw.name.ends_with(".in_proj_qkv.weight");
        let (wexpr, wshape, sexpr, sshape) = if is_qkv && tp > 1 {
            // The [K|K|V] bands, from the same sibling `gdn_kkv_blocked_shards`
            // reads them off.
            let base = &raw.name[..raw.name.len() - "in_proj_qkv.weight".len()];
            let z_dim = b
                .find(&format!("{base}in_proj_z.weight"))
                .and_then(|z| z.shape.first().copied());
            let conv_dim = raw.shape[0];
            let Some(v_dim) =
                z_dim.filter(|&v| v > 0 && conv_dim > v && (conv_dim - v) % 2 == 0)
            else {
                return refuse(
                    b,
                    "the [K|K|V] split needs `in_proj_z.weight` to state V, \
                     and it does not"
                        .to_string(),
                );
            };
            let k_dim = (conv_dim - v_dim) / 2;
            let rb = block[0];
            if k_dim % (rb * tp) != 0 || v_dim % (rb * tp) != 0 {
                return refuse(
                    b,
                    format!(
                        "the [K|K|V] bands (K={k_dim}, V={v_dim}) split {tp} \
                         ways do not fall on {rb}-row scale-block boundaries; \
                         use a tp_size where {rb}*tp divides both, or a BF16 \
                         checkpoint"
                    ),
                );
            }
            let w = || Expr::src(&raw.name).transmute(TensorType::new(raw.shape.clone(), fp8.clone()));
            let (key_lo, key_rows) = b.band(w(), 0, 0, k_dim);
            let (key_hi, _) = b.band(w(), 0, k_dim, k_dim);
            let (value, value_rows) = b.band(w(), 0, 2 * k_dim, v_dim);
            let s = || Expr::src(&scale.name);
            let (skey_lo, skey_rows) = b.band(s(), 0, 0, k_dim / rb);
            let (skey_hi, _) = b.band(s(), 0, k_dim / rb, k_dim / rb);
            let (svalue, svalue_rows) = b.band(s(), 0, 2 * (k_dim / rb), v_dim / rb);
            (
                Expr::concat(0, vec![key_lo, key_hi, value]),
                vec![2 * key_rows + value_rows, raw.shape[1]],
                Expr::concat(0, vec![skey_lo, skey_hi, svalue]),
                vec![2 * skey_rows + svalue_rows, scale.shape[1]],
            )
        } else {
            let axis = b.shard_axis(&raw.name)?;
            if let Some(axis) = axis {
                let index = usize::from(axis);
                if scale.shape[index] % tp != 0 {
                    return refuse(
                        b,
                        format!(
                            "axis {axis} holds {} scale blocks of {}, which \
                             tp_size {tp} does not divide; the split would land \
                             inside a block",
                            scale.shape[index], block[index]
                        ),
                    );
                }
            }
            let wexpr = Expr::src(&raw.name)
                .transmute(TensorType::new(raw.shape.clone(), fp8.clone()));
            let (wexpr, wshape) = b.shard(wexpr, raw.shape.clone(), axis);
            let (sexpr, sshape) = b.shard(Expr::src(&scale.name), scale.shape.clone(), axis);
            (wexpr, wshape, sexpr, sshape)
        };
        // The factors keep their checkpoint dtype, deliberately. A `.cast`
        // here compiled to a TileMap whose operand — the gathered bands — is
        // an anonymous staging buffer, which the CUDA executor types as u8
        // ("unsupported TileMap Cast u8 -> fp32" on the real checkpoint at
        // tp 2), and whose strided single-source form its cast path refuses
        // as non-compact. An `Internal` declaration is always a *typed*
        // buffer on both executors, so the Scale kernel is handed BF16
        // factors and widens them itself.
        let scale_out = b.output_name(&scale.name);
        if let Some(index) = b.define(scale_out.clone(), sexpr, scale.encoding.clone(), Some(sshape))
        {
            b.mark_internal(index);
        }
        b.define(
            b.output_name(&raw.name),
            wexpr.scale_per_block(Expr::out(&scale_out)),
            Encoding::Raw(DType::BF16),
            Some(wshape),
        );
        b.consume(raw.id);
        b.consume(scale.id);
    }
    Ok(())
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
/// `linear_attn.in_proj_qkv.weight`, `conv1d.weight` and `conv1d.bias` all
/// stack two key blocks and one value block on axis 0. Take each block's
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
    for layer in 0..b.facts().num_hidden_layers {
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
        for leaf in ["in_proj_qkv.weight", "conv1d.weight", "conv1d.bias"] {
            let Some(raw) = b.find(&b.source_name(&format!("{la}{leaf}"))) else {
                continue;
            };
            if raw.shape.is_empty() || raw.shape[0] != conv_dim {
                continue;
            }
            // An FP8 `in_proj_qkv` was already banded — with its scales —
            // by `gdn_fp8_dequant`, and its raw tensor is consumed.
            if is_raw(&raw.encoding, DType::F8E4M3) {
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
    for layer in 0..b.facts().num_hidden_layers {
        let prefix = format!("{}{layer}.", b.decoder_layer_prefix_value());
        shared_expert_gate_up_join(b, &prefix);
    }
    shared_expert_gate_up_join(b, "mtp.layers.0.");
}

/// The Metal lowering: rename for MLX's binder, bind in place. Ported from
/// `driver/metal/src/model/qwen3_5/qwen3_5_contract.hpp`; also answers for
/// `qwen3_next` and `qwen3_6`, the mlx-side spellings of the same hybrid.
pub fn author_qwen3_5_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    let has_lm_head = b.tensors().iter().any(|raw| {
        raw.name.starts_with("lm_head.") || raw.name.starts_with("language_model.lm_head.")
    });
    let tied = b.facts().tied_embeddings && !has_lm_head;
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
    let Some(decoder) = mlx::decoder_member(raw_name) else {
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
