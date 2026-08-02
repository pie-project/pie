//! What DeepSeek-V4 binds.
//!
//! Ported from `driver/cuda/src/model/deepseek_v4/deepseek_v4_contract.hpp`.
//! The only family with its own tensor-parallel shard-axis rule: its experts
//! are named `.ffn.experts.w1/w2/w3` rather than `.mlp.experts.gate/up/down`,
//! and the intermediate dim is split within each expert so every rank
//! computes a partial expert output that an all-reduce combines.

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::{Expr, GroupContract, Scales, TensorContract, TensorType};
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding, QuantGranularity, ScaleForm, TensorId, Visibility};

use pie_model_common::builder::{Builder, is_raw, mxfp4_encoding};
use pie_model_common::policy::{Mxfp4MoePolicy, Mxfp4MoeRequest};

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// DeepSeek-V4.
///
/// The routed experts are handed to the driver one of two ways, and this
/// contract picks which. Dequantized and stacked is what the batched expert
/// GEMM wants and is the default. Left packed is the fallback: the forward
/// pass dequantizes a slice per step instead, which is correct and slow, and
/// is worth taking only when the caller asks for it to hold memory down.
///
/// The device rule `resolve_mxfp4_moe` applies is not this family's rule:
/// it asks whether there are native MXFP4 GEMM kernels to fall back *from*,
/// and DeepSeek-V4 has none. The choice here is between eager and per-step,
/// so anything short of an explicit `RoutedDecode` takes the eager path.
pub fn author_deepseek_v4(b: &mut Builder<'_>) -> Result<(), Error> {
    // Ask where the layers are rather than declare it. This family's
    // released checkpoints name them `layers.<L>.`, not the HF
    // `model.layers.<L>.` the default assumes, and the two expert passes
    // below select by that prefix — with it wrong they matched nothing and
    // the forward pass's packed fallback quietly covered for them.
    b.decoder_layer_prefix_any_of(&["model.layers.", "layers."]);
    b.shard_axis_fn(dsv4_shard_axis);
    b.decide_mxfp4_moe(if b.mxfp4_moe_request() == Mxfp4MoeRequest::RoutedDecode {
        Mxfp4MoePolicy::RoutedDecode
    } else {
        Mxfp4MoePolicy::EagerBf16
    });
    if b.stream_routed_experts() {
        // Streaming and stacking are alternatives, not layers: a stack is
        // one slab per layer holding every expert, which is exactly the
        // residency the slab is there to avoid.
        streamed_expert_groups(b)?;
    } else if b.mxfp4_moe() == Mxfp4MoePolicy::EagerBf16 {
        bf16_expert_stacks(b)?;
    }
    block_scales_to_fp32(b)?;
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

fn dsv4_shard_axis(name: &str) -> Result<Option<u8>, Error> {
    // Routed experts: shard the intermediate dim within each expert. w1/w3
    // on axis 0 (gate/up out dim), w2 on axis 1 (down in dim). Weights only:
    // a companion scale reaches here already rewritten to the weight it
    // scales — `Builder::shard_axis` strips the suffix before consulting
    // this — so listing scales again is how the two lists drift apart.
    if name.contains(".ffn.experts.") {
        if name.ends_with(".w1.weight") || name.ends_with(".w3.weight") {
            return Ok(Some(0));
        }
        if name.ends_with(".w2.weight") {
            return Ok(Some(1));
        }
    }
    if name.ends_with(".shared_experts.w1.weight") || name.ends_with(".shared_experts.w3.weight") {
        return Ok(Some(0));
    }
    if name.ends_with(".shared_experts.w2.weight") {
        return Ok(Some(1));
    }
    // Inside the FFN, replication is never the answer, so falling through to
    // it is a bug rather than a default: a checkpoint variant that spells an
    // expert differently would otherwise be replicated silently while the
    // forward went on sharding around it, and the model would answer
    // plausibly and wrongly. Say so instead.
    if name.contains(".ffn.")
        && name.ends_with(".weight")
        && !name.contains(".gate.")
        && !name.contains("_norm.")
        && !name.contains("layernorm")
    {
        return fail(format!(
            "deepseek_v4: no sharding decision for FFN tensor '{name}'; add it to \
             dsv4_shard_axis rather than letting it replicate"
        ));
    }
    // Everything outside the FFN is replicated, which avoids TP
    // communication in the main path.
    Ok(None)
}

/// Decode the block scales that ride beside DeepSeek-V4's FP8 weights.
///
/// The checkpoint stores one byte per tile of the weight, and that byte is
/// OCP Microscaling's E8M0. Only a companion to a **block-FP8** weight is an
/// fp32-bound E8M0 exponent — the routed experts' `.scale` tensors ride
/// beside packed MXFP4 `I8` weights and are consumed as raw bytes, so the
/// F8E4M3 guard on the companion is what keeps them apart.
///
/// This pass publishes the exponent bytes and nothing more — a rename and a
/// shard, which shards at any TP degree. The widening to fp32 is the
/// `scaling(..., F32Factors)` request, answered by the executor once the
/// tensor is materialised. Declared U8 rather than E8M0 because the
/// expansion dispatches on the tensor's dtype and knows only UINT8 and BF16.
fn block_scales_to_fp32(b: &mut Builder<'_>) -> Result<(), Error> {
    const SUFFIX: &str = ".scale";
    for raw in b.tensors().to_vec() {
        if !raw.name.ends_with(SUFFIX) || !is_raw(&raw.encoding, DType::U8) {
            continue;
        }
        let weight = format!("{}.weight", &raw.name[..raw.name.len() - SUFFIX.len()]);
        let Some(companion) = b.find(&weight) else {
            continue;
        };
        if !is_raw(&companion.encoding, DType::F8E4M3) {
            continue;
        }
        let shape = raw.shape.clone();
        let axis = b.shard_axis(&raw.name)?;
        let (expr, local) = b.shard(Expr::src(&raw.name), shape.clone(), axis);
        let id = raw.id;
        let defined = b.define(
            b.output_name(&raw.name),
            expr,
            Encoding::Raw(DType::U8),
            Some(local),
        );
        // The pairing this loop just established, stated rather than
        // dropped. Both shapes are in hand, so the block size is read off
        // them instead of assumed.
        let weight_shape = companion.shape.clone();
        if let Some(defined) = defined
            && let (Some(&scale_cols), Some(&weight_cols)) =
                (shape.last().filter(|&&c| c > 0), weight_shape.last())
        {
            let block = weight_cols / scale_cols;
            if block > 0 {
                b.set_scales(
                    defined,
                    Scales {
                        of: b.output_name(&weight),
                        granularity: QuantGranularity::PerGroup,
                        group_size: block as u32,
                        channel_axis: 0,
                        form: ScaleForm::F32Factors,
                    },
                );
            }
        }
        b.consume(id);
    }
    Ok(())
}

/// Dequantize DeepSeek-V4's routed experts and stack them, at load time.
///
/// The forward pass wants two dense bf16 slabs per layer — `[E, 2I, H]` gate
/// over up, and `[E, H, I]` down — because a batched GEMM over experts wants
/// one base pointer and a stride, not 3E separate tensors. The checkpoint
/// stores the other thing: per expert, `w1`/`w2`/`w3` as packed MXFP4 (an
/// `I8` tensor holding two E2M1 nibbles per byte) beside E8M0 block scales.
///
/// The order matters: the concatenation is *inside* the scale, not outside.
/// One scale node over the whole slab is one instruction per slab whose
/// packed input is a temporary the memory planner can reuse. Sharding sits
/// inside all of it, so each rank dequantizes only the slice it keeps.
fn bf16_expert_stacks(b: &mut Builder<'_>) -> Result<(), Error> {
    const GROUP: i64 = 32;

    // The layer and expert counts are read off the names that are actually
    // present rather than off the config: a `Component` build holds a slice
    // of the checkpoint, and a config-driven loop would ask for tensors that
    // this process was never given.
    let mut layer = 0u32;
    loop {
        let ffn = format!("{}{layer}.ffn.", b.decoder_layer_prefix_value());
        if b.find(&b.source_name(&format!("{ffn}experts.0.w1.weight")))
            .is_none()
        {
            break;
        }
        let mut gate_up = Vec::new();
        let mut gate_up_scales = Vec::new();
        let mut down = Vec::new();
        let mut down_scales = Vec::new();
        let mut consumed: Vec<TensorId> = Vec::new();
        let mut local_inter = 0i64;
        let mut hidden = 0i64;

        let mut expert = 0u32;
        loop {
            let ep = format!("{ffn}experts.{expert}.");
            if b.find(&b.source_name(&format!("{ep}w1.weight"))).is_none() {
                break;
            }
            let names = [
                format!("{ep}w1.weight"),
                format!("{ep}w1.scale"),
                format!("{ep}w3.weight"),
                format!("{ep}w3.scale"),
                format!("{ep}w2.weight"),
                format!("{ep}w2.scale"),
            ];
            let mut parts = Vec::with_capacity(6);
            for name in &names {
                let Some(part) = b.find(&b.source_name(name)) else {
                    return fail(format!(
                        "deepseek_v4 expert stack: {ep} is missing a weight or scale"
                    ));
                };
                parts.push(part);
            }
            // Packed nibbles are stored as `I8`, two elements per byte. A
            // checkpoint that stores its experts some other way is not this
            // pass's to rewrite — leave the whole checkpoint alone rather
            // than half of it, and let `author_dense_contract` publish the
            // experts as they are.
            if !is_raw(&parts[0].encoding, DType::I8) || !is_raw(&parts[4].encoding, DType::I8) {
                return Ok(());
            }

            // `w1`/`w3` are `[I_full, H/2]` packed; `w2` is `[H, I_full/2]`.
            // The logical shapes the transmutes declare unpack the last
            // axis.
            let up_raw = &parts[0].shape;
            let down_raw = &parts[4].shape;
            if up_raw.len() != 2 || down_raw.len() != 2 {
                return fail(format!(
                    "deepseek_v4 expert stack: {ep} expects rank-2 expert weights"
                ));
            }
            let inter_full = up_raw[0];
            let h = up_raw[1] * 2;
            let inter = b.local_extent(inter_full);
            if h % GROUP != 0 || inter % GROUP != 0 {
                return fail(format!(
                    "deepseek_v4 expert stack: {ep} expects both expert dims to be a \
                     multiple of 32"
                ));
            }
            if local_inter != 0 && (inter != local_inter || h != hidden) {
                return fail(format!(
                    "deepseek_v4 expert stack: {ep} disagrees with its siblings on shape"
                ));
            }
            local_inter = inter;
            hidden = h;

            // Every leg is declared rank 3 with a leading 1, so that the
            // outer concatenation over axis 0 is a stack. The transmute
            // carries the rank lift: it already says "read these bytes as
            // this shape and this type". `w1`/`w3` shard the out dim and
            // `w2` the in dim — the same split `dsv4_shard_axis` states,
            // applied to the logical shapes rather than the packed ones.
            let packed = |b: &Builder<'_>, raw: &RawTensor, shape: Vec<i64>, axis: u8| {
                b.shard(
                    Expr::src(&raw.name)
                        .transmute(TensorType::new(shape.clone(), mxfp4_encoding(2))),
                    shape,
                    Some(axis),
                )
                .0
            };
            let factors = |b: &Builder<'_>, raw: &RawTensor, shape: Vec<i64>, axis: u8| {
                b.shard(
                    Expr::src(&raw.name)
                        .transmute(TensorType::new(shape.clone(), Encoding::Raw(DType::E8M0))),
                    shape,
                    Some(axis),
                )
                .0
            };

            gate_up.push(Expr::concat(
                1,
                vec![
                    packed(b, parts[0], vec![1, inter_full, h], 1),
                    packed(b, parts[2], vec![1, inter_full, h], 1),
                ],
            ));
            gate_up_scales.push(Expr::concat(
                1,
                vec![
                    factors(b, parts[1], vec![1, inter_full, h / GROUP], 1),
                    factors(b, parts[3], vec![1, inter_full, h / GROUP], 1),
                ],
            ));
            down.push(packed(b, parts[4], vec![1, down_raw[0], inter_full], 2));
            down_scales.push(factors(
                b,
                parts[5],
                vec![1, down_raw[0], inter_full / GROUP],
                2,
            ));
            consumed.extend(parts.iter().map(|part| part.id));
            expert += 1;
        }
        if gate_up.is_empty() {
            layer += 1;
            continue;
        }
        let experts = gate_up.len() as i64;

        // Named but not bound: `scale_per_block` takes its factors by output
        // name, and the stacked slab is dequantized here, so no kernel ever
        // reads these again.
        let e8m0 = Encoding::Raw(DType::E8M0);
        let gu_scale = format!("{ffn}experts.gate_up.scale");
        let dn_scale = format!("{ffn}experts.down.scale");
        if let Some(gu) = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            e8m0.clone(),
            Some(vec![experts, 2 * local_inter, hidden / GROUP]),
        ) {
            b.mark_internal(gu);
        }
        if let Some(dn) = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            e8m0,
            Some(vec![experts, hidden, local_inter / GROUP]),
        ) {
            b.mark_internal(dn);
        }
        b.define(
            format!("{ffn}experts.gate_up.weight"),
            Expr::concat(0, gate_up).scale_per_block(Expr::out(&gu_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, hidden]),
        );
        b.define(
            format!("{ffn}experts.down.weight"),
            Expr::concat(0, down).scale_per_block(Expr::out(&dn_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, hidden, local_inter]),
        );
        for id in consumed {
            b.consume(id);
        }
        layer += 1;
    }
    Ok(())
}

/// The same experts, as a group per layer, for a driver that will page them.
///
/// [`bf16_expert_stacks`] says "every expert of this layer, concatenated".
/// This says "one expert of this layer", once, with the expert index left
/// standing — the same sentence with the outer `concat` removed and each
/// `src` replaced by the `src_indexed` it was a member of. Everything that
/// made the stack correct is still here in the same order; a group's plan is
/// a whole plan, so all of it runs on the page-in path.
///
/// The declared names carry no layer and no expert — there is one plan, and
/// it is the same plan for every instance of every layer. The driver asks a
/// group by name and gets back `gate_up.weight` and `down.weight`, which is
/// what the per-expert GEMM path wants anyway.
fn streamed_expert_groups(b: &mut Builder<'_>) -> Result<(), Error> {
    const GROUP: i64 = 32;

    let mut layer = 0u32;
    loop {
        let ffn = format!("{}{layer}.ffn.", b.decoder_layer_prefix_value());
        if b.find(&b.source_name(&format!("{ffn}experts.0.w1.weight")))
            .is_none()
        {
            break;
        }

        // Instance 0 is the shape oracle. It has to be: the group's plan is
        // compiled at index 0 and every other instance is then required to
        // match it, so a shape read anywhere else would be a shape the
        // loader is about to reject anyway.
        let names = [
            format!("{ffn}experts.0.w1.weight"),
            format!("{ffn}experts.0.w1.scale"),
            format!("{ffn}experts.0.w3.weight"),
            format!("{ffn}experts.0.w3.scale"),
            format!("{ffn}experts.0.w2.weight"),
            format!("{ffn}experts.0.w2.scale"),
        ];
        let mut proto = Vec::with_capacity(6);
        for name in &names {
            let Some(part) = b.find(&b.source_name(name)) else {
                return fail(format!(
                    "deepseek_v4 expert group: {ffn}experts.0 is missing a weight or scale"
                ));
            };
            proto.push(part);
        }
        if !is_raw(&proto[0].encoding, DType::I8) || !is_raw(&proto[4].encoding, DType::I8) {
            // Not packed MXFP4. Leave the whole checkpoint alone rather than
            // half of it, exactly as the stacking pass does.
            return Ok(());
        }

        let up_raw = proto[0].shape.clone();
        let down_raw = proto[4].shape.clone();
        if up_raw.len() != 2 || down_raw.len() != 2 {
            return fail(format!(
                "deepseek_v4 expert group: {ffn}experts.0 expects rank-2 expert weights"
            ));
        }
        let inter_full = up_raw[0];
        let hidden = up_raw[1] * 2;
        let inter = b.local_extent(inter_full);
        if hidden % GROUP != 0 || inter % GROUP != 0 {
            return fail(format!(
                "deepseek_v4 expert group: {ffn}experts.0 expects both expert dims to be \
                 a multiple of 32"
            ));
        }

        // Count the experts, and claim every tensor all of them read. A
        // group reads through a template, so the sources it consumes are not
        // named by any node and `author_dense_contract` would otherwise
        // publish every one of them.
        let mut experts = 0u32;
        let mut consumed: Vec<TensorId> = Vec::new();
        loop {
            let ep = format!("{ffn}experts.{experts}.");
            if b.find(&b.source_name(&format!("{ep}w1.weight"))).is_none() {
                break;
            }
            for suffix in [
                "w1.weight",
                "w1.scale",
                "w3.weight",
                "w3.scale",
                "w2.weight",
                "w2.scale",
            ] {
                let Some(part) = b.find(&b.source_name(&format!("{ep}{suffix}"))) else {
                    return fail(format!(
                        "deepseek_v4 expert group: {ep} is missing a weight or scale"
                    ));
                };
                consumed.push(part.id);
            }
            experts += 1;
        }
        if experts == 0 {
            layer += 1;
            continue;
        }

        let packed = |b: &Builder<'_>, tmpl: &str, shape: Vec<i64>, axis: u8| {
            b.shard(
                Expr::src_indexed(b.source_name(&format!("{ffn}{tmpl}")))
                    .transmute(TensorType::new(shape.clone(), mxfp4_encoding(2))),
                shape,
                Some(axis),
            )
            .0
        };
        let factors = |b: &Builder<'_>, tmpl: &str, shape: Vec<i64>, axis: u8| {
            b.shard(
                Expr::src_indexed(b.source_name(&format!("{ffn}{tmpl}")))
                    .transmute(TensorType::new(shape.clone(), Encoding::Raw(DType::E8M0))),
                shape,
                Some(axis),
            )
            .0
        };

        let internal = |mut tensor: TensorContract| {
            tensor.visibility = Visibility::Internal;
            tensor
        };
        // Same structure as the stack, one rank lower: with no expert axis
        // to stack along, `w1`/`w3` concatenate along the intermediate dim
        // they were already sharded on, and the scale groups run along the
        // last axis as before. Named the way every bound tensor is named —
        // prefix stripped — because the driver looks a group up beside the
        // weights it binds.
        let e8m0 = Encoding::Raw(DType::E8M0);
        let group = GroupContract {
            name: b.output_name(&format!("{ffn}experts")),
            arity: experts,
            tensors: vec![
                internal(TensorContract::new(
                    "gate_up.scale",
                    Expr::concat(
                        0,
                        vec![
                            factors(
                                b,
                                "experts.{}.w1.scale",
                                vec![inter_full, hidden / GROUP],
                                0,
                            ),
                            factors(
                                b,
                                "experts.{}.w3.scale",
                                vec![inter_full, hidden / GROUP],
                                0,
                            ),
                        ],
                    ),
                    vec![2 * inter, hidden / GROUP],
                    e8m0.clone(),
                )),
                internal(TensorContract::new(
                    "down.scale",
                    factors(
                        b,
                        "experts.{}.w2.scale",
                        vec![down_raw[0], inter_full / GROUP],
                        1,
                    ),
                    vec![down_raw[0], inter / GROUP],
                    e8m0,
                )),
                TensorContract::new(
                    "gate_up.weight",
                    Expr::concat(
                        0,
                        vec![
                            packed(b, "experts.{}.w1.weight", vec![inter_full, hidden], 0),
                            packed(b, "experts.{}.w3.weight", vec![inter_full, hidden], 0),
                        ],
                    )
                    .scale_per_block(Expr::out("gate_up.scale")),
                    vec![2 * inter, hidden],
                    Encoding::Raw(DType::BF16),
                ),
                TensorContract::new(
                    "down.weight",
                    packed(b, "experts.{}.w2.weight", vec![down_raw[0], inter_full], 1)
                        .scale_per_block(Expr::out("down.scale")),
                    vec![down_raw[0], inter],
                    Encoding::Raw(DType::BF16),
                ),
            ],
        };
        b.push_group(group);

        for id in consumed {
            b.consume(id);
        }
        layer += 1;
    }
    Ok(())
}
