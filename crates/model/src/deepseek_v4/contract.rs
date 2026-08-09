//! What DeepSeek-V4 binds.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/deepseek_v4/deepseek_v4_contract.hpp`.
//! The only family with its own tensor-parallel shard-axis rule: its experts
//! are named `.ffn.experts.w1/w2/w3` rather than `.mlp.experts.gate/up/down`,
//! and the intermediate dim is split within each expert so every rank
//! computes a partial expert output that an all-reduce combines.

use model_loader::checkpoint::RawTensor;
use model_loader::contract::{Expr, GroupContract, Scales, TensorContract, TensorType};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, QuantGranularity, ScaleForm, TensorId, Visibility};

use crate::shared::builder::{Builder, is_raw, mxfp4_encoding};
use crate::shared::policy::{Mxfp4MoePolicy, Mxfp4MoeRequest};

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

        // The loops above probe a NAME to decide whether expert `e`
        // exists, so a checkpoint missing exactly that name does not look
        // like a hole -- it looks like the end of the bank, and the walk
        // stops there. Every other missing part is refused by name; that
        // one would silently build a SHORTER bank.
        //
        // Nothing downstream catches it. The manifest measures the
        // router, `[num_experts, hidden]`, and a checkpoint missing a
        // whole expert still carries a full-width router, so the row
        // matches and the load succeeds. Experts are not sharded here --
        // only `inter` is -- so the row's count is the count this bank
        // must have.
        if experts != i64::from(b.shape().n_experts) {
            return fail(format!(
                "deepseek_v4 expert stack: layer {layer} stacked {experts} experts \
                 but the row states {}; the router emits indices this slab has no \
                 rows for",
                b.shape().n_experts
            ));
        }

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
        // UNREACHABLE TODAY, and kept for when it is not. The outer
        // probe above and this loop's own probe are the same name --
        // `{ffn}experts.0.w1.weight` -- so reaching here at all means
        // the first iteration succeeded and `experts >= 1`. A control
        // that deleted this branch changed nothing, which is how that
        // was found.
        //
        // It stays because the check below is an EQUALITY against the
        // row's count: if the two probes ever diverge, a dense layer
        // would arrive here with zero experts and be refused for
        // disagreeing with a row it never contradicted.
        if experts == 0 {
            layer += 1;
            continue;
        }

        // The loops above probe a NAME to decide whether expert `e`
        // exists, so a checkpoint missing exactly that name does not look
        // like a hole -- it looks like the end of the bank, and the walk
        // stops there. Every other missing part is refused by name; that
        // one would silently build a SHORTER bank.
        //
        // Nothing downstream catches it. The manifest measures the
        // router, `[num_experts, hidden]`, and a checkpoint missing a
        // whole expert still carries a full-width router, so the row
        // matches and the load succeeds. Experts are not sharded here --
        // only `inter` is -- so the row's count is the count this bank
        // must have.
        if experts != b.shape().n_experts {
            return fail(format!(
                "deepseek_v4 expert group: layer {layer} grouped {experts} experts \
                 but the row states {}; the router emits indices this group has no \
                 instances for",
                b.shape().n_experts
            ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::Policy;
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;
    use model_loader::types::{BackendKind, FileId};

    const H: i64 = 128; // hidden
    const I: i64 = 64; // expert intermediate

    fn i8e() -> Encoding {
        Encoding::Raw(DType::I8)
    }
    fn u8e() -> Encoding {
        Encoding::Raw(DType::U8)
    }
    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }
    fn fp8() -> Encoding {
        Encoding::Raw(DType::F8E4M3)
    }

    /// The tensors, and HOW MANY ROUTED EXPERTS the fixture means to
    /// ship.
    ///
    /// The count is carried rather than counted back out of the names:
    /// both expert passes now check what they walked against the row's
    /// `n_experts`, so a `run` that recovered the number from the
    /// checkpoint would compare the checkpoint against itself and the
    /// check could never fail.
    struct Ck(Vec<RawTensor>, u32);

    impl Ck {
        fn new() -> Self {
            Self(Vec::new(), 0)
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

    /// One routed expert, packed MXFP4 beside its E8M0 block scales.
    ///
    /// `w1`/`w3` are `[I, H/2]` packed (two nibbles a byte) with `[I, H/32]`
    /// scales; `w2` is `[H, I/2]` with `[H, I/32]`.
    fn expert(ck: Ck, prefix: &str, e: u32) -> Ck {
        let ep = format!("{prefix}experts.{e}.");
        ck.push(&format!("{ep}w1.weight"), &[I, H / 2], i8e())
            .push(&format!("{ep}w1.scale"), &[I, H / 32], u8e())
            .push(&format!("{ep}w3.weight"), &[I, H / 2], i8e())
            .push(&format!("{ep}w3.scale"), &[I, H / 32], u8e())
            .push(&format!("{ep}w2.weight"), &[H, I / 2], i8e())
            .push(&format!("{ep}w2.scale"), &[H, I / 32], u8e())
    }

    /// A one-layer MoE checkpoint with `n` routed experts, under the bare
    /// `layers.` prefix this family's released checkpoints actually use.
    fn moe(n: u32) -> Ck {
        let mut ck = Ck::new().push("model.norm.weight", &[H], bf16());
        for e in 0..n {
            ck = expert(ck, "layers.0.ffn.", e);
        }
        ck.1 = n;
        ck
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

    fn run(ck: Ck, tp_size: u32, policy: &Policy) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: ck.0,
        };
        let enc = StoredEncoding::dense();
        let t = target(tp_size);
        let mut b = Builder::new(
            &meta,
            "deepseek-v4-test",
            LoadShape::mixture(1, 128, ck.1, true),
            &enc,
            &t,
            policy,
        );
        author_deepseek_v4(&mut b)?;
        b.finish()
    }

    fn plain(ck: Ck) -> Result<ModelContract, Error> {
        run(ck, 1, &Policy::default())
    }

    fn names(c: &ModelContract) -> Vec<String> {
        c.tensors.iter().map(|t| t.name.clone()).collect()
    }

    fn shape_of(c: &ModelContract, name: &str) -> Vec<i64> {
        c.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no tensor {name}; have {:?}", names(c)))
            .shape
            .clone()
            .unwrap_or_else(|| panic!("{name} has no declared shape"))
    }

    // ── The shard-axis rule, which is why this family has its own file ─────

    /// Gate and up split their OUT dim; down splits its IN dim.
    ///
    /// This is the whole reason `dsv4_shard_axis` exists. The intermediate
    /// dimension is split within each expert, so every rank computes a
    /// partial expert output and an all-reduce combines them -- which only
    /// works if `w1`/`w3` are cut on axis 0 and `w2` on axis 1. Cutting `w2`
    /// on axis 0 instead splits the HIDDEN dim, and each rank then produces a
    /// slice of the output rather than a partial sum of all of it. The
    /// all-reduce still runs, still returns a tensor of the right shape, and
    /// the model answers fluently from a third of its own weights.
    ///
    /// Asked through the builder rather than the free function, because the
    /// builder is what the loader consults and it strips companion suffixes
    /// on the way in.
    #[test]
    fn gate_and_up_shard_their_out_dim_and_down_shards_its_in_dim() {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: moe(1).0,
        };
        let enc = StoredEncoding::dense();
        let t = target(2);
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "deepseek-v4-test",
            LoadShape::mixture(1, 128, 1, true),
            &enc,
            &t,
            &policy,
        );
        b.shard_axis_fn(dsv4_shard_axis);

        for (name, want) in [
            ("layers.0.ffn.experts.0.w1.weight", Some(0)),
            ("layers.0.ffn.experts.0.w3.weight", Some(0)),
            ("layers.0.ffn.experts.0.w2.weight", Some(1)),
            ("layers.0.ffn.shared_experts.w1.weight", Some(0)),
            ("layers.0.ffn.shared_experts.w3.weight", Some(0)),
            ("layers.0.ffn.shared_experts.w2.weight", Some(1)),
            // Outside the FFN: replicated, to keep TP traffic off the main path.
            ("model.embed_tokens.weight", None),
            ("layers.0.attn.q_proj.weight", None),
            // Inside the FFN but not a projection: the router gate and the
            // norms are named exceptions, not fall-through.
            ("layers.0.ffn.gate.weight", None),
            ("layers.0.ffn.ffn_norm.weight", None),
            ("layers.0.ffn.post_attention_layernorm.weight", None),
        ] {
            assert_eq!(
                b.shard_axis(name).expect("a decision"),
                want,
                "wrong shard axis for {name}"
            );
        }
    }

    /// A companion scale is decided by the weight it scales.
    ///
    /// `Builder::shard_axis` strips the suffix before consulting the family
    /// rule, so `dsv4_shard_axis` never sees a `.scale` and must not list
    /// one. Listing scales again is how the two lists drift apart: a weight
    /// added to one and not the other shards its data on one axis and its
    /// exponents on the other, which dequantizes to numbers with no meaning
    /// and no error.
    #[test]
    fn a_block_scale_shards_on_the_axis_of_the_weight_it_scales() {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: moe(1).0,
        };
        let enc = StoredEncoding::dense();
        let t = target(2);
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "deepseek-v4-test",
            LoadShape::mixture(1, 128, 1, true),
            &enc,
            &t,
            &policy,
        );
        b.shard_axis_fn(dsv4_shard_axis);

        for (weight, scale) in [
            (
                "layers.0.ffn.experts.0.w1.weight",
                "layers.0.ffn.experts.0.w1.scale",
            ),
            (
                "layers.0.ffn.experts.0.w2.weight",
                "layers.0.ffn.experts.0.w2.scale",
            ),
            (
                "layers.0.ffn.shared_experts.w3.weight",
                "layers.0.ffn.shared_experts.w3.scale",
            ),
        ] {
            assert_eq!(
                b.shard_axis(scale).expect("a decision"),
                b.shard_axis(weight).expect("a decision"),
                "{scale} does not follow {weight}"
            );
        }
        // And the family rule itself has never heard of a scale.
        assert_eq!(
            dsv4_shard_axis("layers.0.ffn.experts.0.w1.scale").expect("a decision"),
            None,
            "dsv4_shard_axis answered for a companion; the suffix strip is being bypassed"
        );
    }

    /// An FFN projection this rule has never seen is REFUSED, not replicated.
    ///
    /// Falling through to `Ok(None)` is the dangerous answer here. A
    /// checkpoint variant that spells an expert differently would be
    /// replicated in full while the forward pass went on sharding around it,
    /// and the model would answer plausibly and wrongly -- the failure mode
    /// with no symptom. The refusal names the function to add it to.
    #[test]
    fn an_unrecognised_ffn_projection_is_refused_rather_than_silently_replicated() {
        let e = dsv4_shard_axis("layers.0.ffn.experts.0.w4.weight")
            .expect_err("an unknown FFN projection was accepted");
        let msg = format!("{e:?}");
        assert!(
            msg.contains("dsv4_shard_axis"),
            "the refusal should name the function to extend: {msg}"
        );

        // A spelling the rule does not know, outside `.experts.`, is refused
        // just the same -- the guard is the `.ffn.` prefix, not the word
        // "experts".
        assert!(dsv4_shard_axis("layers.0.ffn.mlp_up.weight").is_err());

        // The three escapes are exact, and each carries its own arm.
        for ok in [
            "layers.0.ffn.gate.weight",            // router logits
            "layers.0.ffn.q_norm.weight",          // `_norm.`
            "layers.0.ffn.input_layernorm.weight", // `layernorm`
            "layers.0.ffn.experts.0.w1.bias",      // not `.weight`
        ] {
            assert_eq!(
                dsv4_shard_axis(ok).expect("should not be refused"),
                None,
                "{ok} should replicate"
            );
        }
    }

    // ── The two ways the routed experts reach the driver ───────────────────

    /// Stacking is the default, and the stack is one slab per layer.
    ///
    /// The forward pass wants `[E, 2I, H]` gate-over-up and `[E, H, I]` down,
    /// because a batched GEMM over experts wants one base pointer and a
    /// stride rather than 3E separate tensors. Everything about the shapes is
    /// load-bearing: the `2I` is gate concatenated with up on the INNER axis,
    /// the leading `E` is the outer stack, and `H` is twice the packed extent
    /// because two E2M1 nibbles ride in each `I8` byte.
    ///
    /// The per-expert sources must also be GONE. A stacked slab that leaves
    /// its inputs published binds the same weights twice and doubles the
    /// residency the stack exists to make contiguous.
    #[test]
    fn the_default_lowering_stacks_every_expert_of_a_layer_into_two_slabs() {
        let c = plain(moe(4)).expect("a four-expert layer");
        let n = names(&c);

        assert_eq!(
            shape_of(&c, "layers.0.ffn.experts.gate_up.weight"),
            vec![4, 2 * I, H],
            "gate_up is not [E, 2I, H]"
        );
        assert_eq!(
            shape_of(&c, "layers.0.ffn.experts.down.weight"),
            vec![4, H, I],
            "down is not [E, H, I]"
        );

        // Not one tensor of the six-per-expert sources survives.
        let leftovers: Vec<&String> = n.iter().filter(|s| s.contains(".experts.0.")).collect();
        assert!(
            leftovers.is_empty(),
            "the stacked sources were published too: {leftovers:?}"
        );

        // The scale slabs exist but are internal: `scale_per_block` takes its
        // factors by output name and the slab is dequantized at load, so no
        // kernel reads them again.
        for slab in ["gate_up", "down"] {
            let s = format!("layers.0.ffn.experts.{slab}.scale");
            let t = c
                .tensors
                .iter()
                .find(|t| t.name == s)
                .unwrap_or_else(|| panic!("no {s}"));
            assert_eq!(
                t.visibility,
                Visibility::Internal,
                "{s} is public, but nothing reads it after the dequantize"
            );
        }
    }

    /// Streaming and stacking are alternatives, not layers.
    ///
    /// A stack is one slab per layer holding every expert, which is exactly
    /// the residency a paging driver is trying to avoid, so asking for both
    /// would defeat the request. The streamed form declares ONE expert with
    /// the index left standing, and the names carry no layer and no expert
    /// because there is one plan reused for every instance.
    #[test]
    fn streaming_replaces_the_stack_rather_than_adding_to_it() {
        let policy = Policy {
            stream_routed_experts: true,
            ..Policy::default()
        };
        let c = run(moe(4), 1, &policy).expect("a streamed four-expert layer");
        let n = names(&c);

        // Nothing resident.
        assert!(
            !n.iter().any(|s| s.contains("experts.gate_up")),
            "streaming still built the resident stack: {n:?}"
        );

        // One group, with the expert count as its arity.
        assert_eq!(c.groups.len(), 1, "expected one group per layer");
        let g = &c.groups[0];
        assert_eq!(g.arity, 4, "the group's arity is not the expert count");

        // The declared names carry no layer and no expert: there is one plan
        // and it is the same plan for every instance of every layer.
        let gn: Vec<&str> = g.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            gn.contains(&"gate_up.weight") && gn.contains(&"down.weight"),
            "the group does not offer the two names the per-expert GEMM asks \
             for: {gn:?}"
        );
        for name in &gn {
            assert!(
                !name.contains("layers.") && !name.contains("experts."),
                "a group tensor carries an instance in its name: {name}"
            );
        }

        // One INSTANCE's shapes. The group is the same structure as the
        // stack ONE RANK LOWER: there is no expert axis to stack along, so
        // the `E` the slab carries is replaced by the group's arity and each
        // instance is rank 2. A group whose tensors were rank 3 would be a
        // stack of one, which is the residency streaming exists to avoid.
        let gs = |want: &str| {
            g.tensors
                .iter()
                .find(|t| t.name == want)
                .unwrap_or_else(|| panic!("no {want} in {gn:?}"))
                .shape
                .clone()
                .unwrap_or_else(|| panic!("{want} has no declared shape"))
        };
        assert_eq!(gs("gate_up.weight"), vec![2 * I, H]);
        assert_eq!(gs("down.weight"), vec![H, I]);
        // Against the resident form of the same checkpoint, which is rank 3.
        let stacked = plain(moe(4)).expect("the resident form");
        assert_eq!(
            shape_of(&stacked, "layers.0.ffn.experts.gate_up.weight"),
            vec![4, 2 * I, H],
            "the two lowerings should differ by exactly the expert axis"
        );

        // The per-expert sources are claimed: a group reads through a
        // template, so no node names them and `publish_remaining` would
        // otherwise emit all 24 of them as ordinary dense tensors.
        let leftovers: Vec<&String> = n.iter().filter(|s| s.contains(".experts.")).collect();
        assert!(
            leftovers.is_empty(),
            "the streamed sources were published as dense tensors: {leftovers:?}"
        );
    }

    /// `RoutedDecode` is the one request that turns the eager stack off.
    ///
    /// The device rule `resolve_mxfp4_moe` applies is not this family's: it
    /// asks whether there are native MXFP4 GEMM kernels to fall back FROM,
    /// and DeepSeek-V4 has none. The choice here is between dequantizing
    /// eagerly at load and dequantizing a slice per step, so everything short
    /// of an explicit `RoutedDecode` -- including `NativeGemm`, which asks for
    /// kernels that do not exist -- takes the eager path.
    #[test]
    fn only_an_explicit_routed_decode_request_leaves_the_experts_packed() {
        for (req, eager) in [
            (Mxfp4MoeRequest::Auto, true),
            (Mxfp4MoeRequest::EagerBf16, true),
            (Mxfp4MoeRequest::NativeGemm, true),
            (Mxfp4MoeRequest::RoutedDecode, false),
        ] {
            let policy = Policy {
                moe_request: req,
                ..Policy::default()
            };
            let c = run(moe(2), 1, &policy).unwrap_or_else(|e| panic!("{req:?}: {e:?}"));
            let stacked = names(&c)
                .iter()
                .any(|s| s == "layers.0.ffn.experts.gate_up.weight");
            assert_eq!(stacked, eager, "{req:?} took the wrong lowering");
        }
    }

    /// The layer prefix is ASKED, not declared.
    ///
    /// This family's released checkpoints name their layers `layers.<L>.`,
    /// not the HF `model.layers.<L>.` the default assumes. With the prefix
    /// wrong both expert passes match nothing, return `Ok(())`, and the
    /// forward pass's packed fallback quietly covers for them -- so the
    /// symptom of getting this wrong is not an error, it is the slow path,
    /// forever. Both spellings have to reach the same two slabs.
    #[test]
    fn both_spellings_of_the_layer_prefix_reach_the_experts() {
        let mut hf = Ck::new().push("model.norm.weight", &[H], bf16());
        for e in 0..2 {
            hf = expert(hf, "model.layers.0.ffn.", e);
        }
        hf.1 = 2;
        let c = plain(hf).expect("an HF-prefixed checkpoint");
        assert_eq!(
            shape_of(&c, "model.layers.0.ffn.experts.gate_up.weight"),
            vec![2, 2 * I, H],
            "the `model.layers.` spelling did not reach the expert pass"
        );

        // And the bare spelling reaches the same two slabs, under its own
        // name. Both are this family; neither is a fallback for the other.
        let c = plain(moe(2)).expect("a bare-prefixed checkpoint");
        assert_eq!(
            shape_of(&c, "layers.0.ffn.experts.gate_up.weight"),
            vec![2, 2 * I, H]
        );
    }

    // ── Refusals and escapes ──────────────────────────────────────────────

    /// An expert missing half of itself is refused by name.
    #[test]
    fn an_expert_missing_a_weight_or_scale_is_refused() {
        // Drop `w3.scale` from expert 1 of a two-expert layer.
        let mut ck = Ck::new().push("model.norm.weight", &[H], bf16());
        ck = expert(ck, "layers.0.ffn.", 0);
        let ep = "layers.0.ffn.experts.1.";
        ck = ck
            .push(&format!("{ep}w1.weight"), &[I, H / 2], i8e())
            .push(&format!("{ep}w1.scale"), &[I, H / 32], u8e())
            .push(&format!("{ep}w3.weight"), &[I, H / 2], i8e())
            .push(&format!("{ep}w2.weight"), &[H, I / 2], i8e())
            .push(&format!("{ep}w2.scale"), &[H, I / 32], u8e());

        let e = plain(ck).expect_err("a half-present expert was accepted");
        let msg = format!("{e:?}");
        assert!(
            msg.contains("experts.1") && msg.contains("missing"),
            "the refusal should name the expert: {msg}"
        );
    }

    /// A checkpoint whose experts are not packed MXFP4 is left ENTIRELY
    /// alone, not half-rewritten.
    ///
    /// The early `return Ok(())` is inside the per-expert loop, so a mixed
    /// checkpoint stops the whole pass rather than stacking the experts it
    /// already visited. Half a rewrite is worse than none: it publishes some
    /// experts as slabs and the rest as raw tensors, and the forward pass can
    /// serve neither shape for the other.
    #[test]
    fn experts_that_are_not_packed_mxfp4_are_left_to_the_dense_pass() {
        let mut ck = Ck::new().push("model.norm.weight", &[H], bf16());
        // Expert 0 is bf16 rather than packed I8.
        let ep = "layers.0.ffn.experts.0.";
        ck = ck
            .push(&format!("{ep}w1.weight"), &[I, H], bf16())
            .push(&format!("{ep}w1.scale"), &[I, H / 32], u8e())
            .push(&format!("{ep}w3.weight"), &[I, H], bf16())
            .push(&format!("{ep}w3.scale"), &[I, H / 32], u8e())
            .push(&format!("{ep}w2.weight"), &[H, I], bf16())
            .push(&format!("{ep}w2.scale"), &[H, I / 32], u8e());

        let c = plain(ck).expect("a bf16-expert checkpoint should load, not fail");
        let n = names(&c);
        assert!(
            !n.iter().any(|s| s.contains("experts.gate_up")),
            "a bf16 checkpoint was stacked anyway: {n:?}"
        );
        assert!(
            n.iter().any(|s| s == "layers.0.ffn.experts.0.w1.weight"),
            "the untouched experts were not published by the dense pass: {n:?}"
        );
    }

    /// Expert dims that are not a multiple of the 32-element MXFP4 group are
    /// refused, in both passes.
    #[test]
    fn expert_dims_that_do_not_divide_the_mxfp4_group_are_refused() {
        let odd = |prefix: &str| {
            // I = 48 is not a multiple of 32; H stays legal so the refusal is
            // attributable to one dim.
            let ep = format!("{prefix}experts.0.");
            Ck::new()
                .push("model.norm.weight", &[H], bf16())
                .push(&format!("{ep}w1.weight"), &[48, H / 2], i8e())
                .push(&format!("{ep}w1.scale"), &[48, H / 32], u8e())
                .push(&format!("{ep}w3.weight"), &[48, H / 2], i8e())
                .push(&format!("{ep}w3.scale"), &[48, H / 32], u8e())
                .push(&format!("{ep}w2.weight"), &[H, 24], i8e())
                .push(&format!("{ep}w2.scale"), &[H, 48 / 32], u8e())
        };

        let e = plain(odd("layers.0.ffn.")).expect_err("stacking accepted a ragged expert");
        assert!(format!("{e:?}").contains("multiple of 32"), "{e:?}");

        let policy = Policy {
            stream_routed_experts: true,
            ..Policy::default()
        };
        let e =
            run(odd("layers.0.ffn."), 1, &policy).expect_err("streaming accepted a ragged expert");
        assert!(format!("{e:?}").contains("multiple of 32"), "{e:?}");
    }

    /// Rank-3 expert weights are refused rather than indexed past their end.
    #[test]
    fn expert_weights_that_are_not_rank_2_are_refused() {
        let ep = "layers.0.ffn.experts.0.";
        let ck = Ck::new()
            .push("model.norm.weight", &[H], bf16())
            .push(&format!("{ep}w1.weight"), &[1, I, H / 2], i8e())
            .push(&format!("{ep}w1.scale"), &[I, H / 32], u8e())
            .push(&format!("{ep}w3.weight"), &[1, I, H / 2], i8e())
            .push(&format!("{ep}w3.scale"), &[I, H / 32], u8e())
            .push(&format!("{ep}w2.weight"), &[1, H, I / 2], i8e())
            .push(&format!("{ep}w2.scale"), &[H, I / 32], u8e());

        let e = plain(ck).expect_err("a rank-3 expert weight was accepted");
        assert!(format!("{e:?}").contains("rank-2"), "{e:?}");
    }

    /// Block scales beside an FP8 weight become fp32 factors; the ones beside
    /// packed MXFP4 experts do not.
    ///
    /// Both are `U8` companions named `.scale`, and only the F8E4M3 guard on
    /// the WEIGHT tells them apart. Widen the guard and the routed experts'
    /// exponent bytes get declared as fp32 factors of a tensor that has
    /// already been dequantized and consumed.
    ///
    /// The group size is read off the two shapes rather than assumed, so a
    /// checkpoint with a different tile width states its own.
    #[test]
    fn a_block_scale_is_only_an_fp32_factor_when_it_rides_beside_an_fp8_weight() {
        let ck = Ck::new()
            .push("model.norm.weight", &[H], bf16())
            .push("layers.0.attn.q_proj.weight", &[H, H], fp8())
            .push("layers.0.attn.q_proj.scale", &[H, H / 64], u8e());

        let c = plain(ck).expect("an fp8 dense layer");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == "layers.0.attn.q_proj.scale")
            .expect("the scale was not published");
        let s = t.scales.as_ref().expect("no Scales attached to the scale");
        assert_eq!(s.of, "layers.0.attn.q_proj.weight");
        assert_eq!(s.form, ScaleForm::F32Factors);
        assert_eq!(
            s.group_size, 64,
            "the block size should be read off the two shapes (128 / 2 = 64)"
        );
        assert_eq!(s.granularity, QuantGranularity::PerGroup);
    }

    /// A `.scale` with no weight beside it, and one beside a weight that is
    /// not FP8, are both passed over rather than refused.
    #[test]
    fn an_orphan_or_non_fp8_block_scale_is_left_alone() {
        let ck = Ck::new()
            .push("model.norm.weight", &[H], bf16())
            // No `orphan.weight` anywhere.
            .push("layers.0.attn.orphan.scale", &[H, 2], u8e())
            // A bf16 weight is not block-FP8.
            .push("layers.0.attn.k_proj.weight", &[H, H], bf16())
            .push("layers.0.attn.k_proj.scale", &[H, 2], u8e());

        let c = plain(ck).expect("neither case should refuse");
        for name in ["layers.0.attn.orphan.scale", "layers.0.attn.k_proj.scale"] {
            let t = c
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{name} was dropped"));
            assert!(
                t.scales.is_none(),
                "{name} was given fp32 factors it does not scale"
            );
        }
    }
    /// A bank SHORTER than the row states is refused, by both passes.
    ///
    /// Both walks probe a name -- `experts.{e}.w1.weight` -- to decide
    /// whether expert `e` exists, so a checkpoint missing exactly that
    /// name does not look like a hole. It looks like the END of the
    /// bank, and the walk stops. Every other missing part is refused by
    /// name; that one would silently build a shorter bank.
    ///
    /// Nothing downstream catches it: the manifest measures the ROUTER,
    /// `[num_experts, hidden]`, and a checkpoint missing a whole expert
    /// still carries a full-width router, so the row matches and the
    /// load succeeds. The fused GEMM then indexes a slab with fewer rows
    /// than the router emits indices for.
    #[test]
    fn a_bank_shorter_than_the_row_states_is_refused_by_both_expert_passes() {
        for streaming in [false, true] {
            let mut ck = moe(3);
            // Say three, ship two: the third expert's probe is simply
            // absent, which is what a truncated upload looks like.
            ck.0.retain(|t| !t.name.starts_with("layers.0.ffn.experts.2."));
            let policy = Policy {
                stream_routed_experts: streaming,
                ..Policy::default()
            };
            let why = run(ck, 1, &policy).expect_err("a short bank is refused");
            let Error::Contract(why) = why else {
                panic!("expected a contract refusal, got {why:?}")
            };
            assert!(
                why.contains("the row states 3") && why.contains("router emits indices"),
                "streaming={streaming}: {why}"
            );
        }
    }

    /// A bank LONGER than the row states is refused too.
    ///
    /// The check is an equality and not a floor. A bank with more
    /// instances than the router can address is a different checkpoint
    /// from the one this row identifies, and building it would bind
    /// weight nothing routes to.
    #[test]
    fn a_bank_longer_than_the_row_states_is_refused_by_both_expert_passes() {
        for streaming in [false, true] {
            let mut ck = moe(2);
            ck.1 = 1;
            let policy = Policy {
                stream_routed_experts: streaming,
                ..Policy::default()
            };
            let why = run(ck, 1, &policy).expect_err("a long bank is refused");
            let Error::Contract(why) = why else {
                panic!("expected a contract refusal, got {why:?}")
            };
            assert!(
                why.contains("the row states 1"),
                "streaming={streaming}: {why}"
            );
        }
    }

    /// A layer with no experts at all is passed over, not refused.
    ///
    /// This family's leading layers are dense, so the walk has to reach
    /// the end of a dense layer and move on. The count check must not
    /// fire there -- a dense layer stacking zero experts against a row
    /// that states 64 is the normal case, not a short bank.
    ///
    /// The two passes reach that answer differently, and only one of
    /// them is testable here. `bf16_expert_stacks` really does walk into
    /// the layer and find nothing. `streamed_expert_groups` breaks out
    /// of its OUTER loop first, so its own `experts == 0` branch cannot
    /// fire -- a control that deleted it changed nothing. The branch is
    /// kept and says so where it lives.
    #[test]
    fn a_dense_layer_is_passed_over_rather_than_counted_against_the_row() {
        for streaming in [false, true] {
            let policy = Policy {
                stream_routed_experts: streaming,
                ..Policy::default()
            };
            let mut ck = Ck::new().push("model.norm.weight", &[H], bf16());
            ck.1 = 64;
            let c = run(ck, 1, &policy).expect("a checkpoint with no experts anywhere");
            assert!(
                !names(&c).iter().any(|n| n.contains("experts.gate_up")),
                "a dense-only checkpoint produced an expert slab"
            );
        }
    }

    /// An expert missing a part that is NOT the walk's probe is refused
    /// by name, and the message locates it.
    #[test]
    fn an_expert_missing_a_part_that_is_not_the_probe_is_refused_and_named() {
        for (streaming, wanted) in [(false, "expert stack"), (true, "expert group")] {
            for suffix in ["w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale"] {
                let mut ck = moe(2);
                let gone = format!("layers.0.ffn.experts.1.{suffix}");
                ck.0.retain(|t| t.name != gone);
                let policy = Policy {
                    stream_routed_experts: streaming,
                    ..Policy::default()
                };
                let why = run(ck, 1, &policy).expect_err("a hole is refused");
                let Error::Contract(why) = why else {
                    panic!("expected a contract refusal, got {why:?}")
                };
                assert!(
                    why.contains("missing a weight or scale") && why.contains(wanted),
                    "a missing {suffix} at streaming={streaming}: {why}"
                );
            }
        }
    }

    /// Siblings that disagree on shape are refused, naming the sibling.
    #[test]
    fn an_expert_that_disagrees_with_its_siblings_on_shape_is_refused() {
        let mut ck = moe(3);
        for t in &mut ck.0 {
            if t.name == "layers.0.ffn.experts.1.w1.weight" {
                t.shape = vec![I * 2, H / 2];
            }
        }
        let why = plain(ck).expect_err("a mismatched sibling is refused");
        let Error::Contract(why) = why else {
            panic!("expected a contract refusal, got {why:?}")
        };
        assert!(
            why.contains("disagrees with its siblings") && why.contains("experts.1."),
            "{why}"
        );
    }
}
