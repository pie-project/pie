use model_loader::checkpoint::RawTensor;
use model_loader::contract::{Expr, GroupContract, Scales, TensorContract, TensorType};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, QuantGranularity, ScaleForm, TensorId, Visibility};

use crate::shared::builder::{Builder, is_raw, mxfp4_encoding};
use crate::shared::policy::{Mxfp4MoePolicy, Mxfp4MoeRequest};

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub fn author_deepseek_v4(b: &mut Builder<'_>) -> Result<(), Error> {
    b.decoder_layer_prefix_any_of(&["model.layers.", "layers."]);
    b.shard_axis_fn(dsv4_shard_axis);
    b.decide_mxfp4_moe(if b.mxfp4_moe_request() == Mxfp4MoeRequest::RoutedDecode {
        Mxfp4MoePolicy::RoutedDecode
    } else {
        Mxfp4MoePolicy::EagerBf16
    });
    if b.stream_routed_experts() {
        streamed_expert_groups(b)?;
    } else if b.mxfp4_moe() == Mxfp4MoePolicy::EagerBf16 {
        bf16_expert_stacks(b)?;
    }
    block_scales_to_fp32(b)?;

    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

fn dsv4_shard_axis(name: &str) -> Result<Option<u8>, Error> {
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

    Ok(None)
}

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

        let weight_shape = companion.shape.clone();
        let scale_cols = *shape.last().unwrap_or(&0);
        let weight_cols = *weight_shape.last().unwrap_or(&0);

        let Some(block) = weight_cols.checked_div(scale_cols).filter(|&b| b > 0) else {
            return fail(format!(
                "deepseek_v4 block scales: '{}' is {shape:?} beside a \
                 {weight_shape:?} weight, which states no block size",
                raw.name
            ));
        };
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
        b.consume(id);
    }
    Ok(())
}

fn bf16_expert_stacks(b: &mut Builder<'_>) -> Result<(), Error> {
    const GROUP: i64 = 32;

    let mut layer = 0u32;
    loop {
        let ffn = format!("{}{layer}.ffn.", b.decoder_layer_prefix_value());
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

            if !is_raw(&parts[0].encoding, DType::I8) || !is_raw(&parts[4].encoding, DType::I8) {
                return Ok(());
            }

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
            break;
        }
        let experts = gate_up.len() as i64;

        if experts != i64::from(b.shape().n_experts) {
            return fail(format!(
                "deepseek_v4 expert stack: layer {layer} stacked {experts} experts \
                 but the row states {}; the router emits indices this slab has no \
                 rows for",
                b.shape().n_experts
            ));
        }

        let e8m0 = Encoding::Raw(DType::E8M0);
        let gu_scale = format!("{ffn}experts.gate_up.scale");
        let dn_scale = format!("{ffn}experts.down.scale");
        let gu = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            e8m0.clone(),
            Some(vec![experts, 2 * local_inter, hidden / GROUP]),
        );
        b.mark_internal(gu);
        let dn = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            e8m0,
            Some(vec![experts, hidden, local_inter / GROUP]),
        );
        b.mark_internal(dn);
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
