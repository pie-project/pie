use model_loader::checkpoint::RawTensor;
use model_loader::contract::{Expr, TensorType};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, TensorId};

use crate::shared::builder::{Builder, is_raw, mxfp4_encoding};
use crate::shared::probe::hf_shard_axis;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

const _EXPERTS_ARE_MXFP4: () = {
    match <super::forward::ShippedW2 as model_dsl::axes::DtypeAxis>::REPR {
        model_dsl::WeightRepr::Mxfp4Marlin => (),
        _ => panic!(
            "kimi-k3's catalogued expert axis moved off MXFP4; \
             the expert stacker spells that axis"
        ),
    }
};

pub fn author_kimi_k3(b: &mut Builder<'_>) -> Result<(), Error> {

    b.source_prefix("language_model.");
    b.shard_axis_fn(kimi_k3_shard_axis);
    b.shard_embed_tokens();
    a_log_bands(b)?;

    bf16_expert_stacks(b,  false)?;

    b.fused_moe_gate_up_tp_slices(false)?;
    b.publish_remaining()
}

fn kimi_k3_shard_axis(name: &str) -> Result<Option<u8>, Error> {

    if [
        ".routed_expert_down_proj.weight",
        ".routed_expert_up_proj.weight",
        ".routed_expert_norm.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
    {
        return Ok(None);
    }

    if [
        ".self_attn.f_a_proj.weight",
        ".self_attn.o_norm.weight",
        ".self_attention_res_proj.weight",
        ".mlp_res_proj.weight",
        ".self_attention_res_norm.weight",
        ".mlp_res_norm.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
    {
        return Ok(None);
    }

    if [
        ".self_attn.g_proj.weight",
        ".self_attn.f_b_proj.weight",
        ".self_attn.b_proj.weight",
        ".self_attn.dt_bias",
        ".self_attn.q_conv1d.weight",
        ".self_attn.k_conv1d.weight",
        ".self_attn.v_conv1d.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
    {
        return Ok(Some(0));
    }

    Ok(hf_shard_axis(name))
}

fn a_log_bands(b: &mut Builder<'_>) -> Result<(), Error> {
    for layer in 0..b.shape().layers {
        let layer_prefix = format!("{}{layer}.self_attn.", b.decoder_layer_prefix_value());
        let (Some(raw), Some(beta)) = (
            b.find(&b.source_name(&format!("{layer_prefix}A_log"))),
            b.find(&b.source_name(&format!("{layer_prefix}b_proj.weight"))),
        ) else {
            continue;
        };
        if raw.shape.len() != 1 || beta.shape.is_empty() {
            return fail(format!(
                "kimi_k3 A_log band: layer {layer} has an unexpected A_log / b_proj rank"
            ));
        }
        let heads = beta.shape[0];
        if raw.shape[0] < heads {
            return fail(format!(
                "kimi_k3 A_log band: layer {layer} has {} gate entries for {heads} heads",
                raw.shape[0]
            ));
        }
        let (banded, rows) = b.band(Expr::src(&raw.name), 0, 0, heads);
        let encoding = raw.encoding.clone();
        let id = raw.id;
        b.define(b.output_name(&raw.name), banded, encoding, Some(vec![rows]));
        b.consume(id);
    }
    Ok(())
}

fn bf16_expert_stacks(b: &mut Builder<'_>, gate_second: bool) -> Result<(), Error> {
    const GROUP: i64 = 32;
    let experts = i64::from(b.shape().n_experts);
    if experts <= 0 {
        return Ok(());
    }

    for layer in 0..b.shape().layers {
        let moe = format!(
            "{}{layer}.block_sparse_moe.",
            b.decoder_layer_prefix_value()
        );
        let prefix = b.source_name(&moe);

        if b.find(&format!("{prefix}experts.0.w1.weight_packed"))
            .is_none()
        {
            continue;
        }

        let mut gate_up = Vec::new();
        let mut gate_up_scales = Vec::new();
        let mut down = Vec::new();
        let mut down_scales = Vec::new();
        let mut consumed: Vec<TensorId> = Vec::new();
        let mut local_inter = 0i64;
        let mut latent = 0i64;

        let packed = |b: &Builder<'_>, raw: &RawTensor, shape: Vec<i64>, axis: u8| {
            b.shard(
                Expr::src(&raw.name).transmute(TensorType::new(shape.clone(), mxfp4_encoding(2))),
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

        for e in 0..experts {
            let ep = format!("{prefix}experts.{e}.");
            let names = [
                format!("{ep}w1.weight_packed"),
                format!("{ep}w1.weight_scale"),
                format!("{ep}w3.weight_packed"),
                format!("{ep}w3.weight_scale"),
                format!("{ep}w2.weight_packed"),
                format!("{ep}w2.weight_scale"),
            ];
            let mut parts = Vec::with_capacity(6);
            for name in &names {
                let Some(part) = b.find(name) else {
                    return fail(format!(
                        "kimi_k3 expert stack: layer {layer} expert {e} is missing a \
                         weight or scale"
                    ));
                };
                parts.push(part);
            }

            if parts
                .iter()
                .any(|part| !is_raw(&part.encoding, DType::U8) || part.shape.len() != 2)
            {
                return Ok(());
            }

            let inter = parts[0].shape[0];
            let latent_here = parts[0].shape[1] * 2;
            if parts[4].shape[0] != latent_here
                || parts[4].shape[1] * 2 != inter
                || parts[1].shape[1] != latent_here / GROUP
                || parts[5].shape[1] != inter / GROUP
            {
                return fail(format!(
                    "kimi_k3 expert stack: layer {layer} expert {e} has inconsistent \
                     MXFP4 shapes"
                ));
            }
            if e == 0 {
                latent = latent_here;
                local_inter = b.local_extent(inter);
            } else if latent_here != latent {
                return fail(format!(
                    "kimi_k3 expert stack: layer {layer} expert {e} changes the latent width"
                ));
            }

            let w1 = packed(b, parts[0], vec![1, inter, latent], 1);
            let w3 = packed(b, parts[2], vec![1, inter, latent], 1);
            let w1s = factors(b, parts[1], vec![1, inter, latent / GROUP], 1);
            let w3s = factors(b, parts[3], vec![1, inter, latent / GROUP], 1);

            let pair = |b: &Builder<'_>, a: &RawTensor, c: &RawTensor, cols: i64| {
                let u8enc = Encoding::Raw(DType::U8);
                let an = b
                    .split(Expr::src(&a.name), 0)
                    .transmute(TensorType::new(vec![local_inter, 1, cols], u8enc.clone()));
                let cn = b
                    .split(Expr::src(&c.name), 0)
                    .transmute(TensorType::new(vec![local_inter, 1, cols], u8enc));
                Expr::concat(1, vec![an, cn])
            };
            gate_up.push(Expr::concat(
                1,
                if gate_second {
                    vec![w3, w1]
                } else {
                    vec![w1, w3]
                },
            ));
            gate_up_scales.push(Expr::concat(
                1,
                if gate_second {
                    vec![w3s, w1s]
                } else {
                    vec![w1s, w3s]
                },
            ));
            down.push(packed(b, parts[4], vec![1, latent, inter], 2));
            down_scales.push(factors(b, parts[5], vec![1, latent, inter / GROUP], 2));

            let ep_out = format!("{moe}experts.{e}.");
            let gu_packed = pair(b, parts[0], parts[2], latent / 2);
            let gu_scale = pair(b, parts[1], parts[3], latent / GROUP);
            let dn_packed = b.split(Expr::src(&parts[4].name), 1);
            let dn_scale = b.split(Expr::src(&parts[5].name), 1);
            let u8enc = Encoding::Raw(DType::U8);
            b.define(
                format!("{ep_out}gate_up.weight_packed"),
                gu_packed,
                u8enc.clone(),
                Some(vec![local_inter, 2, latent / 2]),
            );
            b.define(
                format!("{ep_out}gate_up.weight_scale"),
                gu_scale,
                u8enc.clone(),
                Some(vec![local_inter, 2, latent / GROUP]),
            );
            b.define(
                format!("{ep_out}down.weight_packed"),
                dn_packed,
                u8enc.clone(),
                Some(vec![latent, local_inter / 2]),
            );
            b.define(
                format!("{ep_out}down.weight_scale"),
                dn_scale,
                u8enc,
                Some(vec![latent, local_inter / GROUP]),
            );
            consumed.extend(parts.iter().map(|part| part.id));
        }

        let e8m0 = Encoding::Raw(DType::E8M0);
        let gu_scale = format!("{moe}experts.gate_up.scale");
        let dn_scale = format!("{moe}experts.down.scale");
        let gu = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            e8m0.clone(),
            Some(vec![experts, 2 * local_inter, latent / GROUP]),
        );
        b.mark_internal(gu);
        let dn = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            e8m0,
            Some(vec![experts, latent, local_inter / GROUP]),
        );
        b.mark_internal(dn);
        b.define(
            format!("{moe}experts.gate_up_proj"),
            Expr::concat(0, gate_up).scale_per_block(Expr::out(&gu_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, latent]),
        );
        b.define(
            format!("{moe}experts.down_proj"),
            Expr::concat(0, down).scale_per_block(Expr::out(&dn_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, latent, local_inter]),
        );
        for id in consumed {
            b.consume(id);
        }
    }
    Ok(())
}
