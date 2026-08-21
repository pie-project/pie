use model_dsl::axes::DtypeAxis;
use model_loader::contract::Expr;
use model_loader::error::Error;
use model_loader::types::{BackendKind, DType, Encoding};

use crate::shared::builder::{Builder, int4b8_encoding, is_raw};

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub fn author_deepseek_mla(b: &mut Builder<'_>) -> Result<(), Error> {
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    mla_fused_projection_joins(b)?;
    b.publish_remaining()
}

pub fn author_kimi(b: &mut Builder<'_>) -> Result<(), Error> {
    b.source_prefix("language_model.");
    b.shard_embed_tokens();
    b.replicate_lm_head();
    bf16_expert_stacks(b,  4u64 << 30)?;
    author_deepseek_mla(b)
}

fn mla_fused_projection_joins(b: &mut Builder<'_>) -> Result<(), Error> {
    let mut candidates = Vec::new();
    for layer in 0..b.shape().layers {
        let p = format!("model.layers.{layer}.");
        let s = b.source_name(&p);

        if let Some(candidate) = b.fused_join_candidate(
            format!("{p}self_attn.q_kv_a_proj.fused.weight"),
            &[
                format!("{s}self_attn.q_a_proj.weight"),
                format!("{s}self_attn.kv_a_proj_with_mqa.weight"),
            ],
        ) {
            candidates.push(candidate);
        }

        if let Some(candidate) = b.fused_join_candidate(
            format!("{p}mlp.shared_experts.gate_up_proj.fused.weight"),
            &[
                format!("{s}mlp.shared_experts.gate_proj.weight"),
                format!("{s}mlp.shared_experts.up_proj.weight"),
            ],
        ) {
            candidates.push(candidate);
        }
    }
    b.publish_fused(candidates)
}

fn bf16_expert_stacks(b: &mut Builder<'_>, budget: u64) -> Result<(), Error> {

    const GROUP: i64 = match <super::forward::ShippedW2 as DtypeAxis>::REPR {
        model_dsl::WeightRepr::Scaled {
            layout: model_dsl::ScaleLayout::PerGroup,
            group,
            zero_point: false,
            ..
        } => group as i64,
        _ => panic!(
            "kimi-k2's catalogued expert axis moved off per-group WNA16; \
             this stacker and int4b8_encoding both spell that axis"
        ),
    };
    const CODES_PER_WORD: i64 = 8;

    for layer in 0..b.shape().layers {
        let mlp = format!("model.layers.{layer}.mlp.");
        let mut gate_up = Vec::new();
        let mut gate_up_scales = Vec::new();
        let mut down = Vec::new();
        let mut down_scales = Vec::new();
        let mut local_inter = 0i64;
        let mut hidden = 0i64;

        let mut expert = 0u32;
        loop {
            let ep = format!("{mlp}experts.{expert}.");
            if b.find(&b.source_name(&format!("{ep}gate_proj.weight_packed")))
                .is_none()
            {
                break;
            }
            let names = [
                format!("{ep}gate_proj.weight_packed"),
                format!("{ep}gate_proj.weight_scale"),
                format!("{ep}up_proj.weight_packed"),
                format!("{ep}up_proj.weight_scale"),
                format!("{ep}down_proj.weight_packed"),
                format!("{ep}down_proj.weight_scale"),
            ];
            let mut parts = Vec::with_capacity(6);
            for name in &names {
                let Some(part) = b.find(&b.source_name(name)) else {
                    return fail(format!(
                        "kimi expert stack: {ep} is missing a weight or scale"
                    ));
                };
                parts.push(part);
            }

            if [0usize, 2, 4]
                .iter()
                .any(|&i| !is_raw(&parts[i].encoding, DType::I32))
            {
                return Ok(());
            }
            if [1usize, 3, 5]
                .iter()
                .any(|&i| !is_raw(&parts[i].encoding, DType::BF16))
            {
                return Ok(());
            }

            let up_raw = &parts[0].shape;
            let down_raw = &parts[4].shape;
            if up_raw.len() != 2 || down_raw.len() != 2 {
                return fail(format!(
                    "kimi expert stack: {ep} expects rank-2 expert weights"
                ));
            }
            let inter_full = up_raw[0];
            let h = up_raw[1] * CODES_PER_WORD;
            let inter = b.local_extent(inter_full);
            if h % GROUP != 0 || inter % GROUP != 0 {
                return fail(format!(
                    "kimi expert stack: {ep} expects both expert dims to be a multiple of 32"
                ));
            }
            if local_inter != 0 && (inter != local_inter || h != hidden) {
                return fail(format!(
                    "kimi expert stack: {ep} disagrees with its siblings on shape"
                ));
            }
            local_inter = inter;
            hidden = h;

            let packed = |b: &Builder<'_>, name: &str, shape: Vec<i64>, axis: u8| {
                b.shard(
                    Expr::src(name).transmute(model_loader::contract::TensorType::new(
                        shape.clone(),
                        int4b8_encoding(2),
                    )),
                    shape,
                    Some(axis),
                )
                .0
            };
            let factors = |b: &Builder<'_>, name: &str, shape: Vec<i64>, axis: u8| {
                b.shard(
                    Expr::src(name).transmute(model_loader::contract::TensorType::new(
                        shape.clone(),
                        Encoding::Raw(DType::BF16),
                    )),
                    shape,
                    Some(axis),
                )
                .0
            };

            let gate = packed(b, &parts[0].name, vec![1, inter_full, h], 1);
            let up = packed(b, &parts[2].name, vec![1, inter_full, h], 1);
            let gate_s = factors(b, &parts[1].name, vec![1, inter_full, h / GROUP], 1);
            let up_s = factors(b, &parts[3].name, vec![1, inter_full, h / GROUP], 1);

            gate_up.push(Expr::concat(1, vec![gate, up]));
            gate_up_scales.push(Expr::concat(1, vec![gate_s, up_s]));
            down.push(packed(
                b,
                &parts[4].name,
                vec![1, down_raw[0], inter_full],
                2,
            ));
            down_scales.push(factors(
                b,
                &parts[5].name,
                vec![1, down_raw[0], inter_full / GROUP],
                2,
            ));
            expert += 1;
        }

        if gate_up.is_empty() {

            if b.target().backend == BackendKind::Cuda
                && b.find(&b.source_name(&format!("{mlp}experts.0.gate_proj.weight")))
                    .is_some()
            {
                return fail(format!(
                    "kimi-k2's catalogued CUDA SKU ships `{}` routed experts \
                     (axis W2: `weight_packed` + `weight_scale`, group {GROUP}), \
                     but layer {layer} publishes a bf16 `experts.0.gate_proj.weight` \
                     — the same model in a repr this build's CUDA text has no \
                     routed leg for",
                    <super::forward::ShippedW2 as DtypeAxis>::NAME,
                ));
            }
            continue;
        }
        let experts = gate_up.len() as i64;

        if experts != i64::from(b.shape().n_experts) {
            return fail(format!(
                "kimi expert stack: layer {layer} stacked {experts} experts but the \
                 row states {}; the router emits indices this slab has no rows for",
                b.shape().n_experts
            ));
        }

        let slab_bytes = (experts as u64)
            * 3
            * (local_inter as u64)
            * (hidden as u64)
            * 2
            * u64::from(b.shape().layers);
        if slab_bytes > budget {
            return Ok(());
        }

        let gu_scale = format!("{mlp}experts.gate_up.scale");
        let dn_scale = format!("{mlp}experts.down.scale");
        let gu = b.define(
            gu_scale.clone(),
            Expr::concat(0, gate_up_scales),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, hidden / GROUP]),
        );
        b.mark_internal(gu);
        let dn = b.define(
            dn_scale.clone(),
            Expr::concat(0, down_scales),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, hidden, local_inter / GROUP]),
        );
        b.mark_internal(dn);
        b.define(
            format!("{mlp}experts.gate_up.weight"),
            Expr::concat(0, gate_up).scale_per_block(Expr::out(&gu_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, 2 * local_inter, hidden]),
        );
        b.define(
            format!("{mlp}experts.down.weight"),
            Expr::concat(0, down).scale_per_block(Expr::out(&dn_scale)),
            Encoding::Raw(DType::BF16),
            Some(vec![experts, hidden, local_inter]),
        );
    }
    Ok(())
}
