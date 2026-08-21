use model_loader::contract::{Expr, GroupContract, TensorContract};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, TensorId};

use super::builder::{Builder, logical_dtype};
use super::probe;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub fn hf_moe_expert_stacks(
    b: &mut Builder<'_>,
    gate_second: bool,
    float_only: bool,
) -> Result<(), Error> {
    let num_experts = i64::from(b.shape().n_experts);
    if num_experts <= 0 {
        return Ok(());
    }
    for layer in 0..b.shape().layers {

        let bound = format!("model.layers.{layer}.mlp.experts.");
        let prefix = b.source_name(&bound);
        if b.find(&format!("{prefix}gate_up_proj")).is_some() {
            continue;
        }
        let Some(gate0) = b.find(&format!("{prefix}0.gate_proj.weight")) else {
            continue;
        };
        if gate0.shape.len() != 2 {
            return fail(format!("moe expert stack: '{}' expected 2-D", gate0.name));
        }
        let inter = gate0.shape[0];
        let hidden = gate0.shape[1];
        let dtype = logical_dtype(&gate0.encoding);
        if float_only && dtype != DType::BF16 && dtype != DType::F16 {
            continue;
        }
        if !probe::is_dense_addressable(&gate0.encoding) {
            return fail(format!(
                "moe expert stack: '{}' has a non-affine packed encoding",
                gate0.name
            ));
        }

        if b.stream_routed_experts() {
            hf_moe_streamed_expert_groups(
                b,
                gate_second,
                layer,
                &bound,
                &prefix,
                num_experts,
                inter,
                hidden,
                dtype,
            )?;
            continue;
        }
        if b.target().tp_size != 1 {

            continue;
        }

        let mut gate_up_parts = Vec::with_capacity(num_experts as usize);
        let mut down_parts = Vec::with_capacity(num_experts as usize);
        let mut consumed: Vec<TensorId> = Vec::new();
        for e in 0..num_experts {
            let tag = format!("{prefix}{e}.");
            let (Some(g), Some(u), Some(d)) = (
                b.find(&format!("{tag}gate_proj.weight")),
                b.find(&format!("{tag}up_proj.weight")),
                b.find(&format!("{tag}down_proj.weight")),
            ) else {
                return fail(format!(
                    "moe expert stack: layer {layer} expert {e} missing gate/up/down"
                ));
            };
            let shapes_ok = g.shape.len() == 2
                && u.shape.len() == 2
                && d.shape.len() == 2
                && g.shape == [inter, hidden]
                && u.shape == [inter, hidden]
                && d.shape == [hidden, inter];
            if !shapes_ok {
                return fail(format!(
                    "moe expert stack: layer {layer} expert {e} shape mismatch"
                ));
            }
            if logical_dtype(&g.encoding) != dtype
                || logical_dtype(&u.encoding) != dtype
                || logical_dtype(&d.encoding) != dtype
            {
                return fail(format!(
                    "moe expert stack: layer {layer} expert {e} dtype mismatch"
                ));
            }

            let gate_src = Expr::src(&g.name);
            let up_src = Expr::src(&u.name);
            let halves = if gate_second {
                vec![up_src, gate_src]
            } else {
                vec![gate_src, up_src]
            };
            gate_up_parts.push(Expr::concat(0, halves).transmute(
                model_loader::contract::TensorType::new(
                    vec![1, 2 * inter, hidden],
                    Encoding::Raw(dtype),
                ),
            ));
            down_parts.push(
                Expr::src(&d.name).transmute(model_loader::contract::TensorType::new(
                    vec![1, hidden, inter],
                    Encoding::Raw(dtype),
                )),
            );
            consumed.extend([g.id, u.id, d.id]);
        }

        b.define(
            format!("{bound}gate_up_proj"),
            Expr::concat(0, gate_up_parts),
            Encoding::Raw(dtype),
            Some(vec![num_experts, 2 * inter, hidden]),
        );
        b.define(
            format!("{bound}down_proj"),
            Expr::concat(0, down_parts),
            Encoding::Raw(dtype),
            Some(vec![num_experts, hidden, inter]),
        );
        for id in consumed {
            b.consume(id);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hf_moe_streamed_expert_groups(
    b: &mut Builder<'_>,
    gate_second: bool,
    layer: u32,
    bound: &str,
    prefix: &str,
    num_experts: i64,
    inter: i64,
    hidden: i64,
    dtype: DType,
) -> Result<(), Error> {

    let mut consumed: Vec<TensorId> = Vec::new();
    for e in 0..num_experts {
        let tag = format!("{prefix}{e}.");
        for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            let Some(part) = b.find(&format!("{tag}{suffix}")) else {
                return fail(format!(
                    "moe expert group: layer {layer} expert {e} missing gate/up/down"
                ));
            };
            consumed.push(part.id);
        }
    }

    let gate_src = b.split(
        Expr::src_indexed(format!("{prefix}{{}}.gate_proj.weight")),
        0,
    );
    let up_src = b.split(Expr::src_indexed(format!("{prefix}{{}}.up_proj.weight")), 0);

    let local_inter = b.local_extent(inter);
    let halves = if gate_second {
        vec![up_src, gate_src]
    } else {
        vec![gate_src, up_src]
    };

    let group = GroupContract {
        name: bound[..bound.len() - 1].to_string(),
        arity: num_experts as u32,
        tensors: vec![
            TensorContract::new(
                "gate_up_proj",
                Expr::concat(0, halves),
                vec![2 * local_inter, hidden],
                Encoding::Raw(dtype),
            ),
            TensorContract::new(
                "down_proj",
                b.split(
                    Expr::src_indexed(format!("{prefix}{{}}.down_proj.weight")),
                    1,
                ),
                vec![hidden, local_inter],
                Encoding::Raw(dtype),
            ),
        ],
    };
    b.push_group(group);

    for id in consumed {
        b.consume(id);
    }
    Ok(())
}
