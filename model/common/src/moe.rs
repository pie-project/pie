//! The plain HF MoE source layout, stacked or streamed.
//!
//! Per expert `e`, `mlp.experts.{e}.gate_proj.weight` / `.up_proj.weight` as
//! `[I, H]` and `.down_proj.weight` as `[H, I]`. Not anything qwen-specific:
//! GLM-5.2 ships it too, which is why this lives in `common` and not a
//! generation's own crate. Ported from the tail of the CUDA driver's
//! `model/contract.hpp`.

use pie_loader::contract::{Expr, GroupContract, TensorContract};
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding, TensorId};

use super::builder::{Builder, logical_dtype};
use super::probe;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// Stack per-expert MoE weights into the fused 3-D tensors a fused-MoE
/// forward consumes:
///
/// - `mlp.experts.gate_up_proj` → `[E, 2I, H]`; `gate_second` selects which
///   half leads, matching what the bound driver's activation reads;
/// - `mlp.experts.down_proj` → `[E, H, I]`.
///
/// Built as an expression over the sources, so no GPU-side duplicate exists.
/// A no-op when the checkpoint already ships the fused tensors.
///
/// `float_only` skips layers whose experts are quantised. A quantised expert
/// carries companion scale tensors that this stack does not join, so folding
/// the weights alone would orphan them; families that can ship either layout
/// set it and fall back to their per-expert path for quantised checkpoints.
pub fn hf_moe_expert_stacks(
    b: &mut Builder<'_>,
    gate_second: bool,
    float_only: bool,
) -> Result<(), Error> {
    let num_experts = i64::from(b.facts().num_experts);
    if num_experts <= 0 {
        return Ok(());
    }
    for layer in 0..b.facts().num_hidden_layers {
        // `bound` is the name the bind path uses; `prefix` is where the
        // source tensors actually live, which is the same thing unless the
        // family declared a `source_prefix`.
        let bound = format!("model.layers.{layer}.mlp.experts.");
        let prefix = b.source_name(&bound);
        if b.find(&format!("{prefix}gate_up_proj")).is_some() {
            continue; // already pre-fused; the direct and TP-slice paths take it.
        }
        let Some(gate0) = b.find(&format!("{prefix}0.gate_proj.weight")) else {
            continue; // not a per-expert checkpoint at this layer.
        };
        if gate0.shape.len() != 2 {
            return fail(format!("moe expert stack: '{}' expected 2-D", gate0.name));
        }
        let inter = gate0.shape[0];
        let hidden = gate0.shape[1];
        let dtype = logical_dtype(&gate0.encoding);
        if float_only && dtype != DType::BF16 && dtype != DType::F16 {
            continue; // quantised experts keep the per-expert layout.
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
            // The stack joins E per-expert slabs along a new leading axis,
            // and nothing downstream slices that join per rank; the bind then
            // fails loudly on the missing fused tensor rather than loading
            // silently wrong. The group above has no such join — one instance
            // is one expert — so it shards each half directly and does run
            // under TP.
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
            // One expert slab is gate over up; the leading 1 makes the
            // per-expert concatenation a stack.
            let gate_src = Expr::src(&g.name);
            let up_src = Expr::src(&u.name);
            let halves = if gate_second {
                vec![up_src, gate_src]
            } else {
                vec![gate_src, up_src]
            };
            gate_up_parts.push(Expr::concat(0, halves).transmute(
                pie_loader::contract::TensorType::new(
                    vec![1, 2 * inter, hidden],
                    Encoding::Raw(dtype),
                ),
            ));
            down_parts.push(
                Expr::src(&d.name).transmute(pie_loader::contract::TensorType::new(
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

/// The same experts, declared as a group instead of a stack.
///
/// Structurally this is [`hf_moe_expert_stacks`] with the outer concatenation
/// removed: what is left after dropping the join is one instance, and the
/// instance is the group. `src(".../experts.3.gate_proj.weight")` becomes
/// `src_indexed(".../experts.{}.gate_proj.weight")` and nothing else moves.
///
/// The declared names carry no expert index. There is one plan and it is the
/// same plan for every expert, which is the whole claim a group makes; the
/// driver picks the instance at page-in time. One group per layer, because a
/// name template holds a single `{}`.
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
    // Claim every tensor the template will read. A group reads through a
    // template, so no node names these and `publish_remaining` would
    // otherwise publish all E of them as ordinary resident tensors.
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
    // Rank 2, not 3: with no expert axis to stack along there is nothing to
    // give the leading 1 to.
    //
    // Under TP each half is sharded before the join, never after: a shard of
    // `[gate; up]` on the fused axis would hand rank 0 the whole gate and
    // rank 1 the whole up. `down_proj` is the matching row-parallel half,
    // split on the intermediate axis it contracts over.
    let local_inter = b.local_extent(inter);
    let halves = if gate_second {
        vec![up_src, gate_src]
    } else {
        vec![gate_src, up_src]
    };
    // Named the way the bound tensors beside it are named, minus the trailing
    // dot: the driver resolves a group next to the weights it binds.
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
