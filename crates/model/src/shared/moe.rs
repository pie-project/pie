//! The plain HF MoE source layout, stacked or streamed.
//!
//! Per expert `e`, `mlp.experts.{e}.gate_proj.weight` / `.up_proj.weight` as
//! `[I, H]` and `.down_proj.weight` as `[H, I]`. Not anything qwen-specific:
//! GLM-5.2 ships it too, which is why this lives in `common` and not a
//! generation's own crate. Ported from the tail of the CUDA driver's
//! `model/contract.hpp`.

use model_loader::contract::{Expr, GroupContract, TensorContract};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, TensorId};

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
    let num_experts = i64::from(b.shape().n_experts);
    if num_experts <= 0 {
        return Ok(());
    }
    for layer in 0..b.shape().layers {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::Policy;
    use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;
    use model_loader::types::FileId;

    const EXPERTS: i64 = 3;
    const INTER: i64 = 8;
    const HIDDEN: i64 = 4;

    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }

    fn tensor(id: u32, name: String, shape: Vec<i64>, encoding: Encoding) -> RawTensor {
        RawTensor {
            id: TensorId(id),
            name,
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 0,
            shape,
            encoding,
        }
    }

    /// One layer of per-expert MoE weights, as a checkpoint ships them.
    fn per_expert(experts: i64, encoding: Encoding) -> Vec<RawTensor> {
        let mut ck = Vec::new();
        let mut id = 1;
        for e in 0..experts {
            for (member, shape) in [
                ("gate_proj", vec![INTER, HIDDEN]),
                ("up_proj", vec![INTER, HIDDEN]),
                ("down_proj", vec![HIDDEN, INTER]),
            ] {
                ck.push(tensor(
                    id,
                    format!("model.layers.0.mlp.experts.{e}.{member}.weight"),
                    shape,
                    encoding.clone(),
                ));
                id += 1;
            }
        }
        ck
    }

    fn stack(
        tensors: Vec<RawTensor>,
        gate_second: bool,
        float_only: bool,
        tp_size: u32,
    ) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            tp_size,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "moe-test",
            LoadShape::mixture(1, 0, EXPERTS as u32, true),
            &encoding,
            &target,
            &policy,
        );
        hf_moe_expert_stacks(&mut b, gate_second, float_only)?;
        b.publish_remaining()?;
        b.finish()
    }

    fn refusal(result: Result<ModelContract, Error>) -> String {
        match result {
            Err(Error::Contract(message)) => message,
            Err(other) => panic!("expected a contract refusal, got {other:?}"),
            Ok(_) => panic!("expected a refusal, and the stack succeeded"),
        }
    }

    fn shaped(contract: &ModelContract, suffix: &str) -> Option<Vec<i64>> {
        contract
            .tensors
            .iter()
            .find(|t| t.name.ends_with(suffix))
            .and_then(|t| t.shape.clone())
    }

    /// The sources a declared tensor reads, in the order it reads them.
    ///
    /// Read off the serialized expression rather than restated, because the
    /// ORDER is the fact under test and an accessor that returned a set
    /// would agree with either arrangement.
    fn sources_in_order(contract: &ModelContract, suffix: &str) -> Vec<String> {
        let tensor = contract
            .tensors
            .iter()
            .find(|t| t.name.ends_with(suffix))
            .unwrap_or_else(|| panic!("no tensor ends with '{suffix}'"));
        let json = serde_json::to_value(&tensor.expr).expect("an expression serializes");
        let mut found = Vec::new();
        collect(&json, &mut found);
        found
    }

    /// Every `Src` leaf, in the order the expression names them.
    ///
    /// `Src` serializes as a bare string under its variant key rather than
    /// as an object with a `name` field, so a collector that looked for
    /// `"name"` found nothing and every ordering assertion held over an
    /// empty list.
    fn collect(value: &serde_json::Value, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, inner) in map {
                    match (key.as_str(), inner.as_str()) {
                        ("Src" | "SrcIndexed", Some(text)) => out.push(text.to_string()),
                        _ => collect(inner, out),
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    collect(item, out);
                }
            }
            _ => {}
        }
    }

    /// E per-expert slabs become the two 3-D tensors a fused MoE reads.
    #[test]
    fn per_expert_weights_are_stacked_into_the_fused_pair() {
        let contract = stack(per_expert(EXPERTS, bf16()), false, false, 1)
            .expect("a well-formed checkpoint stacks");
        assert_eq!(
            shaped(&contract, "mlp.experts.gate_up_proj"),
            Some(vec![EXPERTS, 2 * INTER, HIDDEN]),
            "the gate and up halves are joined and the experts stacked"
        );
        assert_eq!(
            shaped(&contract, "mlp.experts.down_proj"),
            Some(vec![EXPERTS, HIDDEN, INTER]),
            "and down is stacked as it stands"
        );
    }

    /// `gate_second` decides which half leads, and it is not cosmetic.
    ///
    /// The bound driver's activation reads one arrangement. Publishing the
    /// other produces a model that loads, runs, and computes the gate
    /// against the wrong half of its own tensor -- the failure this flag
    /// exists to make explicit rather than to leave to a driver-side block
    /// swap over the largest tensor in the model.
    #[test]
    fn gate_second_swaps_which_half_leads() {
        let gate_first = stack(per_expert(EXPERTS, bf16()), false, false, 1).expect("stacks");
        let up_first = stack(per_expert(EXPERTS, bf16()), true, false, 1).expect("stacks");

        let leading = |contract| {
            let sources = sources_in_order(contract, "mlp.experts.gate_up_proj");
            assert_eq!(
                sources.len(),
                2 * EXPERTS as usize,
                "the join reads a gate and an up for every expert"
            );
            sources
        };
        assert!(
            leading(&gate_first)[0].ends_with("0.gate_proj.weight"),
            "gate_second=false puts the gate first: {:?}",
            leading(&gate_first)[0]
        );
        assert!(
            leading(&up_first)[0].ends_with("0.up_proj.weight"),
            "gate_second=true puts the up first: {:?}",
            leading(&up_first)[0]
        );
    }

    /// An expert that is missing a third of itself is refused, by index.
    ///
    /// The refusal has to name the layer and the expert: a mixture has
    /// hundreds of these tensors and "missing gate/up/down" without an
    /// index is a message that sends someone to grep a checkpoint.
    #[test]
    fn an_incomplete_expert_is_refused_and_the_message_says_which() {
        for member in ["gate_proj", "up_proj", "down_proj"] {
            let mut ck = per_expert(EXPERTS, bf16());
            ck.retain(|raw| !raw.name.ends_with(&format!("experts.2.{member}.weight")));
            let message = refusal(stack(ck, false, false, 1));
            assert!(
                message.contains("layer 0") && message.contains("expert 2"),
                "a missing {member} locates the hole: {message}"
            );
        }
    }

    /// An expert whose slabs disagree in shape is refused rather than joined.
    ///
    /// The concatenation would otherwise produce a tensor whose declared
    /// extents do not match its own contents, which the loader discovers
    /// as a byte count and reports about the wrong thing.
    /// Stated for ALL THREE members, because the check is three clauses.
    ///
    /// `shapes_ok` ands together a test of the gate, the up and the down.
    /// A single fixture only exercises the clause it damages, so dropping
    /// either of the other two is invisible -- which is exactly what a
    /// control showed when this damaged only the gate.
    #[test]
    fn an_expert_whose_shape_differs_from_the_first_is_refused() {
        for member in ["gate_proj", "up_proj", "down_proj"] {
            let mut ck = per_expert(EXPERTS, bf16());
            for raw in &mut ck {
                if raw.name.ends_with(&format!("experts.1.{member}.weight")) {
                    raw.shape = vec![INTER + 1, HIDDEN + 1];
                }
            }
            let message = refusal(stack(ck, false, false, 1));
            assert!(
                message.contains("shape mismatch") && message.contains("expert 1"),
                "a wrong {member}: {message}"
            );
        }
    }

    /// Experts of two different dtypes are refused rather than silently one.
    ///
    /// The stack declares ONE encoding for all E slabs, taken from expert
    /// zero. An expert that differs would be read at the wrong width --
    /// every byte after it misaligned.
    /// Stated for all three members, for the reason above.
    #[test]
    fn an_expert_of_another_dtype_is_refused() {
        for member in ["gate_proj", "up_proj", "down_proj"] {
            let mut ck = per_expert(EXPERTS, bf16());
            for raw in &mut ck {
                if raw.name.ends_with(&format!("experts.1.{member}.weight")) {
                    raw.encoding = Encoding::Raw(DType::F32);
                }
            }
            let message = refusal(stack(ck, false, false, 1));
            assert!(
                message.contains("dtype mismatch") && message.contains("expert 1"),
                "a differing {member}: {message}"
            );
        }
    }

    /// A rank-1 expert weight is refused before anything is derived from it.
    ///
    /// `inter` and `hidden` are read off expert zero's two extents, so a
    /// tensor with one extent would index past its own shape.
    #[test]
    fn an_expert_weight_that_is_not_a_matrix_is_refused() {
        let mut ck = per_expert(EXPERTS, bf16());
        for raw in &mut ck {
            if raw.name.ends_with("experts.0.gate_proj.weight") {
                raw.shape = vec![INTER];
            }
        }
        let message = refusal(stack(ck, false, false, 1));
        assert!(message.contains("expected 2-D"), "{message}");
    }

    /// Under TP the stack declines, and the bind fails loudly instead.
    ///
    /// The join makes a new leading axis and nothing downstream slices it
    /// per rank. Declining leaves the fused tensor undeclared, so the bind
    /// path fails on a name it cannot find rather than loading a tensor
    /// that holds every rank's experts.
    #[test]
    fn a_multi_rank_target_leaves_the_stack_unfused() {
        let contract = stack(per_expert(EXPERTS, bf16()), false, false, 2).expect("no refusal");
        assert_eq!(
            shaped(&contract, "mlp.experts.gate_up_proj"),
            None,
            "nothing was fused, and the missing name is the signal"
        );
    }

    /// `float_only` leaves a quantised layer on its per-expert path.
    ///
    /// A quantised expert carries companion scale tensors this stack does
    /// not join, so folding the weights alone would orphan them -- the
    /// scales would be published as ordinary tensors describing a layout
    /// that no longer exists.
    #[test]
    fn float_only_declines_a_quantised_layer_rather_than_orphaning_its_scales() {
        let quantised = Encoding::Raw(DType::F8E4M3);
        let declined =
            stack(per_expert(EXPERTS, quantised.clone()), false, true, 1).expect("no refusal");
        assert_eq!(
            shaped(&declined, "mlp.experts.gate_up_proj"),
            None,
            "float_only=true leaves the quantised layer alone"
        );

        let taken = stack(per_expert(EXPERTS, quantised), false, false, 1).expect("no refusal");
        assert_eq!(
            shaped(&taken, "mlp.experts.gate_up_proj"),
            Some(vec![EXPERTS, 2 * INTER, HIDDEN]),
            "and float_only=false is what makes the skip a CHOICE rather \
             than the only behaviour"
        );
    }

    /// A checkpoint that already ships the fused tensors is left alone.
    #[test]
    fn a_pre_fused_checkpoint_is_not_stacked_again() {
        let mut ck = per_expert(EXPERTS, bf16());
        ck.push(tensor(
            900,
            "model.layers.0.mlp.experts.gate_up_proj".to_string(),
            vec![EXPERTS, 2 * INTER, HIDDEN],
            bf16(),
        ));
        let contract = stack(ck, false, false, 1).expect("no refusal");
        assert_eq!(
            sources_in_order(&contract, "mlp.experts.gate_up_proj"),
            vec!["model.layers.0.mlp.experts.gate_up_proj".to_string()],
            "the shipped tensor is published as itself, not rebuilt from \
             the per-expert slabs beside it"
        );
    }

    /// A mixture with no experts declared does nothing at all.
    #[test]
    fn a_dense_shape_runs_no_expert_pass() {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: per_expert(EXPERTS, bf16()),
        };
        let target = StorageTarget::default();
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "moe-test",
            LoadShape::dense(1, 0, true),
            &encoding,
            &target,
            &policy,
        );
        hf_moe_expert_stacks(&mut b, false, false).expect("a dense shape is not an error");
        let contract = b.publish_remaining().and_then(|()| b.finish()).expect("ok");
        assert_eq!(
            shaped(&contract, "mlp.experts.gate_up_proj"),
            None,
            "no expert count, no stack"
        );
    }
}
