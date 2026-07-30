//! Family declarations.
//!
//! Each function here is a forward pass written as ordinary Rust over a
//! [`TraceBuilder`]; running it *is* the trace. Branches on facts execute
//! now and vanish — a deployment that binds no fused QKV traces three
//! matmuls and no split, and the traced forms differ the way two compiled
//! programs differ, not the way two runtime paths do.

use crate::facts::{LlamaLikeFacts, NormPlacement, QkNorm};
use crate::trace::{ForwardPlan, TraceBuilder};

/// The llama_like decode/prefill body (no structural divergence, so one
/// trace serves both; the emitter picks decode vs prefill attention plans
/// per fire, which is backend knowledge the trace deliberately lacks).
///
/// Mirrors `driver/cuda/src/model/llama_like/llama_like.cpp`
/// (`llama_like_forward_paged`) op for op; the golden test pins that
/// correspondence and the comment there maps each op to the kernel(s) the
/// hand-written pass would launch.
///
/// Norm placement branches the block structure itself (the first fact to
/// do so):
///
/// * `Pre` — norm the stream into the sub-layer, accumulate the output
///   projection straight back (`matmul_add`, the `beta=1` GEMM).
/// * `Post` (olmo2) — the sub-layer reads the stream raw, its output
///   projection lands in scratch (`beta=0`), the norm applies to THAT, and
///   a separate `ResidualAdd` lands it — the hand-written post-norm walk's
///   gemm → `launch_rmsnorm_bf16` → `launch_residual_add_bf16` triplet.
pub fn llama_like(facts: &LlamaLikeFacts) -> ForwardPlan {
    let mut t = TraceBuilder::new("llama_like");
    let q_w = facts.q_width();
    let kv_w = facts.kv_width();
    let post_norm = facts.norm_placement == NormPlacement::Post;

    let mut y = t.embed("embed", facts.hidden);

    for l in 0..facts.layers {
        y = t.layer(l, |t| {
            let w = |name: &str| format!("layer.{l}.{name}");

            // Attention block: (pre-norm) -> qkv -> (q/k norms) -> rope ->
            // append -> attention -> o_proj landed on the residual.
            let x = if post_norm {
                y
            } else {
                t.rmsnorm(y, &w("attn_norm"), facts.norm_variant)
            };
            let (q, k, v) = if facts.fused_qkv {
                let packed = t.matmul(x, &w("qkv"), q_w + 2 * kv_w);
                t.split_qkv(packed, q_w, kv_w)
            } else {
                (
                    t.matmul(x, &w("q_proj"), q_w),
                    t.matmul(x, &w("k_proj"), kv_w),
                    t.matmul(x, &w("v_proj"), kv_w),
                )
            };
            let (q, k) = match facts.qk_norm {
                QkNorm::Off => (q, k),
                QkNorm::PerHead => (
                    t.rmsnorm_per_head(q, &w("q_norm"), facts.head_dim),
                    t.rmsnorm_per_head(k, &w("k_norm"), facts.head_dim),
                ),
                // The global convention IS a plain row RMSNorm over the
                // flattened `[rows, heads * head_dim]` projection — the
                // same op (and kernel) as the block norms, applied to q/k.
                QkNorm::Global => (
                    t.rmsnorm(q, &w("q_norm"), facts.norm_variant),
                    t.rmsnorm(k, &w("k_norm"), facts.norm_variant),
                ),
            };
            let (q, k) = t.rope(q, k, facts.rope);
            t.kv_append(l, k, v);
            let attn = t.attention(l, q, q_w);
            let y_attn = if post_norm {
                // Post-norm: o_proj to scratch, norm the OUTPUT, then the
                // separate residual add — norm placement is an op-order
                // fact, so the trace states all three.
                let o = t.matmul(attn, &w("o_proj"), facts.hidden);
                let o = t.rmsnorm(o, &w("attn_norm"), facts.norm_variant);
                t.residual_add(o, y)
            } else {
                t.matmul_add(attn, &w("o_proj"), y, facts.hidden)
            };

            // MLP block: (pre-norm) -> gate‖up -> swiglu -> down landed.
            let m = if post_norm {
                y_attn
            } else {
                t.rmsnorm(y_attn, &w("mlp_norm"), facts.norm_variant)
            };
            let packed = t.matmul(m, &w("gate_up"), 2 * facts.intermediate);
            let act = t.swiglu(packed, facts.intermediate);
            if post_norm {
                let d = t.matmul(act, &w("down"), facts.hidden);
                let d = t.rmsnorm(d, &w("mlp_norm"), facts.norm_variant);
                t.residual_add(d, y_attn)
            } else {
                t.matmul_add(act, &w("down"), y_attn, facts.hidden)
            }
        });
    }

    let final_norm = t.rmsnorm(y, "final_norm", facts.norm_variant);
    let lm_head = if facts.tied_embeddings { "embed" } else { "lm_head" };
    t.lm_head(final_norm, lm_head, facts.vocab);
    t.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{Dim, OpKind};

    /// The traced form of one qwen3 layer, mapped op-by-op to the kernel
    /// sequence `llama_like_forward_paged` launches on the unfused path.
    /// (The fused decode QKV kernel covers Matmul+SplitQkv+RmsnormPerHead
    /// x2+Rope+KvAppend — an emitter peephole over exactly this adjacency;
    /// see stage1-notes.md for why the trace must stay unfused.)
    ///
    /// | trace op            | hand-written kernel(s)                          |
    /// |---------------------|-------------------------------------------------|
    /// | Rmsnorm(attn_norm)  | launch_rmsnorm_bf16                              |
    /// | Matmul(qkv)         | ops::gemm_act_x_w (qkv_proj_fused)               |
    /// | SplitQkv            | launch_split_qkv_bf16                            |
    /// | RmsnormPerHead x2 + Rope | launch_qk_rmsnorm_rope_bf16 (fused pair)    |
    /// | KvAppend            | launch_write_kv_to_pages                         |
    /// | Attention           | dispatch_attention_flashinfer_{decode,prefill}   |
    /// | Matmul(o_proj)+res  | ops::gemm_act_x_w beta=1                         |
    /// | Rmsnorm(mlp_norm)   | launch_rmsnorm_bf16                              |
    /// | Matmul(gate_up)     | ops::gemm_act_x_w                                |
    /// | Swiglu              | (silu-and-mul kernel)                            |
    /// | Matmul(down)+res    | ops::gemm_act_x_w beta=1                         |
    #[test]
    fn qwen3_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::qwen3_0_6b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { beta_one: false, .. } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",
                "matmul",
                "split_qkv",
                "rmsnorm_per_head",
                "rmsnorm_per_head",
                "rope",
                "kv_append",
                "attention",
                "matmul+res",
                "rmsnorm",
                "matmul",
                "swiglu",
                "matmul+res",
            ]
        );
    }

    #[test]
    fn qwen3_full_plan_shape() {
        let facts = LlamaLikeFacts::qwen3_0_6b();
        let plan = llama_like(&facts);
        // 13 ops per layer + embed + final norm + lm_head.
        assert_eq!(plan.ops.len(), 13 * facts.layers as usize + 3);
        // Weight tying: the lm head names the embedding table.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "embed"
        ));
        // Logits are per-request f32 over the vocab.
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
    }

    #[test]
    fn unfused_binding_traces_three_matmuls() {
        let facts = LlamaLikeFacts {
            fused_qkv: false,
            ..LlamaLikeFacts::qwen3_0_6b()
        };
        let plan = llama_like(&facts);
        let layer0: Vec<_> = plan.layer_ops(0).collect();
        let matmuls = layer0
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Matmul { .. }))
            .count();
        // q, k, v, o_proj, gate_up, down — and no SplitQkv anywhere.
        assert_eq!(matmuls, 6);
        assert!(
            !layer0
                .iter()
                .any(|op| matches!(op.kind, OpKind::SplitQkv { .. }))
        );
    }

    /// Phi-3-mini's traced form: the qk-norm branch folds away (no
    /// RmsnormPerHead anywhere) so Rope follows the projections directly
    /// — the hand-written path's `apply_rope` with no `fuse_qk_norm_rope`
    /// kernel in sight — and the unfused binding (the dense join cannot
    /// re-fuse the contract-split q/k/v bands) traces three projection
    /// matmuls and no SplitQkv.
    #[test]
    fn phi3_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::phi3_mini());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { beta_one: false, .. } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",
                "matmul",
                "matmul",
                "matmul",
                "rope",
                "kv_append",
                "attention",
                "matmul+res",
                "rmsnorm",
                "matmul",
                "swiglu",
                "matmul+res",
            ]
        );
    }

    #[test]
    fn phi3_full_plan_shape() {
        let facts = LlamaLikeFacts::phi3_mini();
        let plan = llama_like(&facts);
        // 12 ops per layer (13 minus the two per-head norms and the
        // SplitQkv, plus the two extra projection matmuls) + embed +
        // final norm + lm_head.
        assert_eq!(plan.ops.len(), 12 * facts.layers as usize + 3);
        // Untied embeddings: the lm head names its own weight, not the
        // embedding table.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "lm_head"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
    }

    /// Mistral-7B-v0.3's traced form: the fused-QKV binding keeps
    /// Matmul(qkv) + SplitQkv, but with no qk-norm the RmsnormPerHead pair
    /// between SplitQkv and Rope folds away — the one branch combination
    /// neither qwen3 (fused + qk-norm) nor phi3 (unfused + no qk-norm) had
    /// run. On this shape the executor's fused decode-QKV peephole can
    /// never fire (its predicate requires qk-norm), so SplitQkv and Rope
    /// launch as the standalone kernels.
    #[test]
    fn mistral_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::mistral_7b_v03());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { beta_one: false, .. } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",
                "matmul",
                "split_qkv",
                "rope",
                "kv_append",
                "attention",
                "matmul+res",
                "rmsnorm",
                "matmul",
                "swiglu",
                "matmul+res",
            ]
        );
    }

    #[test]
    fn mistral_full_plan_shape() {
        let facts = LlamaLikeFacts::mistral_7b_v03();
        let plan = llama_like(&facts);
        // 11 ops per layer (13 minus the two per-head norms) + embed +
        // final norm + lm_head.
        assert_eq!(plan.ops.len(), 11 * facts.layers as usize + 3);
        // Untied embeddings: the lm head names its own weight.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "lm_head"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
    }

    /// OLMo-2-1B's traced form: the post-norm walk. No pre-norm before the
    /// projections — QKV reads the residual stream raw — and each
    /// sub-layer ends with the matmul(beta=0) → rmsnorm → residual_add
    /// triplet instead of one accumulate GEMM. The global qk-norm traces
    /// as plain row Rmsnorm on q and k (weight `[heads * head_dim]`, the
    /// hand-written `rmsnorm_qk` global branch), so no RmsnormPerHead
    /// appears and neither fused peephole (qk-norm+rope, decode-QKV) can
    /// ever fire — both predicates require the per-head convention.
    #[test]
    fn olmo2_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::olmo2_1b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { beta_one: false, .. } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                OpKind::ResidualAdd => "residual_add",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "matmul",        // q_proj — reads y raw: no attn pre-norm
                "matmul",        // k_proj
                "matmul",        // v_proj
                "rmsnorm",       // q_norm (global: row norm over [N, Hq])
                "rmsnorm",       // k_norm
                "rope",
                "kv_append",
                "attention",
                "matmul",        // o_proj, beta=0 — scratch, not the stream
                "rmsnorm",       // attn_norm on the o_proj OUTPUT
                "residual_add",  // y += norm(o_proj(attn))
                "matmul",        // gate_up — reads y raw: no mlp pre-norm
                "swiglu",
                "matmul",        // down, beta=0
                "rmsnorm",       // mlp_norm on the down OUTPUT
                "residual_add",  // y += norm(down(act))
            ]
        );
    }

    #[test]
    fn olmo2_full_plan_shape() {
        let facts = LlamaLikeFacts::olmo2_1b();
        let plan = llama_like(&facts);
        // 16 ops per layer + embed + final norm + lm_head.
        assert_eq!(plan.ops.len(), 16 * facts.layers as usize + 3);
        // Untied embeddings: the lm head names its own weight.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "lm_head"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
        // No RmsnormPerHead anywhere: the global convention is a plain
        // Rmsnorm, and mistaking one for the other is different arithmetic.
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::RmsnormPerHead { .. }))
        );
    }

    /// The global qk-norm's traced Rmsnorm ops carry the q/k projection
    /// shapes (`[Tokens, heads * head_dim]`) — one norm over the flattened
    /// heads, not `heads` norms of `head_dim` — and name the q/k norm
    /// weights.
    #[test]
    fn olmo2_global_qk_norm_is_row_rmsnorm_over_projection_width() {
        let facts = LlamaLikeFacts::olmo2_1b();
        let plan = llama_like(&facts);
        let qk_norms: Vec<_> = plan
            .layer_ops(0)
            .filter(|op| {
                matches!(&op.kind, OpKind::Rmsnorm { weight, .. }
                    if weight.ends_with("q_norm") || weight.ends_with("k_norm"))
            })
            .collect();
        assert_eq!(qk_norms.len(), 2);
        for (op, width) in qk_norms.iter().zip([facts.q_width(), facts.kv_width()]) {
            assert_eq!(
                plan.values[op.outputs[0] as usize].shape.0,
                vec![Dim::Tokens, Dim::Const(width)]
            );
        }
    }

    /// Post-norm residual dataflow: every ResidualAdd consumes the normed
    /// sub-layer output AND the residual stream it lands on, in that order
    /// (the matmul_add convention), and its input really is the Rmsnorm's
    /// output — the norm sits BETWEEN the projection and the add.
    #[test]
    fn olmo2_post_norm_residual_dataflow() {
        let plan = llama_like(&LlamaLikeFacts::olmo2_1b());
        let layer0: Vec<_> = plan.layer_ops(0).collect();
        let adds: Vec<_> = layer0
            .iter()
            .filter(|op| matches!(op.kind, OpKind::ResidualAdd))
            .collect();
        assert_eq!(adds.len(), 2);
        for add in adds {
            assert_eq!(add.inputs.len(), 2, "residual missing on {add:?}");
            let normed = add.inputs[0];
            let norm_op = layer0
                .iter()
                .find(|op| op.outputs.contains(&normed))
                .expect("producer of the add's first operand");
            assert!(
                matches!(&norm_op.kind, OpKind::Rmsnorm { weight, .. }
                    if weight.ends_with("attn_norm") || weight.ends_with("mlp_norm")),
                "post-norm add must consume a block-norm output, got {norm_op:?}"
            );
        }
        // And no beta=1 accumulate anywhere: the residual fold is illegal
        // when a norm sits between the GEMM and the stream.
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::Matmul { beta_one: true, .. }))
        );
    }

    #[test]
    fn residual_dataflow_is_recorded() {
        let plan = llama_like(&LlamaLikeFacts::qwen3_0_6b());
        // Every accumulate consumes two values: the projection input and
        // the residual it adds into.
        for op in &plan.ops {
            if let OpKind::Matmul { beta_one: true, .. } = op.kind {
                assert_eq!(op.inputs.len(), 2, "residual missing on {op:?}");
            }
        }
    }

    /// The traced form is a stable artifact: serialize one layer and pin
    /// it. A representation change must show up as a reviewed diff here,
    /// the same discipline the loader applies to its golden plans.
    #[test]
    fn traced_form_round_trips() {
        let plan = llama_like(&LlamaLikeFacts::qwen3_0_6b());
        let json = serde_json::to_string(&plan).unwrap();
        let back: ForwardPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }
}
