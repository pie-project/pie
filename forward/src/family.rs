//! Family declarations.
//!
//! Each function here is a forward pass written as ordinary Rust over a
//! [`TraceBuilder`]; running it *is* the trace. Branches on facts execute
//! now and vanish — a deployment that binds no fused QKV traces three
//! matmuls and no split, and the traced forms differ the way two compiled
//! programs differ, not the way two runtime paths do.

use crate::facts::{LlamaLikeFacts, NormPlacement, QkNorm, Qwen35MoeMlpFacts};
use crate::trace::{DType, Dim, ForwardPlan, Shape, TraceBuilder};

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

/// One qwen3_5_moe MoE MLP block, traced standalone — the first `dyn`
/// fragment.
///
/// This is a FRAGMENT, not a model: the unit the future qwen3_5 declaration
/// composes per layer (`y += moe_mlp(l, rmsnorm(y, mlp_norm))`), traced
/// against layer 0 with the residual stream as a fragment parameter
/// ([`TraceBuilder::input`]). A full qwen3_5 declaration also needs the
/// hybrid GDN attention vocabulary (`causal_conv1d`, `gated_delta`, gated
/// rmsnorm, per-request recurrent state) — out of scope here, a separate
/// rung; see [`Qwen35MoeMlpFacts`].
///
/// Mirrors `qwen3_5_moe_forward.cpp::run_moe_mlp` launch for launch, in the
/// decode fast path's one-launch-per-op form (the canonical granularity;
/// the prefill path's host-routed per-expert gather/GEMM/scatter loop is a
/// LOWERING of ops 4–7, as is the CUTLASS fused pipeline — both the
/// emitter's per-fire choice):
///
/// | trace op                       | hand-written kernel(s)                      |
/// |--------------------------------|---------------------------------------------|
/// | Rmsnorm(mlp_norm)              | launch_rmsnorm_gemma_bf16                   |
/// | Matmul(router)                 | ops::gemm_act_x_wt_bf16 (router logits)     |
/// | TopK                           | launch_topk_softmax_bf16                    |
/// | Matmul(expert.{e}.gate_up, sel)| grouped GEMM (batched/aligned/CUTLASS)      |
/// | Swiglu                         | launch_chunked_swiglu_bf16 over N*k rows    |
/// | Matmul(expert.{e}.down, sel)   | grouped GEMM (batched/aligned/CUTLASS)      |
/// | WeightedSum                    | launch_token_batched_weighted_sum_bf16      |
/// | Matmul(shared_expert.gate_up)  | ops::gemm_act_x_w                           |
/// | Swiglu                         | launch_chunked_swiglu_bf16                  |
/// | Matmul(shared_expert.down)     | ops::gemm_act_x_w                           |
/// | Matmul(shared_expert_gate)     | ops::gemm_act_x_w ([Tokens, 1] logit)       |
/// | SigmoidGateAdd                 | launch_sigmoid_scalar_gate_add_bf16         |
/// | ResidualAdd                    | launch_residual_add_bf16                    |
///
/// The five shared-expert ops fold away when the facts say the checkpoint
/// has none (`shared_expert_intermediate == 0`, the qwen3_moe shape), the
/// same way llama_like's branches fold: at trace time, leaving no trace.
pub fn qwen3_5_moe_mlp_block(facts: &Qwen35MoeMlpFacts) -> ForwardPlan {
    let mut t = TraceBuilder::new("qwen3_5_moe_mlp_block");
    let hidden = Dim::Const(facts.hidden);

    // The fragment's parameter: the residual stream entering the block.
    let y = t.input(Shape(vec![Dim::Tokens, hidden]), DType::BF16);

    t.layer(0, |t| {
        let l = 0;
        let w = |name: &str| format!("layer.{l}.{name}");

        let m = t.rmsnorm(y, &w("mlp_norm"), facts.norm_variant);

        // Routed experts: router -> topk -> grouped gate_up -> swiglu ->
        // grouped down -> per-token weighted combine.
        let logits = t.matmul(m, &w("router"), facts.num_experts);
        let (experts, weights) = t.topk(logits, facts.top_k);
        let gate_up = t.matmul_per_token(
            m,
            &w("expert.{e}.gate_up"),
            experts,
            2 * facts.moe_intermediate,
        );
        let act = t.swiglu(gate_up, facts.moe_intermediate);
        let down = t.matmul_per_token(act, &w("expert.{e}.down"), experts, facts.hidden);
        let routed = t.weighted_sum(weights, down);

        // Shared expert (qwen3.5/3.6-MoE: always-on dense MLP behind a
        // per-token sigmoid scalar gate; absent on qwen3_moe).
        let combined = if facts.shared_expert_intermediate > 0 {
            let inter = facts.shared_expert_intermediate;
            let packed = t.matmul(m, &w("shared_expert.gate_up"), 2 * inter);
            let act = t.swiglu(packed, inter);
            let shared = t.matmul(act, &w("shared_expert.down"), facts.hidden);
            let gate = t.matmul(m, &w("shared_expert_gate"), 1);
            t.sigmoid_gate_add(shared, gate, routed)
        } else {
            routed
        };

        t.residual_add(combined, y)
    });
    t.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{Dim, NormVariant, OpKind};

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

    /// The MoE block fragment's op sequence, mapped launch for launch to
    /// `run_moe_mlp`'s decode fast path (the table on
    /// [`qwen3_5_moe_mlp_block`]).
    #[test]
    fn moe_block_op_sequence() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match &op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { selector: Some(_), .. } => "matmul_per_token",
                OpKind::Matmul { .. } => "matmul",
                OpKind::TopK { .. } => "topk",
                OpKind::Swiglu { .. } => "swiglu",
                OpKind::WeightedSum { .. } => "weighted_sum",
                OpKind::SigmoidGateAdd => "sigmoid_gate_add",
                OpKind::ResidualAdd => "residual_add",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",           // mlp_norm (gemma fold)
                "matmul",            // router logits [Tokens, E]
                "topk",              // launch_topk_softmax: idx + renormed weights
                "matmul_per_token",  // grouped gate_up over the selected experts
                "swiglu",            // chunked swiglu over [Tokens, k, Im]
                "matmul_per_token",  // grouped down
                "weighted_sum",      // [Tokens, k, H] -> [Tokens, H]
                "matmul",            // shared_expert.gate_up
                "swiglu",
                "matmul",            // shared_expert.down
                "matmul",            // shared_expert_gate: [Tokens, 1] logit
                "sigmoid_gate_add",  // routed + sigmoid(gate) * shared
                "residual_add",      // y += moe_out
            ]
        );
    }

    /// Without a shared expert (qwen3_moe: `shared_expert_intermediate` 0)
    /// the five shared ops fold away at trace time, llama_like-branch
    /// style, and the routed combine lands on the residual directly.
    #[test]
    fn moe_block_without_shared_expert_folds_the_shared_ops() {
        let facts = Qwen35MoeMlpFacts {
            shared_expert_intermediate: 0,
            norm_variant: NormVariant::Plain,
            ..Qwen35MoeMlpFacts::qwen3_5_35b_a3b()
        };
        let plan = qwen3_5_moe_mlp_block(&facts);
        assert_eq!(plan.ops.len(), 8);
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::SigmoidGateAdd))
        );
        assert!(!plan.ops.iter().any(|op| {
            matches!(&op.kind, OpKind::Matmul { weight, .. } if weight.contains("shared"))
        }));
        // The residual add consumes the weighted sum's output directly.
        let add = plan.ops.last().unwrap();
        assert!(matches!(add.kind, OpKind::ResidualAdd));
        let combine = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::WeightedSum { .. }))
            .unwrap();
        assert_eq!(add.inputs[0], combine.outputs[0]);
    }

    /// The dyn dataflow: TopK's index output is the fragment's only
    /// dyn-marked value, both expert matmuls name it as their selector AND
    /// their last input, and their weight names are `{e}` templates.
    #[test]
    fn moe_block_selector_dataflow() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let topk = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::TopK { .. }))
            .unwrap();
        let idx = topk.outputs[0];
        let dyn_values: Vec<_> = plan
            .values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.dyn_axis.is_some())
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(dyn_values, vec![idx]);
        assert_eq!(plan.values[idx as usize].dtype, DType::I32);

        let grouped: Vec<_> = plan
            .ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::Matmul { selector: Some(_), .. }))
            .collect();
        assert_eq!(grouped.len(), 2);
        for op in &grouped {
            let OpKind::Matmul { weight, selector, .. } = &op.kind else {
                unreachable!()
            };
            assert_eq!(*selector, Some(idx));
            assert_eq!(*op.inputs.last().unwrap(), idx);
            assert!(weight.contains("{e}"), "not a template: {weight}");
        }
        assert!(matches!(
            &grouped[0].kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.expert.{e}.gate_up"
        ));
        assert!(matches!(
            &grouped[1].kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.expert.{e}.down"
        ));
    }

    /// Route-expanded shapes: the grouped matmuls and the swiglu between
    /// them carry the `[Tokens, k, ...]` factored form of the driver's
    /// `[N*K, ...]` scratch, and the weighted sum collapses it back.
    #[test]
    fn moe_block_route_expanded_shapes() {
        let facts = Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
        let plan = qwen3_5_moe_mlp_block(&facts);
        let k = Dim::Const(facts.top_k);
        let by_kind = |pred: fn(&OpKind) -> bool| {
            plan.ops
                .iter()
                .filter(move |op| pred(&op.kind))
                .collect::<Vec<_>>()
        };

        let grouped = by_kind(|k| matches!(k, OpKind::Matmul { selector: Some(_), .. }));
        assert_eq!(
            plan.values[grouped[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(2 * facts.moe_intermediate)]
        );
        assert_eq!(
            plan.values[grouped[1].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(facts.hidden)]
        );

        // The routed swiglu keeps the route dims; the shared one is the
        // ordinary dense shape.
        let swiglus = by_kind(|k| matches!(k, OpKind::Swiglu { .. }));
        assert_eq!(
            plan.values[swiglus[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(facts.moe_intermediate)]
        );
        assert_eq!(
            plan.values[swiglus[1].outputs[0] as usize].shape.0,
            vec![
                Dim::Tokens,
                Dim::Const(facts.shared_expert_intermediate)
            ]
        );

        let combine = by_kind(|k| matches!(k, OpKind::WeightedSum { .. }));
        assert!(
            matches!(combine[0].kind, OpKind::WeightedSum { k } if k == facts.top_k)
        );
        assert_eq!(
            plan.values[combine[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, Dim::Const(facts.hidden)]
        );

        // The shared gate logit is the [Tokens, 1] scalar-gate GEMM.
        let gate = plan
            .ops
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::Matmul { weight, .. }
                    if weight.ends_with("shared_expert_gate"))
            })
            .unwrap();
        assert_eq!(
            plan.values[gate.outputs[0] as usize].shape.0,
            vec![Dim::Tokens, Dim::Const(1)]
        );
    }

    /// The fragment parameter is honest dataflow: value 0 is produced by no
    /// op, read by the block's first norm, and landed on by the final
    /// residual add.
    #[test]
    fn moe_block_residual_stream_is_a_fragment_parameter() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));
        assert!(matches!(&plan.ops[0].kind, OpKind::Rmsnorm { weight, .. }
            if weight == "layer.0.mlp_norm"));
        assert_eq!(plan.ops[0].inputs, vec![0]);
        let add = plan.ops.last().unwrap();
        assert!(matches!(add.kind, OpKind::ResidualAdd));
        assert_eq!(*add.inputs.last().unwrap(), 0);
    }

    /// The dyn vocabulary survives serde — selector fields, dyn markers,
    /// rank-3 shapes — and, per the additive rule, none of it appears in a
    /// dyn-free plan's serialization (the goldens pin that byte-for-byte;
    /// this pins the reason).
    #[test]
    fn moe_traced_form_round_trips() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let json = serde_json::to_string(&plan).unwrap();
        let back: ForwardPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);

        let dense = serde_json::to_string(&llama_like(&LlamaLikeFacts::qwen3_0_6b())).unwrap();
        assert!(!dense.contains("selector"));
        assert!(!dense.contains("dyn_axis"));
    }
}
