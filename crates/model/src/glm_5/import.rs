use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{Import, SfBase, copy, pack, stack};

use super::model::{Mlp, Model};

pub fn import_hf<B: SfBase, W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>(
    m: &Model<W1, W2, K, TP>,
) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", copy("norm"));
    i.write("lm_head", copy("lm_head"));
    for (l, w) in m.layers.iter().enumerate() {
        i.write(
            format!("layer.{l}.attn_norm"),
            copy(format!("layer.{l}.input_layernorm")),
        );
        i.write(
            format!("layer.{l}.mlp_norm"),
            copy(format!("layer.{l}.post_attention_layernorm")),
        );
        i.write(
            format!("layer.{l}.q_a_proj"),
            copy(format!("layer.{l}.self_attn.q_a_proj")),
        );
        i.write(
            format!("layer.{l}.q_a_norm"),
            copy(format!("layer.{l}.self_attn.q_a_layernorm")),
        );
        i.write(
            format!("layer.{l}.q_b_proj"),
            copy(format!("layer.{l}.self_attn.q_b_proj")),
        );
        i.write(
            format!("layer.{l}.kv_a_proj"),
            copy(format!("layer.{l}.self_attn.kv_a_proj_with_mqa")),
        );
        i.write(
            format!("layer.{l}.kv_a_norm"),
            copy(format!("layer.{l}.self_attn.kv_a_layernorm")),
        );
        i.write(
            format!("layer.{l}.kv_b_proj"),
            copy(format!("layer.{l}.self_attn.kv_b_proj")),
        );
        i.write(
            format!("layer.{l}.o_proj"),
            copy(format!("layer.{l}.self_attn.o_proj")),
        );
        i.write(
            format!("layer.{l}.index_q_proj"),
            copy(format!("layer.{l}.self_attn.indexer.wq_b")),
        );
        i.write(
            format!("layer.{l}.index_k_proj"),
            copy(format!("layer.{l}.self_attn.indexer.wk")),
        );
        i.write(
            format!("layer.{l}.index_weights"),
            copy(format!("layer.{l}.self_attn.indexer.weights_proj")),
        );
        i.write(
            format!("layer.{l}.index_k_norm"),
            copy(format!("layer.{l}.self_attn.indexer.k_norm")),
        );
        i.write(
            format!("layer.{l}.index_k_norm_bias"),
            copy(format!("layer.{l}.self_attn.indexer.k_norm.bias")),
        );
        match &w.mlp {
            Mlp::Dense { .. } => {
                i.write(
                    format!("layer.{l}.gate_up"),
                    pack([
                        format!("layer.{l}.mlp.gate_proj"),
                        format!("layer.{l}.mlp.up_proj"),
                    ]),
                );
                i.write(
                    format!("layer.{l}.down"),
                    copy(format!("layer.{l}.mlp.down_proj")),
                );
            }
            Mlp::Routed {
                shared, experts, ..
            } => {
                i.write(
                    format!("layer.{l}.router"),
                    copy(format!("layer.{l}.mlp.gate")),
                );
                i.write(
                    format!("layer.{l}.experts_gate_up"),
                    stack((0..*experts).map(|e| {
                        pack([
                            format!("layer.{l}.mlp.experts.{e}.gate_proj"),
                            format!("layer.{l}.mlp.experts.{e}.up_proj"),
                        ])
                    })),
                );
                i.write(
                    format!("layer.{l}.experts_down"),
                    stack(
                        (0..*experts).map(|e| copy(format!("layer.{l}.mlp.experts.{e}.down_proj"))),
                    ),
                );
                if shared.is_some() {
                    i.write(
                        format!("layer.{l}.shared_gate_up"),
                        pack([
                            format!("layer.{l}.mlp.shared_experts.gate_proj"),
                            format!("layer.{l}.mlp.shared_experts.up_proj"),
                        ]),
                    );
                    i.write(
                        format!("layer.{l}.shared_down"),
                        copy(format!("layer.{l}.mlp.shared_experts.down_proj")),
                    );
                }
            }
        }
    }
    i
}
