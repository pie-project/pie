use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{copy, pack, stack, Import, SfBase};

use super::model::{Head, Mixer, Mlp, Model};

pub fn import_hf<B: SfBase, W1: Dtype, W2: Dtype, K: KvDtype>(m: &Model<W1, W2, K>) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", copy("norm"));
    if let Head::Bank(_) = &m.head {
        i.write("lm_head", copy("lm_head"));
    }
    for (l, w) in m.layers.iter().enumerate() {
        i.write(format!("layer.{l}.mixer_norm"), copy(format!("layer.{l}.input_layernorm")));
        i.write(format!("layer.{l}.mlp_norm"), copy(format!("layer.{l}.post_attention_layernorm")));
        if w.res_blend.is_some() {
            i.write(format!("layer.{l}.res_norm"), copy(format!("layer.{l}.self_attention_res_norm")));
            i.write(format!("layer.{l}.res_proj"), copy(format!("layer.{l}.self_attention_res_proj")));
        }
        match &w.mixer {
            Mixer::Mla(a) => {
                i.write(format!("layer.{l}.q_a_proj"), copy(format!("layer.{l}.self_attn.q_a_proj")));
                i.write(format!("layer.{l}.q_a_norm"), copy(format!("layer.{l}.self_attn.q_a_layernorm")));
                i.write(format!("layer.{l}.q_b_proj"), copy(format!("layer.{l}.self_attn.q_b_proj")));
                i.write(format!("layer.{l}.kv_a_proj"), copy(format!("layer.{l}.self_attn.kv_a_proj_with_mqa")));
                i.write(format!("layer.{l}.kv_a_norm"), copy(format!("layer.{l}.self_attn.kv_a_layernorm")));
                i.write(format!("layer.{l}.kv_b_proj"), copy(format!("layer.{l}.self_attn.kv_b_proj")));
                if a.gate.is_some() {
                    i.write(format!("layer.{l}.o_gate"), copy(format!("layer.{l}.self_attn.g_proj")));
                }
                i.write(format!("layer.{l}.o_proj"), copy(format!("layer.{l}.self_attn.o_proj")));
            }
            Mixer::Kda(_) => {
                i.write(format!("layer.{l}.kda_qkv"), pack([
                    format!("layer.{l}.self_attn.q_proj"),
                    format!("layer.{l}.self_attn.k_proj"),
                    format!("layer.{l}.self_attn.v_proj"),
                ]));
                i.write(format!("layer.{l}.kda_conv"), pack([
                    format!("layer.{l}.self_attn.q_conv1d"),
                    format!("layer.{l}.self_attn.k_conv1d"),
                    format!("layer.{l}.self_attn.v_conv1d"),
                ]));
                i.write(format!("layer.{l}.kda_f_a"), copy(format!("layer.{l}.self_attn.f_a_proj")));
                i.write(format!("layer.{l}.kda_f_b"), copy(format!("layer.{l}.self_attn.f_b_proj")));
                i.write(format!("layer.{l}.kda_b"), copy(format!("layer.{l}.self_attn.b_proj")));
                i.write(format!("layer.{l}.kda_dt_bias"), copy(format!("layer.{l}.self_attn.dt_bias")));
                i.write(format!("layer.{l}.kda_a_log"), copy(format!("layer.{l}.self_attn.A_log")));
                i.write(format!("layer.{l}.kda_gate"), copy(format!("layer.{l}.self_attn.g_proj")));
                i.write(format!("layer.{l}.kda_o_norm"), copy(format!("layer.{l}.self_attn.o_norm")));
                i.write(format!("layer.{l}.kda_o_proj"), copy(format!("layer.{l}.self_attn.o_proj")));
            }
        }
        match &w.mlp {
            Mlp::Dense { .. } => {
                i.write(format!("layer.{l}.gate_up"), pack([
                    format!("layer.{l}.mlp.gate_proj"),
                    format!("layer.{l}.mlp.up_proj"),
                ]));
                i.write(format!("layer.{l}.down"), copy(format!("layer.{l}.mlp.down_proj")));
            }
            Mlp::Routed { shared, experts, .. } => {
                i.write(format!("layer.{l}.router"), copy(format!("layer.{l}.block_sparse_moe.gate")));
                i.write(format!("layer.{l}.experts_gate_up"), stack((0..*experts).map(|e| pack([
                    format!("layer.{l}.block_sparse_moe.experts.{e}.w1"),
                    format!("layer.{l}.block_sparse_moe.experts.{e}.w3"),
                ]))));
                i.write(format!("layer.{l}.experts_down"), stack((0..*experts).map(|e| {
                    copy(format!("layer.{l}.block_sparse_moe.experts.{e}.w2"))
                })));
                if shared.is_some() {
                    i.write(format!("layer.{l}.shared_gate_up"), pack([
                        format!("layer.{l}.block_sparse_moe.shared_expert.gate_proj"),
                        format!("layer.{l}.block_sparse_moe.shared_expert.up_proj"),
                    ]));
                    i.write(format!("layer.{l}.shared_down"), copy(format!("layer.{l}.block_sparse_moe.shared_expert.down_proj")));
                }
            }
        }
    }
    i
}
