use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{copy, pack, squeeze, stack, Import, SfBase};

use super::model::{Head, Mixer, Mlp, Model};

pub fn import_hf<B: SfBase, W1: Dtype, K: KvDtype>(m: &Model<W1, K>) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", copy("norm"));
    if let Head::Bank(_) = &m.head {
        i.write("lm_head", copy("lm_head"));
    }
    for (l, w) in m.layers.iter().enumerate() {
        i.write(format!("layer.{l}.mixer_norm"), copy(format!("layer.{l}.input_layernorm")));
        i.write(format!("layer.{l}.mlp_norm"), copy(format!("layer.{l}.post_attention_layernorm")));
        match &w.mixer {
            Mixer::Attn(_) => {
                i.write(format!("layer.{l}.qg_proj"), copy(format!("layer.{l}.self_attn.q_proj")));
                i.write(format!("layer.{l}.k_proj"), copy(format!("layer.{l}.self_attn.k_proj")));
                i.write(format!("layer.{l}.v_proj"), copy(format!("layer.{l}.self_attn.v_proj")));
                i.write(format!("layer.{l}.o_proj"), copy(format!("layer.{l}.self_attn.o_proj")));
                i.write(format!("layer.{l}.q_norm"), copy(format!("layer.{l}.self_attn.q_norm")));
                i.write(format!("layer.{l}.k_norm"), copy(format!("layer.{l}.self_attn.k_norm")));
            }
            Mixer::Gdn(_) => {
                i.write(format!("layer.{l}.in_qkvz"), pack([
                    format!("layer.{l}.linear_attn.in_proj_qkv"),
                    format!("layer.{l}.linear_attn.in_proj_z"),
                ]));
                i.write(format!("layer.{l}.in_ba"), pack([
                    format!("layer.{l}.linear_attn.in_proj_b"),
                    format!("layer.{l}.linear_attn.in_proj_a"),
                ]));
                i.write(format!("layer.{l}.conv"), squeeze(format!("layer.{l}.linear_attn.conv1d"), 1));
                i.write(format!("layer.{l}.dt_bias"), copy(format!("layer.{l}.linear_attn.dt_bias")));
                i.write(format!("layer.{l}.a_log"), copy(format!("layer.{l}.linear_attn.A_log")));
                i.write(format!("layer.{l}.gdn_norm"), copy(format!("layer.{l}.linear_attn.norm")));
                i.write(format!("layer.{l}.out_proj"), copy(format!("layer.{l}.linear_attn.out_proj")));
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
            Mlp::Routed { experts, .. } => {
                i.write(format!("layer.{l}.router"), copy(format!("layer.{l}.mlp.gate")));
                i.write(format!("layer.{l}.experts_gate_up"), stack((0..*experts).map(|e| pack([
                    format!("layer.{l}.mlp.experts.{e}.gate_proj"),
                    format!("layer.{l}.mlp.experts.{e}.up_proj"),
                ]))));
                i.write(format!("layer.{l}.experts_down"), stack((0..*experts).map(|e| {
                    copy(format!("layer.{l}.mlp.experts.{e}.down_proj"))
                })));
                i.write(format!("layer.{l}.shared_gate_up"), pack([
                    format!("layer.{l}.mlp.shared_expert.gate_proj"),
                    format!("layer.{l}.mlp.shared_expert.up_proj"),
                ]));
                i.write(format!("layer.{l}.shared_down"), copy(format!("layer.{l}.mlp.shared_expert.down_proj")));
                i.write(format!("layer.{l}.shared_gate"), copy(format!("layer.{l}.mlp.shared_expert_gate")));
            }
        }
    }
    i
}
