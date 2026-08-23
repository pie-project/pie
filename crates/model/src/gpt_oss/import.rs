use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{copy, deinterleave, Import, SfBase};

use super::model::Model;

pub fn import_hf<B: SfBase, W1: Dtype, W2: Dtype, K: KvDtype>(m: &Model<W1, W2, K>) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", copy("norm"));
    i.write("lm_head", copy("lm_head"));
    for l in 0..m.layers.len() {
        i.write(format!("layer.{l}.attn_norm"), copy(format!("layer.{l}.input_layernorm")));
        i.write(format!("layer.{l}.mlp_norm"), copy(format!("layer.{l}.post_attention_layernorm")));
        i.write(format!("layer.{l}.q_proj"), copy(format!("layer.{l}.self_attn.q_proj")));
        i.write(format!("layer.{l}.q_bias"), copy(format!("layer.{l}.self_attn.q_proj.bias")));
        i.write(format!("layer.{l}.k_proj"), copy(format!("layer.{l}.self_attn.k_proj")));
        i.write(format!("layer.{l}.k_bias"), copy(format!("layer.{l}.self_attn.k_proj.bias")));
        i.write(format!("layer.{l}.v_proj"), copy(format!("layer.{l}.self_attn.v_proj")));
        i.write(format!("layer.{l}.v_bias"), copy(format!("layer.{l}.self_attn.v_proj.bias")));
        i.write(format!("layer.{l}.o_proj"), copy(format!("layer.{l}.self_attn.o_proj")));
        i.write(format!("layer.{l}.o_bias"), copy(format!("layer.{l}.self_attn.o_proj.bias")));
        i.write(format!("layer.{l}.attn_sinks"), copy(format!("layer.{l}.self_attn.sinks")));
        i.write(format!("layer.{l}.router"), copy(format!("layer.{l}.mlp.router")));
        i.write(format!("layer.{l}.router_bias"), copy(format!("layer.{l}.mlp.router.bias")));
        i.write(
            format!("layer.{l}.expert_gate_up_bank"),
            deinterleave(format!("layer.{l}.mlp.experts.gate_up_proj"), 2),
        );
        i.write(
            format!("layer.{l}.expert_gate_up_bias"),
            deinterleave(format!("layer.{l}.mlp.experts.gate_up_proj_bias"), 2),
        );
        i.write(
            format!("layer.{l}.expert_down_bank"),
            copy(format!("layer.{l}.mlp.experts.down_proj")),
        );
        i.write(
            format!("layer.{l}.expert_down_bias"),
            copy(format!("layer.{l}.mlp.experts.down_proj_bias")),
        );
    }
    i
}
