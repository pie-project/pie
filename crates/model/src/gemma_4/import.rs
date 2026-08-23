use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{copy, pack, plus_one, scalar_of, GgufBase, Import, SfBase};

use super::model::{AttnBanks, Model};

pub fn import_hf<B: SfBase, W1: Dtype, K: KvDtype>(m: &Model<W1, K>) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", plus_one("norm"));
    for (l, w) in m.layers.iter().enumerate() {
        i.write(format!("layer.{l}.attn_norm"), plus_one(format!("layer.{l}.input_layernorm")));
        i.write(format!("layer.{l}.post_attn_norm"), plus_one(format!("layer.{l}.post_attention_layernorm")));
        i.write(format!("layer.{l}.pre_ffw_norm"), plus_one(format!("layer.{l}.pre_feedforward_layernorm")));
        i.write(format!("layer.{l}.post_ffw_norm"), plus_one(format!("layer.{l}.post_feedforward_layernorm")));
        i.write(format!("layer.{l}.q_norm"), plus_one(format!("layer.{l}.self_attn.q_norm")));
        match &w.attn.banks {
            AttnBanks::Owned { .. } => {
                i.write(format!("layer.{l}.k_norm"), plus_one(format!("layer.{l}.self_attn.k_norm")));
                i.write(format!("layer.{l}.qkv"), pack([
                    format!("layer.{l}.self_attn.q_proj"),
                    format!("layer.{l}.self_attn.k_proj"),
                    format!("layer.{l}.self_attn.v_proj"),
                ]));
            }
            AttnBanks::Shared { .. } => {
                i.write(format!("layer.{l}.q_proj"), copy(format!("layer.{l}.self_attn.q_proj")));
            }
        }
        i.write(format!("layer.{l}.o_proj"), copy(format!("layer.{l}.self_attn.o_proj")));
        i.write(format!("layer.{l}.gate_up"), pack([
            format!("layer.{l}.mlp.gate_proj"),
            format!("layer.{l}.mlp.up_proj"),
        ]));
        i.write(format!("layer.{l}.down"), copy(format!("layer.{l}.mlp.down_proj")));
    }
    if m.ple.is_some() {
        i.write("ple.table", copy("embed_tokens_per_layer"));
        i.write("ple.model_proj", copy("per_layer_model_projection"));
        i.write("ple.model_norm", plus_one("per_layer_projection_norm"));
        for l in 0..m.layers.len() {
            i.write(format!("layer.{l}.ple_gate"), copy(format!("layer.{l}.per_layer_input_gate")));
            i.write(format!("layer.{l}.ple_proj"), copy(format!("layer.{l}.per_layer_projection")));
            i.write(format!("layer.{l}.ple_norm"), plus_one(format!("layer.{l}.post_per_layer_input_norm")));
            // The HF release files this as a `[1]` tensor of its own, beside
            // the norm rather than inside it -- `layers.{l}.layer_scalar`,
            // which the legacy alias table reaches under the role name
            // `scalar` (`shared/weight_names.rs`). A plain copy, then; the
            // GGUF leg below still says `scalar_of`, whose form nobody has
            // read a file for.
            i.write(format!("layer.{l}.ple_scalar"), copy(format!("layer.{l}.layer_scalar")));
        }
    }
    i
}

pub fn import_gguf<B: GgufBase, W1: Dtype, K: KvDtype>(m: &Model<W1, K>) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("token_embd.weight"));
    i.write("final_norm", copy("output_norm.weight"));
    for (l, w) in m.layers.iter().enumerate() {
        i.write(format!("layer.{l}.attn_norm"), copy(format!("blk.{l}.attn_norm.weight")));
        i.write(format!("layer.{l}.post_attn_norm"), copy(format!("blk.{l}.post_attention_norm.weight")));
        i.write(format!("layer.{l}.pre_ffw_norm"), copy(format!("blk.{l}.ffn_norm.weight")));
        i.write(format!("layer.{l}.post_ffw_norm"), copy(format!("blk.{l}.post_ffw_norm.weight")));
        i.write(format!("layer.{l}.q_norm"), copy(format!("blk.{l}.attn_q_norm.weight")));
        match &w.attn.banks {
            AttnBanks::Owned { .. } => {
                i.write(format!("layer.{l}.k_norm"), copy(format!("blk.{l}.attn_k_norm.weight")));
                i.write(format!("layer.{l}.qkv"), pack([
                    format!("blk.{l}.attn_q.weight"),
                    format!("blk.{l}.attn_k.weight"),
                    format!("blk.{l}.attn_v.weight"),
                ]));
            }
            AttnBanks::Shared { .. } => {
                i.write(format!("layer.{l}.q_proj"), copy(format!("blk.{l}.attn_q.weight")));
            }
        }
        i.write(format!("layer.{l}.o_proj"), copy(format!("blk.{l}.attn_output.weight")));
        i.write(format!("layer.{l}.gate_up"), pack([
            format!("blk.{l}.ffn_gate.weight"),
            format!("blk.{l}.ffn_up.weight"),
        ]));
        i.write(format!("layer.{l}.down"), copy(format!("blk.{l}.ffn_down.weight")));
    }
    if m.ple.is_some() {
        i.write("ple.table", copy("per_layer_token_embd.weight"));
        i.write("ple.model_proj", copy("per_layer_model_proj.weight"));
        i.write("ple.model_norm", copy("per_layer_proj_norm.weight"));
        for l in 0..m.layers.len() {
            i.write(format!("layer.{l}.ple_gate"), copy(format!("blk.{l}.per_layer_inp_gate.weight")));
            i.write(format!("layer.{l}.ple_proj"), copy(format!("blk.{l}.per_layer_proj.weight")));
            i.write(format!("layer.{l}.ple_norm"), copy(format!("blk.{l}.post_per_layer_norm.weight")));
            i.write(format!("layer.{l}.ple_scalar"), scalar_of(format!("blk.{l}.post_per_layer_norm.weight")));
        }
    }
    i
}
