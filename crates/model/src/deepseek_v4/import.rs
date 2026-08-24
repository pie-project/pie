use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{Import, SfBase, copy, pack, stack};

use super::model::{Mlp, Model};

pub fn import_hf<B: SfBase, W1: Dtype, K: KvDtype, const TP: usize>(
    m: &Model<W1, K, TP>,
) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", copy("norm"));
    i.write("hyper.head_scale", copy("hc_head_scale"));
    i.write("hyper.head_base", copy("hc_head_base"));
    for (l, w) in m.layers.iter().enumerate() {
        i.write(
            format!("layer.{l}.attn_mix_scale"),
            copy(format!("layer.{l}.hc_attn_scale")),
        );
        i.write(
            format!("layer.{l}.attn_mix_base"),
            copy(format!("layer.{l}.hc_attn_base")),
        );
        i.write(
            format!("layer.{l}.mlp_mix_scale"),
            copy(format!("layer.{l}.hc_mlp_scale")),
        );
        i.write(
            format!("layer.{l}.mlp_mix_base"),
            copy(format!("layer.{l}.hc_mlp_base")),
        );
        i.write(
            format!("layer.{l}.q_down"),
            copy(format!("layer.{l}.attn.wq_a")),
        );
        i.write(
            format!("layer.{l}.q_norm"),
            copy(format!("layer.{l}.attn.q_norm")),
        );
        i.write(
            format!("layer.{l}.q_up"),
            copy(format!("layer.{l}.attn.wq_b")),
        );
        i.write(
            format!("layer.{l}.kv_down"),
            copy(format!("layer.{l}.attn.wkv")),
        );
        i.write(
            format!("layer.{l}.kv_norm"),
            copy(format!("layer.{l}.attn.kv_norm")),
        );
        i.write(
            format!("layer.{l}.o_down"),
            copy(format!("layer.{l}.attn.wo_a")),
        );
        i.write(
            format!("layer.{l}.o_up"),
            copy(format!("layer.{l}.attn.wo_b")),
        );
        i.write(
            format!("layer.{l}.attn_sink"),
            copy(format!("layer.{l}.attn.sinks")),
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
            Mlp::Routed { experts, .. } => {
                i.write(
                    format!("layer.{l}.router"),
                    copy(format!("layer.{l}.ffn.gate")),
                );
                i.write(
                    format!("layer.{l}.router_bias"),
                    copy(format!("layer.{l}.ffn.gate.bias")),
                );
                i.write(
                    format!("layer.{l}.experts_gate_up"),
                    stack((0..*experts).map(|e| {
                        pack([
                            format!("layer.{l}.ffn.experts.{e}.w1"),
                            format!("layer.{l}.ffn.experts.{e}.w3"),
                        ])
                    })),
                );
                i.write(
                    format!("layer.{l}.experts_down"),
                    stack((0..*experts).map(|e| copy(format!("layer.{l}.ffn.experts.{e}.w2")))),
                );
            }
        }
    }
    i
}
