use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{copy, pack, squeeze, Import, SfBase};

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
            Mlp::Routed { .. } => {
                i.write(format!("layer.{l}.router"), copy(format!("layer.{l}.mlp.gate")));
                // THE EXPERT BANKS SHIP FUSED, AND FUSED IS ALREADY
                // CANONICAL. This table used to read 256 per-expert
                // `gate_proj`/`up_proj`/`down_proj` rows and rebuild the
                // stack with `stack(pack(..))`; the shipped 35B-A3B holds
                // no such tensor. It holds two, per layer:
                //
                //     mlp.experts.gate_up_proj  BF16 [256, 1024, 2048]
                //     mlp.experts.down_proj     BF16 [256, 2048, 512]
                //
                // and with `hidden = 2048`, `inter = 512` those read
                // `[E, 2*inter, hidden]` and `[E, hidden, inter]` — the
                // `[E, out, in]` the declaration states and
                // `moe_grouped_gemm.cuh` indexes. Byte for byte it is what
                // `stack(pack(..))` was building, so the TARGET does not
                // move and no permute verb is owed: a `Permute3` written
                // here would be an identity nothing exercises, and this
                // interpreter's own rule (see `Source::ScalarOf`) is that
                // an unmeasured verb is a wrong weight rather than a
                // refusal. Only the source spelling changes.
                //
                // Which half of the 1024 is `gate` is settled the same way
                // the rest of this file is — off the file. `down_proj` is
                // NOT fused across gate/up: its `[hidden, inter]` matrix
                // has one column per hidden unit, in unit order. Over 16
                // experts of layer 0, per-row norms of the contiguous
                // halves correlate with down's per-column norms at +0.993
                // and +0.996, and the two halves correlate with each other
                // at +0.995; read as interleaved (`0::2` / `1::2`) the same
                // correlations are +0.001, +0.020 and -0.008. So the rows
                // are `[gate(512) ; up(512)]` in unit order — contiguous
                // halves, gate first as the name says, which is exactly the
                // packing `mlp.swiglu` splits and the packing the
                // shared-expert row below still builds by hand.
                i.write(format!("layer.{l}.experts_gate_up"), copy(format!("layer.{l}.mlp.experts.gate_up_proj")));
                i.write(format!("layer.{l}.experts_down"), copy(format!("layer.{l}.mlp.experts.down_proj")));
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
