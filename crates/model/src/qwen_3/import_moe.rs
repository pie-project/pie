use crate::shared::vocabulary::{Member, Vocab, gguf_member};

pub const VOCAB: Vocab = Vocab(&[

    Member::gguf(
        "model.layers.{layer}.self_attn.q_proj",
        "blk.{layer}.attn_q",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.k_proj",
        "blk.{layer}.attn_k",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.v_proj",
        "blk.{layer}.attn_v",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.o_proj",
        "blk.{layer}.attn_output",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.q_norm",
        "blk.{layer}.attn_q_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.k_norm",
        "blk.{layer}.attn_k_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.input_layernorm",
        "blk.{layer}.attn_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.post_attention_layernorm",
        "blk.{layer}.ffn_norm",
    ),
    Member::gguf("model.layers.{layer}.mlp.gate", "blk.{layer}.ffn_gate_inp"),
    Member::gguf(
        "model.layers.{layer}.mlp.experts.{expert}.gate_proj",
        "blk.{layer}.ffn_gate_exps",
    ),
    Member::gguf(
        "model.layers.{layer}.mlp.experts.{expert}.up_proj",
        "blk.{layer}.ffn_up_exps",
    ),
    Member::gguf(
        "model.layers.{layer}.mlp.experts.{expert}.down_proj",
        "blk.{layer}.ffn_down_exps",
    ),

    Member::gguf("model.embed_tokens", "token_embd"),
    Member::gguf("model.norm", "output_norm"),
    Member::gguf("lm_head", "output"),
]);

#[must_use]
pub fn is_stacked(gguf: &str) -> bool {
    if !gguf.ends_with(".weight") {
        return false;
    }
    matches!(
        gguf_member(gguf),
        Some((_, "ffn_gate_exps" | "ffn_up_exps" | "ffn_down_exps"))
    )
}
