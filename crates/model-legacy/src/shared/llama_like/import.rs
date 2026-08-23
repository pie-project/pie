use crate::shared::vocabulary::{Member, Vocab, gguf_member};
use model_loader::checkpoint::Attributes;

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
    Member::same("model.layers.{layer}.self_attn.q_norm"),
    Member::gguf(
        "model.layers.{layer}.input_layernorm",
        "blk.{layer}.attn_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.post_attention_layernorm",
        "blk.{layer}.ffn_norm",
    ),
    Member::same("model.layers.{layer}.post_feedforward_layernorm"),
    Member::gguf("model.layers.{layer}.mlp.gate_proj", "blk.{layer}.ffn_gate"),
    Member::gguf("model.layers.{layer}.mlp.up_proj", "blk.{layer}.ffn_up"),
    Member::gguf("model.layers.{layer}.mlp.down_proj", "blk.{layer}.ffn_down"),
    Member::gguf("model.embed_tokens", "token_embd"),
    Member::gguf("model.norm", "output_norm"),
    Member::gguf("lm_head", "output"),
]);

#[must_use]
pub fn is_derived(gguf: &str) -> bool {
    gguf == "rope_freqs.weight"
}

#[must_use]
pub fn regroup_heads(attributes: &Attributes, gguf: &str) -> Option<u32> {
    let (_, member) = gguf_member(gguf)?;
    let key = match member {
        "attn_q" => "llama.attention.head_count",
        "attn_k" => "llama.attention.head_count_kv",
        _ => return None,
    };
    match attributes.get(key)? {
        model_loader::checkpoint::Attribute::Uint(heads) => u32::try_from(*heads).ok(),
        _ => None,
    }
    .filter(|heads| *heads > 0)
}
