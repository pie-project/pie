use crate::shared::vocabulary::{Member, Vocab};

pub const VOCAB: Vocab = Vocab(&[
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.shared_experts.gate_proj"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.self_attn.kv_a_layernorm"),
    Member::same("model.layers.{layer}.self_attn.kv_a_proj_with_mqa"),
    Member::same("model.layers.{layer}.self_attn.kv_b_proj"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_a_layernorm"),
    Member::same("model.layers.{layer}.self_attn.q_a_proj"),
    Member::same("model.layers.{layer}.self_attn.q_b_proj"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
