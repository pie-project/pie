use crate::shared::vocabulary::{Member, Vocab};

pub const VOCAB: Vocab = Vocab(&[

    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.mlp.experts.down_proj.bias"),
    Member::same("model.layers.{layer}.mlp.experts.down_proj_bias"),
    Member::same("model.layers.{layer}.mlp.experts.gate_proj.bias"),
    Member::same("model.layers.{layer}.mlp.experts.gate_up_proj_bias"),
    Member::same("model.layers.{layer}.mlp.experts.up_proj.bias"),
    Member::same("model.layers.{layer}.mlp.router"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.self_attn.k_proj"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    Member::same("model.layers.{layer}.self_attn.sinks"),
    Member::same("model.layers.{layer}.self_attn.v_proj"),

    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
