use crate::shared::vocabulary::{Member, Vocab};

pub const VOCAB: Vocab = Vocab(&[

    Member::same("model.layers.{layer}.altup.modality_router"),
    Member::same("model.layers.{layer}.altup.router_norm"),
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.laurel.linear_left"),
    Member::same("model.layers.{layer}.laurel.linear_right"),
    Member::same("model.layers.{layer}.laurel.post_laurel_norm"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.up_proj"),
    Member::same("model.layers.{layer}.per_layer_input_gate"),
    Member::same("model.layers.{layer}.per_layer_projection"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.post_feedforward_layernorm"),
    Member::same("model.layers.{layer}.post_per_layer_input_norm"),
    Member::same("model.layers.{layer}.pre_feedforward_layernorm"),
    Member::same("model.layers.{layer}.self_attn.k_norm"),
    Member::same("model.layers.{layer}.self_attn.k_proj"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_norm"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    Member::same("model.layers.{layer}.self_attn.v_norm"),
    Member::same("model.layers.{layer}.self_attn.v_proj"),

    Member::same("model.embed_tokens"),
    Member::same("model.embed_tokens_per_layer"),
    Member::same("lm_head"),
    Member::same("model.norm"),
    Member::same("model.per_layer_model_projection"),
    Member::same("model.per_layer_projection_norm"),
]);
