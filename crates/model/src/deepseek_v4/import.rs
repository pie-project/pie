use crate::shared::vocabulary::{Member, Vocab};

pub const VOCAB: Vocab = Vocab(&[

    Member::same("model.layers.{layer}.attn.kv_norm"),
    Member::same("model.layers.{layer}.attn.q_norm"),
    Member::same("model.layers.{layer}.attn.wkv"),
    Member::same("model.layers.{layer}.attn.wo"),
    Member::same("model.layers.{layer}.attn.wo_a"),
    Member::same("model.layers.{layer}.attn.wo_b"),
    Member::same("model.layers.{layer}.attn.wq"),
    Member::same("model.layers.{layer}.attn.wq_a"),
    Member::same("model.layers.{layer}.attn.wq_b"),
    Member::same("model.layers.{layer}.attn_norm"),
    Member::same("model.layers.{layer}.ffn.gate"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.up_proj"),
    Member::same("model.layers.{layer}.mlp_norm"),

    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
