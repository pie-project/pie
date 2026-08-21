use crate::shared::vocabulary::{Member, Vocab};

pub const VOCAB: Vocab = Vocab(&[

    Member::same("backbone.layers.{layer}.mixer.A_log"),
    Member::same("backbone.layers.{layer}.mixer.D"),
    Member::same("backbone.layers.{layer}.mixer.conv1d"),
    Member::same("backbone.layers.{layer}.mixer.down_proj"),
    Member::same("backbone.layers.{layer}.mixer.dt_bias"),
    Member::same("backbone.layers.{layer}.mixer.experts.{expert}.down_proj"),
    Member::same("backbone.layers.{layer}.mixer.experts.{expert}.up_proj"),
    Member::same("backbone.layers.{layer}.mixer.gate_proj"),
    Member::same("backbone.layers.{layer}.mixer.in_proj"),
    Member::same("backbone.layers.{layer}.mixer.k_norm"),
    Member::same("backbone.layers.{layer}.mixer.k_proj"),
    Member::same("backbone.layers.{layer}.mixer.norm"),
    Member::same("backbone.layers.{layer}.mixer.o_proj"),
    Member::same("backbone.layers.{layer}.mixer.out_proj"),
    Member::same("backbone.layers.{layer}.mixer.q_norm"),
    Member::same("backbone.layers.{layer}.mixer.q_proj"),
    Member::same("backbone.layers.{layer}.mixer.up_proj"),
    Member::same("backbone.layers.{layer}.mixer.v_proj"),
    Member::same("backbone.layers.{layer}.norm"),

    Member::same("backbone.embeddings"),
    Member::same("backbone.norm_f"),
    Member::same("lm_head"),
]);
