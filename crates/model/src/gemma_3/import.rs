//! Gemma 3's tensor names, in pie's vocabulary and the two foreign ones.
//!
//! # The family where a rename is not enough, for the second reason
//!
//! `llama_2::import` has to reorder rows. This one has to change values.
//! Gemma's rmsnorm is `x * (1 + w)`; HuggingFace publishes `w` and pie's
//! kernel adds the one ([`NormVariant::Gemma`](model_ir::trace::NormVariant)),
//! while llama.cpp folds it in at conversion time and publishes `w + 1`. Both
//! files describe the same model. Only one of them matches the kernel.
//!
//! Measured on `gemma-3-270m-it-F16.gguf` against the safetensors release it
//! was converted from, all 18 layers: every tensor whose name ends in a norm
//! is exactly `+1.000000` away, mean and max alike, and no other tensor
//! differs by more than F16 rounding. That is why [`folded_constant`] answers
//! by suffix rather than by listing four names -- the suffix is what the
//! converter itself keys on, and a list would go stale the next time Gemma
//! grows a norm.
//!
//! Checked again end to end on `unsloth/gemma-3-1b-it-GGUF` BF16, which is
//! the smallest Gemma 3 this build actually has a row for -- 270m has none,
//! so that first measurement could compare values but never reach a catalog
//! row. Importing the GGUF and the safetensors release separately gives 345
//! of 346 objects **byte for byte identical**, the tokenizer included; the
//! one that differs is `__meta__/model/config`, where a GGUF carries its
//! key-value block and a checkpoint carries `config.json`. The artifact
//! identifies as `gemma-3-1b` and builds 392 tensors.
//!
//! That the unfold is BIT-exact and not merely close is the load-bearing
//! part, and it is a property of WHERE it happens rather than of the
//! arithmetic: subtracting the one at F32 before the narrowing cast lands
//! back on the same BF16 the checkpoint holds, while rounding `w + 1` to
//! BF16 first and subtracting afterwards turns `w = 0.0123` into `0.0156`.
//!
//! Renaming and stopping would produce an artifact of exactly the right
//! shape, which identifies as the right row, which builds, and which is
//! wrong everywhere by one in the norm -- the failure this module exists to
//! prevent and the only one no later stage can see.
//!
//! # Four norms, and the pair that must not be swapped
//!
//! Gemma 3 is a sandwich: a norm before attention and after it, and a norm
//! before the MLP and after it. llama.cpp names them for what they PRECEDE
//! and HuggingFace for where they SIT, which lines up as
//!
//! | GGUF | HuggingFace |
//! |---|---|
//! | `attn_norm` | `input_layernorm` |
//! | `post_attention_norm` | `post_attention_layernorm` |
//! | `ffn_norm` | `pre_feedforward_layernorm` |
//! | `post_ffw_norm` | `post_feedforward_layernorm` |
//!
//! and the third row is the trap. In `qwen_2::import` and `qwen_3::import`,
//! `ffn_norm` is `post_attention_layernorm`, because those families have one
//! norm between attention and the MLP and both vocabularies name it from
//! their own end. Gemma has two, so the name `post_attention_layernorm` is
//! taken by a DIFFERENT tensor and `ffn_norm` has to go somewhere else.
//! Copying the qwen table would swap two norms that are the same shape, in a
//! model that would load and build.
//!
//! This is the concrete reason these tables are not shared. `shared`'s bar is
//! "more than one generation binds it", and the nine rows that look identical
//! across four families are identical by coincidence of two naming schemes
//! agreeing, not by anything that keeps them agreeing.
//!
//! # What is not here
//!
//! No vision tower. `gemma-3-270m` is text-only, and the multimodal releases
//! put the tower under `v.` names that this map does not carry -- so they are
//! refused by name rather than half-imported. Adding them is rows in
//! [`LAYER`] and [`MODEL`], once there is a file to check them against.
use crate::shared::vocabulary::{Member, Vocab};

/// Every tensor Gemma 3 publishes, and what each vocabulary calls it.
///
/// The four norm rows are the ones to read against the table in this module's
/// documentation; the other nine are the same as every llama-like family's
/// and are written out anyway, because a table assembled from a base plus a
/// delta cannot be read against a file.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
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
    // The four rows where the two vocabularies disagree about position.
    Member::gguf(
        "model.layers.{layer}.input_layernorm",
        "blk.{layer}.attn_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.post_attention_layernorm",
        "blk.{layer}.post_attention_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.pre_feedforward_layernorm",
        "blk.{layer}.ffn_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.post_feedforward_layernorm",
        "blk.{layer}.post_ffw_norm",
    ),
    Member::gguf("model.layers.{layer}.mlp.gate_proj", "blk.{layer}.ffn_gate"),
    Member::gguf("model.layers.{layer}.mlp.up_proj", "blk.{layer}.ffn_up"),
    Member::gguf("model.layers.{layer}.mlp.down_proj", "blk.{layer}.ffn_down"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::gguf("model.embed_tokens", "token_embd"),
    Member::gguf("model.norm", "output_norm"),
    Member::gguf("lm_head", "output"),
]);

/// The constant llama.cpp folded into this tensor, for pie to take back out.
///
/// Keyed on the GGUF name, and by suffix: `Gemma3Model.modify_tensors` adds
/// one to everything whose name ends in `norm.weight` and to nothing else, so
/// asking the same question the converter asked is the only version of this
/// that cannot drift from it. A list of the four names Gemma 3 publishes
/// today would be a second statement of the same rule, and the two could
/// disagree.
///
/// `token_embd` is deliberately not here even though Gemma scales its
/// embeddings by `sqrt(d_model)`: that scaling is not folded into the
/// checkpoint by either side. Both files hold the same embedding, and pie's
/// forward pass applies the factor. This function is only for constants a
/// FILE disagrees about.
#[must_use]
pub fn folded_constant(gguf: &str) -> Option<f32> {
    gguf.ends_with("norm.weight").then_some(-1.0)
}

#[cfg(test)]
mod tests {
    use super::{VOCAB, folded_constant};

    /// Every name a real Gemma 3 GGUF holds is mapped.
    ///
    /// The list is the fifteen distinct patterns read off
    /// `gemma-3-270m-it-F16.gguf` (236 tensors, 18 layers).
    #[test]
    fn every_tensor_a_gemma3_gguf_publishes_has_a_name() {
        let published = [
            (
                "blk.3.attn_k.weight",
                "model.layers.3.self_attn.k_proj.weight",
            ),
            (
                "blk.3.attn_k_norm.weight",
                "model.layers.3.self_attn.k_norm.weight",
            ),
            (
                "blk.3.attn_norm.weight",
                "model.layers.3.input_layernorm.weight",
            ),
            (
                "blk.3.attn_output.weight",
                "model.layers.3.self_attn.o_proj.weight",
            ),
            (
                "blk.3.attn_q.weight",
                "model.layers.3.self_attn.q_proj.weight",
            ),
            (
                "blk.3.attn_q_norm.weight",
                "model.layers.3.self_attn.q_norm.weight",
            ),
            (
                "blk.3.attn_v.weight",
                "model.layers.3.self_attn.v_proj.weight",
            ),
            (
                "blk.3.ffn_down.weight",
                "model.layers.3.mlp.down_proj.weight",
            ),
            (
                "blk.3.ffn_gate.weight",
                "model.layers.3.mlp.gate_proj.weight",
            ),
            (
                "blk.3.ffn_norm.weight",
                "model.layers.3.pre_feedforward_layernorm.weight",
            ),
            ("blk.3.ffn_up.weight", "model.layers.3.mlp.up_proj.weight"),
            (
                "blk.3.post_attention_norm.weight",
                "model.layers.3.post_attention_layernorm.weight",
            ),
            (
                "blk.3.post_ffw_norm.weight",
                "model.layers.3.post_feedforward_layernorm.weight",
            ),
            ("output_norm.weight", "model.norm.weight"),
            ("token_embd.weight", "model.embed_tokens.weight"),
        ];
        for (gguf, hf) in published {
            assert_eq!(VOCAB.from_gguf(gguf).as_deref(), Some(hf), "mapping {gguf}");
        }
    }

    /// The sandwich's four norms land on four different names.
    ///
    /// The failure this guards is specific: `ffn_norm` means
    /// `post_attention_layernorm` in every other family in this crate, and
    /// here that name belongs to `post_attention_norm`. Two same-shaped norms
    /// swapped is a model that loads, identifies, builds, and is wrong.
    #[test]
    fn the_sandwich_norms_do_not_collide() {
        let names: Vec<String> = [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ]
        .iter()
        .map(|m| VOCAB.from_gguf(&format!("blk.0.{m}.weight")).unwrap())
        .collect();
        assert_eq!(
            names,
            [
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.pre_feedforward_layernorm.weight",
                "model.layers.0.post_feedforward_layernorm.weight",
            ]
        );
    }

    /// Exactly the norms carry the fold, and they carry `-1`.
    ///
    /// Both halves matter. Unfolding a projection would corrupt every weight
    /// in the model; not unfolding a norm would leave every norm off by one.
    #[test]
    fn the_fold_is_taken_off_the_norms_and_nothing_else() {
        for norm in [
            "blk.0.attn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.post_ffw_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "output_norm.weight",
        ] {
            assert_eq!(folded_constant(norm), Some(-1.0), "{norm} carries the fold");
        }
        for plain in [
            "blk.0.attn_q.weight",
            "blk.0.ffn_down.weight",
            "token_embd.weight",
            "output.weight",
        ] {
            assert_eq!(folded_constant(plain), None, "{plain} does not");
        }
    }

    /// The index is the layer's, not a fixed one.
    #[test]
    fn the_layer_index_is_carried_through() {
        for layer in [0u32, 7, 17, 199] {
            assert_eq!(
                VOCAB
                    .from_gguf(&format!("blk.{layer}.post_ffw_norm.weight"))
                    .as_deref(),
                Some(format!("model.layers.{layer}.post_feedforward_layernorm.weight").as_str())
            );
        }
    }

    /// A name this map does not know answers `None` rather than something.
    #[test]
    fn an_unknown_name_is_refused_rather_than_guessed() {
        for unknown in [
            "v.blk.0.attn_q.weight",     // the vision tower, which is not here
            "mm.0.weight",               // the multimodal projector, likewise
            "blk.3.ffn_gate_inp.weight", // a mixture's router
            "blk.x.attn_norm.weight",    // not an index
            "blk.3.attn_norm",           // no suffix
            "token_embd",                // no suffix
        ] {
            assert_eq!(VOCAB.from_gguf(unknown), None, "should not map {unknown}");
        }
    }
}
